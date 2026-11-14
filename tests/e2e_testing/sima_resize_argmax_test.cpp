// SimaResizeArgmax hardware suite: the public Graph/Node/Run API on the EV74,
// compared byte for byte with the independent oracle. Covers the SegFormer MLA
// score layouts (INT8 pitch 160, BF16 pitch 152), both index widths,
// align_corners, ties and NaN, batches, frame sequences on one Run, borrowed
// outputs across producer-pool rotation, and admission failures.
#include "../support/bevfusion_kernel_test_utils.h"
#include "../support/resize_argmax_reference.h"

#include <random>

namespace bt = bevfusion_test;
namespace neat = bt::neat;
namespace rr = resize_argmax_test::reference;

namespace {
struct Scores {
  neat::TensorDType dtype = neat::TensorDType::Int8;
  std::vector<std::int64_t> shape;  ///< [h,w,C] or [N,h,w,C]
  std::size_t pitch = 0;            ///< Stored scalars per pixel (>= C).
  std::vector<std::uint8_t> stored; ///< Pixel-pitched storage, padding included.
};

std::size_t width(neat::TensorDType dtype) {
  return dtype == neat::TensorDType::BFloat16 ? 2 : dtype == neat::TensorDType::Float32 ? 4 : 1;
}
rr::Scores oracle_dtype(neat::TensorDType dtype) {
  switch (dtype) {
  case neat::TensorDType::Int8:
    return rr::Scores::Int8;
  case neat::TensorDType::UInt8:
    return rr::Scores::UInt8;
  case neat::TensorDType::BFloat16:
    return rr::Scores::BFloat16;
  default:
    return rr::Scores::Float32;
  }
}

// Deterministic scores: raw mt19937 output only (distributions are not portable).
// `levels` folds values into few distinct scores, forcing class and corner ties.
Scores make_scores(neat::TensorDType dtype, std::vector<std::int64_t> shape, std::size_t pitch,
                   std::uint32_t seed, std::uint32_t levels = 0, std::size_t nans = 0) {
  Scores s{dtype, std::move(shape), pitch, {}};
  const auto c = static_cast<std::size_t>(s.shape.back());
  if (!s.pitch)
    s.pitch = c;
  std::size_t pixels = 1;
  for (std::size_t i = 0; i + 1 < s.shape.size(); ++i)
    pixels *= static_cast<std::size_t>(s.shape[i]);
  const auto bytes = width(dtype);
  s.stored.assign(pixels * s.pitch * bytes, 0xa5); // Padding is never a class.
  std::mt19937 rng(seed);
  for (std::size_t p = 0; p < pixels; ++p)
    for (std::size_t k = 0; k < c; ++k) {
      std::uint32_t r = rng();
      if (levels)
        r = (r % levels) * (256U / levels);
      auto* at = s.stored.data() + (p * s.pitch + k) * bytes;
      if (dtype == neat::TensorDType::Int8 || dtype == neat::TensorDType::UInt8) {
        *at = static_cast<std::uint8_t>(r);
      } else {
        const float v = (static_cast<float>(r % 2001U) - 1000.0f) / 64.0f;
        std::uint32_t bits;
        std::memcpy(&bits, &v, 4);
        if (dtype == neat::TensorDType::BFloat16) {
          const auto top = static_cast<std::uint16_t>(bits >> 16);
          std::memcpy(at, &top, 2);
        } else {
          std::memcpy(at, &bits, 4);
        }
      }
    }
  // Quiet NaNs at deterministic score positions (floating scores only).
  for (std::size_t i = 0; i < nans; ++i) {
    const std::size_t p = (rng() % pixels), k = rng() % c;
    auto* at = s.stored.data() + (p * s.pitch + k) * bytes;
    if (dtype == neat::TensorDType::BFloat16) {
      const std::uint16_t nan = 0x7fc0;
      std::memcpy(at, &nan, 2);
    } else {
      const std::uint32_t nan = 0x7fc00000U;
      std::memcpy(at, &nan, 4);
    }
  }
  return s;
}

// A public tensor over the pitched storage: a strided view when padded.
neat::Tensor tensor(const Scores& s) {
  auto result = neat::Tensor::from_vector(s.stored, {static_cast<std::int64_t>(s.stored.size())},
                                          neat::TensorMemory::EV74);
  result.dtype = s.dtype;
  result.shape = s.shape;
  if (s.pitch != static_cast<std::size_t>(s.shape.back())) {
    const auto w = static_cast<std::int64_t>(width(s.dtype));
    const auto rank = s.shape.size();
    std::vector<std::int64_t> strides(rank);
    strides[rank - 1] = w;
    strides[rank - 2] = static_cast<std::int64_t>(s.pitch) * w;
    for (std::size_t i = rank - 2; i-- > 0;)
      strides[i] = strides[i + 1] * s.shape[i + 1];
    result.strides_bytes = std::move(strides);
  }
  bt::name(result, "scores");
  return result;
}

std::vector<std::int32_t> expected(const Scores& s, const neat::SimaResizeArgmaxOptions& o) {
  const bool batched = s.shape.size() == 4;
  const auto at = [&](std::size_t i) { return static_cast<std::size_t>(s.shape[i]); };
  rr::Input in;
  in.bytes = s.stored.data();
  in.scores = oracle_dtype(s.dtype);
  in.n = batched ? at(0) : 1;
  in.h = at(batched ? 1 : 0);
  in.w = at(batched ? 2 : 1);
  in.c = s.shape.back();
  in.stride_w = s.pitch * width(s.dtype);
  in.stride_h = in.stride_w * in.w;
  in.stride_n = in.stride_h * in.h;
  return rr::resize_argmax(in, static_cast<std::size_t>(o.output_height),
                           static_cast<std::size_t>(o.output_width), o.align_corners);
}

void check(const neat::TensorList& output, const Scores& s, const neat::SimaResizeArgmaxOptions& o,
           const std::string& what) {
  bt::require(output.size() == 1, what + ": one class map");
  std::vector<std::int64_t> shape;
  if (s.shape.size() == 4)
    shape.push_back(s.shape.front());
  shape.push_back(o.output_height);
  shape.push_back(o.output_width);
  const auto reference = expected(s, o);
  if (s.shape.back() <= 256) {
    const std::vector<std::uint8_t> narrow(reference.begin(), reference.end());
    bt::expect_bytes(bt::values<std::uint8_t>(output[0], neat::TensorDType::UInt8, shape, what),
                     narrow, what + " class map");
  } else {
    bt::expect_bytes(bt::values<std::int32_t>(output[0], neat::TensorDType::Int32, shape, what),
                     reference, what + " class map");
  }
}

void run(const Scores& s, const neat::SimaResizeArgmaxOptions& o, const std::string& what) {
  const auto outputs = bt::execute(neat::nodes::SimaResizeArgmax(o), {{tensor(s)}});
  bt::require(outputs.size() == 1, what + ": one output per push");
  check(outputs.front(), s, o, what);
}

using D = neat::TensorDType;

void segformer_int8_pitch160() {
  run(make_scores(D::Int8, {1, 16, 16, 150}, 160, 1), {64, 64, false}, "INT8 pitch 160");
}
void segformer_bf16_pitch152() {
  run(make_scores(D::BFloat16, {1, 12, 12, 150}, 152, 2, 0, 24), {80, 45, false},
      "BF16 pitch 152 with NaN");
}
void int32_indices() {
  run(make_scores(D::Float32, {1, 6, 7, 300}, 0, 3, 0, 5), {11, 13, false},
      "FP32 300 classes -> INT32");
}
void align_corners_ties() {
  run(make_scores(D::UInt8, {1, 6, 6, 9}, 0, 4, 2), {17, 13, true}, "UINT8 align_corners ties");
  run(make_scores(D::Int8, {1, 8, 8, 19}, 0, 5, 4), {32, 32, false}, "INT8 tie-heavy");
}
void rank3_and_batch() {
  run(make_scores(D::Int8, {8, 8, 19}, 0, 6), {32, 24, false}, "rank-3 scores");
  run(make_scores(D::Int8, {2, 8, 8, 19}, 32, 7), {32, 32, false}, "batch 2, pitch 32");
}
void frame_sequence() {
  const neat::SimaResizeArgmaxOptions o{96, 64, false};
  std::vector<Scores> frames;
  std::vector<neat::TensorList> tensors;
  for (std::uint32_t f = 0; f < 3; ++f) {
    frames.push_back(make_scores(D::Int8, {1, 16, 16, 150}, 160, 10 + f));
    tensors.push_back({tensor(frames.back())});
  }
  const auto outputs = bt::execute(neat::nodes::SimaResizeArgmax(o), tensors);
  bt::require(outputs.size() == frames.size(), "frame sequence: one output per push");
  for (std::size_t f = 0; f < frames.size(); ++f)
    check(outputs[f], frames[f], o, "frame " + std::to_string(f));
}
void retained_outputs() {
  const neat::SimaResizeArgmaxOptions o{64, 64, false};
  std::vector<Scores> frames;
  std::vector<neat::TensorList> tensors;
  for (std::uint32_t f = 0; f < 6; ++f) {
    frames.push_back(make_scores(D::Int8, {1, 16, 16, 150}, 160, 20 + f));
    tensors.push_back({tensor(frames.back())});
  }
  bt::execute_retained(neat::nodes::SimaResizeArgmax(o), tensors,
                       [&](const neat::TensorList& output, std::size_t frame) {
                         check(output, frames[frame], o, "retained " + std::to_string(frame));
                       });
}
void admission_failures() {
  auto per_class = tensor(make_scores(D::Int8, {1, 4, 4, 3}, 0, 30));
  per_class.semantic.quant.emplace();
  per_class.semantic.quant->axis = 3;
  per_class.semantic.quant->scales = {0.5f, 0.25f, 0.125f};
  per_class.semantic.quant->zero_points = {0, 0, 0};
  bt::expect_rejected(neat::nodes::SimaResizeArgmax({8, 8, false}), {per_class}, "per-axis",
                      "per-class INT8 quantization");
  bt::expect_rejected(neat::nodes::SimaResizeArgmax({8, 8, false}),
                      {tensor(make_scores(D::Int8, {4, 3}, 0, 31))}, "[h,w,C] or [N,h,w,C]",
                      "rank-2 scores");
  bt::expect_invalid_options([] { neat::nodes::SimaResizeArgmax({0, 8, false}); },
                             "SimaResizeArgmax: output_width", "empty class map");
}
} // namespace

int main(int argc, char** argv) {
  (void)argv;
  return bt::main_test(argc, [] {
    bt::Suite suite("resize_argmax", {"segformer_int8_pitch160", "segformer_bf16_pitch152",
                                      "int32_indices", "align_corners_ties", "rank3_and_batch",
                                      "frame_sequence", "retained_outputs", "admission_failures"});
    suite.run("segformer_int8_pitch160", segformer_int8_pitch160);
    suite.run("segformer_bf16_pitch152", segformer_bf16_pitch152);
    suite.run("int32_indices", int32_indices);
    suite.run("align_corners_ties", align_corners_ties);
    suite.run("rank3_and_batch", rank3_and_batch);
    suite.run("frame_sequence", frame_sequence);
    suite.run("retained_outputs", retained_outputs);
    suite.run("admission_failures", admission_failures);
    suite.finish();
  });
}
