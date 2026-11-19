// SimaCenterHeadDecode hardware suite (A65; the kernels implement no other target): the public
// Graph/Node/Run API compared byte for byte with the known answer. The head maps are dense, or
// padded per cell as the MLA of an SDK 3.0 package writes them.
#include "../support/bevfusion_kernel_test_utils.h"

namespace bt = bevfusion_test;
namespace neat = bt::neat;
namespace ref = bt::ref;

namespace {
std::vector<std::int64_t> map(std::int64_t channels) {
  return {1, 8, 8, channels};
}

neat::TensorList heads(const ref::CenterHeadDecodeCase& c) {
  return {bt::tensor(c.heatmap, map(3), "heatmap"), bt::tensor(c.box, map(16), "box"),
          bt::tensor(c.rot, map(4), "rot")};
}

// BF16 head maps, each cell's channels followed by padding up to `stored` channels. The padding
// holds 30, which must never be read.
neat::Tensor bf16_map(const std::vector<float>& values, std::int64_t channels, std::int64_t stored,
                      std::string_view name) {
  const auto n = static_cast<std::size_t>(channels), width = static_cast<std::size_t>(stored);
  std::vector<std::uint16_t> bits(64 * width, ref::to_bf16(30.0f));
  for (std::size_t cell = 0; cell < 64; ++cell)
    for (std::size_t c = 0; c < n; ++c)
      bits[cell * width + c] = ref::to_bf16(values[cell * n + c]);
  auto tensor = bt::bf16(bits, map(stored), name);
  if (stored != channels) {
    tensor.shape.back() = channels;
    tensor.strides_bytes = {64 * stored * 2, 8 * stored * 2, stored * 2, 2};
  }
  return tensor;
}

void check(const neat::TensorList& output, const ref::CenterHeadDecodeCase& c,
           const std::string& what) {
  bt::require(output.size() == 4, what + ": boxes, scores, labels and count");
  bt::expect_bytes(bt::values<float>(output[0], neat::TensorDType::Float32, {4, 9}, what), c.boxes,
                   what + " boxes");
  bt::expect_bytes(bt::values<float>(output[1], neat::TensorDType::Float32, {4}, what), c.scores,
                   what + " scores");
  bt::expect_bytes(bt::values<std::int32_t>(output[2], neat::TensorDType::Int32, {4}, what),
                   c.labels, what + " labels");
  bt::expect_bytes(bt::values<std::int32_t>(output[3], neat::TensorDType::Int32, {1}, what),
                   std::vector<std::int32_t>{c.count}, what + " count");
}

void fp32_heads() {
  const auto c = ref::centerhead_decode_case();
  check(bt::execute(neat::nodes::SimaCenterHeadDecode(c.options), {heads(c)}).at(0), c,
        "FP32 heads");
}

void bf16_heads() {
  const auto c = ref::centerhead_decode_case();
  for (const bool padded : {false, true}) {
    const neat::TensorList frame{bf16_map(c.heatmap, 3, padded ? 16 : 3, "heatmap"),
                                 bf16_map(c.box, 16, 16, "box"),
                                 bf16_map(c.rot, 4, padded ? 16 : 4, "rot")};
    check(bt::execute(neat::nodes::SimaCenterHeadDecode(c.options), {frame}).at(0), c,
          padded ? "BF16 heads padded to 16 channels" : "BF16 heads");
  }
}

// No logit above the threshold: count 0 and zeroed rows.
void empty_frame() {
  auto c = ref::centerhead_decode_case();
  std::fill(c.heatmap.begin(), c.heatmap.end(), -8.0f);
  std::fill(c.boxes.begin(), c.boxes.end(), 0.0f);
  std::fill(c.scores.begin(), c.scores.end(), 0.0f);
  std::fill(c.labels.begin(), c.labels.end(), 0);
  c.count = 0;
  check(bt::execute(neat::nodes::SimaCenterHeadDecode(c.options), {heads(c)}).at(0), c,
        "empty frame");
}

void reuse() {
  std::vector<neat::TensorList> frames;
  for (int generation = 0; generation < 4; ++generation)
    frames.push_back(heads(ref::centerhead_decode_case(generation)));
  const auto outputs =
      bt::execute(neat::nodes::SimaCenterHeadDecode(ref::centerhead_decode_case().options), frames);
  bt::require(outputs.size() == frames.size(), "one output per push");
  for (int i = 0; i < 4; ++i)
    check(outputs[i], ref::centerhead_decode_case(i), "push " + std::to_string(i));
}

void retained_outputs() {
  std::vector<neat::TensorList> frames;
  for (int generation = 0; generation < 8; ++generation)
    frames.push_back(heads(ref::centerhead_decode_case(generation)));
  bt::execute_retained(neat::nodes::SimaCenterHeadDecode(ref::centerhead_decode_case().options),
                       frames, [](const neat::TensorList& output, std::size_t frame) {
                         check(output, ref::centerhead_decode_case(static_cast<int>(frame)),
                               "retained output " + std::to_string(frame));
                       });
}

// The heads at nonzero offsets of one carrier, as the MLA's packed output places them.
void shared_offset_view() {
  std::vector<bt::SharedCarrier> carriers(3);
  std::vector<neat::TensorList> frames;
  for (int generation = 0; generation < 3; ++generation) {
    const auto c = ref::centerhead_decode_case(generation);
    auto& carrier = carriers[generation];
    carrier.add(std::vector<std::uint8_t>(64, 0x5a));
    const auto heatmap = carrier.add(c.heatmap);
    const auto box = carrier.add(c.box);
    const auto rot = carrier.add(c.rot);
    carrier.seal();
    frames.push_back({carrier.view(heatmap, neat::TensorDType::Float32, map(3), "heatmap"),
                      carrier.view(box, neat::TensorDType::Float32, map(16), "box"),
                      carrier.view(rot, neat::TensorDType::Float32, map(4), "rot")});
  }
  const auto outputs =
      bt::execute(neat::nodes::SimaCenterHeadDecode(ref::centerhead_decode_case().options), frames);
  for (std::size_t i = 0; i < outputs.size(); ++i) {
    check(outputs[i], ref::centerhead_decode_case(static_cast<int>(i)),
          "shared view " + std::to_string(i));
    carriers[i].check();
  }
}

void rejections() {
  const auto c = ref::centerhead_decode_case();
  const auto frame = heads(c);
  bt::expect_rejected(neat::nodes::SimaCenterHeadDecode(c.options), {frame[0], frame[1]},
                      "centerhead_decode admission failed", "a head port missing");
  std::vector<std::int32_t> integers(c.heatmap.size(), 1);
  bt::expect_rejected(neat::nodes::SimaCenterHeadDecode(c.options),
                      {bt::tensor(integers, map(3), "heatmap"), frame[1], frame[2]},
                      "must be FP32 or BF16, got INT32", "INT32 heatmap");
  auto ev74 = c.options;
  ev74.target = neat::SimaKernelTarget::EV74;
  bt::expect_rejected(neat::nodes::SimaCenterHeadDecode(ev74), frame,
                      "centerhead_decode admission failed", "EV74 selection");
  auto empty = c.options;
  empty.tasks.clear();
  bt::expect_invalid_options([&] { neat::nodes::SimaCenterHeadDecode(empty); }, "tasks must hold",
                             "no head tasks");
}
} // namespace

int main(int argc, char**) {
  return bt::main_test(argc, [] {
    bt::Suite suite("SimaCenterHeadDecode", {"fp32_heads", "bf16_heads", "empty_frame",
                                             "reuse_across_pushes", "retained_outputs_in_flight",
                                             "shared_offset_view", "rejects_invalid_requests"});
    suite.run("fp32_heads", fp32_heads);
    suite.run("bf16_heads", bf16_heads);
    suite.run("empty_frame", empty_frame);
    suite.run("reuse_across_pushes", reuse);
    suite.run("retained_outputs_in_flight", retained_outputs);
    suite.run("shared_offset_view", shared_offset_view);
    suite.run("rejects_invalid_requests", rejections);
    suite.finish();
  });
}
