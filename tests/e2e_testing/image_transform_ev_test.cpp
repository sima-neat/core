// Public Graph qualification; the reference below never calls a kernel or device helper.
// DMA writes create fixtures only. Output maps verify completed work, never feed a fallback.
#include <neat.h>
#include "dmabuf_test_utils.h"
#include "gst/GstHelpers.h"
#include "pipeline/TensorAdapters.h"
#include "pipeline/runtime/RunCore.h"
#include "pipeline/runtime/ExecutionGraphRuntime.h"
#include <gstsimaaitensorbuffer_meta.h>
#include <simaai/scheduler/testing/probe.h>
#include <simaai/scheduler/client/tensor_buffer.h>
#include <gst/video/video.h>
#include <algorithm>
#include <array>
#include <bit>
#include <atomic>
#include <cfenv>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <functional>
#include <iostream>
#include <optional>
#include <stdexcept>
#include <thread>

namespace {
using namespace simaai::neat;
namespace gs = simaai::scheduler::client;
using F = ImageFormat;
using I = ResizeInterpolation;
void require(bool ok, const std::string& message) {
  if (!ok)
    throw std::runtime_error(message);
}
const char* name(F f) {
  switch (f) {
  case F::RGB:
    return "RGB";
  case F::BGR:
    return "BGR";
  case F::GRAY8:
    return "GRAY8";
  case F::NV12:
    return "NV12";
  case F::I420:
    return "I420";
  case F::NV16:
    return "NV16";
  }
  throw std::runtime_error("unknown fixture format");
}
bool yuv(F f) {
  return f == F::NV12 || f == F::I420 || f == F::NV16;
}
int vertical(F f) {
  return f == F::NV16 ? 1 : 2;
}
struct Plane {
  int width, height, channels;
  std::vector<std::uint8_t> bytes;
  std::uint8_t at(int x, int y, int c = 0) const {
    x = std::clamp(x, 0, width - 1);
    y = std::clamp(y, 0, height - 1);
    return bytes[(std::size_t(y) * width + x) * channels + c];
  }
  std::uint8_t& at(int x, int y, int c = 0) {
    return bytes[(std::size_t(y) * width + x) * channels + c];
  }
};
struct Image {
  F format;
  int width, height;
  std::vector<Plane> planes;
};
Image image(F f, int w, int h) {
  Image result{f, w, h, {}};
  auto plane = [&](int pw, int ph, int c) {
    result.planes.push_back({pw, ph, c, std::vector<std::uint8_t>(std::size_t(pw) * ph * c)});
  };
  if (!yuv(f))
    plane(w, h, f == F::GRAY8 ? 1 : 3);
  else {
    plane(w, h, 1);
    plane(w / 2, h / vertical(f), f == F::I420 ? 1 : 2);
    if (f == F::I420)
      plane(w / 2, h / 2, 1);
  }
  return result;
}
Image pattern(F f, int w, int h, unsigned seed) {
  auto result = image(f, w, h);
  for (std::size_t p = 0; p < result.planes.size(); ++p) {
    auto& plane = result.planes[p];
    for (int y = 0; y < plane.height; ++y)
      for (int x = 0; x < plane.width; ++x)
        for (int c = 0; c < plane.channels; ++c)
          plane.at(x, y, c) = (17 * x + 29 * y + 61 * c + 83 * p + 37 * seed) & 255;
  }
  return result;
}
// Independently quantize to binary16 using its exact power-of-two lattice.
// All admitted reference intermediates are finite and below 65504.
float half(float x) {
  if (x == 0)
    return x;
  int exponent = 0;
  std::frexp(std::abs(x), &exponent);
  const double step = std::ldexp(1.0, std::max(exponent - 11, -24));
  return float(std::nearbyint(double(x) / step) * step);
}
float cubic(float x) {
  x = std::abs(x);
  if (x <= 1)
    return ((1.25f * x - 2.25f) * (x * x)) + 1;
  if (x < 2)
    return ((3.75f - 0.75f * x) * x - 6) * x + 3;
  return 0;
}
float sample(const Plane& p, float x, float y, int c, I mode, float box_width = 1,
             float box_height = 1) {
  const int ix = int(std::floor(x)), iy = int(std::floor(y));
  if (mode == I::Nearest)
    return p.at(std::clamp(ix, 0, p.width - 1), std::clamp(iy, 0, p.height - 1), c);
  auto get = [&](int sx, int sy) {
    return float(p.at(std::clamp(sx, 0, p.width - 1), std::clamp(sy, 0, p.height - 1), c));
  };
  if (mode == I::Bilinear) {
    const float fx = half(x - ix), fy = half(y - iy);
    const float top = half(get(ix, iy) + half(half(get(ix + 1, iy) - get(ix, iy)) * fx));
    const float bottom =
        half(get(ix, iy + 1) + half(half(get(ix + 1, iy + 1) - get(ix, iy + 1)) * fx));
    return half(top + half(half(bottom - top) * fy));
  }
  float sum = 0;
  if (mode == I::Bicubic) {
    for (int sy = iy - 1; sy <= iy + 2; ++sy)
      for (int sx = ix - 1; sx <= ix + 2; ++sx) {
        const float weight = cubic(x - sx) * cubic(y - sy);
        const float term = get(sx, sy) * weight;
        sum = sum + term;
      }
    return half(std::clamp(sum, 0.0f, 255.0f));
  }
  const int x0 = int(std::floor(x + .5f - box_width / 2));
  const int x1 = int(std::ceil(x + .5f + box_width / 2));
  const int y0 = int(std::floor(y + .5f - box_height / 2));
  const int y1 = int(std::ceil(y + .5f + box_height / 2));
  for (int sy = y0; sy < y1; ++sy)
    for (int sx = x0; sx < x1; ++sx)
      sum += get(sx, sy);
  return half(sum / float((x1 - x0) * (y1 - y0)));
}
std::uint8_t truncate(float x) {
  return std::uint8_t(std::clamp(x, 0.0f, 255.0f));
}
Image resize_reference(const Image& input, int w, int h, I mode, bool contain = false,
                       std::uint8_t pad = 0) {
  auto out = image(input.format, w, h);
  int dw = w, dh = h;
  if (contain) {
    if (input.width * h >= input.height * w)
      dh = int(std::ceil(float(input.height * w) / input.width));
    else
      dw = int(std::ceil(float(input.width * h) / input.height));
  }
  const int dx = (w - dw) / 2, dy = (h - dh) / 2;
  for (std::size_t p = 0; p < out.planes.size(); ++p) {
    auto& target = out.planes[p];
    const auto& source = input.planes[p];
    const int sx = p ? 2 : 1, sy = p ? vertical(input.format) : 1;
    const float rx = float(input.width) / dw, ry = float(input.height) / dh;
    for (int y = 0; y < target.height; ++y)
      for (int x = 0; x < target.width; ++x)
        for (int c = 0; c < target.channels; ++c) {
          if (x * sx < dx || x * sx >= dx + dw || y * sy < dy || y * sy >= dy + dh) {
            target.at(x, y, c) = p ? 128 : pad;
            continue;
          }
          // Full-source centered component coordinates, not floor-resized luma taps.
          const float xx =
              (((float(x * sx) + .5f * (sx - 1) - dx) + .5f) * rx - .5f - .5f * (sx - 1)) / sx;
          const float yy =
              (((float(y * sy) + .5f * (sy - 1) - dy) + .5f) * ry - .5f - .5f * (sy - 1)) / sy;
          target.at(x, y, c) = truncate(sample(source, xx, yy, c, mode, rx, ry));
        }
  }
  return out;
}
std::array<int, 3> rgb_at(const Image& src, int x, int y) {
  const auto& p = src.planes[0];
  if (src.format == F::GRAY8)
    return {p.at(x, y), p.at(x, y), p.at(x, y)};
  if (src.format == F::BGR)
    return {p.at(x, y, 2), p.at(x, y, 1), p.at(x, y, 0)};
  return {p.at(x, y, 0), p.at(x, y, 1), p.at(x, y, 2)};
}
std::uint8_t rounded_code(std::int64_t value, std::int64_t denominator) {
  // A neutral bias makes these numerators nonnegative. Round ties to even once.
  require(value >= 0, "reference chroma numerator");
  auto q = value / denominator;
  const auto remainder = value % denominator;
  if (remainder * 2 > denominator || (remainder * 2 == denominator && (q & 1)))
    ++q;
  return std::uint8_t(std::clamp<std::int64_t>(q, 0, 255));
}
Image color_reference(const Image& src, F format, const ImageColorOptions& color = {}) {
  auto out = image(format, src.width, src.height);
  if (!yuv(src.format) && !yuv(format)) {
    for (int y = 0; y < src.height; ++y)
      for (int x = 0; x < src.width; ++x) {
        const auto rgb = rgb_at(src, x, y);
        for (int c = 0; c < out.planes[0].channels; ++c)
          out.planes[0].at(x, y, c) = rgb[format == F::BGR ? 2 - c : c];
      }
    return out;
  }
  if (!yuv(src.format)) {
    // Published full-range Q16 matrix. Chroma averages unrounded numerators.
    const bool bt709 = color.matrix == ImageColorMatrix::Bt709;
    const std::array<int, 3> yc =
        bt709 ? std::array{13933, 46871, 4732} : std::array{19595, 38470, 7471};
    const std::array<int, 3> uc =
        bt709 ? std::array{-7509, -25259, 32768} : std::array{-11058, -21710, 32768};
    const std::array<int, 3> vc =
        bt709 ? std::array{32768, -29763, -3005} : std::array{32768, -27439, -5329};
    auto dot = [](const auto& a, const auto& b) {
      return std::int64_t(a[0]) * b[0] + std::int64_t(a[1]) * b[1] + std::int64_t(a[2]) * b[2];
    };
    for (int y = 0; y < src.height; ++y)
      for (int x = 0; x < src.width; ++x)
        out.planes[0].at(x, y) = rounded_code(dot(rgb_at(src, x, y), yc), 65536);
    const int rows = vertical(format), cells = 2 * rows;
    for (int y = 0; y < src.height / rows; ++y)
      for (int x = 0; x < src.width / 2; ++x) {
        std::int64_t u = std::int64_t(cells) * 128 * 65536, v = u;
        for (int j = 0; j < rows; ++j)
          for (int i = 0; i < 2; ++i) {
            const auto rgb = rgb_at(src, 2 * x + i, rows * y + j);
            u += dot(rgb, uc);
            v += dot(rgb, vc);
          }
        out.planes[1].at(x, y, 0) = rounded_code(u, cells * 65536);
        if (format == F::I420)
          out.planes[2].at(x, y) = rounded_code(v, cells * 65536);
        else
          out.planes[1].at(x, y, 1) = rounded_code(v, cells * 65536);
      }
    return out;
  }
  require(yuv(format), "YUV-to-packed reference is intentionally a separate neutral-chroma case");
  out.planes[0].bytes = src.planes[0].bytes;
  for (int channel = 0; channel < 2; ++channel) {
    const auto& plane = src.planes[src.format == F::I420 ? channel + 1 : 1];
    const int source_channel = src.format == F::I420 ? 0 : channel;
    auto& destination = out.planes[format == F::I420 ? channel + 1 : 1];
    for (int y = 0; y < destination.height; ++y)
      for (int x = 0; x < destination.width; ++x) {
        const float ix = color.input_siting[0] == ImageChromaSiting::Centered ? .5f : 0;
        const float iy = color.input_siting[1] == ImageChromaSiting::Centered
                             ? .5f * (vertical(src.format) - 1)
                             : 0;
        const float ox = color.output_siting[0] == ImageChromaSiting::Centered ? .5f : 0;
        const float oy = color.output_siting[1] == ImageChromaSiting::Centered
                             ? .5f * (vertical(format) - 1)
                             : 0;
        const float xx = (2 * x + ox - ix) / 2,
                    yy = (vertical(format) * y + oy - iy) / vertical(src.format);
        destination.at(x, y, format == F::I420 ? 0 : channel) =
            truncate(sample(plane, xx, yy, source_channel, I::Bilinear));
      }
  }
  return out;
}
Tensor device_tensor(const Image& input) {
  std::array<gsize, GST_VIDEO_MAX_PLANES> offsets{};
  std::array<gint, GST_VIDEO_MAX_PLANES> strides{};
  std::size_t bytes = 0;
  for (std::size_t p = 0; p < input.planes.size(); ++p) {
    const auto& plane = input.planes[p];
    offsets[p] = bytes;
    strides[p] = ((plane.width * plane.channels + 31) / 16) * 16;
    bytes += std::size_t(strides[p]) * plane.height + 64;
    bytes = (bytes + 63) & ~std::size_t(63);
  }
  std::unique_ptr<GstBuffer, decltype(&gst_buffer_unref)> owner(
      sima_test::allocate_cma_dmabuf(bytes), gst_buffer_unref);
  auto* buffer = owner.get();
  GstMapInfo mapping{};
  require(gst_buffer_map(buffer, &mapping, GST_MAP_WRITE), "fixture DMA map failed");
  std::memset(mapping.data, 0xbd, mapping.size);
  for (std::size_t p = 0; p < input.planes.size(); ++p) {
    const auto& plane = input.planes[p];
    for (int row = 0; row < plane.height; ++row)
      std::memcpy(mapping.data + offsets[p] + std::size_t(row) * strides[p],
                  plane.bytes.data() + std::size_t(row) * plane.width * plane.channels,
                  std::size_t(plane.width) * plane.channels);
  }
  gst_buffer_unmap(buffer, &mapping);
  require(gst_buffer_add_video_meta_full(
              buffer, GST_VIDEO_FRAME_FLAG_NONE, gst_video_format_from_string(name(input.format)),
              input.width, input.height, input.planes.size(), offsets.data(), strides.data()),
          "fixture VideoMeta failed");
  auto* caps = gst_caps_new_simple(
      "video/x-raw", "format", G_TYPE_STRING, name(input.format), "width", G_TYPE_INT, input.width,
      "height", G_TYPE_INT, input.height, "interlace-mode", G_TYPE_STRING, "progressive", nullptr);
  auto* sample = gst_sample_new(buffer, caps, nullptr, nullptr);
  std::unique_ptr<GstCaps, decltype(&gst_caps_unref)> caps_owner(caps, gst_caps_unref);
  std::unique_ptr<GstSample, decltype(&gst_sample_unref)> sample_owner(sample, gst_sample_unref);
  return from_gst_sample(sample);
}
GstBuffer* buffer_of(const Tensor& tensor) {
  require(tensor.storage && tensor.storage->kind == StorageKind::GstSample &&
              tensor.storage->holder,
          "output lost zero-copy GstSample owner");
  return gst_sample_get_buffer(static_cast<GstSample*>(tensor.storage->holder.get()));
}
Tensor only_tensor(const Sample& sample) {
  require(sample.kind != SampleKind::Bundle,
          "one image was published as an unrelated tensor bundle");
  const auto tensors = tensors_from_sample(sample);
  require(tensors.size() == 1, "expected one semantic image/tensor");
  return tensors.front();
}
void check_raw(const Sample& sample, const Image& expected, bool encoder_surface = false) {
  const auto tensor = only_tensor(sample);
  auto* buffer = buffer_of(tensor);
  require(tensor.dtype == TensorDType::UInt8 && tensor.semantic.image.has_value(),
          "raw image semantic contract");
  require(gst_buffer_n_memory(buffer) == 1 &&
              !gst_buffer_get_custom_meta(buffer, SIMA_TENSOR_SET_META_NAME),
          "raw image carrier split or fabricated tensor set");
  (void)sima_test::dmabuf_span(buffer);
  const auto* meta = gst_buffer_get_video_meta(buffer);
  require(meta && meta->format == gst_video_format_from_string(name(expected.format)) &&
              meta->width == unsigned(expected.width) &&
              meta->height == unsigned(expected.height) &&
              meta->n_planes == expected.planes.size() && meta->flags == GST_VIDEO_FRAME_FLAG_NONE,
          "raw output VideoMeta mismatch");
  const auto map = tensor.storage->map(MapMode::Read);
  require(map.data && map.size_bytes >= gst_buffer_get_size(buffer),
          "completed raw image map failed");
  const auto* bytes = static_cast<const std::uint8_t*>(map.data);
  for (std::size_t p = 0; p < expected.planes.size(); ++p) {
    const auto& plane = expected.planes[p];
    const auto row_bytes = plane.width * plane.channels;
    require(meta->stride[p] >= row_bytes &&
                meta->offset[p] + std::size_t(meta->stride[p]) * (plane.height - 1) + row_bytes <=
                    map.size_bytes,
            "raw output span mismatch");
    for (int row = 0; row < plane.height; ++row)
      for (int col = 0; col < row_bytes; ++col) {
        const auto actual = bytes[meta->offset[p] + std::size_t(row) * meta->stride[p] + col];
        const auto want = plane.bytes[std::size_t(row) * row_bytes + col];
        if (actual != want)
          throw std::runtime_error("pixel mismatch plane=" + std::to_string(p) +
                                   " row=" + std::to_string(row) + " byte=" + std::to_string(col) +
                                   " actual=" + std::to_string(actual) +
                                   " expected=" + std::to_string(want));
      }
  }
  if (encoder_surface) {
    const int pitch = (expected.width + 63) / 64 * 64, rows = (expected.height + 7) / 8 * 8;
    require(expected.format == F::NV12 && meta->stride[0] == pitch && meta->stride[1] == pitch &&
                meta->offset[1] == meta->offset[0] + std::size_t(pitch) * rows,
            "encoder surface layout mismatch");
    for (int p = 0; p < 2; ++p) {
      const int visible = p ? expected.height / 2 : expected.height, storage = p ? rows / 2 : rows;
      require(meta->offset[p] + std::size_t(pitch) * storage <= map.size_bytes,
              "encoder storage span missing");
      for (int row = 0; row < storage; ++row)
        for (int col = 0; col < pitch; ++col)
          if (row >= visible || col >= expected.width)
            require(bytes[meta->offset[p] + std::size_t(row) * pitch + col] == (p ? 128 : 0),
                    "encoder declared padding was not initialized");
    }
  }
}
void check_normalized(const Sample& sample, const Image& expected,
                      const NormalizationOptions& options) {
  const auto tensor = only_tensor(sample);
  require(tensor.dtype == TensorDType::BFloat16 && !tensor.semantic.image &&
              tensor.shape == std::vector<std::int64_t>(
                                  {expected.height, expected.width, expected.planes[0].channels}),
          "normalization is not a dense BF16 tensor");
  const auto map = tensor.map(MapMode::Read);
  const auto& plane = expected.planes[0];
  require(map.data && map.size_bytes >= plane.bytes.size() * 2, "normalized output map failed");
  const auto* values = static_cast<const std::uint16_t*>(map.data);
  for (std::size_t i = 0; i < plane.bytes.size(); ++i) {
    const auto storage_channel = i % plane.channels;
    const auto channel = expected.format == F::BGR ? 2 - storage_channel : storage_channel;
    const float offset = float(options.mean[channel] * 255.0);
    const float multiplier = float(1.0 / (options.stddev[channel] * 255.0));
    const float shifted = float(plane.bytes[i]) - offset;
    const float normalized = shifted * multiplier;
    const auto want = std::uint16_t(std::bit_cast<std::uint32_t>(normalized) >> 16);
    if (values[i] != want)
      throw std::runtime_error("BF16 mismatch index=" + std::to_string(i) + " actual=" +
                               std::to_string(values[i]) + " expected=" + std::to_string(want));
  }
}
struct Inspected {
  // Named public inputs use graph execution even for one linear segment.
  // Retain each actual segment owner and GstBin, never reinterpret the root's
  // intentionally empty single-pipeline slot as the graph's live carrier.
  std::vector<std::shared_ptr<const runtime::RunCore>> owners;
  std::vector<std::shared_ptr<GstElement>> pipelines;
  std::vector<GstElement*> runners;
  GstElement* encoder = nullptr;
};
Inspected inspect(Run& run, unsigned expected, bool mixed_ev_a65 = false) {
  Inspected result;
  auto root = run_internal::core(run);
  require(bool(root), "missing live Core owner");
  result.owners.push_back(std::move(root));
  for (std::size_t i = 0; i < result.owners.size(); ++i) {
    const auto core = result.owners[i];
    if (core->graph_execution_) {
      const auto& execution = core->graph_execution();
      require(execution.stages.empty(), "unexpected application/CPU runtime stage");
      for (const auto& pipe : execution.pipelines) {
        require(bool(pipe), "missing compiled pipeline segment");
        std::lock_guard lock(pipe->transport.mu);
        if (pipe->seg.boundary.direct_graph_source) {
          require(!pipe->run_core, "direct graph source unexpectedly owns a Gst pipeline");
          continue;
        }
        require(pipe->transport.built.load(std::memory_order_acquire) && pipe->run_core,
                "image pipeline segment was not materialized by seeded build");
        require(std::find(result.owners.begin(), result.owners.end(), pipe->run_core) ==
                    result.owners.end(),
                "duplicate or cyclic Core segment ownership");
        result.owners.push_back(pipe->run_core);
      }
      continue;
    }
    auto* pipeline = core->pipeline.stream.pipeline_handle();
    require(pipeline && GST_IS_BIN(pipeline), "missing live Core pipeline segment");
    result.pipelines.emplace_back(GST_ELEMENT(gst_object_ref(pipeline)), gst_object_unref);
  }
  require(!result.pipelines.empty(), "graph has no live pipeline segments");
  unsigned a65_stages = 0;
  for (const auto& pipeline : result.pipelines) {
    std::unique_ptr<GstIterator, decltype(&gst_iterator_free)> iterator(
        gst_bin_iterate_recurse(GST_BIN(pipeline.get())), gst_iterator_free);
    struct Value {
      GValue raw = G_VALUE_INIT;
      ~Value() {
        if (G_IS_VALUE(&raw))
          g_value_unset(&raw);
      }
    } holder;
    auto& value = holder.raw;
    GstIteratorResult status;
    while ((status = gst_iterator_next(iterator.get(), &value)) == GST_ITERATOR_OK) {
      auto* element = GST_ELEMENT(g_value_get_object(&value));
      auto* factory = gst_element_get_factory(element);
      const char* type = factory ? gst_plugin_feature_get_name(GST_PLUGIN_FEATURE(factory)) : "";
      require(!g_str_equal(type, "neatprocesscvu") && !g_str_equal(type, "videoconvert") &&
                  !g_str_equal(type, "videoscale"),
              "unexpected legacy/CPU numerical fallback");
      if (g_str_equal(type, "neatkernelrunner")) {
        const auto stage = simaai::scheduler::testing::inspect_stage(element);
        require(stage.valid && stage.members.size() == 1 && stage.members[0].prepared,
                "image transform did not prepare exactly one numerical operation");
        if (mixed_ev_a65 && stage.members[0].target == gs::Target::A65) {
          require(std::string_view(GST_OBJECT_NAME(element)).find("post_cast") !=
                      std::string_view::npos,
                  "unexpected A65 operation in the mixed image chain");
          ++a65_stages;
        } else {
          require(stage.members[0].target == gs::Target::EV74,
                  "image transform did not prepare an actual EV74 operation");
        }
        result.runners.push_back(element);
      }
      if (g_str_equal(type, "neatencoder")) {
        require(!result.encoder, "unexpected multiple encoder instances");
        result.encoder = element;
      }
      g_value_reset(&value);
    }
    require(status == GST_ITERATOR_DONE, "live graph changed during stage inspection");
  }
  require(result.runners.size() == expected, "image transform stage count mismatch");
  require(a65_stages == (mixed_ev_a65 ? 1U : 0U),
          "mixed image chain did not prepare exactly one A65 Cast");
  return result;
}
[[noreturn]] void uncertain(std::string_view message) noexcept {
  // Even a diagnostic/allocation failure must not unwind an uncertain owner.
  try {
    std::cerr << "UNRETIRED image-transform: " << message
              << "; retaining run and all owners for ROOT guarded recovery\n"
              << std::flush;
  } catch (...) {
  }
  for (;;) {
    try {
      std::this_thread::sleep_for(std::chrono::seconds(60));
    } catch (...) {
    }
  }
}
void correlation_diagnostic(const Sample& input, const Sample& output, unsigned frame,
                            std::size_t run_sequence) noexcept {
  // A diagnostic failure must never unwind a possibly still-owned submission.
  try {
    std::fprintf(stderr, "CORRELATION_MISMATCH frame=%u run_sequence=%zu\n", frame, run_sequence);
    const auto print = [](const char* label, const Sample& sample) {
      std::fprintf(
          stderr,
          "%s frame_id=%lld input_seq=%lld orig_input_seq=%lld pts_ns=%lld "
          "stream_id_bytes=%zu stream_id='%s'\n",
          label, static_cast<long long>(sample.frame_id), static_cast<long long>(sample.input_seq),
          static_cast<long long>(sample.orig_input_seq), static_cast<long long>(sample.pts_ns),
          sample.stream_id.size(), sample.stream_id.c_str());
    };
    print("expected", input);
    print("actual", output);
    std::fflush(stderr);
  } catch (...) {
  }
}
void submission_diagnostic(const Run& run, const Inspected& inspection, const char* phase,
                           const char* status, const PullError* error = nullptr,
                           const char* exception = nullptr) noexcept {
  // Read stored diagnostics only: never pop a bus message, change state, or
  // consume another output while a submission's ownership is uncertain.
  try {
    // Establish UNKNOWN before optional getters: even a contended diagnostic
    // mutex cannot prevent the supervisor from recognizing retained owners.
    std::fprintf(stderr, "UNRETIRED image-transform: submission failed; retaining all owners "
                         "while collecting stored diagnostics\n");
    std::fprintf(stderr, "SUBMISSION_FAILURE phase=%s status=%s exception='%s'\n", phase, status,
                 exception ? exception : "");
    std::fflush(stderr);
    const auto public_error = run.last_error();
    std::fprintf(stderr, "run.last_error='%s'\n", public_error.c_str());
    if (error)
      std::fprintf(stderr, "pull_error code='%s' message='%s'\n", error->code.c_str(),
                   error->message.c_str());
    for (std::size_t i = 0; i < inspection.owners.size(); ++i) {
      const auto& owner = inspection.owners[i];
      const auto message = owner->last_error();
      const auto detail = owner->last_error_detail();
      const auto stats = owner->stats();
      const auto input_stats = owner->input_stats();
      std::fprintf(stderr,
                   "owner=%zu graph=%d last_error='%s' code='%s' detail='%s' "
                   "enqueued=%llu pushed=%llu ready=%llu pulled=%llu "
                   "input_push=%llu input_fail=%llu input_pull=%llu dropped=%llu "
                   "renegotiations=%llu renegotiation_blocked=%llu growth_blocked=%llu\n",
                   i, owner->graph_execution_ ? 1 : 0, message.c_str(),
                   detail ? detail->code.c_str() : "", detail ? detail->message.c_str() : "",
                   static_cast<unsigned long long>(stats.inputs_enqueued),
                   static_cast<unsigned long long>(stats.inputs_pushed),
                   static_cast<unsigned long long>(stats.outputs_ready),
                   static_cast<unsigned long long>(stats.outputs_pulled),
                   static_cast<unsigned long long>(input_stats.push_count),
                   static_cast<unsigned long long>(input_stats.push_failures),
                   static_cast<unsigned long long>(input_stats.pull_count),
                   static_cast<unsigned long long>(input_stats.dropped_frames),
                   static_cast<unsigned long long>(input_stats.renegotiations),
                   static_cast<unsigned long long>(input_stats.renegotiation_blocked),
                   static_cast<unsigned long long>(input_stats.growth_blocked));
      const auto summary = owner->diagnostics_summary();
      if (!summary.empty())
        std::fprintf(stderr, "owner=%zu diagnostics: %s\n", i, summary.c_str());
    }
    std::fflush(stderr);
  } catch (...) {
  }
}
const char* pull_status_name(PullStatus status) noexcept {
  switch (status) {
  case PullStatus::Ok:
    return "Ok";
  case PullStatus::Timeout:
    return "Timeout";
  case PullStatus::Closed:
    return "Closed";
  case PullStatus::Error:
    return "Error";
  }
  return "Unknown";
}
std::uint64_t frame_pts(unsigned frame, unsigned frame_rate) {
  return (std::uint64_t(frame) + 1) * 1000000000 / frame_rate;
}
Sample transact(Run& run, const Inspected& inspection, const Tensor& tensor, unsigned frame,
                std::size_t run_sequence, unsigned frame_rate = 1000) {
  Sample input;
  input.kind = SampleKind::TensorSet;
  input.payload_type = PayloadType::Image;
  input.tensors = {tensor};
  input.frame_id = 1000 + frame;
  // Named graphs renumber input_seq at each materialized segment, starting at
  // zero. These sequential, no-drop, one-output-per-input runs use that local
  // ordinal; frame_id/orig_input_seq/PTS remain independently unique across runs.
  input.input_seq = static_cast<std::int64_t>(run_sequence);
  input.orig_input_seq = 3000 + frame;
  input.stream_id = "image-transform-ev";
  input.pts_ns = frame_pts(frame, frame_rate);
  input.duration_ns = frame_pts(frame + 1, frame_rate) - input.pts_ns;
  Sample output;
  // A failed push may already have reached a backend. No destructor or signal
  // can stand in for retirement; park only this genuinely uncertain path.
  try {
    if (!run.push(input)) {
      submission_diagnostic(run, inspection, "push", "false");
      uncertain("push returned false without establishing completion");
    }
    PullError error;
    const auto status = run.pull(20000, output, &error);
    if (status != PullStatus::Ok) {
      submission_diagnostic(run, inspection, "pull", pull_status_name(status), &error);
      uncertain("pull did not establish completion");
    }
  } catch (const std::exception& error) {
    submission_diagnostic(run, inspection, "push/pull", "exception", nullptr, error.what());
    uncertain(error.what());
  } catch (...) {
    submission_diagnostic(run, inspection, "push/pull", "unknown exception");
    uncertain("push/pull threw");
  }
  // A stale or unrelated output is not evidence that this input retired.
  if (output.frame_id != input.frame_id || output.input_seq != input.input_seq ||
      output.orig_input_seq != input.orig_input_seq || output.stream_id != input.stream_id ||
      output.pts_ns != input.pts_ns) {
    correlation_diagnostic(input, output, frame, run_sequence);
    uncertain("pulled output did not establish this frame's completion");
  }
  return output;
}
using Nodes = std::function<void(Graph&)>;
using Check = std::function<void(const Sample&, std::size_t)>;
unsigned cases = 0, frames = 0;
void run_case(const std::string& label, const std::vector<Image>& inputs, const Nodes& nodes,
              unsigned stages, const Check& check, bool mixed_ev_a65 = false) {
  std::cout << "CASE_BEGIN " << label << "\n" << std::flush;
  std::vector<Tensor> tensors;
  for (const auto& input : inputs)
    tensors.push_back(device_tensor(input));
  GraphOptions go;
  go.processcvu_requested_run_target = "EV74";
  if (mixed_ev_a65) {
    go.processcvu.pre_run_target = "EV74";
    go.processcvu.post_run_target = "A65";
  }
  Graph graph(go);
  InputOptions in;
  in.payload_type = PayloadType::Image;
  in.format = name(inputs[0].format);
  in.memory_policy = InputMemoryPolicy::Ev74;
  in.pool_min_buffers = 3;
  in.pool_max_buffers = 3;
  graph.add(nodes::Input("image", in));
  nodes(graph);
  OutputOptions out;
  out.sync = false;
  out.drop = false;
  out.max_buffers = 1;
  graph.add(nodes::Output(out));
  RunOptions options;
  options.output_memory = OutputMemory::ZeroCopy;
  options.queue_depth = 2;
  options.preset = RunPreset::Realtime;
  for (const auto& tensor : tensors)
    options.advanced.max_input_bytes =
        std::max<std::size_t>(options.advanced.max_input_bytes, tensor.storage->size_bytes);
  auto run = [&]() {
    try {
      return graph.build(TensorList{tensors[0]}, options);
    } catch (...) {
      // EV context creation can accept a setup-probe DMA descriptor before any
      // application frame. A failed build has no returned retirement witness.
      uncertain("EV graph startup failed without a verifiable setup ownership barrier");
    }
  }();
  std::optional<Sample> held;
  try {
    const auto inspection = inspect(run, stages, mixed_ev_a65);
    for (std::size_t i = 0; i < inputs.size(); ++i) {
      auto output = transact(run, inspection, tensors[i], frames++, i);
      check(output, i);
      if (!held)
        held = output;
      check(*held, 0);
    }
    run.close();
    check(*held, 0);
    std::cout << "CASE_CLOSED " << label << " frames=" << inputs.size() << "\n" << std::flush;
    ++cases;
  } catch (...) {
    // Every reachable assertion follows a completed pull or cold preparation.
    // No accepted unknown work is unwound through this failure path.
    run.close();
    std::cerr << "CASE_CLOSED_FAILURE " << label << "\n" << std::flush;
    throw;
  }
}
std::vector<Image> inputs(F f, int w, int h) {
  return {pattern(f, w, h, 1), pattern(f, w, h, 2), pattern(f, w, h, 3)};
}
void numerical() {
  for (F f : {F::RGB, F::BGR, F::GRAY8, F::NV12, F::I420, F::NV16})
    for (I interpolation : {I::Nearest, I::Bilinear, I::Bicubic, I::BoxAverage}) {
      // Exact factor two still exercises all interpolation arithmetic and tails.
      auto source = inputs(f, 36, 20);
      std::vector<Image> expected;
      for (const auto& in : source)
        expected.push_back(resize_reference(in, 18, 10, interpolation));
      ResizeOptions o;
      o.width = 18;
      o.height = 10;
      o.interpolation = interpolation;
      run_case(
          std::string("resize-") + name(f) + "-" + std::to_string(int(interpolation)), source,
          [=](Graph& g) { g.add(nodes::Resize(o)); }, 1,
          [&](const Sample& s, std::size_t i) { check_raw(s, expected[i]); });
    }
  for (F in : {F::RGB, F::BGR, F::GRAY8})
    for (F out : {F::NV12, F::I420, F::NV16})
      for (auto matrix : {ImageColorMatrix::Bt601, ImageColorMatrix::Bt709}) {
        const auto source = inputs(in, 18, 10);
        ColorConvertOptions o;
        o.output_format = out;
        o.color.matrix = matrix;
        std::vector<Image> expected;
        for (const auto& v : source)
          expected.push_back(color_reference(v, out, o.color));
        run_case(
            std::string("color-") + name(in) + "-" + name(out) + "-" + std::to_string(int(matrix)),
            source, [=](Graph& g) { g.add(nodes::ColorConvert(o)); }, 1,
            [&](const Sample& s, std::size_t i) { check_raw(s, expected[i]); });
      }
  // The existing full-range YUV-to-packed route must remain reachable from the
  // thin frontend. Neutral chroma gives an independent exact grayscale oracle.
  for (F in : {F::NV12, F::I420, F::NV16})
    for (F out : {F::RGB, F::BGR}) {
      auto source = inputs(in, 18, 10);
      std::vector<Image> expected;
      for (auto& v : source) {
        for (std::size_t p = 1; p < v.planes.size(); ++p)
          std::fill(v.planes[p].bytes.begin(), v.planes[p].bytes.end(), 128);
        auto gray = image(out, v.width, v.height);
        for (int y = 0; y < v.height; ++y)
          for (int x = 0; x < v.width; ++x)
            for (int c = 0; c < 3; ++c)
              gray.planes[0].at(x, y, c) = v.planes[0].at(x, y);
        expected.push_back(std::move(gray));
      }
      ColorConvertOptions o;
      o.output_format = out;
      run_case(
          std::string("neutral-") + name(in) + "-" + name(out), source,
          [=](Graph& g) { g.add(nodes::ColorConvert(o)); }, 1,
          [&](const Sample& s, std::size_t i) { check_raw(s, expected[i]); });
    }
  for (F in : {F::NV12, F::I420, F::NV16})
    for (F out : {F::NV12, F::I420, F::NV16}) {
      const auto source = inputs(in, 18, 10);
      ColorConvertOptions o;
      o.output_format = out;
      std::vector<Image> expected;
      for (const auto& v : source)
        expected.push_back(color_reference(v, out, o.color));
      run_case(
          std::string("direct-") + name(in) + "-" + name(out), source,
          [=](Graph& g) { g.add(nodes::ColorConvert(o)); }, 1,
          [&](const Sample& s, std::size_t i) { check_raw(s, expected[i]); });
    }
  for (F f : {F::NV12, F::I420, F::NV16}) {
    const auto source = inputs(f, 18, f == F::NV16 ? 9 : 10);
    ColorConvertOptions o;
    o.output_format = f;
    o.color.range = ImageYuvRange::Limited;
    o.color.output_siting = {ImageChromaSiting::Cosited, ImageChromaSiting::Cosited};
    std::vector<Image> expected;
    for (const auto& v : source)
      expected.push_back(color_reference(v, f, o.color));
    run_case(
        std::string("limited-resite-") + name(f), source,
        [=](Graph& g) { g.add(nodes::ColorConvert(o)); }, 1,
        [&](const Sample& s, std::size_t i) { check_raw(s, expected[i]); });
  }
  NormalizationOptions normalization;
  normalization.mean = {.125f, .25f, .5f};
  normalization.stddev = {.25f, .5f, 1.0f};
  for (F f : {F::RGB, F::BGR, F::GRAY8}) {
    const auto source = inputs(f, 19, 7);
    run_case(
        std::string("normalization-") + name(f), source,
        [=](Graph& g) { g.add(nodes::Normalization(normalization)); }, 1,
        [&](const Sample& s, std::size_t i) { check_normalized(s, source[i], normalization); });
  }
  {
    const std::vector<Image> source{pattern(F::RGB, 36, 20, 1), pattern(F::RGB, 18, 10, 2),
                                    pattern(F::RGB, 24, 16, 3), pattern(F::RGB, 36, 20, 4),
                                    pattern(F::RGB, 18, 10, 5)};
    ResizeOptions r;
    r.width = 18;
    r.height = 10;
    r.max_input_width = 36;
    r.max_input_height = 20;
    r.interpolation = I::Nearest;
    ColorConvertOptions c;
    c.output_format = F::BGR;
    std::vector<Image> expected;
    for (const auto& v : source)
      expected.push_back(color_reference(resize_reference(v, 18, 10, I::Nearest), F::BGR));
    run_case(
        "dynamic-resize-color-normalize", source,
        [=](Graph& g) {
          g.add(nodes::Resize(r));
          g.add(nodes::ColorConvert(c));
          g.add(nodes::Normalization(normalization));
        },
        3,
        [&](const Sample& s, std::size_t i) {
          check_normalized(s, expected[i], normalization);
          const auto t = only_tensor(s);
          require(t.semantic.preprocess &&
                      t.semantic.preprocess->original_width == source[i].width &&
                      t.semantic.preprocess->original_height == source[i].height,
                  "chained original geometry stale");
          const auto& m = *t.semantic.preprocess;
          require(std::abs(m.affine_m00 - double(source[i].width) / 18) < 1e-12 &&
                      std::abs(m.affine_m11 - double(source[i].height) / 10) < 1e-12 &&
                      m.affine_m01 == 0 && m.affine_m10 == 0 && m.affine_m02 == 0 &&
                      m.affine_m12 == 0,
                  "chained output-to-original affine was replaced rather than composed");
        });
  }
  {
    const auto source = inputs(F::RGB, 18, 10);
    ResizeOptions o;
    o.width = 18;
    o.height = 18;
    o.fit = ResizeFit::Contain;
    o.pad_value = 23;
    o.interpolation = I::Nearest;
    std::vector<Image> expected;
    for (const auto& v : source)
      expected.push_back(resize_reference(v, 18, 18, I::Nearest, true, 23));
    run_case(
        "resize-contain", source, [=](Graph& g) { g.add(nodes::Resize(o)); }, 1,
        [&](const Sample& s, std::size_t i) { check_raw(s, expected[i]); });
  }
  for (F format : {F::NV12, F::I420, F::NV16}) {
    const auto source = inputs(format, 18, 10);
    constexpr std::array<unsigned char, 3> padding{19, 91, 173};
    ResizeOptions o;
    o.width = 18;
    o.height = 18;
    o.fit = ResizeFit::Contain;
    o.interpolation = I::Nearest;
    o.yuv_padding = padding;
    std::vector<Image> expected;
    for (const auto& input : source) {
      // No scaling is needed: center the complete 18x10 image in an 18x18
      // canvas. Independently fill each component, then copy the active rows;
      // chroma has two padding rows per edge for 4:2:0 and four for NV16.
      auto canvas = image(format, 18, 18);
      for (std::size_t p = 0; p < canvas.planes.size(); ++p) {
        auto& plane = canvas.planes[p];
        for (int y = 0; y < plane.height; ++y)
          for (int x = 0; x < plane.width; ++x)
            for (int c = 0; c < plane.channels; ++c)
              plane.at(x, y, c) = padding[p == 0 ? 0 : format == F::I420 ? p : 1 + c];
        const int first_row = p ? 4 / vertical(format) : 4;
        const auto& active = input.planes[p];
        for (int y = 0; y < active.height; ++y)
          for (int x = 0; x < active.width; ++x)
            for (int c = 0; c < active.channels; ++c)
              plane.at(x, y + first_row, c) = active.at(x, y, c);
      }
      expected.push_back(std::move(canvas));
    }
    run_case(
        std::string("resize-contain-padding-") + name(format), source,
        [=](Graph& g) { g.add(nodes::Resize(o)); }, 1,
        [&](const Sample& s, std::size_t i) { check_raw(s, expected[i]); });
  }
  {
    const auto source = inputs(F::RGB, 66, 34);
    ColorConvertOptions o;
    o.output_format = F::NV12;
    o.surface = ImageSurfaceLayout::EncoderNv12;
    std::vector<Image> expected;
    for (const auto& v : source)
      expected.push_back(color_reference(v, F::NV12));
    run_case(
        "encoder-nv12-storage", source, [=](Graph& g) { g.add(nodes::ColorConvert(o)); }, 1,
        [&](const Sample& s, std::size_t i) { check_raw(s, expected[i], true); });
  }
}

// Read-only handoff probes retain identities, never buffers, maps or loans.
// A separate qualification mode: existing default catalogue/counts are unchanged.
// Normalization is EV-only; the public terminal Cast provides a genuine checked
// A65 consumer of that EV-produced BF16 surface, not a CPU output-map substitute.
void mixed_ev_a65() {
  const std::vector<Image> source{pattern(F::RGB, 36, 20, 1),
                                  pattern(F::RGB, 18, 10, 2),
                                  pattern(F::RGB, 36, 20, 3)};
  ResizeOptions resize;
  resize.width = 18;
  resize.height = 10;
  resize.max_input_width = 36;
  resize.max_input_height = 20;
  resize.interpolation = I::Nearest;
  NormalizationOptions normalization;
  normalization.mean = {.125f, .25f, .5f};
  normalization.stddev = {.25f, .5f, 1.f};
  CastOptions cast;
  cast.direction = CastDirection::Bf16ToFp32;
  // Existing public pre/post placement classifies this authored terminal stage
  // as Post. Live stage inspection below verifies the resolved target, not just
  // its name or requested policy.
  cast.element_name = "post_cast";
  std::vector<Image> expected;
  for (const auto& input : source)
    expected.push_back(resize_reference(input, 18, 10, I::Nearest));
  run_case(
      "mixed-ev-a65-resize-normalize-cast", source,
      [=](Graph& graph) {
        graph.add(nodes::Resize(resize));
        graph.add(nodes::Normalization(normalization));
        graph.add(nodes::Cast(cast));
      },
      3,
      [&](const Sample& sample, std::size_t index) {
        const auto tensor = only_tensor(sample);
        const auto& plane = expected[index].planes[0];
        require(tensor.dtype == TensorDType::Float32 && !tensor.semantic.image &&
                    tensor.shape == std::vector<std::int64_t>({10, 18, 3}),
                "mixed chain did not publish a dense FP32 tensor");
        const auto mapped = tensor.map(MapMode::Read);
        require(mapped.data && mapped.size_bytes >= plane.bytes.size() * sizeof(float),
                "mixed chain output map failed");
        const auto* values = static_cast<const std::uint32_t*>(mapped.data);
        for (std::size_t i = 0; i < plane.bytes.size(); ++i) {
          const auto channel = i % 3;
          const float offset = float(normalization.mean[channel] * 255.0);
          const float multiplier = float(1.0 / (normalization.stddev[channel] * 255.0));
          const float shifted = float(plane.bytes[i]) - offset;
          const float normalized = shifted * multiplier;
          // Independent Preproc BF16 truncation followed by exact BF16->FP32
          // expansion. No runtime numerical helper participates in this oracle.
          const auto want = std::bit_cast<std::uint32_t>(normalized) & 0xffff0000U;
          require(values[i] == want,
                  "mixed chain pixel mismatch index=" + std::to_string(i));
        }
        require(tensor.semantic.preprocess &&
                    tensor.semantic.preprocess->original_width == source[index].width &&
                    tensor.semantic.preprocess->original_height == source[index].height,
                "mixed chain lost original geometry metadata");
        const auto& m = *tensor.semantic.preprocess;
        require(std::abs(m.affine_m00 - double(source[index].width) / 18) < 1e-12 &&
                    std::abs(m.affine_m11 - double(source[index].height) / 10) < 1e-12 &&
                    m.affine_m01 == 0 && m.affine_m10 == 0 && m.affine_m02 == 0 && m.affine_m12 == 0,
                "mixed chain lost output-to-original affine metadata");
      },
      true);
}
struct Handoff {
  static constexpr unsigned frame_rate = 30;
  // Three immutable slots: publish with release/acquire; never block streaming threads.
  std::array<sima_test::DmaBufSpan, 3> produced{};
  std::array<std::atomic<bool>, 3> ready{}, consumed{};
  std::atomic<unsigned> received{0}, error{0};
  int width = 130, height = 66;
  void observe(GstBuffer* buffer, bool source) noexcept {
    try {
      const auto span = sima_test::dmabuf_span(buffer);
      const auto* meta = gst_buffer_get_video_meta(buffer);
      const unsigned pitch = (width + 63) / 64 * 64, rows = (height + 7) / 8 * 8;
      if (!meta || meta->format != GST_VIDEO_FORMAT_NV12 || meta->width != unsigned(width) ||
          meta->height != unsigned(height) || meta->stride[0] != int(pitch) ||
          meta->stride[1] != int(pitch) || meta->offset[1] != meta->offset[0] + pitch * rows) {
        error.store(1);
        return;
      }
      const auto pts = GST_BUFFER_PTS(buffer);
      std::size_t index = 0;
      while (index < produced.size() && pts != frame_pts(index, frame_rate))
        ++index;
      if (index == produced.size()) {
        error.store(2);
        return;
      }
      if (source) {
        if (ready[index].load(std::memory_order_acquire)) {
          error.store(3);
          return;
        }
        produced[index] = span;
        ready[index].store(true, std::memory_order_release);
      } else if (!ready[index].load(std::memory_order_acquire) || produced[index] != span) {
        error.store(4);
      } else if (consumed[index].exchange(true)) {
        error.store(6);
      } else {
        received.fetch_add(1);
      }
    } catch (...) {
      error.store(5);
    }
  }
};
struct Probe {
  Handoff* handoff;
  bool source;
};
GstPadProbeReturn observe(GstPad*, GstPadProbeInfo* info, gpointer data) {
  auto* probe = static_cast<Probe*>(data);
  if (auto* buffer = GST_PAD_PROBE_INFO_BUFFER(info))
    probe->handoff->observe(buffer, probe->source);
  return GST_PAD_PROBE_OK;
}
// Small baseline-profile SPS parser: proves actual encoded visible dimensions,
// not merely the authored caps. No decoder/provider fallback is used.
struct Bits {
  std::vector<std::uint8_t> bytes;
  std::size_t at = 0;
  unsigned take(unsigned n) {
    require(n <= 32 && at + n <= bytes.size() * 8, "truncated H264 SPS");
    unsigned v = 0;
    while (n--) {
      v = (v << 1) | ((bytes[at / 8] >> (7 - at % 8)) & 1);
      ++at;
    }
    return v;
  }
  unsigned ue() {
    unsigned zeros = 0;
    while (!take(1)) {
      require(++zeros < 31, "invalid H264 Exp-Golomb");
    }
    return ((1U << zeros) - 1) + take(zeros);
  }
  int se() {
    const auto v = ue();
    return v & 1 ? int((v + 1) / 2) : -int(v / 2);
  }
};
bool check_sps(const std::uint8_t* data, std::size_t size, int width, int height) {
  for (std::size_t i = 0; i + 4 < size; ++i) {
    if (data[i] || data[i + 1] || data[i + 2] != 1 || (data[i + 3] & 31) != 7)
      continue;
    std::size_t end = i + 4;
    while (end + 3 < size && !(data[end] == 0 && data[end + 1] == 0 &&
                               (data[end + 2] == 1 || (data[end + 2] == 0 && data[end + 3] == 1))))
      ++end;
    if (end + 3 >= size)
      end = size;
    Bits bits;
    unsigned zeros = 0;
    for (std::size_t j = i + 4; j < end; ++j) {
      if (zeros >= 2 && data[j] == 3) {
        zeros = 0;
        continue;
      }
      bits.bytes.push_back(data[j]);
      zeros = data[j] == 0 ? zeros + 1 : 0;
    }
    require(bits.take(8) == 66, "encoder did not emit requested baseline SPS");
    bits.take(8);
    bits.take(8);
    bits.ue();
    bits.ue();
    const auto poc = bits.ue();
    if (poc == 0)
      bits.ue();
    else if (poc == 1) {
      bits.take(1);
      bits.se();
      bits.se();
      const auto n = bits.ue();
      require(n < 256, "invalid POC cycle");
      for (unsigned j = 0; j < n; ++j)
        bits.se();
    }
    bits.ue();
    bits.take(1);
    const unsigned wm = bits.ue() + 1, hm = bits.ue() + 1, frame_only = bits.take(1);
    if (!frame_only)
      bits.take(1);
    bits.take(1);
    unsigned l = 0, r = 0, t = 0, b = 0;
    if (bits.take(1)) {
      l = bits.ue();
      r = bits.ue();
      t = bits.ue();
      b = bits.ue();
    }
    require(int(wm * 16 - 2 * (l + r)) == width &&
                int(hm * 16 * (2 - frame_only) - 2 * (2 - frame_only) * (t + b)) == height,
            "encoded SPS visible dimensions mismatch");
    return true;
  }
  return false;
}
// Encoder output slots may remain queued after the last visible frame. A void
// Run::close() is not their ownership barrier: require the retained element's
// completed NULL transition, which only succeeds after the codec stop/destroy
// barrier and callback-thread join. Failure or an unfinished transition parks
// before releasing the retained element, pads, inputs, or probe contexts.
void close_encoder_retired(Run& run, GstElement* encoder) {
  try {
    run.close();
  } catch (...) {
    uncertain("encoder close threw before establishing its ownership barrier");
  }
  if (!encoder)
    uncertain("encoder teardown has no retained element for its ownership proof");
  GstState state = GST_STATE_VOID_PENDING, pending = GST_STATE_VOID_PENDING;
  const auto result = gst_element_get_state(encoder, &state, &pending, 0);
  if (result != GST_STATE_CHANGE_SUCCESS || state != GST_STATE_NULL ||
      pending != GST_STATE_VOID_PENDING)
    uncertain("encoder stop did not establish a completed NULL ownership barrier");
}
void encoder() {
  std::cout << "CASE_BEGIN encoder-nv12-zero-copy\n" << std::flush;
  require(element_exists("neatencoder"), "required neatencoder provider unavailable");
  Handoff handoff;
  Probe produced{&handoff, true}, received{&handoff, false};
  const auto source = inputs(F::RGB, handoff.width, handoff.height);
  std::vector<Tensor> tensors;
  for (const auto& v : source)
    tensors.push_back(device_tensor(v));
  GraphOptions go;
  go.processcvu_requested_run_target = "EV74";
  Graph graph(go);
  InputOptions in;
  in.payload_type = PayloadType::Image;
  in.format = FormatTag::RGB;
  in.fps_n = Handoff::frame_rate;
  in.fps_d = 1;
  in.memory_policy = InputMemoryPolicy::Ev74;
  graph.add(nodes::Input("image", in));
  ColorConvertOptions color;
  color.output_format = F::NV12;
  color.surface = ImageSurfaceLayout::EncoderNv12;
  graph.add(nodes::ColorConvert(color));
  graph.add(nodes::H264EncodeSima(handoff.width, handoff.height, Handoff::frame_rate, 1000,
                                  "baseline", "4.0"));
  OutputOptions out;
  out.sync = false;
  out.drop = false;
  out.max_buffers = 1;
  graph.add(nodes::Output(out));
  RunOptions options;
  options.output_memory = OutputMemory::ZeroCopy;
  options.queue_depth = 2;
  auto run = [&]() {
    try {
      return graph.build(TensorList{tensors[0]}, options);
    } catch (...) {
      // Codec startup can queue output DMA even before the first application
      // frame. Without a returned Run/element, its terminal state is unknown.
      uncertain("encoder graph startup failed without a verifiable ownership barrier");
    }
  }();
  // Public close resets its handle even if a worker owns deferred teardown.
  // Keep the actual Core owner independently until the codec barrier is proven.
  const auto run_owner = run_internal::core(run);
  if (!run_owner)
    uncertain("encoder graph lost its Core ownership witness");
  std::unique_ptr<GstElement, decltype(&gst_object_unref)> encoder_owner(nullptr, gst_object_unref);
  std::shared_ptr<GstPad> from, to;
  std::optional<Inspected> inspection;
  gulong a = 0, b = 0;
  try {
    inspection = inspect(run, 1);
    const auto& stage = *inspection;
    if (!stage.encoder)
      uncertain("encoder graph has no observable encoder ownership barrier");
    encoder_owner.reset(GST_ELEMENT(gst_object_ref(stage.encoder)));
    from = {gst_element_get_static_pad(stage.runners[0], "src"), gst_object_unref};
    to = {gst_element_get_static_pad(stage.encoder, "sink"), gst_object_unref};
    require(from && to, "encoder handoff pads missing");
    a = gst_pad_add_probe(from.get(), GST_PAD_PROBE_TYPE_BUFFER, observe, &produced, nullptr);
    b = gst_pad_add_probe(to.get(), GST_PAD_PROBE_TYPE_BUFFER, observe, &received, nullptr);
    require(a && b, "encoder handoff observation failed");
    bool sps = false;
    for (unsigned i = 0; i < tensors.size(); ++i) {
      auto output = transact(run, stage, tensors[i], frames++, i, Handoff::frame_rate);
      const auto tensor = only_tensor(output);
      require(tensor.semantic.encoded.has_value(), "encoder did not publish encoded output");
      auto map = tensor.map(MapMode::Read);
      require(map.data && map.size_bytes > 5, "empty encoded access unit");
      sps = check_sps(static_cast<const std::uint8_t*>(map.data), map.size_bytes, handoff.width,
                      handoff.height) ||
            sps;
    }
    close_encoder_retired(run, encoder_owner.get());
    gst_pad_remove_probe(from.get(), a);
    a = 0;
    gst_pad_remove_probe(to.get(), b);
    b = 0;
    require(handoff.error.load() == 0,
            "encoder handoff error code=" + std::to_string(handoff.error.load()));
    require(handoff.received.load() == tensors.size() && sps,
            "missing zero-copy handoffs or baseline SPS");
    ++cases;
    std::cout << "CASE_CLOSED encoder-nv12-zero-copy frames=" << tensors.size() << "\n";
  } catch (...) {
    // Inspection can fail while codec startup already owns queued output slots.
    if (!encoder_owner)
      uncertain("encoder preparation failed before retaining its ownership witness");
    close_encoder_retired(run, encoder_owner.get());
    if (a)
      gst_pad_remove_probe(from.get(), a);
    if (b)
      gst_pad_remove_probe(to.get(), b);
    std::cerr << "CASE_CLOSED_FAILURE encoder-nv12-zero-copy\n";
    throw;
  }
}
} // namespace
int main(int argc, char** argv) {
  try {
    const bool encode = argc == 2 && std::string_view(argv[1]) == "--encoder";
    const bool mixed = argc == 2 && std::string_view(argv[1]) == "--mixed-ev-a65";
    require(argc == 1 || encode || mixed,
            "usage: image_transform_ev_test [--encoder|--mixed-ev-a65]");
    require(std::fesetround(FE_TONEAREST) == 0, "reference requires round-to-nearest-even");
    setenv("SIMA_GST_RUN_INSERT_BOUNDARIES", "0", 1);
    require(element_exists("neatkernelrunner"), "required neatkernelrunner provider unavailable");
    if (encode)
      encoder();
    else if (mixed)
      mixed_ev_a65();
    else
      numerical();
    std::cout << "PASS image_transform_ev_test cases=" << cases << " completed_frames=" << frames
              << " all_runs_closed=1\n";
    return 0;
  } catch (const std::exception& error) {
    // Uncertain accepted work never reaches this catch. This suffix is the
    // source-audited natural failure path, not evidence inferred from a timeout.
    std::cerr << "FAIL case=image_transform_ev_test cases=" << cases
              << " completed_frames=" << frames << " error=" << error.what()
              << " source_closed=1\n";
    return 1;
  }
}
