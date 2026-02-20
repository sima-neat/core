#pragma once

#include "test_utils.h"

#include "pipeline/Session.h"
#include "nodes/common/Output.h"
#include "nodes/io/Input.h"

#include <chrono>
#include <cstddef>
#include <string>
#include <thread>

namespace sima_test {

struct InputstreamRenegotiateSpec {
  std::string format;
  std::string gst_element;
  int first_w = 0;
  int first_h = 0;
  int second_w = 0;
  int second_h = 0;
  int stability_frames = 2;
  int timeout_ms = 2000;
  size_t max_input_bytes = 0;
  bool check_output = false;
};

struct InputstreamFormatChangeSpec {
  int width = 0;
  int height = 0;
  size_t max_input_bytes = 0;
  bool expect_throw = false;
  int timeout_ms = 1000;
  int reneg_timeout_ms = 1000;
  uint8_t rgb_fill = 0x7f;
};

inline size_t default_pool_bytes(const InputstreamRenegotiateSpec& spec) {
  if (spec.format == "NV12") {
    return static_cast<size_t>(spec.second_w * spec.second_h * 3 / 2);
  }
  return static_cast<size_t>(spec.second_w * spec.second_h * 3);
}

inline void require_output_shape(const simaai::neat::Sample& out, int w, int h, const char* label) {
  require(out.tensor.has_value(), std::string(label) + ": missing output tensor");
  require(out.tensor->shape.size() >= 2, std::string(label) + ": missing shape");
  require(out.tensor->shape[0] == h, std::string(label) + ": height mismatch");
  require(out.tensor->shape[1] == w, std::string(label) + ": width mismatch");
}

inline size_t default_format_change_pool_bytes(const InputstreamFormatChangeSpec& spec) {
  return static_cast<size_t>(spec.width * spec.height * 3);
}

inline simaai::neat::RunPreset preset_for_stability(int stability_frames) {
  return (stability_frames <= 1) ? simaai::neat::RunPreset::Realtime
                                 : simaai::neat::RunPreset::Balanced;
}

template <typename MakeInputFn>
inline void run_inputstream_renegotiate_test(const InputstreamRenegotiateSpec& spec,
                                             MakeInputFn make_input) {
  using namespace simaai::neat;

  Session p;
  InputOptions src_opt;
  src_opt.media_type = "video/x-raw";
  src_opt.format = spec.format;
  src_opt.use_simaai_pool = false;
  p.add(nodes::Input(src_opt));
  if (!spec.gst_element.empty())
    p.custom(spec.gst_element);
  p.add(nodes::Output(OutputOptions::Latest()));

  auto first = make_input(spec.first_w, spec.first_h);
  auto second = make_input(spec.second_w, spec.second_h);

  RunOptions opt;
  opt.preset = preset_for_stability(spec.stability_frames);
  opt.queue_depth = 2;
  opt.overflow_policy = OverflowPolicy::KeepLatest;
  opt.advanced.max_input_bytes =
      spec.max_input_bytes > 0 ? spec.max_input_bytes : default_pool_bytes(spec);

  Run run = p.build(first, RunMode::Async, opt);

  if (spec.check_output) {
    Sample out = run.push_and_pull(first, spec.timeout_ms);
    require_output_shape(out, spec.first_w, spec.first_h, "initial");
  } else {
    require(run.push(first), "initial push failed");
  }

  if (spec.check_output) {
    Sample out_same = run.push_and_pull(first, spec.timeout_ms);
    require_output_shape(out_same, spec.first_w, spec.first_h, "same caps");
  } else {
    require(run.push(first), "push identical caps failed");
  }
  require(run.input_stats().renegotiations == 0, "unexpected renegotiation on identical caps");

  if (spec.stability_frames > 1) {
    if (spec.check_output) {
      (void)run.push_and_pull(second, spec.timeout_ms);
    } else {
      require(run.push(second), "push new size failed");
    }
    std::this_thread::sleep_for(std::chrono::milliseconds(50));
    require(run.input_stats().renegotiations == 0, "unexpected renegotiation before stability");
    if (spec.check_output) {
      (void)run.push_and_pull(second, spec.timeout_ms);
    } else {
      require(run.push(second), "push new size #2 failed");
    }
    require(wait_for_reneg(run, 1, 1000), "renegotiation not observed after stability");
  } else {
    if (spec.check_output) {
      Sample out2 = run.push_and_pull(second, spec.timeout_ms);
      require_output_shape(out2, spec.second_w, spec.second_h, "after resize");
    } else {
      require(run.push(second), "push new size failed");
    }
    (void)wait_for_reneg(run, 1, 500);
  }

  require(run.input_stats().renegotiations == 1, "expected exactly one renegotiation");
}

inline void run_inputstream_format_change_test(const InputstreamFormatChangeSpec& spec) {
  using namespace simaai::neat;

  require(spec.width > 0 && spec.height > 0, "invalid format-change dimensions");

  simaai::neat::Tensor nv12 = make_nv12_tensor(spec.width, spec.height);
  simaai::neat::Tensor rgb = make_color_tensor(
      spec.width, spec.height, simaai::neat::ImageSpec::PixelFormat::RGB, spec.rgb_fill);

  Session p;
  InputOptions src_opt;
  src_opt.media_type = "video/x-raw";
  src_opt.format = "";
  src_opt.use_simaai_pool = false;
  p.add(nodes::Input(src_opt));
  p.add(nodes::Output(OutputOptions::Latest()));

  RunOptions opt;
  opt.preset = RunPreset::Realtime;
  opt.queue_depth = 2;
  opt.overflow_policy = OverflowPolicy::KeepLatest;
  opt.advanced.max_input_bytes =
      spec.max_input_bytes > 0 ? spec.max_input_bytes : default_format_change_pool_bytes(spec);

  Run run = p.build(nv12, RunMode::Async, opt);

  if (spec.expect_throw) {
    (void)run.push_and_pull(nv12, spec.timeout_ms);
    bool threw = false;
    try {
      (void)run.push_and_pull(rgb, spec.timeout_ms);
    } catch (const std::exception&) {
      threw = true;
    }
    require(threw, "expected format change failure");
    return;
  }

  require(run.push(nv12), "initial NV12 push failed");
  require(run.push(rgb), "push RGB input failed");
  require(wait_for_reneg(run, 1, spec.reneg_timeout_ms),
          "format change renegotiation not observed");
  require(run.input_stats().renegotiations == 1,
          "expected single renegotiation after format change");
  require(run.last_error().empty(), "unexpected error on format change");
}

} // namespace sima_test
