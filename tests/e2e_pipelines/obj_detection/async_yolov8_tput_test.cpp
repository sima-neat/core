#include "pipeline/Graph.h"
#include "nodes/common/Output.h"
#include "nodes/groups/ModelGroups.h"
#include "nodes/io/Input.h"
#include "nodes/sima/SimaBoxDecode.h"
#include "model/Model.h"
#include "pipeline/runtime/RunInternal.h"

#include "e2e_pipelines/e2e_utils.h"
#include "e2e_pipelines/obj_detection/obj_detection_utils.h"
#include "e2e_pipelines/obj_detection/yolov8_test_utils.h"
#include "gst/GstInit.h"
#include "test_utils.h"

#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstdint>
#include <cstdlib>
#include <cctype>
#include <cstring>
#include <filesystem>
#include <iomanip>
#include <iostream>
#include <mutex>
#include <optional>
#include <sstream>
#include <string>
#include <thread>
#include <vector>

namespace fs = std::filesystem;

namespace {

using sima_yolov8_test::append_note;
using sima_yolov8_test::env_bool;
using sima_yolov8_test::env_int;
using sima_yolov8_test::sanitize_note;
using sima_yolov8_test::step_log;

struct AsyncTestConfig {
  int iters = 200;
  int warm = 200;
  int inflight = 4;
  int excluded_preproc_dispatches = 5;
  int topk = 100;
  double min_fps = 350.0;
  float boxdecode_score_threshold = 0.49f;
  float min_score = 0.49f;
  float min_iou = 0.30f;
  bool skip_boxdecode = false;
  bool profile = false;
  bool profile_emit_run_report = false;
  bool profile_skip_extract = true;
  bool profile_skip_validate = false;
  int profile_push_slow_ms = 2;
  int pull_timeout_ms = 1000;
  int max_timeouts = 3;
  int input_width = 544;
  int input_height = 306;
  simaai::neat::BoxDecodeType decode_type = simaai::neat::BoxDecodeType::YoloV8;
};

struct RunSummary {
  bool ok = false;
  int outputs = 0;
  double avg_fps = 0.0;
  std::string note;
  std::string diagnostics;
};

std::string maybe_collect_run_report(simaai::neat::Run& run, bool enabled,
                                     const simaai::neat::MeasureReport* report = nullptr) {
  if (!enabled) {
    return {};
  }
  if (report != nullptr) {
    return report->to_text();
  }
  const std::string last = run.last_error();
  return last.empty() ? std::string{} : ("last_error=" + last);
}

std::optional<simaai::neat::BoxDecodeType> parse_boxdecode_type_token(const char* raw) {
  if (raw == nullptr || *raw == '\0') {
    return std::nullopt;
  }
  std::string token(raw);
  std::transform(token.begin(), token.end(), token.begin(), [](unsigned char c) {
    if (c == '_' || c == '.') {
      return '-';
    }
    return static_cast<char>(std::tolower(c));
  });
  auto is = [&](const char* value) { return token == value; };
  if (is("yolo") || is("yolo-generic"))
    return simaai::neat::BoxDecodeType::Yolo;
  if (is("yolov5") || is("yolo5") || is("yolo-v5"))
    return simaai::neat::BoxDecodeType::YoloV5;
  if (is("yolov5-seg") || is("yolo5-seg") || is("yolo-v5-seg"))
    return simaai::neat::BoxDecodeType::YoloV5Seg;
  if (is("yolov7") || is("yolo7") || is("yolo-v7"))
    return simaai::neat::BoxDecodeType::YoloV7;
  if (is("yolov7-seg") || is("yolo7-seg") || is("yolo-v7-seg"))
    return simaai::neat::BoxDecodeType::YoloV7Seg;
  if (is("yolov8") || is("yolo8") || is("yolo-v8"))
    return simaai::neat::BoxDecodeType::YoloV8;
  if (is("yolov8-seg") || is("yolo8-seg") || is("yolo-v8-seg"))
    return simaai::neat::BoxDecodeType::YoloV8Seg;
  if (is("yolov8-pose") || is("yolo8-pose") || is("yolo-v8-pose"))
    return simaai::neat::BoxDecodeType::YoloV8Pose;
  if (is("yolov9") || is("yolo9") || is("yolo-v9"))
    return simaai::neat::BoxDecodeType::YoloV9;
  if (is("yolov9-seg") || is("yolo9-seg") || is("yolo-v9-seg"))
    return simaai::neat::BoxDecodeType::YoloV9Seg;
  if (is("yolov10") || is("yolo10") || is("yolo-v10"))
    return simaai::neat::BoxDecodeType::YoloV10;
  if (is("yolov10-seg") || is("yolo10-seg") || is("yolo-v10-seg"))
    return simaai::neat::BoxDecodeType::YoloV10Seg;
  if (is("yolo26") || is("yolov26") || is("yolo-v26"))
    return simaai::neat::BoxDecodeType::YoloV26;
  if (is("yolo26-pose") || is("yolov26-pose") || is("yolo-v26-pose"))
    return simaai::neat::BoxDecodeType::YoloV26Pose;
  if (is("yolo26-seg") || is("yolov26-seg") || is("yolo-v26-seg"))
    return simaai::neat::BoxDecodeType::YoloV26Seg;
  if (is("yolov6") || is("yolo6") || is("yolo-v6"))
    return simaai::neat::BoxDecodeType::YoloV6;
  if (is("yolox") || is("yolo-x"))
    return simaai::neat::BoxDecodeType::YoloX;
  return std::nullopt;
}

double env_double(const char* name, double def) {
  const char* raw = std::getenv(name);
  if (!raw || !*raw)
    return def;
  char* end = nullptr;
  const double parsed = std::strtod(raw, &end);
  if (end == raw) {
    return def;
  }
  return parsed;
}

std::uint64_t duration_ns(const std::chrono::steady_clock::time_point& start,
                          const std::chrono::steady_clock::time_point& end) {
  if (end <= start) {
    return 0;
  }
  return static_cast<std::uint64_t>(
      std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count());
}

double ns_to_ms(const std::uint64_t ns) {
  return static_cast<double>(ns) / 1000000.0;
}

struct AtomicNsStats {
  std::atomic<std::uint64_t> count{0};
  std::atomic<std::uint64_t> total_ns{0};
  std::atomic<std::uint64_t> max_ns{0};

  void observe(const std::uint64_t ns) {
    count.fetch_add(1, std::memory_order_relaxed);
    total_ns.fetch_add(ns, std::memory_order_relaxed);
    std::uint64_t prev = max_ns.load(std::memory_order_relaxed);
    while (ns > prev && !max_ns.compare_exchange_weak(prev, ns, std::memory_order_relaxed,
                                                      std::memory_order_relaxed)) {
    }
  }
};

double avg_ms(const AtomicNsStats& stats) {
  const std::uint64_t count = stats.count.load(std::memory_order_relaxed);
  if (count == 0) {
    return 0.0;
  }
  const std::uint64_t total_ns = stats.total_ns.load(std::memory_order_relaxed);
  return ns_to_ms(total_ns) / static_cast<double>(count);
}

double max_ms(const AtomicNsStats& stats) {
  return ns_to_ms(stats.max_ns.load(std::memory_order_relaxed));
}

struct AsyncProfileStats {
  bool enabled = false;
  int push_slow_ms = 2;
  double build_ms = 0.0;
  double warmup_ms = 0.0;
  double push_loop_ms = 0.0;
  double drain_wait_ms = 0.0;
  double verify_ms = 0.0;
  AtomicNsStats push_call;
  AtomicNsStats pull_call;
  AtomicNsStats extract_payload;
  std::atomic<std::uint64_t> slow_pushes{0};
  std::atomic<std::uint64_t> push_failures{0};
  std::atomic<std::uint64_t> pull_timeouts{0};
};

std::string hex_prefix(const std::vector<uint8_t>& payload, size_t max_bytes) {
  std::ostringstream ss;
  ss << std::hex << std::setfill('0');
  const size_t count = std::min(payload.size(), max_bytes);
  for (size_t i = 0; i < count; ++i) {
    ss << std::setw(2) << static_cast<int>(payload[i]);
    if (i + 1 < count)
      ss << " ";
  }
  return ss.str();
}

std::string bbox_payload_summary(const std::vector<uint8_t>& payload, int expected_topk) {
  std::ostringstream ss;
  ss << "bytes=" << payload.size();
  if (payload.size() >= sizeof(uint32_t)) {
    uint32_t header = 0;
    std::memcpy(&header, payload.data(), sizeof(header));
    const size_t data_bytes = payload.size() - sizeof(uint32_t);
    const size_t max_boxes = data_bytes / 24;
    ss << " header=" << header << " data_bytes=" << data_bytes << " data_rem=" << (data_bytes % 24)
       << " max_boxes=" << max_boxes;
    if (expected_topk > 0) {
      const size_t expected_bytes = sizeof(uint32_t) + static_cast<size_t>(expected_topk) * 24;
      ss << " expected_bytes=" << expected_bytes;
    }
  }
  return ss.str();
}

cv::Mat maybe_resize_benchmark_input(const cv::Mat& img, const AsyncTestConfig& cfg) {
  if (cfg.input_width <= 0 || cfg.input_height <= 0 || img.empty()) {
    return img;
  }
  if (img.cols == cfg.input_width && img.rows == cfg.input_height) {
    return img;
  }
  cv::Mat resized;
  cv::resize(img, resized, cv::Size(cfg.input_width, cfg.input_height), 0.0, 0.0, cv::INTER_AREA);
  return resized;
}

simaai::neat::TensorList make_ev74_image_input(const cv::Mat& img) {
  simaai::neat::gst_init_once();
  return simaai::neat::TensorList{simaai::neat::Tensor::from_cv_mat(
      img, simaai::neat::ImageSpec::PixelFormat::BGR, simaai::neat::TensorMemory::EV74)};
}

std::vector<objdet::ExpectedBox>
scale_expected_boxes(const std::vector<objdet::ExpectedBox>& expected, int src_w, int src_h,
                     int dst_w, int dst_h) {
  if (src_w <= 0 || src_h <= 0 || dst_w <= 0 || dst_h <= 0 || (src_w == dst_w && src_h == dst_h)) {
    return expected;
  }
  const float sx = static_cast<float>(dst_w) / static_cast<float>(src_w);
  const float sy = static_cast<float>(dst_h) / static_cast<float>(src_h);
  std::vector<objdet::ExpectedBox> scaled = expected;
  for (auto& b : scaled) {
    b.x1 *= sx;
    b.x2 *= sx;
    b.y1 *= sy;
    b.y2 *= sy;
  }
  return scaled;
}

bool pull_with_timeout(simaai::neat::Run& async, int pull_timeout_ms, int max_timeouts,
                       int& timeout_count, std::optional<simaai::neat::Sample>& out,
                       std::string& err, AsyncProfileStats* profile) {
  while (true) {
    simaai::neat::Sample temp;
    simaai::neat::PullError perr;
    const auto pull_begin = std::chrono::steady_clock::now();
    const simaai::neat::PullStatus status = async.pull(pull_timeout_ms, temp, &perr);
    const auto pull_end = std::chrono::steady_clock::now();
    if (profile && profile->enabled) {
      profile->pull_call.observe(duration_ns(pull_begin, pull_end));
    }
    if (status == simaai::neat::PullStatus::Ok) {
      out = temp;
      timeout_count = 0;
      return true;
    }
    if (status == simaai::neat::PullStatus::Timeout) {
      if (profile && profile->enabled) {
        profile->pull_timeouts.fetch_add(1, std::memory_order_relaxed);
      }
      timeout_count += 1;
      if (max_timeouts > 0 && timeout_count >= max_timeouts) {
        err = "Run::pull: timeout";
        return false;
      }
      continue;
    }
    if (status == simaai::neat::PullStatus::Closed) {
      err = "Run::pull: closed";
      return false;
    }
    err = perr.message.empty() ? "Run::pull: error" : perr.message;
    return false;
  }
}

RunSummary run_yolov8_async_tput(const std::string& tar_gz, const cv::Mat& source_img,
                                 const AsyncTestConfig& cfg) {
  RunSummary res;

  require(!tar_gz.empty(), "Failed to locate yolo_v8s model archive");
  require(!source_img.empty(), "Missing YOLOv8 input image");

  const cv::Mat img = maybe_resize_benchmark_input(source_img, cfg);
  const simaai::neat::TensorList ev74_input = make_ev74_image_input(img);

  simaai::neat::Model::Options model_opt;
  model_opt.preprocess.kind = simaai::neat::InputKind::Image;
  model_opt.preprocess.enable = simaai::neat::AutoFlag::On;
  model_opt.preprocess.normalize.enable = simaai::neat::AutoFlag::On;
  model_opt.preprocess.color_convert.input_format = simaai::neat::PreprocessColorFormat::BGR;
  const int topk = std::max(1, cfg.topk);
  if (!cfg.skip_boxdecode) {
    model_opt.decode_type = cfg.decode_type;
    model_opt.score_threshold = cfg.boxdecode_score_threshold;
    model_opt.nms_iou_threshold = 0.5f;
    model_opt.top_k = topk;
  }
  auto model = simaai::neat::Model(tar_gz, model_opt);

  simaai::neat::Graph p;

  p.add(simaai::neat::nodes::Input());
  p.add(simaai::neat::nodes::groups::Preprocess(model));
  p.add(simaai::neat::nodes::groups::Infer(model));
  if (!cfg.skip_boxdecode) {
    p.add(simaai::neat::nodes::SimaBoxDecode(model, cfg.decode_type, cfg.boxdecode_score_threshold,
                                             0.5f, topk));
  }

  p.add(simaai::neat::nodes::Output());

  const std::vector<objdet::ExpectedBox> expected =
      cfg.skip_boxdecode ? std::vector<objdet::ExpectedBox>{}
                         : scale_expected_boxes(objdet::expected_people_boxes(), source_img.cols,
                                                source_img.rows, img.cols, img.rows);

  step_log("async: before build");
  const auto build_start = std::chrono::steady_clock::now();
  auto async = p.build(ev74_input);
  const auto build_end = std::chrono::steady_clock::now();
  step_log("async: after build");

  const int warm_timeout_ms = 60000;
  const int pull_timeout_ms = std::max(1, cfg.pull_timeout_ms);
  const int max_timeouts = std::max(1, cfg.max_timeouts);
  const bool do_extract = !cfg.skip_boxdecode && !cfg.profile_skip_extract;
  const bool do_validate = do_extract && !cfg.profile_skip_validate;
  AsyncProfileStats profile;
  profile.enabled = cfg.profile;
  profile.push_slow_ms = std::max(1, cfg.profile_push_slow_ms);
  if (profile.enabled) {
    profile.build_ms = ns_to_ms(duration_ns(build_start, build_end));
  }

  try {
    step_log("async: before warmup");
    const auto warmup_start = std::chrono::steady_clock::now();
    for (int i = 0; i < cfg.warm; ++i) {
      (void)async.run(ev74_input, warm_timeout_ms);
    }
    const auto warmup_end = std::chrono::steady_clock::now();
    if (profile.enabled) {
      profile.warmup_ms = ns_to_ms(duration_ns(warmup_start, warmup_end));
    }
    step_log("async: after warmup");
  } catch (const std::exception& e) {
    res.ok = false;
    append_note(res.note, "warmup_error=" + sanitize_note(e.what()));
    res.diagnostics = maybe_collect_run_report(async, cfg.profile_emit_run_report);
    return res;
  }

  for (int i = 0; i < std::max(0, cfg.excluded_preproc_dispatches); ++i) {
    step_log("async: prime preproc dispatch");
    if (!async.push(ev74_input)) {
      res.ok = false;
      append_note(res.note, "primer_push_failed");
      res.diagnostics = maybe_collect_run_report(async, cfg.profile_emit_run_report);
      return res;
    }
    int primer_timeout_count = 0;
    std::optional<simaai::neat::Sample> primer_out;
    std::string primer_err;
    if (!pull_with_timeout(async, warm_timeout_ms, max_timeouts, primer_timeout_count, primer_out,
                           primer_err, nullptr) ||
        !primer_out.has_value()) {
      res.ok = false;
      append_note(res.note,
                  "primer_pull_error=" +
                      sanitize_note(primer_err.empty() ? "missing_primer_output" : primer_err));
      res.diagnostics = maybe_collect_run_report(async, cfg.profile_emit_run_report);
      return res;
    }
    if (do_validate) {
      std::vector<uint8_t> payload;
      std::string err;
      if (!objdet::extract_bbox_payload(*primer_out, i, payload, err)) {
        res.ok = false;
        append_note(res.note, "primer_extract_error=" + sanitize_note(err));
        res.diagnostics = maybe_collect_run_report(async, cfg.profile_emit_run_report);
        return res;
      }
      std::vector<objdet::Box> boxes;
      try {
        boxes = objdet::parse_boxes_strict(payload, img.cols, img.rows, topk, false);
      } catch (const std::exception& ex) {
        res.ok = false;
        append_note(res.note, "primer_bbox_parse_failed=" + sanitize_note(ex.what()));
        res.diagnostics = maybe_collect_run_report(async, cfg.profile_emit_run_report);
        return res;
      }
      const objdet::MatchResult match =
          objdet::match_expected_boxes(boxes, expected, cfg.min_score, cfg.min_iou);
      if (!match.ok) {
        res.ok = false;
        append_note(res.note,
                    "primer_verify_mismatch iter=" + std::to_string(i) + " " + match.note);
        res.diagnostics = maybe_collect_run_report(async, cfg.profile_emit_run_report);
        return res;
      }
    }
  }

  std::string consumer_error;
  std::vector<std::vector<uint8_t>> verify_payloads;
  if (do_extract) {
    verify_payloads.reserve(static_cast<size_t>(cfg.iters));
  }

  const int inflight = std::max(1, cfg.inflight);
  int pushed_count = 0;
  int measured_count = 0;
  int timeout_count = 0;

  simaai::neat::MeasureOptions measure_opt;
  measure_opt.include_plugin_latency = false;
  measure_opt.include_edge_latency = false;
  measure_opt.include_power = false;
  auto measure_scope = async.start_measurement(measure_opt);

  const int seed = std::min(inflight, cfg.iters);
  for (; pushed_count < seed; ++pushed_count) {
    const auto push_start = std::chrono::steady_clock::now();
    const bool pushed = async.push(ev74_input);
    const auto push_end = std::chrono::steady_clock::now();
    if (profile.enabled) {
      const std::uint64_t ns = duration_ns(push_start, push_end);
      profile.push_call.observe(ns);
      if (ns_to_ms(ns) >= static_cast<double>(profile.push_slow_ms)) {
        profile.slow_pushes.fetch_add(1, std::memory_order_relaxed);
      }
    }
    if (!pushed) {
      if (profile.enabled) {
        profile.push_failures.fetch_add(1, std::memory_order_relaxed);
      }
      consumer_error = "Run::push: returned false during seed";
      break;
    }
  }

  const auto start = std::chrono::steady_clock::now();
  const auto push_loop_start = start;
  auto push_loop_end = start;
  auto end = start;

  try {
    while (consumer_error.empty() && measured_count < cfg.iters) {
      std::optional<simaai::neat::Sample> out;
      std::string err;
      if (!pull_with_timeout(async, pull_timeout_ms, max_timeouts, timeout_count, out, err,
                             &profile)) {
        consumer_error = err.empty() ? "Run::pull: error" : err;
        break;
      }
      if (!out.has_value()) {
        consumer_error = "Run::pull: missing output";
        break;
      }

      ++measured_count;
      if (do_extract) {
        std::vector<uint8_t> payload;
        const auto extract_start = std::chrono::steady_clock::now();
        const bool extracted = objdet::extract_bbox_payload(*out, measured_count - 1, payload, err);
        const auto extract_end = std::chrono::steady_clock::now();
        if (profile.enabled) {
          profile.extract_payload.observe(duration_ns(extract_start, extract_end));
        }
        if (!extracted) {
          consumer_error = err;
          break;
        }
        verify_payloads.push_back(std::move(payload));
      }

      // Release the pulled sample before admitting the next frame. Otherwise
      // the benchmark itself can keep zero-copy downstream buffers alive while
      // asking upstream stages for the next output buffer.
      out.reset();

      if (pushed_count < cfg.iters) {
        const auto push_start = std::chrono::steady_clock::now();
        const bool pushed = async.push(ev74_input);
        const auto push_end = std::chrono::steady_clock::now();
        if (profile.enabled) {
          const std::uint64_t ns = duration_ns(push_start, push_end);
          profile.push_call.observe(ns);
          if (ns_to_ms(ns) >= static_cast<double>(profile.push_slow_ms)) {
            profile.slow_pushes.fetch_add(1, std::memory_order_relaxed);
          }
        }
        if (!pushed) {
          if (profile.enabled) {
            profile.push_failures.fetch_add(1, std::memory_order_relaxed);
          }
          consumer_error = "Run::push: returned false";
          break;
        }
        ++pushed_count;
      }
    }
  } catch (const std::exception& e) {
    consumer_error = e.what();
  }

  push_loop_end = std::chrono::steady_clock::now();
  if (profile.enabled) {
    profile.push_loop_ms = ns_to_ms(duration_ns(push_loop_start, push_loop_end));
  }

  async.close_input();
  end = std::chrono::steady_clock::now();
  if (profile.enabled) {
    profile.drain_wait_ms = ns_to_ms(duration_ns(push_loop_end, end));
  }
  const simaai::neat::MeasureReport measure_report = measure_scope.stop();

  const double elapsed_s = std::chrono::duration<double>(end - start).count();

  res.outputs = measured_count;
  res.avg_fps = (elapsed_s > 0.0) ? (static_cast<double>(res.outputs) / elapsed_s) : 0.0;
  res.ok = (res.outputs == cfg.iters);

  if (!consumer_error.empty()) {
    append_note(res.note, "async_pull_error=" + sanitize_note(consumer_error));
    res.ok = false;
  }
  if (elapsed_s <= 0.0) {
    append_note(res.note, "async_timing_incomplete");
    res.ok = false;
  }

  if (do_extract) {
    const size_t payloads = verify_payloads.size();
    if (payloads != static_cast<size_t>(cfg.iters)) {
      append_note(res.note, "verify_payloads=" + std::to_string(payloads));
      res.ok = false;
    }
    if (do_validate) {
      const auto verify_start = std::chrono::steady_clock::now();
      const size_t verify_count = std::min(payloads, static_cast<size_t>(cfg.iters));
      for (size_t i = 0; i < verify_count; ++i) {
        std::vector<objdet::Box> boxes;
        try {
          boxes = objdet::parse_boxes_strict(verify_payloads[i], img.cols, img.rows, topk, false);
        } catch (const std::exception& ex) {
          std::cerr << "[WARN] bbox parse failed iter=" << i << " err=" << ex.what() << " "
                    << bbox_payload_summary(verify_payloads[i], topk)
                    << " prefix=" << hex_prefix(verify_payloads[i], 64) << "\n";
          append_note(res.note, "bbox_parse_failed iter=" + std::to_string(i));
          res.ok = false;
          break;
        }
        const objdet::MatchResult match =
            objdet::match_expected_boxes(boxes, expected, cfg.min_score, cfg.min_iou);
        if (!match.ok) {
          append_note(res.note, "verify_mismatch iter=" + std::to_string(i) + " " + match.note);
          res.ok = false;
          break;
        }
      }
      const auto verify_end = std::chrono::steady_clock::now();
      if (profile.enabled) {
        profile.verify_ms = ns_to_ms(duration_ns(verify_start, verify_end));
      }
    } else {
      append_note(res.note, "profile_skip_validate=1");
    }
  } else if (cfg.skip_boxdecode) {
    append_note(res.note, "skip_boxdecode=1");
  } else {
    append_note(res.note, "profile_skip_extract=1");
  }

  if (!res.ok) {
    const std::string last_err = async.last_error();
    if (!last_err.empty()) {
      append_note(res.note, "async_last_error=" + sanitize_note(last_err));
    }
  }
  if (profile.enabled) {
    const std::uint64_t push_calls = profile.push_call.count.load(std::memory_order_relaxed);
    const std::uint64_t pull_calls = profile.pull_call.count.load(std::memory_order_relaxed);
    const std::uint64_t extract_calls =
        profile.extract_payload.count.load(std::memory_order_relaxed);
    std::cout << "ASYNC_TPUT_PROFILE window inflight=" << inflight << " pushed=" << pushed_count
              << " measured=" << measured_count << "\n";
    std::cout << "ASYNC_TPUT_PROFILE phases build_ms=" << profile.build_ms
              << " warmup_ms=" << profile.warmup_ms << " push_loop_ms=" << profile.push_loop_ms
              << " drain_wait_ms=" << profile.drain_wait_ms
              << " measured_ms=" << (elapsed_s * 1000.0) << " verify_ms=" << profile.verify_ms
              << "\n";
    std::cout << "ASYNC_TPUT_PROFILE push calls=" << push_calls
              << " avg_ms=" << avg_ms(profile.push_call) << " max_ms=" << max_ms(profile.push_call)
              << " slow(>=" << profile.push_slow_ms
              << "ms)=" << profile.slow_pushes.load(std::memory_order_relaxed)
              << " failures=" << profile.push_failures.load(std::memory_order_relaxed) << "\n";
    std::cout << "ASYNC_TPUT_PROFILE pull calls=" << pull_calls
              << " avg_ms=" << avg_ms(profile.pull_call) << " max_ms=" << max_ms(profile.pull_call)
              << " timeouts=" << profile.pull_timeouts.load(std::memory_order_relaxed) << "\n";
    std::cout << "ASYNC_TPUT_PROFILE extract calls=" << extract_calls
              << " avg_ms=" << avg_ms(profile.extract_payload)
              << " max_ms=" << max_ms(profile.extract_payload) << "\n";
    std::cout << "ASYNC_TPUT_PROFILE measured avg_latency_ms=" << measure_report.end_to_end.avg_ms
              << " p50_latency_ms=" << measure_report.end_to_end.p50_ms
              << " max_latency_ms=" << measure_report.end_to_end.max_ms
              << " outputs_pulled=" << measure_report.counters.outputs_pulled << "\n";
    if (cfg.profile_emit_run_report) {
      res.diagnostics = maybe_collect_run_report(async, true, &measure_report);
    }
  }

  return res;
}

} // namespace

int main(int argc, char** argv) {
  try {
    // This is a throughput threshold test; per-frame stage debug logging is
    // intentionally disabled because the DevKit test wrapper enables it by
    // default for diagnostics and that logging materially changes the measured
    // steady-state FPS.
    setenv("SIMA_STAGE_DEBUG", "0", 1);

    const fs::path root = (argc > 1) ? fs::path(argv[1]) : fs::current_path();
    std::error_code ec;
    fs::create_directories(root / "tmp", ec);
    fs::current_path(root, ec);

    const std::string tar_gz = sima_yolov8_test::resolve_yolov8s_tar_or_skip(root);
    cv::Mat img_bgr = sima_yolov8_test::load_people_image_or_skip(root);

    AsyncTestConfig cfg;
    cfg.iters = std::max(1, env_int("SIMA_ASYNC_YOLOV8_ITERS", cfg.iters));
    cfg.warm = std::max(0, env_int("SIMA_ASYNC_YOLOV8_WARM", cfg.warm));
    cfg.inflight = std::max(1, env_int("SIMA_ASYNC_YOLOV8_INFLIGHT", cfg.inflight));
    cfg.excluded_preproc_dispatches =
        std::max(0, env_int("SIMA_ASYNC_YOLOV8_EXCLUDE_PREPROC_DISPATCHES",
                            cfg.excluded_preproc_dispatches));
    cfg.topk = std::max(1, env_int("SIMA_ASYNC_YOLOV8_TOPK", cfg.topk));
    cfg.min_fps = std::max(0.0, env_double("SIMA_ASYNC_YOLOV8_MIN_FPS", cfg.min_fps));
    cfg.skip_boxdecode = env_bool("SIMA_ASYNC_YOLOV8_SKIP_BOXDECODE", false);
    cfg.profile = env_bool("SIMA_ASYNC_YOLOV8_PROFILE", false);
    cfg.profile_emit_run_report = env_bool("SIMA_ASYNC_YOLOV8_PROFILE_REPORT", false);
    cfg.profile_skip_extract =
        env_bool("SIMA_ASYNC_YOLOV8_PROFILE_SKIP_EXTRACT", cfg.profile_skip_extract);
    cfg.profile_skip_validate =
        env_bool("SIMA_ASYNC_YOLOV8_PROFILE_SKIP_VALIDATE", cfg.profile_skip_validate);
    cfg.profile_push_slow_ms =
        std::max(1, env_int("SIMA_ASYNC_YOLOV8_PROFILE_PUSH_SLOW_MS", cfg.profile_push_slow_ms));
    cfg.pull_timeout_ms =
        std::max(1, env_int("SIMA_ASYNC_YOLOV8_PULL_TIMEOUT_MS", cfg.pull_timeout_ms));
    cfg.max_timeouts = std::max(1, env_int("SIMA_ASYNC_YOLOV8_MAX_TIMEOUTS", cfg.max_timeouts));
    cfg.input_width = std::max(0, env_int("SIMA_ASYNC_YOLOV8_INPUT_W", cfg.input_width));
    cfg.input_height = std::max(0, env_int("SIMA_ASYNC_YOLOV8_INPUT_H", cfg.input_height));
    if (const auto parsed =
            parse_boxdecode_type_token(std::getenv("SIMA_ASYNC_YOLOV8_DECODE_TYPE"))) {
      cfg.decode_type = *parsed;
    }
    if (cfg.profile_skip_extract) {
      cfg.profile_skip_validate = true;
    }
    if (cfg.skip_boxdecode) {
      cfg.profile_skip_extract = true;
      cfg.profile_skip_validate = true;
    }

    RunSummary res = run_yolov8_async_tput(tar_gz, img_bgr, cfg);
    if (res.avg_fps <= cfg.min_fps) {
      append_note(res.note, "avg_fps=" + std::to_string(res.avg_fps));
      append_note(res.note, "min_fps=" + std::to_string(cfg.min_fps));
      res.ok = false;
    }

    std::cout << "ASYNC_TPUT outputs=" << res.outputs << " avg_fps=" << res.avg_fps
              << " decode_type=" << simaai::neat::box_decode_type_token(cfg.decode_type)
              << " ok=" << (res.ok ? "1" : "0") << " note=" << res.note << "\n";
    if (!res.diagnostics.empty()) {
      std::cout << "ASYNC_TPUT diagnostics\n" << res.diagnostics << "\n";
    }

    return res.ok ? 0 : 2;
  } catch (const SkipTest& e) {
    std::cout << "[SKIP] " << e.what() << "\n";
    return skip_long_test(e.what());
  } catch (const std::exception& e) {
    if (is_dispatcher_unavailable(e.what())) {
      return skip_long_test("dispatcher unavailable");
    }
    std::cerr << "[ERR] " << e.what() << "\n";
    return 1;
  }
}
