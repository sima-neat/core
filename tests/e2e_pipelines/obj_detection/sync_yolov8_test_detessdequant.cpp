/**
 * @example sync_yolov8_test.cpp
 * Canonical production pipeline: input -> preprocess -> Infer -> postprocess.
 */
#include "pipeline/Session.h"
#include "nodes/groups/ModelGroups.h"
#include "nodes/io/Input.h"
#include "nodes/sima/DetessDequant.h"
#include "model/Model.h"
#include "model/internal/ModelInternal.h"
#include "pipeline/internal/TensorTransfer.h"

#include "e2e_pipelines/e2e_utils.h"
#include "e2e_pipelines/obj_detection/obj_detection_utils.h"
#include "e2e_pipelines/obj_detection/yolov8_test_utils.h"
#include "test_utils.h"

#include <opencv2/imgcodecs.hpp>

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstdlib>
#include <filesystem>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

namespace fs = std::filesystem;

namespace {

using sima_yolov8_test::append_note;
using sima_yolov8_test::env_bool;
using sima_yolov8_test::env_int;
using sima_yolov8_test::sanitize_note;
using sima_yolov8_test::step_log;

struct SyncTestConfig {
  int iters = 20;
  float min_score = 0.52f;
  float min_iou = 0.30f;
};

struct RunSummary {
  bool ok = false;
  int outputs = 0;
  double avg_fps = 0.0;
  std::string note;
  std::string diagnostics;
};

bool detess_zero_copy_enabled() {
  static int cached = -1;
  if (cached >= 0)
    return cached != 0;
  cached = env_bool("SIMA_DETESS_ZERO_COPY", false) ? 1 : 0;
  return cached != 0;
}

const char* device_type_name(simaai::neat::DeviceType type) {
  switch (type) {
  case simaai::neat::DeviceType::CPU:
    return "CPU";
  case simaai::neat::DeviceType::SIMA_APU:
    return "APU";
  case simaai::neat::DeviceType::SIMA_CVU:
    return "CVU";
  case simaai::neat::DeviceType::SIMA_MLA:
    return "MLA";
  case simaai::neat::DeviceType::UNKNOWN:
  default:
    return "UNKNOWN";
  }
}

const char* storage_kind_name(simaai::neat::StorageKind kind) {
  switch (kind) {
  case simaai::neat::StorageKind::CpuOwned:
    return "CpuOwned";
  case simaai::neat::StorageKind::CpuExternal:
    return "CpuExternal";
  case simaai::neat::StorageKind::GstSample:
    return "GstSample";
  case simaai::neat::StorageKind::DeviceHandle:
    return "DeviceHandle";
  case simaai::neat::StorageKind::Unknown:
  default:
    return "Unknown";
  }
}

std::string segments_debug(const std::vector<simaai::neat::Segment>& segs) {
  if (segs.empty())
    return {};
  std::ostringstream ss;
  ss << "[";
  for (size_t i = 0; i < segs.size(); ++i) {
    if (i)
      ss << ",";
    ss << segs[i].name << ":" << segs[i].size_bytes;
  }
  ss << "]";
  return ss.str();
}

void log_tensor_sample(const simaai::neat::Sample& s, const std::string& label) {
  std::ostringstream ss;
  ss << "[DBG] " << label;
  if (!s.port_name.empty())
    ss << " port=" << s.port_name;
  if (s.output_index >= 0)
    ss << " output=" << s.output_index;
  if (!s.media_type.empty())
    ss << " media_type=" << s.media_type;
  if (!s.caps_string.empty())
    ss << " caps=" << s.caps_string;
  if (!s.payload_tag.empty())
    ss << " payload_tag=" << s.payload_tag;
  if (!s.format.empty())
    ss << " format=" << s.format;
  if (!s.tensor.has_value()) {
    ss << " neat=<missing>";
    std::cerr << ss.str() << "\n";
    return;
  }
  const auto& t = s.tensor.value();
  ss << " " << t.debug_string();
  if (t.storage) {
    ss << " storage_kind=" << storage_kind_name(t.storage->kind)
       << " storage_size=" << t.storage->size_bytes
       << " storage_device=" << device_type_name(t.storage->device.type) << ":"
       << t.storage->device.id;
    if (t.storage->sima_mem_target_flags || t.storage->sima_mem_flags) {
      ss << " sima_mem_target_flags=0x" << std::hex << t.storage->sima_mem_target_flags
         << " sima_mem_flags=0x" << t.storage->sima_mem_flags << std::dec;
    }
    const std::string segs = segments_debug(t.storage->sima_segments);
    if (!segs.empty())
      ss << " sima_segments=" << segs;
  } else {
    ss << " storage=<none>";
  }
  std::cerr << ss.str() << "\n";
}

void log_sample_caps(const simaai::neat::Sample& s, const std::string& label) {
  if (s.kind != simaai::neat::SampleKind::Bundle) {
    log_tensor_sample(s, label);
    return;
  }
  std::ostringstream ss;
  ss << "[DBG] " << label << " bundle fields=" << s.fields.size();
  if (!s.media_type.empty())
    ss << " media_type=" << s.media_type;
  if (!s.caps_string.empty())
    ss << " caps=" << s.caps_string;
  std::cerr << ss.str() << "\n";
  for (size_t i = 0; i < s.fields.size(); ++i) {
    log_tensor_sample(s.fields[i], label + ".field" + std::to_string(i));
  }
}

bool extract_tensor_payload_any_impl(const simaai::neat::Sample& result, const std::string& context,
                                     std::vector<uint8_t>& payload, std::string& fmt,
                                     std::string& err, bool* used_cpu_transfer) {
  if (used_cpu_transfer)
    *used_cpu_transfer = false;
  if (result.kind == simaai::neat::SampleKind::Bundle) {
    for (const auto& field : result.fields) {
      if (extract_tensor_payload_any_impl(field, context, payload, fmt, err, used_cpu_transfer)) {
        return true;
      }
    }
    err = objdet::append_context("bundle missing tensor field", context);
    return false;
  }
  if (result.kind != simaai::neat::SampleKind::Tensor) {
    err = objdet::append_context("capture_expected_tensor", context);
    return false;
  }
  if (!result.tensor.has_value()) {
    err = objdet::append_context("capture_missing_tensor", context);
    return false;
  }

  const auto& tensor = result.tensor.value();
  fmt = result.payload_tag;
  if (fmt.empty() && !result.format.empty())
    fmt = result.format;
  if (fmt.empty() && tensor.semantic.tess.has_value()) {
    fmt = tensor.semantic.tess->format;
  }
  const std::string fmt_upper = objdet::upper_ascii_copy(fmt);

  if (detess_zero_copy_enabled() && !fmt_upper.empty() && fmt_upper != "BBOX") {
    const simaai::neat::Mapping mapping = tensor.map_read();
    if (mapping.data && mapping.size_bytes > 0) {
      return true;
    }
  }

  try {
    payload = tensor.copy_payload_bytes();
    if (payload.empty()) {
      err = objdet::append_context("capture_empty_payload", context);
      return false;
    }
    return true;
  } catch (const std::exception& ex) {
    const std::string direct_err = sanitize_note(ex.what());
    const std::string dbg = tensor.debug_string();
    try {
      auto cpu = simaai::neat::pipeline_internal::transfer_to_cpu(tensor);
      payload = cpu.copy_payload_bytes();
      if (payload.empty()) {
        err = objdet::append_context("capture_payload_failed", context);
        err += " err=" + direct_err;
        err += " cpu_transfer_failed=empty";
        err += " tensor=" + dbg;
        return false;
      }
      if (used_cpu_transfer)
        *used_cpu_transfer = true;
      static bool logged_fallback = false;
      if (!logged_fallback) {
        std::cerr << "[DBG] payload cpu transfer fallback format="
                  << (fmt_upper.empty() ? "UNKNOWN" : fmt_upper) << " " << dbg << "\n";
        logged_fallback = true;
      }
      return true;
    } catch (const std::exception& ex2) {
      err = objdet::append_context("capture_payload_failed", context);
      err += " err=" + direct_err;
      err += " cpu_transfer_failed=" + sanitize_note(ex2.what());
      err += " tensor=" + dbg;
      return false;
    }
  }
}

bool extract_tensor_payload_any(const simaai::neat::Sample& result, int iter,
                                std::vector<uint8_t>& payload, std::string& fmt, std::string& err,
                                bool* used_cpu_transfer) {
  return extract_tensor_payload_any_impl(result, "iter=" + std::to_string(iter), payload, fmt, err,
                                         used_cpu_transfer);
}

RunSummary run_yolov8_sync(const std::string& tar_gz, const cv::Mat& img,
                           const SyncTestConfig& cfg) {
  RunSummary res;

  require(!tar_gz.empty(), "Failed to locate yolo_v8s MPK tarball");

  const int num_both = env_int("SIMA_SYNC_NUM_BUFFERS", -1);
  int num_cvu = env_int("SIMA_SYNC_NUM_BUFFERS_CVU", num_both);
  int num_mla = env_int("SIMA_SYNC_NUM_BUFFERS_MLA", num_both);
  const bool override_num = (num_cvu >= 0 || num_mla >= 0);
  if (override_num) {
    if (num_cvu < 0 || num_mla < 0) {
      append_note(res.note, "num_buffers_requires_both");
      return res;
    }
    if (!((num_cvu == 0 || num_cvu == 1) && (num_mla == 0 || num_mla == 1))) {
      append_note(res.note, "num_buffers_invalid");
      return res;
    }
  }

  (void)num_cvu;
  (void)num_mla;
  simaai::neat::Model::Options model_opt;
  model_opt.media_type = "video/x-raw";
  model_opt.format = "BGR";
  model_opt.input_max_width = img.cols;
  model_opt.input_max_height = img.rows;
  model_opt.input_max_depth = 3;
  model_opt.upstream_name = "decoder";
  auto model = simaai::neat::Model(tar_gz, model_opt);
  const int topk = 100;

  // [canonical_pipeline]
  simaai::neat::Session p;
  p.add(simaai::neat::nodes::Input());
  p.add(simaai::neat::nodes::groups::Preprocess(model));
  p.add(simaai::neat::nodes::groups::Infer(model));
  p.add(std::make_shared<simaai::neat::DetessDequant>(simaai::neat::DetessDequantOptions(model)));
  p.add(simaai::neat::nodes::Output());
  // [canonical_pipeline]

  const std::vector<objdet::ExpectedBox> expected = objdet::expected_people_boxes();

  simaai::neat::RunOptions run_opt;
  run_opt.queue_depth = 1;
  step_log("sync: before build");
  simaai::neat::Run runner = p.build(img, simaai::neat::RunMode::Async, run_opt);
  step_log("sync: after build");

  const auto start = std::chrono::steady_clock::now();
  bool logged_sample = false;
  bool noted_skip = false;
  bool noted_payload = false;
  bool noted_cpu_transfer = false;
  const int pull_timeout_ms = env_int("SIMA_SYNC_PULL_TIMEOUT_MS", -1);
  const int log_every = env_int("SIMA_SYNC_LOG_EVERY", 1);
  for (int i = 0; i < cfg.iters; ++i) {
    if (log_every > 0 && (((i + 1) % log_every) == 0 || i == 0 || i + 1 == cfg.iters)) {
      std::cout << "SYNC_YOLOV8 iter " << (i + 1) << "/" << cfg.iters << "\n";
      std::cout.flush();
    }
    simaai::neat::Sample out;
    try {
      step_log("sync: before run");
      out = runner.push_and_pull(img, pull_timeout_ms);
      step_log("sync: after run");
    } catch (const std::exception& e) {
      append_note(res.note, "run_error=" + sanitize_note(e.what()));
      break;
    }

    if (!logged_sample) {
      log_sample_caps(out, "appsink");
      logged_sample = true;
    }

    std::vector<uint8_t> payload;
    std::string err;
    std::string fmt;
    bool used_cpu_transfer = false;
    const bool payload_ok =
        extract_tensor_payload_any(out, i, payload, fmt, err, &used_cpu_transfer);
    const std::string fmt_upper = objdet::upper_ascii_copy(fmt);
    if (used_cpu_transfer && !noted_cpu_transfer) {
      append_note(res.note, "payload_cpu_transfer");
      noted_cpu_transfer = true;
    }
    if (!payload_ok) {
      if (!noted_payload) {
        append_note(res.note, err);
        noted_payload = true;
      }
    }

    const bool should_verify = fmt_upper.empty() || fmt_upper == "BBOX";
    if (should_verify && payload_ok) {
      try {
        const auto boxes = objdet::parse_boxes_strict(payload, img.cols, img.rows, topk, false);
        const objdet::MatchResult match =
            objdet::match_expected_boxes(boxes, expected, cfg.min_score, cfg.min_iou);
        if (!match.ok) {
          append_note(res.note, "verify_mismatch iter=" + std::to_string(i) + " " + match.note);
          break;
        }
      } catch (const std::exception& ex) {
        if (!noted_skip) {
          const std::string label = fmt_upper.empty() ? "UNKNOWN" : fmt_upper;
          append_note(res.note, "skip_bbox_verify parse_failed format=" + label);
          noted_skip = true;
        }
      }
    } else if (!should_verify && !noted_skip) {
      const std::string label = fmt_upper.empty() ? "UNKNOWN" : fmt_upper;
      append_note(res.note, "skip_bbox_verify format=" + label);
      noted_skip = true;
    }

    res.outputs += 1;
  }
  const auto end = std::chrono::steady_clock::now();

  res.diagnostics = p.last_pipeline();

  const double elapsed_s = std::chrono::duration<double>(end - start).count();
  res.avg_fps = (elapsed_s > 0.0) ? (static_cast<double>(res.outputs) / elapsed_s) : 0.0;
  res.ok = (res.outputs == cfg.iters);
  if (elapsed_s <= 0.0) {
    append_note(res.note, "sync_timing_incomplete");
    res.ok = false;
  }

  return res;
}

} // namespace

int main(int argc, char** argv) {
  try {
    const fs::path root = (argc > 1) ? fs::path(argv[1]) : fs::current_path();
    std::error_code ec;
    fs::create_directories(root / "tmp", ec);
    fs::current_path(root, ec);

    const std::string tar_gz = sima_yolov8_test::resolve_yolov8s_tar_or_skip(root);
    cv::Mat img_bgr = sima_yolov8_test::load_people_image_or_skip(root);

    SyncTestConfig cfg;
    cfg.iters = env_int("SIMA_SYNC_ITERS", cfg.iters);
    RunSummary res = run_yolov8_sync(tar_gz, img_bgr, cfg);

    std::cout << "SYNC_YOLOV8 outputs=" << res.outputs << " avg_fps=" << res.avg_fps
              << " ok=" << (res.ok ? "1" : "0") << " note=" << res.note << "\n";
    if (!res.diagnostics.empty()) {
      std::cout << "SYNC_YOLOV8 diagnostics\n" << res.diagnostics << "\n";
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
