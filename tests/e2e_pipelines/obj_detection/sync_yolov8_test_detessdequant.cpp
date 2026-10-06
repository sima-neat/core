#ifndef SIMA_NEAT_INTERNAL
#define SIMA_NEAT_INTERNAL 1
#endif
/**
 * @example sync_yolov8_test.cpp
 * Canonical production pipeline: input -> preprocess -> Infer -> postprocess.
 */
#include "pipeline/Graph.h"
#include "nodes/groups/ModelGroups.h"
#include "nodes/io/Input.h"
#include "nodes/sima/DetessDequant.h"
#include "model/Model.h"
#include "model/internal/ModelInternal.h"
#include "model/internal/ModelPack.h"
#include "pipeline/internal/TensorTransfer.h"
#include "pipeline/internal/TensorBufferEnvelope.h"
#include "dmabuf_test_utils.h"

#include "e2e_pipelines/e2e_utils.h"
#include "e2e_pipelines/obj_detection/obj_detection_utils.h"
#include "e2e_pipelines/obj_detection/yolov8_test_utils.h"
#include "test_utils.h"

#include <opencv2/imgcodecs.hpp>

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <iomanip>
#include <iostream>
#include <limits>
#include <optional>
#include <sstream>
#include <stdexcept>
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

bool detess_deep_debug_enabled() {
  static int cached = -1;
  if (cached >= 0) {
    return cached != 0;
  }
  const char* raw = std::getenv("SIMA_DETESS_DEBUG_ALL");
  if (!raw || !*raw) {
    cached = 1;
    return true;
  }
  cached = (std::strcmp(raw, "0") == 0) ? 0 : 1;
  return cached != 0;
}

void setenv_if_missing(const char* key, const char* value) {
  if (!key || !*key || !value) {
    return;
  }
  const char* cur = std::getenv(key);
  if (cur && *cur) {
    return;
  }
  setenv(key, value, 0);
}

void enable_detess_debug_defaults() {
  if (!detess_deep_debug_enabled()) {
    return;
  }
  setenv_if_missing("SIMA_STAGE_DEBUG", "1");
  setenv_if_missing("SIMA_MANIFEST_ROUTE_DEBUG", "1");
  setenv_if_missing("SIMA_MLA_CONTRACT_DEBUG", "1");
  setenv_if_missing("SIMA_MPK_CONTRACT_DEBUG", "1");
  setenv_if_missing("SIMA_ROUTE_DEBUG", "1");
  setenv_if_missing("SIMA_PULL_TIMEOUT_DIAG", "1");
  setenv_if_missing("SIMA_PULL_TIMEOUT_POOL_DIAG", "1");
  setenv_if_missing("SIMA_DISPATCHER_TRACE", "1");
  setenv_if_missing("SIMA_RPCEVXX_TRACE", "1");
  setenv_if_missing("SIMA_RPCEVXX_CFG_TRACE", "1");
  setenv_if_missing("SIMA_RPMSG_OPEN_TRACE", "1");
  setenv_if_missing("SIMA_RPMSG_RECV_TRACE", "1");
  setenv_if_missing("SIMA_RPMSG_RECV_SLICE_MS", "250");
  setenv_if_missing("SIMA_RPMSG_RECV_PROGRESS_MS", "1000");
  setenv_if_missing("SIMA_RPMSG_SEQ_USE_GRAPH_ID", "1");
  setenv_if_missing("SIMA_PROCESSCVU_POOL_DEBUG", "1");
  setenv_if_missing("SIMA_DETESS_CONFIG_DEBUG", "1");
}

std::string debug_env_snapshot() {
  static constexpr std::array<const char*, 18> kKeys = {
      "SIMA_DETESS_DEBUG_ALL",       "SIMA_STAGE_DEBUG",
      "SIMA_MANIFEST_ROUTE_DEBUG",   "SIMA_MLA_CONTRACT_DEBUG",
      "SIMA_MPK_CONTRACT_DEBUG",     "SIMA_ROUTE_DEBUG",
      "SIMA_PULL_TIMEOUT_DIAG",      "SIMA_PULL_TIMEOUT_POOL_DIAG",
      "SIMA_DISPATCHER_TRACE",       "SIMA_RPCEVXX_TRACE",
      "SIMA_RPCEVXX_CFG_TRACE",      "SIMA_RPMSG_OPEN_TRACE",
      "SIMA_RPMSG_RECV_TRACE",       "SIMA_RPMSG_RECV_SLICE_MS",
      "SIMA_RPMSG_RECV_PROGRESS_MS", "SIMA_RPMSG_SEQ_USE_GRAPH_ID",
      "SIMA_PROCESSCVU_POOL_DEBUG",  "SIMA_DETESS_CONFIG_DEBUG",
  };
  std::ostringstream ss;
  ss << "[DBG] detess debug env";
  for (const char* key : kKeys) {
    const char* val = std::getenv(key);
    ss << " " << key << "=" << ((val && *val) ? val : "<unset>");
  }
  return ss.str();
}

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
  if (s.kind == simaai::neat::SampleKind::TensorSet) {
    for (std::size_t i = 0; i < s.tensors.size(); ++i) {
      simaai::neat::Sample head;
      head.tensor = s.tensors[i];
      log_tensor_sample(head, label + ".head" + std::to_string(i));
    }
    return;
  }
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

struct ExpectedHead {
  std::string name;
  std::vector<std::int64_t> shape;
  std::size_t size_bytes = 0U;
};

std::vector<ExpectedHead> expected_raw_heads(const simaai::neat::Model& model) {
  // Read the loader's MPK-only contract, independently of the runtime output
  // descriptors under test. No model sidecar or decoded-box interpretation.
  const auto& mpk = simaai::neat::internal::ModelAccess::pack(model).mpk_contract();
  require(mpk.has_value(), "sync: expected an MPK manifest contract");
  std::vector<ExpectedHead> heads;
  for (const auto& plugin : mpk->plugins) {
    if (plugin.kernel != "dequantization_transform") {
      continue;
    }
    for (const auto& output : plugin.output_tensors) {
      const auto dtype = objdet::upper_ascii_copy(
          output.logical_dtype.empty() ? output.dtype : output.logical_dtype);
      require(dtype == "FLOAT32" || dtype == "FP32", "sync: MPK head must be FP32");
      // These are dense dequantization outputs: mpk_shape preserves the
      // authored output_shapes rank, including N=1. logical_shape is the
      // loader's legacy geometry projection and intentionally strips that axis.
      const auto& shape = output.mpk_shape;
      require(!output.name.empty() && !shape.empty(), "sync: MPK head identity/shape missing");
      std::size_t bytes = sizeof(float);
      for (const auto dim : shape) {
        require(dim > 0 && static_cast<std::uint64_t>(dim) <=
                               std::numeric_limits<std::size_t>::max() / bytes,
                "sync: invalid MPK head shape");
        bytes *= static_cast<std::size_t>(dim);
      }
      require(bytes == output.size_bytes, "sync: MPK head shape/byte extent mismatch");
      require(std::none_of(heads.begin(), heads.end(),
                           [&](const auto& head) { return head.name == output.name; }),
              "sync: duplicate MPK head identity");
      heads.push_back({output.name, shape, bytes});
    }
  }
  require(heads.size() == 6U, "sync: expected the six MPK-authored YOLOv8 raw heads");
  return heads;
}

sima_test::DmaBufBackingIdentity validate_raw_heads(const simaai::neat::Sample& sample,
                                                    const std::vector<ExpectedHead>& expected) {
  require(sample.kind == simaai::neat::SampleKind::TensorSet &&
              sample.tensors.size() == expected.size(),
          "sync: expected all six raw output tensors");
  simaai::neat::pipeline_internal::TensorBufferView carrier;
  std::string error;
  require(simaai::neat::pipeline_internal::tensor_buffer_view_from_sample(sample, &carrier, &error),
          "sync: raw heads must retain their shared native carrier: " + error);
  const auto backing = sima_test::dmabuf_span(carrier.buffer).backing;
  for (const auto& head : expected) {
    const auto matches = [&](const auto& tensor) { return tensor.route.name == head.name; };
    require(std::count_if(sample.tensors.begin(), sample.tensors.end(), matches) == 1,
            "sync: missing/duplicate MPK output identity " + head.name);
    const auto& tensor = *std::find_if(sample.tensors.begin(), sample.tensors.end(), matches);
    if (tensor.shape != head.shape || tensor.dtype != simaai::neat::TensorDType::Float32 ||
        !tensor.is_dense() || !tensor.is_contiguous()) {
      std::ostringstream detail;
      detail << "sync: raw head shape/dtype/layout mismatch " << head.name
             << " expected=FP32 dense contiguous shape=[";
      for (std::size_t axis = 0; axis < head.shape.size(); ++axis) {
        detail << (axis ? "," : "") << head.shape[axis];
      }
      detail << "] bytes=" << head.size_bytes << " actual=" << tensor.debug_string()
             << " dense=" << tensor.is_dense() << " contiguous=" << tensor.is_contiguous();
      throw std::runtime_error(detail.str());
    }
    const auto mapping = tensor.map_read();
    require(mapping.data && mapping.size_bytes >= head.size_bytes,
            "sync: unreadable raw head " + head.name);
    const auto* data = static_cast<const std::uint8_t*>(mapping.data);
    for (std::size_t offset = 0; offset < head.size_bytes; offset += sizeof(float)) {
      float value = 0.0f;
      std::memcpy(&value, data + offset, sizeof(value));
      if (!std::isfinite(value)) {
        throw std::runtime_error("sync: non-finite raw head " + head.name);
      }
    }
  }
  return backing;
}

RunSummary run_yolov8_sync(const std::string& tar_gz, const cv::Mat& img,
                           const SyncTestConfig& cfg) {
  RunSummary res;

  enable_detess_debug_defaults();
  std::cerr << debug_env_snapshot() << "\n";

  require(!tar_gz.empty(), "Failed to locate yolo_v8s model archive");

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
  model_opt.preprocess.kind = simaai::neat::InputKind::Image;
  model_opt.preprocess.enable = simaai::neat::AutoFlag::On;
  model_opt.preprocess.color_convert.input_format = simaai::neat::PreprocessColorFormat::BGR;
  if (env_bool("SIMA_DETESS_FORCE_STRETCH", false)) {
    model_opt.preprocess.resize.enable = simaai::neat::AutoFlag::On;
    model_opt.preprocess.resize.mode = simaai::neat::ResizeMode::Stretch;
    model_opt.preprocess.resize.width = 640;
    model_opt.preprocess.resize.height = 640;
    std::cerr << "[DBG] forcing preprocess stretch resize 640x640\n";
  }
  model_opt.upstream_name = "decoder";
  auto model = simaai::neat::Model(tar_gz, model_opt);
  const auto raw_heads = expected_raw_heads(model);
  const int topk = 100;

  // [canonical_pipeline]
  simaai::neat::Graph p;
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
  simaai::neat::Run runner = p.build_seeded_internal(
      simaai::neat::Sample{simaai::neat::Sample::from_image(
          img, simaai::neat::ImageSpec::PixelFormat::BGR, simaai::neat::TensorMemory::EV74)},
      simaai::neat::RunMode::Sync, run_opt);
  step_log("sync: after build");

  const auto start = std::chrono::steady_clock::now();
  bool logged_sample = false;
  bool noted_cpu_transfer = false;
  std::optional<simaai::neat::Sample> retained_first;
  simaai::neat::Mapping retained_view;
  std::vector<std::uint8_t> retained_bytes;
  std::optional<sima_test::DmaBufBackingIdentity> retained_backing;
  cv::Mat replacement_input;
  bool saw_raw_heads = false;
  bool checked_retained_view = false;
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
      auto outs = runner.run(
          simaai::neat::Sample{simaai::neat::Sample::from_image(
              i == 1 && retained_first ? replacement_input : img,
              simaai::neat::ImageSpec::PixelFormat::BGR, simaai::neat::TensorMemory::EV74)},
          pull_timeout_ms);
      require(outs.size() == 1, "sync: expected one output sample");
      out = std::move(outs.front());
      step_log("sync: after run");
    } catch (const std::exception& e) {
      append_note(res.note, "run_error=" + sanitize_note(e.what()));
      const std::string last = sanitize_note(runner.last_error());
      if (!last.empty()) {
        append_note(res.note, "runner_last_error=" + last);
      }
      std::cerr << "[DBG] runner.last_error after run_error\n" << runner.last_error() << "\n";
      break;
    }

    if (!logged_sample) {
      log_sample_caps(out, "appsink");
      logged_sample = true;
    }

    if (out.kind == simaai::neat::SampleKind::TensorSet) {
      saw_raw_heads = true;
      try {
        const auto backing = validate_raw_heads(out, raw_heads);
        if (i == 0 && cfg.iters > 1) {
          retained_first = out;
          retained_backing = backing;
          retained_view = retained_first->tensors.front().map_read();
          const auto* begin = static_cast<const std::uint8_t*>(retained_view.data);
          retained_bytes.assign(begin, begin + retained_view.size_bytes);
          replacement_input = cv::Mat::zeros(img.size(), img.type());
        } else if (i == 1 && retained_first) {
          require(retained_backing && backing != *retained_backing,
                  "sync: second output reused the still-retained first DMA allocation");
          require(std::memcmp(retained_view.data, retained_bytes.data(), retained_bytes.size()) ==
                      0,
                  "sync: retained first output changed during second frame execution");
          // Release before producing frame three: this checks two-carrier
          // progress, not unbounded progress while old user outputs stay live.
          retained_view = {};
          retained_first.reset();
          retained_backing.reset();
          retained_bytes.clear();
          checked_retained_view = true;
          append_note(res.note, "six_raw_heads_finite retained_view_stable");
        }
      } catch (const std::exception& ex) {
        append_note(res.note, "raw_head_validation=" + sanitize_note(ex.what()));
        break;
      }
      res.outputs += 1;
      continue;
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
      append_note(res.note, err);
      break; // Unreadable outputs must not count toward a twenty-frame pass.
    }

    if (fmt_upper == "BBOX") {
      try {
        const auto boxes = objdet::parse_boxes_strict(payload, img.cols, img.rows, topk, false);
        const objdet::MatchResult match =
            objdet::match_expected_boxes(boxes, expected, cfg.min_score, cfg.min_iou);
        if (!match.ok) {
          append_note(res.note, "verify_mismatch iter=" + std::to_string(i) + " " + match.note);
          break;
        }
      } catch (const std::exception& ex) {
        append_note(res.note, "bbox_parse_failed=" + sanitize_note(ex.what()));
        break;
      }
    } else {
      append_note(res.note, "unexpected_output_format=" + fmt_upper);
      break; // Raw heads are tensors, never an unlabelled serialized BBOX array.
    }

    res.outputs += 1;
  }
  const auto end = std::chrono::steady_clock::now();
  retained_view = {};
  retained_first.reset();

  res.diagnostics = p.last_pipeline();

  const double elapsed_s = std::chrono::duration<double>(end - start).count();
  res.avg_fps = (elapsed_s > 0.0) ? (static_cast<double>(res.outputs) / elapsed_s) : 0.0;
  res.ok = (res.outputs == cfg.iters);
  if (saw_raw_heads && cfg.iters > 1 && !checked_retained_view) {
    res.ok = false;
  }
  if (elapsed_s <= 0.0) {
    append_note(res.note, "sync_timing_incomplete");
    res.ok = false;
  }

  return res;
}

} // namespace

int main(int argc, char** argv) {
  try {
    enable_detess_debug_defaults();
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
