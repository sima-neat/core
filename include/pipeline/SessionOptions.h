/**
 * @file
 * @ingroup pipeline
 * @brief Runtime options and core Sample/Pull types.
 */
#pragma once

#include "pipeline/SessionReport.h"
#include "pipeline/TensorTypes.h"
#include "pipeline/Tensor.h"
#include "pipeline/TensorCore.h"

#include <cstddef>
#include <functional>
#include <initializer_list>
#include <optional>
#include <string>
#include <vector>

namespace simaai::neat {

struct RtspServerOptions {
  std::string mount = "image";
  int port = 8554;
  // Optional RTP/RTCP UDP port range. When set, the RTSP server will only
  // allocate ports within [rtp_port_base, rtp_port_base + rtp_port_count - 1].
  int rtp_port_base = -1;
  int rtp_port_count = 0;
};

struct ValidateOptions {
  bool parse_launch = true;  // build gst pipeline and verify element naming contract
  bool enforce_names = true; // ensure no unnamed/foreign elements exist
};

enum class RunMode {
  Async,
  Sync,
};

struct SessionOptions {
  int callback_timeout_ms = 1000;
  // Optional element name prefix/suffix applied per session (for uniqueness).
  // Prefix/suffix are sanitized to valid element-name characters.
  std::string element_name_prefix;
  std::string element_name_suffix;
};

struct OutputTensorOptions {
  std::string format = "RGB";
  TensorDType dtype = TensorDType::UInt8;

  int target_width = -1;
  int target_height = -1;
  int target_fps = -1;
};

enum class SampleKind {
  Tensor,
  Bundle,
  Unknown,
};

enum class PullStatus {
  Ok,
  Timeout,
  Closed,
  Error,
};

struct PullError {
  // User-facing message. Should include [code] prefix when code is available.
  std::string message;
  // Canonical machine-triage code. Prefer values from pipeline/ErrorCodes.h.
  std::string code;
  // Optional structured report for runtime/plugin failures.
  std::optional<SessionReport> report;
};

struct Sample {
  SampleKind kind = SampleKind::Unknown;
  bool owned = true;

  std::optional<simaai::neat::Tensor> tensor;
  std::vector<Sample> fields;

  std::string caps_string;
  std::string media_type;
  std::string payload_tag;
  std::string format; // Deprecated: use payload_tag.

  int64_t frame_id = -1;
  std::string stream_id;
  std::string port_name;
  int output_index = -1;
  int64_t input_seq = -1;
  int64_t orig_input_seq = -1;
  int64_t pts_ns = -1;
  int64_t dts_ns = -1;
  int64_t duration_ns = -1;
};

inline Sample make_tensor_sample(const std::string& port_name, simaai::neat::Tensor tensor) {
  Sample out;
  out.kind = SampleKind::Tensor;
  out.port_name = port_name;
  out.tensor = std::move(tensor);
  return out;
}

inline Sample make_bundle_sample(std::initializer_list<Sample> fields) {
  Sample out;
  out.kind = SampleKind::Bundle;
  out.fields = fields;
  return out;
}

} // namespace simaai::neat
