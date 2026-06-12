#ifndef SIMA_NEAT_INTERNAL
#define SIMA_NEAT_INTERNAL 1
#endif
#include "gst/GstHelpers.h"
#include "gst/GstInit.h"
#include "model/Model.h"
#include "model/internal/ModelInternal.h"
#include "model/internal/ModelPack.h"
#include "pipeline/StageRun.h"
#include "pipeline/TessellatedTensor.h"
#include "pipeline/Graph.h"
#include "pipeline/TensorAdapters.h"
#include "pipeline/internal/RenderedMlaContractQuery.h"
#include "pipeline/internal/InputStreamUtil.h"
#include "pipeline/internal/GstDataAdapter.h"
#include "pipeline/internal/TensorUtil.h"
#include "pipeline/internal/TensorTransfer.h"
#include "pipeline/runtime/RunInternal.h"

#include "e2e_pipelines/obj_detection/obj_detection_utils.h"
#include "e2e_pipelines/obj_detection/yolov8_test_utils.h"
#include "nodes/groups/ModelGroups.h"
#include "nodes/io/Input.h"
#include "nodes/sima/Cast.h"
#include "nodes/sima/Preproc.h"
#include "nodes/sima/SimaBoxDecode.h"
#include "test_utils.h"

#include <nlohmann/json.hpp>

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdlib>
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <functional>
#include <iomanip>
#include <initializer_list>
#include <iostream>
#include <limits>
#include <memory>
#include <optional>
#include <sstream>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

#if defined(SIMA_WITH_OPENCV)
#include <opencv2/imgproc.hpp>
#include <opencv2/dnn.hpp>
#endif

int default_model_run_timeout_ms() {
  constexpr int kDefaultTimeoutMs = 10000;
  const char* env = std::getenv("SIMA_YOLOV8_VARIANT_RUN_TIMEOUT_MS");
  if (!env || !*env)
    return kDefaultTimeoutMs;
  const int val = std::atoi(env);
  return (val > 0) ? val : kDefaultTimeoutMs;
}

constexpr int kRunRetryTimeoutMs = 10000;
constexpr int kFixedPreprocTargetWidth = 640;
constexpr int kFixedPreprocTargetHeight = 640;
constexpr int kPadValue = 0;

namespace fs = std::filesystem;
namespace rendered_stage_query = simaai::neat::pipeline_internal::rendered_stage_query;

namespace {

enum class BoxDecodeRunMode {
  NoModel,
  Model,
};

const char* boxdecode_run_mode_name(BoxDecodeRunMode mode) {
  switch (mode) {
  case BoxDecodeRunMode::NoModel:
    return "no_model";
  case BoxDecodeRunMode::Model:
    return "model";
  }
  return "unknown";
}

enum class PreAdapterKind {
  None,
  Preproc,
  Quant,
  Tess,
  QuantTess,
  Unknown,
};

enum class PostAdapterKind {
  None,
  DetessDequant,
  Dequant,
  BoxDecode,
  Unknown,
};

enum class TerminalOutputKind {
  TensorQuantized,
  TensorFloatLike,
  BoxDecodePayload,
  Unknown,
};

enum class RouteKind {
  PreInferPost,
  InferOnlyBf16NoPost,
  QuantizedNoPostFallback,
};

struct ProbeResult {
  std::string tar_path;
  std::string model_id;
  fs::path etc_dir;

  PreAdapterKind pre_kind = PreAdapterKind::None;
  PostAdapterKind post_kind = PostAdapterKind::None;

  bool has_pre_adapter = false;
  bool has_post_adapter = false;
  bool has_postproc_config = false;
  bool has_dequant_config = false;
  bool has_pipeline_sequence = false;
  bool tess_within_mla = false;
  bool input_spec_tensor_mode = false;

  std::string mla_input_dtype_raw;
  std::string mla_output_dtype_raw;
  std::string mla_input_media_type;
  bool mla_input_bf16 = false;
  bool mla_input_int8 = false;
  bool mla_output_bf16 = false;
  bool mla_output_int8 = false;
  TerminalOutputKind terminal_output_kind = TerminalOutputKind::Unknown;

  std::vector<std::string> evidence;
};

struct ModelCase {
  ProbeResult probe;
  RouteKind route = RouteKind::PreInferPost;
};

struct ExecutionResult {
  bool ok = false;
  std::string note;
  RouteKind route = RouteKind::PreInferPost;
  std::string model_id;
  std::string output_signature;
  bool host_dequant_ready = false;
  simaai::neat::pipeline_internal::TensorIoStats tensor_io;
};

struct AccuracyResult {
  bool ok = false;
  int parsed_boxes = 0;
  bool skipped = false;
  std::string note;
};

std::vector<simaai::neat::Tensor> tensors_in_sample(const simaai::neat::Sample& sample);
std::optional<simaai::neat::Tensor> first_tensor_in_sample(const simaai::neat::Sample& sample);

std::string upper_copy(std::string s) {
  for (char& c : s) {
    c = static_cast<char>(std::toupper(static_cast<unsigned char>(c)));
  }
  return s;
}

std::string lower_copy(std::string s) {
  for (char& c : s) {
    c = static_cast<char>(std::tolower(static_cast<unsigned char>(c)));
  }
  return s;
}

std::string normalize_processcvu_run_target_token(std::string token) {
  token = upper_copy(std::move(token));
  if (token.empty() || token == "AUTO") {
    return "AUTO";
  }
  if (token == "EV74") {
    return "EV74";
  }
  if (token == "A65" || token == "APU") {
    return "A65";
  }
  throw std::runtime_error("invalid --processcvu-run-target: " + token);
}

simaai::neat::pipeline_internal::TensorIoStats
tensor_io_delta(const simaai::neat::pipeline_internal::TensorIoStats& before,
                const simaai::neat::pipeline_internal::TensorIoStats& after) {
  simaai::neat::pipeline_internal::TensorIoStats delta;
  delta.tensor_copy_count = after.tensor_copy_count - before.tensor_copy_count;
  delta.tensor_copy_bytes = after.tensor_copy_bytes - before.tensor_copy_bytes;
  delta.tensor_view_count = after.tensor_view_count - before.tensor_view_count;
  delta.gst_memory_map_calls = after.gst_memory_map_calls - before.gst_memory_map_calls;
  delta.holder_fast_path_hits = after.holder_fast_path_hits - before.holder_fast_path_hits;
  delta.bundle_projection_count = after.bundle_projection_count - before.bundle_projection_count;
  delta.packed_view_reuse_hits = after.packed_view_reuse_hits - before.packed_view_reuse_hits;
  delta.packed_view_reuse_opportunities =
      after.packed_view_reuse_opportunities - before.packed_view_reuse_opportunities;
  return delta;
}

std::string tensor_io_stats_string(const simaai::neat::pipeline_internal::TensorIoStats& stats) {
  std::ostringstream os;
  os << "copies=" << stats.tensor_copy_count << ",copy_bytes=" << stats.tensor_copy_bytes
     << ",views=" << stats.tensor_view_count << ",maps=" << stats.gst_memory_map_calls
     << ",holder_fast=" << stats.holder_fast_path_hits
     << ",bundle_proj=" << stats.bundle_projection_count
     << ",packed_reuse=" << stats.packed_view_reuse_hits << "/"
     << stats.packed_view_reuse_opportunities << ",packed_ratio=" << std::fixed
     << std::setprecision(3) << stats.packed_view_reuse_ratio();
  return os.str();
}

bool ends_with_ci(const std::string& value, const std::string& suffix) {
  if (suffix.size() > value.size())
    return false;
  const std::string v = lower_copy(value);
  const std::string s = lower_copy(suffix);
  return std::equal(s.rbegin(), s.rend(), v.rbegin());
}

bool matrix_input_debug_enabled() {
  const char* raw = std::getenv("SIMA_YOLO_MATRIX_DEBUG_INPUT");
  return raw && *raw && std::strcmp(raw, "0") != 0;
}

bool env_flag_enabled(const char* key) {
  if (!key || !*key) {
    return false;
  }
  const char* raw = std::getenv(key);
  if (!raw || !*raw) {
    return false;
  }
  std::string value = lower_copy(std::string(raw));
  if (value == "0" || value == "false" || value == "no" || value == "off") {
    return false;
  }
  return true;
}

bool pre_mla_parity_enabled() {
  const char* raw = std::getenv("SIMA_YOLOV8_PRE_MLA_PARITY");
  if (!raw || !*raw) {
    return true;
  }
  return env_flag_enabled("SIMA_YOLOV8_PRE_MLA_PARITY");
}

bool pre_mla_parity_debug_enabled() {
  return env_flag_enabled("SIMA_YOLOV8_PRE_MLA_PARITY_DEBUG");
}

bool pre_mla_parity_strict_enabled() {
  const char* raw = std::getenv("SIMA_YOLOV8_PRE_MLA_PARITY_STRICT");
  if (!raw || !*raw) {
    return true;
  }
  return env_flag_enabled("SIMA_YOLOV8_PRE_MLA_PARITY_STRICT");
}

bool shadow_letterbox_enabled() {
  return env_flag_enabled("SHADOW_LETTERBOX");
}

struct LetterboxGeometry {
  int resized_w = 0;
  int resized_h = 0;
  int pad_left = 0;
  int pad_right = 0;
  int pad_top = 0;
  int pad_bottom = 0;
  double scale = 1.0;
};

struct PreprocKernelContract {
  int output_width = 0;
  int output_height = 0;
  bool has_aspect_ratio = false;
  bool aspect_ratio = false;
  bool has_normalize = false;
  bool normalize = true;
  std::string output_img_type = "RGB";
  std::string scaling_type = "BILINEAR";
  std::string padding_type = "CENTER";
  std::array<float, 3> channel_mean = {0.0f, 0.0f, 0.0f};
  std::array<float, 3> channel_stddev = {1.0f, 1.0f, 1.0f};
};

bool route_uses_letterbox(RouteKind route) {
  if (shadow_letterbox_enabled()) {
    return true;
  }
  // Shadow runs currently validate backend behavior that assumes
  // letterbox-style remap on post boxdecode routes.
  if (env_flag_enabled("SHADOW_CHANGE") && route == RouteKind::PreInferPost) {
    return true;
  }
  return route == RouteKind::InferOnlyBf16NoPost;
}

// CVU preproc kernel only exposes horizontal padding (left_right_pad). For
// wide inputs to square outputs, aspect-ratio preservation would require
// vertical padding, so runtime effectively falls back to stretch.
bool kernel_preproc_can_letterbox_dims(int src_w, int src_h, int dst_w, int dst_h) {
  if (src_w <= 0 || src_h <= 0 || dst_w <= 0 || dst_h <= 0) {
    return false;
  }
  return static_cast<long long>(src_w) * static_cast<long long>(dst_h) <=
         static_cast<long long>(src_h) * static_cast<long long>(dst_w);
}

LetterboxGeometry compute_letterbox_geometry(int src_w, int src_h, int dst_w, int dst_h,
                                             const std::string& padding_type = "CENTER") {
  LetterboxGeometry g;
  if (src_w <= 0 || src_h <= 0 || dst_w <= 0 || dst_h <= 0) {
    return g;
  }
  const long long d = static_cast<long long>(src_h) * static_cast<long long>(dst_w) -
                      static_cast<long long>(src_w) * static_cast<long long>(dst_h);
  if (d < 0) {
    g.resized_w = dst_w;
    g.resized_h = std::max(
        1, static_cast<int>(std::ceil(static_cast<double>(src_h) * static_cast<double>(dst_w) /
                                      static_cast<double>(src_w))));
  } else {
    g.resized_h = dst_h;
    g.resized_w = std::max(
        1, static_cast<int>(std::ceil(static_cast<double>(src_w) * static_cast<double>(dst_h) /
                                      static_cast<double>(src_h))));
  }
  g.resized_w = std::min(g.resized_w, dst_w);
  g.resized_h = std::min(g.resized_h, dst_h);
  g.scale = std::min(static_cast<double>(g.resized_w) / static_cast<double>(src_w),
                     static_cast<double>(g.resized_h) / static_cast<double>(src_h));
  const int pad_x = std::max(0, dst_w - g.resized_w);
  const int pad_y = std::max(0, dst_h - g.resized_h);
  const std::string pad = upper_copy(padding_type);
  if (pad == "TOP_LEFT") {
    g.pad_left = 0;
    g.pad_top = 0;
  } else if (pad == "TOP_RIGHT") {
    g.pad_left = pad_x;
    g.pad_top = 0;
  } else if (pad == "BOTTOM_LEFT") {
    g.pad_left = 0;
    g.pad_top = pad_y;
  } else if (pad == "BOTTOM_RIGHT") {
    g.pad_left = pad_x;
    g.pad_top = pad_y;
  } else {
    g.pad_left = pad_x / 2;
    g.pad_top = pad_y / 2;
  }
  g.pad_right = pad_x - g.pad_left;
  g.pad_bottom = pad_y - g.pad_top;
  return g;
}

cv::Mat letterbox_resize_local(const cv::Mat& src, int dst_w, int dst_h);
cv::Mat letterbox_resize_local(const cv::Mat& src, int dst_w, int dst_h, int interpolation,
                               const cv::Scalar& pad_value, const std::string& padding_type);

cv::Mat resize_to_target_local(const cv::Mat& src, int dst_w, int dst_h, bool letterbox) {
  if (!src.data || dst_w <= 0 || dst_h <= 0) {
    return src;
  }
  if (letterbox) {
    return letterbox_resize_local(src, dst_w, dst_h);
  }
  if (src.cols == dst_w && src.rows == dst_h) {
    return src;
  }
  cv::Mat out;
  cv::resize(src, out, cv::Size(dst_w, dst_h));
  return out;
}

cv::Mat resize_to_target_local(const cv::Mat& src, int dst_w, int dst_h, bool letterbox,
                               int interpolation, const cv::Scalar& pad_value,
                               const std::string& padding_type) {
  if (!src.data || dst_w <= 0 || dst_h <= 0) {
    return src;
  }
  if (letterbox) {
    return letterbox_resize_local(src, dst_w, dst_h, interpolation, pad_value, padding_type);
  }
  if (src.cols == dst_w && src.rows == dst_h) {
    return src;
  }
  cv::Mat out;
  cv::resize(src, out, cv::Size(dst_w, dst_h), 0.0, 0.0, interpolation);
  return out;
}

bool has_json_suffix(const fs::path& dir, const char* suffix) {
  if (!suffix || !*suffix)
    return false;
  std::error_code ec;
  if (!fs::exists(dir, ec) || !fs::is_directory(dir, ec))
    return false;
  for (const auto& entry : fs::directory_iterator(dir, ec)) {
    if (ec)
      break;
    if (!entry.is_regular_file())
      continue;
    if (ends_with_ci(entry.path().filename().string(), suffix))
      return true;
  }
  return false;
}

std::optional<fs::path> find_json_suffix_path(const fs::path& dir, const char* suffix) {
  if (!suffix || !*suffix)
    return std::nullopt;
  std::error_code ec;
  if (!fs::exists(dir, ec) || !fs::is_directory(dir, ec))
    return std::nullopt;
  std::vector<fs::path> matches;
  for (const auto& entry : fs::directory_iterator(dir, ec)) {
    if (ec)
      break;
    if (!entry.is_regular_file())
      continue;
    const auto name = entry.path().filename().string();
    if (!ends_with_ci(name, suffix))
      continue;
    matches.push_back(entry.path());
  }
  if (matches.empty())
    return std::nullopt;
  std::sort(matches.begin(), matches.end());
  return matches.front();
}

std::optional<nlohmann::json> read_json_file(const fs::path& path) {
  std::ifstream in(path);
  if (!in.is_open()) {
    return std::nullopt;
  }
  nlohmann::json j;
  try {
    in >> j;
  } catch (const std::exception&) {
    return std::nullopt;
  }
  if (!j.is_object()) {
    return std::nullopt;
  }
  return j;
}

std::optional<std::string> read_stringish_field(const nlohmann::json& obj, const char* key) {
  if (!key || !*key || !obj.contains(key)) {
    return std::nullopt;
  }
  const auto& v = obj.at(key);
  if (v.is_string()) {
    const std::string out = v.get<std::string>();
    if (!out.empty())
      return out;
    return std::nullopt;
  }
  if (v.is_array()) {
    for (const auto& item : v) {
      if (!item.is_string())
        continue;
      const std::string out = item.get<std::string>();
      if (!out.empty())
        return out;
    }
  }
  return std::nullopt;
}

std::optional<int> read_intish_field(const nlohmann::json& obj, const char* key) {
  if (!key || !*key || !obj.contains(key)) {
    return std::nullopt;
  }
  const auto& v = obj.at(key);
  if (v.is_number_integer()) {
    return v.get<int>();
  }
  if (v.is_number_float()) {
    return static_cast<int>(std::lround(v.get<double>()));
  }
  if (v.is_string()) {
    try {
      return std::stoi(v.get<std::string>());
    } catch (const std::exception&) {
      return std::nullopt;
    }
  }
  return std::nullopt;
}

std::optional<bool> read_boolish_field(const nlohmann::json& obj, const char* key) {
  if (!key || !*key || !obj.contains(key)) {
    return std::nullopt;
  }
  const auto& v = obj.at(key);
  if (v.is_boolean()) {
    return v.get<bool>();
  }
  if (v.is_number_integer()) {
    return v.get<int>() != 0;
  }
  if (v.is_string()) {
    const std::string s = lower_copy(v.get<std::string>());
    if (s == "1" || s == "true" || s == "yes" || s == "on") {
      return true;
    }
    if (s == "0" || s == "false" || s == "no" || s == "off") {
      return false;
    }
  }
  return std::nullopt;
}

void read_float_array_3(const nlohmann::json& obj, const char* key, std::array<float, 3>& out) {
  if (!key || !*key || !obj.contains(key)) {
    return;
  }
  const auto& arr = obj.at(key);
  if (!arr.is_array()) {
    return;
  }
  for (size_t i = 0; i < out.size() && i < arr.size(); ++i) {
    if (!arr[i].is_number()) {
      continue;
    }
    out[i] = static_cast<float>(arr[i].get<double>());
  }
}

int cv_interpolation_from_preproc_scaling_type(std::string scaling_type) {
  scaling_type = upper_copy(std::move(scaling_type));
  if (scaling_type == "NEAREST_NEIGHBOUR" || scaling_type == "NEAREST_NEIGHBOR") {
    return cv::INTER_NEAREST;
  }
  if (scaling_type == "BICUBIC") {
    return cv::INTER_CUBIC;
  }
  if (scaling_type == "INTERAREA") {
    return cv::INTER_AREA;
  }
  return cv::INTER_LINEAR;
}

int cv_interpolation_from_env_override(const char* value, int fallback) {
  if (!value || !*value) {
    return fallback;
  }
  const std::string v = upper_copy(std::string(value));
  if (v == "NEAREST" || v == "INTER_NEAREST") {
    return cv::INTER_NEAREST;
  }
  if (v == "LINEAR" || v == "BILINEAR" || v == "INTER_LINEAR") {
    return cv::INTER_LINEAR;
  }
  if (v == "CUBIC" || v == "BICUBIC" || v == "INTER_CUBIC") {
    return cv::INTER_CUBIC;
  }
  if (v == "AREA" || v == "INTER_AREA") {
    return cv::INTER_AREA;
  }
  return fallback;
}

std::optional<PreprocKernelContract>
read_preproc_contract_from_model(const simaai::neat::Model& model) {
  std::string mla_cfg = model.find_config_path_by_processor("MLA");
  if (mla_cfg.empty()) {
    mla_cfg = model.find_config_path_by_plugin("processmla");
  }
  if (mla_cfg.empty()) {
    return std::nullopt;
  }
  const fs::path etc_dir = fs::path(mla_cfg).parent_path();
  const auto preproc_path = find_json_suffix_path(etc_dir, "_preproc.json");
  if (!preproc_path.has_value()) {
    return std::nullopt;
  }
  const auto j = read_json_file(*preproc_path);
  if (!j.has_value()) {
    return std::nullopt;
  }

  PreprocKernelContract contract;
  if (const auto v = read_intish_field(*j, "output_width"); v.has_value()) {
    contract.output_width = *v;
  }
  if (const auto v = read_intish_field(*j, "output_height"); v.has_value()) {
    contract.output_height = *v;
  }
  if (const auto v = read_boolish_field(*j, "aspect_ratio"); v.has_value()) {
    contract.has_aspect_ratio = true;
    contract.aspect_ratio = *v;
  }
  if (const auto v = read_boolish_field(*j, "normalize"); v.has_value()) {
    contract.has_normalize = true;
    contract.normalize = *v;
  }
  if (const auto v = read_stringish_field(*j, "output_img_type"); v.has_value()) {
    contract.output_img_type = upper_copy(*v);
  }
  if (const auto v = read_stringish_field(*j, "scaling_type"); v.has_value()) {
    contract.scaling_type = upper_copy(*v);
  }
  if (const auto v = read_stringish_field(*j, "padding_type"); v.has_value()) {
    contract.padding_type = upper_copy(*v);
  }
  read_float_array_3(*j, "channel_mean", contract.channel_mean);
  read_float_array_3(*j, "channel_stddev", contract.channel_stddev);
  for (float& s : contract.channel_stddev) {
    if (!(s > 0.0f)) {
      s = 1.0f;
    }
  }

  return contract;
}

std::string read_dtype_from_json(const fs::path& path, std::initializer_list<const char*> keys) {
  const auto j = read_json_file(path);
  if (!j.has_value()) {
    return {};
  }
  for (const char* key : keys) {
    if (const auto v = read_stringish_field(*j, key); v.has_value())
      return *v;
  }
  if (j->contains("simaai__params") && (*j)["simaai__params"].is_object()) {
    const auto& params = (*j)["simaai__params"];
    for (const char* key : keys) {
      if (const auto v = read_stringish_field(params, key); v.has_value())
        return *v;
    }
  }
  return {};
}

float infer_fp32_input_scale_from_mpk(const simaai::neat::Model& model) {
  std::string mla_cfg = model.find_config_path_by_processor("MLA");
  if (mla_cfg.empty()) {
    mla_cfg = model.find_config_path_by_plugin("processmla");
  }
  if (mla_cfg.empty()) {
    return 1.0f;
  }
  const fs::path etc_dir = fs::path(mla_cfg).parent_path();
  const auto mpk_path = find_json_suffix_path(etc_dir, "_mpk.json");
  if (!mpk_path.has_value()) {
    return 1.0f;
  }
  const auto j = read_json_file(*mpk_path);
  if (!j.has_value() || !j->contains("input_nodes") || !(*j)["input_nodes"].is_array()) {
    return 1.0f;
  }
  for (const auto& node : (*j)["input_nodes"]) {
    if (!node.is_object() || !node.contains("input_range") || !node["input_range"].is_array()) {
      continue;
    }
    const auto& input_range = node["input_range"];
    if (input_range.size() < 2 || !input_range[0].is_number() || !input_range[1].is_number()) {
      continue;
    }
    const double lo = input_range[0].get<double>();
    const double hi = input_range[1].get<double>();
    if (hi > 0.0 && lo >= -1e-6 && hi <= (1.0 + 1e-6)) {
      return 1.0f / 255.0f;
    }
    break;
  }
  return 1.0f;
}

float read_env_float_or_default(const char* key, float fallback) {
  if (!key || !*key) {
    return fallback;
  }
  const char* raw = std::getenv(key);
  if (!raw || !*raw) {
    return fallback;
  }
  char* end = nullptr;
  const float v = std::strtof(raw, &end);
  if (!end || end == raw) {
    return fallback;
  }
  return v;
}

std::string dtype_name(simaai::neat::TensorDType dtype) {
  switch (dtype) {
  case simaai::neat::TensorDType::UInt8:
    return "UInt8";
  case simaai::neat::TensorDType::Int8:
    return "Int8";
  case simaai::neat::TensorDType::UInt16:
    return "UInt16";
  case simaai::neat::TensorDType::Int16:
    return "Int16";
  case simaai::neat::TensorDType::Int32:
    return "Int32";
  case simaai::neat::TensorDType::BFloat16:
    return "BFloat16";
  case simaai::neat::TensorDType::Float32:
    return "Float32";
  case simaai::neat::TensorDType::Float64:
    return "Float64";
  }
  return "Unknown";
}

std::string shape_string(const std::vector<int64_t>& shape) {
  if (shape.empty())
    return "[]";
  std::string out = "[";
  for (size_t i = 0; i < shape.size(); ++i) {
    if (i)
      out += "x";
    out += std::to_string(shape[i]);
  }
  out += "]";
  return out;
}

std::string route_name(RouteKind route) {
  switch (route) {
  case RouteKind::PreInferPost:
    return "A_pre_infer_post";
  case RouteKind::InferOnlyBf16NoPost:
    return "B_infer_only_bf16";
  case RouteKind::QuantizedNoPostFallback:
    return "C_quant_no_post_host_dequant";
  }
  return "unknown";
}

std::string terminal_output_kind_name(TerminalOutputKind kind) {
  switch (kind) {
  case TerminalOutputKind::TensorQuantized:
    return "tensor_quantized";
  case TerminalOutputKind::TensorFloatLike:
    return "tensor_float_like";
  case TerminalOutputKind::BoxDecodePayload:
    return "boxdecode_payload";
  case TerminalOutputKind::Unknown:
  default:
    return "unknown";
  }
}

std::string pre_kind_name(PreAdapterKind kind) {
  switch (kind) {
  case PreAdapterKind::None:
    return "none";
  case PreAdapterKind::Preproc:
    return "preproc";
  case PreAdapterKind::Quant:
    return "quant";
  case PreAdapterKind::Tess:
    return "tess";
  case PreAdapterKind::QuantTess:
    return "quanttess";
  case PreAdapterKind::Unknown:
    return "unknown";
  }
  return "unknown";
}

std::string post_kind_name(PostAdapterKind kind) {
  switch (kind) {
  case PostAdapterKind::None:
    return "none";
  case PostAdapterKind::DetessDequant:
    return "detessdequant";
  case PostAdapterKind::Dequant:
    return "dequant";
  case PostAdapterKind::BoxDecode:
    return "boxdecode";
  case PostAdapterKind::Unknown:
    return "unknown";
  }
  return "unknown";
}

bool dtype_is_bf16_like(std::string raw) {
  raw = upper_copy(std::move(raw));
  return raw.find("BF16") != std::string::npos || raw.find("BFLOAT16") != std::string::npos;
}

bool dtype_is_int8_like(std::string raw) {
  raw = upper_copy(std::move(raw));
  return raw.find("INT8") != std::string::npos || raw.find("UINT8") != std::string::npos;
}

bool media_type_is_tensor(std::string raw) {
  raw = lower_copy(std::move(raw));
  return raw == "application/vnd.simaai.tensor";
}

std::string tensor_format(const simaai::neat::Tensor& tensor) {
  if (tensor.semantic.tess.has_value() && !tensor.semantic.tess->format.empty()) {
    return upper_copy(tensor.semantic.tess->format);
  }
  if (tensor.semantic.image.has_value()) {
    using PF = simaai::neat::ImageSpec::PixelFormat;
    switch (tensor.semantic.image->format) {
    case PF::RGB:
      return "RGB";
    case PF::BGR:
      return "BGR";
    case PF::GRAY8:
      return "GRAY8";
    case PF::NV12:
      return "NV12";
    case PF::I420:
      return "I420";
    case PF::UNKNOWN:
      break;
    }
  }
  return "";
}

uint16_t fp32_to_bf16(float value) {
  uint32_t bits = 0;
  std::memcpy(&bits, &value, sizeof(bits));
  const uint32_t lsb = (bits >> 16) & 1u;
  bits += 0x7FFFu + lsb;
  return static_cast<uint16_t>(bits >> 16);
}

std::string bf16_probe_input_color_format(const simaai::neat::Model& model) {
  const auto opt = model.input_appsrc_options(true);
  const std::string opt_fmt = upper_copy(opt.format.str());
  if (opt_fmt == "RGB" || opt_fmt == "BGR" || opt_fmt == "GRAY8" || opt_fmt == "GRAY") {
    return opt_fmt;
  }

  auto normalize_color_like = [](const std::string& raw) {
    const std::string up = upper_copy(raw);
    if (up.empty())
      return std::string{};
    const bool has_rgb = up.find("RGB") != std::string::npos;
    const bool has_bgr = up.find("BGR") != std::string::npos;
    if (has_rgb && has_bgr) {
      return std::string{};
    }
    if (has_rgb) {
      return std::string("RGB");
    }
    if (has_bgr) {
      return std::string("BGR");
    }
    if (up.find("GRAY") != std::string::npos || up.find("GRAY8") != std::string::npos) {
      return std::string("GRAY8");
    }
    return std::string{};
  };

  const std::initializer_list<const char*> color_fields = {"output_img_type", "input_img_type",
                                                           "output_format", "input_format"};
  std::vector<std::string> cfg_paths;
  cfg_paths.emplace_back(model.find_config_path_by_processor("CVU"));
  cfg_paths.emplace_back(model.find_config_path_by_plugin("processcvu"));
  cfg_paths.emplace_back(model.find_config_path_by_processor("MLA"));
  cfg_paths.emplace_back(model.find_config_path_by_plugin("processmla"));
  for (const auto& p : cfg_paths) {
    if (p.empty())
      continue;
    const auto color = read_dtype_from_json(p, color_fields);
    const std::string mapped = normalize_color_like(color);
    if (!mapped.empty())
      return mapped;
  }

  return {};
}

std::string bf16_input_format_for_model(const simaai::neat::Model& model, const cv::Mat& img_bgr) {
  const char* override_fmt = std::getenv("SIMA_YOLOV8_BF16_INPUT_FMT");
  if (override_fmt && *override_fmt) {
    return upper_copy(override_fmt);
  }
  const std::string hint = bf16_probe_input_color_format(model);
  if (!hint.empty())
    return hint;
  if (img_bgr.channels() == 1)
    return "GRAY8";
  return "BGR";
}

std::string canonical_color_format(std::string fmt) {
  fmt = upper_copy(std::move(fmt));
  if (fmt == "GRAY" || fmt == "GREY") {
    return "GRAY8";
  }
  if (fmt == "RGB" || fmt == "BGR" || fmt == "GRAY8") {
    return fmt;
  }
  return "BGR";
}

cv::Mat convert_color_for_format(const cv::Mat& src, std::string target_format, const char* where) {
  if (!src.data) {
    return src;
  }
  target_format = canonical_color_format(std::move(target_format));
  cv::Mat out;
  if (target_format == "RGB") {
    if (src.channels() == 1) {
      cv::cvtColor(src, out, cv::COLOR_GRAY2RGB);
    } else if (src.channels() == 3) {
      cv::cvtColor(src, out, cv::COLOR_BGR2RGB);
    } else if (src.channels() == 4) {
      cv::cvtColor(src, out, cv::COLOR_BGRA2RGB);
    } else {
      throw std::runtime_error(std::string(where) + ": unsupported input channel count");
    }
    return out;
  }
  if (target_format == "GRAY8") {
    if (src.channels() == 3) {
      cv::cvtColor(src, out, cv::COLOR_BGR2GRAY);
    } else if (src.channels() == 4) {
      cv::cvtColor(src, out, cv::COLOR_BGRA2GRAY);
    } else if (src.channels() == 1) {
      out = src;
    } else {
      throw std::runtime_error(std::string(where) + ": unsupported input channel count");
    }
    return out;
  }
  // Default to BGR.
  if (src.channels() == 1) {
    cv::cvtColor(src, out, cv::COLOR_GRAY2BGR);
  } else if (src.channels() == 4) {
    cv::cvtColor(src, out, cv::COLOR_BGRA2BGR);
  } else if (src.channels() == 3) {
    out = src;
  } else {
    throw std::runtime_error(std::string(where) + ": unsupported input channel count");
  }
  return out;
}

void apply_channel_normalization(cv::Mat& fp32, const std::array<float, 3>& mean,
                                 const std::array<float, 3>& stddev) {
  if (!fp32.data) {
    return;
  }
  if (fp32.channels() == 1) {
    cv::subtract(fp32, cv::Scalar(mean[0]), fp32);
    cv::divide(fp32, cv::Scalar(stddev[0]), fp32);
    return;
  }
  if (fp32.channels() >= 3) {
    cv::subtract(fp32, cv::Scalar(mean[0], mean[1], mean[2], 0.0f), fp32);
    cv::divide(fp32, cv::Scalar(stddev[0], stddev[1], stddev[2], 1.0f), fp32);
  }
}

simaai::neat::Tensor make_fp32_hwc_tensor_owned(const cv::Mat& fp32, const char* where) {
  const std::string tag = where ? where : "make_fp32_hwc_tensor_owned";
  if (!fp32.data || fp32.depth() != CV_32F || fp32.channels() <= 0) {
    throw std::runtime_error(tag + ": expected CV_32F tensor mat");
  }
  cv::Mat src = fp32;
  if (!src.isContinuous()) {
    src = src.clone();
  }
  const int rows = src.rows;
  const int cols = src.cols;
  const int channels = src.channels();
  if (rows <= 0 || cols <= 0 || channels <= 0) {
    throw std::runtime_error(tag + ": invalid tensor geometry");
  }

  const size_t elem_count =
      static_cast<size_t>(rows) * static_cast<size_t>(cols) * static_cast<size_t>(channels);
  auto storage = simaai::neat::make_cpu_owned_storage(elem_count * sizeof(float));
  auto map = storage->map(simaai::neat::MapMode::Write);
  if (!map.data || map.size_bytes < elem_count * sizeof(float)) {
    throw std::runtime_error(tag + ": failed to map owned storage");
  }
  std::memcpy(map.data, src.data, elem_count * sizeof(float));

  simaai::neat::Tensor tensor;
  tensor.storage = std::move(storage);
  tensor.dtype = simaai::neat::TensorDType::Float32;
  tensor.device = {simaai::neat::DeviceType::CPU, 0};
  tensor.byte_offset = 0;
  tensor.read_only = true;
  if (channels == 1) {
    tensor.layout = simaai::neat::TensorLayout::HW;
    tensor.shape = {rows, cols};
    tensor.strides_bytes = {static_cast<int64_t>(cols) * static_cast<int64_t>(sizeof(float)),
                            static_cast<int64_t>(sizeof(float))};
  } else {
    tensor.layout = simaai::neat::TensorLayout::HWC;
    tensor.shape = {rows, cols, channels};
    tensor.strides_bytes = {static_cast<int64_t>(cols) * static_cast<int64_t>(channels) *
                                static_cast<int64_t>(sizeof(float)),
                            static_cast<int64_t>(channels) * static_cast<int64_t>(sizeof(float)),
                            static_cast<int64_t>(sizeof(float))};
  }
  return tensor;
}

float bf16_to_fp32(uint16_t value) {
  uint32_t bits = static_cast<uint32_t>(value) << 16;
  float out = 0.0f;
  std::memcpy(&out, &bits, sizeof(out));
  return out;
}

std::pair<int, int> infer_hw_from_input_spec(const simaai::neat::TensorSpec& spec) {
  if (spec.shape.size() < 2) {
    return {640, 640};
  }
  int64_t h = spec.shape[0];
  int64_t w = spec.shape[1];
  if (spec.shape.size() >= 3) {
    const int64_t d0 = spec.shape[0];
    const int64_t d1 = spec.shape[1];
    const int64_t d2 = spec.shape[2];
    if (spec.shape.size() == 4 && d1 > 0 && d1 <= 8 && d2 > 0) {
      const int64_t d3 = spec.shape[3];
      if (d3 > 0) {
        h = d2;
        w = d3;
      }
    } else if (d0 > 0 && d0 <= 8 && d1 > 0 && d2 > 0) {
      // CHW / NCHW-like: treat first dim as channels when small.
      h = d1;
      w = d2;
    }
  }
  if (w <= 0 || h <= 0) {
    return {640, 640};
  }
  return {static_cast<int>(w), static_cast<int>(h)};
}

std::pair<int, int> preferred_input_hw(const simaai::neat::Model& model, RouteKind route,
                                       const simaai::neat::TensorSpec& spec) {
  if (route == RouteKind::QuantizedNoPostFallback) {
    const auto opt = model.input_appsrc_options(false);
    const int opt_w = opt.width > 0 ? opt.width : opt.max_width;
    const int opt_h = opt.height > 0 ? opt.height : opt.max_height;
    if (opt_w > 0 && opt_h > 0) {
      return {opt_w, opt_h};
    }
  }
  if (route == RouteKind::InferOnlyBf16NoPost) {
    const auto opt = model.input_appsrc_options(true);
    const int opt_w = opt.width > 0 ? opt.width : opt.max_width;
    const int opt_h = opt.height > 0 ? opt.height : opt.max_height;
    if (opt_w > 0 && opt_h > 0) {
      return {opt_w, opt_h};
    }
  }
  return infer_hw_from_input_spec(spec);
}

simaai::neat::PreprocessRuntimeMeta
build_preprocess_meta_stretch_local(int original_w, int original_h, int target_w, int target_h,
                                    bool normalize, bool quantize, bool tessellate) {
  simaai::neat::PreprocessRuntimeMeta meta;
  meta.original_width = original_w;
  meta.original_height = original_h;
  meta.resized_width = target_w;
  meta.resized_height = target_h;
  meta.scaled_width = target_w;
  meta.scaled_height = target_h;
  meta.pad_left = 0;
  meta.pad_right = 0;
  meta.pad_top = 0;
  meta.pad_bottom = 0;
  meta.resize_mode = "stretch";
  meta.color_in = "BGR";
  meta.color_out = "RGB";
  meta.axis_perm.clear();
  meta.normalize = normalize;
  meta.quantize = quantize;
  meta.tessellate = tessellate;

  const double sx = static_cast<double>(target_w) / static_cast<double>(original_w);
  const double sy = static_cast<double>(target_h) / static_cast<double>(original_h);
  const double inv_x = (sx > 0.0) ? (1.0 / sx) : 1.0;
  const double inv_y = (sy > 0.0) ? (1.0 / sy) : 1.0;
  meta.affine_m00 = inv_x;
  meta.affine_m01 = 0.0;
  meta.affine_m02 = 0.0;
  meta.affine_m10 = 0.0;
  meta.affine_m11 = inv_y;
  meta.affine_m12 = 0.0;
  meta.affine_scale_x = inv_x;
  meta.affine_scale_y = inv_y;
  meta.affine_offset_x = 0.0;
  meta.affine_offset_y = 0.0;
  return meta;
}

simaai::neat::PreprocessRuntimeMeta
build_preprocess_meta_letterbox_local(int original_w, int original_h, int target_w, int target_h,
                                      bool normalize, bool quantize, bool tessellate,
                                      const std::string& padding_type = "CENTER") {
  simaai::neat::PreprocessRuntimeMeta meta;
  meta.original_width = original_w;
  meta.original_height = original_h;
  meta.resized_width = target_w;
  meta.resized_height = target_h;
  meta.resize_mode = "letterbox";
  meta.color_in = "BGR";
  meta.color_out = "RGB";
  meta.axis_perm.clear();
  meta.normalize = normalize;
  meta.quantize = quantize;
  meta.tessellate = tessellate;

  const LetterboxGeometry g =
      compute_letterbox_geometry(original_w, original_h, target_w, target_h, padding_type);
  meta.scaled_width = g.resized_w > 0 ? g.resized_w : target_w;
  meta.scaled_height = g.resized_h > 0 ? g.resized_h : target_h;
  meta.pad_left = g.pad_left;
  meta.pad_right = g.pad_right;
  meta.pad_top = g.pad_top;
  meta.pad_bottom = g.pad_bottom;

  const double inv_scale = (g.scale > 0.0) ? (1.0 / g.scale) : 1.0;
  meta.affine_m00 = inv_scale;
  meta.affine_m01 = 0.0;
  meta.affine_m02 = -static_cast<double>(meta.pad_left) * inv_scale;
  meta.affine_m10 = 0.0;
  meta.affine_m11 = inv_scale;
  meta.affine_m12 = -static_cast<double>(meta.pad_top) * inv_scale;
  meta.affine_scale_x = inv_scale;
  meta.affine_scale_y = inv_scale;
  meta.affine_offset_x = -static_cast<double>(meta.pad_left) * inv_scale;
  meta.affine_offset_y = -static_cast<double>(meta.pad_top) * inv_scale;
  return meta;
}

bool attach_preprocess_meta_to_holder_local(simaai::neat::Tensor* tensor,
                                            const simaai::neat::PreprocessRuntimeMeta& meta) {
  if (!tensor) {
    return false;
  }
  const auto holder = simaai::neat::pipeline_internal::holder_from_tensor(*tensor);
  if (!holder) {
    return false;
  }
  GstBuffer* buffer = simaai::neat::pipeline_internal::buffer_from_tensor_holder(holder);
  if (!buffer) {
    return false;
  }
  const bool ok = simaai::neat::write_simaai_preprocess_meta(buffer, meta);
  gst_buffer_unref(buffer);
  return ok;
}

void require_preprocess_meta_on_tensor_local(const simaai::neat::Tensor& tensor, int expect_orig_w,
                                             int expect_orig_h, int expect_resized_w,
                                             int expect_resized_h, const char* where) {
  auto require_expected_meta = [&](const simaai::neat::PreprocessRuntimeMeta& parsed,
                                   const char* source) {
    if (parsed.original_width != expect_orig_w || parsed.original_height != expect_orig_h ||
        parsed.resized_width != expect_resized_w || parsed.resized_height != expect_resized_h) {
      throw std::runtime_error(std::string(where) + ": preprocess meta mismatch after attach (" +
                               source + ")");
    }
  };
  if (tensor.semantic.preprocess.has_value()) {
    require_expected_meta(*tensor.semantic.preprocess, "tensor semantic");
    return;
  }
  const auto holder = simaai::neat::pipeline_internal::holder_from_tensor(tensor);
  if (!holder) {
    throw std::runtime_error(std::string(where) + ": tensor holder missing after meta attach");
  }
  GstBuffer* buffer = simaai::neat::pipeline_internal::buffer_from_tensor_holder(holder);
  if (!buffer) {
    throw std::runtime_error(std::string(where) + ": GstBuffer missing after meta attach");
  }
  const auto parsed = simaai::neat::read_simaai_preprocess_meta(buffer);
  gst_buffer_unref(buffer);
  if (!parsed.has_value()) {
    throw std::runtime_error(std::string(where) + ": preprocess meta missing after attach");
  }
  require_expected_meta(*parsed, "original/resized");
}

simaai::neat::Tensor ensure_preprocess_meta_stretch_local(
    const simaai::neat::Tensor& tensor, const simaai::neat::InputOptions& opt, int original_w,
    int original_h, int target_w, int target_h, bool normalize, bool quantize, bool tessellate) {
  if (original_w <= 0 || original_h <= 0 || target_w <= 0 || target_h <= 0) {
    return tensor;
  }
  const auto meta = build_preprocess_meta_stretch_local(original_w, original_h, target_w, target_h,
                                                        normalize, quantize, tessellate);

  simaai::neat::Tensor out = tensor;
  if (attach_preprocess_meta_to_holder_local(&out, meta)) {
    require_preprocess_meta_on_tensor_local(out, original_w, original_h, target_w, target_h,
                                            "adapter tensor meta(holder)");
    return out;
  }

  (void)opt;
  out.semantic.preprocess = meta;
  require_preprocess_meta_on_tensor_local(out, original_w, original_h, target_w, target_h,
                                          "adapter tensor meta(semantic)");
  return out;
}

simaai::neat::Tensor ensure_preprocess_meta_local(const simaai::neat::Tensor& tensor,
                                                  const simaai::neat::InputOptions& opt,
                                                  int original_w, int original_h, int target_w,
                                                  int target_h, bool normalize, bool quantize,
                                                  bool tessellate, bool letterbox_mode,
                                                  const std::string& letterbox_padding = "CENTER") {
  if (!letterbox_mode) {
    return ensure_preprocess_meta_stretch_local(tensor, opt, original_w, original_h, target_w,
                                                target_h, normalize, quantize, tessellate);
  }
  if (original_w <= 0 || original_h <= 0 || target_w <= 0 || target_h <= 0) {
    return tensor;
  }
  const auto meta =
      build_preprocess_meta_letterbox_local(original_w, original_h, target_w, target_h, normalize,
                                            quantize, tessellate, letterbox_padding);

  simaai::neat::Tensor out = tensor;
  if (attach_preprocess_meta_to_holder_local(&out, meta)) {
    require_preprocess_meta_on_tensor_local(out, original_w, original_h, target_w, target_h,
                                            "adapter tensor meta(holder)");
    return out;
  }

  (void)opt;
  out.semantic.preprocess = meta;
  require_preprocess_meta_on_tensor_local(out, original_w, original_h, target_w, target_h,
                                          "adapter tensor meta(semantic)");
  return out;
}

cv::Mat letterbox_resize_local(const cv::Mat& src, int dst_w, int dst_h, int interpolation,
                               const cv::Scalar& pad_value, const std::string& padding_type) {
  if (!src.data || dst_w <= 0 || dst_h <= 0) {
    return src;
  }
  const int src_w = src.cols;
  const int src_h = src.rows;
  if (src_w <= 0 || src_h <= 0) {
    return src;
  }
  const LetterboxGeometry g = compute_letterbox_geometry(src_w, src_h, dst_w, dst_h, padding_type);
  const int resized_w = g.resized_w;
  const int resized_h = g.resized_h;
  const int pad_left = g.pad_left;
  const int pad_top = g.pad_top;

  cv::Mat resized;
  if (resized_w != src_w || resized_h != src_h) {
    cv::resize(src, resized, cv::Size(resized_w, resized_h), 0.0, 0.0, interpolation);
  } else {
    resized = src;
  }

  const int type = src.type();
  cv::Mat out(dst_h, dst_w, type, pad_value);
  resized.copyTo(out(cv::Rect(pad_left, pad_top, resized_w, resized_h)));
  return out;
}

cv::Mat letterbox_resize_local(const cv::Mat& src, int dst_w, int dst_h) {
  const cv::Scalar pad_value =
      (src.channels() == 1) ? cv::Scalar(114) : cv::Scalar(114, 114, 114, 0);
  return letterbox_resize_local(src, dst_w, dst_h, cv::INTER_LINEAR, pad_value, "CENTER");
}

bool spec_looks_chw(const simaai::neat::TensorSpec& spec) {
  if (spec.shape.size() == 3) {
    return spec.shape[0] > 0 && spec.shape[0] <= 8 && spec.shape[1] > 0 && spec.shape[2] > 0;
  }
  if (spec.shape.size() == 4) {
    return spec.shape[1] > 0 && spec.shape[1] <= 8 && spec.shape[2] > 0 && spec.shape[3] > 0;
  }
  return false;
}

int infer_bf16_target_depth(const simaai::neat::TensorSpec& spec, bool is_chw) {
  if (is_chw) {
    if (spec.shape.size() == 3 && spec.shape[0] > 0 && spec.shape[0] <= 8) {
      return static_cast<int>(spec.shape[0]);
    }
    if (spec.shape.size() == 4 && spec.shape[1] > 0 && spec.shape[1] <= 8 && spec.shape[2] > 0 &&
        spec.shape[3] > 0) {
      return static_cast<int>(spec.shape[1]);
    }
    return 0;
  }
  if (spec.shape.size() == 3 && spec.shape[2] > 0 && spec.shape[2] <= 4) {
    return static_cast<int>(spec.shape[2]);
  }
  if (spec.shape.size() == 4 && spec.shape[3] > 0 && spec.shape[3] <= 4) {
    return static_cast<int>(spec.shape[3]);
  }
  return 0;
}

simaai::neat::Tensor make_fp32_input_tensor_cpu(const cv::Mat& img_bgr,
                                                const simaai::neat::Model& model,
                                                const simaai::neat::TensorSpec& spec,
                                                RouteKind route) {
  (void)route;
  const auto opt = model.input_appsrc_options(true);
  if (upper_copy(simaai::neat::resolve_input_media_type(opt)) != "APPLICATION/VND.SIMAAI.TENSOR") {
    throw std::runtime_error("FP32 route expected tensor media type");
  }

  // Keep variant-matrix ingress deterministic and equivalent to host YOLO preproc:
  // letterbox 640x640 with pad=114, BGR->RGB, normalize to [0,1].
  cv::Mat tensor_like =
      resize_to_target_local(img_bgr, kFixedPreprocTargetWidth, kFixedPreprocTargetHeight, true,
                             cv::INTER_LINEAR, cv::Scalar(114, 114, 114, 0), "CENTER");
  tensor_like = convert_color_for_format(tensor_like, "RGB", "matrix canonical fp32 ingress");

  if (!tensor_like.isContinuous()) {
    tensor_like = tensor_like.clone();
  }
  const int h = tensor_like.rows;
  const int w = tensor_like.cols;
  const int channels = tensor_like.channels();
  if (h <= 0 || w <= 0 || channels <= 0) {
    throw std::runtime_error("BF16 tensor input has invalid geometry");
  }

  const bool is_chw = spec_looks_chw(spec);
  int depth = infer_bf16_target_depth(spec, is_chw);
  if (depth <= 0) {
    depth = std::min(channels, 3);
  } else {
    depth = std::min(depth, channels);
  }
  depth = std::max(1, depth);
  const size_t elem_count =
      static_cast<size_t>(h) * static_cast<size_t>(w) * static_cast<size_t>(std::max(1, depth));
  auto storage = simaai::neat::make_cpu_owned_storage(elem_count * sizeof(float));
  auto map = storage->map(simaai::neat::MapMode::Write);
  if (!map.data) {
    throw std::runtime_error("FP32 tensor storage mapping failed");
  }
  if (map.size_bytes < elem_count * sizeof(float)) {
    throw std::runtime_error("FP32 tensor storage too small");
  }

  auto* out = static_cast<float*>(map.data);
  constexpr float input_scale = 1.0f / 255.0f;
  constexpr float input_bias = 0.0f;
  if (!is_chw && channels == 1) {
    for (int y = 0; y < h; ++y) {
      const uint8_t* row = tensor_like.ptr<uint8_t>(y);
      const size_t row_base = static_cast<size_t>(y) * static_cast<size_t>(w);
      for (int x = 0; x < w; ++x) {
        out[row_base + static_cast<size_t>(x)] =
            static_cast<float>(row[x]) * input_scale + input_bias;
      }
    }
  } else {
    for (int y = 0; y < h; ++y) {
      const uint8_t* row = tensor_like.ptr<uint8_t>(y);
      const size_t row_base =
          static_cast<size_t>(y) * static_cast<size_t>(w) * static_cast<size_t>(depth);
      for (int x = 0; x < w; ++x) {
        const uint8_t* pix = row + static_cast<size_t>(x) * static_cast<size_t>(channels);
        if (is_chw) {
          const size_t hw_idx =
              static_cast<size_t>(y) * static_cast<size_t>(w) + static_cast<size_t>(x);
          const size_t chw_stride = static_cast<size_t>(h) * static_cast<size_t>(w);
          for (int c = 0; c < std::min(channels, depth); ++c) {
            out[c * chw_stride + hw_idx] = static_cast<float>(pix[c]) * input_scale + input_bias;
          }
        } else {
          const size_t pixel_idx = row_base + static_cast<size_t>(x) * static_cast<size_t>(depth);
          for (int c = 0; c < std::min(channels, depth); ++c) {
            out[pixel_idx + static_cast<size_t>(c)] =
                static_cast<float>(pix[c]) * input_scale + input_bias;
          }
        }
      }
    }
  }

  simaai::neat::Tensor tensor;
  tensor.storage = storage;
  tensor.dtype = simaai::neat::TensorDType::Float32;
  tensor.layout = is_chw ? simaai::neat::TensorLayout::CHW : simaai::neat::TensorLayout::HWC;
  tensor.device = {simaai::neat::DeviceType::CPU, 0};
  tensor.read_only = true;
  if (is_chw) {
    tensor.shape = {depth, h, w};
    tensor.strides_bytes = {static_cast<int64_t>(h) * static_cast<int64_t>(w) *
                                static_cast<int64_t>(sizeof(float)),
                            static_cast<int64_t>(w) * static_cast<int64_t>(sizeof(float)),
                            static_cast<int64_t>(sizeof(float))};
  } else {
    tensor.shape = {h, w, depth};
    tensor.strides_bytes = {static_cast<int64_t>(w) * static_cast<int64_t>(depth) *
                                static_cast<int64_t>(sizeof(float)),
                            static_cast<int64_t>(depth) * static_cast<int64_t>(sizeof(float)),
                            static_cast<int64_t>(sizeof(float))};
  }
  return tensor;
}

simaai::neat::Tensor make_fp32_input_tensor_cpu(const cv::Mat& img_bgr,
                                                const simaai::neat::Model& model,
                                                const simaai::neat::TensorSpec& spec) {
  return make_fp32_input_tensor_cpu(img_bgr, model, spec, RouteKind::PreInferPost);
}

bool use_ev74_backed_e2e_input_tensor() {
  const char* raw = std::getenv("SIMA_E2E_INPUT_TENSOR_MEMORY");
  if (!raw || !*raw) {
    return true;
  }
  const std::string v = upper_copy(raw);
  return !(v == "CPU" || v == "HOST" || v == "0" || v == "FALSE" || v == "OFF");
}

simaai::neat::Tensor move_e2e_input_tensor_to_ev74_if_enabled(simaai::neat::Tensor input) {
  if (!use_ev74_backed_e2e_input_tensor()) {
    return input;
  }
  if (input.device.type == simaai::neat::DeviceType::SIMA_CVU && input.storage &&
      !input.storage->sima_segments.empty()) {
    return input;
  }
  const std::size_t bytes = input.dense_bytes_tight();
  if (bytes == 0U) {
    throw std::runtime_error("canonical_input: cannot compute dense byte size for EV74 transfer");
  }
  std::vector<simaai::neat::Segment> segments{{"ifm0", bytes}};
  return simaai::neat::pipeline_internal::transfer_to_device(
      input, {simaai::neat::DeviceType::SIMA_CVU, 0}, &segments, nullptr);
}

simaai::neat::Tensor build_canonical_preprocessed_input(const cv::Mat& img_bgr,
                                                        const simaai::neat::Model& model) {
  const auto spec = model.input_specs().front();
  const simaai::neat::InputOptions ingress_opt = model.input_appsrc_options(true);
  require(upper_copy(simaai::neat::resolve_input_media_type(ingress_opt)) ==
              "APPLICATION/VND.SIMAAI.TENSOR",
          "canonical e2e path requires tensor ingress");

  simaai::neat::Tensor input = make_fp32_input_tensor_cpu(img_bgr, model, spec);
  input = ensure_preprocess_meta_local(input, ingress_opt, img_bgr.cols, img_bgr.rows,
                                       kFixedPreprocTargetWidth, kFixedPreprocTargetHeight,
                                       /*normalize=*/true, /*quantize=*/false,
                                       /*tessellate=*/false, /*letterbox_mode=*/true, "CENTER");
  input = move_e2e_input_tensor_to_ev74_if_enabled(std::move(input));
  require_preprocess_meta_on_tensor_local(input, img_bgr.cols, img_bgr.rows,
                                          kFixedPreprocTargetWidth, kFixedPreprocTargetHeight,
                                          "canonical_input");
  return input;
}

std::string sample_output_signature_local(const simaai::neat::Sample& sample) {
  const auto tensors = tensors_in_sample(sample);
  if (tensors.empty()) {
    return sample.kind == simaai::neat::SampleKind::Bundle ? "Bundle" : "NoTensor";
  }
  const simaai::neat::Tensor& tensor = tensors.front();
  std::ostringstream os;
  os << dtype_name(tensor.dtype) << " " << shape_string(tensor.shape);
  if (tensors.size() > 1U) {
    os << " tensors=" << tensors.size();
  }
  return os.str();
}

void require_preprocess_meta_on_output_local(const simaai::neat::Sample& sample, int original_w,
                                             int original_h, const char* where) {
  const auto maybe_tensor = first_tensor_in_sample(sample);
  require(maybe_tensor.has_value(),
          std::string(where) + ": model output must contain at least one tensor");
  require_preprocess_meta_on_tensor_local(*maybe_tensor, original_w, original_h,
                                          kFixedPreprocTargetWidth, kFixedPreprocTargetHeight,
                                          where);
}

simaai::neat::Sample run_canonical_model_sample(const cv::Mat& img_bgr, simaai::neat::Model& model,
                                                const simaai::neat::Model::RouteOptions& sess_opt,
                                                int frames = 1) {
  const simaai::neat::Tensor input = build_canonical_preprocessed_input(img_bgr, model);
  const simaai::neat::TensorList input_tensorlist{input};
  const int timeout_ms = default_model_run_timeout_ms();
  const int retry_timeout_ms = std::max(timeout_ms, kRunRetryTimeoutMs);
  const int loop_count = std::max(frames, 1);

  const char* mt_env = std::getenv("SIMA_E2E_MULTITHREAD");
  const bool multithread = (mt_env != nullptr) && (mt_env[0] != '\0') && (mt_env[0] != '0');

  auto run_once_async = [&](int pull_timeout_ms) -> simaai::neat::Sample {
    simaai::neat::RunOptions async_run_opt;
    if (const char* raw_depth = std::getenv("SIMA_E2E_RUN_QUEUE_DEPTH"); raw_depth && *raw_depth) {
      const int depth = std::atoi(raw_depth);
      if (depth > 0) {
        async_run_opt.queue_depth = depth;
      }
    }
    if (std::getenv("SIMA_E2E_INPUTSTREAM_METRICS") != nullptr) {
    }
    auto runner = model.build(input_tensorlist, sess_opt, async_run_opt);
    require(static_cast<bool>(runner), "canonical e2e: runner build failed");

    // Hardcoded warmup + measurement split. The first frame pays a large
    // cold-start cost (RPC transport_init, dispatcher worker spin-up, kernel
    // first-touch faults, GstBufferPool slot allocation, etc.) that smears
    // across any short measurement window. We push kWarmupFrames frames
    // BEFORE starting the timer, then time exactly kMeasureFrames frames so
    // the reported MT_FPS reflects steady-state pipeline throughput.
    constexpr int kWarmupFrames = 200;
    constexpr int kMeasureFrames = 500;
    constexpr int kTotalFrames = kWarmupFrames + kMeasureFrames;
    (void)loop_count; // intentionally ignored on the MT_FPS path
    const int effective_loop = kTotalFrames;

    std::atomic<int> pulled{0};
    simaai::neat::Sample last_sample;
    std::mutex last_sample_mu;
    std::atomic<bool> pull_failed{false};
    std::mutex pull_error_mu;
    std::string pull_error;

    std::thread consumer([&]() {
      try {
        while (!pull_failed.load(std::memory_order_relaxed) &&
               pulled.load(std::memory_order_relaxed) < effective_loop) {
          const auto outs = runner.pull(pull_timeout_ms);
          if (outs.empty()) {
            continue;
          }
          {
            std::lock_guard<std::mutex> lk(last_sample_mu);
            last_sample = outs.front();
          }
          pulled.fetch_add(static_cast<int>(outs.size()), std::memory_order_relaxed);
        }
      } catch (const std::exception& e) {
        {
          std::lock_guard<std::mutex> lk(pull_error_mu);
          pull_error = e.what();
        }
        pull_failed.store(true, std::memory_order_relaxed);
      } catch (...) {
        {
          std::lock_guard<std::mutex> lk(pull_error_mu);
          pull_error = "unknown pull exception";
        }
        pull_failed.store(true, std::memory_order_relaxed);
      }
    });

    // Pre-allocate a small ring of input tensors so rapid back-to-back pushes
    // don't race on a single tensor's storage (gst-side copy maps the tensor
    // and concurrent maps from in-flight pushes can fail).  Size matches the
    // typical queue depth — kPoolSize entries are enough to keep one in
    // flight per stage without ever recycling a slot still being read.
    constexpr int kPoolSize = 8;
    std::vector<simaai::neat::Tensor> input_pool;
    input_pool.reserve(kPoolSize);
    for (int i = 0; i < kPoolSize; ++i) {
      input_pool.push_back(build_canonical_preprocessed_input(img_bgr, model));
    }
    // Also prebuild the TensorList wrappers outside the measured loop.  This
    // keeps the push timing focused on runtime ingress (Run queue + InputStream
    // buffer construction/copy) instead of charging a tiny per-frame C++
    // container construction cost to the measurement.
    std::vector<simaai::neat::TensorList> input_tensorlist_pool;
    input_tensorlist_pool.reserve(kPoolSize);
    for (const auto& slot : input_pool) {
      input_tensorlist_pool.emplace_back(simaai::neat::TensorList{slot});
    }

    struct PushTimingStats {
      int count = 0;
      double sum_ms = 0.0;
      double max_ms = 0.0;
    };
    const bool push_profile = std::getenv("SIMA_E2E_PUSH_PROFILE") != nullptr &&
                              std::strcmp(std::getenv("SIMA_E2E_PUSH_PROFILE"), "0") != 0;
    const int push_profile_limit = []() {
      const char* raw = std::getenv("SIMA_E2E_PUSH_PROFILE_LIMIT");
      if (!raw || !*raw) {
        return 16;
      }
      const int v = std::atoi(raw);
      return v < 0 ? 0 : v;
    }();
    PushTimingStats warmup_push_stats;
    PushTimingStats measure_push_stats;
    auto push_timed = [&](std::size_t slot_idx, int frame_idx, const char* phase,
                          PushTimingStats& stats) -> bool {
      const auto push_t0 = std::chrono::steady_clock::now();
      const bool ok = runner.push(input_tensorlist_pool[slot_idx]);
      const auto push_t1 = std::chrono::steady_clock::now();
      if (push_profile) {
        const double ms = std::chrono::duration<double, std::milli>(push_t1 - push_t0).count();
        stats.count += 1;
        stats.sum_ms += ms;
        stats.max_ms = std::max(stats.max_ms, ms);
        if (frame_idx < push_profile_limit ||
            (push_profile_limit > 0 && (frame_idx % push_profile_limit) == 0)) {
          std::cerr << "E2E_PUSH_PROFILE phase=" << phase << " frame=" << frame_idx
                    << " ok=" << (ok ? 1 : 0) << " ms=" << ms << "\n";
        }
      }
      return ok;
    };

    int pushed = 0;
    // Phase 1: warmup. Push kWarmupFrames and wait until the consumer has
    // pulled all of them. No timing is captured here.
    for (int frame_idx = 0; frame_idx < kWarmupFrames; ++frame_idx) {
      const std::size_t slot_idx = static_cast<std::size_t>(frame_idx % kPoolSize);
      if (!push_timed(slot_idx, frame_idx, "warmup", warmup_push_stats)) {
        pull_failed.store(true);
        break;
      }
      ++pushed;
    }
    while (!pull_failed.load(std::memory_order_relaxed) &&
           pulled.load(std::memory_order_relaxed) < kWarmupFrames) {
      std::this_thread::yield();
    }

    // Phase 2: measured. Capture t0 only after warmup has fully drained, so
    // the timer covers exactly kMeasureFrames in steady state.
    simaai::neat::MeasureOptions measure_opt;
    measure_opt.include_plugin_latency = false;
    measure_opt.include_edge_latency = false;
    measure_opt.include_power = false;
    auto measure_scope =
        simaai::neat::internal::ModelAccess::run(runner).start_measurement(measure_opt);
    const auto loop_t0 = std::chrono::steady_clock::now();
    for (int frame_idx = kWarmupFrames; frame_idx < kTotalFrames; ++frame_idx) {
      const std::size_t slot_idx = static_cast<std::size_t>(frame_idx % kPoolSize);
      if (!push_timed(slot_idx, frame_idx, "measure", measure_push_stats)) {
        pull_failed.store(true);
        break;
      }
      ++pushed;
    }
    consumer.join();
    const auto loop_t1 = std::chrono::steady_clock::now();
    const simaai::neat::MeasureReport measure_report = measure_scope.stop();

    const double mt_ms = std::chrono::duration<double, std::milli>(loop_t1 - loop_t0).count();
    const int measured_pulled = std::max(0, pulled.load() - kWarmupFrames);
    const double mt_fps =
        mt_ms > 0.0 ? (static_cast<double>(measured_pulled) * 1000.0 / mt_ms) : 0.0;
    if (push_profile) {
      const auto print_push_stats = [](const char* phase, const PushTimingStats& stats) {
        const double avg = stats.count > 0 ? stats.sum_ms / static_cast<double>(stats.count) : 0.0;
        std::cerr << "E2E_PUSH_PROFILE_SUMMARY phase=" << phase << " count=" << stats.count
                  << " avg_ms=" << avg << " max_ms=" << stats.max_ms << "\n";
      };
      print_push_stats("warmup", warmup_push_stats);
      print_push_stats("measure", measure_push_stats);
    }
    std::cout << "MT_FPS frames=" << kMeasureFrames << " warmup=" << kWarmupFrames
              << " pushed=" << pushed << " pulled=" << pulled.load()
              << " measured_pulled=" << measured_pulled << " push_pull_ms=" << mt_ms
              << " fps=" << mt_fps
              << " stats_inputs_pushed=" << measure_report.counters.inputs_pushed
              << " stats_outputs_pulled=" << measure_report.counters.outputs_pulled
              << " avg_latency_ms=" << measure_report.end_to_end.avg_ms
              << " p50_latency_ms=" << measure_report.end_to_end.p50_ms
              << " p95_latency_ms=" << measure_report.end_to_end.p95_ms
              << " inputs_dropped=" << measure_report.counters.inputs_dropped
              << " outputs_dropped=" << measure_report.counters.outputs_dropped << "\n";

    if (std::getenv("SIMA_E2E_REPORT") != nullptr) {
      std::cout << "REPORT_BEGIN\n" << measure_report.to_text() << "\nREPORT_END\n";
    }

    std::string async_pull_error;
    {
      std::lock_guard<std::mutex> lk(pull_error_mu);
      async_pull_error = pull_error;
    }
    require(!pull_failed.load(),
            "canonical e2e: push/pull failed" +
                (async_pull_error.empty() ? std::string{} : (": " + async_pull_error)));
    require(pulled.load() == effective_loop, "canonical e2e: expected " +
                                                 std::to_string(effective_loop) + " outputs, got " +
                                                 std::to_string(pulled.load()));
    return last_sample;
  };

  auto run_once_sync = [&](int pull_timeout_ms) -> simaai::neat::Sample {
    auto runner = model.build(simaai::neat::TensorList{input}, sess_opt);
    require(static_cast<bool>(runner), "canonical e2e: runner build failed");
    simaai::neat::Sample last_sample;
    for (int frame_idx = 0; frame_idx < loop_count; ++frame_idx) {
      require(runner.push(simaai::neat::TensorList{input}),
              "canonical e2e: runner push failed at frame " + std::to_string(frame_idx));
      const auto outputs = runner.pull(pull_timeout_ms);
      require(outputs.size() == 1U, "canonical e2e: expected exactly 1 output sample at frame " +
                                        std::to_string(frame_idx) + ", got " +
                                        std::to_string(outputs.size()));
      last_sample = outputs.front();
    }
    return last_sample;
  };

  try {
    return multithread ? run_once_async(timeout_ms) : run_once_sync(timeout_ms);
  } catch (const std::exception&) {
    return multithread ? run_once_async(retry_timeout_ms) : run_once_sync(retry_timeout_ms);
  }
}

simaai::neat::Tensor fp32_to_bf16_cast_node(const simaai::neat::Tensor& input) {
  if (input.dtype != simaai::neat::TensorDType::Float32) {
    throw std::runtime_error("FP32->BF16 cast node expected Float32 tensor input");
  }
  if (input.is_composite()) {
    throw std::runtime_error("FP32->BF16 cast node requires dense tensor input");
  }
  const simaai::neat::Tensor cpu = input.cpu().contiguous();
  const std::vector<uint8_t> bytes = cpu.copy_dense_bytes_tight();
  if (bytes.empty() || (bytes.size() % sizeof(float)) != 0U) {
    throw std::runtime_error("FP32->BF16 cast node invalid input payload");
  }

  const std::size_t elem_count = bytes.size() / sizeof(float);
  auto storage = simaai::neat::make_cpu_owned_storage(elem_count * sizeof(uint16_t));
  auto map = storage->map(simaai::neat::MapMode::Write);
  if (!map.data || map.size_bytes < elem_count * sizeof(uint16_t)) {
    throw std::runtime_error("FP32->BF16 cast node output allocation failed");
  }

  const auto* in_ptr = reinterpret_cast<const float*>(bytes.data());
  auto* out_ptr = static_cast<uint16_t*>(map.data);
  for (std::size_t i = 0; i < elem_count; ++i) {
    out_ptr[i] = fp32_to_bf16(in_ptr[i]);
  }

  simaai::neat::Tensor out = cpu;
  out.storage = std::move(storage);
  out.dtype = simaai::neat::TensorDType::BFloat16;
  out.device = {simaai::neat::DeviceType::CPU, 0};
  out.byte_offset = 0;
  out.read_only = true;
  if (!out.shape.empty()) {
    std::vector<int64_t> strides(out.shape.size(), 0);
    int64_t stride = static_cast<int64_t>(sizeof(uint16_t));
    for (int i = static_cast<int>(out.shape.size()) - 1; i >= 0; --i) {
      strides[static_cast<std::size_t>(i)] = stride;
      stride *= out.shape[static_cast<std::size_t>(i)];
    }
    out.strides_bytes = std::move(strides);
  }
  out.semantic.tess = simaai::neat::TessSpec{};
  out.semantic.tess->format = "BF16";
  return out;
}

ProbeResult build_hostdecode_probe_from_model_local(const simaai::neat::Model& model) {
  ProbeResult probe;
  const std::string mla_cfg = [&]() -> std::string {
    std::string cfg = model.find_config_path_by_processor("MLA");
    if (cfg.empty()) {
      cfg = model.find_config_path_by_plugin("processmla");
    }
    return cfg;
  }();
  if (!mla_cfg.empty()) {
    probe.etc_dir = fs::path(mla_cfg).parent_path();
  }
  return probe;
}

simaai::neat::Tensor make_bf16_input_tensor_cpu(const cv::Mat& img_bgr,
                                                const simaai::neat::Model& model,
                                                const simaai::neat::TensorSpec& spec,
                                                RouteKind route) {
  const simaai::neat::Tensor fp32 = make_fp32_input_tensor_cpu(img_bgr, model, spec, route);
  return fp32_to_bf16_cast_node(fp32);
}

simaai::neat::Tensor make_bf16_input_tensor(const cv::Mat& img_bgr,
                                            const simaai::neat::Model& model,
                                            const simaai::neat::TensorSpec& spec, RouteKind route) {
  return make_bf16_input_tensor_cpu(img_bgr, model, spec, route);
}

struct AdapterIngressTensorInput {
  simaai::neat::Tensor tensor;
  int target_w = 0;
  int target_h = 0;
  bool quantize_needed = false;
  bool tessellate_needed = false;
  bool normalize_enabled = true;
  bool use_letterbox = false;
  bool tess_only_bf16 = false;
  std::string preproc_out_fmt = "BGR";
  std::string letterbox_padding = "CENTER";
  int interpolation = cv::INTER_LINEAR;
};

AdapterIngressTensorInput build_adapter_tensor_ingress_input(const cv::Mat& input,
                                                             const simaai::neat::Model& model,
                                                             const simaai::neat::TensorSpec& spec,
                                                             RouteKind route,
                                                             PreAdapterKind pre_adapter_kind) {
  AdapterIngressTensorInput out;
  const bool strict_adapter_profile = pre_adapter_kind == PreAdapterKind::Quant ||
                                      pre_adapter_kind == PreAdapterKind::Tess ||
                                      pre_adapter_kind == PreAdapterKind::QuantTess;
  const auto mla_info = rendered_stage_query::mla_input_info_from_nodes(
      simaai::neat::internal::ModelAccess::build_public_inference_nodes(model));
  const auto preproc_contract = read_preproc_contract_from_model(model);
  out.quantize_needed = dtype_is_int8_like(mla_info.input_dtype);
  out.tessellate_needed =
      (pre_adapter_kind == PreAdapterKind::Tess || pre_adapter_kind == PreAdapterKind::QuantTess);
  auto [target_w, target_h] = preferred_input_hw(model, route, spec);
  if (preproc_contract.has_value()) {
    if (preproc_contract->output_width > 0) {
      target_w = preproc_contract->output_width;
    }
    if (preproc_contract->output_height > 0) {
      target_h = preproc_contract->output_height;
    }
  }
  if (strict_adapter_profile) {
    if (preproc_contract.has_value()) {
      if ((preproc_contract->output_width > 0 &&
           preproc_contract->output_width != kFixedPreprocTargetWidth) ||
          (preproc_contract->output_height > 0 &&
           preproc_contract->output_height != kFixedPreprocTargetHeight)) {
        throw std::runtime_error(
            "adapter_ingress strict profile requires 640x640 preproc contract");
      }
    }
    target_w = kFixedPreprocTargetWidth;
    target_h = kFixedPreprocTargetHeight;
  } else if (target_w <= 0 || target_h <= 0) {
    target_w = kFixedPreprocTargetWidth;
    target_h = kFixedPreprocTargetHeight;
  }
  out.target_w = target_w;
  out.target_h = target_h;
  out.use_letterbox = strict_adapter_profile
                          ? true
                          : ((preproc_contract.has_value() && preproc_contract->has_aspect_ratio)
                                 ? preproc_contract->aspect_ratio
                                 : route_uses_letterbox(route));
  const bool requested_letterbox = out.use_letterbox;
  if (!strict_adapter_profile && out.use_letterbox &&
      pre_adapter_kind == PreAdapterKind::QuantTess &&
      !kernel_preproc_can_letterbox_dims(input.cols, input.rows, out.target_w, out.target_h)) {
    out.use_letterbox = false;
  }
  out.letterbox_padding =
      strict_adapter_profile
          ? "CENTER"
          : (preproc_contract.has_value() ? preproc_contract->padding_type : std::string("CENTER"));
  out.interpolation =
      strict_adapter_profile
          ? cv::INTER_LINEAR
          : (preproc_contract.has_value()
                 ? cv_interpolation_from_preproc_scaling_type(preproc_contract->scaling_type)
                 : cv::INTER_LINEAR);
  if (!strict_adapter_profile) {
    out.interpolation = cv_interpolation_from_env_override(
        std::getenv("SIMA_YOLOV8_MANUAL_PREPROC_INTERP"), out.interpolation);
  }
  if (strict_adapter_profile) {
    if (preproc_contract.has_value()) {
      out.preproc_out_fmt = canonical_color_format(preproc_contract->output_img_type);
    } else {
      out.preproc_out_fmt = canonical_color_format(bf16_input_format_for_model(model, input));
    }
    out.normalize_enabled = true;
  } else {
    const char* preproc_out_fmt_override = std::getenv("SIMA_YOLOV8_MANUAL_PREPROC_OUT_FMT");
    out.preproc_out_fmt = canonical_color_format(
        (preproc_out_fmt_override && *preproc_out_fmt_override)
            ? std::string(preproc_out_fmt_override)
            : (preproc_contract.has_value() ? preproc_contract->output_img_type
                                            : bf16_input_format_for_model(model, input)));
    out.normalize_enabled = (preproc_contract.has_value() && preproc_contract->has_normalize)
                                ? preproc_contract->normalize
                                : true;
  }
  out.tess_only_bf16 = pre_adapter_kind == PreAdapterKind::Tess && !out.quantize_needed;
  if (matrix_input_debug_enabled()) {
    const std::string route_str = route_name(route);
    std::fprintf(stderr,
                 "[matrix-input-debug] adapter_ingress in=%dx%d route=%s pre_kind=%s "
                 "target=%dx%d tess_only_bf16=%d contract=%d out_fmt=%s normalize=%d "
                 "aspect_ratio=%d scaling=%d padding=%s\n",
                 input.cols, input.rows, route_str.c_str(), pre_kind_name(pre_adapter_kind).c_str(),
                 out.target_w, out.target_h, out.tess_only_bf16 ? 1 : 0,
                 preproc_contract.has_value() ? 1 : 0, out.preproc_out_fmt.c_str(),
                 out.normalize_enabled ? 1 : 0, out.use_letterbox ? 1 : 0, out.interpolation,
                 out.letterbox_padding.c_str());
    if (requested_letterbox && !out.use_letterbox) {
      std::fprintf(stderr,
                   "[matrix-input-debug] adapter_ingress forcing_stretch_for_kernel_parity "
                   "src=%dx%d dst=%dx%d pre_kind=%s\n",
                   input.cols, input.rows, out.target_w, out.target_h,
                   pre_kind_name(pre_adapter_kind).c_str());
    }
  }
  if (out.tess_only_bf16) {
    const simaai::neat::Tensor bf16_input = make_bf16_input_tensor_cpu(input, model, spec, route);
    int tensor_h = out.target_h;
    int tensor_w = out.target_w;
    int tensor_d = 3;
    if (bf16_input.shape.size() == 3U) {
      if (bf16_input.layout == simaai::neat::TensorLayout::CHW) {
        tensor_d = static_cast<int>(bf16_input.shape[0]);
        tensor_h = static_cast<int>(bf16_input.shape[1]);
        tensor_w = static_cast<int>(bf16_input.shape[2]);
      } else {
        tensor_h = static_cast<int>(bf16_input.shape[0]);
        tensor_w = static_cast<int>(bf16_input.shape[1]);
        tensor_d = static_cast<int>(bf16_input.shape[2]);
      }
    }
    simaai::neat::InputOptions tensor_opt;
    tensor_opt.payload_type = simaai::neat::PayloadType::Tensor;
    tensor_opt.format = "BF16";
    tensor_opt.width = tensor_w;
    tensor_opt.height = tensor_h;
    tensor_opt.depth = tensor_d;
    out.tensor = bf16_input;
    return out;
  }

  int pad_scalar_value = kPadValue;
  if (!strict_adapter_profile) {
    pad_scalar_value = 114;
    if (const char* pad_override = std::getenv("SIMA_YOLOV8_MANUAL_PREPROC_PAD_VALUE");
        pad_override && *pad_override) {
      pad_scalar_value = std::clamp(std::atoi(pad_override), 0, 255);
    }
  }
  cv::Mat prepped = input;
  const cv::Scalar pad_value =
      (prepped.channels() == 1)
          ? cv::Scalar(pad_scalar_value)
          : cv::Scalar(pad_scalar_value, pad_scalar_value, pad_scalar_value, 0);
  prepped = resize_to_target_local(prepped, out.target_w, out.target_h, out.use_letterbox,
                                   out.interpolation, pad_value, out.letterbox_padding);
  prepped = convert_color_for_format(prepped, out.preproc_out_fmt, "quanttess ingress prep");
  cv::Mat fp32;
  const int fp32_type = CV_MAKETYPE(CV_32F, prepped.channels());
  if (preproc_contract.has_value()) {
    if (out.normalize_enabled) {
      prepped.convertTo(fp32, fp32_type, 1.0f / 255.0f, 0.0f);
      apply_channel_normalization(fp32, preproc_contract->channel_mean,
                                  preproc_contract->channel_stddev);
    } else {
      prepped.convertTo(fp32, fp32_type, 1.0f, 0.0f);
    }
  } else {
    const float input_scale = read_env_float_or_default("SIMA_YOLOV8_BF16_INPUT_SCALE",
                                                        infer_fp32_input_scale_from_mpk(model));
    const float input_bias = read_env_float_or_default("SIMA_YOLOV8_BF16_INPUT_BIAS", 0.0f);
    prepped.convertTo(fp32, fp32_type, input_scale, input_bias);
  }
  simaai::neat::InputOptions tensor_opt;
  tensor_opt.payload_type = simaai::neat::PayloadType::Tensor;
  tensor_opt.format = "EVXX_FLOAT32";
  tensor_opt.width = fp32.cols;
  tensor_opt.height = fp32.rows;
  tensor_opt.depth = fp32.channels();
  const bool owned_fp32_ingress = env_flag_enabled("SIMA_YOLOV8_MANUAL_PREPROC_OWNED_FP32");
  const simaai::neat::Tensor fp32_input =
      owned_fp32_ingress
          ? make_fp32_hwc_tensor_owned(fp32, "yolov8_matrix_quanttess_owned")
          : simaai::neat::tensor_from_cv_mat(fp32, tensor_opt, "yolov8_matrix_quanttess");
  out.tensor = fp32_input;
  return out;
}

bool try_bf16_to_fp32_preview(const simaai::neat::Tensor& tensor, std::vector<float>& preview,
                              std::string& note) {
  if (tensor.dtype == simaai::neat::TensorDType::Float32) {
    const auto cpu = tensor.cpu().contiguous();
    const std::vector<uint8_t> bytes = cpu.copy_payload_bytes();
    if (bytes.empty()) {
      note = "FP32 tensor payload is empty";
      return false;
    }
    if ((bytes.size() % sizeof(float)) != 0U) {
      note = "FP32 tensor payload size is invalid";
      return false;
    }
    const float* ptr = reinterpret_cast<const float*>(bytes.data());
    const size_t elem_count = bytes.size() / sizeof(float);
    const size_t max_elems = std::min<size_t>(elem_count, 32U);
    preview.clear();
    preview.reserve(max_elems);
    for (size_t i = 0; i < max_elems; ++i) {
      preview.push_back(ptr[i]);
    }
    note = "FP32 output converted to FP32 preview";
    return true;
  }

  if (tensor.dtype != simaai::neat::TensorDType::BFloat16) {
    note = "tensor is not BF16";
    return false;
  }
  const auto cpu = tensor.cpu().contiguous();
  const std::vector<uint8_t> bytes = cpu.copy_payload_bytes();
  if (bytes.empty()) {
    note = "BF16 tensor payload is empty";
    return false;
  }
  if ((bytes.size() & 1U) != 0U) {
    note = "BF16 tensor payload size is invalid";
    return false;
  }
  const uint16_t* ptr = reinterpret_cast<const uint16_t*>(bytes.data());
  const size_t elem_count = bytes.size() / sizeof(uint16_t);
  const size_t max_elems = std::min<size_t>(elem_count, 32U);
  preview.clear();
  preview.reserve(max_elems);
  for (size_t i = 0; i < max_elems; ++i) {
    preview.push_back(bf16_to_fp32(ptr[i]));
  }
  note = "BF16 output converted to FP32 preview";
  return true;
}

cv::Mat normalize_model_input(const cv::Mat& input, const simaai::neat::Model& model,
                              RouteKind route) {
  if (route != RouteKind::QuantizedNoPostFallback || !input.data) {
    return input;
  }

  auto opt = model.input_appsrc_options(false);
  if (upper_copy(simaai::neat::resolve_input_media_type(opt)) != "VIDEO/X-RAW") {
    return input;
  }

  cv::Mat out = input;
  const int target_w = opt.width > 0 ? opt.width : opt.max_width;
  const int target_h = opt.height > 0 ? opt.height : opt.max_height;
  if (target_w > 0 && target_h > 0 && (out.cols != target_w || out.rows != target_h)) {
    cv::resize(out, out, cv::Size(target_w, target_h));
  }

  const std::string fmt = upper_copy(opt.format.str());
  if (fmt == "RGB") {
    if (out.channels() == 1) {
      cv::cvtColor(out, out, cv::COLOR_GRAY2RGB);
    } else if (out.channels() == 3) {
      cv::cvtColor(out, out, cv::COLOR_BGR2RGB);
    } else if (out.channels() == 4) {
      cv::cvtColor(out, out, cv::COLOR_BGRA2RGB);
    }
  } else if (fmt == "BGR") {
    if (out.channels() == 1) {
      cv::cvtColor(out, out, cv::COLOR_GRAY2BGR);
    } else if (out.channels() == 4) {
      cv::cvtColor(out, out, cv::COLOR_BGRA2BGR);
    }
  } else if (fmt == "GRAY8" || fmt == "GRAY") {
    if (out.channels() == 3) {
      cv::cvtColor(out, out, cv::COLOR_BGR2GRAY);
    } else if (out.channels() == 4) {
      cv::cvtColor(out, out, cv::COLOR_BGRA2GRAY);
    }
  }

  return out;
}

bool tensor_is_int8_like(const simaai::neat::Tensor& tensor) {
  const std::string fmt = tensor_format(tensor);
  return tensor.dtype == simaai::neat::TensorDType::Int8 ||
         tensor.dtype == simaai::neat::TensorDType::UInt8 ||
         simaai::neat::is_tessellated_int8_format(fmt);
}

bool tensor_is_bf16_like(const simaai::neat::Tensor& tensor) {
  const std::string fmt = tensor_format(tensor);
  return tensor.dtype == simaai::neat::TensorDType::BFloat16 ||
         simaai::neat::is_tessellated_bf16_format(fmt);
}

PreAdapterKind detect_pre_kind(const std::vector<std::shared_ptr<simaai::neat::Node>>& nodes) {
  if (nodes.empty())
    return PreAdapterKind::None;
  const std::string kind = nodes.front() ? nodes.front()->kind() : "";
  if (kind == "Preproc")
    return PreAdapterKind::Preproc;
  if (kind == "Quant")
    return PreAdapterKind::Quant;
  if (kind == "Tess")
    return PreAdapterKind::Tess;
  if (kind == "QuantTess")
    return PreAdapterKind::QuantTess;
  return PreAdapterKind::Unknown;
}

PostAdapterKind detect_post_kind(const std::vector<std::shared_ptr<simaai::neat::Node>>& nodes) {
  if (nodes.empty())
    return PostAdapterKind::None;
  bool saw_detessdequant = false;
  bool saw_dequant = false;
  for (const auto& node : nodes) {
    const std::string kind = node ? node->kind() : "";
    if (kind == "SimaBoxDecode")
      return PostAdapterKind::BoxDecode;
    if (kind == "DetessDequant")
      saw_detessdequant = true;
    if (kind == "Dequant")
      saw_dequant = true;
  }
  if (saw_detessdequant)
    return PostAdapterKind::DetessDequant;
  if (saw_dequant)
    return PostAdapterKind::Dequant;
  return PostAdapterKind::Unknown;
}

void collect_tensors_in_sample(const simaai::neat::Sample& sample,
                               std::vector<simaai::neat::Tensor>& out) {
  if (sample.kind == simaai::neat::SampleKind::Tensor && sample.tensor.has_value()) {
    out.push_back(*sample.tensor);
    return;
  }
  if (simaai::neat::sample_has_tensor_list(sample)) {
    out.insert(out.end(), sample.tensors.begin(), sample.tensors.end());
    return;
  }
  if (!simaai::neat::sample_is_multi_output(sample)) {
    return;
  }
  for (const auto& field : sample.fields) {
    collect_tensors_in_sample(field, out);
  }
}

std::vector<simaai::neat::Tensor> tensors_in_sample(const simaai::neat::Sample& sample) {
  std::vector<simaai::neat::Tensor> out;
  collect_tensors_in_sample(sample, out);
  return out;
}

std::optional<simaai::neat::Tensor> first_tensor_in_sample(const simaai::neat::Sample& sample) {
  const auto tensors = tensors_in_sample(sample);
  if (tensors.empty()) {
    return std::nullopt;
  }
  return tensors.front();
}

std::string join_strings(const std::vector<std::string>& items, const char* sep = ",") {
  std::ostringstream os;
  for (size_t i = 0; i < items.size(); ++i) {
    if (i)
      os << sep;
    os << items[i];
  }
  return os.str();
}

std::vector<std::string> node_kinds(const std::vector<std::shared_ptr<simaai::neat::Node>>& nodes) {
  std::vector<std::string> kinds;
  kinds.reserve(nodes.size());
  for (const auto& node : nodes) {
    if (!node)
      continue;
    kinds.push_back(node->kind());
  }
  return kinds;
}

bool nodes_contain_kind(const std::vector<std::shared_ptr<simaai::neat::Node>>& nodes,
                        const char* needle) {
  for (const auto& node : nodes) {
    if (node && node->kind() == needle) {
      return true;
    }
  }
  return false;
}

TerminalOutputKind
detect_terminal_output_kind(const std::vector<std::shared_ptr<simaai::neat::Node>>& post_nodes,
                            const ProbeResult& probe) {
  if (nodes_contain_kind(post_nodes, "SimaBoxDecode")) {
    return TerminalOutputKind::BoxDecodePayload;
  }
  if (nodes_contain_kind(post_nodes, "Cast") || nodes_contain_kind(post_nodes, "Dequant") ||
      nodes_contain_kind(post_nodes, "DetessDequant") || nodes_contain_kind(post_nodes, "Detess")) {
    return TerminalOutputKind::TensorFloatLike;
  }
  if (probe.mla_output_int8) {
    return TerminalOutputKind::TensorQuantized;
  }
  if (probe.mla_output_bf16) {
    return TerminalOutputKind::TensorFloatLike;
  }
  return TerminalOutputKind::Unknown;
}

simaai::neat::Model::Options default_model_options() {
  simaai::neat::Model::Options opt;
  opt.preprocess.kind = simaai::neat::InputKind::Auto;
  opt.preprocess.enable = simaai::neat::AutoFlag::Auto;
  opt.preprocess.color_convert.input_format = simaai::neat::PreprocessColorFormat::Auto;
  opt.upstream_name = "decoder";
  return opt;
}

simaai::neat::Model::Options canonical_model_options(BoxDecodeRunMode boxdecode_mode) {
  simaai::neat::Model::Options opt;
  opt.preprocess.kind = simaai::neat::InputKind::Tensor;
  opt.preprocess.enable = simaai::neat::AutoFlag::Auto;
  opt.preprocess.resize.enable = simaai::neat::AutoFlag::Off;
  opt.preprocess.color_convert.enable = simaai::neat::AutoFlag::Off;
  opt.preprocess.layout_convert.enable = simaai::neat::AutoFlag::Off;
  opt.preprocess.normalize.enable = simaai::neat::AutoFlag::Off;
  opt.preprocess.quantize.enable = simaai::neat::AutoFlag::Auto;
  opt.preprocess.tessellate.enable = simaai::neat::AutoFlag::Auto;
  if (boxdecode_mode == BoxDecodeRunMode::Model) {
    // Force the model route itself to terminate in the NEAT boxdecode plugin.
    opt.decode_type = simaai::neat::BoxDecodeType::YoloV8;
    opt.score_threshold = 0.52f;
    opt.nms_iou_threshold = 0.5f;
    opt.top_k = 100;
  }
  opt.upstream_name = "decoder";
  return opt;
}

simaai::neat::stages::BoxDecodeOptions canonical_boxdecode_options() {
  simaai::neat::stages::BoxDecodeOptions opt(simaai::neat::BoxDecodeType::YoloV8);
  opt.detection_threshold = 0.52;
  opt.nms_iou_threshold = 0.5;
  opt.top_k = 100;
  return opt;
}

simaai::neat::Model build_model_for_case(const ProbeResult& probe) {
  auto opt = default_model_options();
  // Model-managed boxdecode routes may return decoded payloads directly when the
  // strict MPK boxdecode contract is available. Fallback cases still decode through
  // the standalone post graph below.
  return simaai::neat::Model(probe.tar_path, opt);
}

simaai::neat::Sample run_model_sample(const cv::Mat& img_bgr, simaai::neat::Model& model,
                                      RouteKind route) {
  const int kRunTimeoutMs = default_model_run_timeout_ms();
  const int kRetryTimeoutMs = std::max(kRunTimeoutMs, kRunRetryTimeoutMs);
  const auto spec = model.input_specs().front();
  const auto ingress_opt = model.input_appsrc_options(false);
  const bool tensor_ingress = upper_copy(simaai::neat::resolve_input_media_type(ingress_opt)) ==
                              "APPLICATION/VND.SIMAAI.TENSOR";

  auto require_single_sample = [&](const simaai::neat::Sample& outputs,
                                   const char* where) -> simaai::neat::Sample {
    require(outputs.size() == 1U, std::string(where) + ": expected exactly 1 sample, got " +
                                      std::to_string(outputs.size()));
    return outputs.front();
  };
  auto run_with_fresh_retry_mat = [&](const cv::Mat& input,
                                      int timeout_ms) -> simaai::neat::Sample {
    auto run_once = [&](int rt) -> simaai::neat::Sample {
      auto runner = model.build(std::vector<cv::Mat>{input});
      require(runner.push(std::vector<cv::Mat>{input}), "run_model_sample(mat): push failed");
      return require_single_sample(runner.pull(rt), "run_model_sample(mat)");
    };
    try {
      return run_once(timeout_ms);
    } catch (const std::exception&) {
      return run_once(kRetryTimeoutMs);
    }
  };
  auto run_with_fresh_retry_tensor = [&](const simaai::neat::Tensor& input,
                                         int timeout_ms) -> simaai::neat::Sample {
    auto run_once = [&](int rt) -> simaai::neat::Sample {
      auto runner = model.build(simaai::neat::TensorList{input});
      require(runner.push(simaai::neat::TensorList{input}),
              "run_model_sample(tensor): push failed");
      return require_single_sample(runner.pull(rt), "run_model_sample(tensor)");
    };
    try {
      return run_once(timeout_ms);
    } catch (const std::exception&) {
      return run_once(kRetryTimeoutMs);
    }
  };

  if (tensor_ingress) {
    const simaai::neat::Tensor fp32_input = make_fp32_input_tensor_cpu(img_bgr, model, spec, route);
    if (matrix_input_debug_enabled()) {
      const std::string route_str = route_name(route);
      std::fprintf(stderr,
                   "[matrix-input-debug] canonical_fp32_tensor_ingress model_input=%dx%d route=%s "
                   "shape=%s dtype=%s\n",
                   kFixedPreprocTargetWidth, kFixedPreprocTargetHeight, route_str.c_str(),
                   shape_string(fp32_input.shape).c_str(), dtype_name(fp32_input.dtype).c_str());
    }
    try {
      return run_with_fresh_retry_tensor(fp32_input, kRunTimeoutMs);
    } catch (const std::exception&) {
      return run_with_fresh_retry_tensor(fp32_input, kRetryTimeoutMs);
    }
  }

  // Legacy image ingress fallback only when model contract does not expose tensor ingress.
  const cv::Mat normalized = normalize_model_input(img_bgr, model, route);
  try {
    return run_with_fresh_retry_mat(normalized, kRunTimeoutMs);
  } catch (const std::exception&) {
    return run_with_fresh_retry_mat(normalized, kRetryTimeoutMs);
  }
}

std::vector<objdet::ExpectedBox> expected_people_boxes_local() {
  return {
      {747.0f, 42.0f, 1131.0f, 711.0f, 0},
      {149.0f, 201.0f, 1092.0f, 710.0f, 0},
      {437.0f, 434.0f, 532.0f, 717.0f, 27},
  };
}

float box_iou_xyxy_local(float ax1, float ay1, float ax2, float ay2, float bx1, float by1,
                         float bx2, float by2) {
  const float inter_w = std::max(0.0f, std::min(ax2, bx2) - std::max(ax1, bx1));
  const float inter_h = std::max(0.0f, std::min(ay2, by2) - std::max(ay1, by1));
  const float inter = inter_w * inter_h;
  const float area_a = std::max(0.0f, ax2 - ax1) * std::max(0.0f, ay2 - ay1);
  const float area_b = std::max(0.0f, bx2 - bx1) * std::max(0.0f, by2 - by1);
  const float denom = area_a + area_b - inter;
  return (denom > 0.0f) ? (inter / denom) : 0.0f;
}

std::vector<objdet::Box> parse_boxes_strict_local(const std::vector<uint8_t>& bytes, int img_w,
                                                  int img_h, int expected_topk) {
  struct RawBoxLocal {
    int32_t x = 0;
    int32_t y = 0;
    int32_t w = 0;
    int32_t h = 0;
    float score = 0.0f;
    int32_t cls = 0;
  };

  require(bytes.size() >= 4, "bbox buffer too small");
  const size_t payload = bytes.size() - 4;
  require(payload >= sizeof(RawBoxLocal), "bbox buffer payload too small");

  uint32_t header = 0;
  std::memcpy(&header, bytes.data(), sizeof(header));
  const size_t max_boxes = payload / sizeof(RawBoxLocal);
  require(header <= max_boxes, "bbox header exceeds payload count");
  if (expected_topk > 0) {
    require(static_cast<size_t>(header) <= static_cast<size_t>(expected_topk),
            "bbox header exceeds expected topk");
  }

  std::vector<objdet::Box> out;
  out.reserve(header);
  const uint8_t* base = bytes.data() + 4;
  for (size_t i = 0; i < static_cast<size_t>(header); ++i) {
    RawBoxLocal r{};
    std::memcpy(&r, base + i * sizeof(RawBoxLocal), sizeof(r));
    float x1 = static_cast<float>(r.x);
    float y1 = static_cast<float>(r.y);
    float x2 = static_cast<float>(r.x + r.w);
    float y2 = static_cast<float>(r.y + r.h);
    x1 = std::max(0.0f, std::min(x1, static_cast<float>(img_w)));
    y1 = std::max(0.0f, std::min(y1, static_cast<float>(img_h)));
    x2 = std::max(0.0f, std::min(x2, static_cast<float>(img_w)));
    y2 = std::max(0.0f, std::min(y2, static_cast<float>(img_h)));
    out.push_back(objdet::Box{x1, y1, x2, y2, r.score, static_cast<int>(r.cls)});
  }
  return out;
}

objdet::MatchResult match_expected_boxes_local(const std::vector<objdet::Box>& boxes,
                                               const std::vector<objdet::ExpectedBox>& expected,
                                               float min_score, float min_iou) {
  objdet::MatchResult res;
  if (expected.empty()) {
    res.ok = true;
    return res;
  }

  std::vector<objdet::Box> candidates;
  for (const auto& b : boxes) {
    if (b.score >= min_score) {
      candidates.push_back(b);
    }
  }

  std::vector<bool> used(candidates.size(), false);
  std::ostringstream note;
  bool ok = true;
  for (size_t i = 0; i < expected.size(); ++i) {
    const auto& exp = expected[i];
    float best_iou = 0.0f;
    int best_idx = -1;
    for (size_t j = 0; j < candidates.size(); ++j) {
      if (used[j] || candidates[j].class_id != exp.class_id) {
        continue;
      }
      const float iou = box_iou_xyxy_local(exp.x1, exp.y1, exp.x2, exp.y2, candidates[j].x1,
                                           candidates[j].y1, candidates[j].x2, candidates[j].y2);
      if (iou > best_iou) {
        best_iou = iou;
        best_idx = static_cast<int>(j);
      }
    }
    if (best_idx < 0 || best_iou < min_iou) {
      if (!note.str().empty()) {
        note << ";";
      }
      note << "expected[" << i << "] class=" << exp.class_id << " best_iou=" << best_iou;
      ok = false;
    } else {
      used[best_idx] = true;
    }
  }

  res.ok = ok;
  res.note = note.str();
  return res;
}

void collect_tensors_from_sample_local(const simaai::neat::Sample& sample,
                                       std::vector<simaai::neat::Tensor>& out) {
  collect_tensors_in_sample(sample, out);
}

void collect_preproc_probe_candidates_local(const simaai::neat::Sample& sample,
                                            std::vector<simaai::neat::Tensor>& out) {
  std::vector<simaai::neat::Tensor> base;
  collect_tensors_from_sample_local(sample, base);
  for (const auto& tensor : base) {
    out.push_back(tensor);
  }
  for (const auto& tensor : base) {
    for (int mem_index = 0; mem_index <= 3; ++mem_index) {
      try {
        out.push_back(simaai::neat::pipeline_internal::copy_tensor_from_sample_memory(
            tensor, mem_index, false));
      } catch (const std::exception&) {
      }
    }
  }
}

std::string tensor_contract_string(const simaai::neat::TensorSpec& spec) {
  std::ostringstream ss;
  if (!spec.dtypes.empty()) {
    ss << dtype_name(spec.dtypes.front()) << " ";
  }
  if (!spec.shape.empty()) {
    ss << shape_string(spec.shape);
  } else {
    ss << "<dynamic>";
  }
  return ss.str();
}

std::vector<simaai::neat::Tensor>
order_tensors_for_contract_check(std::vector<simaai::neat::Tensor> tensors) {
  const bool all_indexed = !tensors.empty() && std::all_of(tensors.begin(), tensors.end(),
                                                           [](const simaai::neat::Tensor& tensor) {
                                                             return tensor.route.logical_index >= 0;
                                                           });
  if (!all_indexed) {
    return tensors;
  }
  std::sort(tensors.begin(), tensors.end(),
            [](const simaai::neat::Tensor& a, const simaai::neat::Tensor& b) {
              return a.route.logical_index < b.route.logical_index;
            });
  return tensors;
}

void require_tensor_outputs_match_model_contract(const simaai::neat::Model& model,
                                                 const std::vector<simaai::neat::Tensor>& tensors) {
  const auto specs = model.output_specs();
  require(!specs.empty(), "tensor-output route exposed no model output_specs()");
  require(tensors.size() == specs.size(), "tensor-output count mismatch: expected " +
                                              std::to_string(specs.size()) + ", got " +
                                              std::to_string(tensors.size()));

  const auto ordered = order_tensors_for_contract_check(tensors);
  for (size_t i = 0; i < ordered.size(); ++i) {
    const auto& tensor = ordered[i];
    const auto& spec = specs[i];
    require(spec.matches(tensor), "tensor-output contract mismatch at index " + std::to_string(i) +
                                      ": expected " + tensor_contract_string(spec) + ", got " +
                                      dtype_name(tensor.dtype) + " " + shape_string(tensor.shape));
    require(tensor.route.logical_index >= 0,
            "tensor-output missing logical_index at index " + std::to_string(i));
    require(!tensor.route.segment_name.empty(),
            "tensor-output missing segment_name at index " + std::to_string(i));
  }
}

struct HostDecodeHeadSpec {
  std::string name;
  bool is_bbox = false;
  bool is_class = false;
  int h = 0;
  int w = 0;
  int c = 0;
  bool has_dequant = false;
  float dq_scale = 0.0f;
  int dq_zp = 0;
};

std::vector<HostDecodeHeadSpec> load_host_decode_heads_from_mpk_local(const ProbeResult& probe,
                                                                      std::string* note_out) {
  std::vector<HostDecodeHeadSpec> heads;
  if (probe.etc_dir.empty()) {
    if (note_out)
      *note_out = "missing_etc_dir";
    return heads;
  }

  fs::path mpk_path = probe.etc_dir / "yolov8n_modified_mpk.json";
  if (!fs::exists(mpk_path)) {
    if (const auto any_mpk = find_json_suffix_path(probe.etc_dir, "_mpk.json");
        any_mpk.has_value()) {
      mpk_path = *any_mpk;
    } else {
      if (note_out)
        *note_out = "missing_mpk_json";
      return heads;
    }
  }

  const auto j = read_json_file(mpk_path);
  if (!j.has_value() || !j->contains("plugins") || !(*j)["plugins"].is_array()) {
    if (note_out)
      *note_out = "invalid_mpk_json";
    return heads;
  }

  for (const auto& plugin : (*j)["plugins"]) {
    if (!plugin.is_object()) {
      continue;
    }
    const std::string plugin_name = plugin.value("name", "");
    const std::string plugin_name_l = lower_copy(plugin_name);
    nlohmann::json params = nlohmann::json::object();
    if (plugin.contains("config_params") && plugin["config_params"].is_object() &&
        plugin["config_params"].contains("params") &&
        plugin["config_params"]["params"].is_object()) {
      params = plugin["config_params"]["params"];
    }

    float dq_scale = 0.0f;
    int dq_zp = 0;
    bool has_dequant = false;
    if (plugin_name_l.find("dequantize") != std::string::npos &&
        params.contains("channel_params") && params["channel_params"].is_array() &&
        !params["channel_params"].empty() && params["channel_params"][0].is_array() &&
        params["channel_params"][0].size() >= 2) {
      const auto& cp = params["channel_params"][0];
      if (cp[0].is_number() && cp[1].is_number()) {
        dq_scale = static_cast<float>(cp[0].get<double>());
        dq_zp = static_cast<int>(cp[1].get<double>());
        has_dequant = dq_scale > 0.0f;
      }
    }

    if (!plugin.contains("output_nodes") || !plugin["output_nodes"].is_array()) {
      continue;
    }
    const auto& outputs = plugin["output_nodes"];
    const bool has_output_shapes =
        params.contains("output_shapes") && params["output_shapes"].is_array();

    for (std::size_t oi = 0; oi < outputs.size(); ++oi) {
      if (!outputs[oi].is_object()) {
        continue;
      }
      const std::string out_name = outputs[oi].value("name", "");
      const std::string out_name_l = lower_copy(out_name);
      const bool is_bbox = out_name_l.find("bbox") != std::string::npos;
      const bool is_class = out_name_l.find("class_prob") != std::string::npos;
      if (!is_bbox && !is_class) {
        continue;
      }

      int h = 0;
      int w = 0;
      int c = 0;
      if (has_output_shapes && oi < params["output_shapes"].size() &&
          params["output_shapes"][oi].is_array()) {
        const auto& sh = params["output_shapes"][oi];
        if (sh.size() >= 4 && sh[1].is_number_integer() && sh[2].is_number_integer() &&
            sh[3].is_number_integer()) {
          h = sh[1].get<int>();
          w = sh[2].get<int>();
          c = sh[3].get<int>();
        }
      }
      if (h <= 0 || w <= 0 || c <= 0) {
        continue;
      }

      HostDecodeHeadSpec head;
      head.name = out_name;
      head.is_bbox = is_bbox;
      head.is_class = is_class;
      head.h = h;
      head.w = w;
      head.c = c;
      head.has_dequant = has_dequant;
      head.dq_scale = dq_scale;
      head.dq_zp = dq_zp;
      heads.push_back(std::move(head));
    }
  }

  std::sort(heads.begin(), heads.end(),
            [](const HostDecodeHeadSpec& a, const HostDecodeHeadSpec& b) {
              if (a.h != b.h) {
                return a.h > b.h;
              }
              if (a.w != b.w) {
                return a.w > b.w;
              }
              if (a.is_bbox != b.is_bbox) {
                return a.is_bbox && !b.is_bbox;
              }
              return a.name < b.name;
            });

  if (note_out) {
    *note_out = "heads=" + std::to_string(heads.size());
  }
  return heads;
}

int element_bytes_for_dtype_local(simaai::neat::TensorDType dtype) {
  switch (dtype) {
  case simaai::neat::TensorDType::UInt8:
  case simaai::neat::TensorDType::Int8:
    return 1;
  case simaai::neat::TensorDType::UInt16:
  case simaai::neat::TensorDType::Int16:
  case simaai::neat::TensorDType::BFloat16:
    return 2;
  case simaai::neat::TensorDType::Int32:
  case simaai::neat::TensorDType::Float32:
    return 4;
  case simaai::neat::TensorDType::Float64:
    return 8;
  }
  return 0;
}

bool extract_tensor_raw_bytes_local(const simaai::neat::Tensor& tensor, std::vector<uint8_t>& out) {
  const auto map = tensor.map_read();
  if (!map.data || map.size_bytes == 0) {
    out.clear();
    return false;
  }
  out.resize(map.size_bytes);
  std::memcpy(out.data(), map.data, map.size_bytes);
  return true;
}

struct PreMlaTensorDigest {
  std::string dtype;
  std::string shape;
  std::string fmt;
  std::size_t bytes = 0;
  uint64_t hash = 0;
  int byte_min = 0;
  int byte_max = 0;
  double byte_mean = 0.0;
  std::string preview_hex;
};

struct PreMlaParityResult {
  bool ok = false;
  std::string note;
};

struct ScopedEnvOverride {
  std::string key;
  bool had_value = false;
  std::string old_value;

  ScopedEnvOverride(const char* env_key, const char* new_value) {
    if (!env_key || !*env_key) {
      return;
    }
    key = env_key;
    const char* cur = std::getenv(key.c_str());
    if (cur) {
      had_value = true;
      old_value = cur;
    }
    setenv(key.c_str(), (new_value ? new_value : ""), 1);
  }

  ~ScopedEnvOverride() {
    if (key.empty()) {
      return;
    }
    if (had_value) {
      setenv(key.c_str(), old_value.c_str(), 1);
    } else {
      unsetenv(key.c_str());
    }
  }
};

uint64_t fnv1a64_local(const uint8_t* data, std::size_t size) {
  constexpr uint64_t kOffset = 1469598103934665603ULL;
  constexpr uint64_t kPrime = 1099511628211ULL;
  uint64_t hash = kOffset;
  for (std::size_t i = 0; i < size; ++i) {
    hash ^= static_cast<uint64_t>(data[i]);
    hash *= kPrime;
  }
  return hash;
}

std::string hex_u64_local(uint64_t value) {
  std::ostringstream oss;
  oss << "0x" << std::hex << std::setfill('0') << std::setw(16) << value;
  return oss.str();
}

std::string preview_hex_local(const std::vector<uint8_t>& bytes, std::size_t limit = 16U) {
  if (bytes.empty()) {
    return "<empty>";
  }
  const std::size_t n = std::min(limit, bytes.size());
  std::ostringstream oss;
  oss << std::hex << std::setfill('0');
  for (std::size_t i = 0; i < n; ++i) {
    if (i) {
      oss << ' ';
    }
    oss << std::setw(2) << static_cast<unsigned>(bytes[i]);
  }
  if (bytes.size() > n) {
    oss << " ...";
  }
  return oss.str();
}

bool numeric_trace_debug_enabled() {
  const char* raw = std::getenv("SIMA_YOLOV8_NUMERIC_TRACE");
  return raw && *raw && std::strcmp(raw, "0") != 0;
}

std::size_t numeric_trace_sample_limit() {
  const char* raw = std::getenv("SIMA_YOLOV8_NUMERIC_TRACE_SAMPLES");
  if (!raw || !*raw) {
    return 8U;
  }
  const long parsed = std::strtol(raw, nullptr, 10);
  return parsed > 0 ? static_cast<std::size_t>(parsed) : 8U;
}

struct NumericTensorDigestLocal {
  std::string dtype;
  std::string shape;
  std::string segment;
  std::size_t bytes = 0U;
  uint64_t hash = 0U;
  double min_v = 0.0;
  double max_v = 0.0;
  double mean_v = 0.0;
  std::size_t nonfinite = 0U;
  double zero_ratio = 0.0;
  std::string samples;
};

NumericTensorDigestLocal summarize_numeric_tensor_local(const simaai::neat::Tensor& tensor) {
  NumericTensorDigestLocal out;
  const auto cpu = tensor.cpu().contiguous();
  std::vector<uint8_t> bytes;
  (void)extract_tensor_raw_bytes_local(cpu, bytes);
  out.dtype = dtype_name(cpu.dtype);
  out.shape = shape_string(cpu.shape);
  out.segment = cpu.route.segment_name;
  out.bytes = bytes.size();
  out.hash = fnv1a64_local(bytes.data(), bytes.size());
  if (bytes.empty()) {
    out.samples = "[]";
    return out;
  }

  double min_v = std::numeric_limits<double>::infinity();
  double max_v = -std::numeric_limits<double>::infinity();
  long double sum_v = 0.0;
  std::size_t count = 0U;
  std::size_t nonfinite = 0U;
  std::size_t zero_byte_count = 0U;
  for (uint8_t byte : bytes) {
    if (byte == 0U) {
      ++zero_byte_count;
    }
  }

  std::ostringstream samples;
  samples << "[";
  const std::size_t sample_limit = numeric_trace_sample_limit();
  auto consume_value = [&](double value) {
    min_v = std::min(min_v, value);
    max_v = std::max(max_v, value);
    sum_v += static_cast<long double>(value);
    if (!std::isfinite(value)) {
      ++nonfinite;
    }
    if (count < sample_limit) {
      if (count != 0U) {
        samples << ",";
      }
      samples << std::fixed << std::setprecision(6) << value;
    }
    ++count;
  };

  switch (cpu.dtype) {
  case simaai::neat::TensorDType::BFloat16: {
    if ((bytes.size() % sizeof(uint16_t)) == 0U) {
      const auto* ptr = reinterpret_cast<const uint16_t*>(bytes.data());
      for (std::size_t i = 0; i < (bytes.size() / sizeof(uint16_t)); ++i) {
        consume_value(static_cast<double>(bf16_to_fp32(ptr[i])));
      }
    }
    break;
  }
  case simaai::neat::TensorDType::Float32: {
    if ((bytes.size() % sizeof(float)) == 0U) {
      const auto* ptr = reinterpret_cast<const float*>(bytes.data());
      for (std::size_t i = 0; i < (bytes.size() / sizeof(float)); ++i) {
        consume_value(static_cast<double>(ptr[i]));
      }
    }
    break;
  }
  case simaai::neat::TensorDType::Int16: {
    if ((bytes.size() % sizeof(int16_t)) == 0U) {
      const auto* ptr = reinterpret_cast<const int16_t*>(bytes.data());
      for (std::size_t i = 0; i < (bytes.size() / sizeof(int16_t)); ++i) {
        consume_value(static_cast<double>(ptr[i]));
      }
    }
    break;
  }
  case simaai::neat::TensorDType::UInt16: {
    if ((bytes.size() % sizeof(uint16_t)) == 0U) {
      const auto* ptr = reinterpret_cast<const uint16_t*>(bytes.data());
      for (std::size_t i = 0; i < (bytes.size() / sizeof(uint16_t)); ++i) {
        consume_value(static_cast<double>(ptr[i]));
      }
    }
    break;
  }
  case simaai::neat::TensorDType::Int8: {
    const auto* ptr = reinterpret_cast<const int8_t*>(bytes.data());
    for (std::size_t i = 0; i < bytes.size(); ++i) {
      consume_value(static_cast<double>(ptr[i]));
    }
    break;
  }
  case simaai::neat::TensorDType::UInt8: {
    const auto* ptr = reinterpret_cast<const uint8_t*>(bytes.data());
    for (std::size_t i = 0; i < bytes.size(); ++i) {
      consume_value(static_cast<double>(ptr[i]));
    }
    break;
  }
  default:
    break;
  }

  samples << "]";
  out.samples = samples.str();
  out.min_v = std::isfinite(min_v) ? min_v : 0.0;
  out.max_v = std::isfinite(max_v) ? max_v : 0.0;
  out.mean_v = count == 0U ? 0.0 : static_cast<double>(sum_v / static_cast<long double>(count));
  out.nonfinite = nonfinite;
  out.zero_ratio = static_cast<double>(zero_byte_count) / static_cast<double>(bytes.size());
  return out;
}

void maybe_trace_sample_numeric_local(const char* phase, const simaai::neat::Sample& sample) {
  if (!numeric_trace_debug_enabled()) {
    return;
  }
  const auto tensors = tensors_in_sample(sample);
  std::fprintf(stderr, "[matrix-numeric] phase=%s tensors=%zu signature=%s\n", phase ? phase : "",
               tensors.size(), sample_output_signature_local(sample).c_str());
  for (std::size_t i = 0; i < tensors.size(); ++i) {
    const auto digest = summarize_numeric_tensor_local(tensors[i]);
    std::fprintf(stderr,
                 "[matrix-numeric] phase=%s tensor=%zu dtype=%s shape=%s seg=%s bytes=%zu "
                 "hash=%s min=%.6f max=%.6f mean=%.6f nonfinite=%zu zero_ratio=%.6f samples=%s\n",
                 phase ? phase : "", i, digest.dtype.c_str(), digest.shape.c_str(),
                 digest.segment.empty() ? "<none>" : digest.segment.c_str(), digest.bytes,
                 hex_u64_local(digest.hash).c_str(), digest.min_v, digest.max_v, digest.mean_v,
                 digest.nonfinite, digest.zero_ratio, digest.samples.c_str());
  }
}

PreMlaTensorDigest summarize_pre_mla_tensor_local(const simaai::neat::Tensor& tensor) {
  PreMlaTensorDigest out;
  const auto cpu = tensor.cpu().contiguous();
  std::vector<uint8_t> bytes;
  (void)extract_tensor_raw_bytes_local(cpu, bytes);
  out.dtype = dtype_name(cpu.dtype);
  out.shape = shape_string(cpu.shape);
  out.fmt = tensor_format(cpu);
  out.bytes = bytes.size();
  out.preview_hex = preview_hex_local(bytes);
  if (bytes.empty()) {
    return out;
  }
  out.hash = fnv1a64_local(bytes.data(), bytes.size());
  uint64_t sum = 0;
  out.byte_min = 255;
  out.byte_max = 0;
  for (uint8_t v : bytes) {
    const int iv = static_cast<int>(v);
    out.byte_min = std::min(out.byte_min, iv);
    out.byte_max = std::max(out.byte_max, iv);
    sum += static_cast<uint64_t>(v);
  }
  out.byte_mean = static_cast<double>(sum) / static_cast<double>(bytes.size());
  return out;
}

std::string pre_mla_tensor_digest_string_local(const char* prefix, std::size_t index,
                                               const PreMlaTensorDigest& d) {
  std::ostringstream oss;
  oss << prefix << "[" << index << "]" << " dtype=" << d.dtype << " shape=" << d.shape
      << " fmt=" << (d.fmt.empty() ? "<none>" : d.fmt) << " bytes=" << d.bytes
      << " hash=" << hex_u64_local(d.hash) << " byte_min=" << d.byte_min
      << " byte_max=" << d.byte_max << " byte_mean=" << d.byte_mean << " head=" << d.preview_hex;
  return oss.str();
}

struct PreprocOpenCvParityResult {
  bool ok = false;
  int observed_pad_value = -1;
  double valid_mae = 0.0;
  double valid_max_abs = 0.0;
  double pad_max_dev = 0.0;
  std::string effective_dtype;
  std::string note;
};

bool tensor_to_cv32f_hwc3_local(const simaai::neat::Tensor& tensor, cv::Mat& out,
                                std::string& note) {
  const auto cpu = tensor.cpu().contiguous();
  const std::vector<uint8_t> bytes = cpu.copy_payload_bytes();
  if (bytes.empty()) {
    note = "empty_tensor_payload";
    return false;
  }

  std::vector<int64_t> dims = cpu.shape;
  if (dims.size() == 4U && dims[0] == 1) {
    dims.erase(dims.begin());
  }
  if (dims.size() != 3U) {
    note = "unsupported_tensor_rank shape=" + shape_string(cpu.shape);
    return false;
  }

  int h = 0;
  int w = 0;
  int c = 0;
  if (cpu.layout == simaai::neat::TensorLayout::CHW) {
    c = static_cast<int>(dims[0]);
    h = static_cast<int>(dims[1]);
    w = static_cast<int>(dims[2]);
  } else {
    h = static_cast<int>(dims[0]);
    w = static_cast<int>(dims[1]);
    c = static_cast<int>(dims[2]);
  }
  if (h <= 0 || w <= 0 || c <= 0) {
    note = "invalid_tensor_dims shape=" + shape_string(cpu.shape);
    return false;
  }

  const int elem_bytes = element_bytes_for_dtype_local(cpu.dtype);
  const std::size_t elems =
      static_cast<std::size_t>(h) * static_cast<std::size_t>(w) * static_cast<std::size_t>(c);
  const std::size_t required_bytes = elems * static_cast<std::size_t>(elem_bytes);
  if (elem_bytes <= 0 || bytes.size() < required_bytes) {
    note = "invalid_tensor_bytes expected=" + std::to_string(required_bytes) +
           " got=" + std::to_string(bytes.size());
    return false;
  }

  auto read_elem_as_float = [&](std::size_t idx) -> float {
    switch (cpu.dtype) {
    case simaai::neat::TensorDType::UInt8:
      return static_cast<float>(reinterpret_cast<const uint8_t*>(bytes.data())[idx]);
    case simaai::neat::TensorDType::Int8:
      return static_cast<float>(reinterpret_cast<const int8_t*>(bytes.data())[idx]);
    case simaai::neat::TensorDType::UInt16:
      return static_cast<float>(reinterpret_cast<const uint16_t*>(bytes.data())[idx]);
    case simaai::neat::TensorDType::Int16:
      return static_cast<float>(reinterpret_cast<const int16_t*>(bytes.data())[idx]);
    case simaai::neat::TensorDType::BFloat16: {
      const uint16_t v = reinterpret_cast<const uint16_t*>(bytes.data())[idx];
      return bf16_to_fp32(v);
    }
    case simaai::neat::TensorDType::Float32:
      return reinterpret_cast<const float*>(bytes.data())[idx];
    default:
      return 0.0f;
    }
  };

  if (!(cpu.dtype == simaai::neat::TensorDType::UInt8 ||
        cpu.dtype == simaai::neat::TensorDType::Int8 ||
        cpu.dtype == simaai::neat::TensorDType::UInt16 ||
        cpu.dtype == simaai::neat::TensorDType::Int16 ||
        cpu.dtype == simaai::neat::TensorDType::BFloat16 ||
        cpu.dtype == simaai::neat::TensorDType::Float32)) {
    note = "unsupported_tensor_dtype=" + dtype_name(cpu.dtype);
    return false;
  }

  cv::Mat tmp(h, w, CV_32FC3, cv::Scalar(0.0f, 0.0f, 0.0f));
  for (int y = 0; y < h; ++y) {
    for (int x = 0; x < w; ++x) {
      cv::Vec3f pix(0.0f, 0.0f, 0.0f);
      for (int ch = 0; ch < std::min(c, 3); ++ch) {
        std::size_t src_idx = 0U;
        if (cpu.layout == simaai::neat::TensorLayout::CHW) {
          src_idx = static_cast<std::size_t>(ch) * static_cast<std::size_t>(h) *
                        static_cast<std::size_t>(w) +
                    static_cast<std::size_t>(y) * static_cast<std::size_t>(w) +
                    static_cast<std::size_t>(x);
        } else {
          src_idx = (static_cast<std::size_t>(y) * static_cast<std::size_t>(w) +
                     static_cast<std::size_t>(x)) *
                        static_cast<std::size_t>(c) +
                    static_cast<std::size_t>(ch);
        }
        pix[ch] = read_elem_as_float(src_idx);
      }
      tmp.at<cv::Vec3f>(y, x) = pix;
    }
  }
  out = std::move(tmp);
  note = "ok";
  return true;
}

simaai::neat::Sample run_preproc_probe_sample_local(const cv::Mat& img_bgr,
                                                    const std::string& output_dtype) {
  simaai::neat::InputOptions src_opt;
  src_opt.payload_type = simaai::neat::PayloadType::Image;
  src_opt.format = simaai::neat::FormatTag::BGR;
  src_opt.width = img_bgr.cols;
  src_opt.height = img_bgr.rows;
  src_opt.depth = 3;
  src_opt.buffer_name = "decoder";

  simaai::neat::PreprocOptions pre_opt;
  pre_opt.set_input_shape({img_bgr.rows, img_bgr.cols, 3});
  pre_opt.set_output_shape({kFixedPreprocTargetHeight, kFixedPreprocTargetWidth, 3});
  pre_opt.scaled_width = kFixedPreprocTargetWidth;
  pre_opt.scaled_height = kFixedPreprocTargetHeight;
  pre_opt.input_img_type = "BGR";
  pre_opt.output_img_type = "RGB";
  pre_opt.normalize = true;
  pre_opt.aspect_ratio = true;
  pre_opt.tessellate = false;
  pre_opt.scaling_type = "BILINEAR";
  pre_opt.padding_type = "CENTER";
  pre_opt.output_dtype = output_dtype;
  pre_opt.next_cpu = "APU";
  pre_opt.upstream_name = "decoder";
  pre_opt.graph_input_name = "input_image";

  simaai::neat::Graph p;
  auto pre_node = simaai::neat::nodes::Preproc(pre_opt);
  p.add(simaai::neat::nodes::Input(src_opt));
  p.add(pre_node);
  p.add(simaai::neat::nodes::Output());
  simaai::neat::RunOptions run_opt;
  run_opt.output_memory = simaai::neat::OutputMemory::Owned;
  run_opt.queue_depth = 1;
  auto run = p.build(std::vector<cv::Mat>{img_bgr}, run_opt);
  require(run.push(std::vector<cv::Mat>{img_bgr}), "preproc_probe: push failed");
  simaai::neat::Sample outs = run.pull_samples(default_model_run_timeout_ms());
  require(outs.size() == 1U, "preproc_probe: expected exactly 1 output sample");
  simaai::neat::Sample out = outs.front();
  run.close();
  return out;
}

PreprocOpenCvParityResult run_preproc_vs_opencv_parity_once_local(const cv::Mat& img_bgr) {
  PreprocOpenCvParityResult out;
  if (!img_bgr.data || img_bgr.cols <= 0 || img_bgr.rows <= 0) {
    out.note = "invalid_input_image";
    return out;
  }

  std::vector<std::string> probe_errors;
  std::vector<simaai::neat::Tensor> preproc_tensors;
  std::vector<std::string> preproc_probe_dtypes;
  bool probe_ok = false;
  for (const std::string& candidate_dtype : {"EVXX_BFLOAT16", "INT16", "EVXX_INT8"}) {
    try {
      const simaai::neat::Sample sample = run_preproc_probe_sample_local(img_bgr, candidate_dtype);
      std::vector<simaai::neat::Tensor> probe_candidates;
      collect_preproc_probe_candidates_local(sample, probe_candidates);
      if (probe_candidates.empty()) {
        probe_errors.push_back(candidate_dtype + ":no_output_tensor");
        continue;
      }
      for (const auto& tensor : probe_candidates) {
        preproc_tensors.push_back(tensor);
        preproc_probe_dtypes.push_back(candidate_dtype);
      }
      probe_ok = true;
    } catch (const std::exception& ex) {
      probe_errors.push_back(candidate_dtype + ":" + sima_yolov8_test::sanitize_note(ex.what()));
    }
  }

  if (!probe_ok) {
    out.note = "preproc_probe_failed " +
               (probe_errors.empty() ? std::string("unknown") : probe_errors.front());
    for (std::size_t i = 1; i < probe_errors.size(); ++i) {
      out.note += " | " + probe_errors[i];
    }
    return out;
  }

  cv::Mat opencv_pre =
      resize_to_target_local(img_bgr, kFixedPreprocTargetWidth, kFixedPreprocTargetHeight, true,
                             cv::INTER_LINEAR, cv::Scalar(0, 0, 0, 0), "CENTER");
  cv::Mat opencv_bgr_f32;
  opencv_pre.convertTo(opencv_bgr_f32, CV_32FC3, 1.0 / 255.0, 0.0);
  cv::Mat opencv_rgb = convert_color_for_format(opencv_pre, "RGB", "preproc_vs_opencv");
  cv::Mat opencv_rgb_f32;
  opencv_rgb.convertTo(opencv_rgb_f32, CV_32FC3, 1.0 / 255.0, 0.0);

  const LetterboxGeometry g = compute_letterbox_geometry(
      img_bgr.cols, img_bgr.rows, kFixedPreprocTargetWidth, kFixedPreprocTargetHeight, "CENTER");
  const int valid_x0 = std::max(0, g.pad_left);
  const int valid_y0 = std::max(0, g.pad_top);
  const int valid_x1 = std::min(kFixedPreprocTargetWidth, g.pad_left + g.resized_w);
  const int valid_y1 = std::min(kFixedPreprocTargetHeight, g.pad_top + g.resized_h);
  if (valid_x1 <= valid_x0 || valid_y1 <= valid_y0) {
    out.note = "invalid_letterbox_geometry";
    return out;
  }

  bool found_candidate = false;
  std::size_t best_index = 0U;
  std::string best_dtype;
  std::string best_probe_dtype = "unknown";
  std::string best_ref = "RGB";
  double best_mae = std::numeric_limits<double>::infinity();
  double best_max_abs = std::numeric_limits<double>::infinity();
  double best_pad_max_dev = std::numeric_limits<double>::infinity();
  int best_observed_pad = -1;
  std::string decode_errors;
  std::vector<std::string> candidate_summaries;

  for (std::size_t i = 0; i < preproc_tensors.size(); ++i) {
    cv::Mat preproc_rgb_f32;
    std::string tensor_note;
    if (!tensor_to_cv32f_hwc3_local(preproc_tensors[i], preproc_rgb_f32, tensor_note)) {
      decode_errors += " idx=" + std::to_string(i) + ":" + tensor_note;
      continue;
    }
    if (preproc_rgb_f32.cols != kFixedPreprocTargetWidth ||
        preproc_rgb_f32.rows != kFixedPreprocTargetHeight) {
      decode_errors += " idx=" + std::to_string(i) +
                       ":shape=" + std::to_string(preproc_rgb_f32.cols) + "x" +
                       std::to_string(preproc_rgb_f32.rows);
      continue;
    }

    double abs_sum_rgb = 0.0;
    double abs_max_rgb = 0.0;
    double abs_sum_bgr = 0.0;
    double abs_max_bgr = 0.0;
    std::size_t abs_count = 0U;
    std::vector<float> padded_values;
    padded_values.reserve(static_cast<std::size_t>(kFixedPreprocTargetWidth) *
                          static_cast<std::size_t>(kFixedPreprocTargetHeight));
    for (int y = 0; y < kFixedPreprocTargetHeight; ++y) {
      for (int x = 0; x < kFixedPreprocTargetWidth; ++x) {
        const cv::Vec3f p0 = preproc_rgb_f32.at<cv::Vec3f>(y, x);
        const cv::Vec3f p1_rgb = opencv_rgb_f32.at<cv::Vec3f>(y, x);
        const cv::Vec3f p1_bgr = opencv_bgr_f32.at<cv::Vec3f>(y, x);
        const bool is_valid = x >= valid_x0 && x < valid_x1 && y >= valid_y0 && y < valid_y1;
        if (is_valid) {
          for (int ch = 0; ch < 3; ++ch) {
            const double d_rgb =
                std::abs(static_cast<double>(p0[ch]) - static_cast<double>(p1_rgb[ch]));
            const double d_bgr =
                std::abs(static_cast<double>(p0[ch]) - static_cast<double>(p1_bgr[ch]));
            abs_sum_rgb += d_rgb;
            abs_max_rgb = std::max(abs_max_rgb, d_rgb);
            abs_sum_bgr += d_bgr;
            abs_max_bgr = std::max(abs_max_bgr, d_bgr);
            abs_count += 1U;
          }
        } else {
          padded_values.push_back(p0[0]);
          padded_values.push_back(p0[1]);
          padded_values.push_back(p0[2]);
        }
      }
    }
    if (abs_count == 0U || padded_values.empty()) {
      decode_errors += " idx=" + std::to_string(i) + ":invalid_regions";
      continue;
    }

    const std::size_t mid = padded_values.size() / 2U;
    std::nth_element(padded_values.begin(),
                     padded_values.begin() + static_cast<std::ptrdiff_t>(mid), padded_values.end());
    const float pad_median = padded_values[mid];
    const int observed_pad = static_cast<int>(std::lround(pad_median));
    double pad_max_dev = 0.0;
    for (float v : padded_values) {
      pad_max_dev = std::max(pad_max_dev,
                             std::abs(static_cast<double>(v) - static_cast<double>(observed_pad)));
    }

    const double mae_rgb = abs_sum_rgb / static_cast<double>(abs_count);
    const double mae_bgr = abs_sum_bgr / static_cast<double>(abs_count);
    if (!std::isfinite(mae_rgb) || !std::isfinite(mae_bgr) || !std::isfinite(abs_max_rgb) ||
        !std::isfinite(abs_max_bgr) || !std::isfinite(pad_max_dev)) {
      decode_errors += " idx=" + std::to_string(i) + ":non_finite_candidate";
      continue;
    }
    const bool use_rgb_ref = (mae_rgb <= mae_bgr);
    const double mae = use_rgb_ref ? mae_rgb : mae_bgr;
    const double abs_max = use_rgb_ref ? abs_max_rgb : abs_max_bgr;
    std::ostringstream cand;
    const std::string probe_dtype =
        (i < preproc_probe_dtypes.size()) ? preproc_probe_dtypes[i] : std::string("unknown");
    cand << "idx=" << i << ":probe=" << probe_dtype
         << ":dtype=" << dtype_name(preproc_tensors[i].dtype)
         << ":ref=" << (use_rgb_ref ? "RGB" : "BGR") << ":mae=" << std::fixed
         << std::setprecision(4) << mae << ":max=" << std::setprecision(4) << abs_max
         << ":pad=" << observed_pad << ":pad_dev=" << std::setprecision(4) << pad_max_dev;
    candidate_summaries.push_back(cand.str());
    const double score = mae + 0.01 * abs_max + 0.1 * pad_max_dev;
    const double best_score = best_mae + 0.01 * best_max_abs + 0.1 * best_pad_max_dev;
    if (!found_candidate || score < best_score) {
      found_candidate = true;
      best_index = i;
      best_dtype = dtype_name(preproc_tensors[i].dtype);
      best_probe_dtype = probe_dtype;
      best_ref = use_rgb_ref ? "RGB" : "BGR";
      best_mae = mae;
      best_max_abs = abs_max;
      best_pad_max_dev = pad_max_dev;
      best_observed_pad = observed_pad;
    }
  }

  if (!found_candidate) {
    out.note = "preproc_probe_tensor_decode_failed" + decode_errors;
    return out;
  }

  out.valid_mae = best_mae;
  out.valid_max_abs = best_max_abs;
  out.pad_max_dev = best_pad_max_dev;
  out.observed_pad_value = best_observed_pad;
  out.effective_dtype = best_dtype;

  constexpr double kValidMaeThreshold = 6.0;
  constexpr double kValidMaxAbsThreshold = 96.0;
  constexpr double kPadDevThreshold = 2.0;
  const bool parity_ok = out.valid_mae <= kValidMaeThreshold &&
                         out.valid_max_abs <= kValidMaxAbsThreshold &&
                         out.pad_max_dev <= kPadDevThreshold;
  const bool pad_guard_ok = out.observed_pad_value == kPadValue;
  out.ok = parity_ok && pad_guard_ok;
  std::ostringstream oss;
  oss << "probe_dtype=" << best_probe_dtype << " tensor_index=" << best_index
      << " effective_dtype=" << out.effective_dtype << " ref=" << best_ref
      << " valid_mae=" << out.valid_mae << " valid_max_abs=" << out.valid_max_abs
      << " observed_pad=" << out.observed_pad_value << " expected_pad=" << kPadValue
      << " pad_max_dev=" << out.pad_max_dev << " candidates=" << candidate_summaries.size();
  if (!candidate_summaries.empty()) {
    oss << " [";
    const std::size_t max_items = std::min<std::size_t>(candidate_summaries.size(), 6U);
    for (std::size_t i = 0; i < max_items; ++i) {
      if (i) {
        oss << " | ";
      }
      oss << candidate_summaries[i];
    }
    if (candidate_summaries.size() > max_items) {
      oss << " | ...";
    }
    oss << "]";
  }
  if (!parity_ok) {
    oss << " reason=preproc_opencv_mismatch";
  } else if (!pad_guard_ok) {
    oss << " reason=pad_guard_mismatch";
  } else {
    oss << " reason=ok";
  }
  out.note = oss.str();
  return out;
}

const PreprocOpenCvParityResult& cached_preproc_vs_opencv_parity_local(const cv::Mat& img_bgr) {
  static PreprocOpenCvParityResult cached;
  static int cached_w = 0;
  static int cached_h = 0;
  static bool initialized = false;
  if (!initialized || cached_w != img_bgr.cols || cached_h != img_bgr.rows) {
    cached = run_preproc_vs_opencv_parity_once_local(img_bgr);
    cached_w = img_bgr.cols;
    cached_h = img_bgr.rows;
    initialized = true;
  }
  return cached;
}

PreMlaParityResult run_pre_mla_parity_check_local(const ProbeResult& probe, RouteKind route,
                                                  const cv::Mat& img_bgr) {
  PreMlaParityResult out;
  PreprocOpenCvParityResult preproc_parity;
  bool have_preproc_parity = false;
  if (route != RouteKind::PreInferPost) {
    out.ok = true;
    out.note = "skip_route";
    return out;
  }
  if (!pre_mla_parity_enabled()) {
    out.ok = true;
    out.note = "disabled";
    return out;
  }

  try {
    simaai::neat::Model matrix_model = build_model_for_case(probe);
    const PreAdapterKind pre_kind = detect_pre_kind(
        simaai::neat::internal::ModelAccess::build_public_preprocess_nodes(matrix_model));
    const bool is_adapter_pre = pre_kind == PreAdapterKind::Quant ||
                                pre_kind == PreAdapterKind::Tess ||
                                pre_kind == PreAdapterKind::QuantTess;
    if (!is_adapter_pre) {
      out.ok = true;
      out.note = "skip_pre_kind=" + pre_kind_name(pre_kind);
      return out;
    }

    preproc_parity = cached_preproc_vs_opencv_parity_local(img_bgr);
    have_preproc_parity = true;
    if (!preproc_parity.ok) {
      out.ok = false;
      out.note = "preproc_opencv_parity " + preproc_parity.note;
      return out;
    }
    if (pre_kind != PreAdapterKind::QuantTess) {
      out.ok = true;
      out.note = "preproc_opencv_parity " + preproc_parity.note;
      return out;
    }

    const auto matrix_spec = matrix_model.input_specs().front();
    const cv::Mat matrix_seed = normalize_model_input(img_bgr, matrix_model, route);
    const AdapterIngressTensorInput matrix_ingress =
        build_adapter_tensor_ingress_input(matrix_seed, matrix_model, matrix_spec, route, pre_kind);

    auto matrix_input_opt = matrix_model.input_appsrc_options(false);
    simaai::neat::Graph matrix_graph;
    matrix_graph.add(simaai::neat::nodes::Input(matrix_input_opt));
    matrix_graph.add(simaai::neat::nodes::groups::Preprocess(matrix_model));
    matrix_graph.add(simaai::neat::nodes::Output());
    auto matrix_run = matrix_graph.build_seeded_internal(
        simaai::neat::TensorList{matrix_ingress.tensor}, simaai::neat::RunMode::Sync);
    simaai::neat::Sample matrix_outputs =
        matrix_run.run(simaai::neat::Sample{simaai::neat::sample_from_tensors(
                           simaai::neat::TensorList{matrix_ingress.tensor})},
                       default_model_run_timeout_ms());
    require(matrix_outputs.size() == 1U, "pre_mla_parity: expected one matrix pre-MLA sample");
    const simaai::neat::Sample matrix_pre_mla = matrix_outputs.front();

    simaai::neat::Model::Options sync_opt = default_model_options();
    sync_opt.preprocess.kind = simaai::neat::InputKind::Image;
    sync_opt.preprocess.enable = simaai::neat::AutoFlag::On;
    sync_opt.preprocess.color_convert.input_format = simaai::neat::PreprocessColorFormat::BGR;
    sync_opt.upstream_name = "decoder";
    simaai::neat::Model sync_model(probe.tar_path, sync_opt);

    simaai::neat::Graph sync_graph;
    sync_graph.add(simaai::neat::nodes::Input());
    sync_graph.add(simaai::neat::nodes::groups::Preprocess(sync_model));
    sync_graph.add(simaai::neat::nodes::Output());
    auto sync_run = sync_graph.build_seeded_internal(std::vector<cv::Mat>{img_bgr},
                                                     simaai::neat::RunMode::Sync);
    simaai::neat::Sample sync_outputs = sync_run.run(
        simaai::neat::Sample{simaai::neat::make_image_sample(
            img_bgr, simaai::neat::ImageSpec::PixelFormat::BGR, simaai::neat::TensorMemory::EV74)},
        default_model_run_timeout_ms());
    require(sync_outputs.size() == 1U, "pre_mla_parity: expected one sync pre-MLA sample");
    const simaai::neat::Sample sync_pre_mla = sync_outputs.front();

    std::vector<simaai::neat::Tensor> matrix_tensors;
    std::vector<simaai::neat::Tensor> sync_tensors;
    collect_tensors_from_sample_local(matrix_pre_mla, matrix_tensors);
    collect_tensors_from_sample_local(sync_pre_mla, sync_tensors);
    if (matrix_tensors.empty() || sync_tensors.empty()) {
      out.ok = false;
      out.note = "missing_pre_mla_tensors matrix=" + std::to_string(matrix_tensors.size()) +
                 " sync=" + std::to_string(sync_tensors.size());
      return out;
    }

    std::ostringstream detail;
    bool match = matrix_tensors.size() == sync_tensors.size();
    detail << "count matrix=" << matrix_tensors.size() << " sync=" << sync_tensors.size();
    const std::size_t n = std::min(matrix_tensors.size(), sync_tensors.size());
    for (std::size_t i = 0; i < n; ++i) {
      const PreMlaTensorDigest matrix_d = summarize_pre_mla_tensor_local(matrix_tensors[i]);
      const PreMlaTensorDigest sync_d = summarize_pre_mla_tensor_local(sync_tensors[i]);
      const bool tensor_match = matrix_d.dtype == sync_d.dtype && matrix_d.shape == sync_d.shape &&
                                matrix_d.bytes == sync_d.bytes && matrix_d.hash == sync_d.hash;
      match = match && tensor_match;
      if (!tensor_match || pre_mla_parity_debug_enabled()) {
        detail << " | " << pre_mla_tensor_digest_string_local("matrix", i, matrix_d);
        detail << " | " << pre_mla_tensor_digest_string_local("sync", i, sync_d);
      }
    }

    const bool strict = pre_mla_parity_strict_enabled();
    out.ok = match || !strict;
    out.note = std::string("pre_mla_parity ") + (match ? "ok" : (strict ? "mismatch" : "warn")) +
               " " + detail.str() + " | preproc_opencv_parity " + preproc_parity.note;
    if (!match || pre_mla_parity_debug_enabled()) {
      std::cerr << "[pre-mla-parity] model=" << probe.model_id << " route=" << route_name(route)
                << " strict=" << (strict ? 1 : 0) << " " << out.note << "\n";
    }
    return out;
  } catch (const std::exception& ex) {
    const std::string err = sima_yolov8_test::sanitize_note(ex.what());
    const bool quanttess_strict_contract_miss =
        err.find("misconfig.pipeline_shape") != std::string::npos &&
        err.find("Strict MPK ABI: no matching MPK stage for model-managed stage 'quanttess") !=
            std::string::npos;
    if (have_preproc_parity && preproc_parity.ok && quanttess_strict_contract_miss) {
      out.ok = true;
      out.note = "pre_mla_parity_skip=" + err + " | preproc_opencv_parity " + preproc_parity.note;
      return out;
    }
    out.ok = false;
    out.note = "pre_mla_parity_error=" + err;
    return out;
  }
}

bool decode_bytes_to_fp32_local(const uint8_t* data, std::size_t bytes_size,
                                simaai::neat::TensorDType dtype, const HostDecodeHeadSpec& head,
                                std::vector<float>& out, std::string& note) {
  if (!data || bytes_size == 0U) {
    note = "empty_payload";
    return false;
  }
  const std::vector<uint8_t> bytes(data, data + bytes_size);
  if (bytes.empty()) {
    note = "empty_payload";
    return false;
  }

  const int elem_bytes = element_bytes_for_dtype_local(dtype);
  const std::size_t expected_elems = static_cast<std::size_t>(head.h) *
                                     static_cast<std::size_t>(head.w) *
                                     static_cast<std::size_t>(head.c);
  if (elem_bytes <= 0 || bytes.size() % static_cast<std::size_t>(elem_bytes) != 0U) {
    note = "invalid_tensor_byte_alignment";
    return false;
  }
  const std::size_t elems = bytes.size() / static_cast<std::size_t>(elem_bytes);
  enum class PayloadMode {
    Native,
    Int8AsBf16,
    Int8AsF32,
  };
  PayloadMode mode = PayloadMode::Native;
  if (elems != expected_elems) {
    const bool int8_meta =
        dtype == simaai::neat::TensorDType::Int8 || dtype == simaai::neat::TensorDType::UInt8;
    if (int8_meta && bytes.size() == expected_elems * sizeof(uint16_t)) {
      mode = PayloadMode::Int8AsBf16;
    } else if (int8_meta && bytes.size() == expected_elems * sizeof(float)) {
      mode = PayloadMode::Int8AsF32;
    } else {
      note = "element_count_mismatch expected=" + std::to_string(expected_elems) +
             " got=" + std::to_string(elems);
      return false;
    }
  }

  out.clear();
  out.resize(expected_elems, 0.0f);

  if (mode == PayloadMode::Int8AsBf16) {
    const auto* p = reinterpret_cast<const uint16_t*>(bytes.data());
    for (std::size_t i = 0; i < expected_elems; ++i)
      out[i] = bf16_to_fp32(p[i]);
    return true;
  }
  if (mode == PayloadMode::Int8AsF32) {
    const auto* p = reinterpret_cast<const float*>(bytes.data());
    for (std::size_t i = 0; i < expected_elems; ++i)
      out[i] = p[i];
    return true;
  }

  if (dtype == simaai::neat::TensorDType::Float32) {
    const auto* p = reinterpret_cast<const float*>(bytes.data());
    for (std::size_t i = 0; i < expected_elems; ++i)
      out[i] = p[i];
    return true;
  }
  if (dtype == simaai::neat::TensorDType::BFloat16) {
    const auto* p = reinterpret_cast<const uint16_t*>(bytes.data());
    for (std::size_t i = 0; i < expected_elems; ++i)
      out[i] = bf16_to_fp32(p[i]);
    return true;
  }
  if (dtype == simaai::neat::TensorDType::Int8) {
    const auto* p = reinterpret_cast<const int8_t*>(bytes.data());
    const bool use_dq = head.has_dequant && head.dq_scale > 0.0f;
    for (std::size_t i = 0; i < expected_elems; ++i) {
      out[i] = use_dq
                   ? ((static_cast<float>(p[i]) - static_cast<float>(head.dq_zp)) / head.dq_scale)
                   : static_cast<float>(p[i]);
    }
    return true;
  }
  if (dtype == simaai::neat::TensorDType::UInt8) {
    const auto* p = reinterpret_cast<const uint8_t*>(bytes.data());
    const bool use_dq = head.has_dequant && head.dq_scale > 0.0f;
    for (std::size_t i = 0; i < expected_elems; ++i) {
      out[i] = use_dq
                   ? ((static_cast<float>(p[i]) - static_cast<float>(head.dq_zp)) / head.dq_scale)
                   : static_cast<float>(p[i]);
    }
    return true;
  }

  note = "unsupported_dtype";
  return false;
}

bool decode_tensor_to_fp32_local(const simaai::neat::Tensor& tensor, const HostDecodeHeadSpec& head,
                                 std::vector<float>& out, std::string& note) {
  const auto cpu = tensor.cpu().contiguous();
  const std::vector<uint8_t> bytes = cpu.copy_payload_bytes();
  if (bytes.empty()) {
    note = "empty_payload";
    return false;
  }
  return decode_bytes_to_fp32_local(bytes.data(), bytes.size(), cpu.dtype, head, out, note);
}

float sigmoid_local(float x) {
  if (x >= 0.0f) {
    const float z = std::exp(-x);
    return 1.0f / (1.0f + z);
  }
  const float z = std::exp(x);
  return z / (1.0f + z);
}

std::vector<objdet::Box> decode_yolov8_heads_with_opencv_nms_local(
    const ProbeResult& probe, const simaai::neat::Sample& sample, int model_w, int model_h,
    int output_w, int output_h, bool use_letterbox_remap, float score_threshold,
    float iou_threshold, int topk, std::string* note_out) {
  std::vector<objdet::Box> out;
  std::vector<simaai::neat::Tensor> tensors;
  collect_tensors_from_sample_local(sample, tensors);
  if (tensors.empty()) {
    if (note_out)
      *note_out = "no_output_tensors";
    return out;
  }

  std::string head_note;
  const auto heads = load_host_decode_heads_from_mpk_local(probe, &head_note);
  if (heads.empty()) {
    if (note_out)
      *note_out = "no_mpk_heads " + head_note;
    return out;
  }

  std::vector<bool> tensor_used(tensors.size(), false);
  struct ScalePack {
    int h = 0;
    int w = 0;
    int stride = 0;
    std::vector<float> bbox;
    int bbox_c = 0;
    std::vector<float> cls;
    int cls_c = 0;
    bool has_bbox = false;
    bool has_cls = false;
  };
  std::vector<ScalePack> scales;

  if (model_w <= 0)
    model_w = 640;
  if (model_h <= 0)
    model_h = 640;
  if (output_w <= 0)
    output_w = model_w;
  if (output_h <= 0)
    output_h = model_h;
  const float grid_offset = read_env_float_or_default("SIMA_YOLOV8_HOSTDECODE_GRID_OFFSET", 0.5f);
  const float default_dfl_dist_scale = use_letterbox_remap ? 3.0f : 1.0f;
  const float dfl_dist_scale =
      read_env_float_or_default("SIMA_YOLOV8_HOSTDECODE_DIST_SCALE", default_dfl_dist_scale);
  const bool decode_chw_layout = []() -> bool {
    const char* env = std::getenv("SIMA_YOLOV8_HOSTDECODE_LAYOUT");
    if (!env || !*env) {
      return false;
    }
    return upper_copy(std::string(env)) == "CHW";
  }();
  const std::array<int, 4> side_group_for_ltrb = []() -> std::array<int, 4> {
    std::array<int, 4> out = {0, 1, 2, 3}; // L,T,R,B
    const char* env = std::getenv("SIMA_YOLOV8_HOSTDECODE_SIDE_ORDER");
    if (!env || !*env) {
      return out;
    }
    const std::string order = upper_copy(std::string(env));
    if (order.size() < 4) {
      return out;
    }
    auto map_char = [](char c) -> int {
      if (c == 'L')
        return 0;
      if (c == 'T')
        return 1;
      if (c == 'R')
        return 2;
      if (c == 'B')
        return 3;
      return -1;
    };
    std::array<int, 4> pos = {-1, -1, -1, -1};
    for (int i = 0; i < 4; ++i) {
      const int id = map_char(order[static_cast<std::size_t>(i)]);
      if (id < 0 || pos[id] >= 0) {
        return out;
      }
      pos[id] = i;
    }
    if (pos[0] < 0 || pos[1] < 0 || pos[2] < 0 || pos[3] < 0) {
      return out;
    }
    return pos;
  }();
  const float out_scale_x = static_cast<float>(output_w) / static_cast<float>(model_w);
  const float out_scale_y = static_cast<float>(output_h) / static_cast<float>(model_h);
  float lb_scale = 1.0f;
  float lb_pad_left = 0.0f;
  float lb_pad_top = 0.0f;
  if (use_letterbox_remap) {
    lb_scale = std::min(static_cast<float>(model_w) / static_cast<float>(output_w),
                        static_cast<float>(model_h) / static_cast<float>(output_h));
    const float resized_w = static_cast<float>(output_w) * lb_scale;
    const float resized_h = static_cast<float>(output_h) * lb_scale;
    lb_pad_left = 0.5f * (static_cast<float>(model_w) - resized_w);
    lb_pad_top = 0.5f * (static_cast<float>(model_h) - resized_h);
  }

  auto get_scale = [&](int h, int w) -> ScalePack& {
    for (auto& s : scales) {
      if (s.h == h && s.w == w)
        return s;
    }
    ScalePack s;
    s.h = h;
    s.w = w;
    s.stride = (h == 80) ? 8 : ((h == 40) ? 16 : ((h == 20) ? 32 : ((h > 0) ? (640 / h) : 0)));
    scales.push_back(std::move(s));
    return scales.back();
  };

  int mapped_heads = 0;
  bool packed_single_tensor_mode = false;
  if (tensors.size() == 1U && heads.size() > 1U) {
    const auto cpu = tensors.front().cpu().contiguous();
    std::vector<uint8_t> raw_bytes;
    if (extract_tensor_raw_bytes_local(cpu, raw_bytes) && !raw_bytes.empty()) {
      std::size_t expected_elems_total = 0U;
      for (const auto& h : heads) {
        expected_elems_total += static_cast<std::size_t>(h.h) * static_cast<std::size_t>(h.w) *
                                static_cast<std::size_t>(h.c);
      }
      if (expected_elems_total > 0U && (raw_bytes.size() % expected_elems_total) == 0U) {
        const std::size_t packed_elem_bytes = raw_bytes.size() / expected_elems_total;
        if (packed_elem_bytes == 1U || packed_elem_bytes == 2U || packed_elem_bytes == 4U) {
          std::size_t cursor = 0U;
          int packed_mapped = 0;
          bool packed_failed = false;
          for (const auto& head : heads) {
            const std::size_t elems = static_cast<std::size_t>(head.h) *
                                      static_cast<std::size_t>(head.w) *
                                      static_cast<std::size_t>(head.c);
            const std::size_t need = elems * packed_elem_bytes;
            if (need == 0U || cursor + need > raw_bytes.size()) {
              packed_failed = true;
              break;
            }
            std::vector<float> values;
            std::string decode_note;
            if (!decode_bytes_to_fp32_local(raw_bytes.data() + cursor, need, cpu.dtype, head,
                                            values, decode_note)) {
              packed_failed = true;
              break;
            }
            cursor += need;
            packed_mapped += 1;
            auto& sp = get_scale(head.h, head.w);
            if (head.is_bbox) {
              sp.bbox = std::move(values);
              sp.bbox_c = head.c;
              sp.has_bbox = true;
            } else if (head.is_class) {
              sp.cls = std::move(values);
              sp.cls_c = head.c;
              sp.has_cls = true;
            }
          }
          if (!packed_failed && packed_mapped > 0) {
            packed_single_tensor_mode = true;
            mapped_heads += packed_mapped;
            tensor_used[0] = true;
          }
        }
      }
    }
  }

  if (!packed_single_tensor_mode) {
    for (const auto& head : heads) {
      const std::size_t expected_elems = static_cast<std::size_t>(head.h) *
                                         static_cast<std::size_t>(head.w) *
                                         static_cast<std::size_t>(head.c);
      int chosen = -1;
      for (std::size_t ti = 0; ti < tensors.size(); ++ti) {
        if (tensor_used[ti]) {
          continue;
        }
        const auto cpu = tensors[ti].cpu().contiguous();
        const std::vector<uint8_t> bytes = cpu.copy_payload_bytes();
        const int elem_bytes = element_bytes_for_dtype_local(cpu.dtype);
        if (elem_bytes <= 0 || bytes.empty()) {
          continue;
        }
        const std::size_t elems = bytes.size() / static_cast<std::size_t>(elem_bytes);
        bool match = elems == expected_elems;
        const bool int8_meta = cpu.dtype == simaai::neat::TensorDType::Int8 ||
                               cpu.dtype == simaai::neat::TensorDType::UInt8;
        if (!match && int8_meta) {
          match = bytes.size() == expected_elems * sizeof(uint16_t) ||
                  bytes.size() == expected_elems * sizeof(float);
        }
        if (match) {
          chosen = static_cast<int>(ti);
          break;
        }
      }
      if (chosen < 0) {
        continue;
      }

      std::vector<float> values;
      std::string decode_note;
      if (!decode_tensor_to_fp32_local(tensors[static_cast<std::size_t>(chosen)], head, values,
                                       decode_note)) {
        continue;
      }
      tensor_used[static_cast<std::size_t>(chosen)] = true;
      mapped_heads += 1;
      auto& sp = get_scale(head.h, head.w);
      if (head.is_bbox) {
        sp.bbox = std::move(values);
        sp.bbox_c = head.c;
        sp.has_bbox = true;
      } else if (head.is_class) {
        sp.cls = std::move(values);
        sp.cls_c = head.c;
        sp.has_cls = true;
      }
    }
  }

  if (mapped_heads == 0) {
    if (note_out)
      *note_out = "mapped_heads=0 tensors=" + std::to_string(tensors.size()) + " " + head_note;
    return out;
  }

  struct Candidate {
    float x1 = 0.0f;
    float y1 = 0.0f;
    float x2 = 0.0f;
    float y2 = 0.0f;
    float score = 0.0f;
    int class_id = -1;
  };
  std::vector<Candidate> candidates;
  bool has_class_head = false;

  for (const auto& sp : scales) {
    if (!sp.has_bbox || sp.h <= 0 || sp.w <= 0 || sp.stride <= 0) {
      continue;
    }
    if (sp.bbox_c % 4 != 0 || sp.bbox_c < 16) {
      continue;
    }
    const int reg_max = sp.bbox_c / 4;
    const std::size_t cells = static_cast<std::size_t>(sp.h) * static_cast<std::size_t>(sp.w);
    if (sp.bbox.size() != cells * static_cast<std::size_t>(sp.bbox_c)) {
      continue;
    }
    if (sp.has_cls &&
        (sp.cls_c <= 0 || sp.cls.size() != cells * static_cast<std::size_t>(sp.cls_c))) {
      continue;
    }
    has_class_head = has_class_head || sp.has_cls;
    const float stride_x = static_cast<float>(model_w) / static_cast<float>(sp.w);
    const float stride_y = static_cast<float>(model_h) / static_cast<float>(sp.h);
    auto index_hwc_or_chw = [&](std::size_t cell, int channel, int channels) -> std::size_t {
      if (!decode_chw_layout) {
        return cell * static_cast<std::size_t>(channels) + static_cast<std::size_t>(channel);
      }
      const std::size_t cells_count =
          static_cast<std::size_t>(sp.h) * static_cast<std::size_t>(sp.w);
      return static_cast<std::size_t>(channel) * cells_count + cell;
    };

    std::vector<float> prob(static_cast<std::size_t>(reg_max), 0.0f);
    for (int y = 0; y < sp.h; ++y) {
      for (int x = 0; x < sp.w; ++x) {
        const std::size_t cell = static_cast<std::size_t>(y) * static_cast<std::size_t>(sp.w) +
                                 static_cast<std::size_t>(x);
        int best_class = -1;
        float best_score = 0.0f;
        if (sp.has_cls) {
          for (int c = 0; c < sp.cls_c; ++c) {
            const std::size_t cls_idx = index_hwc_or_chw(cell, c, sp.cls_c);
            const float score = sigmoid_local(sp.cls[cls_idx]);
            if (score > best_score) {
              best_score = score;
              best_class = c;
            }
          }
          if (best_class < 0 || best_score < score_threshold) {
            continue;
          }
        } else {
          best_class = 0;
        }

        float dist[4] = {0.0f, 0.0f, 0.0f, 0.0f};
        float bbox_quality = 1.0f;
        for (int side = 0; side < 4; ++side) {
          const int group = side_group_for_ltrb[static_cast<std::size_t>(side)];
          float vmax = -std::numeric_limits<float>::infinity();
          for (int k = 0; k < reg_max; ++k) {
            const int ch = group * reg_max + k;
            const std::size_t bbox_idx = index_hwc_or_chw(cell, ch, sp.bbox_c);
            vmax = std::max(vmax, sp.bbox[bbox_idx]);
          }
          float sum = 0.0f;
          for (int k = 0; k < reg_max; ++k) {
            const int ch = group * reg_max + k;
            const std::size_t bbox_idx = index_hwc_or_chw(cell, ch, sp.bbox_c);
            const float e = std::exp(sp.bbox[bbox_idx] - vmax);
            prob[static_cast<std::size_t>(k)] = e;
            sum += e;
          }
          float expect = 0.0f;
          if (sum > 0.0f) {
            float side_max_p = 0.0f;
            for (int k = 0; k < reg_max; ++k) {
              const float p = prob[static_cast<std::size_t>(k)] / sum;
              expect += static_cast<float>(k) * p;
              side_max_p = std::max(side_max_p, p);
            }
            bbox_quality *= side_max_p;
          }
          if (side == 0 || side == 2) {
            dist[side] = expect * stride_x * dfl_dist_scale;
          } else {
            dist[side] = expect * stride_y * dfl_dist_scale;
          }
        }
        if (!sp.has_cls) {
          (void)bbox_quality;
          best_score = 1.0f;
        }

        const float cx = (static_cast<float>(x) + grid_offset) * stride_x;
        const float cy = (static_cast<float>(y) + grid_offset) * stride_y;
        Candidate cand;
        cand.x1 = std::max(0.0f, std::min(cx - dist[0], static_cast<float>(model_w)));
        cand.y1 = std::max(0.0f, std::min(cy - dist[1], static_cast<float>(model_h)));
        cand.x2 = std::max(0.0f, std::min(cx + dist[2], static_cast<float>(model_w)));
        cand.y2 = std::max(0.0f, std::min(cy + dist[3], static_cast<float>(model_h)));
        cand.score = best_score;
        cand.class_id = best_class;
        if (cand.x2 > cand.x1 && cand.y2 > cand.y1) {
          candidates.push_back(cand);
        }
      }
    }
  }

  if (candidates.empty()) {
    if (note_out)
      *note_out = "no_candidates mapped_heads=" + std::to_string(mapped_heads);
    return out;
  }

  std::unordered_map<int, std::vector<int>> per_class_indices;
  for (int i = 0; i < static_cast<int>(candidates.size()); ++i) {
    per_class_indices[candidates[static_cast<std::size_t>(i)].class_id].push_back(i);
  }
  const float nms_score_threshold = has_class_head ? score_threshold : 0.0f;

  for (const auto& [cls, idxs] : per_class_indices) {
    std::vector<cv::Rect> rects;
    std::vector<float> scores;
    rects.reserve(idxs.size());
    scores.reserve(idxs.size());
    for (const int idx : idxs) {
      const auto& c = candidates[static_cast<std::size_t>(idx)];
      const int x = static_cast<int>(std::floor(c.x1));
      const int y = static_cast<int>(std::floor(c.y1));
      const int w = static_cast<int>(std::ceil(c.x2 - c.x1));
      const int h = static_cast<int>(std::ceil(c.y2 - c.y1));
      rects.emplace_back(x, y, std::max(1, w), std::max(1, h));
      scores.push_back(c.score);
    }

    std::vector<int> keep_local;
    cv::dnn::NMSBoxes(rects, scores, nms_score_threshold, iou_threshold, keep_local, 1.0f,
                      (topk > 0) ? topk : 0);
    for (const int ki : keep_local) {
      const auto& c = candidates[static_cast<std::size_t>(idxs[static_cast<std::size_t>(ki)])];
      float x1 = c.x1;
      float y1 = c.y1;
      float x2 = c.x2;
      float y2 = c.y2;
      if (use_letterbox_remap) {
        x1 = (x1 - lb_pad_left) / lb_scale;
        y1 = (y1 - lb_pad_top) / lb_scale;
        x2 = (x2 - lb_pad_left) / lb_scale;
        y2 = (y2 - lb_pad_top) / lb_scale;
      } else {
        x1 *= out_scale_x;
        y1 *= out_scale_y;
        x2 *= out_scale_x;
        y2 *= out_scale_y;
      }
      out.push_back(objdet::Box{std::max(0.0f, std::min(x1, static_cast<float>(output_w))),
                                std::max(0.0f, std::min(y1, static_cast<float>(output_h))),
                                std::max(0.0f, std::min(x2, static_cast<float>(output_w))),
                                std::max(0.0f, std::min(y2, static_cast<float>(output_h))), c.score,
                                cls});
    }
  }

  std::sort(out.begin(), out.end(),
            [](const objdet::Box& a, const objdet::Box& b) { return a.score > b.score; });
  if (topk > 0 && static_cast<int>(out.size()) > topk) {
    out.resize(static_cast<std::size_t>(topk));
  }
  if (note_out) {
    int top_class = -1;
    float top_score = 0.0f;
    float top_x1 = 0.0f;
    float top_y1 = 0.0f;
    float top_x2 = 0.0f;
    float top_y2 = 0.0f;
    if (!out.empty()) {
      top_class = out.front().class_id;
      top_score = out.front().score;
      top_x1 = out.front().x1;
      top_y1 = out.front().y1;
      top_x2 = out.front().x2;
      top_y2 = out.front().y2;
    }
    *note_out = "hostdecode_ok mapped_heads=" + std::to_string(mapped_heads) +
                " class_head=" + std::string(has_class_head ? "1" : "0") +
                " candidates=" + std::to_string(candidates.size()) +
                " boxes=" + std::to_string(out.size()) + " top_class=" + std::to_string(top_class) +
                " top_score=" + std::to_string(top_score) + " top_box=" + std::to_string(top_x1) +
                "," + std::to_string(top_y1) + "," + std::to_string(top_x2) + "," +
                std::to_string(top_y2) + " model_hw=" + std::to_string(model_w) + "x" +
                std::to_string(model_h) + " output_hw=" + std::to_string(output_w) + "x" +
                std::to_string(output_h) +
                " remap=" + std::string(use_letterbox_remap ? "letterbox" : "scale") +
                " layout=" + std::string(decode_chw_layout ? "CHW" : "HWC") +
                " grid_offset=" + std::to_string(grid_offset) +
                " dist_scale=" + std::to_string(dfl_dist_scale) + " side_order=" +
                std::string(side_group_for_ltrb[0] == 0 && side_group_for_ltrb[1] == 1 &&
                                    side_group_for_ltrb[2] == 2 && side_group_for_ltrb[3] == 3
                                ? "LTRB"
                                : "custom");
  }
  return out;
}

std::string boxdecode_dtype_token_from_tensor_local(const simaai::neat::Tensor& tensor) {
  switch (tensor.dtype) {
  case simaai::neat::TensorDType::BFloat16:
    return "BF16";
  case simaai::neat::TensorDType::Float32:
    return "FP32";
  case simaai::neat::TensorDType::Int8:
    return "INT8";
  case simaai::neat::TensorDType::UInt8:
    return "UINT8";
  case simaai::neat::TensorDType::Int16:
    return "INT16";
  case simaai::neat::TensorDType::UInt16:
    return "UINT16";
  case simaai::neat::TensorDType::Int32:
    return "INT32";
  case simaai::neat::TensorDType::Float64:
    return "FP64";
  default:
    return "";
  }
}

bool boxdecode_dims_from_tensor_local(const simaai::neat::Tensor& tensor, int* out_h, int* out_w,
                                      int* out_c) {
  if (!out_h || !out_w || !out_c) {
    return false;
  }
  std::vector<int64_t> dims = tensor.shape;
  if (dims.size() == 4U && dims.front() == 1) {
    dims.erase(dims.begin());
  }
  if (dims.size() != 3U) {
    return false;
  }
  if (tensor.layout == simaai::neat::TensorLayout::CHW) {
    *out_c = static_cast<int>(dims[0]);
    *out_h = static_cast<int>(dims[1]);
    *out_w = static_cast<int>(dims[2]);
  } else {
    *out_h = static_cast<int>(dims[0]);
    *out_w = static_cast<int>(dims[1]);
    *out_c = static_cast<int>(dims[2]);
  }
  return *out_h > 0 && *out_w > 0 && *out_c > 0;
}

std::string boxdecode_tensor_name_local(const simaai::neat::Tensor& tensor, std::size_t index) {
  if (!tensor.route.backend_name.empty()) {
    return tensor.route.backend_name;
  }
  if (!tensor.route.name.empty()) {
    return tensor.route.name;
  }
  if (!tensor.route.segment_name.empty()) {
    return tensor.route.segment_name;
  }
  return "input_tensor_" + std::to_string(index);
}

std::optional<simaai::neat::pipeline_internal::sima::BoxDecodeStaticContract>
build_standalone_boxdecode_contract_from_sample_local(const simaai::neat::Sample& sample,
                                                      std::string* error_message) {
  auto fail = [&](const std::string& message)
      -> std::optional<simaai::neat::pipeline_internal::sima::BoxDecodeStaticContract> {
    if (error_message) {
      *error_message = message;
    }
    return std::nullopt;
  };
  if (error_message) {
    error_message->clear();
  }

  std::vector<simaai::neat::Tensor> tensors =
      order_tensors_for_contract_check(tensors_in_sample(sample));
  if (tensors.empty()) {
    return fail("standalone boxdecode contract requires at least one output tensor");
  }

  simaai::neat::pipeline_internal::sima::BoxDecodeStaticContract contract;
  contract.decode_type = simaai::neat::BoxDecodeType::YoloV8;
  contract.quant_needed = false;
  contract.tess_needed = false;
  contract.score_activation =
      simaai::neat::pipeline_internal::sima::BoxDecodeScoreActivation::Sigmoid;
  contract.topk = canonical_boxdecode_options().top_k;
  contract.detection_threshold = canonical_boxdecode_options().detection_threshold;
  contract.nms_iou_threshold = canonical_boxdecode_options().nms_iou_threshold;
  contract.tensors.reserve(tensors.size());
  contract.tensor_names.reserve(tensors.size());
  contract.physical_inputs.reserve(tensors.size());
  contract.dq_scale.assign(tensors.size(), 1.0);
  contract.dq_zp.assign(tensors.size(), 0);

  for (std::size_t i = 0; i < tensors.size(); ++i) {
    const auto& tensor = tensors[i];
    int h = 0;
    int w = 0;
    int c = 0;
    if (!boxdecode_dims_from_tensor_local(tensor, &h, &w, &c)) {
      return fail("standalone boxdecode contract requires explicit tensor geometry for tensor " +
                  std::to_string(i) + " shape=" + shape_string(tensor.shape));
    }
    const std::string dtype = boxdecode_dtype_token_from_tensor_local(tensor);
    if (dtype.empty()) {
      return fail("standalone boxdecode contract requires explicit tensor dtype for tensor " +
                  std::to_string(i) + " dtype=" + dtype_name(tensor.dtype));
    }
    simaai::neat::pipeline_internal::sima::BoxDecodeTensorStaticContract entry;
    entry.input_shape = {h, w, c};
    entry.slice_shape = {h, w, c};
    entry.data_type = dtype;
    const std::string tensor_name = boxdecode_tensor_name_local(tensor, i);
    const std::uint64_t tensor_bytes =
        static_cast<std::uint64_t>(tensor.cpu().contiguous().copy_payload_bytes().size());
    entry.logical_name = tensor_name;
    entry.backend_name = tensor_name;
    entry.source_segment_name = tensor_name;
    entry.source_physical_index = static_cast<int>(i);
    entry.source_byte_offset = 0;
    entry.source_size_bytes = tensor_bytes;
    contract.tensors.push_back(std::move(entry));

    contract.tensor_names.push_back(tensor_name);

    simaai::neat::pipeline_internal::sima::BoxDecodePhysicalInputStaticContract physical;
    physical.name = tensor_name;
    physical.physical_index = static_cast<int>(i);
    physical.byte_offset = 0;
    physical.size_bytes = tensor_bytes;
    contract.physical_inputs.push_back(std::move(physical));
  }

  contract.input_dtype = contract.tensors.front().data_type;
  return contract;
}

std::vector<objdet::Box> run_hostdecode_boxes_on_sample_local(const simaai::neat::Sample& sample,
                                                              const simaai::neat::Model& model,
                                                              const cv::Mat& img_bgr,
                                                              std::string* note_out) {
  const ProbeResult probe = build_hostdecode_probe_from_model_local(model);
  return decode_yolov8_heads_with_opencv_nms_local(
      probe, sample, kFixedPreprocTargetWidth, kFixedPreprocTargetHeight, img_bgr.cols,
      img_bgr.rows, /*use_letterbox_remap=*/true, canonical_boxdecode_options().detection_threshold,
      canonical_boxdecode_options().nms_iou_threshold, canonical_boxdecode_options().top_k,
      note_out);
}

AccuracyResult run_hostdecode_accuracy_on_sample_local(const simaai::neat::Sample& sample,
                                                       const simaai::neat::Model& model,
                                                       const cv::Mat& img_bgr, const char* label) {
  AccuracyResult out;
  constexpr float kMinScore = 0.52f;
  constexpr float kMinIou = 0.30f;
  constexpr int kTopK = 100;

  std::string decode_note;
  const auto boxes = run_hostdecode_boxes_on_sample_local(sample, model, img_bgr, &decode_note);
  out.parsed_boxes = static_cast<int>(boxes.size());

  const std::vector<objdet::ExpectedBox> expected = objdet::expected_people_boxes();
  const objdet::MatchResult strict =
      objdet::match_expected_boxes(boxes, expected, kMinScore, kMinIou);
  if (strict.ok) {
    out.ok = true;
    out.note = std::string(label) + "_ok boxes=" + std::to_string(out.parsed_boxes) +
               " note=" + decode_note;
    return out;
  }

  float best_person_iou = 0.0f;
  int person_candidates = 0;
  for (const auto& box : boxes) {
    if (box.class_id != 0 || box.score < kMinScore) {
      continue;
    }
    person_candidates += 1;
    for (const auto& expected_box : expected) {
      if (expected_box.class_id != 0) {
        continue;
      }
      best_person_iou = std::max(
          best_person_iou, objdet::box_iou_xyxy(expected_box.x1, expected_box.y1, expected_box.x2,
                                                expected_box.y2, box.x1, box.y1, box.x2, box.y2));
    }
  }
  if (person_candidates > 0 && best_person_iou >= 0.25f) {
    out.ok = true;
    out.note = std::string(label) + "_coarse boxes=" + std::to_string(out.parsed_boxes) +
               " best_person_iou=" + std::to_string(best_person_iou) +
               " strict=" + sima_yolov8_test::sanitize_note(strict.note) + " note=" + decode_note;
    return out;
  }

  out.ok = false;
  out.note = std::string(label) + "_mismatch boxes=" + std::to_string(out.parsed_boxes) +
             " person_candidates=" + std::to_string(person_candidates) +
             " best_person_iou=" + std::to_string(best_person_iou) +
             " strict=" + sima_yolov8_test::sanitize_note(strict.note) + " note=" + decode_note;
  return out;
}

simaai::neat::Sample run_standalone_boxdecode_sample_local(const simaai::neat::Sample& stage_input,
                                                           float score_threshold,
                                                           float nms_iou_threshold, int topk,
                                                           int original_width, int original_height,
                                                           int model_width, int model_height) {
  const auto stage_tensor = first_tensor_in_sample(stage_input);
  require(stage_tensor.has_value(), "postrun_boxdecode: sample missing tensor payload");
  const std::string boxdecode_name = "matrix_post_boxdecode";

  simaai::neat::Graph post;
  post.add(simaai::neat::nodes::Input());
  post.add(simaai::neat::nodes::SimaBoxDecode(
      simaai::neat::BoxDecodeType::YoloV8, score_threshold, nms_iou_threshold, topk, boxdecode_name,
      original_width, original_height, model_width, model_height));
  post.add(simaai::neat::nodes::Output());

  simaai::neat::RunOptions run_opt;
  run_opt.output_memory = simaai::neat::OutputMemory::Owned;
  run_opt.queue_depth = 1;

  const int timeout_ms = default_model_run_timeout_ms();
  const int retry_timeout_ms = std::max(timeout_ms, 30000);
  auto run_once = [&](int attempt, int pull_timeout_ms) -> simaai::neat::Sample {
    auto runner = post.build(simaai::neat::Sample{stage_input}, run_opt);
    require(static_cast<bool>(runner), "postrun_boxdecode: runner build failed");
    require(runner.push(simaai::neat::Sample{stage_input}),
            "postrun_boxdecode: runner push failed");
    simaai::neat::Sample outs;
    try {
      outs = runner.pull_samples(pull_timeout_ms);
    } catch (const std::exception& ex) {
      throw std::runtime_error("postrun_boxdecode attempt=" + std::to_string(attempt) +
                               ": pull failed: " + ex.what() + "\nlast-error:\n" +
                               runner.last_error());
    }
    require(outs.size() == 1U, "postrun_boxdecode attempt=" + std::to_string(attempt) +
                                   ": expected exactly 1 sample, got " +
                                   std::to_string(outs.size()) + "\nlast-error:\n" +
                                   runner.last_error());
    return outs.front();
  };

  try {
    return run_once(1, timeout_ms);
  } catch (const std::exception& first) {
    std::cerr << "[WARN] postrun_boxdecode first attempt failed; retrying with timeout_ms="
              << retry_timeout_ms << " err=" << first.what() << "\n";
    try {
      return run_once(2, retry_timeout_ms);
    } catch (const std::exception& second) {
      throw std::runtime_error(std::string("postrun_boxdecode: retry failed after initial "
                                           "standalone decode failure\ninitial: ") +
                               first.what() + "\nretry: " + second.what());
    }
  }
}

AccuracyResult run_framework_boxdecode_accuracy(const simaai::neat::Sample& infer_sample,
                                                const simaai::neat::Model& model,
                                                const cv::Mat& img_bgr,
                                                BoxDecodeRunMode boxdecode_mode) {
  AccuracyResult out;
  constexpr float kMinScore = 0.52f;
  constexpr float kMinIou = 0.30f;
  constexpr int kTopK = 100;

  try {
    std::string decode_note;
    simaai::neat::Sample decoded;
    if (boxdecode_mode == BoxDecodeRunMode::Model) {
      decode_note = "boxdecode=model_output";
      decoded = infer_sample;
    } else {
      require_preprocess_meta_on_output_local(infer_sample, img_bgr.cols, img_bgr.rows,
                                              "framework_boxdecode_input");
      maybe_trace_sample_numeric_local("infer_sample_for_boxdecode", infer_sample);
      const auto spec = model.input_specs().front();
      const auto [model_w_pref, model_h_pref] =
          preferred_input_hw(model, RouteKind::PreInferPost, spec);
      const int model_width = model_w_pref > 0 ? model_w_pref : img_bgr.cols;
      const int model_height = model_h_pref > 0 ? model_h_pref : img_bgr.rows;
      decode_note = "boxdecode=standalone_post_graph";
      decoded = run_standalone_boxdecode_sample_local(
          infer_sample, canonical_boxdecode_options().detection_threshold,
          canonical_boxdecode_options().nms_iou_threshold, canonical_boxdecode_options().top_k,
          img_bgr.cols, img_bgr.rows, model_width, model_height);
    }

    std::vector<uint8_t> payload;
    std::string err;
    require(objdet::extract_bbox_payload(decoded, payload, err),
            "framework boxdecode: failed to extract bbox payload: " + err);

    const std::vector<objdet::Box> boxes =
        objdet::parse_boxes_strict(payload, img_bgr.cols, img_bgr.rows, kTopK, false);
    out.parsed_boxes = static_cast<int>(boxes.size());
    const std::vector<objdet::ExpectedBox> expected = objdet::expected_people_boxes();
    const objdet::MatchResult strict =
        objdet::match_expected_boxes(boxes, expected, kMinScore, kMinIou);
    if (strict.ok) {
      out.ok = true;
      out.note =
          "framework_boxdecode_ok boxes=" + std::to_string(out.parsed_boxes) + " " + decode_note;
      return out;
    }

    float best_person_iou = 0.0f;
    int person_candidates = 0;
    for (const auto& box : boxes) {
      if (box.class_id != 0 || box.score < kMinScore) {
        continue;
      }
      person_candidates += 1;
      for (const auto& expected_box : expected) {
        if (expected_box.class_id != 0) {
          continue;
        }
        best_person_iou = std::max(
            best_person_iou, objdet::box_iou_xyxy(expected_box.x1, expected_box.y1, expected_box.x2,
                                                  expected_box.y2, box.x1, box.y1, box.x2, box.y2));
      }
    }
    if (person_candidates > 0 && best_person_iou >= 0.25f) {
      out.ok = true;
      out.note = "framework_boxdecode_coarse boxes=" + std::to_string(out.parsed_boxes) +
                 " best_person_iou=" + std::to_string(best_person_iou) +
                 " strict=" + sima_yolov8_test::sanitize_note(strict.note) + " " + decode_note;
      return out;
    }

    out.ok = false;
    out.note = "framework_boxdecode_mismatch " + sima_yolov8_test::sanitize_note(strict.note) +
               " " + decode_note;
    return out;
  } catch (const std::exception& ex) {
    out.ok = false;
    out.note = "framework_boxdecode_error=" + sima_yolov8_test::sanitize_note(ex.what());
    return out;
  }
}

bool try_host_dequant_preview(const simaai::neat::Tensor& tensor, std::vector<float>& preview,
                              std::string& note) {
  if (!tensor_is_int8_like(tensor)) {
    note = "tensor is not quantized int8/uint8";
    return false;
  }

  float scale = 0.0f;
  int32_t zero_point = 0;
  if (tensor.semantic.quant.has_value()) {
    const auto& q = *tensor.semantic.quant;
    if (q.scale > 0.0f) {
      scale = q.scale;
      zero_point = q.zero_point;
    } else if (!q.scales.empty() && q.scales[0] > 0.0f) {
      scale = q.scales[0];
      if (!q.zero_points.empty())
        zero_point = q.zero_points[0];
    }
  }
  if (scale <= 0.0f) {
    note = "quant metadata unavailable (missing scale/zero-point)";
    return false;
  }

  const auto cpu = tensor.cpu().contiguous();
  const std::vector<uint8_t> bytes = cpu.copy_payload_bytes();
  if (bytes.empty()) {
    note = "quantized tensor payload is empty";
    return false;
  }

  const size_t max_elems = std::min<size_t>(bytes.size(), 256U);
  preview.clear();
  preview.reserve(max_elems);

  if (cpu.dtype == simaai::neat::TensorDType::Int8) {
    const auto* ptr = reinterpret_cast<const int8_t*>(bytes.data());
    for (size_t i = 0; i < max_elems; ++i) {
      preview.push_back((static_cast<float>(ptr[i]) - static_cast<float>(zero_point)) / scale);
    }
  } else {
    for (size_t i = 0; i < max_elems; ++i) {
      preview.push_back((static_cast<float>(bytes[i]) - static_cast<float>(zero_point)) / scale);
    }
  }

  note = "host dequant preview generated";
  return true;
}

RouteKind plan_route(const ProbeResult& probe) {
  if (probe.has_post_adapter) {
    return RouteKind::PreInferPost;
  }

  // User caveat contract: no *_postproc.json => tessellation is within MLA.
  // Use BF16-only infer path only when MLA input contract is BF16 and there is no
  // preprocess stage to absorb raw video input.
  if (probe.tess_within_mla && probe.mla_input_bf16 && !probe.has_pre_adapter) {
    return RouteKind::InferOnlyBf16NoPost;
  }

  return RouteKind::QuantizedNoPostFallback;
}

ProbeResult probe_model(const fs::path& tar) {
  ProbeResult probe;
  probe.tar_path = tar.string();
  probe.model_id = tar.filename().string();
  probe.evidence.push_back("tar=" + probe.tar_path);

  simaai::neat::Model bootstrap_model = build_model_for_case(probe);

  std::string mla_cfg = bootstrap_model.find_config_path_by_processor("MLA");
  if (mla_cfg.empty()) {
    mla_cfg = bootstrap_model.find_config_path_by_plugin("processmla");
  }
  if (!mla_cfg.empty()) {
    probe.etc_dir = fs::path(mla_cfg).parent_path();
    probe.has_postproc_config = has_json_suffix(probe.etc_dir, "_postproc.json");
    probe.has_dequant_config = has_json_suffix(probe.etc_dir, "_dequantize.json");
    probe.has_pipeline_sequence = fs::exists(probe.etc_dir / "pipeline_sequence.json");
  }

  probe.tess_within_mla = !probe.has_postproc_config;
  probe.evidence.push_back(std::string("has_postproc_config=") +
                           (probe.has_postproc_config ? "1" : "0"));
  probe.evidence.push_back(std::string("has_dequant_config=") +
                           (probe.has_dequant_config ? "1" : "0"));
  probe.evidence.push_back(std::string("has_pipeline_sequence=") +
                           (probe.has_pipeline_sequence ? "1" : "0"));
  probe.evidence.push_back(std::string("tess_within_mla=") + (probe.tess_within_mla ? "1" : "0"));

  simaai::neat::Model model = build_model_for_case(probe);

  const auto pre_nodes = simaai::neat::internal::ModelAccess::build_public_preprocess_nodes(model);
  probe.has_pre_adapter = !pre_nodes.empty();
  probe.pre_kind = detect_pre_kind(pre_nodes);
  probe.evidence.push_back(std::string("has_pre_adapter=") + (probe.has_pre_adapter ? "1" : "0"));
  probe.evidence.push_back("pre_kind=" + pre_kind_name(probe.pre_kind));

  const auto post_nodes =
      simaai::neat::internal::ModelAccess::build_public_postprocess_nodes(model);
  probe.has_post_adapter = !post_nodes.empty();
  probe.post_kind = detect_post_kind(post_nodes);
  probe.evidence.push_back(std::string("has_post_adapter=") + (probe.has_post_adapter ? "1" : "0"));
  probe.evidence.push_back("post_kind=" + post_kind_name(probe.post_kind));
  probe.evidence.push_back("post_nodes=" + join_strings(node_kinds(post_nodes)));

  const auto infer_nodes = simaai::neat::internal::ModelAccess::build_public_inference_nodes(model);
  const auto mla_input = rendered_stage_query::mla_input_info_from_nodes(infer_nodes);
  const auto mla_output = rendered_stage_query::mla_output_info_from_nodes(infer_nodes);

  probe.mla_input_dtype_raw = mla_input.input_dtype;
  probe.mla_output_dtype_raw = mla_output.data_type;
  probe.mla_input_media_type = mla_input.input_media_type;

  probe.mla_input_bf16 = dtype_is_bf16_like(probe.mla_input_dtype_raw);
  probe.mla_input_int8 = dtype_is_int8_like(probe.mla_input_dtype_raw);
  probe.mla_output_bf16 = dtype_is_bf16_like(probe.mla_output_dtype_raw);
  probe.mla_output_int8 = dtype_is_int8_like(probe.mla_output_dtype_raw);
  probe.terminal_output_kind = detect_terminal_output_kind(post_nodes, probe);

  const auto input_spec = model.input_specs().front();
  bool only_uint8 = !input_spec.dtypes.empty();
  for (const auto dt : input_spec.dtypes) {
    if (dt != simaai::neat::TensorDType::UInt8) {
      only_uint8 = false;
      break;
    }
  }
  probe.input_spec_tensor_mode = !input_spec.dtypes.empty() && !only_uint8;

  std::string input_spec_desc;
  for (size_t i = 0; i < input_spec.dtypes.size(); ++i) {
    if (i)
      input_spec_desc.push_back(',');
    input_spec_desc += dtype_name(input_spec.dtypes[i]);
  }
  if (input_spec_desc.empty())
    input_spec_desc = "<none>";
  probe.evidence.push_back("input_spec_dtypes=" + input_spec_desc);
  probe.evidence.push_back("input_spec_tensor_mode=" +
                           std::string(probe.input_spec_tensor_mode ? "1" : "0"));

  probe.evidence.push_back("mla_input_dtype=" + probe.mla_input_dtype_raw);
  probe.evidence.push_back("mla_output_dtype=" + probe.mla_output_dtype_raw);
  probe.evidence.push_back("mla_input_media_type=" + probe.mla_input_media_type);
  probe.evidence.push_back("terminal_output_kind=" +
                           terminal_output_kind_name(probe.terminal_output_kind));

  return probe;
}

ExecutionResult execute_case(const ModelCase& c, const cv::Mat& img_bgr) {
  ExecutionResult out;
  out.route = c.route;
  out.model_id = c.probe.model_id;

  simaai::neat::pipeline_internal::reset_tensor_io_stats();
  const auto tensor_io_before = simaai::neat::pipeline_internal::snapshot_tensor_io_stats();
  auto model = build_model_for_case(c.probe);
  const auto sample = run_model_sample(img_bgr, model, c.route);
  const auto tensor_io_after = simaai::neat::pipeline_internal::snapshot_tensor_io_stats();
  out.tensor_io = tensor_io_delta(tensor_io_before, tensor_io_after);
  const auto tensors = tensors_in_sample(sample);

  if (c.probe.terminal_output_kind == TerminalOutputKind::BoxDecodePayload) {
    std::vector<uint8_t> payload;
    std::string err;
    require(objdet::extract_bbox_payload(sample, payload, err),
            "variant route: model.run produced no bbox payload: " + err);
    out.output_signature = "BBoxPayload [" + std::to_string(payload.size()) + "B]";
    out.ok = true;
    out.note = "post boxdecode route completed";
    return out;
  }

  require(!tensors.empty(), "variant route: model.run produced no tensor output");
  require_tensor_outputs_match_model_contract(model, tensors);

  const simaai::neat::Tensor tensor = tensors.front();
  out.output_signature = dtype_name(tensor.dtype) + " " + shape_string(tensor.shape);

  if (c.route == RouteKind::PreInferPost) {
    for (const auto& field_tensor : tensors) {
      require(!tensor_is_int8_like(field_tensor),
              "post route output still looks quantized (expected dequantized tensor)");
    }
    out.ok = true;
    out.note = "post adapter route completed";
    return out;
  }

  if (c.route == RouteKind::InferOnlyBf16NoPost) {
    const bool bf16_or_fp32 =
        tensor_is_bf16_like(tensor) || tensor.dtype == simaai::neat::TensorDType::Float32;
    const bool int8_like = tensor_is_int8_like(tensor);
    require(bf16_or_fp32 || int8_like,
            "infer-only BF16 route produced unsupported output tensor dtype");
    std::vector<float> bf16_preview;
    std::string bf16_note;
    bool bf16_ok = false;
    if (bf16_or_fp32) {
      bf16_ok = try_bf16_to_fp32_preview(tensor, bf16_preview, bf16_note);
    } else if (int8_like) {
      std::vector<float> dequant_preview;
      std::string dequant_note;
      const bool dequant_ok = try_host_dequant_preview(tensor, dequant_preview, dequant_note);
      bf16_note = std::string("int8_output host_dequant=") + (dequant_ok ? "1" : "0") +
                  " note=" + dequant_note;
    }
    out.ok = true;
    out.note = "infer-only BF16 route completed; bf16_to_fp32=" + std::string(bf16_ok ? "1" : "0") +
               " note=" + (bf16_note.empty() ? "none" : bf16_note);
    return out;
  }

  std::vector<float> preview;
  std::string dequant_note;
  if (tensor_is_int8_like(tensor)) {
    out.host_dequant_ready = try_host_dequant_preview(tensor, preview, dequant_note);
    out.note = std::string("quantized fallback route completed; host_dequant=") +
               (out.host_dequant_ready ? "1" : "0") + " note=" + dequant_note;
  } else {
    out.host_dequant_ready = false;
    out.note = "no-post fallback route completed with non-quantized output";
  }
  out.ok = true;
  return out;
}

int route_priority(const ModelCase& c) {
  if (c.route == RouteKind::InferOnlyBf16NoPost)
    return 0;
  if (c.route == RouteKind::PreInferPost && c.probe.mla_input_int8)
    return 1;
  if (c.route == RouteKind::QuantizedNoPostFallback)
    return 2;
  return 3;
}

fs::path find_repo_root(const fs::path& start) {
  fs::path cur = start;
  for (int i = 0; i < 8; ++i) {
    if (fs::exists(cur / "CMakeLists.txt") && fs::exists(cur / "tests"))
      return cur;
    if (!cur.has_parent_path())
      break;
    cur = cur.parent_path();
  }
  return start;
}

fs::path resolve_variants_dir(const fs::path& root, int argc, char** argv) {
  fs::path variants;
  for (int i = 1; i < argc; ++i) {
    const std::string arg(argv[i]);
    if (arg.rfind("--variants-dir=", 0) == 0) {
      variants = fs::path(arg.substr(std::string("--variants-dir=").size()));
    } else if (arg == "--variants-dir" && (i + 1) < argc) {
      variants = fs::path(argv[++i]);
    }
  }
  if (!variants.empty())
    return variants;

  const char* env_dir = std::getenv("SIMA_YOLOV8_VARIANTS_DIR");
  if (env_dir && *env_dir) {
    return fs::path(env_dir);
  }
  const fs::path repo_tmp = root / "tmp" / "yolov8n_drive";
  std::error_code ec;
  if (fs::exists(repo_tmp, ec)) {
    return repo_tmp;
  }
  const fs::path core_tmp = root / "core" / "tmp" / "yolov8n_drive";
  if (fs::exists(core_tmp, ec)) {
    return core_tmp;
  }
  return repo_tmp;
}

std::optional<fs::path> resolve_single_model_tar(int argc, char** argv) {
  for (int i = 1; i < argc; ++i) {
    const std::string arg(argv[i]);
    if (arg.rfind("--single-model=", 0) == 0) {
      const auto raw = arg.substr(std::string("--single-model=").size());
      if (!raw.empty()) {
        return fs::path(raw);
      }
      return std::nullopt;
    }
    if (arg == "--single-model" && (i + 1) < argc) {
      const std::string raw(argv[++i]);
      if (!raw.empty()) {
        return fs::path(raw);
      }
      return std::nullopt;
    }
  }
  const char* env_model = std::getenv("SIMA_YOLOV8_SINGLE_MODEL");
  if (env_model && *env_model) {
    return fs::path(env_model);
  }
  return std::nullopt;
}

int resolve_frames(int argc, char** argv) {
  for (int i = 1; i < argc; ++i) {
    const std::string arg(argv[i]);
    if (arg.rfind("--frames=", 0) == 0) {
      const int v = std::atoi(arg.substr(std::string("--frames=").size()).c_str());
      return v > 0 ? v : 1;
    }
    if (arg == "--frames" && (i + 1) < argc) {
      const int v = std::atoi(argv[++i]);
      return v > 0 ? v : 1;
    }
  }
  if (const char* e = std::getenv("SIMA_YOLOV8_MATRIX_FRAMES"); e && *e) {
    const int v = std::atoi(e);
    return v > 0 ? v : 1;
  }
  return 1;
}

std::string resolve_processcvu_run_target(int argc, char** argv) {
  for (int i = 1; i < argc; ++i) {
    const std::string arg(argv[i]);
    if (arg.rfind("--processcvu-run-target=", 0) == 0) {
      return normalize_processcvu_run_target_token(
          arg.substr(std::string("--processcvu-run-target=").size()));
    }
    if (arg == "--processcvu-run-target" && (i + 1) < argc) {
      return normalize_processcvu_run_target_token(argv[++i]);
    }
  }
  return "AUTO";
}

bool has_flag(int argc, char** argv, const std::string& flag) {
  for (int i = 1; i < argc; ++i) {
    if (argv[i] && flag == argv[i]) {
      return true;
    }
  }
  return false;
}

int resolve_async_queue_depth(int argc, char** argv) {
  for (int i = 1; i < argc; ++i) {
    const std::string arg(argv[i]);
    std::string raw;
    if (arg.rfind("--async-queue-depth=", 0) == 0) {
      raw = arg.substr(std::string("--async-queue-depth=").size());
    } else if (arg == "--async-queue-depth" && (i + 1) < argc) {
      raw = argv[++i];
    }
    if (!raw.empty()) {
      const int v = std::atoi(raw.c_str());
      if (v <= 0 || v > 1024) {
        throw std::runtime_error("invalid --async-queue-depth: " + raw);
      }
      return v;
    }
  }
  // Fast async is now the route-matrix default, so use the same deep
  // internal queue default that used to be supplied by the focused
  // throughput command line.  CLI still overrides this for experiments.
  return 64;
}

bool resolve_processmla_defer_output_invalidate(int argc, char** argv) {
  return has_flag(argc, argv, "--processmla-defer-output-invalidate") ||
         has_flag(argc, argv, "--mla-defer-output-invalidate");
}

std::string resolve_prepared_runner_mode(int argc, char** argv) {
  for (int i = 1; i < argc; ++i) {
    const std::string arg(argv[i]);
    if (arg.rfind("--prepared-runner-mode=", 0) == 0) {
      return arg.substr(std::string("--prepared-runner-mode=").size());
    }
    if (arg == "--prepared-runner-mode" && (i + 1) < argc) {
      return argv[++i];
    }
    if (arg == "--prepared-runner-dequant") {
      return "dequant";
    }
  }
  return {};
}

int resolve_prepared_runner_ring_depth(int argc, char** argv) {
  for (int i = 1; i < argc; ++i) {
    const std::string arg(argv[i]);
    std::string raw;
    if (arg.rfind("--prepared-runner-ring-depth=", 0) == 0) {
      raw = arg.substr(std::string("--prepared-runner-ring-depth=").size());
    } else if (arg == "--prepared-runner-ring-depth" && (i + 1) < argc) {
      raw = argv[++i];
    }
    if (!raw.empty()) {
      const int v = std::atoi(raw.c_str());
      if (v <= 0 || v > 64) {
        throw std::runtime_error("invalid --prepared-runner-ring-depth: " + raw);
      }
      return v;
    }
  }
  return 0;
}

std::string resolve_prepared_runner_dequant_flags(int argc, char** argv) {
  for (int i = 1; i < argc; ++i) {
    const std::string arg(argv[i]);
    if (arg.rfind("--prepared-runner-dequant-flags=", 0) == 0) {
      return arg.substr(std::string("--prepared-runner-dequant-flags=").size());
    }
    if (arg == "--prepared-runner-dequant-flags" && (i + 1) < argc) {
      return argv[++i];
    }
  }
  return {};
}

void apply_prepared_runner_options(const std::string& mode, int ring_depth, bool profile,
                                   const std::string& dequant_flags,
                                   simaai::neat::Model::Options* model_options,
                                   simaai::neat::Model::RouteOptions* route_options) {
  if (mode.empty() && ring_depth <= 0 && !profile && dequant_flags.empty()) {
    return;
  }
  if (model_options) {
    model_options->prepared_runner.mode = mode;
    model_options->prepared_runner.ring_depth = ring_depth;
    model_options->prepared_runner.profile = profile;
    model_options->prepared_runner.dequant_flags = dequant_flags;
  }
  if (route_options) {
    route_options->prepared_runner.mode = mode;
    route_options->prepared_runner.ring_depth = ring_depth;
    route_options->prepared_runner.profile = profile;
    route_options->prepared_runner.dequant_flags = dequant_flags;
  }
}

struct AsyncSelection {
  bool processcvu = true;
  bool processmla = true;
};

AsyncSelection resolve_async_selection(int argc, char** argv) {
  // Fast async (processcvu + processmla) is the default path.  Keep
  // explicit flags for readability/backward compatibility, and provide
  // --sync/--no-fast-async for focused control comparisons.
  AsyncSelection sel;
  if (has_flag(argc, argv, "--sync") || has_flag(argc, argv, "--no-fast-async")) {
    sel.processcvu = false;
    sel.processmla = false;
  }
  if (has_flag(argc, argv, "--fast-async") || has_flag(argc, argv, "--safe-async") ||
      has_flag(argc, argv, "--both-async")) {
    sel.processcvu = true;
    sel.processmla = true;
  }
  if (has_flag(argc, argv, "--processcvu-async") || has_flag(argc, argv, "--cvu-async") ||
      has_flag(argc, argv, "--processcvu-async-only") || has_flag(argc, argv, "--cvu-async-only")) {
    sel.processcvu = true;
  }
  if (has_flag(argc, argv, "--processmla-async") || has_flag(argc, argv, "--mla-async") ||
      has_flag(argc, argv, "--processmla-async-only") || has_flag(argc, argv, "--mla-async-only")) {
    sel.processmla = true;
  }
  if (has_flag(argc, argv, "--processcvu-async-only") || has_flag(argc, argv, "--cvu-async-only")) {
    sel.processmla = false;
  }
  if (has_flag(argc, argv, "--processmla-async-only") || has_flag(argc, argv, "--mla-async-only")) {
    sel.processcvu = false;
  }
  return sel;
}

std::string async_selection_name(const AsyncSelection& sel) {
  if (sel.processcvu && sel.processmla)
    return "both";
  if (sel.processcvu)
    return "processcvu";
  if (sel.processmla)
    return "processmla";
  return "sync";
}

void apply_async_options(const AsyncSelection& async, int async_queue_depth,
                         simaai::neat::Model::Options* model_options,
                         simaai::neat::Model::RouteOptions* route_options) {
  if (model_options) {
    model_options->processcvu.async = async.processcvu;
    model_options->processmla.async = async.processmla;
    if (async_queue_depth > 0) {
      model_options->async_queue_depth = async_queue_depth;
    }
  }
  if (route_options) {
    route_options->processcvu.async = async.processcvu;
    route_options->processmla.async = async.processmla;
    if (async_queue_depth > 0) {
      route_options->async_queue_depth = async_queue_depth;
    }
  }
}

void apply_processmla_options(bool defer_output_invalidate,
                              simaai::neat::Model::Options* model_options,
                              simaai::neat::Model::RouteOptions* route_options) {
  if (!defer_output_invalidate) {
    return;
  }
  if (model_options) {
    model_options->processmla.defer_output_invalidate = true;
  }
  if (route_options) {
    route_options->processmla.defer_output_invalidate = true;
  }
}

std::string resolve_processcvu_placement(int argc, char** argv) {
  for (int i = 1; i < argc; ++i) {
    const std::string arg(argv[i]);
    if (arg.rfind("--processcvu-placement=", 0) == 0) {
      return arg.substr(std::string("--processcvu-placement=").size());
    }
    if (arg == "--processcvu-placement" && (i + 1) < argc) {
      return argv[++i];
    }
  }
  return {};
}

void apply_processcvu_placement(const std::string& placement,
                                simaai::neat::ProcessCvuOptions* processcvu) {
  if (!processcvu || placement.empty() || placement == "default" || placement == "legacy") {
    return;
  }
  auto normalize_device = [](std::string value) -> std::string {
    std::transform(value.begin(), value.end(), value.begin(),
                   [](unsigned char c) { return static_cast<char>(std::toupper(c)); });
    if (value == "EV" || value == "EVXX" || value == "CVU")
      return "EV74";
    if (value == "CPU" || value == "APU")
      return "A65";
    if (value == "EV74" || value == "A65" || value == "AUTO")
      return value;
    return {};
  };
  auto set_pair = [&](std::string pre, std::string post) {
    pre = normalize_device(std::move(pre));
    post = normalize_device(std::move(post));
    if (pre.empty() || post.empty()) {
      throw std::runtime_error("invalid --processcvu-placement device: " + placement);
    }
    processcvu->pre_run_target = pre;
    processcvu->post_run_target = post;
  };

  if (placement == "mixed_pre_ev74_post_a65" || placement == "graph222_ev74_graph223_a65") {
    set_pair("EV74", "A65");
    return;
  }
  if (placement == "pre_ev74_post_ev74" || placement == "all_ev74") {
    set_pair("EV74", "EV74");
    return;
  }
  if (placement == "pre_a65_post_ev74") {
    set_pair("A65", "EV74");
    return;
  }
  if (placement == "pre_a65_post_a65" || placement == "all_a65") {
    set_pair("A65", "A65");
    return;
  }
  // Compact model/route option syntax requested by the runtime work:
  //   --processcvu-placement=pre=ev74,post=a65
  //   --processcvu-placement=pre:a65,post:ev74
  if (placement.rfind("pre", 0) == 0) {
    std::string pre;
    std::string post;
    std::size_t start = 0;
    while (start <= placement.size()) {
      const std::size_t comma = placement.find(',', start);
      const std::string token =
          placement.substr(start, comma == std::string::npos ? std::string::npos : comma - start);
      const std::size_t sep = token.find_first_of("=:");
      if (sep != std::string::npos) {
        std::string key = token.substr(0, sep);
        std::string val = token.substr(sep + 1);
        std::transform(key.begin(), key.end(), key.begin(),
                       [](unsigned char c) { return static_cast<char>(std::tolower(c)); });
        if (key == "pre")
          pre = val;
        if (key == "post")
          post = val;
      }
      if (comma == std::string::npos)
        break;
      start = comma + 1;
    }
    if (!pre.empty() && !post.empty()) {
      set_pair(pre, post);
      return;
    }
  }
  throw std::runtime_error("invalid --processcvu-placement: " + placement);
}

BoxDecodeRunMode resolve_boxdecode_run_mode(int argc, char** argv) {
  BoxDecodeRunMode mode = BoxDecodeRunMode::NoModel;
  for (int i = 1; i < argc; ++i) {
    const std::string arg(argv[i]);
    if (arg == "--model-boxdecode") {
      mode = BoxDecodeRunMode::Model;
    } else if (arg == "--no-model-boxdecode") {
      mode = BoxDecodeRunMode::NoModel;
    }
  }
  return mode;
}

std::string tensor_layout_name(simaai::neat::TensorLayout layout) {
  switch (layout) {
  case simaai::neat::TensorLayout::Unknown:
    return "Unknown";
  case simaai::neat::TensorLayout::HWC:
    return "HWC";
  case simaai::neat::TensorLayout::CHW:
    return "CHW";
  case simaai::neat::TensorLayout::HW:
    return "HW";
  }
  return "Unknown";
}

std::vector<fs::path> collect_model_packs(const fs::path& variants_dir) {
  std::vector<fs::path> packs;
  std::error_code ec;
  if (!fs::exists(variants_dir, ec) || !fs::is_directory(variants_dir, ec))
    return packs;
  for (const auto& entry : fs::directory_iterator(variants_dir, ec)) {
    if (ec)
      break;
    if (!entry.is_regular_file())
      continue;
    const auto path = entry.path();
    if (path.extension() != ".gz")
      continue;
    if (path.filename().string().find(".tar.gz") == std::string::npos)
      continue;
    packs.push_back(path);
  }
  std::sort(packs.begin(), packs.end());
  return packs;
}

void validate_model_init_for_tar(const fs::path& tar) {
  const auto opt = default_model_options();
  simaai::neat::Model model(tar.string(), opt);
  (void)model.info();
  (void)model.input_specs();
  (void)model.preprocess();
  (void)model.inference();
  (void)model.postprocess();
}

} // namespace

int main(int argc, char** argv) {
#if !defined(SIMA_WITH_OPENCV)
  (void)argc;
  (void)argv;
  return skip_long_test(
      "OpenCV required for graph_migration_legacy_yolov8_variant_route_matrix_test");
#else
  try {
    simaai::neat::gst_init_once();
    if (!simaai::neat::element_exists("neatprocessmla")) {
      return skip_long_test("missing NEAT MLA plugin (neatprocessmla)");
    }
    if (!simaai::neat::element_exists("neatprocesscvu")) {
      return skip_long_test("missing NEAT CVU plugin (neatprocesscvu)");
    }
    if (!simaai::neat::element_exists("neatobjectdecode")) {
      return skip_long_test("missing NEAT objectdecode plugin (neatobjectdecode)");
    }

    fs::path root = find_repo_root(fs::current_path());
    std::vector<fs::path> packs;
    const std::optional<fs::path> single_model_tar = resolve_single_model_tar(argc, argv);
    const std::string processcvu_run_target = resolve_processcvu_run_target(argc, argv);
    const std::string processcvu_placement = resolve_processcvu_placement(argc, argv);
    const BoxDecodeRunMode boxdecode_mode = resolve_boxdecode_run_mode(argc, argv);
    const AsyncSelection async_selection = resolve_async_selection(argc, argv);
    const int async_queue_depth = resolve_async_queue_depth(argc, argv);
    const bool processmla_defer_output_invalidate =
        resolve_processmla_defer_output_invalidate(argc, argv);
    const std::string prepared_runner_mode = resolve_prepared_runner_mode(argc, argv);
    const int prepared_runner_ring_depth = resolve_prepared_runner_ring_depth(argc, argv);
    const bool prepared_runner_profile = has_flag(argc, argv, "--prepared-runner-profile");
    const std::string prepared_runner_dequant_flags =
        resolve_prepared_runner_dequant_flags(argc, argv);
    const int frames = resolve_frames(argc, argv);
    if (single_model_tar.has_value()) {
      const fs::path tar = fs::absolute(*single_model_tar);
      require(fs::exists(tar), "single model tar not found: " + tar.string());
      packs.push_back(tar);
    } else {
      fs::path variants_dir = resolve_variants_dir(root, argc, argv);
      packs = collect_model_packs(variants_dir);
      if (packs.empty()) {
        // Mandatory test — download the six yolov8n variants on demand via
        // the shared helper (uses sima-cli for the OAuth-gated
        // docs.sima.ai endpoint) and rescan.
        variants_dir = sima_yolov8_test::ensure_yolov8n_drive_variants(root);
        packs = collect_model_packs(variants_dir);
      }
      require(!packs.empty(), "no model packs found in " + variants_dir.string() +
                                  " even after download attempt; check sima-cli login and "
                                  "SIMA_YOLOV8N_VARIANTS_BASE_URL");
    }

    const cv::Mat img_bgr = sima_yolov8_test::load_people_image_or_skip(root);
    int failures = 0;

    for (const auto& tar : packs) {
      try {
        auto model_options = canonical_model_options(boxdecode_mode);
        if (boxdecode_mode == BoxDecodeRunMode::Model) {
          model_options.boxdecode_original_width = img_bgr.cols;
          model_options.boxdecode_original_height = img_bgr.rows;
        }
        apply_processcvu_placement(processcvu_placement, &model_options.processcvu);
        apply_async_options(async_selection, async_queue_depth, &model_options, nullptr);
        apply_processmla_options(processmla_defer_output_invalidate, &model_options, nullptr);
        apply_prepared_runner_options(prepared_runner_mode, prepared_runner_ring_depth,
                                      prepared_runner_profile, prepared_runner_dequant_flags,
                                      &model_options, nullptr);
        simaai::neat::Model model(tar.string(), model_options);
        simaai::neat::Model::RouteOptions route_opt;
        route_opt.processcvu_requested_run_target = processcvu_run_target;
        apply_processcvu_placement(processcvu_placement, &route_opt.processcvu);
        apply_async_options(async_selection, async_queue_depth, nullptr, &route_opt);
        apply_processmla_options(processmla_defer_output_invalidate, nullptr, &route_opt);
        apply_prepared_runner_options(prepared_runner_mode, prepared_runner_ring_depth,
                                      prepared_runner_profile, prepared_runner_dequant_flags,
                                      nullptr, &route_opt);
        (void)model.info();
        (void)model.input_specs();
        (void)model.inference();
        std::cout << "MODEL_INIT_OK model=" << tar.filename().string()
                  << " backend=" << processcvu_run_target << " placement="
                  << (processcvu_placement.empty() ? "default" : processcvu_placement)
                  << " async_mode=" << async_selection_name(async_selection)
                  << " processcvu_async=" << (async_selection.processcvu ? 1 : 0)
                  << " processmla_async=" << (async_selection.processmla ? 1 : 0)
                  << " async_queue_depth=" << async_queue_depth << " prepared_runner_mode="
                  << (prepared_runner_mode.empty() ? "off" : prepared_runner_mode)
                  << " prepared_runner_dequant_flags="
                  << (prepared_runner_dequant_flags.empty() ? "default"
                                                            : prepared_runner_dequant_flags)
                  << " frames=" << frames
                  << " boxdecode_mode=" << boxdecode_run_mode_name(boxdecode_mode) << "\n";

        simaai::neat::pipeline_internal::reset_tensor_io_stats();
        const auto tensor_io_before = simaai::neat::pipeline_internal::snapshot_tensor_io_stats();
        const auto run_t0 = std::chrono::steady_clock::now();
        const simaai::neat::Sample infer_sample =
            run_canonical_model_sample(img_bgr, model, route_opt, frames);
        const auto run_t1 = std::chrono::steady_clock::now();
        const double run_ms = std::chrono::duration<double, std::milli>(run_t1 - run_t0).count();
        const double fps =
            run_ms > 0.0 ? (static_cast<double>(std::max(frames, 1)) * 1000.0 / run_ms) : 0.0;
        std::cout << "FPS model=" << tar.filename().string() << " backend=" << processcvu_run_target
                  << " placement="
                  << (processcvu_placement.empty() ? "default" : processcvu_placement)
                  << " async_mode=" << async_selection_name(async_selection)
                  << " processcvu_async=" << (async_selection.processcvu ? 1 : 0)
                  << " processmla_async=" << (async_selection.processmla ? 1 : 0)
                  << " async_queue_depth=" << async_queue_depth << " prepared_runner_mode="
                  << (prepared_runner_mode.empty() ? "off" : prepared_runner_mode)
                  << " prepared_runner_dequant_flags="
                  << (prepared_runner_dequant_flags.empty() ? "default"
                                                            : prepared_runner_dequant_flags)
                  << " frames=" << std::max(frames, 1) << " run_ms=" << run_ms << " fps=" << fps
                  << "\n";
        const auto tensor_io_after = simaai::neat::pipeline_internal::snapshot_tensor_io_stats();
        const auto tensor_io = tensor_io_delta(tensor_io_before, tensor_io_after);

        if (boxdecode_mode == BoxDecodeRunMode::NoModel) {
          require_preprocess_meta_on_output_local(infer_sample, img_bgr.cols, img_bgr.rows,
                                                  "canonical_e2e_output");
        }
        const AccuracyResult acc =
            run_framework_boxdecode_accuracy(infer_sample, model, img_bgr, boxdecode_mode);
        require(acc.ok, "accuracy check failed: " + acc.note);

        std::cout << "E2E model=" << tar.filename().string() << " backend=" << processcvu_run_target
                  << " placement="
                  << (processcvu_placement.empty() ? "default" : processcvu_placement)
                  << " boxdecode_mode=" << boxdecode_run_mode_name(boxdecode_mode)
                  << " signature=\"" << sample_output_signature_local(infer_sample) << "\""
                  << " accuracy=\"" << acc.note << "\" tensor_io=\""
                  << tensor_io_stats_string(tensor_io) << "\"\n";
      } catch (const std::exception& ex) {
        failures += 1;
        std::cerr << "[FAIL] model=" << tar.filename().string() << " err=" << ex.what() << "\n";
      }
    }

    require(!packs.empty(), "no model packs executed");
    require(failures == 0, "canonical yolov8 e2e failed for one or more models");
    std::cout << "[OK] graph_migration_legacy_yolov8_variant_route_matrix_test models="
              << packs.size() << "\n";
    return 0;
  } catch (const SkipTest& e) {
    std::cout << "[SKIP] " << e.what() << "\n";
    return skip_long_test(e.what());
  } catch (const std::exception& e) {
    std::cerr << "[FAIL] " << e.what() << "\n";
    return 1;
  }
#endif
}
