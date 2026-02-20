#include "pipeline/StageRun.h"

#include "builder/ConfigJsonOverride.h"
#include "builder/ConfigJsonProvider.h"
#include "model/internal/ModelInternal.h"
#include "nodes/common/Output.h"
#include "nodes/io/Input.h"
#include "nodes/sima/SimaBoxDecode.h"
#include "pipeline/Session.h"
#include "pipeline/internal/PipelineBuild.h"
#include "pipeline/internal/StageConfig.h"
#include "pipeline/internal/TensorUtil.h"
#include "pipeline/internal/TensorMath.h"
#include "pipeline/TessellatedTensor.h"

#include <nlohmann/json.hpp>

#include <algorithm>
#include <cctype>
#include <cstdlib>
#include <cstdio>
#include <cstring>
#include <functional>
#include <mutex>
#include <sstream>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <vector>

namespace simaai::neat::stages {
using simaai::neat::pipeline_internal::upper_copy;
namespace {

enum class StageKind {
  Preproc,
  Infer,
  MLA,
  BoxDecode,
};

struct StageInputKey {
  std::string media_type;
  std::string format;
  int width = -1;
  int height = -1;
  int depth = -1;
};

bool stage_trace_enabled() {
  const char* v = std::getenv("SIMA_DISPATCHER_TRACE");
  if (!v || !*v)
    return false;
  return std::strcmp(v, "0") != 0 && std::strcmp(v, "false") != 0 && std::strcmp(v, "FALSE") != 0;
}

void stage_trace(const char* label) {
  if (!stage_trace_enabled())
    return;
  std::fprintf(stderr, "[TRACE] %s\n", label);
}

struct StageKey {
  StageKind kind = StageKind::Preproc;
  std::string model_id;
  StageInputKey input;
  BoxDecodeOptions box_opt;
};

struct WireCaps {
  std::string media_type;
  std::string format;
  TensorDims dims;
  std::string buffer_name;
  std::string caps_override;
};

struct WireInput {
  simaai::neat::Tensor tensor;
  InputOptions appsrc;
  WireCaps caps;
};

std::string buffer_name_from_group(const NodeGroup& group);
InputOptions appsrc_for_tensor_wire(const simaai::neat::Tensor& input, const WireCaps& wire);

bool stage_debug_enabled() {
  static int enabled = -1;
  if (enabled >= 0)
    return enabled != 0;
  const char* v = std::getenv("SIMA_STAGE_DEBUG");
  if (!v || !*v) {
    enabled = 0;
    return false;
  }
  if (!std::strcmp(v, "1") || !std::strcmp(v, "true") || !std::strcmp(v, "TRUE") ||
      !std::strcmp(v, "yes") || !std::strcmp(v, "YES") || !std::strcmp(v, "on") ||
      !std::strcmp(v, "ON")) {
    enabled = 1;
    return true;
  }
  enabled = 0;
  return false;
}

const char* dtype_name(TensorDType dtype) {
  switch (dtype) {
  case TensorDType::UInt8:
    return "UInt8";
  case TensorDType::Int8:
    return "Int8";
  case TensorDType::UInt16:
    return "UInt16";
  case TensorDType::Int16:
    return "Int16";
  case TensorDType::Int32:
    return "Int32";
  case TensorDType::BFloat16:
    return "BFloat16";
  case TensorDType::Float32:
    return "Float32";
  case TensorDType::Float64:
    return "Float64";
  }
  return "Unknown";
}

std::string shape_string(const std::vector<int64_t>& shape) {
  if (shape.empty())
    return "";
  std::string out;
  for (size_t i = 0; i < shape.size(); ++i) {
    if (i > 0)
      out.push_back('x');
    out += std::to_string(shape[i]);
  }
  return out;
}

int dtype_bytes(TensorDType dtype);
int tensor_depth_from_shape(const std::vector<int64_t>& shape);

std::string image_format_name(simaai::neat::ImageSpec::PixelFormat fmt) {
  switch (fmt) {
  case simaai::neat::ImageSpec::PixelFormat::RGB:
    return "RGB";
  case simaai::neat::ImageSpec::PixelFormat::BGR:
    return "BGR";
  case simaai::neat::ImageSpec::PixelFormat::GRAY8:
    return "GRAY8";
  case simaai::neat::ImageSpec::PixelFormat::NV12:
    return "NV12";
  case simaai::neat::ImageSpec::PixelFormat::I420:
    return "I420";
  case simaai::neat::ImageSpec::PixelFormat::UNKNOWN:
    break;
  }
  return "";
}

simaai::neat::ImageSpec::PixelFormat image_format_from_string(const std::string& fmt) {
  const std::string up = upper_copy(fmt);
  if (up == "RGB")
    return simaai::neat::ImageSpec::PixelFormat::RGB;
  if (up == "BGR")
    return simaai::neat::ImageSpec::PixelFormat::BGR;
  if (up == "GRAY8" || up == "GRAY")
    return simaai::neat::ImageSpec::PixelFormat::GRAY8;
  if (up == "NV12")
    return simaai::neat::ImageSpec::PixelFormat::NV12;
  if (up == "I420" || up == "YUV420")
    return simaai::neat::ImageSpec::PixelFormat::I420;
  return simaai::neat::ImageSpec::PixelFormat::UNKNOWN;
}

std::string format_from_tensor(const simaai::neat::Tensor& tensor) {
  if (tensor.semantic.tess.has_value()) {
    return upper_copy(tensor.semantic.tess->format);
  }
  if (tensor.semantic.image.has_value()) {
    const std::string fmt = image_format_name(tensor.semantic.image->format);
    if (!fmt.empty())
      return fmt;
  }
  return "";
}

simaai::neat::Tensor require_tessellated_int8(simaai::neat::Tensor tensor, const char* where) {
  const std::string prefix = (where && *where) ? (std::string(where) + ": ") : "";
  if (!tensor.storage) {
    throw std::runtime_error(prefix + "tessellated int8: missing tensor storage");
  }
  const std::string fmt = format_from_tensor(tensor);
  if (!is_tessellated_int8_format(fmt)) {
    throw std::runtime_error(prefix + "tessellated int8: unexpected format: " + fmt);
  }
  if (tensor.dtype == TensorDType::UInt8) {
    tensor.dtype = TensorDType::Int8;
  }
  if (!tensor.semantic.tess.has_value()) {
    simaai::neat::TessSpec tess;
    tess.format = fmt;
    tensor.semantic.tess = tess;
  } else {
    tensor.semantic.tess->format = fmt;
  }
  return tensor;
}

int64_t tensor_dense_bytes_tight(const simaai::neat::Tensor& tensor) {
  if (tensor.shape.empty())
    return 0;
  size_t total = dtype_bytes(tensor.dtype);
  for (const auto dim : tensor.shape) {
    if (dim <= 0)
      return 0;
    total *= static_cast<size_t>(dim);
  }
  return static_cast<int64_t>(total);
}

int64_t tensor_plane_bytes_tight(const simaai::neat::Plane& plane, TensorDType dtype) {
  if (plane.shape.size() < 2)
    return 0;
  const int64_t h = plane.shape[0];
  const int64_t w = plane.shape[1];
  if (h <= 0 || w <= 0)
    return 0;
  return static_cast<int64_t>(dtype_bytes(dtype)) * h * w;
}

int64_t tensor_total_bytes(const simaai::neat::Tensor& tensor) {
  if (tensor.is_composite()) {
    int64_t total = 0;
    for (const auto& plane : tensor.planes) {
      total += tensor_plane_bytes_tight(plane, tensor.dtype);
    }
    return total;
  }
  return tensor_dense_bytes_tight(tensor);
}

const nlohmann::json* config_json_from_group(const NodeGroup& group) {
  for (const auto& node : group.nodes()) {
    if (!node)
      continue;
    auto* provider = dynamic_cast<ConfigJsonProvider*>(node.get());
    if (!provider)
      continue;
    const nlohmann::json* cfg = provider->config_json();
    if (cfg)
      return cfg;
  }
  return nullptr;
}

void dump_config_json(const NodeGroup& group, const char* label) {
  if (!stage_debug_enabled())
    return;
  const nlohmann::json* cfg = config_json_from_group(group);
  if (!cfg) {
    std::fprintf(stderr, "[DBG] StageRun %s config: <none>\n", label ? label : "");
    return;
  }
  std::fprintf(stderr, "[DBG] StageRun %s config:\n%s\n", label ? label : "", cfg->dump(2).c_str());
}

TensorDims dims_from_tensor(const simaai::neat::Tensor& tensor) {
  TensorDims dims;
  int h = (tensor.shape.size() > 0) ? static_cast<int>(tensor.shape[0]) : -1;
  int w = (tensor.shape.size() > 1) ? static_cast<int>(tensor.shape[1]) : -1;
  int d = (tensor.shape.size() > 2) ? static_cast<int>(tensor.shape[2]) : -1;
  if (tensor.is_composite() && !tensor.planes.empty()) {
    const auto& y = tensor.planes.front();
    if (y.shape.size() >= 2) {
      h = (h > 0) ? h : static_cast<int>(y.shape[0]);
      w = (w > 0) ? w : static_cast<int>(y.shape[1]);
    }
  }
  dims.width = w;
  dims.height = h;
  if (d > 0)
    dims.depth = d;
  return dims;
}

int dtype_bytes(TensorDType dtype) {
  switch (dtype) {
  case TensorDType::Int8:
  case TensorDType::UInt8:
    return 1;
  case TensorDType::UInt16:
  case TensorDType::Int16:
  case TensorDType::BFloat16:
    return 2;
  case TensorDType::Int32:
  case TensorDType::Float32:
    return 4;
  case TensorDType::Float64:
    return 8;
  }
  return 1;
}

void apply_tensor_dims(simaai::neat::Tensor& tensor, const TensorDims& dims) {
  if (dims.width <= 0 || dims.height <= 0 || dims.depth <= 0)
    return;
  tensor.shape = {dims.height, dims.width, dims.depth};
  const int64_t elem = dtype_bytes(tensor.dtype);
  tensor.strides_bytes = {dims.width * dims.depth * elem, dims.depth * elem, elem};
}

void apply_tensor_hw(simaai::neat::Tensor& tensor, const TensorDims& dims) {
  if (dims.width > 0 && dims.height > 0 && dims.depth > 0) {
    tensor.shape = {dims.height, dims.width, dims.depth};
    const int64_t elem = dtype_bytes(tensor.dtype);
    tensor.strides_bytes = {dims.width * dims.depth * elem, dims.depth * elem, elem};
  } else if (dims.width > 0 && dims.height > 0) {
    tensor.shape = {dims.height, dims.width};
    const int64_t elem = dtype_bytes(tensor.dtype);
    tensor.strides_bytes = {dims.width * elem, elem};
  }
}

void apply_tensor_size(simaai::neat::Tensor& tensor, int64_t size_bytes) {
  if (size_bytes <= 0)
    return;
  const int elem_size = dtype_bytes(tensor.dtype);
  if (elem_size <= 0)
    return;
  const int64_t elems = size_bytes / elem_size;
  if (elems <= 0)
    return;
  tensor.shape = {1, elems, 1};
  tensor.strides_bytes = {static_cast<int64_t>(elems * elem_size), static_cast<int64_t>(elem_size),
                          static_cast<int64_t>(elem_size)};
}

void apply_tensor_dtype_from_format(simaai::neat::Tensor& tensor, const std::string& fmt) {
  if (is_tessellated_int8_format(fmt)) {
    tensor.dtype = TensorDType::Int8;
  } else if (is_tessellated_bf16_format(fmt)) {
    tensor.dtype = TensorDType::BFloat16;
  }
  if (!fmt.empty()) {
    if (!tensor.semantic.tess.has_value()) {
      simaai::neat::TessSpec tess;
      tess.format = fmt;
      tensor.semantic.tess = tess;
    } else {
      tensor.semantic.tess->format = fmt;
    }
  }
}

void apply_preproc_output_override(simaai::neat::Tensor& tensor, const PreprocOutputInfo& info) {
  // JSON config overrides caps. Preproc caps can remain RGB even when tessellated.
  apply_tensor_dims(tensor, info.dims);
  if (!info.tessellate || info.output_dtype.empty())
    return;
  const std::string fmt = upper_copy(info.output_dtype);
  apply_tensor_dtype_from_format(tensor, fmt);
}

int tessellated_memory_index(const PreprocOutputInfo& info) {
  for (size_t i = 0; i < info.output_memory_order.size(); ++i) {
    const std::string up = upper_copy(info.output_memory_order[i]);
    if (up == "OUTPUT_TESSELLATED_IMAGE") {
      return static_cast<int>(i);
    }
  }
  return info.output_memory_order.empty() ? 0 : 0;
}

void apply_mla_output_override(simaai::neat::Tensor& tensor, const MlaOutputInfo& info) {
  // JSON config overrides caps. MLA caps can be "MLA" even though dtype is INT8/BF16.
  apply_tensor_dims(tensor, info.dims);
  if (!info.data_type.empty()) {
    const std::string fmt = upper_copy(info.data_type);
    apply_tensor_dtype_from_format(tensor, fmt);
  }
  if (info.dims.width <= 0 || info.dims.height <= 0 || info.dims.depth <= 0) {
    apply_tensor_size(tensor, info.size_bytes);
  }
}

bool set_input_buffer_name(nlohmann::json& j, const std::string& name) {
  if (name.empty())
    return false;
  bool changed = false;
  if (j.contains("input_buffers") && j["input_buffers"].is_array()) {
    auto& arr = j["input_buffers"];
    if (arr.size() > 1 && arr[0].is_object()) {
      arr[0]["name"] = name;
      changed = true;
    }
  }
  if (j.contains("buffers") && j["buffers"].is_object()) {
    auto& buffers = j["buffers"];
    if (buffers.contains("input") && buffers["input"].is_array()) {
      auto& arr = buffers["input"];
      if (arr.size() > 1 && arr[0].is_object()) {
        arr[0]["name"] = name;
        changed = true;
      }
    }
  }
  return changed;
}

void override_buffer_name(NodeGroup& group, const std::string& name, const char* tag) {
  if (name.empty())
    return;
  const std::string tag_str = tag ? tag : "";
  for (auto& node : group.nodes_mut()) {
    if (!node)
      continue;
    auto* override = dynamic_cast<ConfigJsonOverride*>(node.get());
    if (!override)
      continue;
    override->override_config_json([&](nlohmann::json& j) { (void)set_input_buffer_name(j, name); },
                                   tag_str);
  }
}

std::string pick_caps_format(const nlohmann::json& caps) {
  if (!caps.contains("sink_pads") || !caps["sink_pads"].is_array() || caps["sink_pads"].empty() ||
      !caps["sink_pads"][0].is_object()) {
    return "";
  }
  const auto& sink = caps["sink_pads"][0];
  if (!sink.contains("params") || !sink["params"].is_array())
    return "";
  for (const auto& param : sink["params"]) {
    if (!param.is_object())
      continue;
    if (!param.contains("name") || !param["name"].is_string())
      continue;
    if (param["name"].get<std::string>() != "format")
      continue;
    if (!param.contains("values") || !param["values"].is_string())
      continue;
    const std::string values = upper_copy(param["values"].get<std::string>());
    if (values.find("BGR") != std::string::npos)
      return "BGR";
    if (values.find("RGB") != std::string::npos)
      return "RGB";
    if (!values.empty()) {
      const size_t comma = values.find(',');
      return values.substr(0, comma == std::string::npos ? values.size() : comma);
    }
  }
  return "";
}

WireCaps build_wire_caps_from_json_or_tensor(const NodeGroup& group,
                                             const simaai::neat::Tensor& input,
                                             const TensorDims* dims_override,
                                             const char* media_type, const char* default_format,
                                             bool use_json_overrides) {
  WireCaps wire;
  wire.media_type = media_type ? media_type : "application/vnd.simaai.tensor";
  wire.format = default_format ? default_format : format_from_tensor(input);
  wire.dims = dims_override ? *dims_override : dims_from_tensor(input);
  wire.buffer_name = buffer_name_from_group(group);

  if (use_json_overrides) {
    const nlohmann::json* cfg = config_json_from_group(group);
    if (cfg && cfg->contains("caps") && (*cfg)["caps"].is_object()) {
      const std::string fmt = pick_caps_format((*cfg)["caps"]);
      if (!fmt.empty())
        wire.format = fmt;
    }
  }

  return wire;
}

WireInput build_wire_input_from_tensor(const simaai::neat::Tensor& input, const WireCaps& wire) {
  WireInput out;
  out.caps = wire;
  // User tensor format is INT8/BF16; wire caps are plugin-expected (MLA/multi-tensor).
  out.tensor = input;
  if (!wire.format.empty()) {
    const std::string fmt = upper_copy(wire.format);
    if (wire.media_type == "video/x-raw") {
      out.tensor.dtype = TensorDType::UInt8;
      out.tensor.semantic.tess.reset();
      if (!out.tensor.semantic.image.has_value()) {
        simaai::neat::ImageSpec image;
        image.format = image_format_from_string(fmt);
        out.tensor.semantic.image = image;
      } else {
        out.tensor.semantic.image->format = image_format_from_string(fmt);
      }
    } else {
      if (!out.tensor.semantic.tess.has_value()) {
        simaai::neat::TessSpec tess;
        tess.format = fmt;
        out.tensor.semantic.tess = tess;
      } else {
        out.tensor.semantic.tess->format = fmt;
      }
    }
  }
  apply_tensor_hw(out.tensor, wire.dims);
  if (out.tensor.layout == TensorLayout::Unknown || out.tensor.layout == TensorLayout::Planar) {
    const int64_t h = (out.tensor.shape.size() > 0) ? out.tensor.shape[0] : -1;
    const int64_t w = (out.tensor.shape.size() > 1) ? out.tensor.shape[1] : -1;
    const int64_t d = (out.tensor.shape.size() > 2) ? out.tensor.shape[2] : -1;
    if (h > 0 && w > 0) {
      out.tensor.layout =
          (out.tensor.shape.size() >= 3 && d > 0) ? TensorLayout::HWC : TensorLayout::HW;
    }
  }
  out.appsrc = appsrc_for_tensor_wire(out.tensor, wire);
  if (!wire.caps_override.empty()) {
    out.appsrc.caps_override = wire.caps_override;
  }
  return out;
}

bool operator==(const StageInputKey& a, const StageInputKey& b) {
  return a.media_type == b.media_type && a.format == b.format && a.width == b.width &&
         a.height == b.height && a.depth == b.depth;
}

bool operator==(const BoxDecodeOptions& a, const BoxDecodeOptions& b) {
  return a.decode_type == b.decode_type && a.original_width == b.original_width &&
         a.original_height == b.original_height && a.detection_threshold == b.detection_threshold &&
         a.nms_iou_threshold == b.nms_iou_threshold && a.top_k == b.top_k;
}

bool operator==(const StageKey& a, const StageKey& b) {
  return a.kind == b.kind && a.model_id == b.model_id && a.input == b.input &&
         a.box_opt == b.box_opt;
}

size_t hash_combine(size_t seed, size_t v) {
  return seed ^ (v + 0x9e3779b97f4a7c15ULL + (seed << 6) + (seed >> 2));
}

struct StageKeyHash {
  size_t operator()(const StageKey& k) const {
    size_t h = std::hash<int>()(static_cast<int>(k.kind));
    h = hash_combine(h, std::hash<std::string>()(k.model_id));
    h = hash_combine(h, std::hash<std::string>()(k.input.media_type));
    h = hash_combine(h, std::hash<std::string>()(k.input.format));
    h = hash_combine(h, std::hash<int>()(k.input.width));
    h = hash_combine(h, std::hash<int>()(k.input.height));
    h = hash_combine(h, std::hash<int>()(k.input.depth));
    h = hash_combine(h, std::hash<std::string>()(k.box_opt.decode_type));
    h = hash_combine(h, std::hash<int>()(k.box_opt.original_width));
    h = hash_combine(h, std::hash<int>()(k.box_opt.original_height));
    h = hash_combine(h, std::hash<double>()(k.box_opt.detection_threshold));
    h = hash_combine(h, std::hash<double>()(k.box_opt.nms_iou_threshold));
    h = hash_combine(h, std::hash<int>()(k.box_opt.top_k));
    return h;
  }
};

std::mutex g_cache_mu;
std::unordered_map<StageKey, std::shared_ptr<Run>, StageKeyHash> g_cache;

std::string buffer_name_from_json(const nlohmann::json& j) {
  if (j.contains("input_buffers") && j["input_buffers"].is_array() && !j["input_buffers"].empty() &&
      j["input_buffers"][0].is_object()) {
    const auto& buf = j["input_buffers"][0];
    if (buf.contains("name") && buf["name"].is_string()) {
      return buf["name"].get<std::string>();
    }
  }

  if (j.contains("buffers") && j["buffers"].is_object()) {
    const auto& buffers = j["buffers"];
    if (buffers.contains("input") && buffers["input"].is_array() && !buffers["input"].empty() &&
        buffers["input"][0].is_object()) {
      const auto& buf = buffers["input"][0];
      if (buf.contains("name") && buf["name"].is_string()) {
        return buf["name"].get<std::string>();
      }
    }
  }

  return "decoder";
}

std::string buffer_name_from_group(const NodeGroup& group) {
  for (const auto& node : group.nodes()) {
    if (!node)
      continue;
    auto* provider = dynamic_cast<ConfigJsonProvider*>(node.get());
    if (!provider)
      continue;
    const nlohmann::json* cfg = provider->config_json();
    if (cfg)
      return buffer_name_from_json(*cfg);
  }
  return "decoder";
}

InputOptions appsrc_for_mat(const cv::Mat& input, const NodeGroup& group) {
  InputOptions opt;
  opt.media_type = "video/x-raw";
  opt.width = input.cols;
  opt.height = input.rows;
  opt.depth = input.channels();
  if (opt.depth == 1) {
    opt.format = "GRAY8";
  } else {
    opt.format = "BGR";
  }
  opt.buffer_name = buffer_name_from_group(group);
  return opt;
}

int tensor_depth_from_shape(const std::vector<int64_t>& shape) {
  if (shape.size() >= 3)
    return static_cast<int>(shape[2]);
  return -1;
}

int shape_dim(const std::vector<int64_t>& shape, size_t index) {
  if (index >= shape.size())
    return -1;
  return static_cast<int>(shape[index]);
}

// Wire caps are for plugin negotiation only; user-facing tensor metadata stays INT8/BF16.
InputOptions appsrc_for_tensor_wire(const simaai::neat::Tensor& input, const WireCaps& wire) {
  InputOptions opt;
  opt.media_type = wire.media_type.empty() ? "application/vnd.simaai.tensor" : wire.media_type;
  const std::string input_fmt = format_from_tensor(input);
  opt.format = wire.format.empty() ? input_fmt : wire.format;
  if (opt.format.empty()) {
    throw std::runtime_error("StageRun: tensor input missing format");
  }
  const int shape_h = shape_dim(input.shape, 0);
  const int shape_w = shape_dim(input.shape, 1);
  const int shape_d = shape_dim(input.shape, 2);
  opt.width = (wire.dims.width > 0) ? wire.dims.width : shape_w;
  opt.height = (wire.dims.height > 0) ? wire.dims.height : shape_h;
  opt.depth = (wire.dims.depth > 0) ? wire.dims.depth : shape_d;
  if (opt.width <= 0 || opt.height <= 0) {
    throw std::runtime_error("StageRun: tensor input missing width/height");
  }
  opt.buffer_name = wire.buffer_name.empty() ? "decoder" : wire.buffer_name;
  return opt;
}

StageInputKey make_input_key(const InputOptions& opt) {
  StageInputKey key;
  key.media_type = opt.media_type;
  key.format = upper_copy(opt.format);
  key.width = opt.width;
  key.height = opt.height;
  key.depth = opt.depth;
  return key;
}

RunOptions stage_run_defaults() {
  RunOptions opt;
  opt.preset = RunPreset::Reliable;
  opt.queue_depth = 1;
  opt.overflow_policy = OverflowPolicy::Block;
  opt.output_memory = OutputMemory::Owned;
  opt.advanced.copy_input = false;
  return opt;
}

int default_timeout_ms() {
  const char* env = std::getenv("SIMA_GST_RUN_INPUT_TIMEOUT_MS");
  if (!env || !*env)
    return 10000;
  return std::max(10, std::atoi(env));
}

simaai::neat::Tensor take_tensor(const Sample& out, const char* where) {
  if (out.kind != SampleKind::Tensor) {
    throw std::runtime_error(std::string(where) + ": expected tensor output");
  }
  if (out.tensor.has_value()) {
    return out.tensor.value();
  }
  throw std::runtime_error(std::string(where) + ": missing tensor output");
}

std::shared_ptr<Run> get_or_build(const StageKey& key, const std::function<Run()>& builder) {
  {
    std::lock_guard<std::mutex> lock(g_cache_mu);
    auto it = g_cache.find(key);
    if (it != g_cache.end())
      return it->second;
  }

  Run run = builder();
  auto handle = std::make_shared<Run>(std::move(run));

  std::lock_guard<std::mutex> lock(g_cache_mu);
  auto it = g_cache.find(key);
  if (it != g_cache.end())
    return it->second;
  g_cache.emplace(key, handle);
  return handle;
}

} // namespace

simaai::neat::Tensor Preproc(const cv::Mat& input, const simaai::neat::Model& model) {
  NodeGroup group = simaai::neat::internal::ModelAccess::build_preprocess_group(model, true);

  const PreprocOutputInfo preproc_info = read_preproc_output_info(group);
  InputOptions src_opt = appsrc_for_mat(input, group);

  StageKey key;
  key.kind = StageKind::Preproc;
  key.model_id = simaai::neat::internal::ModelAccess::model_id(model);
  key.input = make_input_key(src_opt);

  auto runner = get_or_build(key, [&]() {
    RunOptions opt = stage_run_defaults();
    // Keep the GstSample alive so we can copy the tessellated memory.
    opt.output_memory = OutputMemory::ZeroCopy;
    auto input_node = simaai::neat::nodes::Input(src_opt);
    auto output_node = simaai::neat::nodes::Output();
    std::vector<std::shared_ptr<Node>> nodes;
    nodes.reserve(group.size() + 2);
    nodes.push_back(input_node);
    for (const auto& node : group.nodes()) {
      nodes.push_back(node);
    }
    nodes.push_back(output_node);

    pipeline_internal::PipelineBuildContext build_ctx(SessionOptions{});
    build_ctx.apply_name_transform_to_configs(nodes);
    build_ctx.wire_configs_by_order(nodes);

    Session p;
    p.add(input_node);
    p.add(group);
    p.add(output_node);
    return p.build(input, RunMode::Sync, opt);
  });

  const int timeout_ms = default_timeout_ms();
  Sample out = runner->push_and_pull(input, timeout_ms);
  simaai::neat::Tensor tensor;
  // Preproc caps report RGB, but tessellated bytes live in a different GstBuffer memory.
  if (preproc_info.tessellate && out.tensor.has_value()) {
    const int mem_index = tessellated_memory_index(preproc_info);
    tensor = pipeline_internal::copy_tensor_from_sample_memory(*out.tensor, mem_index);
  } else {
    tensor = take_tensor(out, "Preproc");
  }
  apply_preproc_output_override(tensor, preproc_info);
  const std::string pre_fmt = upper_copy(format_from_tensor(tensor));
  if (tensor.dtype == TensorDType::BFloat16 || is_tessellated_bf16_format(pre_fmt)) {
    throw std::runtime_error("Preproc: BF16 path is not supported yet");
  }
  return require_tessellated_int8(std::move(tensor), "Preproc");
}

simaai::neat::Tensor Infer(const simaai::neat::Tensor& input, const simaai::neat::Model& model) {
  NodeGroup group = simaai::neat::internal::ModelAccess::build_infer_group(model, true);
  const MlaOutputInfo infer_info = read_mla_output_info(group);
  const WireCaps wire =
      build_wire_caps_from_json_or_tensor(group, input, nullptr, "video/x-raw", "BGR", true);
  const WireInput wire_input = build_wire_input_from_tensor(input, wire);
  const InputOptions src_opt = wire_input.appsrc;

  StageKey key;
  key.kind = StageKind::Infer;
  key.model_id = simaai::neat::internal::ModelAccess::model_id(model);
  key.input = make_input_key(src_opt);

  stage_trace("StageRun::Infer: before get_or_build");
  auto runner = get_or_build(key, [&]() {
    override_buffer_name(group, src_opt.buffer_name, "infer");
    RunOptions opt = stage_run_defaults();
    // Preserve the GstBuffer so post-processing can reuse plugin metadata.
    opt.output_memory = OutputMemory::ZeroCopy;
    auto input_node = simaai::neat::nodes::Input(src_opt);
    auto output_node = simaai::neat::nodes::Output();
    std::vector<std::shared_ptr<Node>> nodes;
    nodes.reserve(group.size() + 2);
    nodes.push_back(input_node);
    for (const auto& node : group.nodes()) {
      nodes.push_back(node);
    }
    nodes.push_back(output_node);

    pipeline_internal::PipelineBuildContext build_ctx(SessionOptions{});
    build_ctx.apply_name_transform_to_configs(nodes);
    build_ctx.wire_configs_by_order(nodes);

    Session p;
    p.add(input_node);
    p.add(group);
    p.add(output_node);
    return p.build(wire_input.tensor, RunMode::Sync, opt);
  });
  stage_trace("StageRun::Infer: after get_or_build");

  const int timeout_ms = default_timeout_ms();
  stage_trace("StageRun::Infer: before push_and_pull");
  Sample out = runner->push_and_pull(wire_input.tensor, timeout_ms);
  stage_trace("StageRun::Infer: after push_and_pull");
  simaai::neat::Tensor tensor;
  if (out.tensor.has_value() && out.tensor->planes.empty()) {
    tensor = pipeline_internal::copy_tensor_from_sample_memory(*out.tensor, 0);
  } else {
    tensor = take_tensor(out, "Infer");
  }
  apply_mla_output_override(tensor, infer_info);
  if (tensor.layout == TensorLayout::Unknown || tensor.layout == TensorLayout::Planar) {
    const int64_t h = (tensor.shape.size() > 0) ? tensor.shape[0] : -1;
    const int64_t w = (tensor.shape.size() > 1) ? tensor.shape[1] : -1;
    const int64_t d = (tensor.shape.size() > 2) ? tensor.shape[2] : -1;
    if (h > 0 && w > 0) {
      tensor.layout = (tensor.shape.size() >= 3 && d > 0) ? TensorLayout::HWC : TensorLayout::HW;
    }
  }
  const std::string infer_fmt = upper_copy(format_from_tensor(tensor));
  if (tensor.dtype == TensorDType::BFloat16 || is_tessellated_bf16_format(infer_fmt)) {
    throw std::runtime_error("Infer: BF16 path is not supported yet");
  }
  if (stage_debug_enabled()) {
    const int64_t actual_bytes = tensor_total_bytes(tensor);
    const size_t plane0 =
        tensor.planes.empty()
            ? 0
            : static_cast<size_t>(tensor_plane_bytes_tight(tensor.planes[0], tensor.dtype));
    std::fprintf(stderr,
                 "[DBG] StageRun Infer out: format=%s dtype=%s w=%d h=%d shape=%s plane0=%zu "
                 "bytes=%lld infer_size=%lld\n",
                 infer_fmt.c_str(), dtype_name(tensor.dtype),
                 (tensor.shape.size() > 1) ? static_cast<int>(tensor.shape[1]) : -1,
                 (tensor.shape.size() > 0) ? static_cast<int>(tensor.shape[0]) : -1,
                 shape_string(tensor.shape).c_str(), plane0, static_cast<long long>(actual_bytes),
                 static_cast<long long>(infer_info.size_bytes));
  }
  return require_tessellated_int8(std::move(tensor), "Infer");
}

simaai::neat::Tensor MLA(const simaai::neat::Tensor& input, const simaai::neat::Model& model) {
  NodeGroup group = simaai::neat::internal::ModelAccess::build_infer_group(model, true);
  const MlaOutputInfo mla_info = read_mla_output_info(group);
  const WireCaps wire =
      build_wire_caps_from_json_or_tensor(group, input, nullptr, "video/x-raw", "BGR", true);
  const WireInput wire_input = build_wire_input_from_tensor(input, wire);
  const InputOptions src_opt = wire_input.appsrc;

  StageKey key;
  key.kind = StageKind::MLA;
  key.model_id = simaai::neat::internal::ModelAccess::model_id(model);
  key.input = make_input_key(src_opt);

  stage_trace("StageRun::MLA: before get_or_build");
  auto runner = get_or_build(key, [&]() {
    override_buffer_name(group, src_opt.buffer_name, "mla");
    RunOptions opt = stage_run_defaults();
    // Preserve the GstBuffer so BoxDecode can reuse plugin metadata.
    opt.output_memory = OutputMemory::ZeroCopy;
    auto input_node = simaai::neat::nodes::Input(src_opt);
    auto output_node = simaai::neat::nodes::Output();
    std::vector<std::shared_ptr<Node>> nodes;
    nodes.reserve(group.size() + 2);
    nodes.push_back(input_node);
    for (const auto& node : group.nodes()) {
      nodes.push_back(node);
    }
    nodes.push_back(output_node);

    pipeline_internal::PipelineBuildContext build_ctx(SessionOptions{});
    build_ctx.apply_name_transform_to_configs(nodes);
    build_ctx.wire_configs_by_order(nodes);

    Session p;
    p.add(input_node);
    p.add(group);
    p.add(output_node);
    return p.build(wire_input.tensor, RunMode::Sync, opt);
  });
  stage_trace("StageRun::MLA: after get_or_build");

  const int timeout_ms = default_timeout_ms();
  stage_trace("StageRun::MLA: before push_and_pull");
  Sample out = runner->push_and_pull(wire_input.tensor, timeout_ms);
  stage_trace("StageRun::MLA: after push_and_pull");
  simaai::neat::Tensor tensor;
  if (out.tensor.has_value() && out.tensor->planes.empty()) {
    tensor = pipeline_internal::copy_tensor_from_sample_memory(*out.tensor, 0);
  } else {
    tensor = take_tensor(out, "MLA");
  }
  apply_mla_output_override(tensor, mla_info);
  if (tensor.layout == TensorLayout::Unknown || tensor.layout == TensorLayout::Planar) {
    const int64_t h = (tensor.shape.size() > 0) ? tensor.shape[0] : -1;
    const int64_t w = (tensor.shape.size() > 1) ? tensor.shape[1] : -1;
    const int64_t d = (tensor.shape.size() > 2) ? tensor.shape[2] : -1;
    if (h > 0 && w > 0) {
      tensor.layout = (tensor.shape.size() >= 3 && d > 0) ? TensorLayout::HWC : TensorLayout::HW;
    }
  }
  const std::string mla_fmt = upper_copy(format_from_tensor(tensor));
  if (tensor.dtype == TensorDType::BFloat16 || is_tessellated_bf16_format(mla_fmt)) {
    throw std::runtime_error("MLA: BF16 path is not supported yet");
  }
  if (stage_debug_enabled()) {
    const int64_t actual_bytes = tensor_total_bytes(tensor);
    const size_t plane0 =
        tensor.planes.empty()
            ? 0
            : static_cast<size_t>(tensor_plane_bytes_tight(tensor.planes[0], tensor.dtype));
    std::fprintf(stderr,
                 "[DBG] StageRun MLA out: format=%s dtype=%s w=%d h=%d shape=%s plane0=%zu "
                 "bytes=%lld mla_size=%lld\n",
                 mla_fmt.c_str(), dtype_name(tensor.dtype),
                 (tensor.shape.size() > 1) ? static_cast<int>(tensor.shape[1]) : -1,
                 (tensor.shape.size() > 0) ? static_cast<int>(tensor.shape[0]) : -1,
                 shape_string(tensor.shape).c_str(), plane0, static_cast<long long>(actual_bytes),
                 static_cast<long long>(mla_info.size_bytes));
  }
  return require_tessellated_int8(std::move(tensor), "MLA");
}

BoxDecodeResult BoxDecode(const simaai::neat::Tensor& input, const simaai::neat::Model& model,
                          const BoxDecodeOptions& opt) {
  if (opt.original_width <= 0 || opt.original_height <= 0) {
    throw std::runtime_error("BoxDecode: original_width/height are required");
  }

  auto node = simaai::neat::nodes::SimaBoxDecode(model, opt.decode_type, opt.original_width,
                                                 opt.original_height, opt.detection_threshold,
                                                 opt.nms_iou_threshold, opt.top_k);
  NodeGroup group(std::vector<std::shared_ptr<Node>>{node});
  const BoxDecodeInputInfo box_info = read_boxdecode_input_info(group);
  const BoxDecodeExpectedInfo box_expected = read_boxdecode_expected_info(group);
  WireCaps wire = build_wire_caps_from_json_or_tensor(
      group, input, &box_info.dims, "application/vnd.simaai.tensor", "MLA", false);
  wire.caps_override = build_boxdecode_caps_override(group);
  if (!wire.caps_override.empty() && stage_debug_enabled()) {
    std::fprintf(stderr, "[DBG] StageRun BoxDecode caps_override: %s\n",
                 wire.caps_override.c_str());
  }
  const WireInput wire_input = build_wire_input_from_tensor(input, wire);
  const InputOptions src_opt = wire_input.appsrc;

  StageKey key;
  key.kind = StageKind::BoxDecode;
  key.model_id = simaai::neat::internal::ModelAccess::model_id(model);
  key.input = make_input_key(src_opt);
  key.box_opt = opt;

  auto runner = get_or_build(key, [&]() {
    override_buffer_name(group, src_opt.buffer_name, "boxdecode");
    dump_config_json(group, "BoxDecode");
    auto input_node = simaai::neat::nodes::Input(src_opt);
    auto output_node = simaai::neat::nodes::Output();
    std::vector<std::shared_ptr<Node>> nodes;
    nodes.reserve(group.size() + 2);
    nodes.push_back(input_node);
    for (const auto& node : group.nodes()) {
      nodes.push_back(node);
    }
    nodes.push_back(output_node);

    pipeline_internal::PipelineBuildContext build_ctx(SessionOptions{});
    build_ctx.apply_name_transform_to_configs(nodes);
    build_ctx.wire_configs_by_order(nodes);

    Session p;
    p.add(input_node);
    p.add(group);
    p.add(output_node);
    return p.build(wire_input.tensor, RunMode::Sync, stage_run_defaults());
  });

  const int timeout_ms = default_timeout_ms();
  if (stage_debug_enabled()) {
    const int64_t actual_bytes = tensor_total_bytes(wire_input.tensor);
    const size_t plane0 = wire_input.tensor.planes.empty()
                              ? 0
                              : static_cast<size_t>(tensor_plane_bytes_tight(
                                    wire_input.tensor.planes[0], wire_input.tensor.dtype));
    std::fprintf(
        stderr,
        "[DBG] StageRun BoxDecode in: format=%s dtype=%s w=%d h=%d shape=%s plane0=%zu bytes=%lld "
        "expected_bytes=%lld buffer_size=%lld wire=%s/%s %dx%dx%d\n",
        format_from_tensor(wire_input.tensor).c_str(), dtype_name(wire_input.tensor.dtype),
        (wire_input.tensor.shape.size() > 1) ? static_cast<int>(wire_input.tensor.shape[1]) : -1,
        (wire_input.tensor.shape.size() > 0) ? static_cast<int>(wire_input.tensor.shape[0]) : -1,
        shape_string(wire_input.tensor.shape).c_str(), plane0, static_cast<long long>(actual_bytes),
        static_cast<long long>(box_expected.total_bytes),
        static_cast<long long>(box_expected.buffer_size), wire.media_type.c_str(),
        wire.format.c_str(), wire.dims.width, wire.dims.height, wire.dims.depth);
  }
  Sample out;
  const std::shared_ptr<void> holder = pipeline_internal::holder_from_tensor(input);
  if (holder) {
    // Preserve GstSimaMeta by reusing the original MLA buffer when possible.
    out = runner->push_and_pull_holder(holder, timeout_ms);
  } else {
    out = runner->push_and_pull(wire_input.tensor, timeout_ms);
  }
  simaai::neat::Tensor tensor = take_tensor(out, "BoxDecode");
  return decode_bbox_tensor(tensor, opt.original_width, opt.original_height, opt.top_k, true);
}

} // namespace simaai::neat::stages
