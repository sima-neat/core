#include "model/Model.h"
#include "model/internal/ModelInternal.h"
#include "model/internal/ModelPack.h"

#include "nodes/common/Output.h"
#include "nodes/io/Input.h"
#include "nodes/sima/DetessDequant.h"
#include "nodes/sima/Preproc.h"
#include "nodes/sima/QuantTess.h"
#include "nodes/sima/SimaBoxDecode.h"
#include "pipeline/Session.h"
#include "pipeline/internal/EnvUtil.h"
#include "pipeline/internal/InputPolicy.h"
#include "pipeline/internal/InputStreamUtil.h"
#include "pipeline/internal/StageConfig.h"
#include "pipeline/internal/TensorMath.h"

#include <nlohmann/json.hpp>

#include <algorithm>
#include <cctype>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <mutex>
#include <optional>
#include <sstream>
#include <stdexcept>
#include <utility>

namespace simaai::neat {
namespace {
using pipeline_internal::env_int;
using pipeline_internal::upper_copy;

bool env_bool(const char* name, bool def_val = false) {
  const char* env = std::getenv(name);
  if (!env || !*env)
    return def_val;
  std::string v(env);
  std::transform(v.begin(), v.end(), v.begin(),
                 [](unsigned char c) { return static_cast<char>(std::tolower(c)); });
  if (v == "0" || v == "false" || v == "no" || v == "off")
    return false;
  if (v == "1" || v == "true" || v == "yes" || v == "on")
    return true;
  return def_val ? def_val : true;
}

internal::InferenceTerminalPolicy
to_internal_terminal_policy(const Model::InferenceTerminalPolicy& policy) {
  internal::InferenceTerminalPolicy out;
  out.mla_only = policy.mla_only;
  out.last_stage_index = policy.last_stage_index;
  out.last_stage_name = policy.last_stage_name;
  out.last_plugin_id = policy.last_plugin_id;
  out.last_processor = policy.last_processor;
  return out;
}

std::vector<float> vec3_from_array(const std::optional<std::array<float, 3>>& arr,
                                   const std::vector<float>& fallback) {
  if (arr.has_value()) {
    return {(*arr)[0], (*arr)[1], (*arr)[2]};
  }
  return fallback;
}

pipeline_internal::ModelInputPolicyRequest make_model_input_policy_request(
    const Model::Options& opt) {
  pipeline_internal::ModelInputPolicyRequest req;
  req.format = opt.format;
  req.preproc_input_width = opt.preproc.input_width;
  req.preproc_input_height = opt.preproc.input_height;
  req.preproc_input_img_type = opt.preproc.input_img_type;
  req.preproc_normalize = opt.preproc.normalize;
  req.input_max_width = opt.input_max_width;
  req.input_max_height = opt.input_max_height;
  req.input_max_depth = opt.input_max_depth;
  return req;
}

pipeline_internal::ModelInputPolicyResult resolve_model_input_policy(
    const Model::Options& opt) {
  return pipeline_internal::resolve_model_input_policy(make_model_input_policy_request(opt));
}

int default_depth_for_format(const std::string& fmt) {
  return pipeline_internal::default_depth_for_image_format(fmt, -1);
}

std::string resolve_input_format(const Model::Options& opt) {
  return resolve_model_input_policy(opt).resolved_input_format;
}

int resolve_input_width(const Model::Options& opt) {
  return resolve_model_input_policy(opt).resolved_input_width;
}

int resolve_input_height(const Model::Options& opt) {
  return resolve_model_input_policy(opt).resolved_input_height;
}

int resolve_input_depth(const Model::Options& opt) {
  return resolve_model_input_policy(opt).resolved_input_depth;
}

int resolve_max_input_width(const Model::Options& opt) {
  return resolve_model_input_policy(opt).resolved_max_input_width;
}

int resolve_max_input_height(const Model::Options& opt) {
  return resolve_model_input_policy(opt).resolved_max_input_height;
}

int resolve_max_input_depth(const Model::Options& opt) {
  return resolve_model_input_policy(opt).resolved_max_input_depth;
}

bool resolve_normalize(const Model::Options& opt) {
  return resolve_model_input_policy(opt).resolved_normalize;
}

void warn_no_warmup_once() {
  static std::once_flag flag;
  std::call_once(flag, [] {
    std::printf(
        "[WARN] Model::Runner::warmup: warm=0; throughput stability may vary without warmup.\n");
  });
}

void enable_model_debug_env_once() {
  if (!env_bool("NEAT_MODEL_DEBUG", false))
    return;
  static std::once_flag flag;
  std::call_once(flag, []() {
    setenv("SIMA_GST_FLOW_DEBUG", "1", 1);
    setenv("SIMA_GST_ELEMENT_TIMINGS", "1", 1);
    setenv("SIMA_PIPELINE_STRING_DEBUG", "1", 1);
    setenv("SIMA_PULL_TIMEOUT_DIAG", "1", 1);
    setenv("SIMA_GST_RUN_INSERT_BOUNDARIES", "1", 1);
    setenv("SIMA_GST_BOUNDARY_PROBES", "1", 1);
    setenv("SIMA_GST_BOUNDARY_BUFFER_DEBUG", "1", 1);
    setenv("SIMA_GST_BUFFER_DEBUG_LIMIT", "10", 1);
    setenv("SIMA_GST_BUFFER_MEMFLAGS_DEBUG", "1", 1);
    setenv("SIMA_GST_OPTIONS_DEBUG", "1", 1);
    setenv("SIMA_GST_APPSINK_BUFFER_DEBUG", "1", 1);
    setenv("SIMA_GST_BOXDECODE_BUFFER_DEBUG", "1", 1);
    setenv("SIMA_GST_PAD_LINK_DEBUG", "1", 1);
    setenv("SIMA_INPUTSTREAM_META_DEBUG", "1", 1);
    setenv("SIMA_MLA_CONFIG_DEBUG", "1", 1);
    setenv("SIMA_BOXDECODE_WIRE_DEBUG", "1", 1);
    setenv("SIMA_PREPROC_DEBUG_CONFIG", "1", 1);
  });
}

std::string find_pre_adapter_name(const simaai::neat::mpk::SequenceSplit& split,
                                  const char* kernel) {
  if (!kernel || !*kernel)
    return {};
  for (const auto& entry : split.pre) {
    if (entry.kernel == kernel && !entry.name.empty())
      return entry.name;
  }
  return {};
}

std::string resolve_pre_name(const simaai::neat::mpk::SequenceSplit& split, bool use_preproc) {
  if (use_preproc) {
    std::string name = find_pre_adapter_name(split, "preproc");
    if (name.empty())
      name = "preproc";
    return name;
  }
  std::string name = find_pre_adapter_name(split, "quanttess");
  if (name.empty())
    name = "quanttess";
  return name;
}

TensorDType dtype_from_format(const std::string& fmt) {
  const std::string up = upper_copy(fmt);
  if (up.find("UINT8") != std::string::npos)
    return TensorDType::UInt8;
  if (up.find("INT8") != std::string::npos)
    return TensorDType::Int8;
  if (up.find("UINT16") != std::string::npos)
    return TensorDType::UInt16;
  if (up.find("INT16") != std::string::npos)
    return TensorDType::Int16;
  if (up.find("INT32") != std::string::npos)
    return TensorDType::Int32;
  if (up.find("BF16") != std::string::npos || up.find("BFLOAT16") != std::string::npos ||
      up.find("FP16") != std::string::npos) {
    return TensorDType::BFloat16;
  }
  if (up.find("FP64") != std::string::npos)
    return TensorDType::Float64;
  return TensorDType::Float32;
}

ImageSpec::PixelFormat pixel_format_from_string(const std::string& fmt) {
  const std::string up = upper_copy(fmt);
  if (up == "RGB")
    return ImageSpec::PixelFormat::RGB;
  if (up == "BGR")
    return ImageSpec::PixelFormat::BGR;
  if (up == "GRAY" || up == "GRAY8")
    return ImageSpec::PixelFormat::GRAY8;
  if (up == "NV12")
    return ImageSpec::PixelFormat::NV12;
  if (up == "I420")
    return ImageSpec::PixelFormat::I420;
  return ImageSpec::PixelFormat::UNKNOWN;
}

bool is_gray_format(const std::string& fmt) {
  const std::string up = upper_copy(fmt);
  return up == "GRAY" || up == "GRAY8";
}

std::string image_format_name(simaai::neat::ImageSpec::PixelFormat fmt) {
  switch (fmt) {
  case ImageSpec::PixelFormat::RGB:
    return "RGB";
  case ImageSpec::PixelFormat::BGR:
    return "BGR";
  case ImageSpec::PixelFormat::GRAY8:
    return "GRAY8";
  case ImageSpec::PixelFormat::NV12:
    return "NV12";
  case ImageSpec::PixelFormat::I420:
    return "I420";
  case ImageSpec::PixelFormat::UNKNOWN:
    break;
  }
  return "";
}

std::vector<int64_t> shape_from_mla_output(const stages::MlaOutputInfo& info) {
  if (info.dims.width <= 0 || info.dims.height <= 0 || info.dims.depth <= 0) {
    return {};
  }
  if (info.layout == TensorLayout::CHW) {
    return {info.dims.depth, info.dims.height, info.dims.width};
  }
  if (info.layout == TensorLayout::HW) {
    return {info.dims.height, info.dims.width};
  }
  return {info.dims.height, info.dims.width, info.dims.depth};
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

struct InputInfo {
  std::string media_type;
  std::string format;
  int width = 0;
  int height = 0;
  int depth = 0;
};

struct InputKey {
  std::string media_type;
  std::string format;
  int width = 0;
  int height = 0;
  int depth = 0;
};

bool operator==(const InputKey& a, const InputKey& b) {
  return a.media_type == b.media_type && a.format == b.format && a.width == b.width &&
         a.height == b.height && a.depth == b.depth;
}

InputKey input_key_from_appsrc(const InputOptions& opt) {
  InputKey key;
  key.media_type = opt.media_type;
  key.format = upper_copy(opt.format);
  key.width = opt.width;
  key.height = opt.height;
  key.depth = opt.depth;
  return key;
}

#if defined(SIMA_WITH_OPENCV)
InputInfo input_info_from_mat(const cv::Mat& input) {
  InputInfo info;
  info.media_type = "video/x-raw";
  info.width = input.cols;
  info.height = input.rows;
  const int channels = input.channels();
  info.depth = channels;
  info.format = (channels == 1) ? "GRAY8" : "BGR";
  return info;
}
#endif

int64_t shape_dim(const std::vector<int64_t>& shape, size_t index) {
  if (index >= shape.size())
    return -1;
  return shape[index];
}

InputInfo input_info_from_tensor(const simaai::neat::Tensor& tensor, bool image_mode) {
  InputInfo info;
  if (image_mode) {
    info.media_type = "video/x-raw";
    if (tensor.semantic.image.has_value()) {
      info.format = image_format_name(tensor.semantic.image->format);
    }
  } else {
    info.media_type = "application/vnd.simaai.tensor";
    info.format = format_from_tensor(tensor);
  }

  if (info.format.empty()) {
    if (image_mode) {
      info.format = "RGB";
    } else if (tensor.dtype == TensorDType::Float32) {
      info.format = "FP32";
    } else if (tensor.dtype == TensorDType::Int8) {
      info.format = "INT8";
    } else if (tensor.dtype == TensorDType::UInt8) {
      info.format = "UINT8";
    }
  }

  const int64_t shape_h = shape_dim(tensor.shape, 0);
  const int64_t shape_w = shape_dim(tensor.shape, 1);
  const int64_t shape_d = shape_dim(tensor.shape, 2);
  if (tensor.is_composite() && !tensor.planes.empty()) {
    const auto& y = tensor.planes.front();
    if (y.shape.size() >= 2) {
      info.height = (shape_h > 0) ? static_cast<int>(shape_h) : static_cast<int>(y.shape[0]);
      info.width = (shape_w > 0) ? static_cast<int>(shape_w) : static_cast<int>(y.shape[1]);
    }
  } else {
    info.height = (shape_h > 0) ? static_cast<int>(shape_h) : 0;
    info.width = (shape_w > 0) ? static_cast<int>(shape_w) : 0;
  }
  if (shape_d > 0)
    info.depth = static_cast<int>(shape_d);
  return info;
}

InputOptions appsrc_from_info(const InputInfo& info) {
  InputOptions opt;
  opt.media_type = info.media_type;
  opt.format = info.format;
  opt.width = info.width;
  opt.height = info.height;
  opt.depth = info.depth;
  return opt;
}

Tensor make_dummy_tensor(const simaai::neat::InputOptions& opt) {
  Tensor t;
  t.device = {DeviceType::CPU, 0};
  t.read_only = true;

  if (opt.media_type == "application/vnd.simaai.tensor") {
    const int w = (opt.width > 0) ? opt.width : opt.max_width;
    const int h = (opt.height > 0) ? opt.height : opt.max_height;
    const int d = (opt.depth > 0) ? opt.depth : opt.max_depth;
    if (w <= 0 || h <= 0 || d <= 0) {
      throw std::runtime_error(
          "Model::build: missing tensor input shape for dummy input. "
          "Fix: set Model::Options::input_max_width/input_max_height/input_max_depth.");
    }
    t.dtype = dtype_from_format(opt.format);
    t.layout = TensorLayout::HWC;
    t.shape = {h, w, d};
    const std::size_t elem = pipeline_internal::dtype_bytes(t.dtype);
    const std::size_t bytes = static_cast<std::size_t>(w) * static_cast<std::size_t>(h) *
                              static_cast<std::size_t>(d) * elem;
    t.storage = make_cpu_owned_storage(bytes);
    t.strides_bytes = pipeline_internal::contiguous_strides_bytes(t.shape, elem);
    return t;
  }

  const int w = (opt.width > 0) ? opt.width : opt.max_width;
  const int h = (opt.height > 0) ? opt.height : opt.max_height;
  if (w <= 0 || h <= 0) {
    throw std::runtime_error("Model::build: missing image input shape for dummy input. "
                             "Fix: set Model::Options::input_max_width/input_max_height.");
  }
  std::string fmt = opt.format;
  if (fmt.empty())
    fmt = "RGB";
  const auto pix = pixel_format_from_string(fmt);
  if (pix == ImageSpec::PixelFormat::UNKNOWN) {
    throw std::runtime_error("Model::build: unsupported image format");
  }

  t.dtype = TensorDType::UInt8;
  t.semantic.image = ImageSpec{pix, ""};

  if (pix == ImageSpec::PixelFormat::NV12) {
    const std::size_t y_size = static_cast<std::size_t>(w * h);
    const std::size_t uv_size = static_cast<std::size_t>(w * h / 2);
    t.storage = make_cpu_owned_storage(y_size + uv_size);
    t.layout = TensorLayout::HW;
    t.shape = {h, w};

    Plane y;
    y.role = PlaneRole::Y;
    y.shape = {h, w};
    y.strides_bytes = {w, 1};
    y.byte_offset = 0;

    Plane uv;
    uv.role = PlaneRole::UV;
    uv.shape = {h / 2, w};
    uv.strides_bytes = {w, 1};
    uv.byte_offset = static_cast<int64_t>(y_size);

    t.planes = {y, uv};
    return t;
  }

  if (pix == ImageSpec::PixelFormat::I420) {
    const std::size_t y_size = static_cast<std::size_t>(w * h);
    const std::size_t uv_size = static_cast<std::size_t>(w * h / 4);
    t.storage = make_cpu_owned_storage(y_size + uv_size * 2);
    t.layout = TensorLayout::HW;
    t.shape = {h, w};

    Plane y;
    y.role = PlaneRole::Y;
    y.shape = {h, w};
    y.strides_bytes = {w, 1};
    y.byte_offset = 0;

    Plane u;
    u.role = PlaneRole::U;
    u.shape = {h / 2, w / 2};
    u.strides_bytes = {w / 2, 1};
    u.byte_offset = static_cast<int64_t>(y_size);

    Plane v;
    v.role = PlaneRole::V;
    v.shape = {h / 2, w / 2};
    v.strides_bytes = {w / 2, 1};
    v.byte_offset = static_cast<int64_t>(y_size + uv_size);

    t.planes = {y, u, v};
    return t;
  }

  const int depth = (pix == ImageSpec::PixelFormat::GRAY8) ? 1 : 3;
  t.layout = (depth == 1) ? TensorLayout::HW : TensorLayout::HWC;
  t.shape = (depth == 1) ? std::vector<int64_t>{h, w} : std::vector<int64_t>{h, w, depth};
  const std::size_t bytes =
      static_cast<std::size_t>(w) * static_cast<std::size_t>(h) * static_cast<std::size_t>(depth);
  t.storage = make_cpu_owned_storage(bytes);
  t.strides_bytes = pipeline_internal::contiguous_strides_bytes(t.shape, 1);
  return t;
}

simaai::neat::Sample make_bundle_from_tensors(const std::vector<Tensor>& inputs) {
  simaai::neat::Sample bundle;
  bundle.kind = simaai::neat::SampleKind::Bundle;
  bundle.owned = true;
  bundle.fields.reserve(inputs.size());
  for (size_t i = 0; i < inputs.size(); ++i) {
    simaai::neat::Sample field;
    field.kind = simaai::neat::SampleKind::Tensor;
    field.owned = true;
    field.tensor = inputs[i];
    field.port_name = "input" + std::to_string(i);
    bundle.fields.emplace_back(std::move(field));
  }
  return bundle;
}

struct Range {
  int min = 0;
  int max = 0;
  bool valid = false;
};

struct CapsInfo {
  Range width;
  Range height;
  Range depth;
  std::vector<std::string> formats;
};

std::vector<int> parse_ints(const std::string& s) {
  std::vector<int> nums;
  int cur = 0;
  bool in_num = false;
  for (char c : s) {
    if (std::isdigit(static_cast<unsigned char>(c))) {
      cur = cur * 10 + (c - '0');
      in_num = true;
    } else if (in_num) {
      nums.push_back(cur);
      cur = 0;
      in_num = false;
    }
  }
  if (in_num)
    nums.push_back(cur);
  return nums;
}

Range parse_range(const nlohmann::json& values) {
  Range out;
  std::string s;
  if (values.is_string()) {
    s = values.get<std::string>();
  } else if (values.is_number_integer()) {
    out.min = out.max = values.get<int>();
    out.valid = true;
    return out;
  } else {
    return out;
  }
  const std::vector<int> nums = parse_ints(s);
  if (nums.empty())
    return out;
  out.min = nums[0];
  out.max = (nums.size() > 1) ? nums[1] : nums[0];
  if (out.max < out.min)
    std::swap(out.max, out.min);
  out.valid = true;
  return out;
}

std::vector<std::string> parse_value_list(const nlohmann::json& values) {
  std::vector<std::string> out;
  if (values.is_string()) {
    std::string s = values.get<std::string>();
    std::stringstream ss(s);
    std::string tok;
    while (std::getline(ss, tok, ',')) {
      tok.erase(tok.begin(), std::find_if(tok.begin(), tok.end(),
                                          [](unsigned char ch) { return !std::isspace(ch); }));
      tok.erase(
          std::find_if(tok.rbegin(), tok.rend(), [](unsigned char ch) { return !std::isspace(ch); })
              .base(),
          tok.end());
      if (!tok.empty())
        out.push_back(upper_copy(tok));
    }
    return out;
  }
  if (values.is_array()) {
    for (const auto& v : values) {
      if (v.is_string())
        out.push_back(upper_copy(v.get<std::string>()));
    }
  }
  return out;
}

CapsInfo parse_caps_info(const nlohmann::json& cfg) {
  CapsInfo out;
  if (!cfg.contains("caps") || !cfg["caps"].is_object())
    return out;
  const auto& caps = cfg["caps"];
  if (!caps.contains("sink_pads") || !caps["sink_pads"].is_array() || caps["sink_pads"].empty()) {
    return out;
  }
  const auto& pad = caps["sink_pads"][0];
  if (!pad.is_object() || !pad.contains("params") || !pad["params"].is_array()) {
    return out;
  }
  for (const auto& param : pad["params"]) {
    if (!param.is_object())
      continue;
    const std::string name = param.value("name", "");
    if (name.empty())
      continue;
    if (!param.contains("values"))
      continue;
    if (name == "width")
      out.width = parse_range(param["values"]);
    else if (name == "height")
      out.height = parse_range(param["values"]);
    else if (name == "depth")
      out.depth = parse_range(param["values"]);
    else if (name == "format")
      out.formats = parse_value_list(param["values"]);
  }
  return out;
}

void validate_range(const Range& range, int value, const char* label, const char* field) {
  if (!range.valid || value <= 0)
    return;
  if (value < range.min || value > range.max) {
    std::ostringstream ss;
    ss << "Model: " << label << " " << field << " out of bounds: " << value << " (allowed "
       << range.min << "-" << range.max << ")";
    throw std::runtime_error(ss.str());
  }
}

void validate_format(const std::vector<std::string>& formats, const std::string& fmt,
                     const char* label) {
  if (formats.empty() || fmt.empty())
    return;
  const std::string up = upper_copy(fmt);
  for (const auto& allowed : formats) {
    if (allowed == up)
      return;
    if ((allowed == "GRAY" || allowed == "GRAY8") && (up == "GRAY" || up == "GRAY8"))
      return;
  }
  std::ostringstream ss;
  ss << "Model: " << label << " format not allowed: " << fmt;
  throw std::runtime_error(ss.str());
}

void validate_input_against_caps(const InputInfo& info, const nlohmann::json& cfg,
                                 const char* label) {
  CapsInfo caps = parse_caps_info(cfg);
  Range default_range{1, 4096, true};
  if (!caps.width.valid)
    caps.width = default_range;
  if (!caps.height.valid)
    caps.height = default_range;
  if (!caps.depth.valid && info.depth > 0)
    caps.depth = default_range;
  validate_range(caps.width, info.width, label, "width");
  validate_range(caps.height, info.height, label, "height");
  validate_range(caps.depth, info.depth, label, "depth");
  validate_format(caps.formats, info.format, label);
}

std::vector<float> ensure_three(const std::vector<float>& v, float def_val) {
  if (v.empty())
    return {def_val, def_val, def_val};
  if (v.size() == 1)
    return {v[0], v[0], v[0]};
  if (v.size() >= 3)
    return {v[0], v[1], v[2]};
  std::vector<float> out = v;
  while (out.size() < 3)
    out.push_back(def_val);
  return out;
}

bool split_has_kernel(const simaai::neat::mpk::SequenceSplit& split, const char* kernel) {
  if (!kernel || !*kernel)
    return false;
  for (const auto& entry : split.post) {
    if (entry.kernel == kernel)
      return true;
  }
  return false;
}

} // namespace

struct Model::Impl {
  Options options;
  internal::ModelPack pack;
  mutable std::optional<internal::ModelPack> sync_pack;
  std::string model_id;

  mutable std::mutex sync_mu;
  mutable bool sync_ready = false;
  mutable InputKey sync_key{};
  mutable Runner sync_runner{};

  Impl(const std::string& tar_gz, Options opt)
      : options(std::move(opt)),
        pack(tar_gz, options.media_type, resolve_input_format(options),
             resolve_input_width(options), resolve_input_height(options),
             resolve_input_depth(options), resolve_max_input_width(options),
             resolve_max_input_height(options), resolve_max_input_depth(options),
             resolve_normalize(options),
             vec3_from_array(options.preproc.channel_mean, {0.0f, 0.0f, 0.0f}),
             vec3_from_array(options.preproc.channel_stddev, {1.0f, 1.0f, 1.0f}),
             /*preproc_next_cpu=*/{}, options.upstream_name,
             /*num_buffers_cvu=*/4,
             /*num_buffers_mla=*/4,
             /*queue_max_buffers=*/0,
             /*queue_max_time_ns=*/-1,
             /*queue_leaky=*/{}, options.name_suffix,
             to_internal_terminal_policy(options.inference_terminal)) {
    model_id = pack.etc_dir();
    if (!options.name_suffix.empty()) {
      model_id += "::" + options.name_suffix;
    }
  }

  const internal::ModelPack& pack_for_sync() const {
    if (!sync_pack.has_value()) {
      sync_pack = pack.clone_with_buffers(1, 1);
    }
    return *sync_pack;
  }
};

Model::Model(const std::string& mpk_path) : Model(mpk_path, Options{}) {}

Model::Model(const std::string& mpk_path, const Options& opt) {
  Options normalized = opt;
  const auto policy = resolve_model_input_policy(normalized);
  normalized.input_max_width = policy.resolved_max_input_width;
  normalized.input_max_height = policy.resolved_max_input_height;
  normalized.input_max_depth = policy.resolved_max_input_depth;
  impl_ = std::make_unique<Impl>(mpk_path, std::move(normalized));
  enable_model_debug_env_once();
}

Model::Model(Model&&) noexcept = default;
Model& Model::operator=(Model&&) noexcept = default;
Model::~Model() = default;

namespace {

void apply_preproc_overrides(PreprocOptions& pre_opt, const Model::Options& opt,
                             const InputInfo* input) {
  if (!pre_opt.config_json.has_value())
    return;
  nlohmann::json cfg = *pre_opt.config_json;
  const Model::PreprocConfig& pre_cfg = opt.preproc;

  const int observed_w = input ? input->width : 0;
  const int observed_h = input ? input->height : 0;
  const int max_w = (opt.input_max_width > 0) ? opt.input_max_width : observed_w;
  const int max_h = (opt.input_max_height > 0) ? opt.input_max_height : observed_h;
  const int in_w = pre_cfg.input_width.value_or(max_w);
  const int in_h = pre_cfg.input_height.value_or(max_h);
  const std::string in_fmt =
      pre_cfg.input_img_type.value_or(input ? upper_copy(input->format) : std::string{});
  if (in_w > 0)
    cfg["input_width"] = in_w;
  if (in_h > 0)
    cfg["input_height"] = in_h;
  if (!in_fmt.empty())
    cfg["input_img_type"] = upper_copy(in_fmt);

  if (pre_cfg.output_width.has_value())
    cfg["output_width"] = *pre_cfg.output_width;
  if (pre_cfg.output_height.has_value())
    cfg["output_height"] = *pre_cfg.output_height;
  if (pre_cfg.scaled_width.has_value())
    cfg["scaled_width"] = *pre_cfg.scaled_width;
  if (pre_cfg.scaled_height.has_value())
    cfg["scaled_height"] = *pre_cfg.scaled_height;
  if (pre_cfg.output_img_type.has_value()) {
    cfg["output_img_type"] = upper_copy(*pre_cfg.output_img_type);
  }

  const bool has_norm_stats =
      pre_cfg.channel_mean.has_value() || pre_cfg.channel_stddev.has_value();
  const bool enable_norm = (pre_cfg.normalize.has_value() && *pre_cfg.normalize) || has_norm_stats;
  if (enable_norm) {
    cfg["normalize"] = true;
    const std::vector<float> mean3 =
        pre_cfg.channel_mean.has_value()
            ? std::vector<float>{(*pre_cfg.channel_mean)[0], (*pre_cfg.channel_mean)[1],
                                 (*pre_cfg.channel_mean)[2]}
            : ensure_three({}, 0.0f);
    const std::vector<float> std3 =
        pre_cfg.channel_stddev.has_value()
            ? std::vector<float>{(*pre_cfg.channel_stddev)[0], (*pre_cfg.channel_stddev)[1],
                                 (*pre_cfg.channel_stddev)[2]}
            : ensure_three({}, 1.0f);
    cfg["channel_mean"] = mean3;
    cfg["channel_stddev"] = std3;
  }

  if (pre_cfg.scaling_type.has_value())
    cfg["scaling_type"] = *pre_cfg.scaling_type;

  if (pre_cfg.padding_type.has_value())
    cfg["padding_type"] = *pre_cfg.padding_type;

  if (pre_cfg.aspect_ratio.has_value())
    cfg["aspect_ratio"] = *pre_cfg.aspect_ratio;
  pre_opt.config_json = std::move(cfg);
}

void apply_quanttess_overrides(QuantTessOptions& qt_opt, const InputInfo* input) {
  if (!qt_opt.config_json.has_value() || !input)
    return;
  nlohmann::json cfg = *qt_opt.config_json;
  if (input->width > 0)
    cfg["input_width"] = input->width;
  if (input->height > 0)
    cfg["input_height"] = input->height;
  if (input->depth > 0)
    cfg["input_depth"] = input->depth;
  qt_opt.config_json = std::move(cfg);
}

NodeGroup build_preprocess_group_impl(const Model& model, const internal::ModelPack& pack,
                                      const Model::Options& opt, const InputInfo* input,
                                      const std::string& element_name,
                                      const std::string& upstream_name, bool sync) {
  const bool use_preproc = pack.pipeline_type() == internal::PipelineType::Preproc;
  if (use_preproc) {
    PreprocOptions pre_opt(model);
    pre_opt.element_name = element_name;
    if (!upstream_name.empty())
      pre_opt.upstream_name = upstream_name;
    if (sync) {
      pre_opt.num_buffers = 1;
      pre_opt.num_buffers_model = 1;
      pre_opt.num_buffers_locked = true;
    }
    apply_preproc_overrides(pre_opt, opt, input);
    if (pre_opt.config_json.has_value() && input) {
      validate_input_against_caps(*input, *pre_opt.config_json, "preproc");
    }
    return NodeGroup({simaai::neat::nodes::Preproc(std::move(pre_opt))});
  }

  QuantTessOptions qt_opt(model);
  qt_opt.element_name = element_name;
  if (sync) {
    qt_opt.num_buffers = 1;
    qt_opt.num_buffers_model = 1;
    qt_opt.num_buffers_locked = true;
  }
  apply_quanttess_overrides(qt_opt, input);
  if (qt_opt.config_json.has_value() && input) {
    validate_input_against_caps(*input, *qt_opt.config_json, "quanttess");
  }
  return NodeGroup({simaai::neat::nodes::QuantTess(std::move(qt_opt))});
}

NodeGroup build_postprocess_group(const Model& model, const internal::ModelPack& pack,
                                  const Model::Options& opt, bool sync) {
  const auto split = pack.split_sequence();
  if (split_has_kernel(split, "boxdecode")) {
    return NodeGroup({simaai::neat::nodes::SimaBoxDecode(model, opt.decode_type, opt.original_width,
                                                         opt.original_height, opt.score_threshold,
                                                         opt.nms_iou_threshold, opt.top_k)});
  }
  if (split_has_kernel(split, "detessdequant")) {
    DetessDequantOptions det_opt(model);
    if (sync) {
      det_opt.num_buffers = 1;
      det_opt.num_buffers_model = 1;
      det_opt.num_buffers_locked = true;
    }
    return NodeGroup({simaai::neat::nodes::DetessDequant(std::move(det_opt))});
  }
  return NodeGroup{};
}

InputInfo require_input_info(const InputInfo& info, bool tensor_mode) {
  if (info.width <= 0 || info.height <= 0) {
    throw std::runtime_error("Model: input missing width/height");
  }
  if (info.format.empty()) {
    throw std::runtime_error("Model: input missing format");
  }
  if (tensor_mode && info.depth <= 0) {
    throw std::runtime_error("Model: tensor input missing depth");
  }
  return info;
}

NodeGroup build_pipeline_group(const Model& model, const internal::ModelPack& pack,
                               const Model::Options& opt, Model::SessionOptions popt,
                               const InputInfo* input, bool sync) {
  const bool use_preproc = pack.pipeline_type() == internal::PipelineType::Preproc;
  const auto split = pack.split_sequence();
  const std::string pre_name = pack.apply_name_suffix(resolve_pre_name(split, use_preproc));

  std::vector<std::shared_ptr<Node>> nodes;

  InputOptions src_opt;
  if (popt.include_appsrc) {
    if (input) {
      src_opt = appsrc_from_info(require_input_info(*input, !use_preproc));
    } else {
      src_opt = pack.input_appsrc_options(!use_preproc);
    }
    if (!popt.buffer_name.empty()) {
      src_opt.buffer_name = popt.buffer_name;
    } else if (!popt.upstream_name.empty()) {
      src_opt.buffer_name = popt.upstream_name;
    } else if (!popt.name_suffix.empty()) {
      src_opt.buffer_name = std::string("decoder") + popt.name_suffix;
    }
    nodes.push_back(simaai::neat::nodes::Input(src_opt));
  }

  const std::string upstream =
      popt.upstream_name.empty() ? (pre_name.empty() ? "decoder" : pre_name) : popt.upstream_name;

  const std::string pre_upstream =
      (popt.include_appsrc && !src_opt.buffer_name.empty()) ? src_opt.buffer_name : upstream;
  NodeGroup pre_group =
      build_preprocess_group_impl(model, pack, opt, input, pre_name, pre_upstream, sync);
  for (const auto& node : pre_group.nodes()) {
    nodes.push_back(node);
  }

  NodeGroup infer = pack.infer_block(upstream);
  for (const auto& node : infer.nodes()) {
    nodes.push_back(node);
  }

  NodeGroup post = build_postprocess_group(model, pack, opt, sync);
  for (const auto& node : post.nodes()) {
    nodes.push_back(node);
  }

  if (popt.include_appsink) {
    nodes.push_back(simaai::neat::nodes::Output());
  }

  return NodeGroup(std::move(nodes));
}

} // namespace

NodeGroup Model::preprocess() const {
  const auto& pack = impl_->pack;
  const auto split = pack.split_sequence();
  const bool use_preproc = pack.pipeline_type() == internal::PipelineType::Preproc;
  const std::string pre_name = pack.apply_name_suffix(resolve_pre_name(split, use_preproc));
  return build_preprocess_group_impl(*this, pack, impl_->options, nullptr, pre_name, "decoder",
                                     false);
}

NodeGroup Model::inference() const {
  const auto& pack = impl_->pack;
  const auto split = pack.split_sequence();
  const bool use_preproc = pack.pipeline_type() == internal::PipelineType::Preproc;
  const std::string pre_name = pack.apply_name_suffix(resolve_pre_name(split, use_preproc));
  const std::string upstream = pre_name.empty() ? "decoder" : pre_name;
  return pack.infer_block(upstream);
}

NodeGroup Model::postprocess() const {
  const auto& pack = impl_->pack;
  return build_postprocess_group(*this, pack, impl_->options, false);
}

NodeGroup Model::session() const {
  return session(Model::SessionOptions{});
}

NodeGroup Model::session(Model::SessionOptions opt) const {
  internal::ModelPack pack = impl_->pack;
  if (!opt.name_suffix.empty()) {
    pack = pack.clone_with_overrides(std::string{}, opt.name_suffix);
  }
  return build_pipeline_group(*this, pack, impl_->options, std::move(opt), nullptr, false);
}

TensorSpec Model::input_spec() const {
  TensorSpec spec;
  const bool tensor_mode = impl_->pack.pipeline_type() == internal::PipelineType::QuantTess;
  if (tensor_mode) {
    const InputOptions opt = impl_->pack.input_appsrc_options(true);
    const auto split = impl_->pack.split_sequence();
    const bool has_cvu_frontend = !split.pre.empty();
    const auto mla_input = stages::read_mla_input_info(inference());
    int d = (mla_input.dims.depth > 0) ? mla_input.dims.depth : 0;
    if (d <= 0)
      d = (opt.depth > 0) ? opt.depth : opt.max_depth;
    if (d <= 0)
      d = 3;
    spec.dtypes = {TensorDType::Float32};
    if (!has_cvu_frontend && mla_input.dims.width > 0 && mla_input.dims.height > 0 && d > 0) {
      spec.shape = {mla_input.dims.height, mla_input.dims.width, d};
    } else {
      // CVU-fronted model inputs are dynamic at runtime. input_width/input_height
      // in stage JSON are sizing limits, not fixed input geometry.
      spec.shape = {-1, -1, d > 0 ? d : -1};
    }
  } else {
    spec.dtypes = {TensorDType::UInt8};
    const InputOptions opt = impl_->pack.input_appsrc_options(false);
    if (is_gray_format(opt.format)) {
      spec.shape = {-1, -1};
    } else {
      int depth = (opt.depth > 0) ? opt.depth : opt.max_depth;
      if (depth <= 0)
        depth = default_depth_for_format(opt.format);
      spec.shape = {-1, -1, depth > 0 ? depth : -1};
    }
    spec.image_format = pixel_format_from_string(opt.format);
  }
  if (!spec.shape.empty())
    spec.rank = static_cast<int>(spec.shape.size());
  return spec;
}

TensorSpec Model::output_spec() const {
  TensorSpec spec;
  const auto split = impl_->pack.split_sequence();
  const bool has_box = split_has_kernel(split, "boxdecode");
  const bool has_detess = split_has_kernel(split, "detessdequant");
  if (has_box) {
    spec.dtypes = {TensorDType::UInt8};
    spec.rank = -1;
    return spec;
  }
  if (has_detess) {
    NodeGroup post = build_postprocess_group(*this, impl_->pack, impl_->options, false);
    const auto info = stages::read_detessdequant_output_info(post);
    if (!info.outputs.empty()) {
      spec.dtypes = {info.dtype};
      spec.shape = info.outputs.front().shape;
      spec.rank = static_cast<int>(spec.shape.size());
    }
    return spec;
  }
  NodeGroup infer = inference();
  const auto mla_info = stages::read_mla_output_info(infer);
  spec.dtypes = {dtype_from_format(mla_info.data_type)};
  spec.shape = shape_from_mla_output(mla_info);
  if (!spec.shape.empty()) {
    spec.rank = static_cast<int>(spec.shape.size());
  }
  return spec;
}

std::unordered_map<std::string, std::string> Model::metadata() const {
  std::unordered_map<std::string, std::string> out;
  const std::string path = impl_->pack.etc_dir() + "/metadata.json";
  std::ifstream in(path);
  if (!in.is_open())
    return out;
  nlohmann::json j;
  in >> j;
  if (!j.is_object())
    return out;
  for (auto it = j.begin(); it != j.end(); ++it) {
    if (it.value().is_string()) {
      out[it.key()] = it.value().get<std::string>();
    } else {
      out[it.key()] = it.value().dump();
    }
  }
  return out;
}

NodeGroup Model::fragment(Stage stage) const {
  internal::ModelStage s = internal::ModelStage::Full;
  switch (stage) {
  case Stage::Preprocess:
    s = internal::ModelStage::Preprocess;
    break;
  case Stage::Inference:
    s = internal::ModelStage::MlaOnly;
    break;
  case Stage::Postprocess:
    s = internal::ModelStage::Postprocess;
    break;
  case Stage::Full:
    s = internal::ModelStage::Full;
    break;
  }
  return impl_->pack.to_node_group(s);
}

std::string Model::backend_fragment(Stage stage) const {
  internal::ModelStage s = internal::ModelStage::Full;
  switch (stage) {
  case Stage::Preprocess:
    s = internal::ModelStage::Preprocess;
    break;
  case Stage::Inference:
    s = internal::ModelStage::MlaOnly;
    break;
  case Stage::Postprocess:
    s = internal::ModelStage::Postprocess;
    break;
  case Stage::Full:
    s = internal::ModelStage::Full;
    break;
  }
  return impl_->pack.backend_fragment(s);
}

InputOptions Model::input_appsrc_options(bool tensor_mode) const {
  return impl_->pack.input_appsrc_options(tensor_mode);
}

std::string Model::find_config_path_by_plugin(const std::string& plugin_id) const {
  return impl_->pack.find_config_path_by_plugin(plugin_id);
}

std::string Model::find_config_path_by_processor(const std::string& processor) const {
  return impl_->pack.find_config_path_by_processor(processor);
}

std::string Model::infer_output_name() const {
  const auto frag = impl_->pack.fragment(internal::ModelStage::MlaOnly);
  if (frag.elements.empty())
    return {};
  return frag.elements.back();
}

const Model::SessionOptions& Model::default_session_options() {
  static const Model::SessionOptions opt{};
  return opt;
}

const simaai::neat::RunOptions& Model::default_run_options() {
  static const simaai::neat::RunOptions opt{};
  return opt;
}

Model::Runner::Runner(simaai::neat::Run run) : run_(std::move(run)) {}

Model::Runner::operator bool() const noexcept {
  return static_cast<bool>(run_);
}

#if defined(SIMA_WITH_OPENCV)
bool Model::Runner::push(const cv::Mat& input) {
  return run_.push(input);
}
#endif

bool Model::Runner::push(const simaai::neat::Tensor& input) {
  return run_.push(input);
}

bool Model::Runner::push(const simaai::neat::Sample& input) {
  return run_.push(input);
}

std::optional<simaai::neat::Sample> Model::Runner::pull(int timeout_ms) {
  return run_.pull(timeout_ms);
}

#if defined(SIMA_WITH_OPENCV)
simaai::neat::Sample Model::Runner::run(const cv::Mat& input, int timeout_ms) {
  return run_.run(input, timeout_ms);
}
#endif

simaai::neat::Sample Model::Runner::run(const simaai::neat::Tensor& input, int timeout_ms) {
  return run_.run(input, timeout_ms);
}

simaai::neat::Sample Model::Runner::run(const simaai::neat::Sample& input, int timeout_ms) {
  return run_.run(input, timeout_ms);
}

int Model::Runner::warmup(const simaai::neat::Tensor& input, int warm, int timeout_ms) {
  if (warm < 0) {
    warm = env_int("SIMA_ASYNC_WARMUP", 0);
  }
  if (warm <= 0) {
    warn_no_warmup_once();
    return 0;
  }
  for (int i = 0; i < warm; ++i) {
    (void)run_.run(input, timeout_ms);
  }
  return warm;
}

void Model::Runner::close() {
  run_.close();
}

Model::Runner Model::build() {
  return build(Model::SessionOptions{});
}

Model::Runner Model::build(const Model::SessionOptions& opt) {
  const bool tensor_mode = impl_->pack.pipeline_type() == internal::PipelineType::QuantTess;
  const InputOptions src_opt = impl_->pack.input_appsrc_options(tensor_mode);
  const Tensor dummy = make_dummy_tensor(src_opt);
  return build(dummy, opt, RunOptions{});
}

Model::Runner Model::build(const simaai::neat::Tensor& input, const Model::SessionOptions& opt,
                           const simaai::neat::RunOptions& run_opt) {
  internal::ModelPack pack = impl_->pack;
  if (!opt.name_suffix.empty()) {
    pack = pack.clone_with_overrides(std::string{}, opt.name_suffix);
  }
  const bool tensor_mode = pack.pipeline_type() == internal::PipelineType::QuantTess;
  InputInfo info = input_info_from_tensor(input, !tensor_mode);
  info = require_input_info(info, tensor_mode);
  NodeGroup group = build_pipeline_group(*this, pack, impl_->options, opt, &info, false);
  Session p;
  p.add(group);
  Run run = p.build(input, RunMode::Async, run_opt);
  return Runner(std::move(run));
}

Model::Runner Model::build(const simaai::neat::Sample& input, const Model::SessionOptions& opt,
                           const simaai::neat::RunOptions& run_opt) {
  if (input.kind == SampleKind::Tensor && input.tensor.has_value()) {
    return build(*input.tensor, opt, run_opt);
  }
  internal::ModelPack pack = impl_->pack;
  if (!opt.name_suffix.empty()) {
    pack = pack.clone_with_overrides(std::string{}, opt.name_suffix);
  }
  NodeGroup group = build_pipeline_group(*this, pack, impl_->options, opt, nullptr, false);
  Session p;
  p.add(group);
  Run run = p.build(input, RunMode::Async, run_opt);
  return Runner(std::move(run));
}

#if defined(SIMA_WITH_OPENCV)
Model::Runner Model::build(const cv::Mat& input, const Model::SessionOptions& opt,
                           const simaai::neat::RunOptions& run_opt) {
  const bool tensor_mode = impl_->pack.pipeline_type() == internal::PipelineType::QuantTess;
  if (tensor_mode) {
    const InputOptions src_opt = impl_->pack.input_appsrc_options(true);
    simaai::neat::Tensor tensor = simaai::neat::tensor_from_cv_mat(input, src_opt, "Model::build");
    return build(tensor, opt, run_opt);
  }
  internal::ModelPack pack = impl_->pack;
  if (!opt.name_suffix.empty()) {
    pack = pack.clone_with_overrides(std::string{}, opt.name_suffix);
  }
  InputInfo info = input_info_from_mat(input);
  NodeGroup group = build_pipeline_group(*this, pack, impl_->options, opt, &info, false);
  Session p;
  p.add(group);
  Run run = p.build(input, RunMode::Async, run_opt);
  return Runner(std::move(run));
}
#endif

simaai::neat::Sample Model::run(const simaai::neat::Tensor& input, int timeout_ms) {
  const bool tensor_mode = impl_->pack.pipeline_type() == internal::PipelineType::QuantTess;
  InputInfo info = input_info_from_tensor(input, !tensor_mode);
  info = require_input_info(info, tensor_mode);
  const InputKey key = input_key_from_appsrc(appsrc_from_info(info));
  std::lock_guard<std::mutex> lock(impl_->sync_mu);
  if (!impl_->sync_ready || !(impl_->sync_key == key)) {
    internal::ModelPack pack = impl_->pack_for_sync();
    NodeGroup group =
        build_pipeline_group(*this, pack, impl_->options, Model::SessionOptions{}, &info, true);
    Session p;
    p.add(group);
    Run run = p.build(input, RunMode::Sync);
    impl_->sync_runner = Runner(std::move(run));
    impl_->sync_key = key;
    impl_->sync_ready = true;
  }
  return impl_->sync_runner.run(input, timeout_ms);
}

simaai::neat::Sample Model::run(const std::vector<Tensor>& inputs, int timeout_ms) {
  if (inputs.empty()) {
    throw std::runtime_error("Model::run: empty input list");
  }
  if (inputs.size() == 1) {
    return run(inputs.front(), timeout_ms);
  }
  simaai::neat::Sample bundle = make_bundle_from_tensors(inputs);
  return run(bundle, timeout_ms);
}

simaai::neat::Sample Model::run(const simaai::neat::Sample& input, int timeout_ms) {
  if (input.kind == SampleKind::Tensor && input.tensor.has_value()) {
    return run(*input.tensor, timeout_ms);
  }
  internal::ModelPack pack = impl_->pack_for_sync();
  NodeGroup group =
      build_pipeline_group(*this, pack, impl_->options, Model::SessionOptions{}, nullptr, true);
  std::lock_guard<std::mutex> lock(impl_->sync_mu);
  if (!impl_->sync_ready) {
    Session p;
    p.add(group);
    Run run = p.build(input, RunMode::Sync);
    impl_->sync_runner = Runner(std::move(run));
    impl_->sync_ready = true;
  }
  return impl_->sync_runner.run(input, timeout_ms);
}

#if defined(SIMA_WITH_OPENCV)
simaai::neat::Sample Model::run(const cv::Mat& input, int timeout_ms) {
  const bool tensor_mode = impl_->pack.pipeline_type() == internal::PipelineType::QuantTess;
  if (tensor_mode) {
    const InputOptions src_opt = impl_->pack.input_appsrc_options(true);
    simaai::neat::Tensor tensor = simaai::neat::tensor_from_cv_mat(input, src_opt, "Model::run");
    return run(tensor, timeout_ms);
  }

  InputInfo info = input_info_from_mat(input);
  const InputKey key = input_key_from_appsrc(appsrc_from_info(info));
  std::lock_guard<std::mutex> lock(impl_->sync_mu);
  if (!impl_->sync_ready || !(impl_->sync_key == key)) {
    internal::ModelPack pack = impl_->pack_for_sync();
    NodeGroup group =
        build_pipeline_group(*this, pack, impl_->options, Model::SessionOptions{}, &info, true);
    Session p;
    p.add(group);
    Run run = p.build(input, RunMode::Sync);
    impl_->sync_runner = Runner(std::move(run));
    impl_->sync_key = key;
    impl_->sync_ready = true;
  }
  return impl_->sync_runner.run(input, timeout_ms);
}
#endif

namespace internal {
const ModelPack& ModelAccess::pack(const Model& model) {
  return model.impl_->pack;
}

const ModelPack& ModelAccess::pack_for_sync(const Model& model) {
  return model.impl_->pack_for_sync();
}

std::string ModelAccess::model_id(const Model& model) {
  return model.impl_->model_id;
}

NodeGroup ModelAccess::build_preprocess_group(const Model& model, bool sync) {
  const ModelPack& pack = sync ? model.impl_->pack_for_sync() : model.impl_->pack;
  return build_preprocess_group_impl(model, pack, model.impl_->options, nullptr, std::string{},
                                     std::string{}, sync);
}

NodeGroup ModelAccess::build_infer_group(const Model& model, bool sync) {
  const ModelPack& pack = sync ? model.impl_->pack_for_sync() : model.impl_->pack;
  const auto split = pack.split_sequence();
  std::string upstream;
  if (!split.pre.empty()) {
    upstream = split.pre.back().name;
  } else {
    upstream = "decoder";
  }
  upstream = pack.apply_name_suffix(upstream);
  return pack.infer_block(upstream);
}
} // namespace internal

} // namespace simaai::neat
