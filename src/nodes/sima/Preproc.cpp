#include "nodes/sima/Preproc.h"

#include "builder/InputContractConfigurable.h"
#include "model/internal/ModelInternal.h"
#include "model/internal/ModelPack.h"
#include "gst/GstHelpers.h"
#include "pipeline/internal/contract/CompiledNodeContract.h"
#include "pipeline/internal/contract/ContractFacts.h"
#include "pipeline/internal/EnvUtil.h"
#include "pipeline/internal/TensorMath.h"
#include "pipeline/internal/sima/stagesemantics/ProcessCvuStageSemantics.h"

#include <nlohmann/json.hpp>

#include <algorithm>
#include <cctype>
#include <cstdint>
#include <cstdio>
#include <sstream>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace simaai::neat {
using pipeline_internal::upper_copy;

using json = nlohmann::json;

struct Preproc::PreprocConfigHolder {
  json config;
  bool has_config = false;
};

namespace {

int channels_from_format(const std::string& fmt, int fallback) {
  if (!fmt.empty()) {
    const std::string up = upper_copy(fmt);
    if (up == "GRAY" || up == "GRAY8")
      return 1;
    if (up == "RGB" || up == "BGR" || up == "NV12" || up == "I420")
      return 3;
  }
  return (fallback > 0) ? fallback : 3;
}

std::string processcvu_image_type_token(const std::string& fmt) {
  const std::string up = upper_copy(fmt);
  return (up == "GRAY8") ? std::string("GRAY") : up;
}

bool dtype_is_quantized(const std::string& raw) {
  const std::string token = upper_copy(raw);
  return token == "INT8" || token == "EVXX_INT8" || token == "UINT8" || token == "U8";
}

bool tile_geometry_present(const PreprocOptions& opt) {
  return opt.has_slice_shape();
}

bool quant_params_present(const PreprocOptions& opt) {
  return opt.q_scale.has_value() || opt.q_zp.has_value();
}

std::vector<int> compact_shape(std::initializer_list<int> dims) {
  std::vector<int> shape;
  shape.reserve(dims.size());
  for (const int dim : dims) {
    if (dim > 0) {
      shape.push_back(dim);
    }
  }
  return shape;
}

void require_supported_single_output_handoff(const PreprocOptions& opt) {
  if (opt.single_output_handoff) {
    return;
  }
  throw std::runtime_error(
      "Preproc dual-output contract is currently unsupported; use single_output_handoff=true.");
}

void require_quantized_output_params(const PreprocOptions& opt, const char* where) {
  if (!dtype_is_quantized(opt.output_dtype)) {
    return;
  }
  if (!opt.q_scale.has_value() || !opt.q_zp.has_value()) {
    throw std::runtime_error(std::string("Preproc: ") + (where ? where : "strict contract") +
                             " requires explicit q_scale and q_zp for quantized output.");
  }
}

void canonicalize_quantized_output_defaults(PreprocOptions* opt) {
  if (!opt || !dtype_is_quantized(opt->output_dtype)) {
    return;
  }
  if (!opt->q_scale.has_value()) {
    opt->q_scale = 1.0;
  }
  if (!opt->q_zp.has_value()) {
    opt->q_zp = 0;
  }
}

void warn_preproc_option(const std::string& msg) {
  std::fprintf(stderr, "[WARN] Preproc: %s\n", msg.c_str());
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

std::vector<std::string> resolved_preproc_runtime_output_names() {
  return {"output_rgb_image", "output_tessellated_image"};
}

std::vector<int> preproc_dense_shape_hwc(int height, int width, int channels) {
  std::vector<int> shape;
  if (height > 0) {
    shape.push_back(height);
  }
  if (width > 0) {
    shape.push_back(width);
  }
  if (channels > 0) {
    shape.push_back(channels);
  }
  return shape;
}

int preproc_shape_channel_or_default(const std::vector<int>& shape, int fallback) {
  if (shape.size() >= 3 && shape.back() > 0) {
    return shape.back();
  }
  return fallback;
}

void preproc_apply_shape_hwc(const std::vector<int>& shape, int* out_height, int* out_width,
                             int* out_channels) {
  if (out_height) {
    *out_height = shape.size() >= 1 ? shape[0] : 0;
  }
  if (out_width) {
    *out_width = shape.size() >= 2 ? shape[1] : 0;
  }
  if (out_channels) {
    *out_channels = preproc_shape_channel_or_default(shape, 0);
  }
}

std::string resolved_preproc_primary_output_name(const PreprocOptions& opt) {
  return pipeline_internal::sima::stagesemantics::resolve_preproc_primary_output_name(
      resolved_preproc_runtime_output_names(), opt.tessellate);
}

std::string layout_from_format(const std::string& fmt, int channels) {
  const int resolved_channels = channels_from_format(fmt, channels);
  return resolved_channels <= 1 ? "HW" : "HWC";
}

std::string canonical_preproc_output_dtype_or_throw(const std::string& raw_token) {
  const std::string token = upper_copy(raw_token);
  if (token.empty()) {
    return "INT16";
  }
  if (token == "INT8" || token == "EVXX_INT8" || token == "U8" || token == "UINT8") {
    return "INT8";
  }
  if (token == "INT16" || token == "EVXX_INT16") {
    return "INT16";
  }
  if (token == "EVXX_BFLOAT16" || token == "BFLOAT16" || token == "BF16") {
    return "EVXX_BFLOAT16";
  }
  throw std::runtime_error("Preproc: unsupported output_dtype token '" + raw_token + "'");
}

std::string resolve_preproc_output_dtype_policy(const std::string& requested_token) {
  const std::string canonical = canonical_preproc_output_dtype_or_throw(requested_token);
  return canonical;
}

void enforce_preproc_output_dtype_policy(PreprocOptions* opt, json* cfg) {
  if (!opt) {
    throw std::runtime_error("Preproc: invalid dtype policy input");
  }

  std::string requested = opt->output_dtype;
  if (cfg && cfg->is_object() && cfg->contains("output_dtype")) {
    const json& v = (*cfg)["output_dtype"];
    if (!v.is_string()) {
      throw std::runtime_error("Preproc: output_dtype must be a string");
    }
    requested = v.get<std::string>();
  }

  const std::string effective = resolve_preproc_output_dtype_policy(requested);
  opt->output_dtype = effective;
  if (cfg && cfg->is_object()) {
    (*cfg)["output_dtype"] = effective;
  }
}

std::string read_next_cpu_string(const json& j) {
  auto normalize = [](const json& v) -> std::string {
    if (v.is_number_integer()) {
      const int cpu = v.get<int>();
      if (cpu == 0)
        return "APU";
      if (cpu == 1)
        return "CVU";
      if (cpu == 2)
        return "MLA";
      return {};
    }
    if (v.is_string()) {
      const std::string up = upper_copy(v.get<std::string>());
      if (up == "APU" || up == "A65")
        return "APU";
      if (up == "CVU")
        return "CVU";
      if (up == "MLA")
        return "MLA";
    }
    return {};
  };

  if (j.contains("next_cpu")) {
    const std::string val = normalize(j["next_cpu"]);
    if (!val.empty())
      return val;
  }
  if (j.contains("simaai__params") && j["simaai__params"].is_object()) {
    const auto& params = j["simaai__params"];
    if (params.contains("next_cpu")) {
      return normalize(params["next_cpu"]);
    }
  }
  return {};
}

json build_preproc_json(const PreprocOptions& opt);

} // namespace

PreprocOptions::PreprocOptions(const simaai::neat::Model& model) {
  *this = simaai::neat::internal::ModelAccess::build_preprocess_stage_options(model, false);
}

namespace {

json build_preproc_json(const PreprocOptions& opt) {
  require_supported_single_output_handoff(opt);
  const std::string in_fmt = processcvu_image_type_token(opt.input_img_type);
  const std::string out_fmt = processcvu_image_type_token(opt.output_img_type);
  const std::vector<float> mean3 = ensure_three(opt.channel_mean, 0.0f);
  const std::vector<float> std3 = ensure_three(opt.channel_stddev, 1.0f);
  const int input_channels =
      opt.input_channels() > 0 ? opt.input_channels() : channels_from_format(opt.input_img_type, 0);
  const int output_channels = opt.output_channels() > 0
                                  ? opt.output_channels()
                                  : channels_from_format(opt.output_img_type, input_channels);
  const std::vector<int> input_shape = opt.has_input_shape() ? opt.input_shape : std::vector<int>{};
  const std::vector<int> output_shape =
      opt.has_output_shape() ? opt.output_shape : std::vector<int>{};
  const std::vector<int> slice_shape = opt.tessellate ? opt.slice_shape : std::vector<int>{};

  json j;
  j["graph_name"] = opt.graph_name;
  j["node_name"] = opt.node_name;
  if (!opt.cpu.empty()) {
    j["cpu"] = upper_copy(opt.cpu);
  }
  if (!opt.next_cpu.empty()) {
    j["next_cpu"] = upper_copy(opt.next_cpu);
  }

  j["input_buffers"] = json::array({
      {
          {"name", opt.upstream_name},
          {"memories", json::array({
                           {
                               {"segment_name", "parent"},
                               {"graph_input_name", opt.graph_input_name},
                           },
                       })},
      },
  });

  j["primary_output_name"] = resolved_preproc_primary_output_name(opt);
  j["debug"] = opt.debug;

  j["input_shape"] = input_shape;
  j["input_offset"] = opt.input_offset;

  j["output_shapes"] = json::array({output_shape});
  j["scaled_width"] = opt.scaled_width;
  j["scaled_height"] = opt.scaled_height;

  j["batch_size"] = opt.batch_size;
  j["normalize"] = opt.normalize;
  j["aspect_ratio"] = opt.aspect_ratio;
  j["dynamic_input_dims"] = opt.dynamic_input_dims;

  j["tessellate"] = opt.tessellate;
  if (!slice_shape.empty()) {
    j["slice_shape"] = slice_shape;
  }

  j["q_zp"] = opt.q_zp.value_or(0);
  j["q_scale"] = opt.q_scale.value_or(1.0);
  j["channel_mean"] = json::array({mean3[0], mean3[1], mean3[2]});
  j["channel_stddev"] = json::array({std3[0], std3[1], std3[2]});

  j["input_img_type"] = in_fmt;
  j["output_img_type"] = out_fmt;
  j["scaling_type"] = opt.scaling_type;
  j["padding_type"] = opt.padding_type;
  j["pad_value"] = opt.pad_value;
  j["input_stride"] = opt.input_stride;
  j["output_stride"] = opt.output_stride;
  j["output_dtype"] = opt.output_dtype;

  json sink_params = json::array({
      {
          {"name", "format"},
          {"type", "string"},
          {"values", "GRAY, RGB, BGR, I420, NV12"},
          {"json_field", "input_img_type"},
      },
  });
  sink_params.push_back({
      {"name", "width"},
      {"type", "int"},
      {"values", "1 - 4096"},
  });
  sink_params.push_back({
      {"name", "height"},
      {"type", "int"},
      {"values", "1 - 4096"},
  });

  j["caps"] = {
      {"sink_pads", json::array({
                        {
                            {"media_type", "video/x-raw"},
                            {"params", std::move(sink_params)},
                        },
                    })},
      {"src_pads", json::array({
                       {
                           {"media_type", "video/x-raw"},
                           {"params", json::array({
                                          {
                                              {"name", "format"},
                                              {"type", "string"},
                                              {"values", "GRAY, RGB, BGR"},
                                              {"json_field", "output_img_type"},
                                          },
                                          {
                                              {"name", "width"},
                                              {"type", "int"},
                                              {"values", "1 - 4096"},
                                          },
                                          {
                                              {"name", "height"},
                                              {"type", "int"},
                                              {"values", "1 - 4096"},
                                          },
                                      })},
                       },
                   })},
  };

  return j;
}

} // namespace

Preproc::Preproc(PreprocOptions opt) : opt_(std::move(opt)) {
  if (opt_.num_buffers_locked) {
    if (opt_.num_buffers != opt_.num_buffers_model) {
      throw std::runtime_error("Preproc: num_buffers override is not allowed; must match model.");
    }
    if (opt_.num_buffers != 4 && opt_.num_buffers != 1) {
      throw std::runtime_error(
          "Preproc: num_buffers must be 4 (async) or 1 (sync) for model pipelines.");
    }
  }
  if (!opt_.model_managed_contract && !opt_.tessellate && tile_geometry_present(opt_)) {
    warn_preproc_option("tile geometry was provided while tessellate=false; it will be ignored.");
  }
  if (!opt_.model_managed_contract && !dtype_is_quantized(opt_.output_dtype) &&
      quant_params_present(opt_)) {
    warn_preproc_option("quantization parameters were provided while output_dtype is not "
                        "quantized; they will be ignored.");
  }

  auto holder = std::make_shared<PreprocConfigHolder>();
  if (!opt_.model_managed_contract) {
    config_holder_ = std::move(holder);
    return;
  }

  if (!opt_.has_input_shape() || opt_.input_img_type.empty()) {
    throw std::runtime_error("Preproc: invalid model-managed input contract");
  }
  if (!opt_.has_output_shape() || opt_.scaled_width <= 0 || opt_.scaled_height <= 0) {
    throw std::runtime_error(
        "Preproc: model-managed preproc requires explicit output/scaled geometry.");
  }
  if (opt_.output_dtype.empty()) {
    throw std::runtime_error("Preproc: model-managed preproc requires explicit output_dtype.");
  }
  if (opt_.input_channels() <= 0) {
    opt_.set_input_shape(compact_shape(
        {opt_.input_height(), opt_.input_width(), channels_from_format(opt_.input_img_type, 0)}));
  }
  if (opt_.output_img_type.empty()) {
    opt_.output_img_type = opt_.input_img_type;
  }
  if (opt_.output_channels() <= 0) {
    opt_.set_output_shape(
        compact_shape({opt_.output_height(), opt_.output_width(),
                       channels_from_format(opt_.output_img_type, opt_.input_channels())}));
  }
  if (opt_.tessellate) {
    if (!opt_.has_slice_shape()) {
      throw std::runtime_error("Preproc: tessellate=true requires explicit slice_shape.");
    }
  } else if (tile_geometry_present(opt_)) {
    warn_preproc_option("tile geometry was provided while tessellate=false; it will be ignored.");
    opt_.slice_shape.clear();
  }
  require_quantized_output_params(opt_, "model-managed preproc");
  if (!dtype_is_quantized(opt_.output_dtype) && quant_params_present(opt_)) {
    warn_preproc_option("quantization parameters were provided while output_dtype is not "
                        "quantized; they will be ignored.");
  }

  holder->config = build_preproc_json(opt_);
  enforce_preproc_output_dtype_policy(&opt_, &holder->config);
  holder->has_config = true;
  config_path_.clear();
  config_holder_ = std::move(holder);
}

void Preproc::materialize_config_from_input_contract(const InputContract& contract) {
  if (contract.width <= 0 || contract.height <= 0 || contract.format.empty()) {
    throw std::runtime_error("Preproc: missing input w/h/format from upstream input contract.");
  }

  opt_.set_input_shape(compact_shape({contract.height, contract.width, 0}));
  const std::string contract_format = upper_copy(contract.format);
  if (!opt_.model_managed_contract || opt_.input_img_type.empty()) {
    opt_.input_img_type = contract_format;
  }
  if (opt_.input_channels() <= 0) {
    opt_.set_input_shape(compact_shape(
        {opt_.input_height(), opt_.input_width(),
         (contract.depth > 0) ? contract.depth : channels_from_format(opt_.input_img_type, 0)}));
  }
  if (opt_.model_managed_contract) {
    if (!opt_.has_output_shape() || opt_.scaled_width <= 0 || opt_.scaled_height <= 0) {
      throw std::runtime_error(
          "Preproc: model-managed preproc requires explicit output/scaled geometry.");
    }
    if (opt_.output_dtype.empty()) {
      throw std::runtime_error("Preproc: model-managed preproc requires explicit output_dtype.");
    }
  } else {
    if (opt_.output_width() <= 0 || opt_.output_height() <= 0) {
      opt_.set_output_shape(
          compact_shape({opt_.output_height() > 0 ? opt_.output_height() : contract.height,
                         opt_.output_width() > 0 ? opt_.output_width() : contract.width,
                         opt_.output_channels()}));
    }
    if (opt_.scaled_width <= 0) {
      opt_.scaled_width = opt_.output_width();
    }
    if (opt_.scaled_height <= 0) {
      opt_.scaled_height = opt_.output_height();
    }
  }
  if (opt_.output_img_type.empty()) {
    opt_.output_img_type = opt_.input_img_type;
  }
  if (opt_.output_channels() <= 0) {
    opt_.set_output_shape(
        compact_shape({opt_.output_height(), opt_.output_width(),
                       channels_from_format(opt_.output_img_type, opt_.input_channels())}));
  }

  const bool quantize_enabled = dtype_is_quantized(opt_.output_dtype);
  if (opt_.tessellate) {
    if (!opt_.has_slice_shape()) {
      throw std::runtime_error("Preproc: tessellate=true requires explicit slice_shape.");
    }
  } else if (tile_geometry_present(opt_)) {
    warn_preproc_option("tile geometry was provided while tessellate=false; it will be ignored.");
    opt_.slice_shape.clear();
  }
  if (opt_.model_managed_contract) {
    require_quantized_output_params(opt_, "model-managed preproc");
  } else {
    canonicalize_quantized_output_defaults(&opt_);
  }
  if (!quantize_enabled && quant_params_present(opt_)) {
    warn_preproc_option("quantization parameters were provided while output_dtype is not "
                        "quantized; they will be ignored.");
  }

  auto holder = config_holder_ ? config_holder_ : std::make_shared<PreprocConfigHolder>();
  holder->config = build_preproc_json(opt_);
  enforce_preproc_output_dtype_policy(&opt_, &holder->config);
  holder->has_config = true;
  config_path_.clear();
  config_holder_ = std::move(holder);
}

void Preproc::apply_input_contract(const InputContract& contract, std::string* err) {
  try {
    materialize_config_from_input_contract(contract);
    if (err) {
      err->clear();
    }
  } catch (const std::exception& ex) {
    if (err) {
      *err = ex.what();
      return;
    }
    throw;
  }
}

NodeContractDefinition Preproc::contract_definition() const {
  require_supported_single_output_handoff(opt_);
  NodeContractDefinition def;
  def.node_kind = kind();
  def.plugin_kind = "processcvu";

  ContractPortSpec input;
  input.port_id = opt_.graph_input_name.empty() ? "input_image" : opt_.graph_input_name;
  input.media_type = "video/x-raw";
  input.format = upper_copy(opt_.input_img_type);
  input.dtype = "UINT8";
  input.layout = "HWC";
  def.inputs.push_back(std::move(input));

  ContractPortSpec output;
  output.port_id = opt_.single_output_handoff ? resolved_preproc_primary_output_name(opt_)
                                              : std::string("preproc_outputs");
  output.media_type = "video/x-raw";
  output.format = upper_copy(opt_.output_img_type);
  output.dtype = upper_copy(opt_.output_dtype);
  output.layout = layout_from_format(opt_.output_img_type, opt_.output_channels());
  def.outputs.push_back(std::move(output));

  def.fields.push_back(
      {"input_shape", ContractFieldSource::InputOnly, ContractOverridePolicy::Forbidden, true});
  def.fields.push_back(
      {"input_img_type", ContractFieldSource::InputOnly, ContractOverridePolicy::Forbidden, true});
  def.fields.push_back({"output_shapes", ContractFieldSource::BuilderOption,
                        ContractOverridePolicy::BuilderOnly, true});
  def.fields.push_back({"output_img_type", ContractFieldSource::BuilderOption,
                        ContractOverridePolicy::BuilderOnly, true});
  def.fields.push_back({"tessellate", ContractFieldSource::BuilderOption,
                        ContractOverridePolicy::BuilderOnly, true});
  def.fields.push_back({"slice_shape", ContractFieldSource::BuilderOption,
                        ContractOverridePolicy::BuilderOnly, false});
  return def;
}

bool Preproc::compile_node_contract(const ContractCompileInput& input, CompiledNodeContract* out,
                                    std::string* err) const {
  const std::string element_name = element_names(input.node_index).empty()
                                       ? std::string("preproc")
                                       : element_names(input.node_index).front();
  try {
    require_supported_single_output_handoff(opt_);
    const auto compiled =
        pipeline_internal::sima::stagesemantics::build_processcvu_compiled_contract_from_options(
            options());
    return pipeline_internal::sima::stagesemantics::build_processcvu_node_contract(
        kind(), element_name, element_name, contract_definition(), compiled, out, err);
  } catch (const std::exception& ex) {
    if (err) {
      *err = ex.what();
    }
    return false;
  }
}

void Preproc::apply_compiled_contract(const CompiledNodeContract& contract, std::string* err) {
  if (err) {
    err->clear();
  }
  if (!contract.processcvu.has_value()) {
    return;
  }
  const auto& payload = contract.processcvu->payload;
  if (!payload.input_shapes.empty()) {
    opt_.input_shape = payload.input_shapes[0];
  }
  if (payload.tessellate == 1) {
    const int output_height =
        payload.scaled_height > 0 ? payload.scaled_height : opt_.input_height();
    const int output_width = payload.scaled_width > 0 ? payload.scaled_width : opt_.input_width();
    opt_.set_output_shape(compact_shape({output_height, output_width, opt_.output_channels()}));
    if (!payload.output_shapes.empty()) {
      const int output_channels = preproc_shape_channel_or_default(payload.output_shapes[0], 0);
      if (output_channels > 0) {
        opt_.set_output_shape(
            compact_shape({opt_.output_height(), opt_.output_width(), output_channels}));
      }
    }
  } else {
    if (!payload.output_shapes.empty()) {
      opt_.output_shape = payload.output_shapes[0];
    }
  }
  if (!payload.slice_shapes.empty()) {
    opt_.slice_shape = payload.slice_shapes[0];
  } else {
    opt_.slice_shape.clear();
  }
  opt_.scaled_width = payload.scaled_width;
  opt_.scaled_height = payload.scaled_height;
}

std::string Preproc::backend_fragment(int node_index) const {
  std::ostringstream ss;
  require_element("neatprocesscvu", "Preproc::backend_fragment");
  const char* factory = "neatprocesscvu";
  const std::string name = opt_.element_name.empty()
                               ? ("n" + std::to_string(node_index) + "_preproc")
                               : opt_.element_name;
  const std::string stage_id = opt_.model_managed_contract ? name : std::string();
  if ((!config_holder_ || !config_holder_->has_config) && !opt_.model_managed_contract) {
    throw std::runtime_error(
        "Preproc: standalone preproc requires actual input contract before pipeline build. "
        "Build with input so width/height/format can be derived from the upstream input.");
  }
  ss << factory << " name=" << name;
  if (!stage_id.empty()) {
    ss << " stage-id=" << stage_id;
  }
  if (opt_.num_buffers > 0) {
    ss << " num-buffers=" << opt_.num_buffers;
  }
  return ss.str();
}

std::vector<std::string> Preproc::element_names(int node_index) const {
  if (!opt_.element_name.empty())
    return {opt_.element_name};
  return {"n" + std::to_string(node_index) + "_preproc"};
}

OutputSpec Preproc::output_spec(const OutputSpec& input) const {
  if ((!config_holder_ || !config_holder_->has_config) && !opt_.model_managed_contract) {
    InputContract contract;
    contract.media_type = input.media_type;
    contract.format = input.format;
    contract.width = input.width;
    contract.height = input.height;
    contract.depth = input.depth;
    const_cast<Preproc*>(this)->materialize_config_from_input_contract(contract);
  }
  if (opt_.model_managed_contract && (opt_.output_width() <= 0 || opt_.output_height() <= 0)) {
    throw std::runtime_error(
        "Preproc::output_spec: model-managed preproc requires explicit output geometry.");
  }
  OutputSpec out;
  out.media_type = "video/x-raw";
  out.format = upper_copy(opt_.output_img_type);
  out.width = (opt_.output_width() > 0) ? opt_.output_width() : opt_.input_width();
  out.height = (opt_.output_height() > 0) ? opt_.output_height() : opt_.input_height();
  out.depth =
      (opt_.output_channels() > 0) ? opt_.output_channels() : channels_from_format(out.format, 3);
  out.layout = (out.format == "GRAY" || out.format == "GRAY8") ? "HW" : "HWC";
  out.dtype = "UInt8";
  if (upper_copy(opt_.next_cpu) == "APU") {
    out.memory = "SystemMemory";
  } else if (!input.memory.empty()) {
    out.memory = input.memory;
  } else {
    out.memory = "SimaAI";
  }
  out.certainty = SpecCertainty::Derived;
  out.note = "neatprocesscvu";
  out.byte_size = expected_byte_size(out);
  return out;
}

const nlohmann::json* Preproc::config_json() const {
  if (!config_holder_ || !config_holder_->has_config)
    return nullptr;
  return &config_holder_->config;
}

} // namespace simaai::neat

namespace simaai::neat::nodes {

std::shared_ptr<simaai::neat::Node> Preproc(PreprocOptions opt) {
  return std::make_shared<simaai::neat::Preproc>(std::move(opt));
}

} // namespace simaai::neat::nodes
