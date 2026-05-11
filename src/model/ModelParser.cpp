#include "model/internal/ModelParser.h"
#include "model/internal/RoutePlanner.h"

#include "pipeline/internal/TensorMath.h"

#include <nlohmann/json.hpp>

#include <algorithm>
#include <cctype>
#include <fstream>
#include <unordered_map>
#include <unordered_set>

namespace simaai::neat::internal {
namespace {
using json = nlohmann::json;

std::string kernel_kind_name(ParsedKernelKind kind) {
  switch (kind) {
  case ParsedKernelKind::Preproc:
    return "preproc";
  case ParsedKernelKind::Quantize:
    return "quantize";
  case ParsedKernelKind::Tessellate:
    return "tessellate";
  case ParsedKernelKind::QuantTess:
    return "quanttess";
  case ParsedKernelKind::Cast:
    return "cast";
  case ParsedKernelKind::DetessDequant:
    return "detessdequant";
  case ParsedKernelKind::Detessellate:
    return "detessellate";
  case ParsedKernelKind::Dequantize:
    return "dequantize";
  case ParsedKernelKind::BoxDecode:
    return "boxdecode";
  case ParsedKernelKind::Unknown:
    break;
  }
  return "unknown";
}

ParsedKernelKind canonical_kernel_kind(const std::string& raw_kernel) {
  const std::string kernel = pipeline_internal::lower_copy(raw_kernel);
  if (kernel.empty()) {
    return ParsedKernelKind::Unknown;
  }
  if (kernel.find("detessdequant") != std::string::npos) {
    return ParsedKernelKind::DetessDequant;
  }
  if (kernel.find("detess") != std::string::npos) {
    return ParsedKernelKind::Detessellate;
  }
  if (kernel.find("dequant") != std::string::npos) {
    return ParsedKernelKind::Dequantize;
  }
  if (kernel.find("boxdecode") != std::string::npos) {
    return ParsedKernelKind::BoxDecode;
  }
  if (kernel.find("quanttess") != std::string::npos ||
      (kernel.find("quant") != std::string::npos && kernel.find("tess") != std::string::npos)) {
    return ParsedKernelKind::QuantTess;
  }
  if (kernel.find("quant") != std::string::npos) {
    return ParsedKernelKind::Quantize;
  }
  if (kernel.find("tess") != std::string::npos) {
    return ParsedKernelKind::Tessellate;
  }
  if (kernel.find("cast") != std::string::npos) {
    return ParsedKernelKind::Cast;
  }
  if (kernel.find("preproc") != std::string::npos) {
    return ParsedKernelKind::Preproc;
  }
  return ParsedKernelKind::Unknown;
}

std::string canonical_dtype_for_signal(std::string raw) {
  for (char& c : raw) {
    c = static_cast<char>(std::toupper(static_cast<unsigned char>(c)));
  }
  if (raw.find("BFLOAT16") != std::string::npos || raw.find("BF16") != std::string::npos) {
    return "BF16";
  }
  if (raw.find("FLOAT32") != std::string::npos || raw == "FP32") {
    return "FP32";
  }
  if (raw.find("FLOAT16") != std::string::npos || raw == "FP16") {
    return "FP16";
  }
  if (raw.find("INT8") != std::string::npos) {
    return "INT8";
  }
  if (raw.find("UINT8") != std::string::npos) {
    return "UINT8";
  }
  if (raw.find("INT16") != std::string::npos) {
    return "INT16";
  }
  if (raw.find("UINT16") != std::string::npos) {
    return "UINT16";
  }
  if (raw.find("INT32") != std::string::npos) {
    return "INT32";
  }
  if (raw.find("UINT32") != std::string::npos) {
    return "UINT32";
  }
  return raw;
}

bool dtype_is_float_like_local(const std::string& raw_dtype) {
  const std::string dtype = canonical_dtype_for_signal(raw_dtype);
  if (dtype.empty()) {
    return false;
  }
  return dtype == "BF16" || dtype == "FP16" || dtype == "FP32";
}

bool dtype_is_quantized_like_local(const std::string& raw_dtype) {
  const std::string dtype = canonical_dtype_for_signal(raw_dtype);
  if (dtype.empty() || dtype_is_float_like_local(dtype)) {
    return false;
  }
  return dtype == "INT8" || dtype == "UINT8" || dtype == "INT16" || dtype == "UINT16" ||
         dtype == "INT32" || dtype == "UINT32";
}

std::string primary_tensor_dtype(
    const std::vector<pipeline_internal::sima::MpkTensorContract>& tensors) {
  for (const auto& tensor : tensors) {
    if (!tensor.logical_dtype.empty()) {
      return canonical_dtype_for_signal(tensor.logical_dtype);
    }
    if (!tensor.dtype.empty()) {
      return canonical_dtype_for_signal(tensor.dtype);
    }
  }
  return {};
}

int primary_tensor_rank(const std::vector<pipeline_internal::sima::MpkTensorContract>& tensors) {
  for (const auto& tensor : tensors) {
    if (!tensor.logical_shape.empty()) {
      return static_cast<int>(tensor.logical_shape.size());
    }
    if (!tensor.mpk_shape.empty()) {
      return static_cast<int>(tensor.mpk_shape.size());
    }
  }
  return 0;
}

void push_unique(std::vector<std::string>* values, const std::string& value) {
  if (!values || value.empty()) {
    return;
  }
  if (std::find(values->begin(), values->end(), value) != values->end()) {
    return;
  }
  values->push_back(value);
}

void parse_mpk_execution_order(const pipeline_internal::sima::MpkContract& contract,
                               ParsedModelInfo* out) {
  if (!out) {
    return;
  }
  out->execution_order = pipeline_internal::sima::plugins_in_execution_order(contract);
  std::vector<bool> seen(contract.plugins.size(), false);
  for (const std::size_t idx : out->execution_order) {
    if (idx < seen.size()) {
      seen[idx] = true;
    }
  }
  for (std::size_t idx = 0; idx < contract.plugins.size(); ++idx) {
    if (!seen[idx]) {
      out->execution_order.push_back(idx);
    }
  }
}

void parse_mpk_edges(const pipeline_internal::sima::MpkContract& contract, ParsedModelInfo* out) {
  if (!out) {
    return;
  }
  out->edges = contract.edges;
}

void parse_mpk_mla_io(const pipeline_internal::sima::MpkContract& contract, ParsedModelInfo* out) {
  if (!out) {
    return;
  }
  const auto* mla = pipeline_internal::sima::get_mla_stage_io_contract(contract);
  if (!mla) {
    out->warnings.push_back("model-parser: MLA stage missing in MPK contract");
    return;
  }

  auto mla_idx =
      pipeline_internal::sima::find_plugin_index_by_name_or_id(contract, mla->name);
  if (!mla_idx.has_value() && !mla->plugin_id.empty()) {
    mla_idx = pipeline_internal::sima::find_plugin_index_by_name_or_id(contract, mla->plugin_id);
  }
  if (mla_idx.has_value()) {
    out->mla_plugin_index = static_cast<int>(*mla_idx);
  }

  out->outputs.physical.reserve(mla->output_tensors.size());
  for (std::size_t i = 0; i < mla->output_tensors.size(); ++i) {
    const auto& src = mla->output_tensors[i];
    ParsedPhysicalOutput dst;
    dst.name = src.name.empty() ? ("mla_output_" + std::to_string(i)) : src.name;
    dst.dtype = src.dtype;
    dst.shape = src.mpk_shape;
    dst.size_bytes = src.size_bytes;
    out->outputs.physical.push_back(std::move(dst));
  }

  out->outputs.logical.reserve(mla->output_tensors.size());
  for (std::size_t i = 0; i < mla->output_tensors.size(); ++i) {
    const auto& src = mla->output_tensors[i];
    ParsedLogicalOutput dst;
    dst.name = src.name.empty() ? ("mla_output_" + std::to_string(i)) : src.name;
    dst.dtype = src.logical_dtype.empty() ? src.dtype : src.logical_dtype;
    dst.shape = src.logical_shape.empty() ? src.mpk_shape : src.logical_shape;
    dst.size_bytes = src.size_bytes;
    dst.source_plugin = src.logical_source_plugin;
    dst.source_kernel = src.logical_source_kernel;
    dst.source_sequence = src.logical_source_sequence;
    dst.physical_index = src.tensor_index;
    if (dst.physical_index < 0 ||
        static_cast<std::size_t>(dst.physical_index) >= out->outputs.physical.size()) {
      dst.physical_index = out->outputs.physical.empty()
                               ? -1
                               : static_cast<int>(std::min<std::size_t>(
                                     i, out->outputs.physical.size() - 1U));
    }
    out->outputs.logical.push_back(std::move(dst));
  }
  out->outputs.packed_output =
      out->outputs.physical.size() == 1U && out->outputs.logical.size() > 1U;
}

void parse_mpk_plugins(const pipeline_internal::sima::MpkContract& contract, ParsedModelInfo* out) {
  if (!out) {
    return;
  }

  std::unordered_map<std::size_t, std::size_t> rank_by_index;
  rank_by_index.reserve(out->execution_order.size());
  for (std::size_t rank = 0; rank < out->execution_order.size(); ++rank) {
    rank_by_index[out->execution_order[rank]] = rank;
  }

  std::size_t mla_rank = out->execution_order.size();
  if (out->mla_plugin_index >= 0) {
    const auto rank_it = rank_by_index.find(static_cast<std::size_t>(out->mla_plugin_index));
    if (rank_it != rank_by_index.end()) {
      mla_rank = rank_it->second;
    }
  }

  out->plugins.reserve(out->execution_order.size());
  for (const std::size_t plugin_idx : out->execution_order) {
    if (plugin_idx >= contract.plugins.size()) {
      continue;
    }
    const auto& src = contract.plugins[plugin_idx];
    std::string kernel_source = src.kernel;
    if (kernel_source.empty()) {
      kernel_source = src.name;
    }
    const auto kind = canonical_kernel_kind(kernel_source);
    const auto rank_it = rank_by_index.find(plugin_idx);
    const std::size_t rank = (rank_it == rank_by_index.end()) ? out->execution_order.size()
                                                               : rank_it->second;

    ParsedKernelStage dst;
    dst.name = src.name;
    dst.plugin_id = src.plugin_id;
    dst.processor = src.processor;
    dst.kernel = src.kernel;
    dst.kind = kind;
    dst.sequence = src.sequence;
    dst.before_mla = rank < mla_rank;
    dst.after_mla = rank > mla_rank;
    out->plugins.push_back(dst);

    const std::string canonical = kernel_kind_name(kind);
    const std::string in_dtype = primary_tensor_dtype(src.input_tensors);
    const std::string out_dtype = primary_tensor_dtype(src.output_tensors);
    const int in_rank = primary_tensor_rank(src.input_tensors);
    const int out_rank = primary_tensor_rank(src.output_tensors);
    const bool dtype_transition =
        !in_dtype.empty() && !out_dtype.empty() && in_dtype != out_dtype;

    const bool hint_pre_quant =
        dst.before_mla && dtype_transition && !dtype_is_quantized_like_local(in_dtype) &&
        dtype_is_quantized_like_local(out_dtype);
    const bool suppress_post_dequant_hint = (canonical == "unpacktransform");
    const bool hint_post_dequant =
        dst.after_mla && !suppress_post_dequant_hint && dtype_transition &&
        dtype_is_quantized_like_local(in_dtype) && dtype_is_float_like_local(out_dtype);
    const bool hint_cast =
        dtype_transition && !hint_pre_quant && !hint_post_dequant;
    const bool hint_pre_tess = dst.before_mla && in_rank >= 3 && out_rank > 0 && out_rank < in_rank;
    const bool hint_post_detess =
        dst.after_mla && in_rank > 0 && in_rank <= 2 && out_rank >= 3;

    if (dst.before_mla) {
      bool emitted_pre_signal = false;
      if (hint_pre_quant) {
        push_unique(&out->pre_kernels, "quantize");
        out->needs.pre_quantization = true;
        out->capabilities.has_pre_quantization = true;
        emitted_pre_signal = true;
      }
      if (hint_pre_tess) {
        push_unique(&out->pre_kernels, "tessellate");
        out->needs.pre_tessellation = true;
        out->capabilities.has_pre_tessellation = true;
        emitted_pre_signal = true;
      }
      if (hint_cast) {
        push_unique(&out->pre_kernels, "cast");
        out->needs.pre_cast = true;
        out->capabilities.has_pre_cast = true;
        emitted_pre_signal = true;
      }

      if (!emitted_pre_signal) {
        push_unique(&out->pre_kernels, canonical);
        if (kind == ParsedKernelKind::Quantize || kind == ParsedKernelKind::QuantTess) {
          out->needs.pre_quantization = true;
          out->capabilities.has_pre_quantization = true;
        }
        if (kind == ParsedKernelKind::Tessellate || kind == ParsedKernelKind::QuantTess) {
          out->needs.pre_tessellation = true;
          out->capabilities.has_pre_tessellation = true;
        }
        if (kind == ParsedKernelKind::Cast) {
          out->needs.pre_cast = true;
          out->capabilities.has_pre_cast = true;
        }
      }
    }

    if (dst.after_mla) {
      bool emitted_post_signal = false;
      if (hint_post_detess) {
        push_unique(&out->post_kernels, "detessellate");
        out->needs.post_detessellation = true;
        out->capabilities.has_post_detessellation = true;
        emitted_post_signal = true;
      }
      if (hint_post_dequant) {
        push_unique(&out->post_kernels, "dequantize");
        out->needs.post_dequantization = true;
        out->capabilities.has_post_dequantization = true;
        emitted_post_signal = true;
      }
      if (hint_cast) {
        push_unique(&out->post_kernels, "cast");
        out->needs.post_cast = true;
        out->capabilities.has_post_cast = true;
        emitted_post_signal = true;
      }

      if (!emitted_post_signal) {
        push_unique(&out->post_kernels, canonical);
        if (kind == ParsedKernelKind::DetessDequant || kind == ParsedKernelKind::Detessellate) {
          out->needs.post_detessellation = true;
          out->capabilities.has_post_detessellation = true;
        }
        if (kind == ParsedKernelKind::DetessDequant || kind == ParsedKernelKind::Dequantize) {
          out->needs.post_dequantization = true;
          out->capabilities.has_post_dequantization = true;
        }
        if (kind == ParsedKernelKind::Cast) {
          out->needs.post_cast = true;
          out->capabilities.has_post_cast = true;
        }
      }
      if (kind == ParsedKernelKind::BoxDecode) {
        push_unique(&out->post_kernels, "boxdecode");
        out->capabilities.has_post_boxdecode = true;
      }
    }
  }

  out->needs.tess_needed = out->needs.pre_tessellation || out->needs.post_detessellation;
  bool mla_output_quantized = false;
  if (!out->outputs.physical.empty()) {
    const std::string mla_dtype = canonical_dtype_for_signal(out->outputs.physical.front().dtype);
    mla_output_quantized = dtype_is_quantized_like_local(mla_dtype);
  }
  const bool has_explicit_post_kernels = !out->post_kernels.empty();
  if (out->needs.post_dequantization) {
    out->needs.quant_needed = true;
  } else if (!has_explicit_post_kernels) {
    out->needs.quant_needed = mla_output_quantized;
  } else {
    out->needs.quant_needed = false;
  }
  out->needs.post_detessellation = out->needs.tess_needed;

  if (out->needs.pre_cast && !out->needs.post_cast) {
    // Do not auto-heal cast symmetry at parse time; the route policy must decide.
    out->warnings.push_back(
        "model-parser: cast symmetry unresolved (pre_cast detected without post_cast)");
  }
}

const json* params_or_root(const json& cfg) {
  if (cfg.contains("simaai__params") && cfg.at("simaai__params").is_object()) {
    return &cfg.at("simaai__params");
  }
  return &cfg;
}

std::size_t read_array_count(const json& obj, const char* key) {
  if (!key || !*key || !obj.contains(key)) {
    return 0U;
  }
  const auto& v = obj.at(key);
  if (!v.is_array()) {
    return 0U;
  }
  return v.size();
}

void parse_mla_config_consistency(const ModelPack& pack, ParsedModelInfo* out) {
  if (!out) {
    return;
  }
  const std::string cfg_path = pack.find_config_path_by_processor("EV74");
  if (cfg_path.empty()) {
    return;
  }

  std::ifstream in(cfg_path);
  if (!in.is_open()) {
    return;
  }
  json cfg;
  try {
    in >> cfg;
  } catch (const std::exception&) {
    return;
  }
  if (!cfg.is_object()) {
    return;
  }

  const json* params = params_or_root(cfg);
  const std::size_t params_outputs = read_array_count(*params, "outputs");
  const std::size_t params_dtypes = read_array_count(*params, "data_type");
  const std::size_t physical_outputs = out->outputs.physical.size();

  if (physical_outputs > 0U && params_outputs > 0U && params_outputs != physical_outputs) {
    out->warnings.push_back("model-parser: MLA config outputs count (" +
                            std::to_string(params_outputs) +
                            ") differs from MPK physical output count (" +
                            std::to_string(physical_outputs) + ")");
  }
  if (params_dtypes > 0U && params_outputs > 0U && params_dtypes != params_outputs) {
    out->warnings.push_back("model-parser: MLA config data_type count (" +
                            std::to_string(params_dtypes) +
                            ") differs from MLA config outputs count (" +
                            std::to_string(params_outputs) + ")");
  }
}

} // namespace

std::vector<std::string> validate_parsed_model_contract(const ParsedModelInfo& parsed) {
  std::vector<std::string> warnings;
  if (parsed.plugins.empty()) {
    warnings.push_back("model-parser: parsed plugin list is empty");
  }
  if (parsed.execution_order.empty()) {
    warnings.push_back("model-parser: execution order is empty");
  }
  if (parsed.mla_plugin_index < 0) {
    warnings.push_back("model-parser: MLA plugin index unresolved");
  }
  if (parsed.outputs.physical.empty()) {
    warnings.push_back("model-parser: physical MLA outputs are empty");
  }
  if (parsed.outputs.logical.empty()) {
    warnings.push_back("model-parser: logical MLA outputs are empty");
  }

  std::unordered_set<std::string> names;
  for (const auto& out : parsed.outputs.physical) {
    if (out.name.empty()) {
      warnings.push_back("model-parser: physical output with empty name");
      continue;
    }
    if (!names.insert(out.name).second) {
      warnings.push_back("model-parser: duplicate physical output name '" + out.name + "'");
    }
  }

  if (parsed.outputs.packed_output) {
    warnings.push_back("model-parser: packed-output topology detected (1 physical output, " +
                       std::to_string(parsed.outputs.logical.size()) + " logical outputs)");
  }

  return warnings;
}

ParsedModelInfo parse_model_from_pack(const ModelPack& pack) {
  ParsedModelInfo out;
  const auto& maybe_contract = pack.mpk_contract();
  if (!maybe_contract.has_value()) {
    out.warnings.push_back("model-parser: MPK contract is unavailable");
    return out;
  }

  const auto& contract = *maybe_contract;
  out.mpk_json_path = contract.mpk_json_path;
  out.model_name = contract.model_name;

  parse_mpk_execution_order(contract, &out);
  parse_mpk_mla_io(contract, &out);
  parse_mpk_plugins(contract, &out);
  parse_mpk_edges(contract, &out);
  parse_mla_config_consistency(pack, &out);

  auto validation = validate_parsed_model_contract(out);
  out.warnings.insert(out.warnings.end(), validation.begin(), validation.end());
  return out;
}

bool parse_model_semantics_from_pack(const ModelPack& pack, ModelSemantics* out) {
  if (!out) {
    return false;
  }
  *out = ModelSemantics{};

  const ParsedModelInfo parsed = parse_model_from_pack(pack);
  out->quant_needed = parsed.needs.quant_needed;
  out->tess_needed = parsed.needs.tess_needed;
  out->pre_quant_needed = parsed.needs.pre_quantization;
  out->pre_tess_needed = parsed.needs.pre_tessellation;
  out->has_pre_adapter = !parsed.pre_kernels.empty();
  out->has_post_adapter = !parsed.post_kernels.empty();
  out->has_post_boxdecode = parsed.capabilities.has_post_boxdecode;
  out->has_pre_quant_adapter = parsed.capabilities.has_pre_quantization;
  out->has_pre_tess_adapter = parsed.capabilities.has_pre_tessellation;
  out->has_pre_quanttess_adapter =
      parsed.capabilities.has_pre_quantization && parsed.capabilities.has_pre_tessellation;
  out->has_post_dequant_adapter = parsed.capabilities.has_post_dequantization;
  out->has_post_detess_adapter = parsed.capabilities.has_post_detessellation;
  out->has_post_cast_adapter = parsed.capabilities.has_post_cast;
  out->pre_cast_needed = parsed.needs.pre_cast;
  out->post_cast_needed = parsed.needs.post_cast;
  out->cast_symmetry_ok = !out->pre_cast_needed || out->post_cast_needed;
  out->output_physical_count = parsed.outputs.physical.size();
  out->output_logical_count = parsed.outputs.logical.size();

  const auto& maybe_contract = pack.mpk_contract();
  if (maybe_contract.has_value()) {
    const auto logical_outputs =
        pipeline_internal::sima::get_mla_logical_outputs_contract(*maybe_contract);
    if (!logical_outputs.empty()) {
      out->mla_output_dtype_raw =
          canonical_dtype_for_signal(logical_outputs.front().dtype);
    }
    const auto* mla = pipeline_internal::sima::get_mla_stage_io_contract(*maybe_contract);
    if (mla) {
      if (!mla->input_tensors.empty()) {
        out->mla_input_dtype_raw = primary_tensor_dtype(mla->input_tensors);
      }
      if (!mla->output_tensors.empty() && out->mla_output_dtype_raw.empty()) {
        out->mla_output_dtype_raw = primary_tensor_dtype(mla->output_tensors);
      }
    }
  }
  if (out->mla_output_dtype_raw.empty() && !parsed.outputs.physical.empty()) {
    out->mla_output_dtype_raw = canonical_dtype_for_signal(parsed.outputs.physical.front().dtype);
  }

  return true;
}

} // namespace simaai::neat::internal
