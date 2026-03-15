#include "pipeline/internal/sima/SimaPluginStaticManifestResolver.h"

#include "pipeline/internal/sima/MlaStaticContractExtractor.h"
#include "pipeline/internal/sima/StageTransformRuleRegistry.h"
#include "pipeline/internal/TensorMath.h"

#include <algorithm>
#include <array>
#include <cctype>
#include <fstream>
#include <limits>
#include <optional>
#include <unordered_map>
#include <unordered_set>

namespace simaai::neat::pipeline_internal::sima {
namespace {

using json = nlohmann::json;

std::string lower_copy_local(std::string s) {
  std::transform(s.begin(), s.end(), s.begin(),
                 [](unsigned char c) { return static_cast<char>(std::tolower(c)); });
  return s;
}

const json* params_or_root(const json& root) {
  if (root.contains("simaai__params") && root["simaai__params"].is_object()) {
    return &root["simaai__params"];
  }
  return &root;
}

bool read_json_from_file(const std::string& path, json& out) {
  if (path.empty())
    return false;
  std::ifstream in(path);
  if (!in.is_open())
    return false;
  try {
    in >> out;
  } catch (const std::exception&) {
    return false;
  }
  return true;
}

template <typename T> std::vector<T> read_numeric_values_any(const json& value) {
  std::vector<T> out;
  if (value.is_array()) {
    out.reserve(value.size());
    for (const auto& item : value) {
      if (item.is_number_integer()) {
        out.push_back(static_cast<T>(item.get<std::int64_t>()));
      } else if (item.is_number()) {
        out.push_back(static_cast<T>(item.get<double>()));
      }
    }
    return out;
  }
  if (value.is_number_integer()) {
    out.push_back(static_cast<T>(value.get<std::int64_t>()));
  } else if (value.is_number()) {
    out.push_back(static_cast<T>(value.get<double>()));
  }
  return out;
}

std::optional<std::string> read_string_value_any(const json& value) {
  if (value.is_string()) {
    return value.get<std::string>();
  }
  return std::nullopt;
}

std::vector<std::string> read_string_values_any(const json& value) {
  std::vector<std::string> out;
  if (value.is_array()) {
    out.reserve(value.size());
    for (const auto& item : value) {
      if (item.is_string()) {
        out.push_back(item.get<std::string>());
      }
    }
    return out;
  }
  if (value.is_string()) {
    out.push_back(value.get<std::string>());
  }
  return out;
}

template <typename T>
std::vector<T> read_numeric_field_alias(const json& root, const json& params,
                                        std::initializer_list<const char*> keys) {
  for (const char* key : keys) {
    if (!key || !*key) {
      continue;
    }
    if (params.contains(key)) {
      auto values = read_numeric_values_any<T>(params.at(key));
      if (!values.empty()) {
        return values;
      }
    }
    if (root.contains(key)) {
      auto values = read_numeric_values_any<T>(root.at(key));
      if (!values.empty()) {
        return values;
      }
    }
  }
  return {};
}

template <typename T>
std::optional<T> read_scalar_field_alias(const json& root, const json& params,
                                         std::initializer_list<const char*> keys) {
  for (const char* key : keys) {
    if (!key || !*key) {
      continue;
    }
    if (params.contains(key)) {
      auto values = read_numeric_values_any<T>(params.at(key));
      if (!values.empty()) {
        return values.front();
      }
    }
    if (root.contains(key)) {
      auto values = read_numeric_values_any<T>(root.at(key));
      if (!values.empty()) {
        return values.front();
      }
    }
  }
  return std::nullopt;
}

std::optional<std::string> read_string_field_alias(const json& root, const json& params,
                                                   std::initializer_list<const char*> keys) {
  for (const char* key : keys) {
    if (!key || !*key) {
      continue;
    }
    if (params.contains(key)) {
      const auto value = read_string_value_any(params.at(key));
      if (value.has_value() && !value->empty()) {
        return value;
      }
    }
    if (root.contains(key)) {
      const auto value = read_string_value_any(root.at(key));
      if (value.has_value() && !value->empty()) {
        return value;
      }
    }
  }
  return std::nullopt;
}

std::vector<std::string> read_string_field_alias_values(const json& root, const json& params,
                                                        std::initializer_list<const char*> keys) {
  for (const char* key : keys) {
    if (!key || !*key) {
      continue;
    }
    if (params.contains(key)) {
      auto values = read_string_values_any(params.at(key));
      if (!values.empty()) {
        return values;
      }
    }
    if (root.contains(key)) {
      auto values = read_string_values_any(root.at(key));
      if (!values.empty()) {
        return values;
      }
    }
  }
  return {};
}

int to_non_negative_int(std::int64_t value) {
  if (value < 0) {
    return 0;
  }
  if (value > static_cast<std::int64_t>(std::numeric_limits<int>::max())) {
    return std::numeric_limits<int>::max();
  }
  return static_cast<int>(value);
}

std::string normalize_layout(const std::string& layout_raw) {
  const std::string up = upper_copy(layout_raw);
  if (up.find("NCHW") != std::string::npos || up.find("CHW") != std::string::npos) {
    return "CHW";
  }
  if (up.find("NHWC") != std::string::npos || up.find("HWC") != std::string::npos) {
    return "HWC";
  }
  if (up.find("HW") != std::string::npos) {
    return "HW";
  }
  return {};
}

template <typename T> T pick_or_default(const std::vector<T>& values, std::size_t index, T def) {
  if (values.empty()) {
    return def;
  }
  if (index < values.size()) {
    return values[index];
  }
  return values.front();
}

std::string pick_str_or_default(const std::vector<std::string>& values, std::size_t index,
                                const std::string& def) {
  if (values.empty()) {
    return def;
  }
  if (index < values.size()) {
    return values[index];
  }
  return values.front();
}

std::vector<std::int64_t> shape_from_layout(int width, int height, int depth,
                                            const std::string& layout_raw) {
  if (width <= 0 || height <= 0) {
    return {};
  }
  const std::string layout = normalize_layout(layout_raw);
  if (layout == "CHW") {
    if (depth <= 0) {
      return {};
    }
    return {depth, height, width};
  }
  if (layout == "HW") {
    return {height, width};
  }
  if (depth <= 0) {
    return {};
  }
  return {height, width, depth};
}

std::size_t tensor_count_from_fields(const std::vector<std::int64_t>& widths,
                                     const std::vector<std::int64_t>& heights,
                                     const std::vector<std::int64_t>& depths,
                                     const std::vector<std::string>& dtypes,
                                     const std::vector<std::string>& layouts) {
  std::size_t count = 0;
  count = std::max<std::size_t>(count, widths.size());
  count = std::max<std::size_t>(count, heights.size());
  count = std::max<std::size_t>(count, depths.size());
  count = std::max<std::size_t>(count, dtypes.size());
  count = std::max<std::size_t>(count, layouts.size());
  return count;
}

std::vector<TensorStaticSpec> extract_tensor_specs_from_stage_config(
    const json& root, const json& params, std::initializer_list<const char*> width_keys,
    std::initializer_list<const char*> height_keys, std::initializer_list<const char*> depth_keys,
    std::initializer_list<const char*> layout_keys, std::initializer_list<const char*> dtype_keys,
    const std::string& semantic_prefix) {
  const auto widths = read_numeric_field_alias<std::int64_t>(root, params, width_keys);
  const auto heights = read_numeric_field_alias<std::int64_t>(root, params, height_keys);
  auto depths = read_numeric_field_alias<std::int64_t>(root, params, depth_keys);
  if (depths.empty()) {
    depths =
        read_numeric_field_alias<std::int64_t>(root, params, {"input_channels", "output_channels"});
  }
  const auto layouts = read_string_field_alias_values(root, params, layout_keys);
  const auto dtypes = read_string_field_alias_values(root, params, dtype_keys);
  const std::size_t count = tensor_count_from_fields(widths, heights, depths, dtypes, layouts);
  if (count == 0) {
    return {};
  }

  std::vector<TensorStaticSpec> tensors;
  tensors.reserve(count);
  for (std::size_t i = 0; i < count; ++i) {
    const int width = to_non_negative_int(pick_or_default<std::int64_t>(widths, i, 0));
    const int height = to_non_negative_int(pick_or_default<std::int64_t>(heights, i, 0));
    const int depth = to_non_negative_int(pick_or_default<std::int64_t>(depths, i, 0));
    const std::string layout_raw = pick_str_or_default(layouts, i, "HWC");
    const std::string layout = normalize_layout(layout_raw);
    const auto shape = shape_from_layout(width, height, depth, layout);
    if (shape.empty()) {
      continue;
    }

    TensorStaticSpec tensor;
    tensor.tensor_index = static_cast<int>(i);
    tensor.shape = shape;
    tensor.dtype = pick_str_or_default(dtypes, i, "INT8");
    tensor.layout = layout.empty() ? "HWC" : layout;
    tensor.max_w = width;
    tensor.max_h = height;
    tensor.max_stride = 0;
    tensor.semantic_tag = semantic_prefix + "_" + std::to_string(i);
    tensors.push_back(std::move(tensor));
  }
  return tensors;
}

std::vector<QuantStaticSpec> build_quant_specs_from_vectors(const std::vector<double>& q_scale,
                                                            const std::vector<std::int64_t>& q_zp,
                                                            const std::vector<std::int64_t>& q_axis,
                                                            std::size_t tensor_count_hint) {
  if (q_scale.empty() || q_zp.empty()) {
    return {};
  }

  const std::size_t tensor_count =
      std::max<std::size_t>(static_cast<std::size_t>(1), tensor_count_hint);
  const bool per_tensor_scalars =
      tensor_count > 1 && q_scale.size() >= tensor_count && q_zp.size() >= tensor_count;

  std::vector<QuantStaticSpec> out;
  if (per_tensor_scalars) {
    out.reserve(tensor_count);
    for (std::size_t i = 0; i < tensor_count; ++i) {
      QuantStaticSpec quant;
      quant.granularity = QuantGranularity::PerTensor;
      quant.axis = -1;
      quant.scales.push_back(q_scale[i]);
      quant.zero_points.push_back(q_zp[i]);
      out.push_back(std::move(quant));
    }
    return out;
  }

  QuantStaticSpec quant;
  quant.granularity = (q_scale.size() > 1 || q_zp.size() > 1) ? QuantGranularity::PerAxis
                                                              : QuantGranularity::PerTensor;
  quant.axis = q_axis.empty() ? -1 : static_cast<int>(q_axis.front());
  quant.scales = q_scale;
  quant.zero_points = q_zp;
  out.push_back(std::move(quant));
  return out;
}

std::vector<QuantStaticSpec> extract_quant_specs_from_config(
    const json& root, const json& params, std::initializer_list<const char*> scale_keys,
    std::initializer_list<const char*> zp_keys, std::size_t tensor_count_hint) {
  const auto q_scale = read_numeric_field_alias<double>(root, params, scale_keys);
  const auto q_zp = read_numeric_field_alias<std::int64_t>(root, params, zp_keys);
  if (q_scale.empty() || q_zp.empty()) {
    return {};
  }
  const auto q_axis = read_numeric_field_alias<std::int64_t>(root, params, {"q_axis", "axis"});
  return build_quant_specs_from_vectors(q_scale, q_zp, q_axis, tensor_count_hint);
}

bool is_processmla_plugin(const std::string& plugin) {
  const std::string lower = lower_copy_local(plugin);
  return lower.find("processmla") != std::string::npos;
}

bool is_processcvu_plugin(const std::string& plugin) {
  const std::string lower = lower_copy_local(plugin);
  return lower.find("processcvu") != std::string::npos;
}

bool is_boxdecode_plugin(const std::string& plugin) {
  const std::string lower = lower_copy_local(plugin);
  return lower.find("boxdecode") != std::string::npos;
}

bool is_tessellate_plugin(const std::string& plugin) {
  const std::string lower = lower_copy_local(plugin);
  return lower.find("tessellate") != std::string::npos;
}

bool is_dequantize_plugin(const std::string& plugin) {
  const std::string lower = lower_copy_local(plugin);
  return lower.find("dequantize") != std::string::npos;
}

bool requires_static_quant_contract(const StageStaticSpec& stage) {
  const std::string kernel = lower_copy_local(stage.kernel_kind);
  return kernel == "detessdequant" || kernel == "detessellate" || kernel == "slicedequant" ||
         kernel == "quantize" || kernel == "dequantize";
}

std::string infer_kernel_kind(const PipelineElementSpec& element) {
  const std::string plugin = lower_copy_local(element.plugin);
  const std::string path = lower_copy_local(element.config_path);
  const std::string stage_name = lower_copy_local(element.element_name);
  const std::string stage_id = lower_copy_local(element.stage_id);
  if (is_processmla_plugin(plugin))
    return "mla";
  if (stage_name.find("detessdequant") != std::string::npos ||
      stage_id.find("detessdequant") != std::string::npos) {
    return "detessdequant";
  }
  if (stage_name.find("detessellate") != std::string::npos ||
      stage_id.find("detessellate") != std::string::npos) {
    return "detessellate";
  }
  if (stage_name.find("slicedequant") != std::string::npos ||
      stage_id.find("slicedequant") != std::string::npos) {
    return "slicedequant";
  }
  if (stage_name.find("dequantize") != std::string::npos ||
      stage_id.find("dequantize") != std::string::npos) {
    return "dequantize";
  }
  if (stage_name.find("tessellate") != std::string::npos ||
      stage_id.find("tessellate") != std::string::npos) {
    return "tessellate";
  }
  if (stage_name.find("quantize") != std::string::npos ||
      stage_id.find("quantize") != std::string::npos) {
    return "quantize";
  }
  if (stage_name.find("boxdecode") != std::string::npos ||
      stage_id.find("boxdecode") != std::string::npos) {
    return "boxdecode";
  }
  if (stage_name.find("quanttess") != std::string::npos ||
      stage_id.find("quanttess") != std::string::npos) {
    return "quanttess";
  }
  if (stage_name.find("preproc") != std::string::npos ||
      stage_id.find("preproc") != std::string::npos) {
    return "preproc";
  }

  if (plugin.find("detessdequant") != std::string::npos ||
      path.find("detessdequant") != std::string::npos)
    return "detessdequant";
  if (plugin.find("detessellate") != std::string::npos ||
      path.find("detessellate") != std::string::npos)
    return "detessellate";
  if (plugin.find("slicedequant") != std::string::npos ||
      path.find("slicedequant") != std::string::npos)
    return "slicedequant";
  if (plugin.find("dequantize") != std::string::npos ||
      path.find("dequantize") != std::string::npos || is_dequantize_plugin(plugin))
    return "dequantize";
  if (plugin.find("tessellate") != std::string::npos ||
      path.find("tessellate") != std::string::npos || is_tessellate_plugin(plugin))
    return "tessellate";
  if (plugin.find("quantize") != std::string::npos || path.find("quantize") != std::string::npos)
    return "quantize";
  if (is_boxdecode_plugin(plugin))
    return "boxdecode";
  if (path.find("preproc") != std::string::npos)
    return "preproc";
  if (path.find("quanttess") != std::string::npos)
    return "quanttess";

  if (is_processcvu_plugin(plugin))
    return "cvu";

  return {};
}

void append_trace(StageStaticSpec& stage, std::string field, std::string source_used,
                  bool conflict = false) {
  ResolutionTrace trace;
  trace.field = std::move(field);
  trace.source_used = std::move(source_used);
  trace.fallback_chain = {"infer", "caps/property/context", "json"};
  trace.conflict = conflict;
  stage.resolution_trace.push_back(std::move(trace));
}

void set_stage_output_quant(StageStaticSpec& stage, std::vector<QuantStaticSpec> quant,
                            const std::string& source_used) {
  if (quant.empty()) {
    return;
  }
  stage.output_quant = std::move(quant);
  append_trace(stage, "output_quant", source_used);
}

bool is_upstream_mla_input_quant_kernel(const std::string& kernel_kind) {
  const std::string kernel = lower_copy_local(kernel_kind);
  return kernel == "preproc" || kernel == "quanttess" || kernel == "quantize" || kernel == "cvu";
}

bool is_downstream_mla_output_quant_kernel(const std::string& kernel_kind) {
  const std::string kernel = lower_copy_local(kernel_kind);
  return kernel == "detessdequant" || kernel == "detessellate" || kernel == "slicedequant" ||
         kernel == "dequantize" || kernel == "boxdecode";
}

void build_sink_pad_tensor_index_map(StageStaticSpec& stage,
                                     ManifestBuildDiagnostics* diagnostics) {
  stage.sink_pad_tensor_index_map.clear();
  stage.sink_pad_tensor_index_map.reserve(stage.inputs.size());

  std::unordered_set<int> seen;
  bool duplicate = false;
  for (std::size_t i = 0; i < stage.inputs.size(); ++i) {
    const int tensor_idx =
        (stage.inputs[i].tensor_index >= 0) ? stage.inputs[i].tensor_index : static_cast<int>(i);
    stage.sink_pad_tensor_index_map.push_back(tensor_idx);
    if (!seen.insert(tensor_idx).second) {
      duplicate = true;
    }
  }

  if (!stage.sink_pad_tensor_index_map.empty()) {
    append_trace(stage, "sink_pad_tensor_index_map", "infer:stage_input_order", duplicate);
  }
  if (duplicate && diagnostics) {
    diagnostics->errors.push_back(
        "Ambiguous sink_pad_tensor_index_map for stage '" + stage.element_name +
        "' (duplicate tensor indices; deterministic pad mapping unavailable)");
  }
}

struct StageInputRouteHint {
  int tensor_index = -1;
  std::string cm_input_name;
  std::string source_segment_name;
};

struct StageOutputRouteHint {
  int output_slot = 0;
  std::string cm_output_name;
  std::string segment_name;
};

std::optional<int> read_optional_int(const nlohmann::json& value) {
  if (value.is_number_integer()) {
    return value.get<int>();
  }
  if (value.is_number()) {
    return static_cast<int>(value.get<double>());
  }
  return std::nullopt;
}

std::vector<StageInputRouteHint> parse_input_route_hints_from_stage_config(const json& cfg) {
  std::vector<StageInputRouteHint> hints;
  if (!cfg.is_object() || !cfg.contains("input_buffers")) {
    return hints;
  }
  const auto& input_buffers = cfg.at("input_buffers");
  if (!input_buffers.is_array() && !input_buffers.is_object()) {
    return hints;
  }

  auto parse_memories = [](const nlohmann::json& entry,
                           std::size_t default_tensor_index) -> std::vector<StageInputRouteHint> {
    std::vector<StageInputRouteHint> out;
    if (!entry.is_object() || !entry.contains("memories") || !entry.at("memories").is_array()) {
      return out;
    }
    std::optional<int> maybe_tensor_idx = std::nullopt;
    if (entry.contains("tensor_index")) {
      maybe_tensor_idx = read_optional_int(entry.at("tensor_index"));
    }
    const auto& memories = entry.at("memories");
    if (memories.size() == 1 && memories.front().is_object()) {
      StageInputRouteHint hint;
      hint.tensor_index =
          maybe_tensor_idx.has_value() ? *maybe_tensor_idx : static_cast<int>(default_tensor_index);
      if (memories.front().contains("graph_input_name") &&
          memories.front().at("graph_input_name").is_string()) {
        hint.cm_input_name = memories.front().at("graph_input_name").get<std::string>();
      }
      if (memories.front().contains("segment_name") &&
          memories.front().at("segment_name").is_string()) {
        hint.source_segment_name = memories.front().at("segment_name").get<std::string>();
      }
      out.push_back(std::move(hint));
      return out;
    }

    for (std::size_t mem_idx = 0; mem_idx < memories.size(); ++mem_idx) {
      if (!memories[mem_idx].is_object()) {
        continue;
      }
      StageInputRouteHint hint;
      if (maybe_tensor_idx.has_value()) {
        hint.tensor_index = *maybe_tensor_idx + static_cast<int>(mem_idx);
      } else {
        hint.tensor_index = static_cast<int>(mem_idx);
      }
      if (memories[mem_idx].contains("graph_input_name") &&
          memories[mem_idx].at("graph_input_name").is_string()) {
        hint.cm_input_name = memories[mem_idx].at("graph_input_name").get<std::string>();
      }
      if (memories[mem_idx].contains("segment_name") &&
          memories[mem_idx].at("segment_name").is_string()) {
        hint.source_segment_name = memories[mem_idx].at("segment_name").get<std::string>();
      }
      out.push_back(std::move(hint));
    }
    return out;
  };

  if (input_buffers.is_array()) {
    for (std::size_t i = 0; i < input_buffers.size(); ++i) {
      auto parsed = parse_memories(input_buffers[i], i);
      hints.insert(hints.end(), parsed.begin(), parsed.end());
    }
    return hints;
  }

  std::size_t i = 0;
  for (const auto& item : input_buffers.items()) {
    auto parsed = parse_memories(item.value(), i);
    hints.insert(hints.end(), parsed.begin(), parsed.end());
    ++i;
  }
  return hints;
}

std::vector<StageOutputRouteHint> parse_output_route_hints_from_stage_config(const json& cfg) {
  std::vector<StageOutputRouteHint> hints;
  if (!cfg.is_object() || !cfg.contains("output_memory_order")) {
    return hints;
  }
  const auto& output_order = cfg.at("output_memory_order");
  if (!output_order.is_array() && !output_order.is_object()) {
    return hints;
  }

  auto push_output_name = [&](const nlohmann::json& entry, int slot_hint) {
    if (!entry.is_string()) {
      return;
    }
    const std::string name = entry.get<std::string>();
    if (name.empty()) {
      return;
    }
    StageOutputRouteHint hint;
    hint.output_slot = slot_hint;
    hint.cm_output_name = name;
    hint.segment_name = name;
    hints.push_back(std::move(hint));
  };

  if (output_order.is_array()) {
    for (std::size_t i = 0; i < output_order.size(); ++i) {
      push_output_name(output_order[i], static_cast<int>(i));
    }
    return hints;
  }

  int i = 0;
  for (const auto& item : output_order.items()) {
    push_output_name(item.value(), i);
    ++i;
  }
  return hints;
}

bool is_preproc_stage(const StageStaticSpec& stage) {
  return lower_copy_local(stage.kernel_kind) == "preproc";
}

std::vector<StageOutputRouteHint>
ensure_preproc_output_route_coverage(const StageStaticSpec& stage,
                                     const std::optional<json>& stage_config,
                                     std::vector<StageOutputRouteHint> hints) {
  if (!is_preproc_stage(stage)) {
    return hints;
  }

  bool next_cpu_apu = false;
  if (stage_config.has_value() && stage_config->is_object()) {
    const json* params_ptr = params_or_root(*stage_config);
    const json& params = params_ptr ? *params_ptr : *stage_config;
    if (const auto next_cpu = read_string_field_alias(*stage_config, params, {"next_cpu"});
        next_cpu.has_value()) {
      const std::string up = lower_copy_local(*next_cpu);
      next_cpu_apu = (up == "apu" || up == "a65" || up == "tvm");
    }
  }

  auto has_name = [&](const std::string& name) {
    return std::any_of(hints.begin(), hints.end(), [&](const StageOutputRouteHint& hint) {
      return hint.cm_output_name == name || hint.segment_name == name;
    });
  };
  auto append_name = [&](const std::string& name) {
    if (has_name(name)) {
      return;
    }
    StageOutputRouteHint hint;
    hint.output_slot = static_cast<int>(hints.size());
    hint.cm_output_name = name;
    hint.segment_name = name;
    hints.push_back(std::move(hint));
  };

  if (next_cpu_apu) {
    append_name("output_rgb_image");
    append_name("output_tessellated_image");
  } else {
    append_name("output_tessellated_image");
    append_name("output_rgb_image");
  }

  for (std::size_t i = 0; i < hints.size(); ++i) {
    hints[i].output_slot = static_cast<int>(i);
  }
  return hints;
}

std::optional<int> output_slot_for_tensor(const StageStaticSpec& upstream, int tensor_index) {
  if (tensor_index < 0) {
    return std::nullopt;
  }
  for (std::size_t i = 0; i < upstream.outputs.size(); ++i) {
    if (upstream.outputs[i].tensor_index == tensor_index) {
      return static_cast<int>(i);
    }
  }
  if (static_cast<std::size_t>(tensor_index) < upstream.outputs.size()) {
    return tensor_index;
  }
  return std::nullopt;
}

void build_manifest_v2_routes(std::vector<StageStaticSpec>& stages,
                              const std::vector<std::optional<json>>& parsed_stage_configs) {
  if (stages.empty()) {
    return;
  }

  std::vector<std::vector<StageInputRouteHint>> input_hints(stages.size());
  std::vector<std::vector<StageOutputRouteHint>> output_hints(stages.size());
  for (std::size_t i = 0; i < stages.size(); ++i) {
    if (i < parsed_stage_configs.size() && parsed_stage_configs[i].has_value()) {
      input_hints[i] = parse_input_route_hints_from_stage_config(*parsed_stage_configs[i]);
      output_hints[i] = parse_output_route_hints_from_stage_config(*parsed_stage_configs[i]);
    }
  }

  for (std::size_t i = 0; i < stages.size(); ++i) {
    auto& stage = stages[i];
    stage.sink_routes.clear();
    stage.output_order.clear();

    const auto& hints = input_hints[i];
    std::unordered_map<int, StageInputRouteHint> hints_by_tensor;
    std::unordered_map<int, StageInputRouteHint> hints_by_pad;
    for (const auto& hint : hints) {
      if (hint.tensor_index >= 0) {
        hints_by_tensor[hint.tensor_index] = hint;
      }
    }
    for (std::size_t pad_idx = 0; pad_idx < hints.size(); ++pad_idx) {
      hints_by_pad[static_cast<int>(pad_idx)] = hints[pad_idx];
    }

    // Topology-first: route structure comes from resolved stage input contracts.
    // Stage config hints (input_buffers) only decorate names/segments.
    const std::size_t base_route_count =
        std::max<std::size_t>(stage.sink_pad_tensor_index_map.size(), stage.inputs.size());
    const std::size_t route_count =
        !hints.empty() ? hints.size() : ((base_route_count > 0) ? base_route_count : hints.size());

    const StageStaticSpec* upstream = (i > 0) ? &stages[i - 1] : nullptr;
    const std::string upstream_stage_id =
        upstream ? (!upstream->logical_stage_id.empty() ? upstream->logical_stage_id
                                                        : upstream->element_name)
                 : "__session_input__";

    for (std::size_t pad_idx = 0; pad_idx < route_count; ++pad_idx) {
      StageSinkRoute route;
      route.sink_pad_index = static_cast<int>(pad_idx);
      route.required = true;
      route.src_stage_id = upstream_stage_id;

      if (pad_idx < stage.sink_pad_tensor_index_map.size()) {
        route.tensor_index = stage.sink_pad_tensor_index_map[pad_idx];
      } else if (pad_idx < stage.inputs.size()) {
        route.tensor_index = (stage.inputs[pad_idx].tensor_index >= 0)
                                 ? stage.inputs[pad_idx].tensor_index
                                 : static_cast<int>(pad_idx);
      } else if (pad_idx < hints.size() && hints[pad_idx].tensor_index >= 0) {
        route.tensor_index = hints[pad_idx].tensor_index;
      } else {
        route.tensor_index = static_cast<int>(pad_idx);
      }

      if (const auto hint_it = hints_by_tensor.find(route.tensor_index);
          hint_it != hints_by_tensor.end()) {
        route.cm_input_name = hint_it->second.cm_input_name;
        route.source_segment_name = hint_it->second.source_segment_name;
      } else if (const auto hint_pad_it = hints_by_pad.find(static_cast<int>(pad_idx));
                 hint_pad_it != hints_by_pad.end()) {
        route.cm_input_name = hint_pad_it->second.cm_input_name;
        route.source_segment_name = hint_pad_it->second.source_segment_name;
      }
      if (route.cm_input_name.empty() && !hints.empty()) {
        route.cm_input_name = hints.front().cm_input_name;
        if (route.source_segment_name.empty()) {
          route.source_segment_name = hints.front().source_segment_name;
        }
      }

      if (upstream) {
        route.src_output_slot = output_slot_for_tensor(*upstream, route.tensor_index)
                                    .value_or(static_cast<int>(pad_idx));
      } else {
        route.src_output_slot = static_cast<int>(pad_idx);
        route.source_segment_name.clear();
      }

      stage.sink_routes.push_back(std::move(route));
    }

    auto out_hints = output_hints[i];
    const std::optional<json> stage_cfg =
        (i < parsed_stage_configs.size()) ? parsed_stage_configs[i] : std::optional<json>{};
    out_hints = ensure_preproc_output_route_coverage(stage, stage_cfg, std::move(out_hints));
    if (out_hints.empty()) {
      continue;
    }
    std::unordered_map<int, StageOutputRouteHint> out_hints_by_slot;
    for (const auto& hint : out_hints) {
      out_hints_by_slot[hint.output_slot] = hint;
    }

    const std::size_t output_route_count = out_hints.size();
    stage.output_order.reserve(output_route_count);
    for (std::size_t slot = 0; slot < output_route_count; ++slot) {
      StageOutputRoute route;
      route.output_slot = static_cast<int>(slot);

      if (const auto hit = out_hints_by_slot.find(static_cast<int>(slot));
          hit != out_hints_by_slot.end()) {
        route.cm_output_name = hit->second.cm_output_name;
        route.segment_name = hit->second.segment_name;
      }

      if (route.cm_output_name.empty()) {
        route.cm_output_name = "ofm" + std::to_string(slot);
      }
      if (route.segment_name.empty()) {
        route.segment_name = route.cm_output_name;
      }

      stage.output_order.push_back(std::move(route));
    }
  }
}

std::optional<std::size_t> find_mla_index(const std::vector<PipelineElementSpec>& elements) {
  for (std::size_t i = 0; i < elements.size(); ++i) {
    if (is_processmla_plugin(elements[i].plugin))
      return i;
  }
  return std::nullopt;
}

std::optional<std::string> read_model_id_from_mla_config(const json& root) {
  if (!root.is_object())
    return std::nullopt;
  const json* params_ptr = params_or_root(root);
  if (!params_ptr || !params_ptr->is_object())
    return std::nullopt;
  const json& params = *params_ptr;
  if (params.contains("model_path") && params["model_path"].is_string()) {
    return params["model_path"].get<std::string>();
  }
  if (root.contains("model_path") && root["model_path"].is_string()) {
    return root["model_path"].get<std::string>();
  }
  return std::nullopt;
}

} // namespace

SimaPluginStaticManifest resolve_manifest_from_pipeline(const std::string& pipeline_string,
                                                        const std::string& session_id,
                                                        ManifestBuildDiagnostics* diagnostics) {
  if (diagnostics) {
    diagnostics->warnings.clear();
    diagnostics->errors.clear();
  }

  SimaPluginStaticManifest manifest;
  manifest.manifest_version = 2;
  manifest.session_id = session_id;

  const std::vector<PipelineElementSpec> elements = parse_pipeline_elements(pipeline_string);
  if (elements.empty())
    return manifest;

  std::vector<std::optional<json>> parsed_stage_configs(elements.size());
  std::vector<std::string> inferred_kernel_kinds(elements.size());
  for (std::size_t i = 0; i < elements.size(); ++i) {
    inferred_kernel_kinds[i] = infer_kernel_kind(elements[i]);
    const auto& cfg_path = elements[i].config_path;
    if (cfg_path.empty()) {
      continue;
    }
    json cfg;
    if (read_json_from_file(cfg_path, cfg)) {
      parsed_stage_configs[i] = std::move(cfg);
    } else if (diagnostics) {
      diagnostics->warnings.push_back("Failed to load stage config path=" + cfg_path);
    }
  }

  std::optional<std::size_t> mla_index = find_mla_index(elements);
  std::optional<json> mla_config;
  std::optional<MlaStaticContract> mla_contract;
  if (mla_index.has_value()) {
    const auto& mla_element = elements[*mla_index];
    if (parsed_stage_configs[*mla_index].has_value()) {
      mla_config = parsed_stage_configs[*mla_index];
    } else if (!mla_element.config_path.empty() && diagnostics) {
      diagnostics->warnings.push_back("Failed to load MLA config path=" + mla_element.config_path);
    }
    if (mla_config.has_value()) {
      std::string mla_error;
      mla_contract = extract_mla_static_contract(*mla_config, &mla_error);
      if (!mla_contract.has_value() && diagnostics) {
        diagnostics->warnings.push_back("MLA static contract unavailable: " + mla_error);
      }
      if (mla_contract.has_value()) {
        if (const auto model_id = read_model_id_from_mla_config(*mla_config);
            model_id.has_value()) {
          manifest.model_id = *model_id;
        }
      }
    } else if (diagnostics) {
      diagnostics->warnings.push_back("MLA stage has no parseable config JSON");
    }
  }

  std::optional<std::vector<QuantStaticSpec>> mla_input_quant_from_upstream;
  std::optional<std::vector<QuantStaticSpec>> mla_output_quant_from_downstream;
  if (mla_index.has_value()) {
    for (std::size_t rev = *mla_index; rev > 0; --rev) {
      const std::size_t i = rev - 1;
      if (!parsed_stage_configs[i].has_value()) {
        continue;
      }
      if (!is_upstream_mla_input_quant_kernel(inferred_kernel_kinds[i])) {
        continue;
      }
      const json& cfg = *parsed_stage_configs[i];
      const json* params_ptr = params_or_root(cfg);
      const json& params = params_ptr ? *params_ptr : cfg;
      const std::size_t tensor_hint = mla_contract.has_value() ? mla_contract->inputs.size() : 1;
      auto quant = extract_quant_specs_from_config(cfg, params, {"q_scale", "quant_scale"},
                                                   {"q_zp", "zero_point"}, tensor_hint);
      if (!quant.empty()) {
        mla_input_quant_from_upstream = std::move(quant);
        break;
      }
    }

    for (std::size_t i = *mla_index + 1; i < elements.size(); ++i) {
      if (!parsed_stage_configs[i].has_value()) {
        continue;
      }
      if (!is_downstream_mla_output_quant_kernel(inferred_kernel_kinds[i])) {
        continue;
      }
      const json& cfg = *parsed_stage_configs[i];
      const json* params_ptr = params_or_root(cfg);
      const json& params = params_ptr ? *params_ptr : cfg;
      const std::size_t tensor_hint = mla_contract.has_value() ? mla_contract->outputs.size() : 1;
      auto quant =
          extract_quant_specs_from_config(cfg, params, {"q_scale", "quant_scale", "dq_scale"},
                                          {"q_zp", "zero_point", "dq_zp"}, tensor_hint);
      if (!quant.empty()) {
        mla_output_quant_from_downstream = std::move(quant);
        break;
      }
    }
  }

  if (mla_contract.has_value() && mla_contract->output_quant.empty() &&
      mla_output_quant_from_downstream.has_value()) {
    mla_contract->output_quant = *mla_output_quant_from_downstream;
  }

  manifest.stages.reserve(elements.size());
  for (std::size_t i = 0; i < elements.size(); ++i) {
    const auto& element = elements[i];
    StageStaticSpec stage;
    stage.element_name = element.element_name.empty() ? (element.plugin + "_" + std::to_string(i))
                                                      : element.element_name;
    stage.plugin_kind = element.plugin;
    stage.logical_stage_id = element.stage_id.empty() ? stage.element_name : element.stage_id;
    stage.kernel_kind = inferred_kernel_kinds[i];
    const bool model_managed_stage = !element.stage_id.empty();
    stage.runtime_defaults = nlohmann::json::object();
    if (!element.config_path.empty()) {
      stage.runtime_defaults["config_path"] = element.config_path;
    }
    if (element.decode_type_property.has_value() && !element.decode_type_property->empty()) {
      stage.runtime_defaults["decode_type"] = *element.decode_type_property;
      append_trace(stage, "runtime.decode_type", "property");
    }
    if (element.detection_threshold_property.has_value()) {
      stage.runtime_defaults["detection_threshold"] = *element.detection_threshold_property;
      append_trace(stage, "runtime.detection_threshold", "property");
    }
    if (element.nms_iou_threshold_property.has_value()) {
      stage.runtime_defaults["nms_iou_threshold"] = *element.nms_iou_threshold_property;
      append_trace(stage, "runtime.nms_iou_threshold", "property");
    }
    if (element.topk_property.has_value()) {
      stage.runtime_defaults["topk"] = *element.topk_property;
      append_trace(stage, "runtime.topk", "property");
    }
    const std::optional<json>& stage_config = parsed_stage_configs[i];
    // Model-managed boxdecode stages still rely on packaged config JSON for
    // decode/runtime defaults when the node does not set explicit properties.
    const bool allow_model_managed_json_defaults = is_boxdecode_plugin(stage.plugin_kind);
    if (stage_config.has_value() && (!model_managed_stage || allow_model_managed_json_defaults)) {
      const json* stage_params_ptr = params_or_root(*stage_config);
      const json& stage_params = stage_params_ptr ? *stage_params_ptr : *stage_config;

      if (!stage.runtime_defaults.contains("decode_type")) {
        if (const auto decode =
                read_string_field_alias(*stage_config, stage_params, {"decode_type"});
            decode.has_value()) {
          stage.runtime_defaults["decode_type"] = *decode;
          append_trace(stage, "runtime.decode_type", "json_fallback:stage_config");
        }
      }
      if (!stage.runtime_defaults.contains("detection_threshold")) {
        if (const auto det = read_scalar_field_alias<double>(*stage_config, stage_params,
                                                             {"detection_threshold"});
            det.has_value()) {
          stage.runtime_defaults["detection_threshold"] = *det;
          append_trace(stage, "runtime.detection_threshold", "json_fallback:stage_config");
        }
      }
      if (!stage.runtime_defaults.contains("nms_iou_threshold")) {
        if (const auto nms =
                read_scalar_field_alias<double>(*stage_config, stage_params, {"nms_iou_threshold"});
            nms.has_value()) {
          stage.runtime_defaults["nms_iou_threshold"] = *nms;
          append_trace(stage, "runtime.nms_iou_threshold", "json_fallback:stage_config");
        }
      }
      if (!stage.runtime_defaults.contains("topk")) {
        if (const auto topk = read_scalar_field_alias<int>(*stage_config, stage_params, {"topk"});
            topk.has_value()) {
          stage.runtime_defaults["topk"] = *topk;
          append_trace(stage, "runtime.topk", "json_fallback:stage_config");
        }
      }
    }

    const bool is_mla_stage = mla_index.has_value() && *mla_index == i;
    const bool is_upstream_of_mla = mla_index.has_value() && i < *mla_index;
    const bool is_downstream_of_mla = mla_index.has_value() && i > *mla_index;
    const auto transform_rule = default_stage_transform_rules().lookup(stage.kernel_kind);

    if (is_mla_stage && mla_contract.has_value()) {
      stage.inputs = mla_contract->inputs;
      stage.outputs = mla_contract->outputs;
      stage.output_quant = mla_contract->output_quant;
      append_trace(stage, "inputs", "infer:mla_contract");
      append_trace(stage, "outputs", "infer:mla_contract");
      if (!stage.output_quant.empty()) {
        append_trace(stage, "output_quant", "infer:mla_contract");
        if (mla_output_quant_from_downstream.has_value()) {
          append_trace(stage, "output_quant", "json_fallback:downstream_stage_config");
        }
      }
    }

    if (is_mla_stage && mla_input_quant_from_upstream.has_value() &&
        !mla_input_quant_from_upstream->empty()) {
      nlohmann::json input_quant = nlohmann::json::array();
      for (const auto& quant : *mla_input_quant_from_upstream) {
        input_quant.push_back(to_json(quant));
      }
      stage.runtime_defaults["input_quant"] = std::move(input_quant);
      if (!mla_input_quant_from_upstream->front().scales.empty()) {
        stage.runtime_defaults["input_q_scale"] = mla_input_quant_from_upstream->front().scales;
      }
      if (!mla_input_quant_from_upstream->front().zero_points.empty()) {
        stage.runtime_defaults["input_q_zp"] = mla_input_quant_from_upstream->front().zero_points;
      }
      append_trace(stage, "runtime.input_quant", "json_fallback:upstream_stage_config");
    }

    if (!is_mla_stage && mla_contract.has_value() && transform_rule.has_value()) {
      const auto apply_tensor_source = [&](StageTensorSource source, bool for_inputs) {
        if (source == StageTensorSource::MlaInputs && is_upstream_of_mla) {
          if (for_inputs) {
            stage.inputs = mla_contract->inputs;
            append_trace(stage, "inputs", "infer:mla_transform_registry");
          } else {
            stage.outputs = mla_contract->inputs;
            append_trace(stage, "outputs", "infer:mla_transform_registry");
          }
        } else if (source == StageTensorSource::MlaOutputs && is_downstream_of_mla) {
          if (for_inputs) {
            stage.inputs = mla_contract->outputs;
            append_trace(stage, "inputs", "infer:mla_transform_registry");
          } else {
            stage.outputs = mla_contract->outputs;
            append_trace(stage, "outputs", "infer:mla_transform_registry");
          }
        }
      };

      apply_tensor_source(transform_rule->input_source, /*for_inputs=*/true);
      apply_tensor_source(transform_rule->output_source, /*for_inputs=*/false);

      if (transform_rule->propagate_output_quant && is_downstream_of_mla &&
          !mla_contract->output_quant.empty()) {
        stage.output_quant = mla_contract->output_quant;
        append_trace(stage, "output_quant", "infer:mla_transform_registry");
      }
    }

    if (stage.inputs.empty() && stage_config.has_value()) {
      const json* stage_params_ptr = params_or_root(*stage_config);
      const json& stage_params = stage_params_ptr ? *stage_params_ptr : *stage_config;
      auto inferred_inputs = extract_tensor_specs_from_stage_config(
          *stage_config, stage_params, {"input_width", "slice_width"},
          {"input_height", "slice_height"}, {"input_depth", "slice_depth"},
          {"input_format", "slice_format"}, {"input_dtype", "data_type"}, "json_input");
      if (!inferred_inputs.empty()) {
        stage.inputs = std::move(inferred_inputs);
        append_trace(stage, "inputs", "json_fallback:stage_config");
      }
    }

    if (stage.outputs.empty() && stage_config.has_value()) {
      const json* stage_params_ptr = params_or_root(*stage_config);
      const json& stage_params = stage_params_ptr ? *stage_params_ptr : *stage_config;
      auto inferred_outputs = extract_tensor_specs_from_stage_config(
          *stage_config, stage_params, {"output_width", "slice_width"},
          {"output_height", "slice_height"}, {"output_depth", "slice_depth"},
          {"output_format", "slice_format"}, {"data_type", "output_dtype"}, "json_output");
      if (!inferred_outputs.empty()) {
        stage.outputs = std::move(inferred_outputs);
        append_trace(stage, "outputs", "json_fallback:stage_config");
      }
    }

    if (stage.output_quant.empty() && stage_config.has_value()) {
      const json* stage_params_ptr = params_or_root(*stage_config);
      const json& stage_params = stage_params_ptr ? *stage_params_ptr : *stage_config;
      const std::size_t tensor_hint = std::max<std::size_t>(
          static_cast<std::size_t>(1), std::max(stage.inputs.size(), stage.outputs.size()));
      const std::string kernel = lower_copy_local(stage.kernel_kind);
      if (kernel == "preproc" || kernel == "quanttess" || kernel == "quantize" || kernel == "cvu") {
        auto quant =
            extract_quant_specs_from_config(*stage_config, stage_params, {"q_scale", "quant_scale"},
                                            {"q_zp", "zero_point"}, tensor_hint);
        set_stage_output_quant(stage, std::move(quant), "json_fallback:stage_config");
      }
      if (requires_static_quant_contract(stage) && stage.output_quant.empty()) {
        auto quant = extract_quant_specs_from_config(*stage_config, stage_params,
                                                     {"q_scale", "quant_scale", "dq_scale"},
                                                     {"q_zp", "zero_point", "dq_zp"}, tensor_hint);
        set_stage_output_quant(stage, std::move(quant), "json_fallback:stage_config");
      }
    }

    if (diagnostics && model_managed_stage) {
      auto require_runtime_default = [&](const char* key) {
        if (stage.runtime_defaults.contains(key)) {
          return;
        }
        diagnostics->errors.push_back("Missing required runtime field '" + std::string(key) +
                                      "' for model-managed stage '" + stage.element_name +
                                      "' (fallback_chain=property->json)");
      };

      if (is_boxdecode_plugin(stage.plugin_kind)) {
        require_runtime_default("decode_type");
        require_runtime_default("detection_threshold");
        require_runtime_default("nms_iou_threshold");
        require_runtime_default("topk");
      }

      if (requires_static_quant_contract(stage) && stage.output_quant.empty()) {
        diagnostics->errors.push_back(
            "Missing required quant contract for model-managed stage '" + stage.element_name +
            "' (fallback_chain=runtime-meta->context static quant->json)");
      }

      const bool requires_inputs = requires_static_quant_contract(stage);
      if (requires_inputs && stage.inputs.empty()) {
        diagnostics->errors.push_back(
            "Missing required input tensor contract for model-managed stage '" +
            stage.element_name + "' (fallback_chain=infer->context static fields)");
      }
    }

    build_sink_pad_tensor_index_map(stage, diagnostics);

    manifest.stages.push_back(std::move(stage));
  }

  build_manifest_v2_routes(manifest.stages, parsed_stage_configs);
  for (auto& stage : manifest.stages) {
    if (!stage.sink_routes.empty()) {
      append_trace(stage, "sink_routes", "infer:manifest_v2_route_compiler");
    }
    if (!stage.output_order.empty()) {
      append_trace(stage, "output_order", "infer:manifest_v2_route_compiler");
    }
  }

  return manifest;
}

} // namespace simaai::neat::pipeline_internal::sima
