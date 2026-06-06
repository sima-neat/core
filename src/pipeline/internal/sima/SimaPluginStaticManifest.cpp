#include "pipeline/internal/sima/SimaPluginStaticManifest.h"

#include "gst/SimaPluginStaticManifestAbi.h"
#include "pipeline/internal/TensorMath.h"
#include "pipeline/internal/EnvUtil.h"
#include "pipeline/internal/sima/BoxDecodeTypeUtils.h"
#include "pipeline/internal/sima/TensorSemanticsUtil.h"

#include <algorithm>
#include <atomic>
#include <cctype>
#include <cerrno>
#include <cstdio>
#include <cstring>
#include <cstdlib>
#include <memory>
#include <limits>
#include <sstream>
#include <unordered_map>

namespace simaai::neat::pipeline_internal::sima {
namespace {

std::string lower_copy(std::string s) {
  std::transform(s.begin(), s.end(), s.begin(),
                 [](unsigned char c) { return static_cast<char>(std::tolower(c)); });
  return s;
}

bool manifest_context_debug_enabled() {
  return pipeline_internal::env_bool("SIMA_MANIFEST_CONTEXT_DEBUG", false);
}

const char* element_factory_name(GstElement* element) {
  if (!element) {
    return "<null>";
  }
  GstElementFactory* factory = gst_element_get_factory(element);
  if (!factory) {
    return "<no-factory>";
  }
  const gchar* name = gst_plugin_feature_get_name(GST_PLUGIN_FEATURE(factory));
  return (name && *name) ? name : "<unnamed-factory>";
}

bool manifest_context_debug_relevant_element(GstElement* element) {
  if (!element) {
    return false;
  }
  const std::string factory = lower_copy(element_factory_name(element));
  return factory.find("neatprocessmla") != std::string::npos ||
         factory.find("neatprocesscvu") != std::string::npos ||
         factory.find("neatdetess") != std::string::npos ||
         factory.find("neatobjectdecode") != std::string::npos ||
         factory.find("neatboxdecode") != std::string::npos;
}

void manifest_context_debug_log(const char* action, GstElement* element, GstContext* context,
                                const char* extra = nullptr) {
  if (!manifest_context_debug_enabled() || !manifest_context_debug_relevant_element(element)) {
    return;
  }
  std::fprintf(stderr,
               "[manifest-context-debug] action=%s element=%s factory=%s context=%p extra=%s\n",
               action ? action : "<unset>", element ? GST_ELEMENT_NAME(element) : "<null>",
               element_factory_name(element), static_cast<void*>(context), extra ? extra : "");
}

std::string trim_copy_local(const std::string& s) {
  return trim_copy(s);
}

void set_boxdecode_decode_type_token_abi_safe(SimaPluginBoxDecodeStagePayload& payload,
                                              const gchar* decode_token) {
  payload.decode_type = decode_token;
  // Defensive ABI compatibility write:
  // Some mixed builds may still interpret the first boxdecode payload bytes as
  // a legacy enum slot; always stamp the first bytes as a pointer token.
  std::memcpy(&payload, &decode_token, sizeof(decode_token));
}

std::vector<std::string> split_pipeline_segments(const std::string& pipeline_string) {
  std::vector<std::string> out;
  if (pipeline_string.empty())
    return out;

  bool in_single_quote = false;
  bool in_double_quote = false;
  std::size_t start = 0;
  for (std::size_t i = 0; i < pipeline_string.size(); ++i) {
    const char c = pipeline_string[i];
    if (c == '\'' && !in_double_quote) {
      in_single_quote = !in_single_quote;
      continue;
    }
    if (c == '"' && !in_single_quote) {
      in_double_quote = !in_double_quote;
      continue;
    }
    if (c == '!' && !in_single_quote && !in_double_quote) {
      const std::string seg = trim_copy_local(pipeline_string.substr(start, i - start));
      if (!seg.empty()) {
        out.push_back(seg);
      }
      start = i + 1;
    }
  }
  const std::string tail = trim_copy_local(pipeline_string.substr(start));
  if (!tail.empty()) {
    out.push_back(tail);
  }
  return out;
}

std::string parse_first_token(const std::string& seg) {
  std::size_t i = 0;
  while (i < seg.size() && std::isspace(static_cast<unsigned char>(seg[i])))
    ++i;
  std::size_t j = i;
  while (j < seg.size() && !std::isspace(static_cast<unsigned char>(seg[j])))
    ++j;
  return (j > i) ? seg.substr(i, j - i) : std::string{};
}

std::optional<std::string> parse_property_value(const std::string& seg, const char* key) {
  if (!key || !*key)
    return std::nullopt;
  const std::string needle = std::string(key) + "=";
  std::size_t pos = seg.find(needle);
  if (pos == std::string::npos)
    return std::nullopt;
  std::size_t i = pos + needle.size();
  while (i < seg.size() && std::isspace(static_cast<unsigned char>(seg[i])))
    ++i;
  if (i >= seg.size())
    return std::nullopt;

  if (seg[i] == '"') {
    const std::size_t j = seg.find('"', i + 1);
    if (j == std::string::npos || j <= i + 1)
      return std::nullopt;
    return seg.substr(i + 1, j - i - 1);
  }

  std::size_t j = i;
  while (j < seg.size() && !std::isspace(static_cast<unsigned char>(seg[j])) && seg[j] != '!')
    ++j;
  if (j <= i)
    return std::nullopt;
  return seg.substr(i, j - i);
}

std::optional<double> parse_double_property(const std::string& seg, const char* key) {
  const auto raw = parse_property_value(seg, key);
  if (!raw.has_value() || raw->empty())
    return std::nullopt;
  char* end = nullptr;
  errno = 0;
  const double value = std::strtod(raw->c_str(), &end);
  if (errno != 0 || end == raw->c_str() || (end && *end != '\0'))
    return std::nullopt;
  return value;
}

std::optional<int> parse_int_property(const std::string& seg, const char* key) {
  const auto raw = parse_property_value(seg, key);
  if (!raw.has_value() || raw->empty())
    return std::nullopt;
  char* end = nullptr;
  errno = 0;
  const long value = std::strtol(raw->c_str(), &end, 10);
  if (errno != 0 || end == raw->c_str() || (end && *end != '\0'))
    return std::nullopt;
  if (value < std::numeric_limits<int>::min() || value > std::numeric_limits<int>::max())
    return std::nullopt;
  return static_cast<int>(value);
}

const char* quant_granularity_name(QuantGranularity granularity) {
  switch (granularity) {
  case QuantGranularity::PerTensor:
    return "per_tensor";
  case QuantGranularity::PerAxis:
    return "per_axis";
  }
  return "per_tensor";
}

std::optional<QuantGranularity> quant_granularity_from_json(const nlohmann::json& j) {
  if (!j.is_string())
    return std::nullopt;
  const std::string name = lower_copy(j.get<std::string>());
  if (name == "per_tensor")
    return QuantGranularity::PerTensor;
  if (name == "per_axis")
    return QuantGranularity::PerAxis;
  return std::nullopt;
}

template <typename T> void read_number_vector(const nlohmann::json& j, std::vector<T>& out) {
  out.clear();
  if (j.is_array()) {
    for (const auto& item : j) {
      if (item.is_number_integer()) {
        out.push_back(static_cast<T>(item.get<std::int64_t>()));
      } else if (item.is_number()) {
        out.push_back(static_cast<T>(item.get<double>()));
      }
    }
    return;
  }
  if (j.is_number_integer()) {
    out.push_back(static_cast<T>(j.get<std::int64_t>()));
  } else if (j.is_number()) {
    out.push_back(static_cast<T>(j.get<double>()));
  }
}

template <typename T>
void read_number_vector_key(const nlohmann::json& obj, const char* key, std::vector<T>& out) {
  if (!obj.contains(key)) {
    out.clear();
    return;
  }
  read_number_vector(obj.at(key), out);
}

bool read_string_key(const nlohmann::json& obj, const char* key, std::string& out) {
  out.clear();
  if (!obj.contains(key))
    return false;
  const auto& value = obj.at(key);
  if (value.is_string()) {
    out = value.get<std::string>();
    return true;
  }
  return false;
}

bool read_int_key(const nlohmann::json& obj, const char* key, int& out) {
  if (!obj.contains(key))
    return false;
  const auto& value = obj.at(key);
  if (value.is_number_integer()) {
    out = value.get<int>();
    return true;
  }
  if (value.is_number()) {
    out = static_cast<int>(value.get<double>());
    return true;
  }
  return false;
}

bool read_bool_key(const nlohmann::json& obj, const char* key, bool& out) {
  if (!obj.contains(key))
    return false;
  const auto& value = obj.at(key);
  if (value.is_boolean()) {
    out = value.get<bool>();
    return true;
  }
  if (value.is_number_integer()) {
    out = value.get<int>() != 0;
    return true;
  }
  return false;
}

} // namespace

std::vector<PipelineElementSpec> parse_pipeline_elements(const std::string& pipeline_string) {
  std::vector<PipelineElementSpec> out;
  const std::vector<std::string> segments = split_pipeline_segments(pipeline_string);
  out.reserve(segments.size());

  for (std::size_t i = 0; i < segments.size(); ++i) {
    const std::string& seg = segments[i];
    const std::string plugin = parse_first_token(seg);
    if (plugin.empty())
      continue;

    PipelineElementSpec spec;
    spec.element_index = i;
    spec.plugin = lower_copy(plugin);
    spec.fragment = seg;
    if (const auto name = parse_property_value(seg, "name"); name.has_value())
      spec.element_name = *name;
    if (const auto stage_id = parse_property_value(seg, "stage-id"); stage_id.has_value()) {
      spec.stage_id = *stage_id;
    } else if (const auto stage_id_snake = parse_property_value(seg, "stage_id");
               stage_id_snake.has_value()) {
      spec.stage_id = *stage_id_snake;
    }
    if (const auto cfg = parse_property_value(seg, "config"); cfg.has_value())
      spec.config_path = *cfg;
    if (const auto decode = parse_property_value(seg, "decode-type"); decode.has_value()) {
      if (const auto parsed = parse_box_decode_type_token(*decode); parsed.has_value()) {
        spec.decode_type_property = *parsed;
      }
    } else if (const auto decode_snake = parse_property_value(seg, "decode_type");
               decode_snake.has_value()) {
      if (const auto parsed = parse_box_decode_type_token(*decode_snake); parsed.has_value()) {
        spec.decode_type_property = *parsed;
      }
    }
    if (const auto det = parse_double_property(seg, "detection-threshold"); det.has_value()) {
      spec.detection_threshold_property = *det;
    } else if (const auto det_snake = parse_double_property(seg, "detection_threshold");
               det_snake.has_value()) {
      spec.detection_threshold_property = *det_snake;
    }
    if (const auto nms = parse_double_property(seg, "nms-iou-threshold"); nms.has_value()) {
      spec.nms_iou_threshold_property = *nms;
    } else if (const auto nms_snake = parse_double_property(seg, "nms_iou_threshold");
               nms_snake.has_value()) {
      spec.nms_iou_threshold_property = *nms_snake;
    }
    if (const auto topk = parse_int_property(seg, "topk"); topk.has_value())
      spec.topk_property = *topk;
    if (const auto model_path = parse_property_value(seg, "model-path"); model_path.has_value()) {
      spec.model_path_property = *model_path;
    } else if (const auto model_path_snake = parse_property_value(seg, "model_path");
               model_path_snake.has_value()) {
      spec.model_path_property = *model_path_snake;
    }
    if (const auto batch = parse_int_property(seg, "batch-size"); batch.has_value()) {
      spec.batch_size_property = *batch;
    } else if (const auto batch_snake = parse_int_property(seg, "batch_size");
               batch_snake.has_value()) {
      spec.batch_size_property = *batch_snake;
    }
    if (const auto batch_model = parse_int_property(seg, "batch-sz-model");
        batch_model.has_value()) {
      spec.batch_sz_model_property = *batch_model;
    } else if (const auto batch_model_snake = parse_int_property(seg, "batch_sz_model");
               batch_model_snake.has_value()) {
      spec.batch_sz_model_property = *batch_model_snake;
    }

    out.push_back(std::move(spec));
  }

  return out;
}

nlohmann::json to_json(const QuantStaticSpec& spec) {
  nlohmann::json j = nlohmann::json::object();
  j["granularity"] = quant_granularity_name(spec.granularity);
  j["axis"] = spec.axis;
  j["scales"] = spec.scales;
  j["zero_points"] = spec.zero_points;
  return j;
}

nlohmann::json to_json(const TensorStaticSpec& spec) {
  nlohmann::json j = nlohmann::json::object();
  j["tensor_index"] = spec.tensor_index;
  j["shape"] = spec.shape;
  j["dtype"] = spec.dtype;
  j["layout"] = spec.layout;
  j["max_w"] = spec.max_w;
  j["max_h"] = spec.max_h;
  j["max_stride"] = spec.max_stride;
  j["semantic_tag"] = spec.semantic_tag;
  return j;
}

nlohmann::json to_json(const ResolutionTrace& trace) {
  nlohmann::json j = nlohmann::json::object();
  j["field"] = trace.field;
  j["source_used"] = trace.source_used;
  j["fallback_chain"] = trace.fallback_chain;
  j["conflict"] = trace.conflict;
  return j;
}

nlohmann::json to_json(const StageOutputRoute& route) {
  nlohmann::json j = nlohmann::json::object();
  j["output_slot"] = route.output_slot;
  j["logical_output_index"] = route.logical_output_index;
  j["cm_output_name"] = route.cm_output_name;
  j["segment_name"] = route.segment_name;
  return j;
}

nlohmann::json to_json(const PhysicalBufferStaticSpec& spec) {
  nlohmann::json j = nlohmann::json::object();
  j["physical_index"] = spec.physical_index;
  j["allocator_index"] = spec.allocator_index;
  j["source_physical_index"] = spec.source_physical_index;
  j["size_bytes"] = spec.size_bytes;
  j["source_byte_offset"] = spec.source_byte_offset;
  j["device_kind"] = static_cast<int>(spec.device_kind);
  j["memory_flags"] = spec.memory_flags;
  j["segment_name_id"] = spec.segment_name_id;
  j["segment_name"] = spec.segment_name;
  return j;
}

nlohmann::json to_json(const LogicalTensorStaticSpec& spec) {
  nlohmann::json j = nlohmann::json::object();
  j["logical_index"] = spec.logical_index;
  j["backend_output_index"] = spec.backend_output_index;
  j["physical_index"] = spec.physical_index;
  j["output_slot"] = spec.output_slot;
  j["tensor_index"] = spec.tensor_index;
  j["byte_offset"] = spec.byte_offset;
  j["size_bytes"] = spec.size_bytes;
  j["shape"] = spec.shape;
  j["stride_bytes"] = spec.stride_bytes;
  j["dtype"] = spec.dtype;
  j["dtype_source"] = dtype_source_name(spec.dtype_source);
  j["layout"] = spec.layout;
  j["logical_name_id"] = spec.logical_name_id;
  j["logical_name"] = spec.logical_name;
  j["backend_name_id"] = spec.backend_name_id;
  j["backend_name"] = spec.backend_name;
  j["segment_name_id"] = spec.segment_name_id;
  j["segment_name"] = spec.segment_name;
  if (spec.quant.has_value()) {
    j["quant"] = to_json(*spec.quant);
  }
  return j;
}

nlohmann::json to_json(const LogicalInputStaticSpec& spec) {
  nlohmann::json j = nlohmann::json::object();
  j["logical_index"] = spec.logical_index;
  j["backend_input_index"] = spec.backend_input_index;
  j["physical_index"] = spec.physical_index;
  j["shape"] = spec.shape;
  j["stride_bytes"] = spec.stride_bytes;
  j["byte_offset"] = spec.byte_offset;
  j["size_bytes"] = spec.size_bytes;
  j["dtype"] = spec.dtype;
  j["dtype_source"] = dtype_source_name(spec.dtype_source);
  j["layout"] = spec.layout;
  j["logical_name_id"] = spec.logical_name_id;
  j["logical_name"] = spec.logical_name;
  j["backend_name_id"] = spec.backend_name_id;
  j["backend_name"] = spec.backend_name;
  j["segment_name_id"] = spec.segment_name_id;
  j["segment_name"] = spec.segment_name;
  j["materialization_kind"] = static_cast<int>(spec.materialization_kind);
  if (spec.quant.has_value()) {
    j["quant"] = to_json(*spec.quant);
  }
  return j;
}

nlohmann::json to_json(const InputBindingStaticSpec& spec) {
  nlohmann::json j = nlohmann::json::object();
  j["sink_pad_index"] = spec.sink_pad_index;
  j["local_logical_input_index"] = spec.local_logical_input_index;
  j["src_stage_index"] = spec.src_stage_index;
  j["src_stage_id"] = spec.src_stage_id;
  j["src_logical_output_index"] = spec.src_logical_output_index;
  j["src_output_slot"] = spec.src_output_slot;
  j["src_physical_output_index"] = spec.src_physical_output_index;
  j["required"] = spec.required;
  j["cm_input_name_id"] = spec.cm_input_name_id;
  j["cm_input_name"] = spec.cm_input_name;
  j["source_segment_name_id"] = spec.source_segment_name_id;
  j["source_segment_name"] = spec.source_segment_name;
  return j;
}

nlohmann::json to_json(const StageStaticSpec& spec) {
  nlohmann::json j = nlohmann::json::object();
  j["element_name"] = spec.element_name;
  j["logical_stage_id"] = spec.logical_stage_id;
  j["plugin_kind"] = spec.plugin_kind;
  j["kernel_kind"] = spec.kernel_kind;
  j["name_table"] = spec.name_table;
  if (spec.payload_kind == StagePayloadKind::ProcessCvu &&
      !spec.processcvu.exact_stage_name_or_id.empty()) {
    j["processcvu_exact_stage_name_or_id"] = spec.processcvu.exact_stage_name_or_id;
  }

  nlohmann::json logical_inputs = nlohmann::json::array();
  for (const auto& input : spec.logical_inputs)
    logical_inputs.push_back(to_json(input));
  j["logical_inputs"] = std::move(logical_inputs);

  nlohmann::json input_bindings = nlohmann::json::array();
  for (const auto& binding : spec.input_bindings)
    input_bindings.push_back(to_json(binding));
  j["input_bindings"] = std::move(input_bindings);

  nlohmann::json physical_inputs = nlohmann::json::array();
  for (const auto& physical : spec.physical_inputs)
    physical_inputs.push_back(to_json(physical));
  j["physical_inputs"] = std::move(physical_inputs);

  nlohmann::json physical_outputs = nlohmann::json::array();
  for (const auto& physical : spec.physical_outputs)
    physical_outputs.push_back(to_json(physical));
  j["physical_outputs"] = std::move(physical_outputs);

  nlohmann::json logical_outputs = nlohmann::json::array();
  for (const auto& logical : spec.logical_outputs)
    logical_outputs.push_back(to_json(logical));
  j["logical_outputs"] = std::move(logical_outputs);

  nlohmann::json output_routes = nlohmann::json::array();
  for (const auto& route : spec.output_order)
    output_routes.push_back(to_json(route));
  j["output_order"] = std::move(output_routes);

  nlohmann::json q = nlohmann::json::array();
  for (const auto& quant : spec.output_quant)
    q.push_back(to_json(quant));
  j["output_quant"] = std::move(q);

  nlohmann::json trace = nlohmann::json::array();
  for (const auto& t : spec.resolution_trace)
    trace.push_back(to_json(t));
  j["resolution_trace"] = std::move(trace);

  if (spec.consumer_keeps_distinct_physical_inputs) {
    j["consumer_keeps_distinct_physical_inputs"] = true;
  }
  if (!spec.elf_ifm_symbol_names.empty()) {
    j["elf_ifm_symbol_names"] = spec.elf_ifm_symbol_names;
  }
  if (!spec.elf_ofm_symbol_names.empty()) {
    j["elf_ofm_symbol_names"] = spec.elf_ofm_symbol_names;
  }

  if (spec.payload_kind == StagePayloadKind::ProcessCvu && !spec.processcvu.run_target.empty()) {
    j["processcvu_run_target"] = spec.processcvu.run_target;
  }
  if (spec.payload_kind == StagePayloadKind::ProcessCvu &&
      !spec.processcvu.requested_run_target.empty()) {
    j["processcvu_requested_run_target"] = spec.processcvu.requested_run_target;
  }
  if (spec.payload_kind == StagePayloadKind::ProcessCvu &&
      !spec.processcvu.resolved_exec_backend.empty()) {
    j["processcvu_resolved_exec_backend"] = spec.processcvu.resolved_exec_backend;
  }
  if (spec.payload_kind == StagePayloadKind::ProcessCvu &&
      !spec.processcvu.run_target_resolution_reason.empty()) {
    j["processcvu_run_target_resolution_reason"] = spec.processcvu.run_target_resolution_reason;
  }
  if (spec.payload_kind == StagePayloadKind::ProcessCvu && spec.processcvu.opt_flags != 0U) {
    j["processcvu_opt_flags"] = spec.processcvu.opt_flags;
  }

  return j;
}

nlohmann::json to_json(const SimaPluginStaticManifest& manifest) {
  nlohmann::json j = nlohmann::json::object();
  j["session_id"] = manifest.session_id;
  j["model_id"] = manifest.model_id;
  j["stages"] = nlohmann::json::array();
  for (const auto& stage : manifest.stages) {
    j["stages"].push_back(to_json(stage));
  }
  return j;
}

std::string serialize_manifest_json(const SimaPluginStaticManifest& manifest) {
  return to_json(manifest).dump();
}

std::optional<SimaPluginStaticManifest> parse_manifest_json(const std::string& manifest_json,
                                                            std::string* error_message) {
  if (error_message)
    error_message->clear();

  nlohmann::json root;
  try {
    root = nlohmann::json::parse(manifest_json);
  } catch (const std::exception& e) {
    if (error_message) {
      *error_message = std::string("manifest parse error: ") + e.what();
    }
    return std::nullopt;
  }

  if (!root.is_object()) {
    if (error_message)
      *error_message = "manifest root must be a JSON object";
    return std::nullopt;
  }

  SimaPluginStaticManifest manifest;
  read_string_key(root, "session_id", manifest.session_id);
  read_string_key(root, "model_id", manifest.model_id);

  if (!root.contains("stages") || !root["stages"].is_array()) {
    if (error_message)
      *error_message = "manifest missing stages[]";
    return std::nullopt;
  }

  for (const auto& stage_j : root["stages"]) {
    if (!stage_j.is_object())
      continue;
    StageStaticSpec stage;
    read_string_key(stage_j, "element_name", stage.element_name);
    read_string_key(stage_j, "logical_stage_id", stage.logical_stage_id);
    read_string_key(stage_j, "plugin_kind", stage.plugin_kind);
    read_string_key(stage_j, "kernel_kind", stage.kernel_kind);
    if (stage_j.contains("name_table") && stage_j["name_table"].is_array()) {
      for (const auto& name_j : stage_j["name_table"]) {
        if (name_j.is_string()) {
          stage.name_table.push_back(name_j.get<std::string>());
        }
      }
    }

    if (stage_j.contains("logical_inputs") && stage_j["logical_inputs"].is_array()) {
      for (const auto& in_j : stage_j["logical_inputs"]) {
        if (!in_j.is_object())
          continue;
        LogicalInputStaticSpec input;
        read_int_key(in_j, "logical_index", input.logical_index);
        read_int_key(in_j, "backend_input_index", input.backend_input_index);
        read_int_key(in_j, "physical_index", input.physical_index);
        read_number_vector_key<std::int64_t>(in_j, "shape", input.shape);
        read_number_vector_key<std::int64_t>(in_j, "stride_bytes", input.stride_bytes);
        if (in_j.contains("byte_offset") && in_j["byte_offset"].is_number_integer()) {
          input.byte_offset = in_j["byte_offset"].get<std::int64_t>();
        } else if (in_j.contains("byte_offset") && in_j["byte_offset"].is_number()) {
          input.byte_offset = static_cast<std::int64_t>(in_j["byte_offset"].get<double>());
        }
        if (in_j.contains("size_bytes") && in_j["size_bytes"].is_number_unsigned()) {
          input.size_bytes = in_j["size_bytes"].get<std::uint64_t>();
        } else if (in_j.contains("size_bytes") && in_j["size_bytes"].is_number_integer()) {
          const auto raw = in_j["size_bytes"].get<std::int64_t>();
          input.size_bytes = raw > 0 ? static_cast<std::uint64_t>(raw) : 0U;
        } else if (in_j.contains("size_bytes") && in_j["size_bytes"].is_number()) {
          const auto raw = in_j["size_bytes"].get<double>();
          input.size_bytes = raw > 0.0 ? static_cast<std::uint64_t>(raw) : 0U;
        }
        read_string_key(in_j, "dtype", input.dtype);
        if (in_j.contains("dtype_source") && in_j["dtype_source"].is_string()) {
          input.dtype_source = dtype_source_from_name(in_j["dtype_source"].get<std::string>());
        }
        read_string_key(in_j, "layout", input.layout);
        read_int_key(in_j, "logical_name_id", input.logical_name_id);
        read_string_key(in_j, "logical_name", input.logical_name);
        read_int_key(in_j, "backend_name_id", input.backend_name_id);
        read_string_key(in_j, "backend_name", input.backend_name);
        read_int_key(in_j, "segment_name_id", input.segment_name_id);
        read_string_key(in_j, "segment_name", input.segment_name);
        {
          int materialization_kind = static_cast<int>(TensorMaterializationKind::Direct);
          read_int_key(in_j, "materialization_kind", materialization_kind);
          input.materialization_kind = static_cast<TensorMaterializationKind>(materialization_kind);
        }
        if (in_j.contains("quant") && in_j["quant"].is_object()) {
          QuantStaticSpec quant;
          const auto& quant_j = in_j["quant"];
          if (quant_j.contains("granularity")) {
            if (const auto g = quant_granularity_from_json(quant_j["granularity"]); g.has_value()) {
              quant.granularity = *g;
            }
          }
          read_int_key(quant_j, "axis", quant.axis);
          read_number_vector_key<double>(quant_j, "scales", quant.scales);
          read_number_vector_key<std::int64_t>(quant_j, "zero_points", quant.zero_points);
          input.quant = std::move(quant);
        }
        stage.logical_inputs.push_back(std::move(input));
      }
    }

    if (stage_j.contains("input_bindings") && stage_j["input_bindings"].is_array()) {
      for (const auto& in_j : stage_j["input_bindings"]) {
        if (!in_j.is_object())
          continue;
        InputBindingStaticSpec binding;
        read_int_key(in_j, "sink_pad_index", binding.sink_pad_index);
        read_int_key(in_j, "local_logical_input_index", binding.local_logical_input_index);
        read_int_key(in_j, "src_stage_index", binding.src_stage_index);
        read_string_key(in_j, "src_stage_id", binding.src_stage_id);
        read_int_key(in_j, "src_logical_output_index", binding.src_logical_output_index);
        read_int_key(in_j, "src_output_slot", binding.src_output_slot);
        read_int_key(in_j, "src_physical_output_index", binding.src_physical_output_index);
        read_bool_key(in_j, "required", binding.required);
        read_int_key(in_j, "cm_input_name_id", binding.cm_input_name_id);
        read_string_key(in_j, "cm_input_name", binding.cm_input_name);
        read_int_key(in_j, "source_segment_name_id", binding.source_segment_name_id);
        read_string_key(in_j, "source_segment_name", binding.source_segment_name);
        stage.input_bindings.push_back(std::move(binding));
      }
    }

    if (stage_j.contains("physical_inputs") && stage_j["physical_inputs"].is_array()) {
      for (const auto& in_j : stage_j["physical_inputs"]) {
        if (!in_j.is_object())
          continue;
        PhysicalBufferStaticSpec physical;
        read_int_key(in_j, "physical_index", physical.physical_index);
        read_int_key(in_j, "allocator_index", physical.allocator_index);
        read_int_key(in_j, "source_physical_index", physical.source_physical_index);
        if (in_j.contains("size_bytes") && in_j["size_bytes"].is_number_unsigned()) {
          physical.size_bytes = in_j["size_bytes"].get<std::uint64_t>();
        } else if (in_j.contains("size_bytes") && in_j["size_bytes"].is_number_integer()) {
          physical.size_bytes = static_cast<std::uint64_t>(in_j["size_bytes"].get<std::int64_t>());
        }
        if (in_j.contains("source_byte_offset") && in_j["source_byte_offset"].is_number_integer()) {
          physical.source_byte_offset = in_j["source_byte_offset"].get<std::int64_t>();
        } else if (in_j.contains("source_byte_offset") && in_j["source_byte_offset"].is_number()) {
          physical.source_byte_offset =
              static_cast<std::int64_t>(in_j["source_byte_offset"].get<double>());
        }
        int device_kind = static_cast<int>(DeviceKind::Unknown);
        read_int_key(in_j, "device_kind", device_kind);
        physical.device_kind = static_cast<DeviceKind>(device_kind);
        if (in_j.contains("memory_flags") && in_j["memory_flags"].is_number_unsigned()) {
          physical.memory_flags = in_j["memory_flags"].get<std::uint64_t>();
        } else if (in_j.contains("memory_flags") && in_j["memory_flags"].is_number_integer()) {
          physical.memory_flags =
              static_cast<std::uint64_t>(in_j["memory_flags"].get<std::int64_t>());
        }
        read_int_key(in_j, "segment_name_id", physical.segment_name_id);
        read_string_key(in_j, "segment_name", physical.segment_name);
        stage.physical_inputs.push_back(std::move(physical));
      }
    }

    if (stage_j.contains("physical_outputs") && stage_j["physical_outputs"].is_array()) {
      for (const auto& out_j : stage_j["physical_outputs"]) {
        if (!out_j.is_object())
          continue;
        PhysicalBufferStaticSpec physical;
        read_int_key(out_j, "physical_index", physical.physical_index);
        read_int_key(out_j, "allocator_index", physical.allocator_index);
        read_int_key(out_j, "source_physical_index", physical.source_physical_index);
        if (out_j.contains("size_bytes") && out_j["size_bytes"].is_number_unsigned()) {
          physical.size_bytes = out_j["size_bytes"].get<std::uint64_t>();
        } else if (out_j.contains("size_bytes") && out_j["size_bytes"].is_number_integer()) {
          physical.size_bytes = static_cast<std::uint64_t>(out_j["size_bytes"].get<std::int64_t>());
        }
        if (out_j.contains("source_byte_offset") &&
            out_j["source_byte_offset"].is_number_integer()) {
          physical.source_byte_offset = out_j["source_byte_offset"].get<std::int64_t>();
        } else if (out_j.contains("source_byte_offset") &&
                   out_j["source_byte_offset"].is_number()) {
          physical.source_byte_offset =
              static_cast<std::int64_t>(out_j["source_byte_offset"].get<double>());
        }
        int device_kind = static_cast<int>(DeviceKind::Unknown);
        read_int_key(out_j, "device_kind", device_kind);
        physical.device_kind = static_cast<DeviceKind>(device_kind);
        if (out_j.contains("memory_flags") && out_j["memory_flags"].is_number_unsigned()) {
          physical.memory_flags = out_j["memory_flags"].get<std::uint64_t>();
        } else if (out_j.contains("memory_flags") && out_j["memory_flags"].is_number_integer()) {
          physical.memory_flags =
              static_cast<std::uint64_t>(out_j["memory_flags"].get<std::int64_t>());
        }
        read_int_key(out_j, "segment_name_id", physical.segment_name_id);
        read_string_key(out_j, "segment_name", physical.segment_name);
        stage.physical_outputs.push_back(std::move(physical));
      }
    }

    if (stage_j.contains("logical_outputs") && stage_j["logical_outputs"].is_array()) {
      for (const auto& out_j : stage_j["logical_outputs"]) {
        if (!out_j.is_object())
          continue;
        LogicalTensorStaticSpec tensor;
        read_int_key(out_j, "logical_index", tensor.logical_index);
        read_int_key(out_j, "backend_output_index", tensor.backend_output_index);
        read_int_key(out_j, "physical_index", tensor.physical_index);
        read_int_key(out_j, "output_slot", tensor.output_slot);
        read_int_key(out_j, "tensor_index", tensor.tensor_index);
        if (out_j.contains("byte_offset") && out_j["byte_offset"].is_number_integer()) {
          tensor.byte_offset = out_j["byte_offset"].get<std::int64_t>();
        }
        if (out_j.contains("size_bytes") && out_j["size_bytes"].is_number_unsigned()) {
          tensor.size_bytes = out_j["size_bytes"].get<std::uint64_t>();
        } else if (out_j.contains("size_bytes") && out_j["size_bytes"].is_number_integer()) {
          tensor.size_bytes = static_cast<std::uint64_t>(out_j["size_bytes"].get<std::int64_t>());
        }
        read_number_vector_key<std::int64_t>(out_j, "shape", tensor.shape);
        read_number_vector_key<std::int64_t>(out_j, "stride_bytes", tensor.stride_bytes);
        read_string_key(out_j, "dtype", tensor.dtype);
        if (out_j.contains("dtype_source") && out_j["dtype_source"].is_string()) {
          tensor.dtype_source = dtype_source_from_name(out_j["dtype_source"].get<std::string>());
        }
        read_string_key(out_j, "layout", tensor.layout);
        read_int_key(out_j, "logical_name_id", tensor.logical_name_id);
        read_string_key(out_j, "logical_name", tensor.logical_name);
        read_int_key(out_j, "backend_name_id", tensor.backend_name_id);
        read_string_key(out_j, "backend_name", tensor.backend_name);
        read_int_key(out_j, "segment_name_id", tensor.segment_name_id);
        read_string_key(out_j, "segment_name", tensor.segment_name);
        if (out_j.contains("quant") && out_j["quant"].is_object()) {
          QuantStaticSpec quant;
          const auto& quant_j = out_j["quant"];
          if (quant_j.contains("granularity")) {
            if (const auto g = quant_granularity_from_json(quant_j["granularity"]); g.has_value()) {
              quant.granularity = *g;
            }
          }
          read_int_key(quant_j, "axis", quant.axis);
          read_number_vector_key<double>(quant_j, "scales", quant.scales);
          read_number_vector_key<std::int64_t>(quant_j, "zero_points", quant.zero_points);
          tensor.quant = std::move(quant);
        }
        stage.logical_outputs.push_back(std::move(tensor));
      }
    }

    if (stage_j.contains("output_order") && stage_j["output_order"].is_array()) {
      for (const auto& route_j : stage_j["output_order"]) {
        if (!route_j.is_object())
          continue;
        StageOutputRoute route;
        read_int_key(route_j, "output_slot", route.output_slot);
        read_int_key(route_j, "logical_output_index", route.logical_output_index);
        read_string_key(route_j, "cm_output_name", route.cm_output_name);
        read_string_key(route_j, "segment_name", route.segment_name);
        stage.output_order.push_back(std::move(route));
      }
    }

    if (stage_j.contains("output_quant") && stage_j["output_quant"].is_array()) {
      for (const auto& q_j : stage_j["output_quant"]) {
        if (!q_j.is_object())
          continue;
        QuantStaticSpec quant;
        if (q_j.contains("granularity")) {
          if (const auto g = quant_granularity_from_json(q_j["granularity"]); g.has_value()) {
            quant.granularity = *g;
          }
        }
        read_int_key(q_j, "axis", quant.axis);
        read_number_vector_key<double>(q_j, "scales", quant.scales);
        read_number_vector_key<std::int64_t>(q_j, "zero_points", quant.zero_points);
        stage.output_quant.push_back(std::move(quant));
      }
    }

    if (stage_j.contains("resolution_trace") && stage_j["resolution_trace"].is_array()) {
      for (const auto& t_j : stage_j["resolution_trace"]) {
        if (!t_j.is_object())
          continue;
        ResolutionTrace trace;
        read_string_key(t_j, "field", trace.field);
        read_string_key(t_j, "source_used", trace.source_used);
        if (t_j.contains("fallback_chain") && t_j["fallback_chain"].is_array()) {
          for (const auto& f : t_j["fallback_chain"]) {
            if (f.is_string()) {
              trace.fallback_chain.push_back(f.get<std::string>());
            }
          }
        }
        read_bool_key(t_j, "conflict", trace.conflict);
        stage.resolution_trace.push_back(std::move(trace));
      }
    }

    read_bool_key(stage_j, "consumer_keeps_distinct_physical_inputs",
                  stage.consumer_keeps_distinct_physical_inputs);
    if (stage_j.contains("elf_ifm_symbol_names") && stage_j["elf_ifm_symbol_names"].is_array()) {
      stage.elf_ifm_symbol_names.clear();
      for (const auto& s : stage_j["elf_ifm_symbol_names"]) {
        if (s.is_string()) {
          stage.elf_ifm_symbol_names.push_back(s.get<std::string>());
        }
      }
    }
    if (stage_j.contains("elf_ofm_symbol_names") && stage_j["elf_ofm_symbol_names"].is_array()) {
      stage.elf_ofm_symbol_names.clear();
      for (const auto& s : stage_j["elf_ofm_symbol_names"]) {
        if (s.is_string()) {
          stage.elf_ofm_symbol_names.push_back(s.get<std::string>());
        }
      }
    }

    if (stage.payload_kind == StagePayloadKind::ProcessCvu) {
      read_string_key(stage_j, "processcvu_run_target", stage.processcvu.run_target);
      read_string_key(stage_j, "processcvu_requested_run_target",
                      stage.processcvu.requested_run_target);
      read_string_key(stage_j, "processcvu_resolved_exec_backend",
                      stage.processcvu.resolved_exec_backend);
      read_string_key(stage_j, "processcvu_run_target_resolution_reason",
                      stage.processcvu.run_target_resolution_reason);
      read_string_key(stage_j, "processcvu_exact_stage_name_or_id",
                      stage.processcvu.exact_stage_name_or_id);
      int processcvu_opt_flags = 0;
      if (read_int_key(stage_j, "processcvu_opt_flags", processcvu_opt_flags) &&
          processcvu_opt_flags >= 0) {
        stage.processcvu.opt_flags = static_cast<std::uint32_t>(processcvu_opt_flags);
      }
    }

    manifest.stages.push_back(std::move(stage));
  }

  return manifest;
}

namespace {

bool manifest_stage_debug_enabled() {
  return pipeline_internal::env_bool("SIMA_STAGE_DEBUG", false);
}

class ManifestAccessorRegistry {
public:
  explicit ManifestAccessorRegistry(SimaPluginStaticManifest manifest)
      : manifest_(std::move(manifest)), session_id_(manifest_.session_id),
        model_id_(manifest_.model_id) {
    accessor_.abi_version = SIMA_PLUGIN_STATIC_MANIFEST_ABI_VERSION;
    accessor_.user_data = this;
    accessor_.session_id = &ManifestAccessorRegistry::session_id_cb;
    accessor_.model_id = &ManifestAccessorRegistry::model_id_cb;
    accessor_.stage_count = &ManifestAccessorRegistry::stage_count_cb;
    accessor_.stage_by_index = &ManifestAccessorRegistry::stage_by_index_cb;
    accessor_.stage_by_element_name = &ManifestAccessorRegistry::stage_by_element_cb;
    accessor_.stage_by_logical_stage_id = &ManifestAccessorRegistry::stage_by_id_cb;

    stage_storage_.reserve(manifest_.stages.size());
    stage_by_element_.reserve(manifest_.stages.size());
    stage_by_id_.reserve(manifest_.stages.size());
    for (std::size_t i = 0; i < manifest_.stages.size(); ++i) {
      const auto& stage = manifest_.stages[i];
      stage_storage_.push_back(build_stage_storage(stage));
      if (!stage.element_name.empty()) {
        stage_by_element_.emplace(stage.element_name, i);
      }
      if (!stage.logical_stage_id.empty()) {
        stage_by_id_.emplace(stage.logical_stage_id, i);
      }
    }
  }

  const SimaPluginStaticManifestAccessor* accessor() const {
    return &accessor_;
  }

  const SimaPluginStageSpec* stage_by_element_name(const gchar* element_name) const {
    return stage_by_element_cb(const_cast<ManifestAccessorRegistry*>(this), element_name);
  }

  const SimaPluginStageSpec* stage_by_logical_stage_id(const gchar* logical_stage_id) const {
    return stage_by_id_cb(const_cast<ManifestAccessorRegistry*>(this), logical_stage_id);
  }

private:
  struct StageAbiStorage {
    std::vector<const gchar*> name_table;
    std::vector<std::vector<gint64>> logical_input_shapes;
    std::vector<std::vector<gint64>> logical_input_stride_bytes;
    std::vector<std::vector<guint8>> logical_input_axis_semantics;
    std::vector<std::vector<gdouble>> logical_input_quant_scales;
    std::vector<std::vector<gint64>> logical_input_quant_zps;
    std::vector<SimaPluginQuantSpec> logical_input_quant_specs;
    std::vector<SimaPluginLogicalInput> logical_inputs;
    std::vector<SimaPluginInputBinding> input_bindings;
    std::vector<SimaPluginPhysicalBuffer> physical_inputs;
    std::vector<SimaPluginPhysicalBuffer> physical_outputs;
    std::vector<const gchar*> processmla_dispatcher_output_names;
    std::vector<guint64> processmla_dispatcher_output_sizes;
    std::vector<std::vector<gint64>> logical_output_shapes;
    std::vector<std::vector<gint64>> logical_output_stride_bytes;
    std::vector<std::vector<guint8>> logical_output_axis_semantics;
    std::vector<std::vector<gdouble>> logical_output_quant_scales;
    std::vector<std::vector<gint64>> logical_output_quant_zps;
    std::vector<SimaPluginQuantSpec> logical_output_quant_specs;
    std::vector<SimaPluginLogicalTensor> logical_outputs;
    std::vector<SimaPluginOutputRoute> output_order;
    std::vector<std::string> output_route_cm_output_names;
    std::vector<std::string> output_route_segment_names;
    std::vector<std::vector<gdouble>> quant_scales;
    std::vector<std::vector<gint64>> quant_zps;
    std::vector<SimaPluginQuantSpec> output_quant;
    std::vector<const gchar*> required_meta_fields;
    std::vector<const gchar*> processcvu_default_output_names;
    std::vector<std::string> processcvu_default_output_name_storage;
    std::vector<gint> processcvu_runtime_output_logical_index_array;
    std::vector<gint> processcvu_runtime_output_output_slot_array;
    std::vector<gint> processcvu_runtime_output_physical_index_array;
    std::vector<const gchar*> processcvu_runtime_output_dtype_array;
    std::vector<gint> processcvu_runtime_output_transport_kind_array;
    std::vector<gint> processcvu_runtime_output_semantic_kind_array;
    std::vector<gdouble> processcvu_q_scale_array;
    std::vector<gint> processcvu_q_zp_array;
    std::vector<sima_ev_tensor_desc> processcvu_input_tensors;
    std::vector<sima_ev_tensor_desc> processcvu_output_tensors;
    std::vector<gdouble> processcvu_dq_scale_array;
    std::vector<gint> processcvu_dq_zp_array;
    std::vector<gdouble> processcvu_channel_mean;
    std::vector<gdouble> processcvu_channel_stddev;
    std::vector<sima_ev_shape_desc> boxdecode_slice_shapes;
    SimaPluginStageSpec spec{};
  };

  template <typename ShapeT>
  static std::vector<guint8> derive_axis_semantics_from_layout(const std::vector<ShapeT>& shape,
                                                               const std::string& layout) {
    const std::string normalized = tensorsemantics::normalize_layout_token(layout);
    if (normalized.empty()) {
      return {};
    }
    std::vector<guint8> semantics(SIMA_EV_MAX_RANK, SIMA_EV_AXIS_UNKNOWN);
    tensorsemantics::fill_axis_semantics_from_shape_layout(shape, normalized, semantics.data());
    semantics.resize(shape.size());
    return semantics;
  }

  static SimaPluginDeviceKind device_kind_to_abi(DeviceKind kind) {
    switch (kind) {
    case DeviceKind::Cpu:
      return SIMA_PLUGIN_DEVICE_KIND_CPU;
    case DeviceKind::Mla:
      return SIMA_PLUGIN_DEVICE_KIND_MLA;
    case DeviceKind::Evxx:
      return SIMA_PLUGIN_DEVICE_KIND_EVXX;
    case DeviceKind::Unknown:
    default:
      return SIMA_PLUGIN_DEVICE_KIND_UNKNOWN;
    }
  }

  static SimaPluginStagePayloadKind payload_kind_to_abi(StagePayloadKind kind) {
    switch (kind) {
    case StagePayloadKind::ProcessCvu:
      return SIMA_PLUGIN_STAGE_PAYLOAD_PROCESSCVU;
    case StagePayloadKind::ProcessMla:
      return SIMA_PLUGIN_STAGE_PAYLOAD_PROCESSMLA;
    case StagePayloadKind::BoxDecode:
      return SIMA_PLUGIN_STAGE_PAYLOAD_BOXDECODE;
    case StagePayloadKind::DetessDequant:
      return SIMA_PLUGIN_STAGE_PAYLOAD_DETESSDEQUANT;
    case StagePayloadKind::Quant:
      return SIMA_PLUGIN_STAGE_PAYLOAD_QUANT;
    case StagePayloadKind::Tess:
      return SIMA_PLUGIN_STAGE_PAYLOAD_TESS;
    case StagePayloadKind::Dequant:
      return SIMA_PLUGIN_STAGE_PAYLOAD_DEQUANT;
    case StagePayloadKind::QuantTess:
      return SIMA_PLUGIN_STAGE_PAYLOAD_QUANTTESS;
    case StagePayloadKind::None:
    default:
      return SIMA_PLUGIN_STAGE_PAYLOAD_NONE;
    }
  }

  static SimaPluginProcessCvuGraphFamily processcvu_family_to_abi(ProcessCvuGraphFamily family) {
    switch (family) {
    case ProcessCvuGraphFamily::Preproc:
      return SIMA_PLUGIN_PROCESSCVU_GRAPH_FAMILY_PREPROC;
    case ProcessCvuGraphFamily::Quant:
      return SIMA_PLUGIN_PROCESSCVU_GRAPH_FAMILY_QUANT;
    case ProcessCvuGraphFamily::Tess:
      return SIMA_PLUGIN_PROCESSCVU_GRAPH_FAMILY_TESS;
    case ProcessCvuGraphFamily::QuantTess:
      return SIMA_PLUGIN_PROCESSCVU_GRAPH_FAMILY_QUANTTESS;
    case ProcessCvuGraphFamily::CastTess:
      return SIMA_PLUGIN_PROCESSCVU_GRAPH_FAMILY_CASTTESS;
    case ProcessCvuGraphFamily::Detess:
      return SIMA_PLUGIN_PROCESSCVU_GRAPH_FAMILY_DETESS;
    case ProcessCvuGraphFamily::DetessCast:
      return SIMA_PLUGIN_PROCESSCVU_GRAPH_FAMILY_DETESSCAST;
    case ProcessCvuGraphFamily::Dequant:
      return SIMA_PLUGIN_PROCESSCVU_GRAPH_FAMILY_DEQUANT;
    case ProcessCvuGraphFamily::DetessDequant:
      return SIMA_PLUGIN_PROCESSCVU_GRAPH_FAMILY_DETESSDEQUANT;
    case ProcessCvuGraphFamily::Cast:
      return SIMA_PLUGIN_PROCESSCVU_GRAPH_FAMILY_CAST;
    case ProcessCvuGraphFamily::VisualFrontend:
      return SIMA_PLUGIN_PROCESSCVU_GRAPH_FAMILY_VISUAL_FRONTEND;
    case ProcessCvuGraphFamily::Unknown:
    default:
      return SIMA_PLUGIN_PROCESSCVU_GRAPH_FAMILY_UNKNOWN;
    }
  }

  static SimaPluginProcessCvuOutputTransportKind
  processcvu_output_transport_to_abi(ProcessCvuOutputTransportKind kind) {
    switch (kind) {
    case ProcessCvuOutputTransportKind::Dense:
      return SIMA_PLUGIN_PROCESSCVU_OUTPUT_TRANSPORT_DENSE;
    case ProcessCvuOutputTransportKind::Packed:
      return SIMA_PLUGIN_PROCESSCVU_OUTPUT_TRANSPORT_PACKED;
    case ProcessCvuOutputTransportKind::Unknown:
    default:
      return SIMA_PLUGIN_PROCESSCVU_OUTPUT_TRANSPORT_UNKNOWN;
    }
  }

  static SimaPluginProcessCvuOutputSemanticKind
  processcvu_output_semantic_to_abi(ProcessCvuOutputSemanticKind kind) {
    switch (kind) {
    case ProcessCvuOutputSemanticKind::Image:
      return SIMA_PLUGIN_PROCESSCVU_OUTPUT_SEMANTIC_IMAGE;
    case ProcessCvuOutputSemanticKind::TessellatedImage:
      return SIMA_PLUGIN_PROCESSCVU_OUTPUT_SEMANTIC_TESSELLATED_IMAGE;
    case ProcessCvuOutputSemanticKind::QuantizedTensor:
      return SIMA_PLUGIN_PROCESSCVU_OUTPUT_SEMANTIC_QUANTIZED_TENSOR;
    case ProcessCvuOutputSemanticKind::QuantTessTensor:
      return SIMA_PLUGIN_PROCESSCVU_OUTPUT_SEMANTIC_QUANTTESS_TENSOR;
    case ProcessCvuOutputSemanticKind::Tensor:
      return SIMA_PLUGIN_PROCESSCVU_OUTPUT_SEMANTIC_TENSOR;
    case ProcessCvuOutputSemanticKind::Unknown:
    default:
      return SIMA_PLUGIN_PROCESSCVU_OUTPUT_SEMANTIC_UNKNOWN;
    }
  }

  static StageAbiStorage build_stage_storage(const StageStaticSpec& stage) {
    StageAbiStorage out;

    if (pipeline_internal::env_bool("SIMA_RENDER_STAGE_DEBUG", false) &&
        stage.logical_stage_id.find("post_dequant") != std::string::npos) {
      std::fprintf(
          stderr,
          "[stage-storage-debug] stage=%s id=%s payload_kind=%d logical_inputs=%zu "
          "logical_outputs=%zu output_order=%zu physical_inputs=%zu physical_outputs=%zu\n",
          stage.element_name.c_str(), stage.logical_stage_id.c_str(),
          static_cast<int>(stage.payload_kind), stage.logical_inputs.size(),
          stage.logical_outputs.size(), stage.output_order.size(), stage.physical_inputs.size(),
          stage.physical_outputs.size());
    }
    if (pipeline_internal::env_bool("SIMA_MLA_STAGE_STORAGE_DEBUG", false) &&
        stage.payload_kind == StagePayloadKind::ProcessMla) {
      std::fprintf(stderr,
                   "[stage-storage-mla] stage=%s id=%s physical_outputs=%zu logical_outputs=%zu\n",
                   stage.element_name.c_str(), stage.logical_stage_id.c_str(),
                   stage.physical_outputs.size(), stage.logical_outputs.size());
      for (std::size_t i = 0; i < stage.physical_outputs.size(); ++i) {
        const auto& physical = stage.physical_outputs[i];
        std::fprintf(
            stderr,
            "[stage-storage-mla]   physical[%zu] idx=%d seg=%s size=%llu src_idx=%d src_off=%lld\n",
            i, physical.physical_index,
            physical.segment_name.empty() ? "<empty>" : physical.segment_name.c_str(),
            static_cast<unsigned long long>(physical.size_bytes), physical.source_physical_index,
            static_cast<long long>(physical.source_byte_offset));
      }
    }

    out.name_table.reserve(stage.name_table.size());
    for (const auto& name : stage.name_table) {
      out.name_table.push_back(name.empty() ? nullptr : name.c_str());
    }

    out.logical_input_shapes.reserve(stage.logical_inputs.size());
    out.logical_input_stride_bytes.reserve(stage.logical_inputs.size());
    out.logical_input_axis_semantics.reserve(stage.logical_inputs.size());
    out.logical_input_quant_scales.reserve(stage.logical_inputs.size());
    out.logical_input_quant_zps.reserve(stage.logical_inputs.size());
    out.logical_input_quant_specs.reserve(stage.logical_inputs.size());
    out.logical_inputs.reserve(stage.logical_inputs.size());
    for (const auto& input : stage.logical_inputs) {
      out.logical_input_shapes.emplace_back();
      auto& shape_store = out.logical_input_shapes.back();
      shape_store.reserve(input.shape.size());
      for (const auto dim : input.shape) {
        shape_store.push_back(static_cast<gint64>(dim));
      }

      out.logical_input_stride_bytes.emplace_back();
      auto& stride_store = out.logical_input_stride_bytes.back();
      stride_store.reserve(input.stride_bytes.size());
      for (const auto dim : input.stride_bytes) {
        stride_store.push_back(static_cast<gint64>(dim));
      }

      out.logical_input_axis_semantics.push_back(
          derive_axis_semantics_from_layout(input.shape, input.layout));
      auto& axis_semantics_store = out.logical_input_axis_semantics.back();

      const SimaPluginQuantSpec* quant_ptr = nullptr;
      if (input.quant.has_value()) {
        out.logical_input_quant_scales.emplace_back();
        auto& scale_store = out.logical_input_quant_scales.back();
        scale_store.reserve(input.quant->scales.size());
        for (const auto value : input.quant->scales) {
          scale_store.push_back(static_cast<gdouble>(value));
        }

        out.logical_input_quant_zps.emplace_back();
        auto& zp_store = out.logical_input_quant_zps.back();
        zp_store.reserve(input.quant->zero_points.size());
        for (const auto value : input.quant->zero_points) {
          zp_store.push_back(static_cast<gint64>(value));
        }

        SimaPluginQuantSpec quant{};
        quant.granularity =
            static_cast<gint>(input.quant->granularity == QuantGranularity::PerAxis ? 1 : 0);
        quant.axis = input.quant->axis;
        quant.scales = scale_store.empty() ? nullptr : scale_store.data();
        quant.scales_len = static_cast<guint>(scale_store.size());
        quant.zero_points = zp_store.empty() ? nullptr : zp_store.data();
        quant.zero_points_len = static_cast<guint>(zp_store.size());
        out.logical_input_quant_specs.push_back(quant);
        quant_ptr = &out.logical_input_quant_specs.back();
      }

      SimaPluginLogicalInput abi{};
      abi.logical_index = input.logical_index;
      abi.backend_input_index = input.backend_input_index;
      abi.physical_index = input.physical_index;
      abi.shape = shape_store.empty() ? nullptr : shape_store.data();
      abi.shape_len = static_cast<guint>(shape_store.size());
      abi.stride_bytes = stride_store.empty() ? nullptr : stride_store.data();
      abi.stride_bytes_len = static_cast<guint>(stride_store.size());
      abi.byte_offset = static_cast<gint64>(input.byte_offset);
      abi.size_bytes = static_cast<guint64>(input.size_bytes);
      abi.dtype = input.dtype.empty() ? nullptr : input.dtype.c_str();
      abi.axis_semantics = axis_semantics_store.empty() ? nullptr : axis_semantics_store.data();
      abi.axis_semantics_len = static_cast<guint>(axis_semantics_store.size());
      abi.logical_name_id = input.logical_name_id;
      abi.logical_name = input.logical_name.empty() ? nullptr : input.logical_name.c_str();
      abi.backend_name_id = input.backend_name_id;
      abi.backend_name = input.backend_name.empty() ? nullptr : input.backend_name.c_str();
      abi.segment_name_id = input.segment_name_id;
      abi.segment_name = input.segment_name.empty() ? nullptr : input.segment_name.c_str();
      abi.materialization_kind =
          static_cast<SimaPluginTensorMaterializationKind>(input.materialization_kind);
      abi.quant = quant_ptr;
      out.logical_inputs.push_back(abi);
    }

    out.input_bindings.reserve(stage.input_bindings.size());
    for (const auto& binding : stage.input_bindings) {
      SimaPluginInputBinding abi{};
      abi.sink_pad_index = binding.sink_pad_index;
      abi.local_logical_input_index = binding.local_logical_input_index;
      abi.src_stage_index = binding.src_stage_index;
      abi.src_stage_id = binding.src_stage_id.empty() ? nullptr : binding.src_stage_id.c_str();
      abi.src_logical_output_index = binding.src_logical_output_index;
      abi.src_output_slot = binding.src_output_slot;
      abi.src_physical_output_index = binding.src_physical_output_index;
      abi.src_physical_size_bytes = static_cast<guint64>(binding.src_physical_size_bytes);
      abi.src_physical_byte_offset = static_cast<gint64>(binding.src_physical_byte_offset);
      abi.required = binding.required ? TRUE : FALSE;
      abi.cm_input_name_id = binding.cm_input_name_id;
      abi.cm_input_name = binding.cm_input_name.empty() ? nullptr : binding.cm_input_name.c_str();
      abi.source_segment_name_id = binding.source_segment_name_id;
      abi.source_segment_name =
          binding.source_segment_name.empty() ? nullptr : binding.source_segment_name.c_str();
      out.input_bindings.push_back(abi);
    }

    out.physical_inputs.reserve(stage.physical_inputs.size());
    for (const auto& physical : stage.physical_inputs) {
      SimaPluginPhysicalBuffer abi{};
      abi.physical_index = physical.physical_index;
      abi.allocator_index = physical.allocator_index;
      abi.size_bytes = static_cast<guint64>(physical.size_bytes);
      abi.device_kind = device_kind_to_abi(physical.device_kind);
      abi.memory_flags = static_cast<guint64>(physical.memory_flags);
      abi.segment_name_id = physical.segment_name_id;
      abi.segment_name = physical.segment_name.empty() ? nullptr : physical.segment_name.c_str();
      abi.source_physical_index = physical.source_physical_index;
      abi.source_byte_offset = static_cast<gint64>(physical.source_byte_offset);
      out.physical_inputs.push_back(abi);
    }

    out.physical_outputs.reserve(stage.physical_outputs.size());
    for (const auto& physical : stage.physical_outputs) {
      SimaPluginPhysicalBuffer abi{};
      abi.physical_index = physical.physical_index;
      abi.allocator_index = physical.allocator_index;
      abi.size_bytes = static_cast<guint64>(physical.size_bytes);
      abi.device_kind = device_kind_to_abi(physical.device_kind);
      abi.memory_flags = static_cast<guint64>(physical.memory_flags);
      abi.segment_name_id = physical.segment_name_id;
      abi.segment_name = physical.segment_name.empty() ? nullptr : physical.segment_name.c_str();
      abi.source_physical_index = physical.source_physical_index;
      abi.source_byte_offset = static_cast<gint64>(physical.source_byte_offset);
      out.physical_outputs.push_back(abi);
    }

    out.logical_output_shapes.reserve(stage.logical_outputs.size());
    out.logical_output_stride_bytes.reserve(stage.logical_outputs.size());
    out.logical_output_axis_semantics.reserve(stage.logical_outputs.size());
    out.logical_output_quant_scales.reserve(stage.logical_outputs.size());
    out.logical_output_quant_zps.reserve(stage.logical_outputs.size());
    out.logical_output_quant_specs.reserve(stage.logical_outputs.size());
    out.logical_outputs.reserve(stage.logical_outputs.size());
    for (const auto& logical : stage.logical_outputs) {
      out.logical_output_shapes.emplace_back();
      auto& shape_store = out.logical_output_shapes.back();
      shape_store.reserve(logical.shape.size());
      for (const auto dim : logical.shape) {
        shape_store.push_back(static_cast<gint64>(dim));
      }

      out.logical_output_stride_bytes.emplace_back();
      auto& stride_store = out.logical_output_stride_bytes.back();
      stride_store.reserve(logical.stride_bytes.size());
      for (const auto dim : logical.stride_bytes) {
        stride_store.push_back(static_cast<gint64>(dim));
      }

      out.logical_output_axis_semantics.push_back(
          derive_axis_semantics_from_layout(logical.shape, logical.layout));
      auto& axis_semantics_store = out.logical_output_axis_semantics.back();

      const SimaPluginQuantSpec* quant_ptr = nullptr;
      if (logical.quant.has_value()) {
        out.logical_output_quant_scales.emplace_back();
        auto& scale_store = out.logical_output_quant_scales.back();
        scale_store.reserve(logical.quant->scales.size());
        for (const auto value : logical.quant->scales) {
          scale_store.push_back(static_cast<gdouble>(value));
        }

        out.logical_output_quant_zps.emplace_back();
        auto& zp_store = out.logical_output_quant_zps.back();
        zp_store.reserve(logical.quant->zero_points.size());
        for (const auto value : logical.quant->zero_points) {
          zp_store.push_back(static_cast<gint64>(value));
        }

        SimaPluginQuantSpec quant{};
        quant.granularity =
            static_cast<gint>(logical.quant->granularity == QuantGranularity::PerAxis ? 1 : 0);
        quant.axis = logical.quant->axis;
        quant.scales = scale_store.empty() ? nullptr : scale_store.data();
        quant.scales_len = static_cast<guint>(scale_store.size());
        quant.zero_points = zp_store.empty() ? nullptr : zp_store.data();
        quant.zero_points_len = static_cast<guint>(zp_store.size());
        out.logical_output_quant_specs.push_back(quant);
        quant_ptr = &out.logical_output_quant_specs.back();
      }

      SimaPluginLogicalTensor abi{};
      abi.logical_index = logical.logical_index;
      abi.backend_output_index = logical.backend_output_index;
      abi.physical_index = logical.physical_index;
      abi.output_slot = logical.output_slot;
      abi.tensor_index = logical.tensor_index;
      abi.byte_offset = static_cast<gint64>(logical.byte_offset);
      abi.size_bytes = static_cast<guint64>(logical.size_bytes);
      abi.shape = shape_store.empty() ? nullptr : shape_store.data();
      abi.shape_len = static_cast<guint>(shape_store.size());
      abi.stride_bytes = stride_store.empty() ? nullptr : stride_store.data();
      abi.stride_bytes_len = static_cast<guint>(stride_store.size());
      abi.dtype = logical.dtype.empty() ? nullptr : logical.dtype.c_str();
      abi.axis_semantics = axis_semantics_store.empty() ? nullptr : axis_semantics_store.data();
      abi.axis_semantics_len = static_cast<guint>(axis_semantics_store.size());
      abi.logical_name_id = logical.logical_name_id;
      abi.logical_name = logical.logical_name.empty() ? nullptr : logical.logical_name.c_str();
      abi.backend_name_id = logical.backend_name_id;
      abi.backend_name = logical.backend_name.empty() ? nullptr : logical.backend_name.c_str();
      abi.segment_name_id = logical.segment_name_id;
      abi.segment_name = logical.segment_name.empty() ? nullptr : logical.segment_name.c_str();
      abi.quant = quant_ptr;
      out.logical_outputs.push_back(abi);
    }

    out.output_route_cm_output_names.clear();
    out.output_route_segment_names.clear();
    out.output_route_cm_output_names.reserve(stage.output_order.size());
    out.output_route_segment_names.reserve(stage.output_order.size());
    out.output_order.reserve(stage.output_order.size());
    for (const auto& route : stage.output_order) {
      out.output_route_cm_output_names.push_back(route.cm_output_name);
      out.output_route_segment_names.push_back(route.segment_name);
      SimaPluginOutputRoute oroute{};
      oroute.output_slot = route.output_slot;
      oroute.cm_output_name = out.output_route_cm_output_names.back().empty()
                                  ? nullptr
                                  : out.output_route_cm_output_names.back().c_str();
      oroute.segment_name = out.output_route_segment_names.back().empty()
                                ? nullptr
                                : out.output_route_segment_names.back().c_str();
      out.output_order.push_back(oroute);
    }

    out.quant_scales.reserve(stage.output_quant.size());
    out.quant_zps.reserve(stage.output_quant.size());
    out.output_quant.reserve(stage.output_quant.size());
    for (const auto& q : stage.output_quant) {
      out.quant_scales.emplace_back();
      auto& scale_store = out.quant_scales.back();
      scale_store.reserve(q.scales.size());
      for (const auto value : q.scales) {
        scale_store.push_back(static_cast<gdouble>(value));
      }

      out.quant_zps.emplace_back();
      auto& zp_store = out.quant_zps.back();
      zp_store.reserve(q.zero_points.size());
      for (const auto value : q.zero_points) {
        zp_store.push_back(static_cast<gint64>(value));
      }

      SimaPluginQuantSpec qs{};
      qs.granularity = static_cast<gint>(q.granularity == QuantGranularity::PerAxis ? 1 : 0);
      qs.axis = q.axis;
      qs.scales = scale_store.empty() ? nullptr : scale_store.data();
      qs.scales_len = static_cast<guint>(scale_store.size());
      qs.zero_points = zp_store.empty() ? nullptr : zp_store.data();
      qs.zero_points_len = static_cast<guint>(zp_store.size());
      out.output_quant.push_back(qs);
    }

    out.required_meta_fields.reserve(stage.required_preprocess_meta_fields.size());
    for (const auto& field : stage.required_preprocess_meta_fields) {
      out.required_meta_fields.push_back(field.c_str());
    }

    out.spec.element_name = stage.element_name.empty() ? nullptr : stage.element_name.c_str();
    out.spec.logical_stage_id =
        stage.logical_stage_id.empty() ? nullptr : stage.logical_stage_id.c_str();
    out.spec.plugin_kind = stage.plugin_kind.empty() ? nullptr : stage.plugin_kind.c_str();
    out.spec.kernel_kind = stage.kernel_kind.empty() ? nullptr : stage.kernel_kind.c_str();
    out.spec.name_table = out.name_table.empty() ? nullptr : out.name_table.data();
    out.spec.name_table_len = static_cast<guint>(out.name_table.size());
    out.spec.logical_inputs = out.logical_inputs.empty() ? nullptr : out.logical_inputs.data();
    out.spec.logical_inputs_len = static_cast<guint>(out.logical_inputs.size());
    out.spec.input_bindings = out.input_bindings.empty() ? nullptr : out.input_bindings.data();
    out.spec.input_bindings_len = static_cast<guint>(out.input_bindings.size());
    out.spec.physical_inputs = out.physical_inputs.empty() ? nullptr : out.physical_inputs.data();
    out.spec.physical_inputs_len = static_cast<guint>(out.physical_inputs.size());
    out.spec.physical_outputs =
        out.physical_outputs.empty() ? nullptr : out.physical_outputs.data();
    out.spec.physical_outputs_len = static_cast<guint>(out.physical_outputs.size());
    out.spec.logical_outputs = out.logical_outputs.empty() ? nullptr : out.logical_outputs.data();
    out.spec.logical_outputs_len = static_cast<guint>(out.logical_outputs.size());
    out.spec.output_order = out.output_order.empty() ? nullptr : out.output_order.data();
    out.spec.output_order_len = static_cast<guint>(out.output_order.size());
    out.spec.output_quant = out.output_quant.empty() ? nullptr : out.output_quant.data();
    out.spec.output_quant_len = static_cast<guint>(out.output_quant.size());
    out.spec.required_preprocess_meta_fields =
        out.required_meta_fields.empty() ? nullptr : out.required_meta_fields.data();
    out.spec.required_preprocess_meta_fields_len =
        static_cast<guint>(out.required_meta_fields.size());
    out.spec.payload_kind = payload_kind_to_abi(stage.payload_kind);

    switch (stage.payload_kind) {
    case StagePayloadKind::ProcessCvu: {
      out.spec.payload.processcvu.graph_family =
          stage.processcvu.graph_family.empty() ? nullptr : stage.processcvu.graph_family.c_str();
      out.spec.payload.processcvu.graph_family_kind =
          processcvu_family_to_abi(stage.processcvu.graph_family_enum);
      out.spec.payload.processcvu.graph_name =
          stage.processcvu.graph_name.empty() ? nullptr : stage.processcvu.graph_name.c_str();
      out.spec.payload.processcvu.requested_run_target =
          stage.processcvu.requested_run_target.empty()
              ? nullptr
              : stage.processcvu.requested_run_target.c_str();
      out.spec.payload.processcvu.run_target =
          stage.processcvu.run_target.empty() ? nullptr : stage.processcvu.run_target.c_str();
      out.spec.payload.processcvu.resolved_exec_backend =
          stage.processcvu.resolved_exec_backend.empty()
              ? nullptr
              : stage.processcvu.resolved_exec_backend.c_str();
      out.spec.payload.processcvu.run_target_resolution_reason =
          stage.processcvu.run_target_resolution_reason.empty()
              ? nullptr
              : stage.processcvu.run_target_resolution_reason.c_str();
      out.spec.payload.processcvu.default_input_name =
          stage.processcvu.default_input_name.empty() ? nullptr
                                                      : stage.processcvu.default_input_name.c_str();
      out.processcvu_default_output_name_storage.clear();
      out.processcvu_default_output_name_storage.reserve(
          stage.processcvu.default_output_names.size());
      out.processcvu_default_output_names.clear();
      out.processcvu_default_output_names.reserve(stage.processcvu.default_output_names.size());
      for (const auto& output_name : stage.processcvu.default_output_names) {
        out.processcvu_default_output_name_storage.push_back(output_name);
        out.processcvu_default_output_names.push_back(
            out.processcvu_default_output_name_storage.back().c_str());
      }
      out.spec.payload.processcvu.default_output_names =
          out.processcvu_default_output_names.empty() ? nullptr
                                                      : out.processcvu_default_output_names.data();
      out.spec.payload.processcvu.default_output_names_len =
          static_cast<guint>(out.processcvu_default_output_names.size());
      out.spec.payload.processcvu.primary_output_name =
          stage.processcvu.primary_output_name.empty()
              ? nullptr
              : stage.processcvu.primary_output_name.c_str();
      out.spec.payload.processcvu.primary_output_transport_kind =
          processcvu_output_transport_to_abi(stage.processcvu.primary_output_transport_kind);
      out.spec.payload.processcvu.primary_output_semantic_kind =
          processcvu_output_semantic_to_abi(stage.processcvu.primary_output_semantic_kind);
      out.spec.payload.processcvu.input_img_type = stage.processcvu.input_img_type.empty()
                                                       ? nullptr
                                                       : stage.processcvu.input_img_type.c_str();
      out.spec.payload.processcvu.output_img_type = stage.processcvu.output_img_type.empty()
                                                        ? nullptr
                                                        : stage.processcvu.output_img_type.c_str();
      out.spec.payload.processcvu.scaling_type =
          stage.processcvu.scaling_type.empty() ? nullptr : stage.processcvu.scaling_type.c_str();
      out.spec.payload.processcvu.padding_type =
          stage.processcvu.padding_type.empty() ? nullptr : stage.processcvu.padding_type.c_str();
      out.spec.payload.processcvu.input_dtype =
          stage.processcvu.input_dtype.empty() ? nullptr : stage.processcvu.input_dtype.c_str();
      out.spec.payload.processcvu.output_dtype =
          stage.processcvu.output_dtype.empty() ? nullptr : stage.processcvu.output_dtype.c_str();
      out.spec.payload.processcvu.out_dtype =
          stage.processcvu.out_dtype.empty() ? nullptr : stage.processcvu.out_dtype.c_str();

      out.spec.payload.processcvu.pad_value = stage.processcvu.pad_value;
      out.spec.payload.processcvu.scaled_width = stage.processcvu.scaled_width;
      out.spec.payload.processcvu.scaled_height = stage.processcvu.scaled_height;
      out.spec.payload.processcvu.input_stride = stage.processcvu.input_stride;
      out.spec.payload.processcvu.output_stride = stage.processcvu.output_stride;
      out.spec.payload.processcvu.input_offset = stage.processcvu.input_offset;
      out.spec.payload.processcvu.batch_size = stage.processcvu.batch_size;
      out.spec.payload.processcvu.round_off = stage.processcvu.round_off;
      out.spec.payload.processcvu.byte_align = stage.processcvu.byte_align;
      out.spec.payload.processcvu.graph_id = stage.processcvu.graph_id;
      out.spec.payload.processcvu.width = stage.processcvu.width;
      out.spec.payload.processcvu.height = stage.processcvu.height;
      out.spec.payload.processcvu.threshold = stage.processcvu.threshold;
      out.spec.payload.processcvu.max_features = stage.processcvu.max_features;
      out.spec.payload.processcvu.grid_x = stage.processcvu.grid_x;
      out.spec.payload.processcvu.grid_y = stage.processcvu.grid_y;
      out.spec.payload.processcvu.min_px_dist = stage.processcvu.min_px_dist;
      out.spec.payload.processcvu.descriptor_words = stage.processcvu.descriptor_words;
      out.spec.payload.processcvu.num_points = stage.processcvu.num_points;
      out.spec.payload.processcvu.win_half = stage.processcvu.win_half;
      out.spec.payload.processcvu.max_iters = stage.processcvu.max_iters;
      out.spec.payload.processcvu.max_level = stage.processcvu.max_level;
      out.spec.payload.processcvu.detect_new_features = stage.processcvu.detect_new_features;
      out.spec.payload.processcvu.fast_threshold = stage.processcvu.fast_threshold;
      out.spec.payload.processcvu.debug = stage.processcvu.debug;
      out.spec.payload.processcvu.opt_flags = stage.processcvu.opt_flags;
      out.spec.payload.processcvu.canonical_contract =
          stage.processcvu.canonical_contract ? TRUE : FALSE;
      out.spec.payload.processcvu.preproc_single_output_handoff =
          stage.processcvu.preproc_single_output_handoff ? TRUE : FALSE;
      out.spec.payload.processcvu.aspect_ratio = stage.processcvu.aspect_ratio;
      out.spec.payload.processcvu.normalize = stage.processcvu.normalize;
      out.spec.payload.processcvu.tessellate = stage.processcvu.tessellate;
      out.spec.payload.processcvu.q_scale = stage.processcvu.q_scale;
      out.spec.payload.processcvu.q_zp = stage.processcvu.q_zp;
      out.spec.payload.processcvu.has_q_scale = stage.processcvu.has_q_scale ? TRUE : FALSE;
      out.spec.payload.processcvu.has_q_zp = stage.processcvu.has_q_zp ? TRUE : FALSE;

      out.processcvu_q_scale_array.clear();
      out.processcvu_q_scale_array.reserve(stage.processcvu.q_scale_list.size());
      for (const auto value : stage.processcvu.q_scale_list) {
        out.processcvu_q_scale_array.push_back(static_cast<gdouble>(value));
      }
      out.spec.payload.processcvu.q_scale_array =
          out.processcvu_q_scale_array.empty() ? nullptr : out.processcvu_q_scale_array.data();
      out.spec.payload.processcvu.q_scale_array_len =
          static_cast<guint>(out.processcvu_q_scale_array.size());

      out.processcvu_q_zp_array.clear();
      out.processcvu_q_zp_array.reserve(stage.processcvu.q_zp_list.size());
      for (const auto value : stage.processcvu.q_zp_list) {
        out.processcvu_q_zp_array.push_back(static_cast<gint>(value));
      }
      out.spec.payload.processcvu.q_zp_array =
          out.processcvu_q_zp_array.empty() ? nullptr : out.processcvu_q_zp_array.data();
      out.spec.payload.processcvu.q_zp_array_len =
          static_cast<guint>(out.processcvu_q_zp_array.size());

      out.spec.payload.processcvu.num_in_tensor = stage.processcvu.num_in_tensor;
      out.processcvu_input_tensors = stage.processcvu.input_tensors;
      out.spec.payload.processcvu.input_tensors =
          out.processcvu_input_tensors.empty() ? nullptr : out.processcvu_input_tensors.data();
      out.spec.payload.processcvu.input_tensors_len =
          static_cast<guint>(out.processcvu_input_tensors.size());
      out.processcvu_output_tensors = stage.processcvu.output_tensors;
      out.spec.payload.processcvu.output_tensors =
          out.processcvu_output_tensors.empty() ? nullptr : out.processcvu_output_tensors.data();
      out.spec.payload.processcvu.output_tensors_len =
          static_cast<guint>(out.processcvu_output_tensors.size());

      out.processcvu_runtime_output_logical_index_array.clear();
      out.processcvu_runtime_output_logical_index_array.reserve(
          stage.processcvu.runtime_output_logical_index_list.size());
      for (const auto value : stage.processcvu.runtime_output_logical_index_list) {
        out.processcvu_runtime_output_logical_index_array.push_back(static_cast<gint>(value));
      }
      out.spec.payload.processcvu.runtime_output_logical_index_array =
          out.processcvu_runtime_output_logical_index_array.empty()
              ? nullptr
              : out.processcvu_runtime_output_logical_index_array.data();
      out.spec.payload.processcvu.runtime_output_logical_index_array_len =
          static_cast<guint>(out.processcvu_runtime_output_logical_index_array.size());

      out.processcvu_runtime_output_output_slot_array.clear();
      out.processcvu_runtime_output_output_slot_array.reserve(
          stage.processcvu.runtime_output_output_slot_list.size());
      for (const auto value : stage.processcvu.runtime_output_output_slot_list) {
        out.processcvu_runtime_output_output_slot_array.push_back(static_cast<gint>(value));
      }
      out.spec.payload.processcvu.runtime_output_output_slot_array =
          out.processcvu_runtime_output_output_slot_array.empty()
              ? nullptr
              : out.processcvu_runtime_output_output_slot_array.data();
      out.spec.payload.processcvu.runtime_output_output_slot_array_len =
          static_cast<guint>(out.processcvu_runtime_output_output_slot_array.size());

      out.processcvu_runtime_output_physical_index_array.clear();
      out.processcvu_runtime_output_physical_index_array.reserve(
          stage.processcvu.runtime_output_physical_index_list.size());
      for (const auto value : stage.processcvu.runtime_output_physical_index_list) {
        out.processcvu_runtime_output_physical_index_array.push_back(static_cast<gint>(value));
      }
      out.spec.payload.processcvu.runtime_output_physical_index_array =
          out.processcvu_runtime_output_physical_index_array.empty()
              ? nullptr
              : out.processcvu_runtime_output_physical_index_array.data();
      out.spec.payload.processcvu.runtime_output_physical_index_array_len =
          static_cast<guint>(out.processcvu_runtime_output_physical_index_array.size());

      out.processcvu_runtime_output_dtype_array.clear();
      out.processcvu_runtime_output_dtype_array.reserve(
          stage.processcvu.runtime_output_dtype_list.size());
      for (const auto& value : stage.processcvu.runtime_output_dtype_list) {
        out.processcvu_runtime_output_dtype_array.push_back(value.empty() ? nullptr
                                                                          : value.c_str());
      }
      out.spec.payload.processcvu.runtime_output_dtype_array =
          out.processcvu_runtime_output_dtype_array.empty()
              ? nullptr
              : out.processcvu_runtime_output_dtype_array.data();
      out.spec.payload.processcvu.runtime_output_dtype_array_len =
          static_cast<guint>(out.processcvu_runtime_output_dtype_array.size());

      out.processcvu_runtime_output_transport_kind_array.clear();
      out.processcvu_runtime_output_transport_kind_array.reserve(
          stage.processcvu.runtime_output_transport_kind_list.size());
      for (const auto value : stage.processcvu.runtime_output_transport_kind_list) {
        out.processcvu_runtime_output_transport_kind_array.push_back(
            static_cast<gint>(processcvu_output_transport_to_abi(value)));
      }
      out.spec.payload.processcvu.runtime_output_transport_kind_array =
          out.processcvu_runtime_output_transport_kind_array.empty()
              ? nullptr
              : out.processcvu_runtime_output_transport_kind_array.data();
      out.spec.payload.processcvu.runtime_output_transport_kind_array_len =
          static_cast<guint>(out.processcvu_runtime_output_transport_kind_array.size());

      out.processcvu_runtime_output_semantic_kind_array.clear();
      out.processcvu_runtime_output_semantic_kind_array.reserve(
          stage.processcvu.runtime_output_semantic_kind_list.size());
      for (const auto value : stage.processcvu.runtime_output_semantic_kind_list) {
        out.processcvu_runtime_output_semantic_kind_array.push_back(
            static_cast<gint>(processcvu_output_semantic_to_abi(value)));
      }
      out.spec.payload.processcvu.runtime_output_semantic_kind_array =
          out.processcvu_runtime_output_semantic_kind_array.empty()
              ? nullptr
              : out.processcvu_runtime_output_semantic_kind_array.data();
      out.spec.payload.processcvu.runtime_output_semantic_kind_array_len =
          static_cast<guint>(out.processcvu_runtime_output_semantic_kind_array.size());

      out.processcvu_dq_scale_array.clear();
      out.processcvu_dq_scale_array.reserve(stage.processcvu.dq_scale_list.size());
      for (const auto value : stage.processcvu.dq_scale_list) {
        out.processcvu_dq_scale_array.push_back(static_cast<gdouble>(value));
      }
      out.spec.payload.processcvu.dq_scale_array =
          out.processcvu_dq_scale_array.empty() ? nullptr : out.processcvu_dq_scale_array.data();
      out.spec.payload.processcvu.dq_scale_array_len =
          static_cast<guint>(out.processcvu_dq_scale_array.size());

      out.processcvu_dq_zp_array.clear();
      out.processcvu_dq_zp_array.reserve(stage.processcvu.dq_zp_list.size());
      for (const auto value : stage.processcvu.dq_zp_list) {
        out.processcvu_dq_zp_array.push_back(static_cast<gint>(value));
      }
      out.spec.payload.processcvu.dq_zp_array =
          out.processcvu_dq_zp_array.empty() ? nullptr : out.processcvu_dq_zp_array.data();
      out.spec.payload.processcvu.dq_zp_array_len =
          static_cast<guint>(out.processcvu_dq_zp_array.size());

      out.processcvu_channel_mean.clear();
      out.processcvu_channel_mean.reserve(stage.processcvu.channel_mean.size());
      for (const auto mean : stage.processcvu.channel_mean) {
        out.processcvu_channel_mean.push_back(static_cast<gdouble>(mean));
      }
      out.processcvu_channel_stddev.clear();
      out.processcvu_channel_stddev.reserve(stage.processcvu.channel_stddev.size());
      for (const auto stdv : stage.processcvu.channel_stddev) {
        out.processcvu_channel_stddev.push_back(static_cast<gdouble>(stdv));
      }
      out.spec.payload.processcvu.channel_mean =
          out.processcvu_channel_mean.empty() ? nullptr : out.processcvu_channel_mean.data();
      out.spec.payload.processcvu.channel_mean_len =
          static_cast<guint>(out.processcvu_channel_mean.size());
      out.spec.payload.processcvu.channel_stddev =
          out.processcvu_channel_stddev.empty() ? nullptr : out.processcvu_channel_stddev.data();
      out.spec.payload.processcvu.channel_stddev_len =
          static_cast<guint>(out.processcvu_channel_stddev.size());
      break;
    }
    case StagePayloadKind::ProcessMla:
      out.spec.payload.processmla.model_path =
          stage.processmla.model_path.empty() ? nullptr : stage.processmla.model_path.c_str();
      out.spec.payload.processmla.batch_size = stage.processmla.batch_size;
      out.spec.payload.processmla.batch_sz_model = stage.processmla.batch_sz_model;
      out.processmla_dispatcher_output_names.clear();
      out.processmla_dispatcher_output_names.reserve(
          stage.processmla.dispatcher_output_names.size());
      for (const auto& name : stage.processmla.dispatcher_output_names) {
        out.processmla_dispatcher_output_names.push_back(name.empty() ? nullptr : name.c_str());
      }
      out.spec.payload.processmla.dispatcher_output_names =
          out.processmla_dispatcher_output_names.empty()
              ? nullptr
              : out.processmla_dispatcher_output_names.data();
      out.spec.payload.processmla.dispatcher_output_names_len =
          static_cast<guint>(out.processmla_dispatcher_output_names.size());
      out.processmla_dispatcher_output_sizes.clear();
      out.processmla_dispatcher_output_sizes.reserve(
          stage.processmla.dispatcher_output_sizes.size());
      for (const auto value : stage.processmla.dispatcher_output_sizes) {
        out.processmla_dispatcher_output_sizes.push_back(static_cast<guint64>(value));
      }
      out.spec.payload.processmla.dispatcher_output_sizes =
          out.processmla_dispatcher_output_sizes.empty()
              ? nullptr
              : out.processmla_dispatcher_output_sizes.data();
      out.spec.payload.processmla.dispatcher_output_sizes_len =
          static_cast<guint>(out.processmla_dispatcher_output_sizes.size());
      break;
    case StagePayloadKind::BoxDecode:
      set_boxdecode_decode_type_token_abi_safe(
          out.spec.payload.boxdecode, is_box_decode_type_specified(stage.boxdecode.decode_type)
                                          ? box_decode_type_token(stage.boxdecode.decode_type)
                                          : nullptr);
      out.spec.payload.boxdecode.decode_type_option =
          stage.boxdecode.decode_type_option.has_value()
              ? box_decode_type_option_token(*stage.boxdecode.decode_type_option)
              : nullptr;
      out.spec.payload.boxdecode.input_dtype =
          stage.boxdecode.input_dtype.empty() ? nullptr : stage.boxdecode.input_dtype.c_str();
      out.spec.payload.boxdecode.score_activation =
          static_cast<gint>(stage.boxdecode.score_activation);
      out.spec.payload.boxdecode.tess_needed = stage.boxdecode.tess_needed ? 1 : 0;
      out.spec.payload.boxdecode.quant_needed = stage.boxdecode.quant_needed ? 1 : 0;
      out.spec.payload.boxdecode.model_owned_flags = stage.boxdecode.model_owned_flags ? 1 : 0;
      out.spec.payload.boxdecode.quant_contract_required =
          stage.boxdecode.quant_contract_required ? 1 : 0;
      out.spec.payload.boxdecode.detection_threshold = stage.boxdecode.detection_threshold;
      out.spec.payload.boxdecode.nms_iou_threshold = stage.boxdecode.nms_iou_threshold;
      out.spec.payload.boxdecode.topk = stage.boxdecode.topk;
      out.spec.payload.boxdecode.num_classes = stage.boxdecode.num_classes;

      out.boxdecode_slice_shapes = stage.boxdecode.slice_shapes;
      out.spec.payload.boxdecode.slice_shapes =
          out.boxdecode_slice_shapes.empty() ? nullptr : out.boxdecode_slice_shapes.data();
      out.spec.payload.boxdecode.slice_shapes_len =
          static_cast<guint>(out.boxdecode_slice_shapes.size());
      break;
    case StagePayloadKind::DetessDequant:
      out.spec.payload.detessdequant.reserved = stage.detessdequant.reserved;
      break;
    case StagePayloadKind::Quant:
      out.spec.payload.quant.reserved = stage.quant.reserved;
      break;
    case StagePayloadKind::Tess:
      out.spec.payload.tess.reserved = stage.tess.reserved;
      break;
    case StagePayloadKind::Dequant:
      out.spec.payload.dequant.reserved = stage.dequant.reserved;
      break;
    case StagePayloadKind::QuantTess:
      out.spec.payload.quanttess.reserved = stage.quanttess.reserved;
      break;
    case StagePayloadKind::None:
    default:
      break;
    }

    if (manifest_stage_debug_enabled() &&
        out.spec.payload_kind == SIMA_PLUGIN_STAGE_PAYLOAD_PROCESSCVU) {
      const auto& payload = out.spec.payload.processcvu;
      std::fprintf(stderr,
                   "[manifest-stage-debug] stage=%s id=%s spec=%p payload=%p "
                   "graph=%s family=%s input_tensors=%p len=%u output_tensors=%p len=%u "
                   "dq_scale_array=%p len=%u "
                   "dq_zp_array=%p len=%u default_output_names=%p len=%u\n",
                   out.spec.element_name ? out.spec.element_name : "<null>",
                   out.spec.logical_stage_id ? out.spec.logical_stage_id : "<null>",
                   static_cast<const void*>(&out.spec), static_cast<const void*>(&payload),
                   payload.graph_name ? payload.graph_name : "<null>",
                   payload.graph_family ? payload.graph_family : "<null>",
                   static_cast<const void*>(payload.input_tensors), payload.input_tensors_len,
                   static_cast<const void*>(payload.output_tensors), payload.output_tensors_len,
                   static_cast<const void*>(payload.dq_scale_array), payload.dq_scale_array_len,
                   static_cast<const void*>(payload.dq_zp_array), payload.dq_zp_array_len,
                   static_cast<const void*>(payload.default_output_names),
                   payload.default_output_names_len);
    }

    return out;
  }

  static const gchar* session_id_cb(gpointer user_data) {
    if (!user_data) {
      return nullptr;
    }
    const auto* self = static_cast<const ManifestAccessorRegistry*>(user_data);
    return self->session_id_.empty() ? nullptr : self->session_id_.c_str();
  }

  static const gchar* model_id_cb(gpointer user_data) {
    if (!user_data) {
      return nullptr;
    }
    const auto* self = static_cast<const ManifestAccessorRegistry*>(user_data);
    return self->model_id_.empty() ? nullptr : self->model_id_.c_str();
  }

  static guint stage_count_cb(gpointer user_data) {
    if (!user_data) {
      return 0;
    }
    const auto* self = static_cast<const ManifestAccessorRegistry*>(user_data);
    return static_cast<guint>(self->stage_storage_.size());
  }

  static const SimaPluginStageSpec* stage_by_index_cb(gpointer user_data, guint index) {
    if (!user_data) {
      return nullptr;
    }
    const auto* self = static_cast<const ManifestAccessorRegistry*>(user_data);
    if (index >= self->stage_storage_.size()) {
      return nullptr;
    }
    if (manifest_stage_debug_enabled()) {
      const auto* stage = &self->stage_storage_[index].spec;
      if (stage->payload_kind == SIMA_PLUGIN_STAGE_PAYLOAD_PROCESSCVU) {
        const auto& payload = stage->payload.processcvu;
        std::fprintf(stderr,
                     "[manifest-stage-debug] accessor index=%u stage=%s id=%s spec=%p payload=%p "
                     "logical_outputs_len=%u output_order_len=%u "
                     "graph=%s family=%s input_tensors=%p len=%u output_tensors=%p len=%u "
                     "dq_scale_array=%p len=%u "
                     "dq_zp_array=%p len=%u default_output_names=%p len=%u\n",
                     index, stage->element_name ? stage->element_name : "<null>",
                     stage->logical_stage_id ? stage->logical_stage_id : "<null>",
                     static_cast<const void*>(stage), static_cast<const void*>(&payload),
                     stage->logical_outputs_len, stage->output_order_len,
                     payload.graph_name ? payload.graph_name : "<null>",
                     payload.graph_family ? payload.graph_family : "<null>",
                     static_cast<const void*>(payload.input_tensors), payload.input_tensors_len,
                     static_cast<const void*>(payload.output_tensors), payload.output_tensors_len,
                     static_cast<const void*>(payload.dq_scale_array), payload.dq_scale_array_len,
                     static_cast<const void*>(payload.dq_zp_array), payload.dq_zp_array_len,
                     static_cast<const void*>(payload.default_output_names),
                     payload.default_output_names_len);
      }
    }
    return &self->stage_storage_[index].spec;
  }

  static const SimaPluginStageSpec* stage_by_element_cb(gpointer user_data,
                                                        const gchar* element_name) {
    if (!user_data || !element_name || !*element_name) {
      return nullptr;
    }
    const auto* self = static_cast<const ManifestAccessorRegistry*>(user_data);
    const auto it = self->stage_by_element_.find(element_name);
    const bool debug_lookup = pipeline_internal::env_bool("SIMA_MLA_CONTRACT_DEBUG", false);
    if (debug_lookup) {
      std::fprintf(stderr, "[MLA-CONTRACT][Accessor] lookup_by_element key=%s hit=%d\n",
                   element_name, it == self->stage_by_element_.end() ? 0 : 1);
    }
    if (it == self->stage_by_element_.end()) {
      return nullptr;
    }
    if (debug_lookup) {
      const auto& spec = self->stage_storage_[it->second].spec;
      std::fprintf(stderr, "[MLA-CONTRACT][Accessor]   stage=%s id=%s payload_kind=%d\n",
                   spec.element_name ? spec.element_name : "<null>",
                   spec.logical_stage_id ? spec.logical_stage_id : "<null>",
                   static_cast<int>(spec.payload_kind));
    }
    return &self->stage_storage_[it->second].spec;
  }

  static const SimaPluginStageSpec* stage_by_id_cb(gpointer user_data,
                                                   const gchar* logical_stage_id) {
    if (!user_data || !logical_stage_id || !*logical_stage_id) {
      return nullptr;
    }
    const auto* self = static_cast<const ManifestAccessorRegistry*>(user_data);
    const auto it = self->stage_by_id_.find(logical_stage_id);
    const bool debug_lookup = pipeline_internal::env_bool("SIMA_MLA_CONTRACT_DEBUG", false);
    if (debug_lookup) {
      std::fprintf(stderr, "[MLA-CONTRACT][Accessor] lookup_by_id key=%s hit=%d\n",
                   logical_stage_id, it == self->stage_by_id_.end() ? 0 : 1);
    }
    if (it == self->stage_by_id_.end()) {
      return nullptr;
    }
    if (debug_lookup) {
      const auto& spec = self->stage_storage_[it->second].spec;
      std::fprintf(stderr, "[MLA-CONTRACT][Accessor]   stage=%s id=%s payload_kind=%d\n",
                   spec.element_name ? spec.element_name : "<null>",
                   spec.logical_stage_id ? spec.logical_stage_id : "<null>",
                   static_cast<int>(spec.payload_kind));
    }
    return &self->stage_storage_[it->second].spec;
  }

  SimaPluginStaticManifest manifest_;
  std::string session_id_;
  std::string model_id_;
  SimaPluginStaticManifestAccessor accessor_{};
  std::vector<StageAbiStorage> stage_storage_;
  std::unordered_map<std::string, std::size_t> stage_by_element_;
  std::unordered_map<std::string, std::size_t> stage_by_id_;
};

class ManifestContextOwner {
public:
  explicit ManifestContextOwner(SimaPluginStaticManifest manifest)
      : registry_(std::move(manifest)) {
    handle_.abi_version = SIMA_PLUGIN_STATIC_MANIFEST_ABI_VERSION;
    handle_.user_data = this;
    handle_.ref = &ManifestContextOwner::ref_cb;
    handle_.unref = &ManifestContextOwner::unref_cb;
    handle_.accessor = &ManifestContextOwner::accessor_cb;
  }

  const SimaPluginStaticManifestHandle* handle() const {
    return &handle_;
  }

private:
  static void ref_cb(gpointer user_data) {
    if (user_data) {
      static_cast<ManifestContextOwner*>(user_data)->ref();
    }
  }

  static void unref_cb(gpointer user_data) {
    if (user_data) {
      static_cast<ManifestContextOwner*>(user_data)->unref();
    }
  }

  static const SimaPluginStaticManifestAccessor* accessor_cb(gpointer user_data) {
    if (!user_data) {
      return nullptr;
    }
    return static_cast<ManifestContextOwner*>(user_data)->registry_.accessor();
  }

  void ref() {
    ref_count_.fetch_add(1U, std::memory_order_relaxed);
  }

  void unref() {
    if (ref_count_.fetch_sub(1U, std::memory_order_acq_rel) == 1U) {
      delete this;
    }
  }

  std::atomic<guint> ref_count_{1U};
  ManifestAccessorRegistry registry_;
  SimaPluginStaticManifestHandle handle_{};
};

GQuark manifest_context_registration_quark() {
  static GQuark quark =
      g_quark_from_static_string("sima-plugin-static-manifest-context-registration");
  return quark;
}

struct ManifestContextRegistration {
  GstElement* pipeline = nullptr;
  GstContext* context = nullptr;
  gulong deep_element_added_handler = 0;
};

static constexpr const char* kManifestContextDeepAddedHookKey =
    "sima-plugin-static-manifest-deep-element-added-hook";

void pipeline_deep_element_added_cb(GstBin* pipeline_bin, GstBin*, GstElement* element, gpointer);

void install_manifest_context_deep_added_hook(GstElement* element) {
  if (!element || !GST_IS_BIN(element)) {
    return;
  }
  if (g_object_get_data(G_OBJECT(element), kManifestContextDeepAddedHookKey)) {
    return;
  }
  g_object_set_data(G_OBJECT(element), kManifestContextDeepAddedHookKey, const_cast<char*>("1"));
  g_signal_connect(element, "deep-element-added", G_CALLBACK(pipeline_deep_element_added_cb),
                   nullptr);
}

void manifest_context_registration_destroy(gpointer data) {
  auto* registration = static_cast<ManifestContextRegistration*>(data);
  if (!registration) {
    return;
  }
  if (registration->deep_element_added_handler != 0 && registration->pipeline &&
      G_IS_OBJECT(registration->pipeline) &&
      g_signal_handler_is_connected(registration->pipeline,
                                    registration->deep_element_added_handler)) {
    g_signal_handler_disconnect(registration->pipeline, registration->deep_element_added_handler);
  }
  if (registration->context) {
    gst_context_unref(registration->context);
    registration->context = nullptr;
  }
  delete registration;
}

bool element_has_equivalent_manifest_context(GstElement* element, GstContext* context) {
  if (!element || !context || !sima_plugin_manifest_context_matches(context)) {
    return false;
  }

  GstContext* existing = gst_element_get_context(element, SIMA_PLUGIN_STATIC_MANIFEST_CONTEXT_TYPE);
  if (!existing) {
    return false;
  }

  const auto* existing_handle = sima_plugin_manifest_context_handle(existing);
  const auto* requested_handle = sima_plugin_manifest_context_handle(context);
  const bool equivalent = existing == context || (existing_handle && requested_handle &&
                                                  existing_handle == requested_handle);
  gst_context_unref(existing);
  return equivalent;
}

void apply_manifest_context_recursive(GstElement* element, GstContext* context) {
  if (!element || !context) {
    return;
  }

  if (element_has_equivalent_manifest_context(element, context)) {
    manifest_context_debug_log("set_context_recursive_skip_existing", element, context);
  } else {
    manifest_context_debug_log("set_context_recursive", element, context);
    gst_element_set_context(element, context);
  }
  install_manifest_context_deep_added_hook(element);
  if (!GST_IS_BIN(element)) {
    return;
  }

  GstIterator* it = gst_bin_iterate_recurse(GST_BIN(element));
  if (!it) {
    return;
  }

  GValue item = G_VALUE_INIT;
  gboolean done = FALSE;
  while (!done) {
    switch (gst_iterator_next(it, &item)) {
    case GST_ITERATOR_OK: {
      GstElement* child = GST_ELEMENT(g_value_get_object(&item));
      if (child && child != element) {
        if (element_has_equivalent_manifest_context(child, context)) {
          manifest_context_debug_log("set_context_recursive_child_skip_existing", child, context);
        } else {
          manifest_context_debug_log("set_context_recursive_child", child, context);
          gst_element_set_context(child, context);
        }
      }
      g_value_reset(&item);
      break;
    }
    case GST_ITERATOR_RESYNC:
      gst_iterator_resync(it);
      break;
    case GST_ITERATOR_DONE:
      done = TRUE;
      break;
    case GST_ITERATOR_ERROR:
    default:
      done = TRUE;
      break;
    }
  }
  g_value_unset(&item);
  gst_iterator_free(it);
}

void pipeline_deep_element_added_cb(GstBin* pipeline_bin, GstBin*, GstElement* element, gpointer) {
  auto* pipeline = GST_ELEMENT(pipeline_bin);
  if (!pipeline || !element) {
    return;
  }

  auto* registration = static_cast<ManifestContextRegistration*>(
      g_object_get_qdata(G_OBJECT(pipeline), manifest_context_registration_quark()));
  if (!registration || !registration->context) {
    return;
  }

  manifest_context_debug_log("deep_element_added", element, registration->context);
  apply_manifest_context_recursive(element, registration->context);
}

} // namespace

bool attach_manifest_context(GstElement* pipeline, const SimaPluginStaticManifest& manifest,
                             std::string* error_message) {
  if (error_message)
    error_message->clear();
  if (!pipeline) {
    if (error_message)
      *error_message = "attach manifest context failed: pipeline is null";
    return false;
  }

  GstContext* context = gst_context_new(SIMA_PLUGIN_STATIC_MANIFEST_CONTEXT_TYPE, TRUE);
  if (!context) {
    if (error_message)
      *error_message = "attach manifest context failed: gst_context_new returned null";
    return false;
  }

  auto* owner = new (std::nothrow) ManifestContextOwner(manifest);
  if (!owner) {
    if (error_message) {
      *error_message = "attach manifest context failed: unable to create manifest owner";
    }
    gst_context_unref(context);
    return false;
  }

  GstStructure* structure = gst_context_writable_structure(context);

  GValue handle_value = G_VALUE_INIT;
  g_value_init(&handle_value, sima_plugin_static_manifest_handle_get_type());
  g_value_take_boxed(&handle_value, const_cast<SimaPluginStaticManifestHandle*>(owner->handle()));
  gst_structure_take_value(structure, SIMA_PLUGIN_STATIC_MANIFEST_KEY_HANDLE, &handle_value);
  if (!manifest.session_id.empty()) {
    gst_structure_set(structure, SIMA_PLUGIN_STATIC_MANIFEST_KEY_SESSION_ID, G_TYPE_STRING,
                      manifest.session_id.c_str(), nullptr);
  }
  if (!manifest.model_id.empty()) {
    gst_structure_set(structure, SIMA_PLUGIN_STATIC_MANIFEST_KEY_MODEL_ID, G_TYPE_STRING,
                      manifest.model_id.c_str(), nullptr);
  }

  auto* registration = new (std::nothrow) ManifestContextRegistration();
  if (!registration) {
    if (error_message) {
      *error_message = "attach manifest context failed: unable to create manifest registration";
    }
    gst_context_unref(context);
    return false;
  }
  registration->pipeline = pipeline;
  registration->context = gst_context_ref(context);
  if (GST_IS_BIN(pipeline)) {
    registration->deep_element_added_handler = g_signal_connect(
        pipeline, "deep-element-added", G_CALLBACK(pipeline_deep_element_added_cb), nullptr);
  }
  g_object_set_qdata_full(G_OBJECT(pipeline), manifest_context_registration_quark(), registration,
                          manifest_context_registration_destroy);
  if (manifest_context_debug_enabled()) {
    std::fprintf(stderr,
                 "[manifest-context-debug] action=attach pipeline=%s context=%p stages=%zu\n",
                 GST_ELEMENT_NAME(pipeline), static_cast<void*>(context), manifest.stages.size());
  }
  apply_manifest_context_recursive(pipeline, context);
  gst_context_unref(context);
  return true;
}

} // namespace simaai::neat::pipeline_internal::sima
