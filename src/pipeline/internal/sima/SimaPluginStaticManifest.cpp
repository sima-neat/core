#include "pipeline/internal/sima/SimaPluginStaticManifest.h"

#include "gst/SimaPluginStaticManifestAbi.h"
#include "pipeline/internal/TensorMath.h"
#include "pipeline/internal/sima/BoxDecodeTypeUtils.h"
#include "pipeline/internal/sima/TensorSemanticsUtil.h"

#include <simaai/scheduler/client/stage.h>

#include <algorithm>
#include <cctype>
#include <cerrno>
#include <cstdio>
#include <cstring>
#include <cstdlib>
#include <memory>
#include <limits>
#include <sstream>
#include <stdexcept>
#include <unordered_map>

namespace simaai::neat::pipeline_internal::sima {
namespace {
static_assert(static_cast<int>(static_contract::ArenaStorageDomain::Cma) ==
              SIMA_PLUGIN_FRAME_ARENA_STORAGE_CMA);
static_assert(static_cast<int>(static_contract::ArenaStorageDomain::Dms) ==
              SIMA_PLUGIN_FRAME_ARENA_STORAGE_DMS);
static_assert(static_cast<int>(static_contract::ArenaStorageDomain::HostCpu) ==
              SIMA_PLUGIN_FRAME_ARENA_STORAGE_HOST_CPU);
static_assert(static_cast<int>(static_contract::ArenaStorageDomain::Mixed) ==
              SIMA_PLUGIN_FRAME_ARENA_STORAGE_MIXED);

std::string lower_copy(std::string s) {
  std::transform(s.begin(), s.end(), s.begin(),
                 [](unsigned char c) { return static_cast<char>(std::tolower(c)); });
  return s;
}

std::string trim_copy_local(const std::string& s) {
  return trim_copy(s);
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

bool read_u32_key(const nlohmann::json& obj, const char* key, std::uint32_t& out) {
  if (!obj.contains(key) || !obj[key].is_number_integer()) {
    return false;
  }
  std::uint64_t value = 0U;
  if (obj[key].is_number_unsigned()) {
    value = obj[key].get<std::uint64_t>();
  } else {
    const auto signed_value = obj[key].get<std::int64_t>();
    if (signed_value < 0) {
      return false;
    }
    value = static_cast<std::uint64_t>(signed_value);
  }
  if (value > std::numeric_limits<std::uint32_t>::max()) {
    return false;
  }
  out = static_cast<std::uint32_t>(value);
  return true;
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
  j["address_source"] = static_cast<int>(spec.address_source);
  j["size_bytes"] = spec.size_bytes;
  j["source_byte_offset"] = spec.source_byte_offset;
  j["device_kind"] = static_cast<int>(spec.device_kind);
  j["memory_flags"] = spec.memory_flags;
  j["segment_name_id"] = spec.segment_name_id;
  j["segment_name"] = spec.segment_name;
  j["required_alignment_bytes"] = spec.required_alignment_bytes;
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
  if (spec.image_publication) {
    const auto& image = *spec.image_publication;
    auto planes = nlohmann::json::array();
    for (const auto& plane : image.planes)
      planes.push_back({{"backend_output_index", plane.backend_output_index},
                        {"byte_offset", plane.byte_offset},
                        {"stride", plane.stride},
                        {"visible_rows", plane.visible_rows},
                        {"visible_row_bytes", plane.visible_row_bytes}});
    j["image_publication"] = {{"format", gst_video_format_to_string(image.format)},
                              {"width", image.width},
                              {"height", image.height},
                              {"planes", std::move(planes)}};
  }
  if (spec.frame_arena_size_bytes != 0U) {
    if (!spec.frame_arena_allocations.empty()) {
      auto parts = nlohmann::json::array();
      for (const auto& part : spec.frame_arena_allocations)
        parts.push_back(
            {{"byte_offset", part.byte_offset},
             {"size_bytes", part.size_bytes},
             {"storage_domain", static_cast<int>(part.placement.domain)},
             {"required_device_access", part.placement.required_device_access},
             {"allocation_policy", static_cast<int>(part.placement.allocation_policy)}});
      j["frame_arena_allocations"] = std::move(parts);
    }
    j["frame_arena_size_bytes"] = spec.frame_arena_size_bytes;
    j["frame_arena_role"] = static_cast<int>(spec.frame_arena_role);
    j["frame_arena_storage_domain"] = static_cast<int>(spec.frame_arena_storage_domain);
    j["frame_arena_provenance"] = static_cast<int>(spec.frame_arena_provenance);
    j["frame_arena_allocation_policy"] = static_cast<int>(spec.frame_arena_allocation_policy);
    j["frame_arena_required_device_access"] = spec.frame_arena_required_device_access;
    j["frame_arena_escape_policy"] = static_cast<int>(spec.frame_arena_escape_policy);
  }
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
  if (spec.payload_kind == StagePayloadKind::ProcessMla && spec.processmla.dmabuf_plan_contract) {
    j["processmla_dmabuf_plan_contract"] = true;
    j["processmla_executable_bytes"] = spec.processmla.executable_bytes;
    j["processmla_executable_sha256"] = spec.processmla.executable_sha256;
  }
  if (spec.payload_kind == StagePayloadKind::ProcessCvu && spec.processcvu.dmabuf_plan_contract) {
    j["processcvu_dmabuf_plan_contract"] = true;
    j["processcvu_descriptor_abi_id"] = spec.processcvu.descriptor_abi_id;
    j["processcvu_descriptor_contract_version"] = spec.processcvu.descriptor_contract_version;
    j["processcvu_binding_schema_version"] = spec.processcvu.binding_schema_version;
    j["processcvu_supported_placement_mask"] = spec.processcvu.supported_placement_mask;
    j["processcvu_allowed_frame_patch_mask"] = spec.processcvu.allowed_frame_patch_mask;
    j["processcvu_maximum_members"] = spec.processcvu.maximum_members;
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
  j["abi_version"] = SIMA_PLUGIN_STATIC_MANIFEST_ABI_VERSION;
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

  int abi_version = 30; // Unversioned historical JSON has strict placement only.
  if (root.contains("abi_version")) {
    const auto& version = root["abi_version"];
    if (!version.is_number_integer()) {
      if (error_message)
        *error_message = "manifest abi_version must be an integer";
      return std::nullopt;
    }
    if (version != 30 && version != SIMA_PLUGIN_STATIC_MANIFEST_ABI_VERSION) {
      if (error_message)
        *error_message = "unsupported manifest ABI version";
      return std::nullopt;
    }
    abi_version = version.get<int>();
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
    if (stage_j.contains("image_publication")) {
      try {
        // This is metadata only: no executable or numerical policy is restored.
        // Preserve the ordered physical-relative descriptor exactly. Admission
        // rechecks it against the independently checked numerical contract.
        const auto& value = stage_j.at("image_publication");
        const auto integer = [](const nlohmann::json& object, const char* key,
                                std::uint64_t maximum, bool positive) {
          const auto& number = object.at(key);
          if (!number.is_number_integer() ||
              (!number.is_number_unsigned() && number.get<std::int64_t>() < 0))
            throw std::invalid_argument("image publication integer is negative or non-integral");
          const auto result = number.get<std::uint64_t>();
          if (result > maximum || (positive && result == 0))
            throw std::invalid_argument("image publication integer is out of range");
          return result;
        };
        auto image = std::make_shared<::simaai::scheduler::client::TensorStageImagePublication>();
        image->format = gst_video_format_from_string(value.at("format").get<std::string>().c_str());
        if (image->format == GST_VIDEO_FORMAT_UNKNOWN || image->format == GST_VIDEO_FORMAT_ENCODED)
          throw std::invalid_argument("image publication requires a raw video format");
        image->width = integer(value, "width", std::numeric_limits<guint>::max(), true);
        image->height = integer(value, "height", std::numeric_limits<guint>::max(), true);
        const auto& planes = value.at("planes");
        if (!planes.is_array() || planes.empty() || planes.size() > GST_VIDEO_MAX_PLANES)
          throw std::invalid_argument("image publication requires bounded nonempty planes");
        for (const auto& entry : planes) {
          ::simaai::scheduler::client::TensorStageImagePlane plane;
          plane.backend_output_index =
              integer(entry, "backend_output_index", std::numeric_limits<int>::max(), false);
          plane.byte_offset =
              integer(entry, "byte_offset", std::numeric_limits<gsize>::max(), false);
          plane.stride = integer(entry, "stride", std::numeric_limits<gint>::max(), true);
          plane.visible_rows =
              integer(entry, "visible_rows", std::numeric_limits<guint>::max(), true);
          plane.visible_row_bytes =
              integer(entry, "visible_row_bytes", std::numeric_limits<guint>::max(), true);
          if (plane.visible_row_bytes > static_cast<guint>(plane.stride) ||
              std::any_of(image->planes.begin(), image->planes.end(), [&](const auto& prior) {
                return prior.backend_output_index == plane.backend_output_index;
              }))
            throw std::invalid_argument("image publication has an invalid or duplicate plane");
          image->planes.push_back(plane);
        }
        stage.image_publication = std::move(image);
      } catch (const std::exception& ex) {
        if (error_message)
          *error_message = std::string("manifest image_publication: ") + ex.what();
        return std::nullopt;
      }
    }
    if (stage_j.contains("frame_arena_size_bytes") &&
        stage_j["frame_arena_size_bytes"].is_number_unsigned()) {
      stage.frame_arena_size_bytes = stage_j["frame_arena_size_bytes"].get<std::uint64_t>();
    } else if (stage_j.contains("frame_arena_size_bytes") &&
               stage_j["frame_arena_size_bytes"].is_number_integer()) {
      const auto raw = stage_j["frame_arena_size_bytes"].get<std::int64_t>();
      stage.frame_arena_size_bytes = raw > 0 ? static_cast<std::uint64_t>(raw) : 0U;
    }
    int frame_arena_role = static_cast<int>(FrameArenaRole::None);
    read_int_key(stage_j, "frame_arena_role", frame_arena_role);
    if (frame_arena_role >= static_cast<int>(FrameArenaRole::None) &&
        frame_arena_role <= static_cast<int>(FrameArenaRole::ReuseInput)) {
      stage.frame_arena_role = static_cast<FrameArenaRole>(frame_arena_role);
    }
    int frame_arena_storage_domain = static_cast<int>(static_contract::ArenaStorageDomain::Unknown);
    int frame_arena_provenance =
        static_cast<int>(static_contract::ArenaAllocationProvenance::Unknown);
    int frame_arena_allocation_policy =
        static_cast<int>(static_contract::ArenaAllocationPolicy::RequiredDomain);
    if (stage_j.contains("frame_arena_allocation_policy")) {
      if (abi_version < 31 || !stage_j["frame_arena_allocation_policy"].is_number_integer() ||
          (stage_j["frame_arena_allocation_policy"] != 0 &&
           stage_j["frame_arena_allocation_policy"] != 1)) {
        if (error_message)
          *error_message = "allocation policy requires current manifest ABI";
        return std::nullopt;
      }
      try {
        frame_arena_allocation_policy = stage_j["frame_arena_allocation_policy"].get<int>();
      } catch (const std::exception&) {
        if (error_message)
          *error_message = "manifest allocation policy is out of range";
        return std::nullopt;
      }
    } else if (abi_version >= 31 && stage.frame_arena_size_bytes != 0U) {
      if (error_message)
        *error_message = "manifest frame arena is missing allocation policy";
      return std::nullopt;
    }
    const bool required_domain =
        frame_arena_allocation_policy ==
        static_cast<int>(static_contract::ArenaAllocationPolicy::RequiredDomain);
    const bool preferred_dms =
        frame_arena_allocation_policy ==
        static_cast<int>(static_contract::ArenaAllocationPolicy::PreferDmsThenCma);
    int frame_arena_required_device_access = 0;
    int frame_arena_escape_policy =
        static_cast<int>(static_contract::ArenaEscapePolicy::InternalOnly);
    read_int_key(stage_j, "frame_arena_storage_domain", frame_arena_storage_domain);
    read_int_key(stage_j, "frame_arena_provenance", frame_arena_provenance);
    read_int_key(stage_j, "frame_arena_required_device_access", frame_arena_required_device_access);
    read_int_key(stage_j, "frame_arena_escape_policy", frame_arena_escape_policy);
    constexpr int kKnownDeviceAccess =
        static_cast<int>(static_contract::ArenaDeviceAccess::CpuA65) |
        static_cast<int>(static_contract::ArenaDeviceAccess::Mla) |
        static_cast<int>(static_contract::ArenaDeviceAccess::Ev74);
    const bool known_domain =
        frame_arena_storage_domain == static_cast<int>(static_contract::ArenaStorageDomain::Cma) ||
        frame_arena_storage_domain == static_cast<int>(static_contract::ArenaStorageDomain::Dms) ||
        (abi_version == SIMA_PLUGIN_STATIC_MANIFEST_ABI_VERSION &&
         (frame_arena_storage_domain ==
              static_cast<int>(static_contract::ArenaStorageDomain::HostCpu) ||
          frame_arena_storage_domain ==
              static_cast<int>(static_contract::ArenaStorageDomain::Mixed)));
    const bool external_unknown_domain =
        frame_arena_role == static_cast<int>(FrameArenaRole::ReuseInput) &&
        frame_arena_provenance ==
            static_cast<int>(static_contract::ArenaAllocationProvenance::ExternalAdopted) &&
        frame_arena_storage_domain ==
            static_cast<int>(static_contract::ArenaStorageDomain::Unknown);
    const bool provenance_role_valid =
        (frame_arena_provenance ==
             static_cast<int>(static_contract::ArenaAllocationProvenance::CoreAllocated) &&
         known_domain) ||
        (frame_arena_provenance ==
             static_cast<int>(static_contract::ArenaAllocationProvenance::ExternalAdopted) &&
         frame_arena_role == static_cast<int>(FrameArenaRole::ReuseInput));
    const bool allocation_policy_valid =
        required_domain ||
        (preferred_dms && stage.frame_arena_size_bytes != 0U && known_domain &&
         frame_arena_storage_domain == static_cast<int>(static_contract::ArenaStorageDomain::Dms) &&
         frame_arena_provenance ==
             static_cast<int>(static_contract::ArenaAllocationProvenance::CoreAllocated) &&
         frame_arena_required_device_access > 0 &&
         !(frame_arena_required_device_access &
           static_cast<int>(static_contract::ArenaDeviceAccess::Ev74)));
    const bool host_domain = frame_arena_storage_domain ==
                             static_cast<int>(static_contract::ArenaStorageDomain::HostCpu);
    const bool host_escape = frame_arena_escape_policy ==
                             static_cast<int>(static_contract::ArenaEscapePolicy::CpuOnlyPublic);
    const bool host_valid =
        (!host_domain && !host_escape) ||
        (host_domain && host_escape && abi_version == SIMA_PLUGIN_STATIC_MANIFEST_ABI_VERSION &&
         required_domain && frame_arena_role == static_cast<int>(FrameArenaRole::Allocate) &&
         frame_arena_provenance ==
             static_cast<int>(static_contract::ArenaAllocationProvenance::CoreAllocated) &&
         frame_arena_required_device_access ==
             static_cast<int>(static_contract::ArenaDeviceAccess::CpuA65));
    const bool arena_placement_valid =
        host_valid && allocation_policy_valid &&
        (stage.frame_arena_size_bytes == 0U ||
         (frame_arena_role >= static_cast<int>(FrameArenaRole::Allocate) &&
          frame_arena_role <= static_cast<int>(FrameArenaRole::ReuseInput) &&
          (known_domain || external_unknown_domain) && provenance_role_valid &&
          frame_arena_required_device_access > 0 &&
          (frame_arena_required_device_access & ~kKnownDeviceAccess) == 0 &&
          frame_arena_escape_policy >=
              static_cast<int>(static_contract::ArenaEscapePolicy::InternalOnly) &&
          frame_arena_escape_policy <=
              static_cast<int>(static_contract::ArenaEscapePolicy::CpuOnlyPublic) &&
          !(frame_arena_storage_domain ==
                static_cast<int>(static_contract::ArenaStorageDomain::Dms) &&
            (frame_arena_required_device_access &
             static_cast<int>(static_contract::ArenaDeviceAccess::Ev74)) != 0)));
    if (!arena_placement_valid) {
      if (error_message) {
        *error_message = "manifest stage has invalid frame-arena placement";
      }
      return std::nullopt;
    }
    stage.frame_arena_allocation_policy =
        static_cast<static_contract::ArenaAllocationPolicy>(frame_arena_allocation_policy);
    stage.frame_arena_storage_domain =
        static_cast<static_contract::ArenaStorageDomain>(frame_arena_storage_domain);
    stage.frame_arena_provenance =
        static_cast<static_contract::ArenaAllocationProvenance>(frame_arena_provenance);
    stage.frame_arena_required_device_access =
        static_cast<std::uint32_t>(frame_arena_required_device_access);
    stage.frame_arena_escape_policy =
        static_cast<static_contract::ArenaEscapePolicy>(frame_arena_escape_policy);
    if (stage_j.contains("frame_arena_allocations")) {
      try {
        const auto& parts = stage_j.at("frame_arena_allocations");
        if (abi_version < 32 || !parts.is_array() || parts.empty())
          throw std::invalid_argument("partitioned arenas require ABI 32");
        std::vector<SimaPluginFrameArenaAllocation> abi_parts;
        for (const auto& part : parts) {
          const auto unsigned_value = [&](const char* key) {
            const auto& value = part.at(key);
            if (!value.is_number_integer() ||
                (!value.is_number_unsigned() && value.get<std::int64_t>() < 0))
              throw std::invalid_argument("arena allocation fields must be nonnegative integers");
            return value.get<std::uint64_t>();
          };
          const auto offset = unsigned_value("byte_offset");
          const auto bytes = unsigned_value("size_bytes");
          const auto domain = unsigned_value("storage_domain");
          const auto access = unsigned_value("required_device_access");
          const auto policy = unsigned_value("allocation_policy");
          constexpr std::uint64_t known_access = SIMA_PLUGIN_FRAME_ARENA_ACCESS_CPU_A65 |
                                                 SIMA_PLUGIN_FRAME_ARENA_ACCESS_MLA |
                                                 SIMA_PLUGIN_FRAME_ARENA_ACCESS_EV74;
          if ((domain != SIMA_PLUGIN_FRAME_ARENA_STORAGE_CMA &&
               domain != SIMA_PLUGIN_FRAME_ARENA_STORAGE_DMS &&
               !(host_domain && domain == SIMA_PLUGIN_FRAME_ARENA_STORAGE_HOST_CPU)) ||
              access == 0 || (access & ~known_access) || policy > 1 ||
              (policy == static_cast<unsigned>(
                             static_contract::ArenaAllocationPolicy::PreferDmsThenCma) &&
               (domain != SIMA_PLUGIN_FRAME_ARENA_STORAGE_DMS ||
                (access & SIMA_PLUGIN_FRAME_ARENA_ACCESS_EV74))))
            throw std::invalid_argument("invalid allocation domain, access mask or policy");
          abi_parts.push_back({offset, bytes,
                               static_cast<SimaPluginFrameArenaStorageDomain>(domain),
                               static_cast<guint>(access)});
          static_contract::ArenaPlacement placement;
          placement.domain = static_cast<static_contract::ArenaStorageDomain>(domain);
          placement.provenance = stage.frame_arena_provenance;
          placement.required_device_access = static_cast<std::uint32_t>(access);
          placement.allocation_policy = static_cast<static_contract::ArenaAllocationPolicy>(policy);
          placement.escape = stage.frame_arena_escape_policy;
          stage.frame_arena_allocations.push_back({offset, bytes, placement});
        }
        SimaPluginStageSpec probe{};
        probe.frame_arena_size_bytes = stage.frame_arena_size_bytes;
        probe.frame_arena_storage_domain =
            static_cast<SimaPluginFrameArenaStorageDomain>(stage.frame_arena_storage_domain);
        probe.frame_arena_provenance =
            static_cast<SimaPluginFrameArenaProvenance>(stage.frame_arena_provenance);
        probe.frame_arena_allocation_policy =
            static_cast<SimaPluginFrameArenaAllocationPolicy>(stage.frame_arena_allocation_policy);
        probe.frame_arena_required_device_access = stage.frame_arena_required_device_access;
        probe.frame_arena_allocations = abi_parts.data();
        probe.frame_arena_allocations_len = abi_parts.size();
        const bool host_parts_valid =
            host_domain && abi_parts.size() == 1 && abi_parts[0].byte_offset == 0 &&
            abi_parts[0].size_bytes == stage.frame_arena_size_bytes &&
            abi_parts[0].storage_domain == SIMA_PLUGIN_FRAME_ARENA_STORAGE_HOST_CPU &&
            abi_parts[0].required_device_access == SIMA_PLUGIN_FRAME_ARENA_ACCESS_CPU_A65 &&
            stage.frame_arena_allocations[0].placement.allocation_policy ==
                static_contract::ArenaAllocationPolicy::RequiredDomain;
        if (host_domain ? !host_parts_valid : !sima_plugin_frame_arena_allocations_valid(&probe))
          throw std::invalid_argument("invalid allocation coverage or reachability");
      } catch (const std::exception& ex) {
        if (error_message)
          *error_message = std::string("manifest frame-arena allocations: ") + ex.what();
        return std::nullopt;
      }
    } else if (stage.frame_arena_storage_domain == static_contract::ArenaStorageDomain::Mixed) {
      if (error_message)
        *error_message = "partitioned arena is missing allocations";
      return std::nullopt;
    }
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
        int address_source = 0;
        read_int_key(in_j, "address_source", address_source);
        physical.address_source = static_cast<PhysicalAddressSource>(address_source);
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
        if (in_j.contains("required_alignment_bytes") &&
            in_j["required_alignment_bytes"].is_number_unsigned()) {
          physical.required_alignment_bytes = in_j["required_alignment_bytes"].get<std::uint64_t>();
        } else if (in_j.contains("required_alignment_bytes") &&
                   in_j["required_alignment_bytes"].is_number_integer()) {
          const auto raw = in_j["required_alignment_bytes"].get<std::int64_t>();
          physical.required_alignment_bytes = raw > 0 ? static_cast<std::uint64_t>(raw) : 0U;
        }
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
        int address_source = 0;
        read_int_key(out_j, "address_source", address_source);
        physical.address_source = static_cast<PhysicalAddressSource>(address_source);
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
        if (out_j.contains("required_alignment_bytes") &&
            out_j["required_alignment_bytes"].is_number_unsigned()) {
          physical.required_alignment_bytes =
              out_j["required_alignment_bytes"].get<std::uint64_t>();
        } else if (out_j.contains("required_alignment_bytes") &&
                   out_j["required_alignment_bytes"].is_number_integer()) {
          const auto raw = out_j["required_alignment_bytes"].get<std::int64_t>();
          physical.required_alignment_bytes = raw > 0 ? static_cast<std::uint64_t>(raw) : 0U;
        }
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
    read_bool_key(stage_j, "processmla_dmabuf_plan_contract",
                  stage.processmla.dmabuf_plan_contract);
    if (stage_j.contains("processmla_executable_bytes") &&
        stage_j["processmla_executable_bytes"].is_number_unsigned()) {
      stage.processmla.executable_bytes =
          stage_j["processmla_executable_bytes"].get<std::uint64_t>();
    }
    read_string_key(stage_j, "processmla_executable_sha256", stage.processmla.executable_sha256);
    read_bool_key(stage_j, "processcvu_dmabuf_plan_contract",
                  stage.processcvu.dmabuf_plan_contract);
    read_u32_key(stage_j, "processcvu_descriptor_abi_id", stage.processcvu.descriptor_abi_id);
    read_u32_key(stage_j, "processcvu_descriptor_contract_version",
                 stage.processcvu.descriptor_contract_version);
    read_u32_key(stage_j, "processcvu_binding_schema_version",
                 stage.processcvu.binding_schema_version);
    read_u32_key(stage_j, "processcvu_supported_placement_mask",
                 stage.processcvu.supported_placement_mask);
    read_u32_key(stage_j, "processcvu_allowed_frame_patch_mask",
                 stage.processcvu.allowed_frame_patch_mask);
    read_u32_key(stage_j, "processcvu_maximum_members", stage.processcvu.maximum_members);
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

} // namespace simaai::neat::pipeline_internal::sima
