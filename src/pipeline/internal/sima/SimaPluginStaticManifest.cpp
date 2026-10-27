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
#include <type_traits>
#include <unordered_map>

namespace simaai::neat::pipeline_internal::sima {
namespace {

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
  if (spec.frame_layout) {
    namespace client = ::simaai::scheduler::client;
    const auto span_json = [](const client::FrameSpan& span) {
      return nlohmann::json{{"allocation_index", span.allocation_index},
                            {"byte_offset", span.byte_offset},
                            {"size_bytes", span.size_bytes}};
    };
    const auto policy_name = [](client::AllocationPolicy policy) {
      switch (policy) {
      case client::AllocationPolicy::RequireCma:
        return "RequireCma";
      case client::AllocationPolicy::RequireDms:
        return "RequireDms";
      case client::AllocationPolicy::PreferDmsThenCma:
        return "PreferDmsThenCma";
      case client::AllocationPolicy::PreserveImported:
        return "PreserveImported";
      case client::AllocationPolicy::RequireHostCpu:
        return "RequireHostCpu";
      }
      return "Unknown";
    };
    const auto& layout = *spec.frame_layout;
    auto allocations = nlohmann::json::array();
    for (const auto& allocation : layout.allocations)
      allocations.push_back({{"size_bytes", allocation.size_bytes},
                             {"required_device_access", allocation.required_device_access},
                             {"policy", policy_name(allocation.policy)}});
    auto inputs = nlohmann::json::array();
    for (const auto& input : layout.inputs) {
      auto source = std::visit(
          [&](const auto& binding) {
            using Source = std::decay_t<decltype(binding)>;
            nlohmann::json result{{"pad", binding.pad}};
            if constexpr (std::is_same_v<Source, client::KnownFrameSpan>) {
              result["source"] = "KnownFrameSpan";
              result["span"] = span_json(binding.span);
            } else {
              result["byte_offset"] = binding.byte_offset;
              result["size_bytes"] = binding.size_bytes;
              if constexpr (std::is_same_v<Source, client::DirectMemory>) {
                result["source"] = "DirectMemory";
                result["memory_index"] = binding.memory_index;
              } else {
                result["source"] = "PhysicalBinding";
                result["binding_index"] = binding.binding_index;
              }
            }
            return result;
          },
          input.source);
      source["view_offset"] = input.view_offset;
      inputs.push_back(std::move(source));
    }
    auto outputs = nlohmann::json::array();
    for (const auto& output : layout.outputs)
      outputs.push_back({{"tensor", output.tensor}, {"span", span_json(output.span)}});
    j["frame_layout"] = {{"ownership", layout.ownership == client::FrameOwnership::Allocate
                                           ? "Allocate"
                                           : "ReuseInput"},
                         {"source_pad", layout.source_pad},
                         {"allocations", std::move(allocations)},
                         {"inputs", std::move(inputs)},
                         {"outputs", std::move(outputs)}};
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

} // namespace simaai::neat::pipeline_internal::sima
