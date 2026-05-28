#include "pipeline/internal/TerminalOutputContractQuery.h"

#include "pipeline/internal/TensorMath.h"

#include <algorithm>
#include <cctype>
#include <cstdint>
#include <limits>
#include <sstream>
#include <string_view>

namespace simaai::neat::pipeline_internal::terminal_output_contract {
namespace {

using sima::LogicalTensorStaticSpec;
using sima::PhysicalBufferStaticSpec;
using sima::ProcessCvuGraphFamily;
using sima::StagePayloadKind;
using sima::StageStaticSpec;

std::string upper_copy(std::string value) {
  std::transform(value.begin(), value.end(), value.begin(),
                 [](unsigned char c) { return static_cast<char>(std::toupper(c)); });
  return value;
}

std::string lower_copy(std::string value) {
  std::transform(value.begin(), value.end(), value.begin(),
                 [](unsigned char c) { return static_cast<char>(std::tolower(c)); });
  return value;
}

bool contains_token(std::string_view haystack, std::string_view needle) {
  if (haystack.empty() || needle.empty()) {
    return false;
  }
  std::string lower_haystack(haystack);
  std::string lower_needle(needle);
  lower_haystack = lower_copy(std::move(lower_haystack));
  lower_needle = lower_copy(std::move(lower_needle));
  return lower_haystack.find(lower_needle) != std::string::npos;
}

bool is_boundary_view_token(std::string_view token) {
  return contains_token(token, "slice") || contains_token(token, "batchflatten") ||
         contains_token(token, "batch_flatten") || contains_token(token, "offset_view") ||
         contains_token(token, "offsetview") || contains_token(token, "view_transform") ||
         contains_token(token, "passthrough_view");
}

bool stage_identity_matches(const StageStaticSpec& stage, const std::string& key) {
  if (key.empty()) {
    return false;
  }
  const std::string lower_key = lower_copy(key);
  return stage.logical_stage_id == key || stage.element_name == key || stage.plugin_kind == key ||
         stage.kernel_kind == key || lower_copy(stage.logical_stage_id) == lower_key ||
         lower_copy(stage.element_name) == lower_key || lower_copy(stage.plugin_kind) == lower_key ||
         lower_copy(stage.kernel_kind) == lower_key;
}

bool stage_has_publishable_outputs(const StageStaticSpec& stage) {
  return !stage.physical_outputs.empty() || !stage.logical_outputs.empty() ||
         !stage.processmla.dispatcher_output_sizes.empty();
}

std::optional<TensorDType> tensor_dtype_from_static_token(std::string token) {
  token = upper_copy(std::move(token));
  constexpr const char* kPrefix = "EVXX_";
  if (token.rfind(kPrefix, 0U) == 0U) {
    token.erase(0U, std::char_traits<char>::length(kPrefix));
  }
  if (token.find("BFLOAT16") != std::string::npos || token.find("BF16") != std::string::npos) {
    return TensorDType::BFloat16;
  }
  if (token.find("FLOAT64") != std::string::npos || token.find("FP64") != std::string::npos ||
      token == "F64") {
    return TensorDType::Float64;
  }
  if (token.find("FLOAT32") != std::string::npos || token.find("FP32") != std::string::npos ||
      token == "F32") {
    return TensorDType::Float32;
  }
  if (token.find("UINT16") != std::string::npos || token == "U16") {
    return TensorDType::UInt16;
  }
  if (token.find("INT16") != std::string::npos || token == "S16") {
    return TensorDType::Int16;
  }
  if (token.find("INT32") != std::string::npos || token == "S32") {
    return TensorDType::Int32;
  }
  if (token.find("UINT8") != std::string::npos || token == "U8") {
    return TensorDType::UInt8;
  }
  if (token.find("INT8") != std::string::npos || token == "S8") {
    return TensorDType::Int8;
  }
  return std::nullopt;
}

TensorLayout tensor_layout_from_static_token(std::string token) {
  token = upper_copy(std::move(token));
  if (token == "HW") {
    return TensorLayout::HW;
  }
  if (token == "HWC" || token == "NHWC") {
    return TensorLayout::HWC;
  }
  if (token == "CHW" || token == "NCHW") {
    return TensorLayout::CHW;
  }
  return TensorLayout::Unknown;
}

std::uint64_t logical_span_bytes(const LogicalTensorStaticSpec& logical, std::size_t elem_bytes) {
  if (elem_bytes == 0U) {
    return 0U;
  }
  if (!logical.shape.empty() && logical.stride_bytes.size() == logical.shape.size()) {
    std::uint64_t max_offset = 0U;
    for (std::size_t i = 0; i < logical.shape.size(); ++i) {
      if (logical.shape[i] <= 0 || logical.stride_bytes[i] < 0) {
        return 0U;
      }
      if (logical.shape[i] == 1) {
        continue;
      }
      const auto dim = static_cast<std::uint64_t>(logical.shape[i] - 1);
      const auto stride = static_cast<std::uint64_t>(logical.stride_bytes[i]);
      if (dim != 0U && stride > std::numeric_limits<std::uint64_t>::max() / dim) {
        return 0U;
      }
      const std::uint64_t delta = dim * stride;
      if (max_offset > std::numeric_limits<std::uint64_t>::max() - delta) {
        return 0U;
      }
      max_offset += delta;
    }
    if (max_offset > std::numeric_limits<std::uint64_t>::max() - elem_bytes) {
      return 0U;
    }
    return max_offset + elem_bytes;
  }

  if (!logical.shape.empty()) {
    std::uint64_t elements = 1U;
    for (const auto dim : logical.shape) {
      if (dim <= 0) {
        return 0U;
      }
      const auto u_dim = static_cast<std::uint64_t>(dim);
      if (elements > std::numeric_limits<std::uint64_t>::max() / u_dim) {
        return 0U;
      }
      elements *= u_dim;
    }
    if (elements > std::numeric_limits<std::uint64_t>::max() / elem_bytes) {
      return 0U;
    }
    return elements * elem_bytes;
  }

  return logical.size_bytes;
}

const PhysicalBufferStaticSpec* physical_for_logical(const StageStaticSpec& stage,
                                                     const LogicalTensorStaticSpec& logical,
                                                     std::size_t* vector_index = nullptr) {
  if (logical.physical_index >= 0) {
    for (std::size_t i = 0; i < stage.physical_outputs.size(); ++i) {
      if (stage.physical_outputs[i].physical_index == logical.physical_index) {
        if (vector_index) {
          *vector_index = i;
        }
        return &stage.physical_outputs[i];
      }
    }
    const auto requested = static_cast<std::size_t>(logical.physical_index);
    if (requested < stage.physical_outputs.size() && stage.physical_outputs[requested].physical_index < 0) {
      if (vector_index) {
        *vector_index = requested;
      }
      return &stage.physical_outputs[requested];
    }
  }
  if (stage.physical_outputs.size() == 1U) {
    if (vector_index) {
      *vector_index = 0U;
    }
    return &stage.physical_outputs.front();
  }
  return nullptr;
}

bool logical_name_looks_like_boundary_view(const LogicalTensorStaticSpec& logical) {
  return is_boundary_view_token(logical.logical_name) || is_boundary_view_token(logical.backend_name) ||
         is_boundary_view_token(logical.segment_name);
}

bool logical_outputs_are_producer_local(const StageStaticSpec& stage) {
  if (stage.logical_outputs.empty() || stage.physical_outputs.empty()) {
    return false;
  }
  for (const auto& logical : stage.logical_outputs) {
    if (!sima::dtype_source_is_public_logical_contract_authoritative(logical.dtype_source)) {
      return false;
    }
    const auto dtype = tensor_dtype_from_static_token(logical.dtype);
    if (!dtype.has_value()) {
      return false;
    }
    std::size_t physical_index = 0U;
    const auto* physical = physical_for_logical(stage, logical, &physical_index);
    if (!physical || physical->size_bytes == 0U) {
      return false;
    }
    const std::size_t elem_bytes = dtype_bytes(*dtype);
    const std::uint64_t span = std::max<std::uint64_t>(logical_span_bytes(logical, elem_bytes),
                                                       logical.size_bytes);
    if (span == 0U || logical.byte_offset < 0) {
      return false;
    }
    const auto offset = static_cast<std::uint64_t>(logical.byte_offset);
    if (offset > physical->size_bytes || span > physical->size_bytes - offset) {
      return false;
    }

    // A boundary/view name on a non-view terminal producer is evidence that a
    // downstream edge contract leaked into public publication.  Do not use it
    // as the user-visible terminal contract; synthesize the raw producer bytes
    // instead.
    if (logical_name_looks_like_boundary_view(logical) &&
        classify_stage_for_publication(stage) != StagePublicationRole::BoundaryView) {
      return false;
    }

    (void)physical_index;
  }
  return true;
}

std::string fallback_physical_name(const StageStaticSpec& stage, std::size_t index) {
  const auto usable_public_name = [](std::string_view name) {
    return !name.empty() && !is_boundary_view_token(name);
  };
  const std::size_t output_count = !stage.processmla.dispatcher_output_sizes.empty()
                                       ? stage.processmla.dispatcher_output_sizes.size()
                                       : stage.physical_outputs.size();

  if (index < stage.physical_outputs.size() &&
      usable_public_name(stage.physical_outputs[index].segment_name)) {
    return stage.physical_outputs[index].segment_name;
  }
  if (index < stage.processmla.dispatcher_output_names.size() &&
      usable_public_name(stage.processmla.dispatcher_output_names[index])) {
    return stage.processmla.dispatcher_output_names[index];
  }
  if (output_count <= 1U && usable_public_name(stage.logical_stage_id)) {
    return stage.logical_stage_id;
  }
  if (usable_public_name(stage.logical_stage_id)) {
    return stage.logical_stage_id + "/output_" + std::to_string(index);
  }
  if (output_count <= 1U && usable_public_name(stage.element_name)) {
    return stage.element_name;
  }
  if (usable_public_name(stage.element_name)) {
    return stage.element_name + "/output_" + std::to_string(index);
  }
  return "output" + std::to_string(index);
}

std::vector<PhysicalBufferStaticSpec> raw_physical_outputs_for_terminal(const StageStaticSpec& stage) {
  std::vector<PhysicalBufferStaticSpec> out;
  if (!stage.processmla.dispatcher_output_sizes.empty()) {
    out.reserve(stage.processmla.dispatcher_output_sizes.size());
    for (std::size_t i = 0; i < stage.processmla.dispatcher_output_sizes.size(); ++i) {
      PhysicalBufferStaticSpec physical;
      physical.physical_index = static_cast<int>(i);
      physical.allocator_index = static_cast<int>(i);
      physical.source_physical_index = static_cast<int>(i);
      physical.source_byte_offset = 0;
      physical.device_kind = sima::DeviceKind::Mla;
      physical.size_bytes = stage.processmla.dispatcher_output_sizes[i];
      physical.segment_name = fallback_physical_name(stage, i);
      out.push_back(std::move(physical));
    }
    return out;
  }

  out = stage.physical_outputs;
  for (std::size_t i = 0; i < out.size(); ++i) {
    if (out[i].physical_index < 0) {
      out[i].physical_index = static_cast<int>(i);
    }
    if (out[i].allocator_index < 0) {
      out[i].allocator_index = static_cast<int>(i);
    }
    if (out[i].source_physical_index < 0) {
      out[i].source_physical_index = out[i].physical_index;
    }
    if (out[i].segment_name.empty()) {
      out[i].segment_name = fallback_physical_name(stage, i);
    }
  }
  return out;
}

std::vector<LogicalTensorStaticSpec>
raw_logical_outputs_from_physical(const std::vector<PhysicalBufferStaticSpec>& physical_outputs) {
  std::vector<LogicalTensorStaticSpec> out;
  out.reserve(physical_outputs.size());
  for (std::size_t i = 0; i < physical_outputs.size(); ++i) {
    const auto& physical = physical_outputs[i];
    if (physical.size_bytes == 0U ||
        physical.size_bytes > static_cast<std::uint64_t>(std::numeric_limits<std::int64_t>::max())) {
      continue;
    }
    LogicalTensorStaticSpec logical;
    logical.logical_index = static_cast<int>(i);
    logical.backend_output_index = static_cast<int>(i);
    logical.physical_index = physical.physical_index >= 0 ? physical.physical_index : static_cast<int>(i);
    logical.output_slot = static_cast<int>(i);
    logical.tensor_index = static_cast<int>(i);
    logical.byte_offset = 0;
    logical.size_bytes = physical.size_bytes;
    logical.shape = {static_cast<std::int64_t>(physical.size_bytes)};
    logical.stride_bytes = {1};
    logical.dtype = "UINT8";
    logical.dtype_source = sima::DTypeSource::InternalContract;
    logical.layout.clear();
    logical.logical_name = !physical.segment_name.empty() ? physical.segment_name
                                                         : ("output" + std::to_string(i));
    logical.backend_name = logical.logical_name;
    logical.segment_name = logical.logical_name;
    out.push_back(std::move(logical));
  }
  return out;
}

bool logical_to_override_entry(const StageStaticSpec& stage, const LogicalTensorStaticSpec& logical,
                               OutputTensorOverrideEntry* entry, std::string* error_message) {
  if (!entry) {
    return false;
  }
  const auto dtype = tensor_dtype_from_static_token(logical.dtype);
  if (!dtype.has_value()) {
    if (error_message) {
      *error_message = "unsupported or unknown terminal output dtype '" + logical.dtype + "'";
    }
    return false;
  }
  std::size_t memory_index = 0U;
  const auto* physical = physical_for_logical(stage, logical, &memory_index);
  if (!physical) {
    if (error_message) {
      *error_message = "terminal output logical tensor has no matching physical buffer";
    }
    return false;
  }
  (void)physical;

  entry->shape = logical.shape;
  entry->strides_bytes = logical.stride_bytes;
  entry->byte_offset = logical.byte_offset;
  entry->memory_index = static_cast<int>(memory_index);
  entry->logical_output_index = logical.logical_index >= 0 ? logical.logical_index : 0;
  entry->route_slot = logical.output_slot >= 0 ? logical.output_slot : entry->logical_output_index;
  entry->dtype = *dtype;
  entry->layout = tensor_layout_from_static_token(logical.layout);
  entry->name = !logical.logical_name.empty() ? logical.logical_name
                                              : (!logical.backend_name.empty() ? logical.backend_name
                                                                              : "output");
  entry->segment_name = !logical.segment_name.empty() ? logical.segment_name : entry->name;
  return true;
}

bool endpoint_has_output_selector(const PublicOutputEndpointSelector& endpoint) {
  return !endpoint.output_segment_name.empty() || endpoint.output_slot >= 0 ||
         endpoint.route_slot >= 0;
}

bool string_matches_endpoint_name(std::string_view value,
                                  const PublicOutputEndpointSelector& endpoint) {
  return !value.empty() && !endpoint.output_segment_name.empty() &&
         value == endpoint.output_segment_name;
}

bool route_matches_endpoint(const sima::StageOutputRoute& route,
                            const PublicOutputEndpointSelector& endpoint) {
  if (!endpoint_has_output_selector(endpoint)) {
    return true;
  }
  if (endpoint.output_slot >= 0 && route.output_slot == endpoint.output_slot) {
    return true;
  }
  if (endpoint.route_slot >= 0 &&
      (route.output_slot == endpoint.route_slot ||
       route.logical_output_index == endpoint.route_slot ||
       route.tensor_index == endpoint.route_slot)) {
    return true;
  }
  return string_matches_endpoint_name(route.cm_output_name, endpoint) ||
         string_matches_endpoint_name(route.segment_name, endpoint);
}

bool logical_matches_endpoint(const LogicalTensorStaticSpec& logical,
                              const PublicOutputEndpointSelector& endpoint) {
  if (!endpoint_has_output_selector(endpoint)) {
    return true;
  }
  if (endpoint.output_slot >= 0 && logical.output_slot == endpoint.output_slot) {
    return true;
  }
  if (endpoint.route_slot >= 0 &&
      (logical.output_slot == endpoint.route_slot ||
       logical.logical_index == endpoint.route_slot ||
       logical.tensor_index == endpoint.route_slot)) {
    return true;
  }
  return string_matches_endpoint_name(logical.logical_name, endpoint) ||
         string_matches_endpoint_name(logical.backend_name, endpoint) ||
         string_matches_endpoint_name(logical.segment_name, endpoint);
}

bool physical_matches_endpoint(const PhysicalBufferStaticSpec& physical,
                               const PublicOutputEndpointSelector& endpoint) {
  if (!endpoint_has_output_selector(endpoint)) {
    return true;
  }
  if (endpoint.output_slot >= 0 && physical.physical_index == endpoint.output_slot) {
    return true;
  }
  if (endpoint.route_slot >= 0 && physical.physical_index == endpoint.route_slot) {
    return true;
  }
  return string_matches_endpoint_name(physical.segment_name, endpoint);
}

bool stage_outputs_match_endpoint(const StageStaticSpec& stage,
                                  const PublicOutputEndpointSelector& endpoint) {
  if (!endpoint_has_output_selector(endpoint)) {
    return true;
  }
  for (const auto& route : stage.output_order) {
    if (route_matches_endpoint(route, endpoint)) {
      return true;
    }
  }
  for (const auto& logical : stage.logical_outputs) {
    if (logical_matches_endpoint(logical, endpoint)) {
      return true;
    }
  }
  for (const auto& physical : stage.physical_outputs) {
    if (physical_matches_endpoint(physical, endpoint)) {
      return true;
    }
  }
  return false;
}

bool filter_publication_stage_for_endpoint(StageStaticSpec* publication,
                                           const PublicOutputEndpointSelector& endpoint,
                                           std::string* error_message) {
  if (!publication || !endpoint_has_output_selector(endpoint)) {
    return true;
  }

  std::vector<LogicalTensorStaticSpec> selected_logical;
  selected_logical.reserve(publication->logical_outputs.size());
  for (const auto& logical : publication->logical_outputs) {
    if (logical_matches_endpoint(logical, endpoint)) {
      selected_logical.push_back(logical);
    }
  }

  if (selected_logical.empty() && !publication->output_order.empty()) {
    for (const auto& route : publication->output_order) {
      if (!route_matches_endpoint(route, endpoint)) {
        continue;
      }
      for (const auto& logical : publication->logical_outputs) {
        const bool logical_match =
            (route.logical_output_index >= 0 && logical.logical_index == route.logical_output_index) ||
            (route.output_slot >= 0 && logical.output_slot == route.output_slot) ||
            (route.tensor_index >= 0 && logical.tensor_index == route.tensor_index) ||
            (!route.cm_output_name.empty() &&
             (logical.logical_name == route.cm_output_name ||
              logical.backend_name == route.cm_output_name)) ||
            (!route.segment_name.empty() && logical.segment_name == route.segment_name);
        if (logical_match) {
          selected_logical.push_back(logical);
        }
      }
    }
  }

  if (selected_logical.empty()) {
    if (error_message) {
      *error_message = "terminal publication endpoint selector did not match any logical output";
    }
    return false;
  }

  publication->logical_outputs = std::move(selected_logical);
  std::vector<sima::StageOutputRoute> selected_routes;
  selected_routes.reserve(publication->output_order.size());
  for (const auto& route : publication->output_order) {
    if (route_matches_endpoint(route, endpoint)) {
      selected_routes.push_back(route);
    }
  }
  if (selected_routes.empty()) {
    selected_routes.reserve(publication->logical_outputs.size());
    for (const auto& logical : publication->logical_outputs) {
      sima::StageOutputRoute route;
      route.output_slot = logical.output_slot;
      route.logical_output_index = logical.logical_index;
      route.tensor_index = logical.tensor_index;
      route.cm_output_name = logical.logical_name;
      route.segment_name = logical.segment_name;
      selected_routes.push_back(std::move(route));
    }
  }
  publication->output_order = std::move(selected_routes);
  return true;
}

} // namespace

StagePublicationRole classify_stage_for_publication(const sima::StageStaticSpec& stage) {
  if (!stage_has_publishable_outputs(stage)) {
    return StagePublicationRole::TransportOnly;
  }

  if (is_boundary_view_token(stage.kernel_kind) || is_boundary_view_token(stage.plugin_kind) ||
      is_boundary_view_token(stage.logical_stage_id) || is_boundary_view_token(stage.element_name)) {
    return StagePublicationRole::BoundaryView;
  }

  switch (stage.payload_kind) {
  case StagePayloadKind::ProcessMla:
  case StagePayloadKind::BoxDecode:
    return StagePublicationRole::RealProducer;
  case StagePayloadKind::DetessDequant:
  case StagePayloadKind::Quant:
  case StagePayloadKind::Tess:
  case StagePayloadKind::Dequant:
  case StagePayloadKind::QuantTess:
    return StagePublicationRole::MaterializedTransform;
  case StagePayloadKind::ProcessCvu:
    switch (stage.processcvu.graph_family_enum) {
    case ProcessCvuGraphFamily::Preproc:
    case ProcessCvuGraphFamily::Quant:
    case ProcessCvuGraphFamily::Tess:
    case ProcessCvuGraphFamily::QuantTess:
    case ProcessCvuGraphFamily::CastTess:
    case ProcessCvuGraphFamily::Detess:
    case ProcessCvuGraphFamily::Dequant:
    case ProcessCvuGraphFamily::DetessCast:
    case ProcessCvuGraphFamily::DetessDequant:
    case ProcessCvuGraphFamily::Cast:
      return StagePublicationRole::MaterializedTransform;
    case ProcessCvuGraphFamily::Unknown:
      return StagePublicationRole::RealProducer;
    }
    break;
  case StagePayloadKind::None:
    break;
  }

  return stage_has_publishable_outputs(stage) ? StagePublicationRole::RealProducer
                                              : StagePublicationRole::TransportOnly;
}

const sima::StageStaticSpec* find_terminal_real_producer_for_endpoint(
    const sima::SimaPluginStaticManifest& manifest, const PublicOutputEndpointSelector& endpoint) {
  if (manifest.stages.empty()) {
    return nullptr;
  }

  std::size_t start = manifest.stages.size() - 1U;
  if (!endpoint.terminal_stage_key.empty()) {
    bool found = false;
    for (std::size_t i = 0; i < manifest.stages.size(); ++i) {
      if (stage_identity_matches(manifest.stages[i], endpoint.terminal_stage_key)) {
        start = i;
        found = true;
      }
    }
    if (!found) {
      return nullptr;
    }
  } else if (endpoint_has_output_selector(endpoint)) {
    for (std::size_t i = 0; i < manifest.stages.size(); ++i) {
      const auto& stage = manifest.stages[i];
      if (stage_outputs_match_endpoint(stage, endpoint)) {
        start = i;
      }
    }
  }

  for (std::size_t i = start + 1U; i-- > 0U;) {
    const auto& stage = manifest.stages[i];
    const StagePublicationRole role = classify_stage_for_publication(stage);
    if (role == StagePublicationRole::RealProducer || role == StagePublicationRole::MaterializedTransform) {
      return &stage;
    }
    if (i == 0U) {
      break;
    }
  }
  return nullptr;
}

sima::StageStaticSpec make_publication_stage_for_terminal(const sima::StageStaticSpec& terminal_stage) {
  StageStaticSpec publication = terminal_stage;
  if (logical_outputs_are_producer_local(publication)) {
    return publication;
  }

  publication.physical_outputs = raw_physical_outputs_for_terminal(terminal_stage);
  publication.logical_outputs = raw_logical_outputs_from_physical(publication.physical_outputs);
  publication.output_order.clear();
  publication.output_order.reserve(publication.logical_outputs.size());
  for (const auto& logical : publication.logical_outputs) {
    sima::StageOutputRoute route;
    route.output_slot = logical.output_slot;
    route.logical_output_index = logical.logical_index;
    route.tensor_index = logical.tensor_index;
    route.cm_output_name = logical.logical_name;
    route.segment_name = logical.segment_name;
    publication.output_order.push_back(std::move(route));
  }
  return publication;
}

bool validate_publication_stage_strict(const sima::StageStaticSpec& publication_stage,
                                       std::string* error_message) {
  if (publication_stage.physical_outputs.empty()) {
    if (error_message) {
      *error_message = "terminal publication stage has no physical outputs";
    }
    return false;
  }
  if (publication_stage.logical_outputs.empty()) {
    if (error_message) {
      *error_message = "terminal publication stage has no logical outputs";
    }
    return false;
  }

  for (const auto& logical : publication_stage.logical_outputs) {
    if (!sima::dtype_source_is_public_logical_contract_authoritative(logical.dtype_source)) {
      if (error_message) {
        *error_message = "terminal publication output '" + logical.logical_name +
                         "' dtype '" + logical.dtype +
                         "' does not have authoritative public provenance";
      }
      return false;
    }
    const auto dtype = tensor_dtype_from_static_token(logical.dtype);
    if (!dtype.has_value()) {
      if (error_message) {
        *error_message = "terminal publication output '" + logical.logical_name +
                         "' has unsupported dtype '" + logical.dtype + "'";
      }
      return false;
    }
    if (logical.shape.empty()) {
      if (error_message) {
        *error_message = "terminal publication output '" + logical.logical_name + "' has no shape";
      }
      return false;
    }
    const auto* physical = physical_for_logical(publication_stage, logical);
    if (!physical) {
      if (error_message) {
        *error_message = "terminal publication output '" + logical.logical_name +
                         "' references missing physical index " +
                         std::to_string(logical.physical_index);
      }
      return false;
    }
    if (logical.byte_offset < 0) {
      if (error_message) {
        *error_message = "terminal publication output '" + logical.logical_name +
                         "' has negative byte offset";
      }
      return false;
    }
    const std::uint64_t offset = static_cast<std::uint64_t>(logical.byte_offset);
    if (offset > physical->size_bytes) {
      if (error_message) {
        *error_message = "terminal publication output '" + logical.logical_name +
                         "' starts past its physical buffer";
      }
      return false;
    }
    const std::uint64_t span =
        std::max<std::uint64_t>(logical_span_bytes(logical, dtype_bytes(*dtype)), logical.size_bytes);
    if (span == 0U || span > physical->size_bytes - offset) {
      if (error_message) {
        std::ostringstream oss;
        oss << "terminal publication output '" << logical.logical_name
            << "' span does not fit physical buffer: span=" << span << " offset=" << offset
            << " physical_size=" << physical->size_bytes;
        *error_message = oss.str();
      }
      return false;
    }
  }
  return true;
}

std::optional<OutputTensorOverride> build_output_override_from_manifest(
    const sima::SimaPluginStaticManifest& manifest, const PublicOutputEndpointSelector& endpoint,
    std::string* error_message) {
  const StageStaticSpec* terminal = find_terminal_real_producer_for_endpoint(manifest, endpoint);
  if (!terminal) {
    if (error_message) {
      *error_message = "could not resolve terminal real producer for public output";
    }
    return std::nullopt;
  }

  StageStaticSpec publication = make_publication_stage_for_terminal(*terminal);
  std::string endpoint_error;
  if (!filter_publication_stage_for_endpoint(&publication, endpoint, &endpoint_error)) {
    if (error_message) {
      *error_message = endpoint_error;
    }
    return std::nullopt;
  }
  std::string validation_error;
  if (!validate_publication_stage_strict(publication, &validation_error)) {
    if (error_message) {
      *error_message = validation_error;
    }
    return std::nullopt;
  }

  OutputTensorOverride out;
  out.outputs.reserve(publication.logical_outputs.size());
  for (const auto& logical : publication.logical_outputs) {
    OutputTensorOverrideEntry entry;
    std::string entry_error;
    if (!logical_to_override_entry(publication, logical, &entry, &entry_error)) {
      if (error_message) {
        *error_message = entry_error;
      }
      return std::nullopt;
    }
    if (endpoint.route_slot >= 0 && out.outputs.empty()) {
      entry.route_slot = endpoint.route_slot;
    }
    out.outputs.push_back(std::move(entry));
  }
  return out;
}

} // namespace simaai::neat::pipeline_internal::terminal_output_contract
