#define SIMA_NEAT_INTERNAL 1
#include "pipeline/internal/sima/static_contract/DmabufPlanContractProjection.h"
#include "pipeline/internal/sima/static_contract/FrameSlotArenaPlan.h"
#include "pipeline/internal/sima/static_contract/PhysicalExecutionPlan.h"
#include "pipeline/internal/sima/TensorSemanticsUtil.h"
#include "pipeline/internal/sima/stagesemantics/ProcessCvuRuntimeConfigAdapterInternal.h"
#include "pipeline/internal/sima/stagesemantics/ProcessCvuStageSemantics.h"
#include "gst/SimaPluginStaticManifestAbi.h"

#include <algorithm>
#include <cctype>
#include <iterator>
#include <limits>
#include <optional>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>

namespace simaai::neat::pipeline_internal::sima::static_contract {
namespace {

bool fail(std::string* error, std::string detail) {
  if (error) {
    *error = std::move(detail);
  }
  return false;
}

bool checked_add(const std::uint64_t lhs, const std::uint64_t rhs,
                 std::uint64_t* result) {
  if (!result || lhs > std::numeric_limits<std::uint64_t>::max() - rhs) {
    return false;
  }
  *result = lhs + rhs;
  return true;
}

std::optional<std::uint64_t> checked_align_up(const std::uint64_t value,
                                              const std::uint64_t alignment) {
  if (alignment == 0U || (alignment & (alignment - 1U)) != 0U) {
    return std::nullopt;
  }
  const auto mask = alignment - 1U;
  std::uint64_t rounded = 0U;
  if (!checked_add(value, mask, &rounded)) {
    return std::nullopt;
  }
  return rounded & ~mask;
}

bool exact_logical_tensor_span(const TensorShape& shape,
                               std::span<const std::int64_t> stride_bytes,
                               const std::string& dtype, std::uint64_t* span) {
  if (!span || shape.empty() || dtype.empty() ||
      (!stride_bytes.empty() && stride_bytes.size() != shape.size())) {
    return false;
  }
  std::uint32_t ev_dtype = 0U;
  if (!tensorsemantics::dtype_token_to_ev(dtype, &ev_dtype)) {
    return false;
  }
  const int element_bytes = sima_ev_elem_size_bytes(ev_dtype);
  if (element_bytes <= 0) {
    return false;
  }
  std::uint64_t result = static_cast<std::uint64_t>(element_bytes);
  if (stride_bytes.empty()) {
    for (const auto dim : shape) {
      if (dim <= 0 ||
          static_cast<std::uint64_t>(dim) >
              std::numeric_limits<std::uint64_t>::max() / result) {
        return false;
      }
      result *= static_cast<std::uint64_t>(dim);
    }
  } else {
    for (std::size_t axis = 0; axis < shape.size(); ++axis) {
      const auto dim = shape[axis];
      const auto stride = stride_bytes[axis];
      if (dim <= 0 || stride < 0 || (dim > 1 && stride == 0)) {
        return false;
      }
      const auto extent = static_cast<std::uint64_t>(dim - 1);
      const auto step = static_cast<std::uint64_t>(stride);
      if (extent != 0U && step > std::numeric_limits<std::uint64_t>::max() / extent) {
        return false;
      }
      if (!checked_add(result, extent * step, &result)) {
        return false;
      }
    }
  }
  *span = result;
  return true;
}

const LogicalTensorStaticSpec*
find_exact_upstream_output(const ValueSpec& value,
                           std::span<const LogicalTensorStaticSpec> upstream_outputs,
                           std::string* error) {
  const LogicalTensorStaticSpec* match = nullptr;
  for (const auto& output : upstream_outputs) {
    const bool exact_name_match = output.logical_name == value.name ||
                                  output.backend_name == value.name ||
                                  output.segment_name == value.name;
    if (!exact_name_match) {
      continue;
    }
    if (match != nullptr && match != &output) {
      fail(error, "MLA input value '" + value.name +
                      "' matches more than one upstream TensorBuffer output");
      return nullptr;
    }
    match = &output;
  }
  return match;
}

const OpSpec* producer_of(const ModelExecutionPlan& plan, const ValueId value_id) {
  for (const auto& op : plan.ops()) {
    if (std::find(op.outputs.begin(), op.outputs.end(), value_id) != op.outputs.end()) {
      return &op;
    }
  }
  return nullptr;
}

ValueId root_value_id(const ModelExecutionPlan& plan, const ValueId value_id) {
  ValueId root = value_id;
  for (std::size_t remaining = plan.values().size() + 1U; remaining > 0U; --remaining) {
    const auto* value = plan.value(root);
    if (!value || !value->read_expression.has_value()) {
      return root;
    }
    root = value->read_expression->source_value_id;
  }
  return value_id;
}

std::optional<FrameSlotArenaPlan> compile_frame_arena(const ModelExecutionPlan& plan,
                                                      std::string* error) {
  std::string physical_error;
  auto physical = PhysicalExecutionLowerer::lower(plan, &physical_error);
  if (!physical) {
    fail(error, physical_error.empty()
                    ? "frame-slot arena could not lower the physical command DAG"
                    : std::move(physical_error));
    return std::nullopt;
  }
  const auto detached_roots = detached_mla_output_roots(plan, *physical);
  return FrameSlotArenaPlan::compile(
      plan, *physical, FrameSlotArenaReuse::DisjointLifetimes,
      kLegacyEvoCmaRegionAlignmentBytes, error,
      kModalixProductionArenaDmsPolicy, detached_roots);
}

bool assign_physical_region(const FrameSlotArenaPlan& arena, const ValueSpec& value,
                            PhysicalBufferStaticSpec* physical, std::string* error) {
  if (!physical) {
    return fail(error, "frame-slot physical projection is null");
  }
  const auto* region = arena.region(value.id);
  if (!region || region->size_bytes != physical->size_bytes ||
      physical->required_alignment_bytes == 0U ||
      region->byte_offset % physical->required_alignment_bytes != 0U ||
      region->byte_offset > static_cast<std::uint64_t>(std::numeric_limits<std::int64_t>::max())) {
    return fail(error, "frame-slot region contradicts value '" + value.name + "'");
  }
  physical->source_byte_offset = static_cast<std::int64_t>(region->byte_offset);
  return true;
}

const ValueSpec* find_materialized_value_for_logical(const ModelExecutionPlan& plan,
                                                     const LogicalTensorStaticSpec& logical,
                                                     std::string* error) {
  // Names are compiler identities, not runtime binding keys.  Try the most
  // backend-specific spelling first and require that spelling to identify one
  // materialized value.  A shared-parent segment such as "output_tensor" is
  // intentionally only the final fallback.
  for (const auto* name : {&logical.backend_name, &logical.logical_name, &logical.segment_name}) {
    if (name->empty()) {
      continue;
    }
    const ValueSpec* result = nullptr;
    for (const auto& value : plan.values()) {
      if (value.read_expression.has_value() || value.name != *name) {
        continue;
      }
      if (result) {
        fail(error, "frame-slot logical output name is ambiguous: '" + *name + "'");
        return nullptr;
      }
      result = &value;
    }
    if (result) {
      return result;
    }
  }
  return nullptr;
}

struct CvuOutputPlacement {
  const ValueSpec* value = nullptr;
  std::uint64_t byte_offset = 0U;
  std::uint64_t required_alignment_bytes = 0U;
};

std::optional<CvuOutputPlacement>
resolve_cvu_output_placement(const ModelExecutionPlan& plan, const FrameSlotArenaPlan& arena,
                             const ProcessCvuMlaBoundary boundary,
                             std::span<const BackendPortSpec> ports,
                             const LogicalTensorStaticSpec& logical, const std::size_t output_index,
                             const std::size_t output_count, std::string* error) {
  const ValueSpec* value = find_materialized_value_for_logical(plan, logical, error);
  if (!value && error && !error->empty()) {
    return std::nullopt;
  }
  if (!value && output_count == ports.size()) {
    value = plan.value(ports[output_index].value_id);
  }
  if (!value && boundary == ProcessCvuMlaBoundary::Outputs &&
      output_count == plan.model_outputs().size()) {
    value = plan.value(plan.model_outputs()[output_index].value_id);
  }
  if (!value || value->read_expression.has_value()) {
    fail(error, "ProcessCVU logical output has no materialized execution-plan value: "
                "logical='" +
                    logical.logical_name + "' backend='" + logical.backend_name + "' segment='" +
                    logical.segment_name + "'");
    return std::nullopt;
  }

  const auto alignment_for_boundary = [&]() -> std::uint64_t {
    if (ports.size() == output_count) {
      return ports[output_index].required_alignment_bytes;
    }
    if (ports.size() == 1U) {
      return ports.front().required_alignment_bytes;
    }
    return 0U;
  };

  // An exact batch-one Pack is a placement expression for the preceding CVU
  // outputs.  The CVU writes each child directly into its compiler-authored
  // parent offset; no Pack job or copy is scheduled.
  if (boundary == ProcessCvuMlaBoundary::Inputs && ports.size() == 1U) {
    for (const auto& op : plan.ops()) {
      if (op.kind != OpKind::Pack || op.outputs.size() != 1U ||
          op.outputs.front() != ports.front().value_id) {
        continue;
      }
      const auto* pack = std::get_if<PackOpConfig>(&op.config);
      if (!pack || pack->components.size() != op.inputs.size()) {
        break;
      }
      for (const auto& component : pack->components) {
        if (component.value_id != value->id) {
          continue;
        }
        const auto* parent = arena.region(op.outputs.front());
        if (!parent || component.parent_offset >
                           std::numeric_limits<std::uint64_t>::max() - parent->byte_offset) {
          fail(error, "ProcessCVU Pack child has no parent frame-arena placement");
          return std::nullopt;
        }
        return CvuOutputPlacement{value, parent->byte_offset + component.parent_offset, 16U};
      }
    }
  }

  const auto* region = arena.region(value->id);
  const auto alignment = alignment_for_boundary();
  if (!region || alignment == 0U || region->byte_offset % alignment != 0U) {
    fail(error, "ProcessCVU value '" + value->name + "' has no aligned frame-arena placement");
    return std::nullopt;
  }
  return CvuOutputPlacement{value, region->byte_offset, alignment};
}

std::optional<int>
resolve_pack_parent_physical_source(const ModelExecutionPlan& plan, const ValueSpec& packed_value,
                                    std::span<const LogicalTensorStaticSpec> upstream_outputs,
                                    std::string* error) {
  const auto* pack = producer_of(plan, packed_value.id);
  if (!pack || pack->kind != OpKind::Pack || pack->outputs.size() != 1U ||
      pack->outputs.front() != packed_value.id || pack->inputs.empty()) {
    return std::nullopt;
  }
  const auto* config = std::get_if<PackOpConfig>(&pack->config);
  if (!config || config->materializes ||
      (config->components.size() != pack->inputs.size() &&
       (config->batch_count == 0U ||
        config->spans.size() != pack->inputs.size() * config->batch_count))) {
    fail(error, "MLA packed IFM has no exact Pack component placement");
    return std::nullopt;
  }

  if (!config->spans.empty()) {
    const auto* parent_binding = packed_value.storage_binding
                                     ? &*packed_value.storage_binding
                                     : nullptr;
    if (!parent_binding || config->parent_required_bytes != packed_value.required_bytes) {
      fail(error, "MLA packed IFM has no exact parent storage binding");
      return std::nullopt;
    }
    int physical_index = -1;
    std::vector<std::pair<std::uint64_t, std::uint64_t>> destinations;
    destinations.reserve(config->spans.size());
    for (const auto input_id : pack->inputs) {
      const auto* child_value = plan.value(input_id);
      const auto* child_binding = child_value && child_value->storage_binding
                                      ? &*child_value->storage_binding
                                      : nullptr;
      const auto* child = child_value
                              ? find_exact_upstream_output(*child_value, upstream_outputs, error)
                              : nullptr;
      if (!child_value || !child_binding || !child ||
          child_binding->carrier_id != parent_binding->carrier_id ||
          child->logical_index < 0 || child->physical_index < 0 || child->byte_offset < 0 ||
          child->size_bytes != child_value->required_bytes ||
          static_cast<std::uint64_t>(child->byte_offset) != child_binding->byte_offset ||
          child->stride_bytes != child_binding->stride_bytes) {
        fail(error, "MLA packed IFM child does not publish its exact shared-carrier view");
        return std::nullopt;
      }
      if (physical_index < 0) {
        physical_index = child->physical_index;
      } else if (physical_index != child->physical_index) {
        fail(error, "MLA packed IFM children do not share one physical parent");
        return std::nullopt;
      }
    }
    for (const auto& span : config->spans) {
      std::uint64_t end = 0U;
      if (!checked_add(span.parent_offset, span.stored_bytes, &end)) {
        fail(error, "MLA packed IFM span overflows");
        return std::nullopt;
      }
      destinations.emplace_back(span.parent_offset, end);
    }
    std::sort(destinations.begin(), destinations.end());
    std::uint64_t covered = 0U;
    for (const auto& [begin, end] : destinations) {
      if (begin != covered) {
        fail(error, "MLA packed IFM child views leave a gap or overlap");
        return std::nullopt;
      }
      covered = end;
    }
    if (physical_index < 0 || covered != packed_value.required_bytes) {
      fail(error, "MLA packed IFM children do not cover the exact parent");
      return std::nullopt;
    }
    return physical_index;
  }

  int physical_index = -1;
  std::uint64_t covered_bytes = 0U;
  for (std::size_t component_index = 0; component_index < pack->inputs.size(); ++component_index) {
    const auto input_id = pack->inputs[component_index];
    PackComponentPlacement placement;
    if (!config->components.empty()) {
      placement = config->components[component_index];
    } else {
      const auto found = std::find_if(
          config->spans.begin(), config->spans.end(), [&](const PackSpan& span) {
            return span.value_id == input_id && span.batch_index == 0U;
          });
      if (found == config->spans.end() || found->source_byte_offset != 0U ||
          found->logical_bytes != plan.value(input_id)->required_bytes ||
          found->stored_bytes != found->logical_bytes || found->padding_policy != "none") {
        fail(error, "MLA packed IFM has a non-direct Pack member");
        return std::nullopt;
      }
      placement = {found->value_id, found->parent_offset, found->stored_bytes};
    }
    const auto* child_value = plan.value(input_id);
    if (!child_value) {
      fail(error, "MLA packed IFM references a missing child value");
      return std::nullopt;
    }
    const auto* child = find_exact_upstream_output(*child_value, upstream_outputs, error);
    if (!child) {
      if (error && error->empty()) {
        *error = "MLA packed IFM child '" + child_value->name +
                 "' has no exact upstream TensorBuffer view";
      }
      return std::nullopt;
    }
    if (placement.value_id != input_id || placement.parent_offset != covered_bytes ||
        placement.stored_bytes != child_value->required_bytes || child->logical_index < 0 ||
        child->physical_index < 0 || child->byte_offset < 0 ||
        child->size_bytes != child_value->required_bytes ||
        static_cast<std::uint64_t>(child->byte_offset) != placement.parent_offset) {
      fail(error, "MLA packed IFM child '" + child_value->name +
                      "' does not exactly cover the ordered parent carrier");
      return std::nullopt;
    }
    if (physical_index < 0) {
      physical_index = child->physical_index;
    } else if (child->physical_index != physical_index) {
      fail(error, "MLA packed IFM children do not share one physical parent");
      return std::nullopt;
    }
    if (placement.stored_bytes > std::numeric_limits<std::uint64_t>::max() - covered_bytes) {
      fail(error, "MLA packed IFM child extents overflow");
      return std::nullopt;
    }
    covered_bytes += placement.stored_bytes;
  }
  if (physical_index < 0 || covered_bytes != packed_value.required_bytes) {
    fail(error, "MLA packed IFM children do not exactly cover the MLA port length");
    return std::nullopt;
  }
  return physical_index;
}

std::vector<const ValueSpec*> read_views_consumed_after_mla(const ModelExecutionPlan& plan,
                                                            const ValueId root_value_id) {
  std::vector<const ValueSpec*> result;
  std::unordered_set<ValueId> seen;
  for (const auto& op : plan.ops()) {
    if (op.kind == OpKind::Unpack || op.kind == OpKind::Slice) {
      continue;
    }
    for (const auto input_id : op.inputs) {
      const auto* value = plan.value(input_id);
      if (!value || !value->read_expression.has_value() ||
          value->read_expression->source_value_id != root_value_id ||
          !seen.emplace(input_id).second) {
        continue;
      }
      result.push_back(value);
    }
  }
  return result;
}

struct PublicationTransportView {
  std::string carrier_dtype;
  std::optional<std::string> carrier_layout;
  std::uint64_t physical_span = 0U;
};

// A terminal MLA cut publishes the raw transport view, not the semantic value
// produced by a removed Detess/Cast tail. In legacy packed-OFM manifests the
// Unpack tensor type is the exact byte-carrier authority even when a typed
// downstream Detess proves that ObjectDecode must interpret those bytes as
// BF16. Trace only the address relations which preserve that carrier; never
// let the downstream semantic dtype/layout change the producer catalogue.
std::optional<PublicationTransportView> resolve_publication_transport_view(
    const ModelExecutionPlan& plan, const ValueSpec& value, std::string* error) {
  if (!value.logical_shape.has_value()) {
    fail(error, "MLA publication view has no exact shape");
    return std::nullopt;
  }

  std::string carrier_dtype = value.logical_dtype.value_or(std::string{});
  std::optional<std::string> carrier_layout = value.logical_layout;
  ValueId cursor = value.id;
  std::unordered_set<ValueId> visited;
  for (std::size_t remaining = plan.values().size() + 1U; remaining > 0U;
       --remaining) {
    if (!visited.emplace(cursor).second) {
      fail(error, "MLA publication address lineage contains a cycle");
      return std::nullopt;
    }
    const auto* producer = producer_of(plan, cursor);
    if (!producer) {
      break;
    }
    const auto output =
        std::find(producer->outputs.begin(), producer->outputs.end(), cursor);
    if (output == producer->outputs.end()) {
      fail(error, "MLA publication address lineage lost its output identity");
      return std::nullopt;
    }
    const auto output_index = static_cast<std::size_t>(
        std::distance(producer->outputs.begin(), output));
    if (producer->kind == OpKind::Unpack) {
      const auto* unpack = std::get_if<UnpackOpConfig>(&producer->config);
      if (!unpack || producer->inputs.size() != 1U ||
          unpack->tensor_types.size() != producer->outputs.size() ||
          output_index >= unpack->tensor_types.size() ||
          unpack->tensor_types[output_index].empty()) {
        fail(error, "packed OFM Unpack view has no exact carrier dtype");
        return std::nullopt;
      }
      carrier_dtype = unpack->tensor_types[output_index];
      // The flattened Unpack view does not author a physical HWC layout.
      // Any HWC evidence propagated from a removed Detess belongs solely to
      // the consumer contract.
      carrier_layout.reset();
      break;
    }
    if (producer->kind == OpKind::Slice ||
        producer->kind == OpKind::Reshape) {
      if (producer->inputs.size() != 1U || producer->outputs.size() != 1U ||
          output_index != 0U) {
        fail(error, "MLA publication address relation is ambiguous");
        return std::nullopt;
      }
      cursor = producer->inputs.front();
      continue;
    }
    if (producer->kind == OpKind::PassThrough) {
      if (producer->inputs.size() != producer->outputs.size() ||
          output_index >= producer->inputs.size()) {
        fail(error, "MLA publication PassThrough relation is ambiguous");
        return std::nullopt;
      }
      cursor = producer->inputs[output_index];
      continue;
    }
    break;
  }

  std::span<const std::int64_t> strides;
  if (value.read_expression.has_value()) {
    strides = value.read_expression->stride_bytes;
  } else if (value.storage_binding.has_value()) {
    strides = value.storage_binding->stride_bytes;
  }
  std::uint64_t physical_span = 0U;
  if (!exact_logical_tensor_span(*value.logical_shape, strides, carrier_dtype,
                                 &physical_span)) {
    fail(error, "MLA publication carrier shape/stride/dtype is not exact");
    return std::nullopt;
  }
  const auto* binding = value.storage_binding ? &*value.storage_binding : nullptr;
  if (!binding || binding->physical_span != physical_span ||
      binding->stride_bytes.size() != strides.size() ||
      !std::equal(binding->stride_bytes.begin(), binding->stride_bytes.end(),
                  strides.begin(), strides.end())) {
    fail(error,
         "MLA publication carrier view disagrees with normalized storage");
    return std::nullopt;
  }
  return PublicationTransportView{std::move(carrier_dtype),
                                  std::move(carrier_layout), physical_span};
}

bool project_terminal_mla_publications(
    const ModelExecutionPlan& plan, const std::span<const BackendPortSpec> outputs,
    MlaStaticContract* contract, std::string* error) {
  if (!contract || outputs.empty() || plan.model_outputs().empty()) {
    return fail(error, "terminal MLA publication has no exact output contract");
  }

  std::unordered_map<ValueId, std::size_t> port_for_root;
  port_for_root.reserve(outputs.size());
  for (std::size_t port_index = 0; port_index < outputs.size(); ++port_index) {
    const auto& port = outputs[port_index];
    const auto root = root_value_id(plan, port.value_id);
    if (port.port_index != port_index ||
        !port_for_root.emplace(root, port_index).second) {
      return fail(error, "terminal MLA publication has an ambiguous physical output root");
    }
  }

  contract->logical_outputs.clear();
  contract->logical_outputs.reserve(plan.model_outputs().size());
  for (std::size_t publication_index = 0;
       publication_index < plan.model_outputs().size(); ++publication_index) {
    const auto& publication = plan.model_outputs()[publication_index];
    const auto* value = plan.value(publication.value_id);
    if (!value || publication.public_index != publication_index ||
        publication_index >
            static_cast<std::size_t>(std::numeric_limits<int>::max()) ||
        !value->logical_shape.has_value()) {
      return fail(error, "terminal MLA publication has sparse or untyped public metadata");
    }
    const auto root = root_value_id(plan, publication.value_id);
    const auto found = port_for_root.find(root);
    if (found == port_for_root.end()) {
      return fail(error, "terminal MLA public value does not map to an exact physical port");
    }
    const auto port_index = found->second;
    const auto* root_value = plan.value(root);
    if (!root_value) {
      return fail(error, "terminal MLA physical output root is missing");
    }

    std::uint64_t byte_offset = 0U;
    std::span<const std::int64_t> strides;
    if (value->read_expression.has_value()) {
      const auto& expression = *value->read_expression;
      if (root_value_id(plan, expression.source_value_id) != root) {
        return fail(error, "terminal MLA public read expression has a foreign root");
      }
      byte_offset = expression.byte_offset;
      strides = std::span<const std::int64_t>(expression.stride_bytes);
    } else if (value->storage_binding.has_value() &&
               !value->storage_binding->stride_bytes.empty()) {
      strides = std::span<const std::int64_t>(
          value->storage_binding->stride_bytes);
    }

    const auto transport =
        resolve_publication_transport_view(plan, *value, error);
    if (byte_offset > static_cast<std::uint64_t>(
                          std::numeric_limits<std::int64_t>::max()) ||
        !transport.has_value() ||
        byte_offset > outputs[port_index].physical_extent_bytes ||
        transport->physical_span >
            outputs[port_index].physical_extent_bytes - byte_offset) {
      if (!transport.has_value()) {
        return false;
      }
      return fail(error, "terminal MLA public view exceeds its physical output port");
    }

    const int logical_index = static_cast<int>(publication_index);
    const int physical_index = static_cast<int>(port_index);
    LogicalTensorStaticSpec logical;
    logical.logical_index = logical_index;
    logical.backend_output_index = physical_index;
    logical.physical_index = physical_index;
    logical.output_slot = logical_index;
    logical.tensor_index = logical_index;
    logical.byte_offset = static_cast<std::int64_t>(byte_offset);
    logical.size_bytes = value->required_bytes;
    logical.shape = *value->logical_shape;
    logical.stride_bytes.assign(strides.begin(), strides.end());
    logical.dtype = transport->carrier_dtype;
    logical.dtype_source = DTypeSource::InternalContract;
    logical.logical_name = publication.name.empty() ? value->name : publication.name;
    logical.backend_name = value->name;
    logical.segment_name = root_value->name;
    if (transport->carrier_layout.has_value()) {
      logical.layout = *transport->carrier_layout;
    }
    contract->logical_outputs.push_back(std::move(logical));
  }
  return true;
}

std::uint32_t cvu_semantic_op_code(const OpKind kind) noexcept {
  switch (kind) {
  case OpKind::Cast:
    return SIMA_CVU_SEMANTIC_OP_CAST;
  case OpKind::Quantize:
    return SIMA_CVU_SEMANTIC_OP_QUANTIZE;
  case OpKind::Tessellate:
    return SIMA_CVU_SEMANTIC_OP_TESSELLATE;
  case OpKind::Detessellate:
    return SIMA_CVU_SEMANTIC_OP_DETESSELLATE;
  case OpKind::Dequantize:
    return SIMA_CVU_SEMANTIC_OP_DEQUANTIZE;
  default:
    return SIMA_CVU_SEMANTIC_OP_NONE;
  }
}

struct PhysicalCvuCohortView {
  SimaCvuCapabilityAbiRecord capability{};
  std::string implementation_id;
  std::uint32_t batch_size = 0U;
  std::vector<const PhysicalCommandMember*> members;
};

std::optional<PhysicalCvuCohortView> resolve_physical_cvu_cohort(
    const ModelExecutionPlan& plan, const PhysicalExecutionPlan& physical_plan,
    const std::span<const PhysicalCommandId> command_ids, std::string* error) {
  if (command_ids.empty()) {
    fail(error, "ProcessCVU projection references no physical command");
    return std::nullopt;
  }
  PhysicalCvuCohortView result;
  std::optional<PhysicalCohortId> cohort_id;
  std::uint32_t graph_id = 0U;
  std::uint32_t maximum_members = 0U;
  for (const auto command_id : command_ids) {
    if (command_id >= physical_plan.commands.size()) {
      fail(error, "ProcessCVU projection references a missing physical command");
      return std::nullopt;
    }
    const auto& command = physical_plan.commands[command_id];
    if (command.id != command_id || command.engine != PhysicalEngine::Cvu ||
        command.graph_id == 0U || command.members.empty() || command.maximum_members == 0U ||
        command.members.size() > command.maximum_members ||
        (cohort_id.has_value() && *cohort_id != command.cohort_id) ||
        (!result.implementation_id.empty() &&
         result.implementation_id != command.implementation_id) ||
        (result.batch_size != 0U && result.batch_size != command.batch_size) ||
        (graph_id != 0U && graph_id != command.graph_id) ||
        (maximum_members != 0U && maximum_members != command.maximum_members)) {
      fail(error, "ProcessCVU projection references an incompatible physical cohort");
      return std::nullopt;
    }
    cohort_id = command.cohort_id;
    graph_id = command.graph_id;
    maximum_members = command.maximum_members;
    result.batch_size = command.batch_size;
    result.implementation_id = command.implementation_id;
    for (const auto& member : command.members) {
      result.members.push_back(&member);
    }
  }

  std::sort(result.members.begin(), result.members.end(),
            [](const auto* left, const auto* right) {
              return left->ordinal < right->ordinal;
            });
  if (result.members.empty() ||
      std::adjacent_find(result.members.begin(), result.members.end(),
                         [](const auto* left, const auto* right) {
                           return left->ordinal == right->ordinal;
                         }) != result.members.end() ||
      !sima_cvu_capability_abi_lookup(graph_id, &result.capability) ||
      result.capability.maximum_members != maximum_members ||
      result.capability.semantic_pattern_length == 0U || result.batch_size == 0U ||
      result.batch_size > static_cast<std::uint32_t>(std::numeric_limits<int>::max())) {
    fail(error, "ProcessCVU physical cohort has no exact generated capability");
    return std::nullopt;
  }

  const std::string implementation_prefix =
      "cvu.graph" + std::to_string(graph_id) + ".";
  if (result.implementation_id.rfind(implementation_prefix, 0U) != 0U) {
    fail(error, "ProcessCVU physical implementation identity contradicts its graph id");
    return std::nullopt;
  }
  for (const auto* member : result.members) {
    if (member->semantic_chain.size() != result.capability.semantic_pattern_length ||
        member->outer_inputs.size() != 1U || member->outer_outputs.size() != 1U ||
        std::any_of(member->semantic_chain.begin(), member->semantic_chain.end(),
                    [&](const auto op_id) {
                      return op_id >= plan.ops().size() || plan.ops()[op_id].id != op_id;
                    })) {
      fail(error, "ProcessCVU physical cohort has no exact registered member chain");
      return std::nullopt;
    }
    for (std::size_t index = 0; index < member->semantic_chain.size(); ++index) {
      const auto& op = plan.ops()[member->semantic_chain[index]];
      const auto expected_semantic =
          index == 0U ? result.capability.semantic_op0 : result.capability.semantic_op1;
      if (cvu_semantic_op_code(op.kind) != expected_semantic || op.processor != "EV74" ||
          op.inputs.size() != 1U || op.outputs.size() != 1U ||
          (index > 0U &&
           !resolve_exact_private_ordered_relation_path(
               plan, member->semantic_chain[index - 1U], op.id))) {
        fail(error, "ProcessCVU member chain is not one registered connected transform");
        return std::nullopt;
      }
    }
    const auto& first = plan.ops()[member->semantic_chain.front()];
    const auto& last = plan.ops()[member->semantic_chain.back()];
    if (member->outer_inputs.front() != first.inputs.front() ||
        member->outer_outputs.front() != last.outputs.front()) {
      fail(error, "ProcessCVU member outer bindings contradict its semantic chain");
      return std::nullopt;
    }
  }
  return result;
}

bool apply_processcvu_implementation_contract(ProcessCvuStagePayload* payload,
                                              std::string* error,
                                              const std::optional<std::uint32_t> graph_id =
                                                  std::nullopt) {
  if (!payload) {
    return fail(error, "ProcessCVU implementation projection is null");
  }
  if (graph_id.has_value()) {
    SimaCvuCapabilityAbiRecord capability{};
    if (!sima_cvu_capability_abi_lookup(*graph_id, &capability)) {
      return fail(error, "dmabuf-plan ProcessCVU has no generated physical ABI record");
    }
    const auto expected_family =
        stagesemantics::canonical_processcvu_graph_family(capability.canonical_token);
    const auto actual_family =
        stagesemantics::canonical_processcvu_graph_family(payload->graph_family);
    if (expected_family.empty() || actual_family != expected_family) {
      return fail(error, "dmabuf-plan ProcessCVU family contradicts its generated graph");
    }
    payload->graph_id = static_cast<int>(capability.graph_id);
    payload->graph_name = capability.canonical_token;
    payload->descriptor_abi_id = capability.descriptor_abi_id;
    payload->descriptor_contract_version = capability.descriptor_contract_version;
    payload->binding_schema_version = capability.binding_schema_version;
    payload->supported_placement_mask = capability.supported_placement_mask;
    payload->allowed_frame_patch_mask = capability.allowed_frame_patch_mask;
    payload->maximum_members = capability.maximum_members;
    return true;
  }
  switch (payload->graph_family_enum) {
  case ProcessCvuGraphFamily::Preproc:
    payload->graph_id = 200;
    payload->graph_name = "preproc";
    break;
  case ProcessCvuGraphFamily::Cast:
    payload->graph_id = 221;
    payload->graph_name = "cast";
    break;
  case ProcessCvuGraphFamily::Quant:
    payload->graph_id = 222;
    payload->graph_name = "quantize";
    break;
  case ProcessCvuGraphFamily::Dequant:
    payload->graph_id = 223;
    payload->graph_name = "dequantize";
    break;
  case ProcessCvuGraphFamily::Tess:
    payload->graph_id = 2;
    payload->graph_name = "tessellate";
    break;
  case ProcessCvuGraphFamily::Detess:
    payload->graph_id = 3;
    payload->graph_name = "detessellate";
    break;
  default:
    return fail(error, "dmabuf-plan ProcessCVU does not support this graph family (family=" +
                           std::to_string(static_cast<int>(payload->graph_family_enum)) + ")");
  }
  SimaCvuCapabilityAbiRecord capability{};
  if (!sima_cvu_capability_abi_lookup(static_cast<std::uint32_t>(payload->graph_id), &capability) ||
      payload->graph_name != capability.canonical_token) {
    return fail(error, "dmabuf-plan ProcessCVU has no generated physical ABI record");
  }
  payload->descriptor_abi_id = capability.descriptor_abi_id;
  payload->descriptor_contract_version = capability.descriptor_contract_version;
  payload->binding_schema_version = capability.binding_schema_version;
  payload->supported_placement_mask = capability.supported_placement_mask;
  payload->allowed_frame_patch_mask = capability.allowed_frame_patch_mask;
  payload->maximum_members = capability.maximum_members;
  return true;
}

std::vector<int> cvu_shape_from_tensor(const TensorShape& source) {
  std::vector<int> result;
  result.reserve(source.size());
  for (const auto dim : source) {
    if (dim <= 0 || dim > std::numeric_limits<int>::max()) {
      return {};
    }
    result.push_back(static_cast<int>(dim));
  }
  return result;
}

std::vector<int> cvu_shape(const ValueSpec& value, const TensorShape& fallback = {}) {
  return cvu_shape_from_tensor(value.logical_shape.has_value() ? *value.logical_shape : fallback);
}

std::string cvu_dtype(const ValueSpec& value, std::string fallback = {}) {
  std::string dtype = value.logical_dtype.value_or(std::move(fallback));
  std::transform(dtype.begin(), dtype.end(), dtype.begin(), [](const unsigned char c) {
    return static_cast<char>(std::toupper(c));
  });
  if (dtype == "FLOAT32") return "FP32";
  if (dtype == "FLOAT16") return "FP16";
  if (dtype == "BFLOAT16") return "BF16";
  return dtype;
}

bool build_cvu_dense_desc_with_geometry(const ValueSpec& value,
                                        const TensorShape& logical_shape,
                                        const std::string& logical_layout,
                                        const std::string& fallback_dtype,
                                        sima_ev_tensor_desc* descriptor,
                                        std::string* error) {
  const auto shape = cvu_shape_from_tensor(logical_shape);
  const auto dtype = cvu_dtype(value, fallback_dtype);
  std::string detail;
  const bool valid = tensorsemantics::normalize_layout_token(logical_layout).empty()
                         ? tensorsemantics::build_generic_dense_tensor_desc(
                               shape, dtype, descriptor, &detail, "missing tensor descriptor",
                               "invalid tensor rank", "invalid tensor dimension",
                               "invalid tensor dtype", "invalid tensor stride")
                         : tensorsemantics::build_dense_tensor_desc(
                               shape, dtype, logical_layout, descriptor, &detail,
                               "missing tensor descriptor", "invalid tensor rank",
                               "invalid tensor dimension", "invalid tensor dtype",
                               "invalid tensor stride");
  if (!valid) {
    return fail(error, "ProcessCVU could not author dense tensor descriptor: " + detail);
  }

  // A tensor descriptor describes the exact addressed view, not every byte in
  // its backing carrier.  QMLA ports may have harmless tail padding (and, for
  // multi-row outputs, an exact compiler-authored row pitch).  Keep that
  // larger extent in the physical-buffer contract while authoring the CVU
  // descriptor from the value's semantic bytes and storage strides.
  std::uint64_t dense_span = 0U;
  if (!exact_logical_tensor_span(logical_shape, {}, dtype, &dense_span) ||
      dense_span != value.required_bytes) {
    return fail(error, "ProcessCVU dense tensor bytes contradict its exact typed value");
  }
  std::span<const std::int64_t> authored_strides;
  if (value.storage_binding && !value.storage_binding->stride_bytes.empty()) {
    authored_strides = value.storage_binding->stride_bytes;
    if (authored_strides.size() != descriptor->shape.rank) {
      return fail(error, "ProcessCVU dense tensor storage stride rank is invalid");
    }
    for (std::size_t axis = 0; axis < authored_strides.size(); ++axis) {
      if (authored_strides[axis] <= 0) {
        return fail(error, "ProcessCVU dense tensor storage stride is invalid");
      }
      descriptor->layout.strided.strides_bytes[axis] = authored_strides[axis];
    }
  }
  std::uint64_t addressed_span = 0U;
  if (!exact_logical_tensor_span(logical_shape, authored_strides, dtype, &addressed_span) ||
      addressed_span < dense_span ||
      (value.storage_binding && addressed_span > value.storage_binding->physical_span)) {
    return fail(error, "ProcessCVU dense tensor view exceeds its exact physical carrier");
  }
  descriptor->storage.nbytes = addressed_span;
  return addressed_span != 0U;
}

bool build_cvu_dense_desc(const ValueSpec& value, const TensorShape& fallback_shape,
                          const std::string& fallback_dtype, sima_ev_tensor_desc* descriptor,
                          std::string* error) {
  const TensorShape logical_shape =
      value.logical_shape.has_value() ? *value.logical_shape : fallback_shape;
  return build_cvu_dense_desc_with_geometry(
      value, logical_shape, value.logical_layout.value_or(""), fallback_dtype,
      descriptor, error);
}

bool build_cvu_tiled_desc(const ValueSpec& value, const TensorShape& frame_shape,
                          const TensorShape& raw_tile_shape, const std::string& frame_type,
                          const bool c16_packed, sima_ev_tensor_desc* descriptor,
                          std::string* error) {
  const auto shape = !frame_shape.empty() ? cvu_shape_from_tensor(frame_shape)
                                          : cvu_shape(value);
  std::vector<int> raw_tiles;
  raw_tiles.reserve(raw_tile_shape.size());
  for (const auto dim : raw_tile_shape) {
    if (dim <= 0 || dim > std::numeric_limits<int>::max()) {
      return fail(error, "ProcessCVU tiled descriptor has an invalid slice shape");
    }
    raw_tiles.push_back(static_cast<int>(dim));
  }
  std::vector<int> tiles;
  std::string detail;
  if (!tensorsemantics::normalize_tile_shape(
          shape, raw_tiles, &tiles, &detail, "missing tile shape",
          "tile rank is incompatible with frame rank", "invalid tile dimension")) {
    return fail(error, "ProcessCVU could not normalize tiled tensor descriptor: " + detail);
  }
  const auto dtype = cvu_dtype(value, frame_type);
  const auto layout = value.logical_layout.value_or("");
  if (!tensorsemantics::build_tiled_tensor_desc(
          shape, tiles, dtype, layout, 16U, descriptor, &detail,
          "missing tensor descriptor", "invalid tensor rank", "invalid tensor dimension",
          "invalid tensor dtype", "tile rank mismatch", "invalid tile dimension")) {
    return fail(error, "ProcessCVU could not author tiled tensor descriptor: " + detail);
  }
  if (c16_packed) {
    descriptor->layout.tiled.flags &=
        ~static_cast<std::uint32_t>(SIMA_EV_TILED_FLAG_COMPACT_CHANNELS);
  }
  descriptor->storage.nbytes = value.storage_binding
                                   ? value.storage_binding->physical_span
                                   : value.required_bytes;
  return descriptor->storage.nbytes != 0U;
}

std::optional<int> cvu_rounding_mode(const std::string& raw) {
  std::string token = raw;
  std::transform(token.begin(), token.end(), token.begin(), [](const unsigned char c) {
    return static_cast<char>(std::toupper(c));
  });
  if (token == "RT_ZERO" || token == "TOZERO") return 0;
  if (token == "RT_EVEN" || token == "TONEAREST") return 1;
  if (token == "RT_POSITVE_INFINITY" || token == "TOPOSITIVEINFINITY") return 2;
  if (token == "RT_NEGATIVE_INFINITY" || token == "TONEGATIVEINFINITY") return 3;
  return std::nullopt;
}

} // namespace

std::optional<CompiledProcessCvuContract> build_dmabuf_plan_processcvu_command_contract(
    const ModelExecutionPlan& plan, const PhysicalExecutionPlan& physical_plan,
    const std::span<const PhysicalCommandId> command_ids, const FrameSlotArenaPlan& arena,
    std::string* error) {
  const auto cohort = resolve_physical_cvu_cohort(plan, physical_plan, command_ids, error);
  if (!cohort) {
    return std::nullopt;
  }

  stagesemantics::CompiledProcessCvuRuntimeConfig authored;
  authored.graph_family =
      stagesemantics::canonical_processcvu_graph_family(cohort->capability.canonical_token);
  authored.graph_name = cohort->capability.canonical_token;
  authored.graph_id = static_cast<int>(cohort->capability.graph_id);
  // Existing ProcessCVU ABI encodes the canonical 16-byte tile alignment as
  // the sentinel value 1; the runtime-config adapter resolves 1 -> 16.
  authored.byte_align = 1;
  authored.batch_size = static_cast<int>(cohort->batch_size);
  authored.input_tensors.reserve(cohort->members.size());
  authored.output_tensors.reserve(cohort->members.size());
  authored.input_shapes.reserve(cohort->members.size());
  authored.output_shapes.reserve(cohort->members.size());
  authored.slice_shapes.reserve(cohort->members.size());
  authored.runtime_input_names.reserve(cohort->members.size());
  authored.runtime_output_names.reserve(cohort->members.size());
  authored.physical_input_names.reserve(cohort->members.size());
  authored.physical_output_names.reserve(cohort->members.size());
  authored.published_output_names.reserve(cohort->members.size());
  authored.runtime_output_logical_index_list.reserve(cohort->members.size());
  authored.runtime_output_output_slot_list.reserve(cohort->members.size());
  authored.runtime_output_physical_index_list.reserve(cohort->members.size());
  authored.runtime_output_dtype_list.reserve(cohort->members.size());
  authored.runtime_output_transport_kind_list.reserve(cohort->members.size());
  authored.runtime_output_semantic_kind_list.reserve(cohort->members.size());
  authored.runtime_output_logical_shapes.reserve(cohort->members.size());
  authored.runtime_output_logical_layout_list.reserve(cohort->members.size());

  std::optional<int> uniform_rounding;
  bool detesscast_requires_lane_split = false;
  for (std::size_t index = 0; index < cohort->members.size(); ++index) {
    const auto& member = *cohort->members[index];
    const auto& first_op = plan.ops()[member.semantic_chain.front()];
    const auto& last_op = plan.ops()[member.semantic_chain.back()];
    const auto* input = plan.value(member.outer_inputs.front());
    const auto* output = plan.value(member.outer_outputs.front());
    if (!input || !output || !input->storage_binding || !output->storage_binding ||
        !input->logical_shape || !input->logical_dtype || !output->logical_shape ||
        !output->logical_dtype) {
      fail(error, "ProcessCVU physical member has no exact typed outer values");
      return std::nullopt;
    }

    const TessellateOpConfig* tess = nullptr;
    const DetessellateOpConfig* detess = nullptr;
    const QuantizeOpConfig* quant = nullptr;
    const DequantizeOpConfig* dequant = nullptr;
    const OpSpec* tess_op = nullptr;
    for (const auto op_id : member.semantic_chain) {
      const auto& op = plan.ops()[op_id];
      if (op.kind == OpKind::Tessellate) {
        tess = std::get_if<TessellateOpConfig>(&op.config);
        tess_op = &op;
      } else if (op.kind == OpKind::Detessellate) {
        detess = std::get_if<DetessellateOpConfig>(&op.config);
      } else if (op.kind == OpKind::Quantize) {
        quant = std::get_if<QuantizeOpConfig>(&op.config);
      } else if (op.kind == OpKind::Dequantize) {
        dequant = std::get_if<DequantizeOpConfig>(&op.config);
      }
    }

    sima_ev_tensor_desc input_descriptor{};
    sima_ev_tensor_desc output_descriptor{};
    if (detess) {
      if (!build_cvu_tiled_desc(*input, detess->frame_shape, detess->slice_shape,
                                detess->frame_type, detess->align_c16 || detess->cblock,
                                &input_descriptor, error)) {
        return std::nullopt;
      }
      authored.slice_shapes.push_back(cvu_shape_from_tensor(detess->slice_shape));
      detesscast_requires_lane_split =
          detesscast_requires_lane_split ||
          (cvu_dtype(*input, detess->frame_type) == "BF16" && detess->align_c16 &&
           detess->cblock && last_op.kind == OpKind::Cast);
    } else {
      const auto fallback = first_op.input_shapes.empty() ? TensorShape{}
                                                          : first_op.input_shapes.front();
      if (!build_cvu_dense_desc(*input, fallback, {}, &input_descriptor, error)) {
        return std::nullopt;
      }
    }
    if (tess) {
      TensorShape frame_shape;
      if (tess_op && !tess_op->input_shapes.empty()) {
        frame_shape = tess_op->input_shapes.front();
      } else if (tess_op) {
        const auto* frame = plan.value(tess_op->inputs.front());
        if (frame && frame->logical_shape) frame_shape = *frame->logical_shape;
      }
      if (!build_cvu_tiled_desc(*output, frame_shape, tess->slice_shape, tess->frame_type,
                                tess->align_c16 || tess->cblock, &output_descriptor, error)) {
        return std::nullopt;
      }
      authored.slice_shapes.push_back(cvu_shape_from_tensor(tess->slice_shape));
    } else {
      const auto fallback = last_op.output_shapes.empty() ? TensorShape{}
                                                          : last_op.output_shapes.front();
      // Graph 227 executes detessellation and dequantization before any proved
      // address-only rank view.  Its physical endpoint descriptor therefore
      // has the registered Detess frame geometry even when the published
      // ValueSpec retains the post-view logical shape.
      const bool graph227_reverse_geometry =
          cohort->capability.graph_id == 227U && detess != nullptr && dequant != nullptr;
      const bool output_descriptor_built =
          graph227_reverse_geometry
              ? build_cvu_dense_desc_with_geometry(*output, detess->frame_shape, "HWC", {},
                                                   &output_descriptor, error)
              : build_cvu_dense_desc(*output, fallback, {}, &output_descriptor, error);
      if (!output_descriptor_built) {
        return std::nullopt;
      }
    }
    authored.input_tensors.push_back(input_descriptor);
    authored.output_tensors.push_back(output_descriptor);
    authored.input_shapes.push_back(cvu_shape(*input));
    authored.output_shapes.push_back(cvu_shape(*output));
    authored.runtime_output_logical_shapes.push_back(cvu_shape(*output));

    const auto input_name = "input_tensor_" + std::to_string(index);
    // Canonical facts select published outputs by their runtime output names.
    // Use the exact authored outer ValueSpec identity for every member rather
    // than introducing a second, synthetic naming domain.
    const auto& output_name = output->name;
    authored.runtime_input_names.push_back(input_name);
    authored.runtime_output_names.push_back(output_name);
    authored.physical_input_names.push_back(input_name);
    authored.physical_output_names.push_back(output_name);
    authored.published_output_names.push_back(output->name);
    authored.runtime_output_logical_index_list.push_back(static_cast<int>(index));
    authored.runtime_output_output_slot_list.push_back(static_cast<int>(index));
    authored.runtime_output_physical_index_list.push_back(static_cast<int>(index));
    authored.runtime_output_dtype_list.push_back(cvu_dtype(*output));
    authored.runtime_output_transport_kind_list.push_back(
        tess ? ProcessCvuOutputTransportKind::Packed : ProcessCvuOutputTransportKind::Dense);
    authored.runtime_output_semantic_kind_list.push_back(
        tess && quant ? ProcessCvuOutputSemanticKind::QuantTessTensor
        : tess         ? ProcessCvuOutputSemanticKind::TessellatedImage
        : quant        ? ProcessCvuOutputSemanticKind::QuantizedTensor
                       : ProcessCvuOutputSemanticKind::Tensor);
    authored.runtime_output_logical_layout_list.push_back(
        output->logical_layout.value_or(""));

    const auto append_qparams = [&](const std::vector<QuantizationSpec>& params) -> bool {
      if (params.size() != 1U) {
        return fail(error, "ProcessCVU quantized member requires exactly one scale/zero-point");
      }
      authored.q_scale_list.push_back(params.front().scale);
      authored.q_zp_list.push_back(params.front().zero_point);
      return true;
    };
    if (quant) {
      const auto rounding = cvu_rounding_mode(quant->rounding);
      if (!rounding || (uniform_rounding && *uniform_rounding != *rounding) ||
          !append_qparams(quant->channel_params)) {
        if (rounding && uniform_rounding && *uniform_rounding != *rounding) {
          fail(error, "ProcessCVU grouped quantize members disagree on rounding mode");
        }
        return std::nullopt;
      }
      uniform_rounding = *rounding;
    }
    if (dequant && !append_qparams(dequant->channel_params)) {
      return std::nullopt;
    }
  }

  authored.default_input_name = authored.runtime_input_names.front();
  authored.primary_output_name = authored.published_output_names.front();
  authored.input_dtype = cvu_dtype(*plan.value(cohort->members.front()->outer_inputs.front()));
  authored.output_dtype = cvu_dtype(*plan.value(cohort->members.front()->outer_outputs.front()));
  authored.out_dtype = authored.output_dtype;
  authored.round_off = uniform_rounding.value_or(0);
  authored.tessellate = (!authored.slice_shapes.empty() ? 1 : 0);
  authored.primary_output_transport_kind =
      authored.runtime_output_transport_kind_list.front();
  authored.primary_output_semantic_kind = authored.runtime_output_semantic_kind_list.front();
  if (!authored.q_scale_list.empty()) {
    authored.has_q_scale = true;
    authored.q_scale = authored.q_scale_list.front();
    authored.has_q_zp = true;
    authored.q_zp = authored.q_zp_list.front();
  }
  authored.opt_flags =
      stagesemantics::processcvu_detesscast_optimized_flags(detesscast_requires_lane_split);

  try {
    auto compiled =
        stagesemantics::build_processcvu_compiled_contract_from_runtime_config(authored);
    if (!apply_dmabuf_plan_processcvu_command_projection(
            plan, physical_plan, command_ids, arena, &compiled.payload,
            &compiled.runtime_contract, &compiled.exposed_view, error)) {
      return std::nullopt;
    }
    return compiled;
  } catch (const std::exception& ex) {
    fail(error, std::string("ProcessCVU command contract assembly failed: ") + ex.what());
    return std::nullopt;
  }
}

static std::optional<std::vector<PhysicalPortSource>>
resolve_mla_input_physical_sources_impl(
    const ModelExecutionPlan& plan, const std::size_t mla_stage_index,
    const FrameSlotArenaPlan* arena,
    std::span<const LogicalTensorStaticSpec> upstream_outputs, std::string* error) {
  const auto stage = plan.mla_stage(mla_stage_index);
  const auto stage_inputs = plan.backend_ports(mla_stage_index, BackendPortDirection::Input);
  if (!stage || stage_inputs.empty()) {
    fail(error, "MLA physical input projection references a missing or empty exact stage");
    return std::nullopt;
  }

  std::vector<PhysicalPortSource> result;
  result.reserve(stage_inputs.size());
  std::unordered_set<int> seen_sources;
  for (std::size_t port_index = 0; port_index < stage_inputs.size(); ++port_index) {
    const auto& port = stage_inputs[port_index];
    if (port.port_index != port_index) {
      fail(error, "MLA physical input projection has a sparse backend port order");
      return std::nullopt;
    }
    const auto* value = plan.value(port.value_id);
    if (!value) {
      fail(error, "MLA physical input projection references a missing graph value");
      return std::nullopt;
    }

    std::optional<int> source_physical_index;
    // All materialized internal values live in one retained frame-arena
    // DMA-BUF. MLA ports remain distinct backend bindings, but they select
    // different offsets of source physical memory zero instead of requiring
    // one fake GstMemory per semantic value.
    const auto root_id = root_value_id(plan, port.value_id);
    const bool arena_bound = arena && arena->region(root_id) != nullptr;
    const auto model_input =
        std::find(plan.model_inputs().begin(), plan.model_inputs().end(), port.value_id);
    if (arena_bound) {
      source_physical_index = 0;
    } else if (model_input != plan.model_inputs().end()) {
      const auto index = std::distance(plan.model_inputs().begin(), model_input);
      if (index > std::numeric_limits<int>::max()) {
        fail(error, "MLA model-input physical carrier index overflows int");
        return std::nullopt;
      }
      source_physical_index = static_cast<int>(index);
    } else {
      const auto* match = find_exact_upstream_output(*value, upstream_outputs, error);
      if (match && match->physical_index >= 0 && match->byte_offset == 0 &&
          match->size_bytes == port.physical_extent_bytes) {
        source_physical_index = match->physical_index;
      } else {
        source_physical_index =
            resolve_pack_parent_physical_source(plan, *value, upstream_outputs, error);
        if (!source_physical_index.has_value()) {
          if (error && error->empty()) {
            *error = "MLA input value '" + value->name +
                     "' has no exact upstream physical carrier or Pack proof";
          }
          return std::nullopt;
        }
      }
    }

    if (!arena_bound && !seen_sources.emplace(*source_physical_index).second) {
      fail(error, "MLA input projection aliases two IFM ports to one physical carrier");
      return std::nullopt;
    }
    result.push_back(PhysicalPortSource{port.value_id, *source_physical_index});
  }
  if (error) {
    error->clear();
  }
  return result;
}

std::optional<std::vector<PhysicalPortSource>> resolve_mla_input_physical_sources(
    const ModelExecutionPlan& plan, const std::size_t mla_stage_index,
    std::span<const LogicalTensorStaticSpec> upstream_outputs, std::string* error) {
  return resolve_mla_input_physical_sources_impl(plan, mla_stage_index, nullptr,
                                                 upstream_outputs, error);
}

std::optional<std::vector<PhysicalPortSource>> resolve_mla_input_physical_sources(
    const ModelExecutionPlan& plan, const std::size_t mla_stage_index,
    const FrameSlotArenaPlan& arena,
    std::span<const LogicalTensorStaticSpec> upstream_outputs, std::string* error) {
  return resolve_mla_input_physical_sources_impl(plan, mla_stage_index, &arena,
                                                 upstream_outputs, error);
}

std::optional<std::vector<PhysicalPortSource>>
resolve_mla_input_physical_sources(const ModelExecutionPlan& plan,
                                   std::span<const LogicalTensorStaticSpec> upstream_outputs,
                                   std::string* error) {
  if (plan.mla_stage_count() != 1U) {
    fail(error, "single-stage MLA input projection is ambiguous for a multi-stage plan");
    return std::nullopt;
  }
  return resolve_mla_input_physical_sources(plan, 0U, upstream_outputs, error);
}

MlaOutputCarrierPolicy select_mla_output_carrier_policy(
    const ModelExecutionPlan& plan, const PhysicalExecutionPlan& physical_plan,
    const std::size_t mla_stage_index) {
  const auto* stage = plan.mla_stage(mla_stage_index);
  if (!stage || stage->key.op_id >= physical_plan.command_for_semantic_op.size()) {
    return MlaOutputCarrierPolicy::SharedFrameArena;
  }
  const auto command_id = physical_plan.command_for_semantic_op[stage->key.op_id];
  if (!command_id.has_value() || *command_id >= physical_plan.commands.size()) {
    return MlaOutputCarrierPolicy::SharedFrameArena;
  }
  const auto& command = physical_plan.commands[*command_id];
  if (command.engine != PhysicalEngine::Mla || !command.successors.empty() ||
      command.outputs.empty() || plan.model_outputs().empty()) {
    return MlaOutputCarrierPolicy::SharedFrameArena;
  }

  std::vector<ValueId> physical_roots;
  physical_roots.reserve(command.outputs.size());
  for (const auto output : command.outputs) {
    physical_roots.push_back(root_value_id(plan, output));
  }
  std::sort(physical_roots.begin(), physical_roots.end());
  physical_roots.erase(std::unique(physical_roots.begin(), physical_roots.end()),
                       physical_roots.end());

  std::vector<ValueId> public_roots;
  public_roots.reserve(plan.model_outputs().size());
  for (const auto& output : plan.model_outputs()) {
    public_roots.push_back(root_value_id(plan, output.value_id));
  }
  std::sort(public_roots.begin(), public_roots.end());
  public_roots.erase(std::unique(public_roots.begin(), public_roots.end()),
                     public_roots.end());

  // Exact equality is intentional.  A subset would detach storage while a
  // hidden device consumer still addresses the graph arena; a superset would
  // make a different producer's public carrier appear to be owned by MLA.
  if (physical_roots != public_roots) {
    return MlaOutputCarrierPolicy::SharedFrameArena;
  }

  std::unordered_set<CarrierId> external_carriers;
  for (const auto input_id : plan.model_inputs()) {
    const auto* value = plan.value(root_value_id(plan, input_id));
    if (value && value->storage_binding.has_value()) {
      external_carriers.emplace(value->storage_binding->carrier_id);
    }
  }
  std::unordered_set<CarrierId> output_carriers;
  for (const auto root : physical_roots) {
    const auto* value = plan.value(root);
    if (!value || !value->storage_binding.has_value()) {
      return MlaOutputCarrierPolicy::SharedFrameArena;
    }
    output_carriers.emplace(value->storage_binding->carrier_id);
  }
  const bool has_retained_internal_carrier =
      std::any_of(physical_plan.commands.begin(), physical_plan.commands.end(),
                  [&](const auto& physical_command) {
                    const auto retained = [&](const ValueId id) {
                      const auto* value = plan.value(root_value_id(plan, id));
                      return value && value->storage_binding.has_value() &&
                             !external_carriers.contains(
                                 value->storage_binding->carrier_id) &&
                             !output_carriers.contains(
                                 value->storage_binding->carrier_id);
                    };
                    return std::any_of(physical_command.inputs.begin(),
                                       physical_command.inputs.end(), retained) ||
                           std::any_of(physical_command.outputs.begin(),
                                       physical_command.outputs.end(), retained);
                  });
  return has_retained_internal_carrier
             ? MlaOutputCarrierPolicy::SeparateCpuVisible
             : MlaOutputCarrierPolicy::SharedFrameArena;
}

std::vector<ValueId> detached_mla_output_roots(
    const ModelExecutionPlan& plan,
    const PhysicalExecutionPlan& physical_plan) {
  std::vector<ValueId> roots;
  for (std::size_t stage_index = 0; stage_index < plan.mla_stage_count();
       ++stage_index) {
    if (select_mla_output_carrier_policy(plan, physical_plan, stage_index) !=
        MlaOutputCarrierPolicy::SeparateCpuVisible) {
      continue;
    }
    for (const auto& port :
         plan.backend_ports(stage_index, BackendPortDirection::Output)) {
      roots.push_back(root_value_id(plan, port.value_id));
    }
  }
  std::sort(roots.begin(), roots.end());
  roots.erase(std::unique(roots.begin(), roots.end()), roots.end());
  return roots;
}

MlaOutputCarrierPolicy mla_output_carrier_policy_from_arena(
    const ModelExecutionPlan& plan, const FrameSlotArenaPlan& arena,
    const std::size_t mla_stage_index) {
  const auto outputs =
      plan.backend_ports(mla_stage_index, BackendPortDirection::Output);
  if (outputs.empty()) {
    return MlaOutputCarrierPolicy::SharedFrameArena;
  }
  return std::all_of(outputs.begin(), outputs.end(), [&](const auto& port) {
           return arena.is_detached_root(root_value_id(plan, port.value_id));
         })
             ? MlaOutputCarrierPolicy::SeparateCpuVisible
             : MlaOutputCarrierPolicy::SharedFrameArena;
}

bool apply_dmabuf_plan_contract_projection(const ModelExecutionPlan& plan,
                                           const std::size_t mla_stage_index,
                                           const FrameSlotArenaPlan& arena,
                                           const MlaOutputCarrierPolicy output_carrier_policy,
                                           MlaStaticContract* contract,
                                           std::span<const PhysicalPortSource> input_sources,
                                           std::string* error) {
  if (!contract) {
    return fail(error, "MLA contract projection is null");
  }
  const auto* stage = plan.mla_stage(mla_stage_index);
  const auto inputs = plan.backend_ports(mla_stage_index, BackendPortDirection::Input);
  const auto outputs = plan.backend_ports(mla_stage_index, BackendPortDirection::Output);
  if (!stage || inputs.empty() || outputs.empty()) {
    return fail(error, "MLA projection requires one exact non-empty backend stage");
  }
  if (output_carrier_policy !=
      mla_output_carrier_policy_from_arena(plan, arena, mla_stage_index)) {
    return fail(error,
                "MLA output-carrier policy contradicts the frame-arena authority");
  }

  const auto validate = [&](const std::span<const BackendPortSpec> ports,
                            const std::vector<PhysicalBufferStaticSpec>& physical,
                            const char* kind) {
    if (ports.size() != physical.size()) {
      return fail(error, std::string(kind) + " projection arity mismatch");
    }
    for (std::size_t index = 0; index < ports.size(); ++index) {
      const auto& port = ports[index];
      const auto& projected = physical[index];
      const auto* value = plan.value(port.value_id);
      if (port.port_index != index || projected.physical_index != static_cast<int>(index) ||
          projected.size_bytes != (value ? value->required_bytes : 0U) || !value ||
          projected.segment_name.empty() ||
          projected.segment_name != value->name) {
        return fail(error, std::string(kind) + "[" + std::to_string(index) +
                               "] differs from decoded MPK+ELF plan");
      }
    }
    return true;
  };

  if (!validate(inputs, contract->physical_inputs, "IFM") ||
      !validate(outputs, contract->dispatcher_physical_outputs, "OFM")) {
    return false;
  }

  // Alignment is an execution-plan property. The decoded legacy stage
  // contract intentionally does not invent it; project the Core-owned value
  // only after the MPK/ELF port identity has been reconciled exactly.
  for (std::size_t index = 0; index < inputs.size(); ++index) {
    contract->physical_inputs[index].size_bytes = inputs[index].physical_extent_bytes;
    contract->physical_inputs[index].required_alignment_bytes =
        inputs[index].required_alignment_bytes;
  }
  for (std::size_t index = 0; index < outputs.size(); ++index) {
    contract->dispatcher_physical_outputs[index].size_bytes =
        outputs[index].physical_extent_bytes;
    contract->dispatcher_physical_outputs[index].required_alignment_bytes =
        outputs[index].required_alignment_bytes;
  }

  if (input_sources.size() != inputs.size()) {
    return fail(error, "IFM projection has no complete physical carrier map");
  }
  std::unordered_set<int> seen_input_sources;
  contract->input_bindings.clear();
  contract->input_bindings.reserve(inputs.size());
  for (std::size_t index = 0; index < inputs.size(); ++index) {
    const auto& port = inputs[index];
    const auto& projected = input_sources[index];
    auto& physical = contract->physical_inputs[index];
    const auto root_id = root_value_id(plan, port.value_id);
    // OFM ownership never changes IFM placement. Internal inputs remain exact
    // offsets in the retained upstream frame arena; only true public inputs
    // lack a region and select TensorBuffer physical bindings independently.
    const auto* input_region = arena.region(root_id);
    if (projected.value_id != port.value_id || projected.source_physical_index < 0 ||
        (!input_region && !seen_input_sources.emplace(projected.source_physical_index).second)) {
      return fail(error, "IFM projection has an invalid or aliased physical carrier");
    }
    // Every retained arena region is a byte range in one physical DMA-BUF.
    // The upstream logical selector remains useful for route identity, but it
    // must not leak into the direct physical binding as a second memory index.
    physical.source_physical_index = input_region ? 0 : projected.source_physical_index;
    std::uint64_t source_byte_offset = 0U;
    if (input_region) {
      const auto* port_value = plan.value(port.value_id);
      const auto* binding = port_value && port_value->storage_binding.has_value()
                                ? &*port_value->storage_binding
                                : nullptr;
      if (!binding || binding->carrier_id != input_region->carrier_id ||
          binding->byte_offset > input_region->size_bytes ||
          port.physical_extent_bytes > input_region->size_bytes - binding->byte_offset ||
          !checked_add(input_region->byte_offset, binding->byte_offset, &source_byte_offset) ||
          source_byte_offset >
              static_cast<std::uint64_t>(std::numeric_limits<std::int64_t>::max()) ||
          source_byte_offset % port.required_alignment_bytes != 0U) {
        return fail(error, "IFM frame-slot view offset is invalid or unrepresentable");
      }
      physical.source_byte_offset = static_cast<std::int64_t>(source_byte_offset);
    } else {
      // Direct QMLA inputs remain imported public DMA-BUFs; ProcessMLA is the
      // first frame-arena producer in that route.
      physical.source_byte_offset = 0;
    }

    TensorStaticSpec* logical_input = nullptr;
    for (auto& candidate : contract->logical_inputs) {
      if (candidate.tensor_index != static_cast<int>(index)) {
        continue;
      }
      if (logical_input != nullptr) {
        return fail(error, "IFM projection has ambiguous logical input metadata");
      }
      logical_input = &candidate;
    }
    if (!logical_input) {
      return fail(error, "IFM projection has no logical input metadata");
    }
    logical_input->parent_carrier = false;

    InputBindingStaticSpec binding;
    binding.sink_pad_index = 0;
    binding.local_logical_input_index = static_cast<int>(index);
    binding.src_logical_output_index = projected.source_physical_index;
    binding.src_output_slot = projected.source_physical_index;
    binding.src_physical_output_index = physical.source_physical_index;
    binding.src_physical_size_bytes = port.physical_extent_bytes;
    binding.src_physical_byte_offset = static_cast<std::int64_t>(source_byte_offset);
    binding.required = true;
    const auto* value = plan.value(port.value_id);
    binding.cm_input_name = value ? value->name : std::string{};
    binding.source_segment_name = binding.cm_input_name;
    contract->input_bindings.push_back(std::move(binding));
  }

  // ProcessMLA always allocates/submits the real backend OFMs. For a packed
  // 1/1 stage, Unpack and Slice have already lowered to immutable address/read
  // expressions, so publish those logical views over the one physical OFM.
  // No runtime unpack/slice job or materialization is introduced.
  contract->physical_outputs.clear();
  contract->logical_outputs.clear();
  contract->physical_outputs.reserve(outputs.size());
  std::uint64_t separate_output_cursor = 0U;
  std::uint64_t separate_output_alignment = 0U;
  for (std::size_t index = 0; index < outputs.size(); ++index) {
    const auto& port = outputs[index];
    const auto* value = plan.value(port.value_id);
    if (!value || port.port_index != index ||
        index > static_cast<std::size_t>(std::numeric_limits<int>::max())) {
      return fail(error, "OFM projection has an invalid backend port");
    }
    const int slot = static_cast<int>(index);

    PhysicalBufferStaticSpec physical;
    physical.physical_index = slot;
    physical.allocator_index = slot;
    physical.source_physical_index = slot;
    physical.size_bytes = port.physical_extent_bytes;
    physical.source_byte_offset = 0;
    physical.device_kind = DeviceKind::Mla;
    physical.segment_name = value->name;
    physical.required_alignment_bytes = port.required_alignment_bytes;
    if (output_carrier_policy == MlaOutputCarrierPolicy::SeparateCpuVisible) {
      const auto offset = checked_align_up(separate_output_cursor,
                                           port.required_alignment_bytes);
      if (!offset || *offset >
                         static_cast<std::uint64_t>(
                             std::numeric_limits<std::int64_t>::max()) ||
          !checked_add(*offset, port.physical_extent_bytes,
                       &separate_output_cursor)) {
        return fail(error, "separate MLA output carrier layout overflows");
      }
      physical.source_byte_offset = static_cast<std::int64_t>(*offset);
      separate_output_alignment = std::max(
          separate_output_alignment, static_cast<std::uint64_t>(port.required_alignment_bytes));
    } else if (!assign_physical_region(arena, *value, &physical, error)) {
      return false;
    }
    contract->physical_outputs.push_back(std::move(physical));
  }

  if (output_carrier_policy == MlaOutputCarrierPolicy::SeparateCpuVisible) {
    // A terminal publication chain has no materializing consumer for the old
    // consumer-oriented read-view query to discover.  The model-output table
    // is the ordered public authority; map each composed root-relative view
    // back to one exact backend port instead of silently publishing raw OFMs.
    if (!project_terminal_mla_publications(plan, outputs, contract, error)) {
      return false;
    }
  } else {
    std::vector<const ValueSpec*> packed_read_views;
    if (outputs.size() == 1U) {
      packed_read_views = read_views_consumed_after_mla(plan, outputs.front().value_id);
    }
    // One affine child is still a read expression.  Cardinality must not decide
    // whether its root-relative offset/strides survive projection: a monolithic
    // OFM followed by one Slice is just as much a logical view as an Unpack with
    // many children.
    if (!packed_read_views.empty()) {
      contract->logical_outputs.reserve(packed_read_views.size());
      for (std::size_t index = 0; index < packed_read_views.size(); ++index) {
        const auto& value = *packed_read_views[index];
        const auto& expression = *value.read_expression;
        const auto* producer = producer_of(plan, value.id);
        if (index > static_cast<std::size_t>(std::numeric_limits<int>::max()) ||
            expression.byte_offset >
                static_cast<std::uint64_t>(std::numeric_limits<std::int64_t>::max()) ||
            !value.logical_shape.has_value() || !producer) {
          return fail(error, "packed OFM read view cannot be represented in the static manifest");
        }
        const auto transport =
            resolve_publication_transport_view(plan, value, error);
        if (expression.stride_bytes.empty() || !transport.has_value() ||
            expression.byte_offset > outputs.front().physical_extent_bytes ||
            transport->physical_span >
                outputs.front().physical_extent_bytes - expression.byte_offset) {
          if (!transport.has_value()) {
            return false;
          }
          return fail(error, "packed OFM read view has no exact typed shape/stride contract");
        }
        const int slot = static_cast<int>(index);
        LogicalTensorStaticSpec logical;
        logical.logical_index = slot;
        logical.backend_output_index = 0;
        logical.physical_index = 0;
        logical.output_slot = slot;
        logical.tensor_index = slot;
        logical.byte_offset = static_cast<std::int64_t>(expression.byte_offset);
        logical.size_bytes = value.required_bytes;
        logical.shape = *value.logical_shape;
        logical.stride_bytes = expression.stride_bytes;
        logical.dtype = transport->carrier_dtype;
        logical.dtype_source = DTypeSource::InternalContract;
        logical.logical_name = value.name;
        logical.backend_name = value.name;
        logical.segment_name = plan.value(outputs.front().value_id)->name;
        if (transport->carrier_layout) {
          logical.layout = *transport->carrier_layout;
        }
        contract->logical_outputs.push_back(std::move(logical));
      }
    } else {
      contract->logical_outputs.reserve(outputs.size());
      for (std::size_t index = 0; index < outputs.size(); ++index) {
        const auto& port = outputs[index];
        const auto* value = plan.value(port.value_id);
        const int slot = static_cast<int>(index);

        LogicalTensorStaticSpec logical;
        logical.logical_index = slot;
        logical.backend_output_index = slot;
        logical.physical_index = slot;
        logical.output_slot = slot;
        logical.tensor_index = slot;
        logical.byte_offset = 0;
        logical.size_bytes = value->required_bytes;
        logical.logical_name = value->name;
        logical.backend_name = value->name;
        logical.segment_name = value->name;
        std::uint64_t logical_span = 0U;
        if (!value->logical_dtype || !value->logical_shape ||
            !exact_logical_tensor_span(*value->logical_shape, {}, *value->logical_dtype,
                                       &logical_span) ||
            logical_span > port.physical_extent_bytes) {
          return fail(error,
                      "MLA OFM '" + value->name +
                          "' has no exact typed logical output contract (dtype=" +
                          value->logical_dtype.value_or("<missing>") + ", rank=" +
                          std::to_string(value->logical_shape ? value->logical_shape->size() : 0U) +
                          ", logical_span=" + std::to_string(logical_span) +
                          ", physical_extent=" + std::to_string(port.physical_extent_bytes) + ")");
        }
        logical.dtype = *value->logical_dtype;
        logical.dtype_source = DTypeSource::InternalContract;
        logical.shape = *value->logical_shape;
        if (value->storage_binding && !value->storage_binding->stride_bytes.empty()) {
          logical.stride_bytes = value->storage_binding->stride_bytes;
        }
        if (value->logical_layout) {
          logical.layout = *value->logical_layout;
        }
        contract->logical_outputs.push_back(std::move(logical));
      }
    }
  }

  std::vector<std::string> ifm_symbols;
  std::vector<std::string> ofm_symbols;
  ifm_symbols.reserve(inputs.size());
  ofm_symbols.reserve(outputs.size());
  for (const auto& port : inputs) {
    ifm_symbols.push_back(port.elf_symbol);
  }
  for (const auto& port : outputs) {
    ofm_symbols.push_back(port.elf_symbol);
  }
  contract->elf_ifm_symbol_names = std::move(ifm_symbols);
  contract->elf_ofm_symbol_names = std::move(ofm_symbols);
  contract->consumer_keeps_distinct_physical_inputs = inputs.size() > 1U;
  if (output_carrier_policy == MlaOutputCarrierPolicy::SeparateCpuVisible) {
    const auto allocation =
        checked_align_up(separate_output_cursor, separate_output_alignment);
    if (!allocation || *allocation == 0U) {
      return fail(error, "separate MLA output carrier allocation is invalid");
    }
    contract->frame_arena_size_bytes = *allocation;
    contract->frame_arena_storage_domain = ArenaStorageDomain::Dms;
    contract->frame_arena_provenance = ArenaAllocationProvenance::CoreAllocated;
    contract->frame_arena_required_device_access =
        static_cast<std::uint32_t>(ArenaDeviceAccess::Mla) |
        static_cast<std::uint32_t>(ArenaDeviceAccess::CpuA65);
    contract->frame_arena_escape_policy = ArenaEscapePolicy::CpuMappablePublic;
    contract->frame_arena_role = FrameArenaRole::Allocate;
  } else {
    contract->frame_arena_size_bytes = arena.allocation_bytes();
    contract->frame_arena_storage_domain = arena.placement().domain;
    contract->frame_arena_provenance = arena.placement().provenance;
    contract->frame_arena_required_device_access =
        arena.placement().required_device_access;
    contract->frame_arena_escape_policy = arena.placement().escape;
    const bool has_internal_input =
        std::any_of(inputs.begin(), inputs.end(), [&](const BackendPortSpec& port) {
          return arena.region(root_value_id(plan, port.value_id)) != nullptr;
        });
    contract->frame_arena_role =
        has_internal_input ? FrameArenaRole::ReuseInput : FrameArenaRole::Allocate;
  }
  if (error) {
    error->clear();
  }
  return true;
}

bool apply_dmabuf_plan_contract_projection(const ModelExecutionPlan& plan,
                                           const std::size_t mla_stage_index,
                                           const FrameSlotArenaPlan& arena,
                                           MlaStaticContract* contract,
                                           const std::span<const PhysicalPortSource> input_sources,
                                           std::string* error) {
  const auto output_carrier_policy =
      mla_output_carrier_policy_from_arena(plan, arena, mla_stage_index);
  return apply_dmabuf_plan_contract_projection(plan, mla_stage_index, arena, output_carrier_policy,
                                               contract, input_sources, error);
}

bool apply_dmabuf_plan_contract_projection(const ModelExecutionPlan& plan,
                                           const std::size_t mla_stage_index,
                                           MlaStaticContract* contract,
                                           const std::span<const PhysicalPortSource> input_sources,
                                           std::string* error) {
  auto arena = compile_frame_arena(plan, error);
  if (!arena) {
    return false;
  }
  const auto output_carrier_policy =
      mla_output_carrier_policy_from_arena(plan, *arena, mla_stage_index);
  return apply_dmabuf_plan_contract_projection(plan, mla_stage_index, *arena, output_carrier_policy,
                                               contract, input_sources, error);
}

bool apply_dmabuf_plan_contract_projection(const ModelExecutionPlan& plan,
                                           MlaStaticContract* contract,
                                           const std::span<const PhysicalPortSource> input_sources,
                                           std::string* error) {
  if (plan.mla_stage_count() != 1U) {
    return fail(error, "single-stage MLA contract projection is ambiguous for a multi-stage plan");
  }
  return apply_dmabuf_plan_contract_projection(plan, 0U, contract, input_sources, error);
}

bool apply_dmabuf_plan_processcvu_contract_projection(
    const ModelExecutionPlan& plan, const std::size_t adjacent_mla_stage_index,
    const FrameSlotArenaPlan& arena, ProcessCvuMlaBoundary boundary,
    ProcessCvuStagePayload* payload, ::simaai::neat::CompiledRuntimeContract* runtime,
    ::simaai::neat::CompiledExposedView* exposed_view, std::string* error) {
  if (!payload || !runtime || !exposed_view) {
    return fail(error, "ProcessCVU contract projection is null");
  }
  if (!apply_processcvu_implementation_contract(payload, error)) {
    return false;
  }

  const auto wanted_direction = boundary == ProcessCvuMlaBoundary::Inputs
                                    ? BackendPortDirection::Input
                                    : BackendPortDirection::Output;
  const auto ports = plan.backend_ports(adjacent_mla_stage_index, wanted_direction);
  if (ports.empty() || runtime->logical_outputs.empty()) {
    return fail(error, "ProcessCVU alignment projection has an empty MLA boundary");
  }
  for (std::size_t index = 0; index < ports.size(); ++index) {
    const auto alignment = ports[index].required_alignment_bytes;
    if (ports[index].port_index != index || alignment == 0U ||
        (alignment & (alignment - 1U)) != 0U) {
      return fail(error, "ProcessCVU alignment projection has an invalid MLA port order or "
                         "alignment");
    }
  }
  const auto output_count = runtime->logical_outputs.size();
  if (output_count != ports.size() && ports.size() != 1U) {
    return fail(error, "ProcessCVU materialized outputs cannot be reconciled with the MLA "
                       "boundary port order");
  }

  const DeviceKind output_device = runtime->physical_outputs.empty()
                                       ? DeviceKind::Evxx
                                       : runtime->physical_outputs.front().device_kind;
  const std::uint64_t output_memory_flags =
      runtime->physical_outputs.empty() ? 0U : runtime->physical_outputs.front().memory_flags;
  std::vector<PhysicalBufferStaticSpec> projected_outputs;
  projected_outputs.reserve(output_count);
  bool exact_pack_children = false;
  for (std::size_t index = 0; index < output_count; ++index) {
    auto placement = resolve_cvu_output_placement(
        plan, arena, boundary, ports, runtime->logical_outputs[index], index, output_count, error);
    if (!placement || !placement->value ||
        placement->byte_offset >
            static_cast<std::uint64_t>(std::numeric_limits<std::int64_t>::max())) {
      return false;
    }
    PhysicalBufferStaticSpec physical;
    physical.physical_index = static_cast<int>(index);
    physical.allocator_index = static_cast<int>(index);
    physical.source_physical_index = static_cast<int>(index);
    physical.size_bytes = placement->value->required_bytes;
    physical.source_byte_offset = static_cast<std::int64_t>(placement->byte_offset);
    physical.device_kind = output_device;
    physical.memory_flags = output_memory_flags;
    physical.segment_name = placement->value->name;
    physical.required_alignment_bytes = placement->required_alignment_bytes;
    exact_pack_children = exact_pack_children || placement->required_alignment_bytes == 16U;
    projected_outputs.push_back(std::move(physical));

    auto& logical = runtime->logical_outputs[index];
    logical.backend_output_index = static_cast<int>(index);
    logical.physical_index = static_cast<int>(index);
    logical.byte_offset = 0;
    logical.segment_name = placement->value->name;
  }
  runtime->physical_outputs = std::move(projected_outputs);
  if (payload->runtime_output_physical_index_list.size() == output_count) {
    for (std::size_t index = 0; index < output_count; ++index) {
      payload->runtime_output_physical_index_list[index] = static_cast<int>(index);
    }
  }
  if (!exact_pack_children) {
    for (auto& exposed : exposed_view->exposed_logical_outputs) {
      const auto projected =
          std::find_if(runtime->logical_outputs.begin(), runtime->logical_outputs.end(),
                       [&](const LogicalTensorStaticSpec& logical) {
                         return logical.logical_index == exposed.logical_index;
                       });
      if (projected == runtime->logical_outputs.end()) {
        return fail(error, "ProcessCVU exposed output has no projected runtime output");
      }
      exposed.backend_output_index = projected->backend_output_index;
      exposed.physical_index = projected->physical_index;
      exposed.byte_offset = projected->byte_offset;
      exposed.segment_name = projected->segment_name;
    }
    for (auto& route : exposed_view->exposed_output_order) {
      const auto logical = std::find_if(exposed_view->exposed_logical_outputs.begin(),
                                        exposed_view->exposed_logical_outputs.end(),
                                        [&](const LogicalTensorStaticSpec& output) {
                                          return output.logical_index == route.logical_output_index;
                                        });
      if (logical != exposed_view->exposed_logical_outputs.end()) {
        route.segment_name = logical->segment_name;
      }
    }
  }

  runtime->frame_arena_size_bytes = arena.allocation_bytes();
  runtime->frame_arena_storage_domain = arena.placement().domain;
  runtime->frame_arena_provenance = arena.placement().provenance;
  runtime->frame_arena_required_device_access =
      arena.placement().required_device_access;
  runtime->frame_arena_escape_policy = arena.placement().escape;
  runtime->frame_arena_role = boundary == ProcessCvuMlaBoundary::Inputs
                                  ? FrameArenaRole::Allocate
                                  : FrameArenaRole::ReuseInput;
  runtime->consumer_keeps_distinct_physical_inputs =
      boundary == ProcessCvuMlaBoundary::Inputs && ports.size() > 1U;

  payload->dmabuf_plan_contract = true;
  if (error) {
    error->clear();
  }
  return true;
}

bool apply_dmabuf_plan_processcvu_command_projection(
    const ModelExecutionPlan& plan, const PhysicalExecutionPlan& physical_plan,
    const std::span<const PhysicalCommandId> command_ids,
    const FrameSlotArenaPlan& arena, ProcessCvuStagePayload* payload,
    ::simaai::neat::CompiledRuntimeContract* runtime,
    ::simaai::neat::CompiledExposedView* exposed_view, std::string* error) {
  if (!payload || !runtime || !exposed_view || command_ids.empty()) {
    return fail(error, "ProcessCVU command projection is null or references a missing command");
  }
  const auto cohort = resolve_physical_cvu_cohort(plan, physical_plan, command_ids, error);
  if (!cohort) {
    return false;
  }

  const OpSpec* first = nullptr;
  std::vector<ValueId> inputs;
  std::vector<ValueId> outputs;
  inputs.reserve(cohort->members.size());
  outputs.reserve(cohort->members.size());
  for (const auto* member : cohort->members) {
    first = first ? first : &plan.ops()[member->semantic_chain.front()];
    inputs.push_back(member->outer_inputs.front());
    outputs.push_back(member->outer_outputs.front());
  }
  if (!apply_processcvu_implementation_contract(payload, error,
                                                cohort->capability.graph_id)) {
    return false;
  }
  if (static_cast<std::uint32_t>(payload->graph_id) != cohort->capability.graph_id ||
      payload->maximum_members != cohort->capability.maximum_members ||
      payload->graph_name != cohort->capability.canonical_token) {
    return fail(error, "ProcessCVU payload contradicts its physical implementation");
  }
  if (payload->input_tensors.size() != cohort->members.size() ||
      payload->output_tensors.size() != cohort->members.size() ||
      runtime->logical_inputs.size() != cohort->members.size() ||
      runtime->input_bindings.size() != cohort->members.size() ||
      runtime->logical_outputs.size() != cohort->members.size()) {
    return fail(error, "ProcessCVU command cohort outputs disagree with its compiled "
                       "runtime contract: plan=" + std::to_string(cohort->members.size()) +
                       " runtime=" + std::to_string(runtime->logical_outputs.size()));
  }

  const DeviceKind input_device = runtime->physical_inputs.empty()
                                      ? DeviceKind::Evxx
                                      : runtime->physical_inputs.front().device_kind;
  const std::uint64_t input_memory_flags =
      runtime->physical_inputs.empty() ? 0U : runtime->physical_inputs.front().memory_flags;
  std::vector<PhysicalBufferStaticSpec> physical_inputs;
  physical_inputs.reserve(inputs.size());
  for (std::size_t index = 0; index < inputs.size(); ++index) {
    const auto* value = plan.value(inputs[index]);
    const auto* binding = value && value->storage_binding ? &*value->storage_binding : nullptr;
    const auto* carrier = binding ? plan.carrier(binding->carrier_id) : nullptr;
    const auto* region = value ? arena.region(root_value_id(plan, value->id)) : nullptr;
    const std::uint64_t arena_offset = region ? region->byte_offset : 0U;
    if (!value || !binding || !carrier ||
        index > static_cast<std::size_t>(std::numeric_limits<int>::max()) ||
        binding->physical_span == 0U || carrier->required_alignment_bytes == 0U ||
        binding->byte_offset > std::numeric_limits<std::uint64_t>::max() - arena_offset ||
        arena_offset + binding->byte_offset >
            static_cast<std::uint64_t>(std::numeric_limits<std::int64_t>::max())) {
      return fail(error, "ProcessCVU command input has no exact carrier");
    }

    const int physical_index = static_cast<int>(index);
    PhysicalBufferStaticSpec physical;
    physical.physical_index = physical_index;
    physical.allocator_index = physical_index;
    physical.source_physical_index = physical_index;
    physical.size_bytes = binding->physical_span;
    physical.source_byte_offset =
        static_cast<std::int64_t>(arena_offset + binding->byte_offset);
    physical.device_kind = input_device;
    physical.memory_flags = input_memory_flags;
    physical.segment_name = "value_" + std::to_string(value->id);
    physical.required_alignment_bytes = carrier->required_alignment_bytes;
    physical_inputs.push_back(std::move(physical));

    auto& logical = runtime->logical_inputs[index];
    logical.logical_index = physical_index;
    logical.backend_input_index = physical_index;
    logical.physical_index = physical_index;
    const bool offset_view =
        value->read_expression.has_value() || binding->kind == StorageBindingKind::View ||
        binding->source_value_id.has_value() || binding->byte_offset != 0U;
    // Tiled C16 lane layout is described by the tensor descriptor and graph225
    // option flags. It is not a request to materialize/repack the input.
    logical.materialization_kind = offset_view ? TensorMaterializationKind::OffsetView
                                               : TensorMaterializationKind::Direct;
    logical.byte_offset = 0;
    logical.size_bytes = value->required_bytes;
    logical.stride_bytes = binding->stride_bytes;
    logical.logical_name = value->name;
    logical.backend_name = value->name;
    logical.segment_name = physical_inputs.back().segment_name;
    if (value->logical_dtype) {
      logical.dtype = *value->logical_dtype;
    }
    if (value->logical_shape) {
      logical.shape = *value->logical_shape;
    }
    if (value->logical_layout) {
      logical.layout = *value->logical_layout;
    }
    const bool application_boundary_input =
        std::find(plan.model_inputs().begin(), plan.model_inputs().end(), value->id) !=
        plan.model_inputs().end();
    if (application_boundary_input && logical.shape.size() >= 4U &&
        logical.shape.front() == 1) {
      // ModelExecutionPlan retains AFE's explicit N=1 semantic axis.  Neat's
      // frame-at-a-time application tensor/caps boundary is intentionally
      // unbatched, while the CVU descriptor still uses the original op shape.
      // Project only the wire-facing logical view; storage bytes and payload
      // descriptor geometry remain unchanged.
      logical.shape.erase(logical.shape.begin());
      if (logical.stride_bytes.size() == logical.shape.size() + 1U) {
        logical.stride_bytes.erase(logical.stride_bytes.begin());
      }
    }

    auto& route = runtime->input_bindings[index];
    // Every frame-arena region arrives on the one upstream arena carrier pad;
    // the source selectors below identify the member's exact logical/physical
    // view within that carrier. Public carriers remain distinct input pads.
    route.sink_pad_index = region ? 0 : physical_index;
    route.local_logical_input_index = physical_index;
    route.src_logical_output_index = physical_index;
    route.src_output_slot = physical_index;
    route.src_physical_output_index = physical_index;
    route.src_physical_size_bytes = binding->physical_span;
    route.src_physical_byte_offset = 0;
    route.required = true;
    route.cm_input_name = value->name;
    route.source_segment_name = physical_inputs.back().segment_name;
  }
  runtime->physical_inputs = std::move(physical_inputs);

  const DeviceKind output_device = runtime->physical_outputs.empty()
                                       ? DeviceKind::Evxx
                                       : runtime->physical_outputs.front().device_kind;
  const std::uint64_t output_memory_flags =
      runtime->physical_outputs.empty() ? 0U : runtime->physical_outputs.front().memory_flags;
  std::vector<PhysicalBufferStaticSpec> physical_outputs;
  physical_outputs.reserve(outputs.size());
  for (std::size_t index = 0; index < outputs.size(); ++index) {
    const auto* value = plan.value(outputs[index]);
    const auto* binding = value && value->storage_binding ? &*value->storage_binding : nullptr;
    const auto* carrier = binding ? plan.carrier(binding->carrier_id) : nullptr;
    const auto* region = value ? arena.region(root_value_id(plan, value->id)) : nullptr;
    if (!value || !binding || !carrier || !region ||
        index > static_cast<std::size_t>(std::numeric_limits<int>::max()) ||
        binding->byte_offset > std::numeric_limits<std::uint64_t>::max() - region->byte_offset ||
        region->byte_offset + binding->byte_offset >
            static_cast<std::uint64_t>(std::numeric_limits<std::int64_t>::max()) ||
        binding->physical_span == 0U || carrier->required_alignment_bytes == 0U ||
        (region->byte_offset + binding->byte_offset) %
                carrier->required_alignment_bytes !=
            0U) {
      return fail(error, "ProcessCVU command output has no exact frame-arena carrier");
    }
    const int physical_index = static_cast<int>(index);
    PhysicalBufferStaticSpec physical;
    physical.physical_index = physical_index;
    physical.allocator_index = physical_index;
    physical.source_physical_index = physical_index;
    physical.size_bytes = binding->physical_span;
    physical.source_byte_offset = static_cast<std::int64_t>(
        region->byte_offset + binding->byte_offset);
    physical.device_kind = output_device;
    physical.memory_flags = output_memory_flags;
    physical.segment_name = "value_" + std::to_string(value->id);
    physical.required_alignment_bytes = carrier->required_alignment_bytes;
    physical_outputs.push_back(std::move(physical));

    auto& logical = runtime->logical_outputs[index];
    logical.logical_index = static_cast<int>(index);
    logical.backend_output_index = static_cast<int>(index);
    logical.physical_index = physical_index;
    logical.output_slot = static_cast<int>(index);
    logical.tensor_index = static_cast<int>(index);
    logical.byte_offset = 0;
    logical.size_bytes = value->required_bytes;
    logical.stride_bytes = binding->stride_bytes;
    logical.logical_name = value->name;
    logical.backend_name = value->name;
    logical.segment_name = physical_outputs[static_cast<std::size_t>(physical_index)].segment_name;
    if (value->logical_dtype) {
      logical.dtype = *value->logical_dtype;
    }
    if (value->logical_shape) {
      logical.shape = *value->logical_shape;
    }
    if (value->logical_layout) {
      logical.layout = *value->logical_layout;
    }
  }
  runtime->physical_outputs = std::move(physical_outputs);

  for (auto& exposed : exposed_view->exposed_logical_outputs) {
    const auto match = std::find_if(runtime->logical_outputs.begin(), runtime->logical_outputs.end(),
                                    [&](const LogicalTensorStaticSpec& value) {
                                      return value.logical_index == exposed.logical_index;
                                    });
    if (match == runtime->logical_outputs.end()) {
      return fail(error, "ProcessCVU exposed output has no exact command output");
    }
    exposed = *match;
  }
  for (auto& route : exposed_view->exposed_output_order) {
    if (route.logical_output_index < 0 ||
        static_cast<std::size_t>(route.logical_output_index) >= runtime->logical_outputs.size()) {
      return fail(error, "ProcessCVU exposed output order references a missing command output");
    }
    route.segment_name =
        runtime->logical_outputs[static_cast<std::size_t>(route.logical_output_index)].segment_name;
  }

  const bool consumes_internal_carrier = std::any_of(
      inputs.begin(), inputs.end(), [&](const ValueId input) {
        return arena.region(root_value_id(plan, input)) != nullptr;
      });
  const bool continues_existing_output_carrier = std::any_of(
      outputs.begin(), outputs.end(), [&](const ValueId output) {
        const auto* region = arena.region(root_value_id(plan, output));
        return region && first && region->lifetime.first_sequence < first->sequence;
      });
  runtime->frame_arena_size_bytes = arena.allocation_bytes();
  runtime->frame_arena_storage_domain = arena.placement().domain;
  runtime->frame_arena_provenance = arena.placement().provenance;
  runtime->frame_arena_required_device_access =
      arena.placement().required_device_access;
  runtime->frame_arena_escape_policy = arena.placement().escape;
  runtime->frame_arena_role =
      (consumes_internal_carrier || continues_existing_output_carrier)
          ? FrameArenaRole::ReuseInput
          : FrameArenaRole::Allocate;
  runtime->consumer_keeps_distinct_physical_inputs = inputs.size() > 1U;
  payload->dmabuf_plan_contract = true;
  if (payload->runtime_output_physical_index_list.size() == outputs.size()) {
    for (std::size_t index = 0; index < outputs.size(); ++index) {
      payload->runtime_output_physical_index_list[index] =
          runtime->logical_outputs[index].physical_index;
    }
  }
  if (error) {
    error->clear();
  }
  return true;
}

std::optional<std::vector<PhysicalCommandId>>
resolve_model_managed_preproc_ingress_commands(
    const ModelExecutionPlan& plan, const PhysicalExecutionPlan& physical_plan,
    std::string* error) {
  const auto mla_inputs = plan.backend_ports(0U, BackendPortDirection::Input);
  const auto* first_mla = plan.mla_stage(0U);
  if (!first_mla || mla_inputs.size() != 1U || plan.model_inputs().size() != 1U) {
    fail(error,
         "model-managed preproc absorption requires exactly one public input and one first-MLA "
         "IFM");
    return std::nullopt;
  }

  std::vector<OpId> reversed_ops;
  ValueId cursor = mla_inputs.front().value_id;
  for (std::size_t remaining = plan.ops().size() + 1U; remaining > 0U; --remaining) {
    const auto* producer = producer_of(plan, cursor);
    if (!producer || (producer->kind != OpKind::Cast && producer->kind != OpKind::Quantize &&
                      producer->kind != OpKind::Tessellate)) {
      break;
    }
    if (producer->inputs.size() != 1U || producer->outputs.size() != 1U ||
        producer->outputs.front() != cursor) {
      fail(error, "model-managed preproc absorption found a non-linear ingress transform");
      return std::nullopt;
    }
    reversed_ops.push_back(producer->id);
    cursor = producer->inputs.front();
  }
  if (reversed_ops.empty() || cursor != plan.model_inputs().front()) {
    fail(error,
         "model-managed preproc absorption has no exact transform chain from the public input "
         "to the first MLA");
    return std::nullopt;
  }
  std::reverse(reversed_ops.begin(), reversed_ops.end());

  // Absorption is legal only for a private linear prefix. A published or
  // side-consumed intermediate is an observable barrier and must keep its
  // original physical command.
  for (std::size_t index = 0; index < reversed_ops.size(); ++index) {
    const auto& op = plan.ops()[reversed_ops[index]];
    const ValueId output = op.outputs.front();
    if (std::any_of(plan.model_outputs().begin(), plan.model_outputs().end(),
                    [&](const ModelOutputSpec& published) {
                      return published.value_id == output;
                    })) {
      fail(error, "model-managed preproc absorption crosses a published ingress value");
      return std::nullopt;
    }
    const OpId expected_consumer =
        index + 1U < reversed_ops.size() ? reversed_ops[index + 1U] : first_mla->key.op_id;
    std::size_t consumers = 0U;
    for (const auto& consumer : plan.ops()) {
      if (std::find(consumer.inputs.begin(), consumer.inputs.end(), output) ==
          consumer.inputs.end()) {
        continue;
      }
      ++consumers;
      if (consumer.id != expected_consumer) {
        fail(error, "model-managed preproc absorption crosses an ingress fan-out or barrier");
        return std::nullopt;
      }
    }
    if (consumers != 1U) {
      fail(error, "model-managed preproc absorption found an unconsumed ingress value");
      return std::nullopt;
    }
  }

  if (physical_plan.command_for_semantic_op.size() != plan.ops().size()) {
    fail(error, "model-managed preproc absorption has no complete physical-command index");
    return std::nullopt;
  }
  std::vector<PhysicalCommandId> commands;
  for (const auto op_id : reversed_ops) {
    const auto command_id = physical_plan.command_for_semantic_op[op_id];
    if (!command_id || *command_id >= physical_plan.commands.size()) {
      fail(error, "model-managed preproc absorption references an unlowered semantic op");
      return std::nullopt;
    }
    if (commands.empty() || commands.back() != *command_id) {
      commands.push_back(*command_id);
    }
  }

  std::unordered_set<OpId> wanted_ops(reversed_ops.begin(), reversed_ops.end());
  for (const auto command_id : commands) {
    const auto& command = physical_plan.commands[command_id];
    if (command.engine != PhysicalEngine::Cvu ||
        command.role != PhysicalCommandRole::Ingress || command.members.size() != 1U ||
        command.inputs.size() != 1U || command.outputs.size() != 1U ||
        command.members.front().semantic_chain.empty()) {
      fail(error,
           "model-managed preproc absorption requires singleton strict CVU Ingress commands");
      return std::nullopt;
    }
    for (const auto semantic : command.members.front().semantic_chain) {
      if (wanted_ops.erase(semantic) != 1U) {
        fail(error,
             "model-managed preproc absorption command contains an unrelated semantic op");
        return std::nullopt;
      }
    }
  }
  if (!wanted_ops.empty()) {
    fail(error, "model-managed preproc absorption does not cover its exact semantic prefix");
    return std::nullopt;
  }
  if (error) {
    error->clear();
  }
  return commands;
}

bool project_model_managed_preproc_contract(
    const ModelExecutionPlan& plan, const PhysicalExecutionPlan& physical_plan,
    const FrameSlotArenaPlan& arena, ::simaai::neat::CompiledProcessCvuContract* contract,
    std::vector<PhysicalCommandId>* absorbed_command_ids, std::string* error) {
  if (!contract || !absorbed_command_ids) {
    return fail(error, "model-managed preproc projection received a null contract/output");
  }
  auto commands = resolve_model_managed_preproc_ingress_commands(plan, physical_plan, error);
  if (!commands) {
    return false;
  }

  const auto mla_inputs = plan.backend_ports(0U, BackendPortDirection::Input);
  const auto* target = mla_inputs.size() == 1U ? plan.value(mla_inputs.front().value_id) : nullptr;
  if (!target || !target->logical_dtype || !target->logical_shape ||
      !target->logical_layout || !target->storage_binding) {
    return fail(error,
                "model-managed preproc target has no exact typed first-MLA storage contract");
  }

  bool has_quant = false;
  bool has_tess = false;
  const QuantizeOpConfig* quant = nullptr;
  const TessellateOpConfig* tess = nullptr;
  const OpSpec* tess_op = nullptr;
  for (const auto command_id : *commands) {
    for (const auto op_id : physical_plan.commands[command_id].members.front().semantic_chain) {
      const auto& op = plan.ops()[op_id];
      if (op.kind == OpKind::Quantize) {
        has_quant = true;
        quant = std::get_if<QuantizeOpConfig>(&op.config);
      } else if (op.kind == OpKind::Tessellate) {
        has_tess = true;
        tess = std::get_if<TessellateOpConfig>(&op.config);
        tess_op = &op;
      }
    }
  }
  if ((contract->payload.tessellate == 1) != has_tess) {
    return fail(error,
                "model-managed preproc tessellation contradicts the absorbed ingress prefix");
  }
  if (has_tess &&
      (!tess || contract->payload.slice_shapes.size() != 1U ||
       std::vector<std::int64_t>(contract->payload.slice_shapes.front().begin(),
                                 contract->payload.slice_shapes.front().end()) !=
           tess->slice_shape)) {
    return fail(error,
                "model-managed preproc tile geometry contradicts the absorbed ingress prefix");
  }
  if (has_quant) {
    std::string rounding = quant ? quant->rounding : std::string{};
    std::transform(rounding.begin(), rounding.end(), rounding.begin(),
                   [](const unsigned char c) { return static_cast<char>(std::toupper(c)); });
    const bool exact_rounding = rounding == "TONEAREST" || rounding == "RT_EVEN";
    if (!quant || !exact_rounding || quant->channel_params.size() != 1U ||
        !contract->payload.has_q_scale || !contract->payload.has_q_zp ||
        contract->payload.q_scale != quant->channel_params.front().scale ||
        contract->payload.q_zp != quant->channel_params.front().zero_point) {
      return fail(error,
                  "model-managed preproc quantization contradicts the absorbed ingress prefix");
    }
  } else if (contract->payload.has_q_scale || contract->payload.has_q_zp) {
    return fail(error,
                "model-managed preproc authors quantization without an absorbed quantize op");
  }

  // The graph-200 descriptor keeps both firmware-internal output identities
  // (RGB and, when enabled, tessellated), but a model-managed preprocessor is
  // explicitly a single-output handoff.  The immutable execution plan has no
  // values named after those firmware pointers; its exact authority is the
  // sole first-MLA port selected above.  Narrow the already-compiled contract
  // to its authored primary output before the ordinary one-port projection so
  // that the resolver maps that one output to the materialized MLA ingress
  // rather than attempting to resolve an unexposed internal pointer by name.
  auto& runtime = contract->runtime_contract;
  const auto& primary_output_name = contract->payload.primary_output_name;
  const auto selected = std::find_if(
      runtime.logical_outputs.begin(), runtime.logical_outputs.end(),
      [&](const LogicalTensorStaticSpec& logical) {
        return logical.logical_name == primary_output_name ||
               logical.backend_name == primary_output_name ||
               logical.segment_name == primary_output_name;
      });
  if (!contract->payload.preproc_single_output_handoff ||
      primary_output_name.empty() || selected == runtime.logical_outputs.end() ||
      std::find_if(std::next(selected), runtime.logical_outputs.end(),
                   [&](const LogicalTensorStaticSpec& logical) {
                     return logical.logical_name == primary_output_name ||
                            logical.backend_name == primary_output_name ||
                            logical.segment_name == primary_output_name;
                   }) != runtime.logical_outputs.end() ||
      selected->physical_index < 0 ||
      static_cast<std::size_t>(selected->physical_index) >= runtime.physical_outputs.size()) {
    return fail(error,
                "model-managed preproc primary output has no unique internal graph-200 "
                "identity");
  }
  LogicalTensorStaticSpec selected_logical = *selected;
  PhysicalBufferStaticSpec selected_physical =
      runtime.physical_outputs[static_cast<std::size_t>(selected->physical_index)];
  selected_logical.logical_index = 0;
  selected_logical.backend_output_index = 0;
  selected_logical.physical_index = 0;
  selected_logical.output_slot = 0;
  selected_logical.tensor_index = 0;
  selected_logical.byte_offset = 0;
  selected_physical.physical_index = 0;
  selected_physical.allocator_index = 0;
  selected_physical.source_physical_index = 0;
  selected_physical.source_byte_offset = 0;
  runtime.logical_outputs = {selected_logical};
  runtime.physical_outputs = {std::move(selected_physical)};
  runtime.output_order = {StageOutputRoute{0, 0, 0, primary_output_name,
                                           selected_logical.segment_name}};
  contract->exposed_view.primary_output_name = primary_output_name;
  contract->exposed_view.exposed_logical_outputs = {selected_logical};
  contract->exposed_view.exposed_output_order = runtime.output_order;

  if (!apply_dmabuf_plan_processcvu_contract_projection(
          plan, 0U, arena, ProcessCvuMlaBoundary::Inputs, &contract->payload,
          &contract->runtime_contract, &contract->exposed_view, error)) {
    return false;
  }
  if (contract->runtime_contract.logical_outputs.size() != 1U ||
      contract->runtime_contract.physical_outputs.size() != 1U ||
      contract->runtime_contract.logical_inputs.size() != 1U ||
      contract->runtime_contract.physical_inputs.size() != 1U ||
      contract->runtime_contract.input_bindings.size() != 1U ||
      contract->payload.input_tensors.size() != 1U) {
    return fail(error,
                "model-managed preproc strict handoff must expose one input and one "
                "first-MLA carrier");
  }

  const auto& logical = contract->runtime_contract.logical_outputs.front();
  if (has_tess) {
    contract->runtime_contract.logical_outputs.front().size_bytes = target->required_bytes;
    contract->runtime_contract.physical_outputs.front().size_bytes =
        target->storage_binding->physical_span;
  }
  std::uint32_t logical_dtype = 0U;
  std::uint32_t target_dtype = 0U;
  if (!tensorsemantics::dtype_token_to_ev(logical.dtype, &logical_dtype) ||
      !tensorsemantics::dtype_token_to_ev(*target->logical_dtype, &target_dtype) ||
      logical_dtype != target_dtype || logical.shape != *target->logical_shape ||
      logical.layout != *target->logical_layout || logical.size_bytes != target->required_bytes ||
      contract->runtime_contract.physical_outputs.front().size_bytes !=
          target->storage_binding->physical_span) {
    return fail(error,
                "model-managed preproc output contradicts the exact first-MLA tensor/carrier");
  }

  // Graph 200 has two firmware-internal output pointers, but the strict
  // single-output mode relocates exactly one of them into the frame arena.
  // Collapse every output-indexed typed field to that compiler-proved MLA
  // ingress value.  Leaving both legacy internal descriptors here would make
  // graph 200 look like a two-member transform even though it has one input
  // and one selected output.
  sima_ev_tensor_desc selected_output{};
  if (has_tess) {
    TensorShape frame_shape;
    if (tess_op && !tess_op->input_shapes.empty()) {
      frame_shape = tess_op->input_shapes.front();
    } else if (tess_op && !tess_op->inputs.empty()) {
      const auto* frame = plan.value(tess_op->inputs.front());
      if (frame && frame->logical_shape) {
        frame_shape = *frame->logical_shape;
      }
    }
    if (!tess || frame_shape.empty() ||
        !build_cvu_tiled_desc(*target, frame_shape, tess->slice_shape,
                              tess->frame_type, tess->align_c16 || tess->cblock,
                              &selected_output, error)) {
      return false;
    }
  } else if (!build_cvu_dense_desc(*target, {}, {}, &selected_output, error)) {
    return false;
  }

  auto& payload = contract->payload;
  payload.default_output_names = {payload.primary_output_name};
  payload.output_tensors = {selected_output};
  payload.output_shapes = {cvu_shape(*target)};
  payload.runtime_output_logical_index_list = {0};
  payload.runtime_output_output_slot_list = {0};
  payload.runtime_output_physical_index_list = {0};
  payload.runtime_output_dtype_list = {cvu_dtype(*target)};
  payload.runtime_output_transport_kind_list = {
      has_tess ? ProcessCvuOutputTransportKind::Packed
               : ProcessCvuOutputTransportKind::Dense};
  payload.runtime_output_semantic_kind_list = {
      ProcessCvuOutputSemanticKind::Image};
  payload.runtime_output_logical_shapes = {cvu_shape(*target)};
  payload.runtime_output_logical_layout_list = {*target->logical_layout};
  payload.num_in_tensor = 1;

  // A public video input is still one concrete DMA-BUF carrier.  It has no
  // producer stage in the model plan, so author its sole physical/logical
  // selector explicitly instead of leaving sentinel -1 values that the
  // strict binder must reject as ambiguous.
  auto& physical_input = contract->runtime_contract.physical_inputs.front();
  physical_input.physical_index = 0;
  physical_input.allocator_index = 0;
  physical_input.source_physical_index = 0;
  physical_input.source_byte_offset = 0;
  auto& logical_input = contract->runtime_contract.logical_inputs.front();
  logical_input.logical_index = 0;
  logical_input.backend_input_index = 0;
  logical_input.physical_index = 0;
  logical_input.byte_offset = 0;
  auto& input_binding = contract->runtime_contract.input_bindings.front();
  input_binding.sink_pad_index = 0;
  input_binding.local_logical_input_index = 0;
  input_binding.src_logical_output_index = 0;
  input_binding.src_output_slot = 0;
  input_binding.src_physical_output_index = 0;
  input_binding.src_physical_size_bytes = physical_input.size_bytes;
  input_binding.src_physical_byte_offset = 0;
  input_binding.source_segment_name = physical_input.segment_name;

  contract->physical_command_role = PhysicalCommandRole::Ingress;
  *absorbed_command_ids = std::move(*commands);
  if (error) {
    error->clear();
  }
  return true;
}

bool apply_dmabuf_plan_processcvu_contract_projection(
    const ModelExecutionPlan& plan, const std::size_t adjacent_mla_stage_index,
    const ProcessCvuMlaBoundary boundary, ProcessCvuStagePayload* payload,
    ::simaai::neat::CompiledRuntimeContract* runtime,
    ::simaai::neat::CompiledExposedView* exposed_view, std::string* error) {
  auto arena = compile_frame_arena(plan, error);
  return arena && apply_dmabuf_plan_processcvu_contract_projection(plan, adjacent_mla_stage_index,
                                                                   *arena, boundary, payload,
                                                                   runtime, exposed_view, error);
}

bool apply_dmabuf_plan_processcvu_contract_projection(
    const ModelExecutionPlan& plan, const ProcessCvuMlaBoundary boundary,
    ProcessCvuStagePayload* payload, ::simaai::neat::CompiledRuntimeContract* runtime,
    ::simaai::neat::CompiledExposedView* exposed_view, std::string* error) {
  if (plan.mla_stage_count() != 1U) {
    return fail(error,
                "single-stage ProcessCVU/MLA projection is ambiguous for a multi-stage plan");
  }
  return apply_dmabuf_plan_processcvu_contract_projection(plan, 0U, boundary, payload, runtime,
                                                          exposed_view, error);
}

} // namespace simaai::neat::pipeline_internal::sima::static_contract
