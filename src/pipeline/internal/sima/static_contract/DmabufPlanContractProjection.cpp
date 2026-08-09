#define SIMA_NEAT_INTERNAL 1
#include "pipeline/internal/sima/static_contract/DmabufPlanContractProjection.h"
#include "pipeline/internal/sima/static_contract/FrameSlotArenaPlan.h"

#include <algorithm>
#include <limits>
#include <optional>
#include <string>
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
  const auto* value = plan.value(value_id);
  return value && value->read_expression.has_value()
             ? value->read_expression->source_value_id
             : value_id;
}

std::optional<FrameSlotArenaPlan>
compile_frame_arena(const ModelExecutionPlan& plan, std::string* error) {
  return FrameSlotArenaPlan::compile(
      plan, FrameSlotArenaReuse::DisjointLifetimes,
      kLegacyEvoCmaRegionAlignmentBytes, error);
}

bool assign_physical_region(const FrameSlotArenaPlan& arena,
                            const ValueSpec& value,
                            PhysicalBufferStaticSpec* physical,
                            std::string* error) {
  if (!physical) {
    return fail(error, "frame-slot physical projection is null");
  }
  const auto* region = arena.region(value.id);
  if (!region || region->size_bytes != physical->size_bytes ||
      physical->required_alignment_bytes == 0U ||
      region->byte_offset % physical->required_alignment_bytes != 0U ||
      region->byte_offset >
          static_cast<std::uint64_t>(std::numeric_limits<std::int64_t>::max())) {
    return fail(error, "frame-slot region contradicts value '" + value.name + "'");
  }
  physical->source_byte_offset =
      static_cast<std::int64_t>(region->byte_offset);
  return true;
}

const ValueSpec* find_materialized_value_for_logical(
    const ModelExecutionPlan& plan, const LogicalTensorStaticSpec& logical,
    std::string* error) {
  // Names are compiler identities, not runtime binding keys.  Try the most
  // backend-specific spelling first and require that spelling to identify one
  // materialized value.  A shared-parent segment such as "output_tensor" is
  // intentionally only the final fallback.
  for (const auto* name :
       {&logical.backend_name, &logical.logical_name, &logical.segment_name}) {
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

std::optional<CvuOutputPlacement> resolve_cvu_output_placement(
    const ModelExecutionPlan& plan, const FrameSlotArenaPlan& arena,
    const ProcessCvuMlaBoundary boundary,
    std::span<const BackendPortSpec* const> ports,
    const LogicalTensorStaticSpec& logical, const std::size_t output_index,
    const std::size_t output_count, std::string* error) {
  const ValueSpec* value =
      find_materialized_value_for_logical(plan, logical, error);
  if (!value && error && !error->empty()) {
    return std::nullopt;
  }
  if (!value && output_count == ports.size()) {
    value = plan.value(ports[output_index]->value_id);
  }
  if (!value && boundary == ProcessCvuMlaBoundary::Outputs &&
      output_count == plan.model_outputs().size()) {
    value = plan.value(plan.model_outputs()[output_index].value_id);
  }
  if (!value || value->read_expression.has_value()) {
    fail(error, "ProcessCVU logical output has no materialized execution-plan value: "
                    "logical='" +
                    logical.logical_name + "' backend='" + logical.backend_name +
                    "' segment='" + logical.segment_name + "'");
    return std::nullopt;
  }

  const auto alignment_for_boundary = [&]() -> std::uint64_t {
    if (ports.size() == output_count) {
      return ports[output_index]->required_alignment_bytes;
    }
    if (ports.size() == 1U) {
      return ports.front()->required_alignment_bytes;
    }
    return 0U;
  };

  // An exact batch-one Pack is a placement expression for the preceding CVU
  // outputs.  The CVU writes each child directly into its compiler-authored
  // parent offset; no Pack job or copy is scheduled.
  if (boundary == ProcessCvuMlaBoundary::Inputs && ports.size() == 1U) {
    for (const auto& op : plan.ops()) {
      if (op.kind != OpKind::Pack || op.outputs.size() != 1U ||
          op.outputs.front() != ports.front()->value_id) {
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
                           std::numeric_limits<std::uint64_t>::max() -
                               parent->byte_offset) {
          fail(error, "ProcessCVU Pack child has no parent frame-arena placement");
          return std::nullopt;
        }
        return CvuOutputPlacement{value,
                                  parent->byte_offset + component.parent_offset,
                                  16U};
      }
    }
  }

  const auto* region = arena.region(value->id);
  const auto alignment = alignment_for_boundary();
  if (!region || alignment == 0U ||
      region->byte_offset % alignment != 0U) {
    fail(error, "ProcessCVU value '" + value->name +
                    "' has no aligned frame-arena placement");
    return std::nullopt;
  }
  return CvuOutputPlacement{value, region->byte_offset, alignment};
}

std::optional<int>
resolve_pack_parent_physical_source(
    const ModelExecutionPlan& plan, const ValueSpec& packed_value,
    std::span<const LogicalTensorStaticSpec> upstream_outputs,
    std::string* error) {
  const auto* pack = producer_of(plan, packed_value.id);
  if (!pack || pack->kind != OpKind::Pack || pack->outputs.size() != 1U ||
      pack->outputs.front() != packed_value.id || pack->inputs.empty()) {
    return std::nullopt;
  }
  const auto* config = std::get_if<PackOpConfig>(&pack->config);
  if (!config || config->components.size() != pack->inputs.size()) {
    fail(error, "MLA packed IFM has no exact Pack component placement");
    return std::nullopt;
  }

  int physical_index = -1;
  std::uint64_t covered_bytes = 0U;
  for (std::size_t component_index = 0;
       component_index < pack->inputs.size(); ++component_index) {
    const auto input_id = pack->inputs[component_index];
    const auto& placement = config->components[component_index];
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
    if (placement.value_id != input_id ||
        placement.parent_offset != covered_bytes ||
        placement.stored_bytes != child_value->required_bytes ||
        child->logical_index < 0 || child->physical_index < 0 ||
        child->byte_offset < 0 ||
        child->size_bytes != child_value->required_bytes ||
        static_cast<std::uint64_t>(child->byte_offset) !=
            placement.parent_offset) {
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
    if (placement.stored_bytes >
        std::numeric_limits<std::uint64_t>::max() - covered_bytes) {
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

std::vector<const ValueSpec*>
read_views_consumed_after_mla(const ModelExecutionPlan& plan, const ValueId root_value_id) {
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

} // namespace

std::optional<std::vector<PhysicalPortSource>>
resolve_mla_input_physical_sources(
    const ModelExecutionPlan& plan,
    std::span<const LogicalTensorStaticSpec> upstream_outputs,
    std::string* error) {
  std::vector<const BackendPortSpec*> inputs;
  for (const auto& port : plan.backend_ports()) {
    if (port.direction == BackendPortDirection::Input) {
      if (port.stage_index != 0U) {
        fail(error, "MLA physical input projection currently requires one MLA stage");
        return std::nullopt;
      }
      inputs.push_back(&port);
    }
  }
  std::sort(inputs.begin(), inputs.end(),
            [](const auto* lhs, const auto* rhs) { return lhs->port_index < rhs->port_index; });

  std::vector<PhysicalPortSource> result;
  result.reserve(inputs.size());
  std::unordered_set<int> seen_sources;
  for (std::size_t port_index = 0; port_index < inputs.size(); ++port_index) {
    const auto& port = *inputs[port_index];
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
    const auto model_input =
        std::find(plan.model_inputs().begin(), plan.model_inputs().end(), port.value_id);
    if (model_input != plan.model_inputs().end()) {
      const auto index = std::distance(plan.model_inputs().begin(), model_input);
      if (index > std::numeric_limits<int>::max()) {
        fail(error, "MLA model-input physical carrier index overflows int");
        return std::nullopt;
      }
      source_physical_index = static_cast<int>(index);
    } else {
      const auto* match = find_exact_upstream_output(*value, upstream_outputs, error);
      if (match && match->physical_index >= 0 && match->byte_offset == 0 &&
          match->size_bytes == value->required_bytes) {
        source_physical_index = match->physical_index;
      } else {
        source_physical_index = resolve_pack_parent_physical_source(
            plan, *value, upstream_outputs, error);
        if (!source_physical_index.has_value()) {
          if (error && error->empty()) {
            *error = "MLA input value '" + value->name +
                     "' has no exact upstream physical carrier or Pack proof";
          }
          return std::nullopt;
        }
      }
    }

    if (!seen_sources.emplace(*source_physical_index).second) {
      fail(error, "MLA input projection aliases two IFM ports to one physical carrier");
      return std::nullopt;
    }
    result.push_back(
        PhysicalPortSource{port.value_id, *source_physical_index});
  }
  if (error) {
    error->clear();
  }
  return result;
}

bool apply_dmabuf_plan_contract_projection(const ModelExecutionPlan& plan,
                                           MlaStaticContract* contract,
                                           std::span<const PhysicalPortSource> input_sources,
                                           std::string* error) {
  if (!contract) {
    return fail(error, "MLA contract projection is null");
  }
  auto arena = compile_frame_arena(plan, error);
  if (!arena) {
    return false;
  }

  std::vector<const BackendPortSpec*> inputs;
  std::vector<const BackendPortSpec*> outputs;
  for (const auto& port : plan.backend_ports()) {
    (port.direction == BackendPortDirection::Input ? inputs : outputs).push_back(&port);
  }
  const auto by_port_index = [](const BackendPortSpec* lhs, const BackendPortSpec* rhs) {
    return lhs->port_index < rhs->port_index;
  };
  std::sort(inputs.begin(), inputs.end(), by_port_index);
  std::sort(outputs.begin(), outputs.end(), by_port_index);

  if (inputs.empty() || outputs.empty() ||
      std::any_of(inputs.begin(), inputs.end(),
                  [](const auto* port) { return port->stage_index != 0U; }) ||
      std::any_of(outputs.begin(), outputs.end(),
                  [](const auto* port) { return port->stage_index != 0U; })) {
    return fail(error, "MLA projection requires one non-empty backend stage");
  }

  const auto validate = [&](const std::vector<const BackendPortSpec*>& ports,
                            const std::vector<PhysicalBufferStaticSpec>& physical,
                            const char* kind) {
    if (ports.size() != physical.size()) {
      return fail(error, std::string(kind) + " projection arity mismatch");
    }
    for (std::size_t index = 0; index < ports.size(); ++index) {
      const auto& port = *ports[index];
      const auto& projected = physical[index];
      const auto* value = plan.value(port.value_id);
      if (port.port_index != index || projected.physical_index != static_cast<int>(index) ||
          projected.size_bytes != port.required_bytes || !value || projected.segment_name.empty() ||
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
    contract->physical_inputs[index].required_alignment_bytes =
        inputs[index]->required_alignment_bytes;
  }
  for (std::size_t index = 0; index < outputs.size(); ++index) {
    contract->dispatcher_physical_outputs[index].required_alignment_bytes =
        outputs[index]->required_alignment_bytes;
  }

  if (input_sources.size() != inputs.size()) {
    return fail(error, "IFM projection has no complete physical carrier map");
  }
  std::unordered_set<int> seen_input_sources;
  contract->input_bindings.clear();
  contract->input_bindings.reserve(inputs.size());
  for (std::size_t index = 0; index < inputs.size(); ++index) {
    const auto& port = *inputs[index];
    const auto& projected = input_sources[index];
    auto& physical = contract->physical_inputs[index];
    if (projected.value_id != port.value_id ||
        projected.source_physical_index < 0 ||
        !seen_input_sources.emplace(projected.source_physical_index).second) {
      return fail(error, "IFM projection has an invalid or aliased physical carrier");
    }
    physical.source_physical_index = projected.source_physical_index;
    const auto root_id = root_value_id(plan, port.value_id);
    if (const auto* region = arena->region(root_id)) {
      if (region->byte_offset > static_cast<std::uint64_t>(
                                    std::numeric_limits<std::int64_t>::max())) {
        return fail(error, "IFM frame-slot offset cannot be represented");
      }
      physical.source_byte_offset =
          static_cast<std::int64_t>(region->byte_offset);
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
    binding.src_physical_output_index = projected.source_physical_index;
    binding.src_physical_size_bytes = port.required_bytes;
    binding.src_physical_byte_offset = 0;
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
  for (std::size_t index = 0; index < outputs.size(); ++index) {
    const auto& port = *outputs[index];
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
    physical.size_bytes = port.required_bytes;
    physical.source_byte_offset = 0;
    physical.device_kind = DeviceKind::Mla;
    physical.segment_name = value->name;
    physical.required_alignment_bytes = port.required_alignment_bytes;
    if (!assign_physical_region(*arena, *value, &physical, error)) {
      return false;
    }
    contract->physical_outputs.push_back(std::move(physical));
  }

  std::vector<const ValueSpec*> packed_read_views;
  if (outputs.size() == 1U) {
    packed_read_views = read_views_consumed_after_mla(plan, outputs.front()->value_id);
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
      if (index > static_cast<std::size_t>(std::numeric_limits<int>::max()) ||
          expression.byte_offset >
              static_cast<std::uint64_t>(std::numeric_limits<std::int64_t>::max()) ||
          !value.logical_shape.has_value()) {
        return fail(error, "packed OFM read view cannot be represented in the static manifest");
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
      logical.logical_name = value.name;
      logical.backend_name = value.name;
      logical.segment_name = plan.value(outputs.front()->value_id)->name;
      if (value.logical_dtype) {
        logical.dtype = *value.logical_dtype;
      }
      if (value.logical_layout) {
        logical.layout = *value.logical_layout;
      }
      contract->logical_outputs.push_back(std::move(logical));
    }
  } else {
    contract->logical_outputs.reserve(outputs.size());
    for (std::size_t index = 0; index < outputs.size(); ++index) {
      const auto& port = *outputs[index];
      const auto* value = plan.value(port.value_id);
      const int slot = static_cast<int>(index);

      LogicalTensorStaticSpec logical;
      logical.logical_index = slot;
      logical.backend_output_index = slot;
      logical.physical_index = slot;
      logical.output_slot = slot;
      logical.tensor_index = slot;
      logical.byte_offset = 0;
      logical.size_bytes = port.required_bytes;
      logical.logical_name = value->name;
      logical.backend_name = value->name;
      logical.segment_name = value->name;
      if (value->logical_dtype) {
        logical.dtype = *value->logical_dtype;
      }
      if (value->logical_shape) {
        logical.shape = *value->logical_shape;
      }
      if (value->logical_layout) {
        logical.layout = *value->logical_layout;
      }
      contract->logical_outputs.push_back(std::move(logical));
    }
  }

  std::vector<std::string> ifm_symbols;
  std::vector<std::string> ofm_symbols;
  ifm_symbols.reserve(inputs.size());
  ofm_symbols.reserve(outputs.size());
  for (const auto* port : inputs) {
    ifm_symbols.push_back(port->elf_symbol);
  }
  for (const auto* port : outputs) {
    ofm_symbols.push_back(port->elf_symbol);
  }
  contract->elf_ifm_symbol_names = std::move(ifm_symbols);
  contract->elf_ofm_symbol_names = std::move(ofm_symbols);
  contract->consumer_keeps_distinct_physical_inputs = inputs.size() > 1U;
  contract->frame_arena_size_bytes = arena->allocation_bytes();
  const bool has_internal_input = std::any_of(
      inputs.begin(), inputs.end(), [&](const BackendPortSpec* port) {
        return arena->region(root_value_id(plan, port->value_id)) != nullptr;
      });
  contract->frame_arena_role = has_internal_input
                                   ? FrameArenaRole::ReuseInput
                                   : FrameArenaRole::Allocate;
  if (error) {
    error->clear();
  }
  return true;
}

bool apply_dmabuf_plan_processcvu_contract_projection(
    const ModelExecutionPlan& plan, ProcessCvuMlaBoundary boundary,
    ProcessCvuStagePayload* payload, ::simaai::neat::CompiledRuntimeContract* runtime,
    ::simaai::neat::CompiledExposedView* exposed_view,
    std::string* error) {
  if (!payload || !runtime || !exposed_view) {
    return fail(error, "ProcessCVU contract projection is null");
  }
  auto arena = compile_frame_arena(plan, error);
  if (!arena) {
    return false;
  }

  switch (payload->graph_family_enum) {
  case ProcessCvuGraphFamily::Cast:
    payload->graph_id = 221;
    break;
  case ProcessCvuGraphFamily::Quant:
    payload->graph_id = 222;
    break;
  case ProcessCvuGraphFamily::Dequant:
    payload->graph_id = 223;
    break;
  case ProcessCvuGraphFamily::Tess:
    payload->graph_id = 2;
    break;
  case ProcessCvuGraphFamily::QuantTess:
    payload->graph_id = 226;
    break;
  case ProcessCvuGraphFamily::CastTess:
    payload->graph_id = 224;
    break;
  case ProcessCvuGraphFamily::Detess:
    payload->graph_id = 3;
    break;
  case ProcessCvuGraphFamily::DetessCast:
    payload->graph_id = 225;
    break;
  case ProcessCvuGraphFamily::DetessDequant:
    payload->graph_id = 227;
    break;
  default:
    return fail(error, "dmabuf-plan ProcessCVU does not support this graph family "
                       "(family=" +
                           std::to_string(static_cast<int>(payload->graph_family_enum)) + ")");
  }

  std::vector<const BackendPortSpec*> ports;
  const auto wanted_direction =
      boundary == ProcessCvuMlaBoundary::Inputs
          ? BackendPortDirection::Input
          : BackendPortDirection::Output;
  for (const auto& port : plan.backend_ports()) {
    if (port.direction == wanted_direction) {
      ports.push_back(&port);
    }
  }
  std::sort(ports.begin(), ports.end(),
            [](const auto* lhs, const auto* rhs) {
              return lhs->port_index < rhs->port_index;
            });
  if (ports.empty() || runtime->logical_outputs.empty()) {
    return fail(error, "ProcessCVU alignment projection has an empty MLA boundary");
  }
  for (std::size_t index = 0; index < ports.size(); ++index) {
    const auto alignment = ports[index]->required_alignment_bytes;
    if (ports[index]->port_index != index || alignment == 0U ||
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

  const DeviceKind output_device =
      runtime->physical_outputs.empty()
          ? DeviceKind::Evxx
          : runtime->physical_outputs.front().device_kind;
  const std::uint64_t output_memory_flags =
      runtime->physical_outputs.empty()
          ? 0U
          : runtime->physical_outputs.front().memory_flags;
  std::vector<PhysicalBufferStaticSpec> projected_outputs;
  projected_outputs.reserve(output_count);
  bool exact_pack_children = false;
  for (std::size_t index = 0; index < output_count; ++index) {
    auto placement = resolve_cvu_output_placement(
        plan, *arena, boundary, ports, runtime->logical_outputs[index], index,
        output_count, error);
    if (!placement || !placement->value ||
        placement->byte_offset > static_cast<std::uint64_t>(
                                     std::numeric_limits<std::int64_t>::max())) {
      return false;
    }
    PhysicalBufferStaticSpec physical;
    physical.physical_index = static_cast<int>(index);
    physical.allocator_index = static_cast<int>(index);
    physical.source_physical_index = static_cast<int>(index);
    physical.size_bytes = placement->value->required_bytes;
    physical.source_byte_offset =
        static_cast<std::int64_t>(placement->byte_offset);
    physical.device_kind = output_device;
    physical.memory_flags = output_memory_flags;
    physical.segment_name = placement->value->name;
    physical.required_alignment_bytes =
        placement->required_alignment_bytes;
    exact_pack_children =
        exact_pack_children || placement->required_alignment_bytes == 16U;
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
      payload->runtime_output_physical_index_list[index] =
          static_cast<int>(index);
    }
  }
  if (!exact_pack_children) {
    for (auto& exposed : exposed_view->exposed_logical_outputs) {
      const auto projected = std::find_if(
          runtime->logical_outputs.begin(), runtime->logical_outputs.end(),
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
      const auto logical = std::find_if(
          exposed_view->exposed_logical_outputs.begin(),
          exposed_view->exposed_logical_outputs.end(),
          [&](const LogicalTensorStaticSpec& output) {
            return output.logical_index == route.logical_output_index;
          });
      if (logical != exposed_view->exposed_logical_outputs.end()) {
        route.segment_name = logical->segment_name;
      }
    }
  }

  runtime->frame_arena_size_bytes = arena->allocation_bytes();
  runtime->frame_arena_role =
      boundary == ProcessCvuMlaBoundary::Inputs
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

} // namespace simaai::neat::pipeline_internal::sima::static_contract
