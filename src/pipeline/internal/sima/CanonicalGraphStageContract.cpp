#include "pipeline/internal/sima/CanonicalGraphStageContract.h"

#include <algorithm>
#include <limits>
#include <unordered_map>

namespace simaai::neat::pipeline_internal::sima {
namespace {
bool fail(std::string* error, std::string message) {
  if (error)
    *error = std::move(message);
  return false;
}
std::size_t scalar_bytes(std::string dtype) {
  dtype = tensorsemantics::upper_copy_ascii(std::move(dtype));
  if (dtype == "INT8" || dtype == "UINT8")
    return 1;
  if (dtype == "BF16" || dtype == "BFLOAT16" || dtype == "FP16" || dtype == "FLOAT16" ||
      dtype == "INT16" || dtype == "UINT16")
    return 2;
  if (dtype == "FP32" || dtype == "FLOAT32" || dtype == "INT32" || dtype == "UINT32")
    return 4;
  if (dtype == "FP64" || dtype == "FLOAT64")
    return 8;
  return 0;
}
struct TensorFacts {
  std::string name, dtype, layout;
  DTypeSource dtype_source = DTypeSource::Unknown;
  std::vector<std::int64_t> shape, strides;
  std::uint64_t bytes = 0, end = 0;
};
bool tensor_facts(const MpkTensorContract& tensor, std::string_view fallback_dtype,
                  const sima_ev_tensor_desc& descriptor, TensorFacts* facts, std::string* error) {
  facts->name = tensor.name.empty() ? tensor.segment_name : tensor.name;
  facts->shape = tensor.logical_shape.empty() ? tensor.mpk_shape : tensor.logical_shape;
  facts->dtype = !fallback_dtype.empty()         ? std::string(fallback_dtype)
                 : !tensor.logical_dtype.empty() ? tensor.logical_dtype
                                                 : tensor.dtype;
  facts->dtype_source =
      !tensor.logical_dtype.empty() ? tensor.logical_dtype_source : tensor.dtype_source;
  if (!fallback_dtype.empty())
    facts->dtype_source = DTypeSource::InternalContract;
  if (facts->name.empty() || tensor.segment_name.empty() || tensor.tensor_index < 0 ||
      tensor.physical_index < 0 || tensor.byte_offset < 0 || facts->shape.empty())
    return fail(error, "canonical graph tensor is missing shape or graph-owned identity");
  std::uint64_t dense = scalar_bytes(facts->dtype);
  if (!dense)
    return fail(error, "canonical graph tensor has an unknown semantic dtype");
  std::vector<std::int64_t> dense_strides(facts->shape.size());
  for (std::size_t axis = facts->shape.size(); axis-- > 0;) {
    if (facts->shape[axis] <= 0 || dense > std::uint64_t(INT64_MAX) ||
        std::uint64_t(facts->shape[axis]) > UINT64_MAX / dense)
      return fail(error, "canonical graph tensor shape or byte extent overflows");
    dense_strides[axis] = static_cast<std::int64_t>(dense);
    dense *= std::uint64_t(facts->shape[axis]);
  }
  // Missing model extents may use the already authored exact descriptor, never
  // an invented maximum input image envelope or an upstream carrier substitute.
  facts->bytes = tensor.size_bytes ? tensor.size_bytes : std::max(dense, descriptor.storage.nbytes);
  if (std::uint64_t(tensor.byte_offset) > UINT64_MAX - facts->bytes)
    return fail(error, "canonical graph tensor offset plus extent overflows");
  facts->end = std::uint64_t(tensor.byte_offset) + facts->bytes;
  facts->strides = tensor.stride_bytes;
  if (facts->strides.empty()) {
    if (tensor.materialization_kind != MpkTensorMaterializationKind::Direct &&
        facts->bytes != dense)
      return fail(error, "canonical graph non-dense view requires explicit strides");
    facts->strides = std::move(dense_strides);
  } else if (facts->strides.size() != facts->shape.size() ||
             std::any_of(facts->strides.begin(), facts->strides.end(),
                         [](auto stride) { return stride < 0; })) {
    return fail(error, "canonical graph tensor strides do not match its shape");
  }
  facts->layout = tensorsemantics::layout_token_from_ev_tensor_desc(descriptor);
  if (facts->layout.empty()) {
    if (facts->shape.size() == 2)
      facts->layout = "HW";
    else if (facts->shape.size() == 3 || facts->shape.size() == 4)
      facts->layout = "HWC";
  }
  return true;
}
TensorMaterializationKind materialization(MpkTensorMaterializationKind kind) {
  switch (kind) {
  case MpkTensorMaterializationKind::OffsetView:
    return TensorMaterializationKind::OffsetView;
  case MpkTensorMaterializationKind::Bf16LaneSplitRepack:
    return TensorMaterializationKind::Bf16LaneSplitRepack;
  default:
    return TensorMaterializationKind::Direct;
  }
}
// A declared compiler physical carrier is authoritative, even when it is larger
// than this logical view. It must never be shrunk or changed to a direct carrier.
bool merge_physical(std::vector<PhysicalBufferStaticSpec>& existing,
                    std::vector<PhysicalBufferStaticSpec> inferred, std::string* error) {
  if (existing.empty()) {
    existing = std::move(inferred);
    return true;
  }
  for (const auto& requested : inferred) {
    const auto found = std::find_if(existing.begin(), existing.end(), [&](const auto& value) {
      return value.physical_index == requested.physical_index;
    });
    if (found == existing.end() || found->size_bytes < requested.size_bytes)
      return fail(error, "canonical graph tensor exceeds its compiler-owned physical carrier");
  }
  return true;
}
} // namespace

bool complete_graph_kernel_contract(StageStaticSpec* stage,
                                    std::span<const MpkTensorContract> inputs,
                                    std::span<const MpkTensorContract> outputs,
                                    std::string_view canonical_input_dtype,
                                    std::string_view canonical_output_dtype,
                                    std::string_view input_slot,
                                    std::span<const std::string> output_slots, std::string* error) {
  if (error)
    error->clear();
  if (!stage || inputs.empty() || outputs.empty() || inputs.size() > INT_MAX ||
      outputs.size() > INT_MAX)
    return fail(error, "canonical graph completion requires a stage and bounded nonempty IO");
  if (stage->processcvu.input_tensors.size() != inputs.size() ||
      stage->processcvu.output_tensors.size() != outputs.size())
    return fail(error, "canonical graph completion requires matching authored tensor descriptors");
  StageStaticSpec result = *stage;
  std::vector<LogicalInputStaticSpec> logical_inputs;
  std::vector<PhysicalBufferStaticSpec> physical_inputs, physical_outputs;
  std::vector<LogicalTensorStaticSpec> logical_outputs;
  std::vector<InputBindingStaticSpec> bindings;
  std::vector<StageOutputRoute> routes;
  std::unordered_map<int, std::size_t> input_ids, output_ids;
  for (std::size_t index = 0; index < inputs.size(); ++index) {
    const auto& tensor = inputs[index];
    TensorFacts facts;
    if (!tensor_facts(tensor, canonical_input_dtype, result.processcvu.input_tensors[index], &facts,
                      error))
      return false;
    if (tensor.source_physical_index < 0 || tensor.source_byte_offset < 0)
      return fail(error, "canonical graph input is missing upstream physical-carrier authority");
    LogicalInputStaticSpec logical;
    logical.logical_index = static_cast<int>(index);
    logical.backend_input_index = tensor.tensor_index;
    logical.physical_index = tensor.physical_index;
    logical.shape = facts.shape;
    logical.stride_bytes = facts.strides;
    logical.byte_offset = tensor.byte_offset;
    logical.size_bytes = facts.bytes;
    logical.dtype = facts.dtype;
    logical.dtype_source = facts.dtype_source;
    logical.layout = facts.layout;
    logical.logical_name = logical.backend_name = facts.name;
    logical.segment_name = tensor.segment_name;
    logical.materialization_kind = materialization(tensor.materialization_kind);
    logical_inputs.push_back(std::move(logical));
    const auto [position, inserted] =
        input_ids.emplace(tensor.physical_index, physical_inputs.size());
    if (inserted) {
      PhysicalBufferStaticSpec physical;
      physical.physical_index = physical.allocator_index = tensor.physical_index;
      physical.source_physical_index = tensor.source_physical_index;
      physical.source_byte_offset = tensor.source_byte_offset;
      physical.size_bytes = facts.end;
      physical.segment_name = tensor.segment_name;
      physical_inputs.push_back(std::move(physical));
    } else {
      auto& physical = physical_inputs[position->second];
      if (physical.source_physical_index != tensor.source_physical_index ||
          physical.source_byte_offset != tensor.source_byte_offset ||
          physical.segment_name != tensor.segment_name)
        return fail(error, "canonical graph input aliases disagree on physical-carrier identity");
      physical.size_bytes = std::max(physical.size_bytes, facts.end);
    }
    InputBindingStaticSpec binding;
    binding.sink_pad_index = binding.local_logical_input_index = static_cast<int>(index);
    binding.src_logical_output_index = binding.src_output_slot = static_cast<int>(index);
    binding.src_physical_output_index = tensor.source_physical_index;
    // Scheduler adds the physical carrier base to this logical-view offset.
    // The upstream base belongs to physical.source_byte_offset exactly once.
    binding.src_physical_byte_offset = tensor.byte_offset;
    binding.src_physical_size_bytes = facts.bytes;
    binding.cm_input_name =
        inputs.size() == 1 && !input_slot.empty() ? std::string(input_slot) : facts.name;
    binding.source_segment_name = tensor.segment_name;
    bindings.push_back(std::move(binding));
  }
  for (std::size_t index = 0; index < outputs.size(); ++index) {
    const auto& tensor = outputs[index];
    TensorFacts facts;
    if (!tensor_facts(tensor, canonical_output_dtype, result.processcvu.output_tensors[index],
                      &facts, error))
      return false;
    LogicalTensorStaticSpec logical;
    logical.logical_index = logical.output_slot = static_cast<int>(index);
    logical.backend_output_index = logical.tensor_index = tensor.tensor_index;
    logical.physical_index = tensor.physical_index;
    logical.byte_offset = tensor.byte_offset;
    logical.size_bytes = facts.bytes;
    logical.shape = facts.shape;
    logical.stride_bytes = facts.strides;
    logical.dtype = facts.dtype;
    logical.dtype_source = facts.dtype_source;
    logical.layout = facts.layout;
    logical.logical_name = logical.backend_name =
        index < output_slots.size() && !output_slots[index].empty() ? output_slots[index]
                                                                    : facts.name;
    logical.segment_name = tensor.segment_name;
    if (index < result.processcvu.runtime_output_logical_layout_list.size()) {
      const auto& authored = result.processcvu.runtime_output_logical_layout_list[index];
      if (!authored.empty() && tensorsemantics::normalize_layout_token(authored) != facts.layout)
        return fail(error, "canonical graph output layout conflicts with descriptor semantics");
    }
    routes.push_back({logical.output_slot, logical.logical_index, logical.tensor_index,
                      logical.logical_name, logical.segment_name});
    logical_outputs.push_back(std::move(logical));
    const auto [position, inserted] =
        output_ids.emplace(tensor.physical_index, physical_outputs.size());
    if (inserted) {
      PhysicalBufferStaticSpec physical;
      physical.physical_index = physical.allocator_index = tensor.physical_index;
      physical.size_bytes = facts.end;
      physical.segment_name = tensor.segment_name;
      physical_outputs.push_back(std::move(physical));
    } else {
      auto& physical = physical_outputs[position->second];
      if (physical.segment_name != tensor.segment_name)
        return fail(error, "canonical graph outputs disagree on shared physical-carrier identity");
      physical.size_bytes = std::max(physical.size_bytes, facts.end);
    }
  }
  if (!merge_physical(result.physical_inputs, std::move(physical_inputs), error) ||
      !merge_physical(result.physical_outputs, std::move(physical_outputs), error))
    return false;
  if (result.logical_inputs.empty())
    result.logical_inputs = std::move(logical_inputs);
  if (result.logical_outputs.empty())
    result.logical_outputs = std::move(logical_outputs);
  if (result.input_bindings.empty()) {
    // If the compiler already projected carrier identities, inferred bindings
    // must use that same authority rather than mixing MPK IDs with its offsets.
    for (std::size_t index = 0; index < bindings.size(); ++index) {
      const auto logical = std::find_if(
          result.logical_inputs.begin(), result.logical_inputs.end(), [&](const auto& value) {
            return value.backend_input_index == inputs[index].tensor_index;
          });
      if (logical == result.logical_inputs.end())
        return fail(error, "canonical graph binding has no authored logical input");
      const auto physical = std::find_if(
          result.physical_inputs.begin(), result.physical_inputs.end(),
          [&](const auto& value) { return value.physical_index == logical->physical_index; });
      if (physical == result.physical_inputs.end() || logical->byte_offset < 0 ||
          std::uint64_t(logical->byte_offset) > physical->size_bytes ||
          logical->size_bytes > physical->size_bytes - std::uint64_t(logical->byte_offset))
        return fail(error, "canonical graph logical input exceeds its authored physical carrier");
      auto& binding = bindings[index];
      binding.local_logical_input_index = logical->logical_index;
      binding.src_physical_byte_offset = logical->byte_offset;
      binding.src_physical_size_bytes = logical->size_bytes;
      if (physical->source_physical_index >= 0)
        binding.src_physical_output_index = physical->source_physical_index;
      if (!physical->segment_name.empty())
        binding.source_segment_name = physical->segment_name;
    }
    result.input_bindings = std::move(bindings);
  }
  if (result.output_order.empty())
    result.output_order = std::move(routes);
  // The selected MPK IO contract owns scalar meaning and the runtime slots.
  // Stale application-facing aliases must not contradict the just-authored
  // descriptors at the numerical boundary.
  auto& policy = result.processcvu;
  if (!canonical_input_dtype.empty())
    policy.input_dtype = canonical_input_dtype;
  if (!canonical_output_dtype.empty())
    policy.output_dtype = canonical_output_dtype;
  policy.out_dtype = policy.output_dtype;
  policy.default_input_name =
      input_slot.empty()
          ? (inputs.front().name.empty() ? inputs.front().segment_name : inputs.front().name)
          : std::string(input_slot);
  policy.default_output_names.clear();
  policy.runtime_output_dtype_list.clear();
  for (std::size_t index = 0; index < outputs.size(); ++index) {
    const auto& tensor = outputs[index];
    policy.default_output_names.push_back(
        index < output_slots.size() && !output_slots[index].empty() ? output_slots[index]
        : tensor.name.empty()                                       ? tensor.segment_name
                                                                    : tensor.name);
    policy.runtime_output_dtype_list.push_back(
        !canonical_output_dtype.empty() ? std::string(canonical_output_dtype)
        : !tensor.logical_dtype.empty() ? tensor.logical_dtype
                                        : tensor.dtype);
  }
  // Graph completion may start with no chosen public name. Resolve it from
  // actual compiled publication authority, never an invented tensor identity.
  const auto names_primary = [&](const auto& logical) {
    return logical.logical_name == policy.primary_output_name ||
           logical.backend_name == policy.primary_output_name;
  };
  if (policy.primary_output_name.empty() ||
      std::none_of(result.logical_outputs.begin(), result.logical_outputs.end(), names_primary)) {
    policy.primary_output_name.clear();
    for (const auto& route : result.output_order) {
      if (!route.cm_output_name.empty()) {
        policy.primary_output_name = route.cm_output_name;
        break;
      }
    }
    if (policy.primary_output_name.empty()) {
      const auto& logical = result.logical_outputs.front();
      policy.primary_output_name =
          logical.logical_name.empty() ? logical.backend_name : logical.logical_name;
    }
    if (policy.primary_output_name.empty())
      return fail(error, "canonical graph output has no publication name authority");
  }
  // This marker records Core model-contract completion, not kernel admission.
  // The canonical numerical check still runs before any hardware context opens.
  policy.canonical_contract = true;
  *stage = std::move(result);
  return true;
}
} // namespace simaai::neat::pipeline_internal::sima
