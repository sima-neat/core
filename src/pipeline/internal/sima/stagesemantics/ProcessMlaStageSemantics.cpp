#include "pipeline/internal/sima/stagesemantics/ProcessMlaStageSemantics.h"
#include "pipeline/internal/sima/StaticSpecBuilders.h"

#include <algorithm>
#include <cctype>

namespace simaai::neat::pipeline_internal::sima::stagesemantics {
namespace {

std::string upper_copy_local(std::string value) {
  std::transform(value.begin(), value.end(), value.begin(),
                 [](unsigned char c) { return static_cast<char>(std::toupper(c)); });
  return value;
}

std::size_t mla_dtype_size_bytes(const std::string& dtype_raw) {
  const std::string dtype = upper_copy_local(dtype_raw);
  if (dtype == "FP32" || dtype == "FLOAT" || dtype == "FLOAT32" || dtype == "INT32" ||
      dtype == "UINT32") {
    return 4U;
  }
  if (dtype == "BF16" || dtype == "FP16" || dtype == "FLOAT16" || dtype == "INT16" ||
      dtype == "UINT16") {
    return 2U;
  }
  return 1U;
}

std::uint64_t tensor_static_logical_size_bytes(const TensorStaticSpec& tensor) {
  if (tensor.max_stride > 0) {
    return static_cast<std::uint64_t>(tensor.max_stride);
  }
  if (!tensor.shape.empty()) {
    std::uint64_t total = static_cast<std::uint64_t>(mla_dtype_size_bytes(tensor.dtype));
    for (const auto dim : tensor.shape) {
      if (dim <= 0) {
        return 0U;
      }
      total *= static_cast<std::uint64_t>(dim);
    }
    return total;
  }
  return 0U;
}

PhysicalBufferStaticSpec synthesize_mla_physical_input(const TensorStaticSpec& tensor, int index) {
  return specbuilders::build_physical_buffer_static_spec(
      index, index, tensor_static_logical_size_bytes(tensor), DeviceKind::Unknown,
      tensor.semantic_tag, index, 0);
}

LogicalInputStaticSpec mla_logical_input_from_facts(const TensorStaticSpec& tensor,
                                                    const PhysicalBufferStaticSpec& physical,
                                                    int index) {
  const std::string logical_name = !tensor.semantic_tag.empty()
                                       ? tensor.semantic_tag
                                       : ("input_tensor_" + std::to_string(index));
  return specbuilders::build_logical_input_static_spec(
      index, tensor.tensor_index >= 0 ? tensor.tensor_index : index,
      physical.physical_index >= 0 ? physical.physical_index : index, tensor.shape, tensor.dtype,
      tensor.layout, logical_name, logical_name, physical.segment_name,
      std::max<std::int64_t>(0, physical.source_byte_offset));
}

// Returns the firmware-side IFM slot name for input `index`. If the ELF
// topology has been parsed and surfaces a placeholder symbol at that index,
// use it (e.g. "data.ifm.persistent.input_00/MLA_0/placeholder_0_0.b0");
// otherwise fall back to the legacy synthesized "ifmN" naming. This keeps
// the carrier path correct for both monolithic and native-multi-IFM .elfs.
std::string mla_ifm_backend_name_for_index(const std::vector<std::string>& elf_ifm_symbol_names,
                                           std::size_t index) {
  if (index < elf_ifm_symbol_names.size() && !elf_ifm_symbol_names[index].empty()) {
    return elf_ifm_symbol_names[index];
  }
  return "ifm" + std::to_string(index);
}

LogicalInputStaticSpec
mla_carrier_logical_input_from_physical(const PhysicalBufferStaticSpec& physical, int index,
                                        const std::vector<std::string>& elf_ifm_symbol_names) {
  const std::string backend_name =
      mla_ifm_backend_name_for_index(elf_ifm_symbol_names, static_cast<std::size_t>(index));
  const std::string logical_name =
      !physical.segment_name.empty() ? physical.segment_name : backend_name;
  return specbuilders::build_logical_input_static_spec(
      index, index, physical.physical_index >= 0 ? physical.physical_index : index, {}, "INT8", {},
      logical_name, backend_name, physical.segment_name, 0, physical.size_bytes);
}

InputBindingStaticSpec mla_input_binding_from_facts(const LogicalInputStaticSpec& logical,
                                                    const PhysicalBufferStaticSpec& physical) {
  return specbuilders::build_input_binding_static_spec(
      0, logical.logical_index, logical.backend_name, physical.segment_name, -1, -1,
      physical.source_physical_index >= 0 ? physical.source_physical_index
                                          : physical.physical_index,
      physical.size_bytes, std::max<std::int64_t>(0, physical.source_byte_offset), true);
}

InputBindingStaticSpec
mla_input_binding_with_defaults(const LogicalInputStaticSpec& logical,
                                const PhysicalBufferStaticSpec& physical,
                                const InputBindingStaticSpec* explicit_binding) {
  InputBindingStaticSpec binding =
      explicit_binding ? *explicit_binding : mla_input_binding_from_facts(logical, physical);
  if (binding.local_logical_input_index < 0) {
    binding.local_logical_input_index = logical.logical_index;
  }
  if (binding.cm_input_name.empty()) {
    binding.cm_input_name = logical.backend_name;
  }
  if (binding.source_segment_name.empty()) {
    binding.source_segment_name = physical.segment_name;
  }
  if (binding.src_physical_output_index < 0) {
    binding.src_physical_output_index = physical.source_physical_index >= 0
                                            ? physical.source_physical_index
                                            : physical.physical_index;
  }
  const std::uint64_t logical_or_physical_size =
      logical.size_bytes > 0U ? logical.size_bytes : physical.size_bytes;
  if (binding.src_physical_size_bytes == 0U) {
    binding.src_physical_size_bytes = logical_or_physical_size;
  }
  if (physical.size_bytes > 0U && (binding.src_physical_size_bytes == 0U ||
                                   binding.src_physical_size_bytes > physical.size_bytes)) {
    binding.src_physical_size_bytes = physical.size_bytes;
  }
  if (binding.src_physical_byte_offset < 0) {
    binding.src_physical_byte_offset = std::max<std::int64_t>(0, physical.source_byte_offset);
  }
  return binding;
}

StageOutputRoute mla_output_route_from_logical(const LogicalTensorStaticSpec& logical,
                                               int fallback_index) {
  return specbuilders::build_output_route_static_spec(
      logical.output_slot >= 0 ? logical.output_slot : fallback_index,
      logical.logical_index >= 0 ? logical.logical_index : fallback_index, logical.tensor_index,
      !logical.backend_name.empty() ? logical.backend_name : logical.segment_name,
      logical.segment_name);
}

void populate_mla_node_contract_common(const std::string& node_kind,
                                       const std::string& element_name,
                                       const std::string& logical_stage_id,
                                       const NodeContractDefinition& definition,
                                       CompiledMlaContract compiled, CompiledNodeContract* out) {
  out->node_kind = node_kind;
  out->plugin_kind = "processmla";
  out->element_name = element_name;
  out->logical_stage_id = logical_stage_id.empty() ? element_name : logical_stage_id;
  out->definition = definition;
  out->processmla = std::move(compiled);
  out->renderable = true;
}

} // namespace

CompiledMlaContract build_mla_compiled_contract(const MlaStaticContract& contract) {
  const bool include_dispatcher_output_names = contract.dispatcher_physical_outputs.size() > 1U;
  const auto subset = plugin_contracts::extract_processmla_contract_subset_from_static_contract(
      contract, include_dispatcher_output_names);
  return build_mla_compiled_contract_from_subset(subset, contract);
}

CompiledMlaContract
build_mla_compiled_contract_from_subset(const plugin_contracts::ProcessMlaContractSubset& subset,
                                        const MlaStaticContract& contract) {
  CompiledMlaContract compiled;
  compiled.payload = plugin_contracts::build_processmla_payload_from_subset(subset);
  compiled.inputs = contract.inputs;
  compiled.dispatcher_physical_outputs = contract.dispatcher_physical_outputs;
  if (compiled.payload.batch_size <= 0) {
    compiled.payload.batch_size = 1;
  }
  if (compiled.payload.batch_sz_model <= 0) {
    compiled.payload.batch_sz_model = compiled.payload.batch_size;
  }
  compiled.runtime_contract.plugin_kind = "processmla";
  compiled.runtime_contract.consumer_keeps_distinct_physical_inputs =
      contract.consumer_keeps_distinct_physical_inputs;
  compiled.runtime_contract.elf_ifm_symbol_names = contract.elf_ifm_symbol_names;
  compiled.runtime_contract.elf_ofm_symbol_names = contract.elf_ofm_symbol_names;

  compiled.runtime_contract.physical_inputs = contract.physical_inputs;
  const auto& logical_inputs =
      contract.logical_inputs.empty() ? contract.inputs : contract.logical_inputs;
  if (!logical_inputs.empty()) {
    compiled.runtime_contract.logical_inputs.reserve(logical_inputs.size());
    for (std::size_t i = 0; i < logical_inputs.size(); ++i) {
      PhysicalBufferStaticSpec physical =
          (i < compiled.runtime_contract.physical_inputs.size())
              ? compiled.runtime_contract.physical_inputs[i]
              : synthesize_mla_physical_input(logical_inputs[i], static_cast<int>(i));
      auto logical = mla_logical_input_from_facts(logical_inputs[i], physical, static_cast<int>(i));
      logical.size_bytes = tensor_static_logical_size_bytes(logical_inputs[i]);
      compiled.runtime_contract.logical_inputs.push_back(std::move(logical));
      if (i >= compiled.runtime_contract.physical_inputs.size()) {
        compiled.runtime_contract.physical_inputs.push_back(std::move(physical));
      }
    }
  } else if (!compiled.runtime_contract.physical_inputs.empty()) {
    compiled.runtime_contract.logical_inputs.reserve(
        compiled.runtime_contract.physical_inputs.size());
    for (std::size_t i = 0; i < compiled.runtime_contract.physical_inputs.size(); ++i) {
      compiled.runtime_contract.logical_inputs.push_back(mla_carrier_logical_input_from_physical(
          compiled.runtime_contract.physical_inputs[i], static_cast<int>(i),
          compiled.runtime_contract.elf_ifm_symbol_names));
    }
  }

  const bool explicit_bindings_align_with_carriers =
      contract.input_bindings.size() == compiled.runtime_contract.physical_inputs.size();
  compiled.runtime_contract.input_bindings.reserve(
      compiled.runtime_contract.physical_inputs.size());
  for (std::size_t i = 0; i < compiled.runtime_contract.physical_inputs.size(); ++i) {
    const auto& physical = compiled.runtime_contract.physical_inputs[i];
    const auto& logical =
        compiled.runtime_contract
            .logical_inputs[std::min(i, compiled.runtime_contract.logical_inputs.size() - 1U)];
    const InputBindingStaticSpec* explicit_binding =
        explicit_bindings_align_with_carriers ? &contract.input_bindings[i] : nullptr;
    compiled.runtime_contract.input_bindings.push_back(
        mla_input_binding_with_defaults(logical, physical, explicit_binding));
  }

  compiled.runtime_contract.physical_outputs = contract.physical_outputs;
  compiled.runtime_contract.logical_outputs = contract.logical_outputs;
  compiled.runtime_contract.output_quant = contract.output_quant;
  for (std::size_t i = 0; i < contract.logical_outputs.size(); ++i) {
    compiled.runtime_contract.output_order.push_back(
        mla_output_route_from_logical(contract.logical_outputs[i], static_cast<int>(i)));
  }
  return compiled;
}

bool build_mla_node_contract(const std::string& node_kind, const std::string& element_name,
                             const std::string& logical_stage_id,
                             const NodeContractDefinition& definition,
                             const CompiledMlaContract& compiled, CompiledNodeContract* out,
                             std::string* error_message) {
  if (!out) {
    if (error_message) {
      *error_message = node_kind + " contract compile: output is null";
    }
    return false;
  }
  populate_mla_node_contract_common(node_kind, element_name, logical_stage_id, definition, compiled,
                                    out);
  if (error_message) {
    error_message->clear();
  }
  return true;
}

} // namespace simaai::neat::pipeline_internal::sima::stagesemantics
