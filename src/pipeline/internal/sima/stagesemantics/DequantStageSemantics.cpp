#include "pipeline/internal/sima/stagesemantics/DequantStageSemantics.h"

#include "pipeline/internal/sima/StaticSpecBuilders.h"
#include "pipeline/internal/sima/stagesemantics/ProcessCvuStageSemantics.h"

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <sstream>

namespace simaai::neat::pipeline_internal::sima::stagesemantics {
namespace {

bool dequant_contract_compare_enabled() {
  const char* env = std::getenv("SIMA_DEQUANT_CONTRACT_COMPARE");
  return env && env[0] != '\0' && std::strcmp(env, "0") != 0;
}

bool dequant_contract_compare_exit_after_dump() {
  const char* env = std::getenv("SIMA_DEQUANT_CONTRACT_COMPARE_EXIT_AFTER_DUMP");
  return env && env[0] != '\0' && std::strcmp(env, "0") != 0;
}

bool dequant_contract_compare_matches(const std::string& element_name,
                                      const std::string& logical_stage_id) {
  if (!dequant_contract_compare_enabled()) {
    return false;
  }
  const char* env = std::getenv("SIMA_DEQUANT_CONTRACT_COMPARE_STAGE_MATCH");
  if (!env || env[0] == '\0') {
    return true;
  }
  const std::string needle(env);
  return element_name.find(needle) != std::string::npos ||
         logical_stage_id.find(needle) != std::string::npos;
}

std::string ints64_dbg_dequant_local(const std::vector<std::int64_t>& values) {
  std::ostringstream oss;
  oss << "[";
  for (std::size_t i = 0; i < values.size(); ++i) {
    if (i != 0U) {
      oss << ",";
    }
    oss << values[i];
  }
  oss << "]";
  return oss.str();
}

std::string doubles_dbg_dequant_local(const std::vector<double>& values) {
  std::ostringstream oss;
  oss << "[";
  for (std::size_t i = 0; i < values.size(); ++i) {
    if (i != 0U) {
      oss << ",";
    }
    oss << values[i];
  }
  oss << "]";
  return oss.str();
}

const char* quant_granularity_dbg_dequant_local(const QuantGranularity granularity) {
  switch (granularity) {
  case QuantGranularity::PerTensor:
    return "PerTensor";
  case QuantGranularity::PerAxis:
    return "PerAxis";
  }
  return "Unknown";
}

std::string quant_dbg_dequant_local(const std::optional<QuantStaticSpec>& quant) {
  if (!quant.has_value()) {
    return "<none>";
  }
  std::ostringstream oss;
  oss << "{granularity=" << quant_granularity_dbg_dequant_local(quant->granularity)
      << ",axis=" << quant->axis << ",scales=" << doubles_dbg_dequant_local(quant->scales)
      << ",zero_points=" << ints64_dbg_dequant_local(quant->zero_points) << "}";
  return oss.str();
}

const char* device_kind_dbg_dequant_local(const DeviceKind kind) {
  switch (kind) {
  case DeviceKind::Unknown:
    return "Unknown";
  case DeviceKind::Cpu:
    return "Cpu";
  case DeviceKind::Mla:
    return "Mla";
  case DeviceKind::Evxx:
    return "Evxx";
  }
  return "Unknown";
}

std::string logical_input_dbg_dequant_local(const LogicalInputStaticSpec& spec) {
  std::ostringstream oss;
  oss << "{logical_index=" << spec.logical_index
      << ",backend_input_index=" << spec.backend_input_index
      << ",physical_index=" << spec.physical_index
      << ",shape=" << ints64_dbg_dequant_local(spec.shape)
      << ",stride_bytes=" << ints64_dbg_dequant_local(spec.stride_bytes)
      << ",byte_offset=" << spec.byte_offset << ",size_bytes=" << spec.size_bytes << ",dtype=\""
      << spec.dtype << "\"" << ",layout=\"" << spec.layout << "\"" << ",logical_name=\""
      << spec.logical_name << "\"" << ",backend_name=\"" << spec.backend_name << "\""
      << ",segment_name=\"" << spec.segment_name << "\""
      << ",quant=" << quant_dbg_dequant_local(spec.quant) << "}";
  return oss.str();
}

std::string binding_dbg_dequant_local(const InputBindingStaticSpec& spec) {
  std::ostringstream oss;
  oss << "{sink_pad_index=" << spec.sink_pad_index
      << ",local_logical_input_index=" << spec.local_logical_input_index
      << ",src_stage_index=" << spec.src_stage_index << ",src_stage_id=\"" << spec.src_stage_id
      << "\"" << ",src_logical_output_index=" << spec.src_logical_output_index
      << ",src_output_slot=" << spec.src_output_slot
      << ",src_physical_output_index=" << spec.src_physical_output_index
      << ",src_physical_size_bytes=" << spec.src_physical_size_bytes
      << ",src_physical_byte_offset=" << spec.src_physical_byte_offset
      << ",required=" << (spec.required ? 1 : 0) << ",cm_input_name=\"" << spec.cm_input_name
      << "\"" << ",source_segment_name=\"" << spec.source_segment_name << "\"}";
  return oss.str();
}

std::string physical_dbg_dequant_local(const PhysicalBufferStaticSpec& spec) {
  std::ostringstream oss;
  oss << "{physical_index=" << spec.physical_index << ",allocator_index=" << spec.allocator_index
      << ",source_physical_index=" << spec.source_physical_index
      << ",size_bytes=" << spec.size_bytes << ",source_byte_offset=" << spec.source_byte_offset
      << ",device_kind=" << device_kind_dbg_dequant_local(spec.device_kind)
      << ",memory_flags=" << spec.memory_flags << ",segment_name=\"" << spec.segment_name << "\"}";
  return oss.str();
}

std::string logical_output_dbg_dequant_local(const LogicalTensorStaticSpec& spec) {
  std::ostringstream oss;
  oss << "{logical_index=" << spec.logical_index
      << ",backend_output_index=" << spec.backend_output_index
      << ",physical_index=" << spec.physical_index << ",output_slot=" << spec.output_slot
      << ",tensor_index=" << spec.tensor_index << ",byte_offset=" << spec.byte_offset
      << ",size_bytes=" << spec.size_bytes << ",shape=" << ints64_dbg_dequant_local(spec.shape)
      << ",stride_bytes=" << ints64_dbg_dequant_local(spec.stride_bytes) << ",dtype=\""
      << spec.dtype << "\"" << ",layout=\"" << spec.layout << "\"" << ",logical_name=\""
      << spec.logical_name << "\"" << ",backend_name=\"" << spec.backend_name << "\""
      << ",segment_name=\"" << spec.segment_name << "\""
      << ",quant=" << quant_dbg_dequant_local(spec.quant) << "}";
  return oss.str();
}

std::string route_dbg_dequant_local(const StageOutputRoute& route) {
  std::ostringstream oss;
  oss << "{output_slot=" << route.output_slot
      << ",logical_output_index=" << route.logical_output_index
      << ",tensor_index=" << route.tensor_index << ",cm_output_name=\"" << route.cm_output_name
      << "\"" << ",segment_name=\"" << route.segment_name << "\"}";
  return oss.str();
}

void dump_dequant_contract_compare_local(const std::string& element_name,
                                         const std::string& logical_stage_id,
                                         const std::string& plugin_kind,
                                         const CompiledDequantContract& compiled) {
  std::fprintf(stderr,
               "[dequant-compare] element=%s stage_id=%s plugin_kind=%s logical_inputs=%zu "
               "input_bindings=%zu physical_inputs=%zu logical_outputs=%zu physical_outputs=%zu "
               "output_order=%zu\n",
               element_name.c_str(), logical_stage_id.c_str(), plugin_kind.c_str(),
               compiled.runtime_contract.logical_inputs.size(),
               compiled.runtime_contract.input_bindings.size(),
               compiled.runtime_contract.physical_inputs.size(),
               compiled.runtime_contract.logical_outputs.size(),
               compiled.runtime_contract.physical_outputs.size(),
               compiled.runtime_contract.output_order.size());
  for (std::size_t i = 0; i < compiled.runtime_contract.logical_inputs.size(); ++i) {
    std::fprintf(
        stderr, "[dequant-compare] runtime.logical_input index=%zu value=%s\n", i,
        logical_input_dbg_dequant_local(compiled.runtime_contract.logical_inputs[i]).c_str());
  }
  for (std::size_t i = 0; i < compiled.runtime_contract.input_bindings.size(); ++i) {
    std::fprintf(stderr, "[dequant-compare] runtime.input_binding index=%zu value=%s\n", i,
                 binding_dbg_dequant_local(compiled.runtime_contract.input_bindings[i]).c_str());
  }
  for (std::size_t i = 0; i < compiled.runtime_contract.physical_inputs.size(); ++i) {
    std::fprintf(stderr, "[dequant-compare] runtime.physical_input index=%zu value=%s\n", i,
                 physical_dbg_dequant_local(compiled.runtime_contract.physical_inputs[i]).c_str());
  }
  for (std::size_t i = 0; i < compiled.runtime_contract.logical_outputs.size(); ++i) {
    std::fprintf(
        stderr, "[dequant-compare] runtime.logical_output index=%zu value=%s\n", i,
        logical_output_dbg_dequant_local(compiled.runtime_contract.logical_outputs[i]).c_str());
  }
  for (std::size_t i = 0; i < compiled.runtime_contract.physical_outputs.size(); ++i) {
    std::fprintf(stderr, "[dequant-compare] runtime.physical_output index=%zu value=%s\n", i,
                 physical_dbg_dequant_local(compiled.runtime_contract.physical_outputs[i]).c_str());
  }
  for (std::size_t i = 0; i < compiled.runtime_contract.output_order.size(); ++i) {
    std::fprintf(stderr, "[dequant-compare] runtime.output_order index=%zu value=%s\n", i,
                 route_dbg_dequant_local(compiled.runtime_contract.output_order[i]).c_str());
  }
}

void populate_dequant_node_contract_common(
    const std::string& node_kind, const std::string& plugin_kind, const std::string& element_name,
    const std::string& logical_stage_id, const NodeContractDefinition& definition,
    CompiledDequantContract compiled, CompiledNodeContract* out) {
  out->node_kind = node_kind;
  out->plugin_kind = plugin_kind.empty() ? "dequant" : plugin_kind;
  out->element_name = element_name;
  out->logical_stage_id = logical_stage_id.empty() ? element_name : logical_stage_id;
  out->definition = definition;
  compiled.runtime_contract.plugin_kind = out->plugin_kind;
  out->dequant = std::move(compiled);
  out->renderable = true;
}

} // namespace

CompiledDequantContract
build_dequant_compiled_contract_from_facts(const DequantCanonicalFacts& facts) {
  CompiledDequantContract compiled;
  compiled.runtime_contract.plugin_kind = "dequant";

  compiled.runtime_contract.logical_inputs.push_back(specbuilders::build_logical_input_static_spec(
      0, 0, 0, {}, facts.input_dtype, facts.layout, facts.input_name, facts.input_name,
      facts.input_name, 0, 0, TensorMaterializationKind::Direct, facts.input_quant));
  compiled.runtime_contract.input_bindings.push_back(
      specbuilders::build_input_binding_static_spec(0, 0, facts.input_name, facts.input_name));
  compiled.runtime_contract.logical_outputs.push_back(
      specbuilders::build_logical_output_static_spec(0, 0, 0, 0, 0, {}, facts.output_dtype,
                                                     facts.layout, facts.output_name,
                                                     facts.output_name, facts.output_name));
  compiled.runtime_contract.physical_outputs.push_back(
      specbuilders::build_physical_buffer_static_spec(0, 0, 0U, DeviceKind::Cpu,
                                                      facts.output_name));
  compiled.runtime_contract.output_order.push_back(
      specbuilders::build_output_route_static_spec(0, 0, 0, facts.output_name, facts.output_name));

  return compiled;
}

CompiledDequantContract build_dequant_compiled_contract_from_upstream(
    const CompiledRuntimeContract& upstream, const DequantCanonicalFacts& facts, std::string* err) {
  CompiledDequantContract compiled = build_dequant_compiled_contract_from_facts(facts);
  if (upstream.logical_outputs.size() != 1U) {
    if (err) {
      *err = "dequant contract compile requires exactly one upstream logical output";
    }
    return {};
  }

  const auto& input = upstream.logical_outputs.front();
  auto& logical_input = compiled.runtime_contract.logical_inputs.front();
  logical_input.logical_index = input.logical_index >= 0 ? input.logical_index : 0;
  logical_input.backend_input_index = input.backend_output_index;
  logical_input.physical_index = input.physical_index;
  logical_input.shape = input.shape;
  logical_input.stride_bytes = input.stride_bytes;
  logical_input.byte_offset = input.byte_offset;
  logical_input.size_bytes = input.size_bytes;
  logical_input.dtype = input.dtype.empty() ? facts.input_dtype : input.dtype;
  logical_input.layout = input.layout.empty() ? facts.layout : input.layout;
  logical_input.logical_name = input.logical_name.empty() ? facts.input_name : input.logical_name;
  logical_input.backend_name = input.backend_name.empty() ? facts.input_name : input.backend_name;
  logical_input.segment_name = input.segment_name.empty() ? facts.input_name : input.segment_name;
  logical_input.quant = input.quant.has_value() ? input.quant : facts.input_quant;

  auto& binding = compiled.runtime_contract.input_bindings.front();
  binding.local_logical_input_index = logical_input.logical_index;
  binding.src_logical_output_index = input.logical_index;
  binding.src_output_slot = input.output_slot;
  binding.src_physical_output_index = input.physical_index;
  binding.src_physical_size_bytes = input.size_bytes;
  binding.src_physical_byte_offset = input.byte_offset;
  binding.cm_input_name = logical_input.backend_name;
  binding.source_segment_name = logical_input.segment_name;

  auto& logical_output = compiled.runtime_contract.logical_outputs.front();
  logical_output.logical_index = input.logical_index >= 0 ? input.logical_index : 0;
  logical_output.backend_output_index = 0;
  logical_output.physical_index = 0;
  logical_output.output_slot = input.output_slot >= 0 ? input.output_slot : 0;
  logical_output.tensor_index = 0;
  logical_output.byte_offset = 0;
  logical_output.shape = input.shape;
  logical_output.logical_name = facts.output_name;
  logical_output.backend_name = facts.output_name;
  logical_output.segment_name = facts.output_name;
  logical_output.dtype = facts.output_dtype;
  logical_output.layout = input.layout.empty() ? facts.layout : input.layout;
  logical_output.quant.reset();

  logical_output.stride_bytes.clear();
  if (!logical_output.shape.empty()) {
    const std::int64_t elem_bytes = sizeof(float);
    std::uint64_t total_elems = 1U;
    for (const auto dim : logical_output.shape) {
      total_elems *= static_cast<std::uint64_t>(dim > 0 ? dim : 0);
    }
    logical_output.size_bytes = total_elems * static_cast<std::uint64_t>(elem_bytes);
    logical_output.stride_bytes.resize(logical_output.shape.size(), 0);
    std::int64_t running = elem_bytes;
    for (std::size_t i = logical_output.shape.size(); i-- > 0;) {
      logical_output.stride_bytes[i] = running;
      running *= logical_output.shape[i];
    }
  } else {
    logical_output.size_bytes = 0U;
  }

  compiled.runtime_contract.physical_inputs.clear();
  if (!upstream.physical_outputs.empty()) {
    compiled.runtime_contract.physical_inputs.push_back(upstream.physical_outputs.front());
  } else {
    compiled.runtime_contract.physical_inputs.push_back(
        specbuilders::build_physical_buffer_static_spec(input.physical_index, input.physical_index,
                                                        input.size_bytes, DeviceKind::Cpu,
                                                        logical_input.segment_name));
  }
  compiled.runtime_contract.physical_outputs.clear();
  compiled.runtime_contract.physical_outputs.push_back(
      specbuilders::build_physical_buffer_static_spec(0, 0, logical_output.size_bytes,
                                                      DeviceKind::Cpu, facts.output_name));
  compiled.runtime_contract.output_order.clear();
  compiled.runtime_contract.output_order.push_back(specbuilders::build_output_route_static_spec(
      logical_output.output_slot, logical_output.logical_index, logical_output.tensor_index,
      logical_output.backend_name, logical_output.segment_name));
  if (err) {
    err->clear();
  }
  return compiled;
}

bool build_dequant_node_contract(const std::string& node_kind, const std::string& plugin_kind,
                                 const std::string& element_name,
                                 const std::string& logical_stage_id,
                                 const NodeContractDefinition& definition,
                                 const CompiledDequantContract& compiled, CompiledNodeContract* out,
                                 std::string* error_message) {
  if (!out) {
    if (error_message) {
      *error_message = node_kind + " contract compile: output is null";
    }
    return false;
  }
  populate_dequant_node_contract_common(node_kind, plugin_kind, element_name, logical_stage_id,
                                        definition, compiled, out);
  if (dequant_contract_compare_matches(element_name, logical_stage_id)) {
    dump_dequant_contract_compare_local(element_name, logical_stage_id,
                                        plugin_kind.empty() ? std::string("dequant") : plugin_kind,
                                        compiled);
    if (dequant_contract_compare_exit_after_dump()) {
      std::fflush(stderr);
      std::fflush(stdout);
      std::_Exit(0);
    }
  }
  if (error_message) {
    error_message->clear();
  }
  return true;
}

} // namespace simaai::neat::pipeline_internal::sima::stagesemantics
