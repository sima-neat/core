#include "nodes/sima/internal/KernelRequestStage.h"

#include "gst/GstHelpers.h"
#include "pipeline/internal/contract/CompiledNodeContractQuery.h"
#include "pipeline/internal/contract/ContractFacts.h"
#include "pipeline/internal/TensorBufferEnvelope.h"
#include "pipeline/internal/sima/KernelStageRecipe.h"
#include "pipeline/internal/sima/StaticSpecBuilders.h"
#include <simaai/scheduler/client/kernel.h>

#include <algorithm>
#include <optional>
#include <stdexcept>
#include <utility>

namespace simaai::neat::nodes_internal::kernel_requests {
namespace {
namespace si = pipeline_internal::sima;
namespace specs = si::specbuilders;

const char* ingress_dtype_token(int dtype) {
  switch (dtype) {
  case SIMA_TENSOR_SET_DTYPE_UINT8_V1:
    return "UINT8";
  case SIMA_TENSOR_SET_DTYPE_INT8_V1:
    return "INT8";
  case SIMA_TENSOR_SET_DTYPE_UINT16_V1:
    return "UINT16";
  case SIMA_TENSOR_SET_DTYPE_INT16_V1:
    return "INT16";
  case SIMA_TENSOR_SET_DTYPE_BF16_V1:
    return "BF16";
  case SIMA_TENSOR_SET_DTYPE_INT32_V1:
    return "INT32";
  case SIMA_TENSOR_SET_DTYPE_FP32_V1:
    return "FP32";
  default:
    return nullptr;
  }
}

std::optional<PerTensorQuant> per_tensor(const std::optional<si::QuantStaticSpec>& quant) {
  if (!quant || quant->granularity != si::QuantGranularity::PerTensor ||
      quant->scales.size() != 1 || quant->zero_points.size() != 1)
    return std::nullopt;
  return PerTensorQuant{quant->scales.front(), quant->zero_points.front()};
}

std::optional<si::QuantStaticSpec> static_quant(const std::optional<PerTensorQuant>& quant) {
  if (!quant)
    return std::nullopt;
  si::QuantStaticSpec result;
  result.scales = {quant->scale};
  result.zero_points = {quant->zero_point};
  return result;
}

// Graph-ingress carriers, resolved from the current storage envelope rather
// than producer route provenance. This is cold admission, not a per-frame copy.
// `tensor_bytes` holds each tensor's own extent; its logical output spans the
// rest of the carrier.
struct IngressTensors {
  CompiledRuntimeContract contract;
  std::vector<std::uint64_t> tensor_bytes;
};

IngressTensors ingress_tensors(const Sample& sample, const std::string& node) {
  IngressTensors ingress_result;
  auto& result = ingress_result.contract;
  const auto tensors = tensors_from_sample(sample, true);
  for (const auto& tensor : tensors) {
    if (!tensor.storage || !tensor.storage->size_bytes || tensor.is_composite() ||
        tensor.semantic.tess)
      throw std::invalid_argument(node + " ingress requires typed single-plane tensors; "
                                         "detessellate image/tile inputs first");
  }
  std::string error;
  const auto holder = pipeline_internal::sample_to_gst_envelope_holder(sample, &error);
  pipeline_internal::TensorBufferView ingress;
  if (!holder || !pipeline_internal::tensor_buffer_descriptor_from_sample(
                     static_cast<GstSample*>(holder.get()), &ingress, &error))
    throw std::invalid_argument(node + ": cannot resolve ingress carriers: " + error);
  if (ingress.tensors.size() != tensors.size())
    throw std::invalid_argument(node + " ingress tensor count changed during projection");
  for (std::size_t i = 0; i < ingress.tensors.size(); ++i) {
    const auto& tensor = ingress.tensors[i];
    const char* dtype = ingress_dtype_token(tensor.dtype);
    if (!dtype)
      throw std::invalid_argument(node + " ingress tensor " + std::to_string(i) +
                                  " has an unsupported dtype");
    const int physical = tensor.memory_index;
    if (physical < 0 || static_cast<guint>(physical) >= gst_buffer_n_memory(ingress.buffer))
      throw std::invalid_argument(node + " ingress tensor has no physical carrier");
    const auto bytes =
        gst_memory_get_sizes(gst_buffer_peek_memory(ingress.buffer, physical), nullptr, nullptr);
    const auto offset = tensor.byte_offset;
    if (offset < 0 || static_cast<std::uint64_t>(offset) >= bytes || !tensor.size_bytes ||
        tensor.size_bytes > bytes - static_cast<std::uint64_t>(offset))
      throw std::invalid_argument(node + " ingress tensor offset exceeds its backing memory");
    const auto name =
        tensor.logical_name.empty() ? "input_" + std::to_string(i) : tensor.logical_name;
    auto logical = specs::build_logical_output_static_spec(
        static_cast<int>(i), static_cast<int>(i), physical, static_cast<int>(i),
        static_cast<int>(i), tensor.shape, dtype, "", name, name, name, offset,
        bytes - static_cast<std::uint64_t>(offset));
    logical.stride_bytes = tensor.stride_bytes;
    if (const auto& quant = tensors[i].semantic.quant; quant && quant->axis < 0)
      logical.quant = static_quant(PerTensorQuant{quant->scale, quant->zero_point});
    result.logical_outputs.push_back(std::move(logical));
    ingress_result.tensor_bytes.push_back(tensor.size_bytes);
    const auto existing =
        std::find_if(result.physical_outputs.begin(), result.physical_outputs.end(),
                     [physical](const auto& value) { return value.physical_index == physical; });
    if (existing == result.physical_outputs.end())
      result.physical_outputs.push_back(specs::build_physical_buffer_static_spec(
          physical, physical, bytes, si::DeviceKind::Cpu, name));
    else if (existing->size_bytes != bytes)
      throw std::invalid_argument(node + " ingress tensors disagree on shared carrier extent");
  }
  return ingress_result;
}

// A legacy ProcessCVU producer may describe an output in MLA-tiled storage that
// its logical shape does not reveal. These kernels read dense rows; refuse such
// a producer instead of reinterpreting tile bytes.
bool producer_output_tiled(const CompiledNodeContract* producer, int backend_output_index) {
  if (!producer || backend_output_index < 0)
    return false;
  const si::ProcessCvuStagePayload* legacy = nullptr;
  if (producer->processcvu)
    legacy = &producer->processcvu->payload;
  else if (producer->transport && producer->transport->processcvu_payload)
    legacy = &*producer->transport->processcvu_payload;
  if (!legacy || static_cast<std::size_t>(backend_output_index) >= legacy->output_tensors.size())
    return false;
  return legacy->output_tensors[backend_output_index].descriptor().layout_kind ==
         SIMA_EV_LAYOUT_TILED;
}

std::uint64_t dense_bytes(const std::vector<std::int64_t>& shape, rt::DType dtype) {
  std::uint64_t bytes = scalar_bytes(dtype);
  for (const auto dimension : shape)
    bytes *= static_cast<std::uint64_t>(dimension);
  return bytes;
}
} // namespace

bool compile_kernel_request_stage(const std::string& node_kind, const std::string& element_name,
                                  const NodeContractDefinition& definition,
                                  const ContractCompileInput& input, const RequestBuilder& build,
                                  CompiledNodeContract* output, std::string* error) {
  try {
    if (!output)
      throw std::invalid_argument(node_kind + ": missing contract output");
    const auto* upstream = compiled_runtime_contract_from_stage(input.immediate_upstream);
    std::optional<IngressTensors> ingress;
    if (!upstream && input.ingress.ingress_sample) {
      ingress = ingress_tensors(*input.ingress.ingress_sample, node_kind);
      upstream = &ingress->contract;
    }
    if (!upstream || upstream->logical_outputs.empty())
      throw std::invalid_argument(node_kind + " requires a typed upstream tensor contract");

    std::vector<TensorFact> facts;
    facts.reserve(upstream->logical_outputs.size());
    for (const auto& source : upstream->logical_outputs) {
      if (!ingress && producer_output_tiled(input.immediate_upstream, source.backend_output_index))
        throw std::invalid_argument(node_kind + ": upstream tensor " + source.logical_name +
                                    " is MLA-tiled; detessellate it before this Node");
      facts.push_back({source.shape, dtype_from_token(source.dtype, node_kind.c_str()),
                       source.stride_bytes, source.logical_name, per_tensor(source.quant)});
    }
    const auto plan = build(facts);
    if (plan.inputs.size() != facts.size() || plan.outputs.size() != plan.published.size())
      throw std::logic_error(node_kind + ": request builder changed port arity");

    auto checked = ::simaai::scheduler::client::check_kernel(
        {plan.operation, plan.target, plan.inputs, plan.outputs, plan.parameters});
    if (const auto* failure = std::get_if<rt::Error>(&checked))
      throw std::invalid_argument(node_kind + ": " + plan.operation +
                                  " admission failed: " + failure->field + ": " + failure->message);
    auto executable = std::get<rt::KernelExecutable>(std::move(checked));
    auto inspected = ::simaai::scheduler::client::inspect_kernel(executable);
    if (const auto* failure = std::get_if<rt::Error>(&inspected))
      throw std::invalid_argument(node_kind + ": " + failure->message);
    const auto contract = std::get<rt::KernelInspection>(std::move(inspected));
    if (contract.target() != plan.target ||
        contract.input_requirements().size() != plan.inputs.size() ||
        contract.output_requirements().size() != plan.outputs.size())
      throw std::invalid_argument(node_kind + ": checked executable differs from its request");

    CompiledKernelContract compiled;
    auto& runtime = compiled.runtime_contract;
    runtime.plugin_kind = "kernel";
    runtime.physical_inputs = upstream->physical_outputs;
    ::simaai::scheduler::client::TensorStageMember member;
    member.executable = std::move(executable);
    for (std::size_t index = 0; index < upstream->logical_outputs.size(); ++index) {
      const auto& source = upstream->logical_outputs[index];
      if (source.byte_offset < 0)
        throw std::invalid_argument(node_kind + " input has a negative byte offset");
      const auto physical = std::find_if(
          runtime.physical_inputs.begin(), runtime.physical_inputs.end(),
          [&](const auto& value) { return value.physical_index == source.physical_index; });
      if (physical == runtime.physical_inputs.end())
        throw std::invalid_argument(node_kind + " input has no authored physical carrier");
      const auto required = contract.input_requirements()[index].storage_bytes;
      if (static_cast<std::uint64_t>(source.byte_offset) > physical->size_bytes ||
          required > physical->size_bytes - static_cast<std::uint64_t>(source.byte_offset))
        throw std::invalid_argument(node_kind + " input " + std::to_string(index) + " holds " +
                                    std::to_string(physical->size_bytes) +
                                    " bytes; the checked request reads " +
                                    std::to_string(required));
      // Several tensors may share one carrier: the read must also stay inside
      // this tensor, not run on into its neighbour. Zero means unknown.
      const std::uint64_t own =
          ingress ? ingress->tensor_bytes[index] : static_cast<std::uint64_t>(source.size_bytes);
      if (own && required > own)
        throw std::invalid_argument(node_kind + " input " + std::to_string(index) + " is " +
                                    std::to_string(own) + " bytes; the checked request reads " +
                                    std::to_string(required));
      auto logical = specs::build_logical_input_static_spec(
          static_cast<int>(index), static_cast<int>(index), source.physical_index, source.shape,
          source.dtype, source.layout, source.logical_name, source.backend_name,
          source.segment_name, source.byte_offset, source.size_bytes,
          si::TensorMaterializationKind::Direct, source.quant);
      logical.stride_bytes = source.stride_bytes;
      runtime.logical_inputs.push_back(std::move(logical));
      runtime.input_bindings.push_back(specs::build_input_binding_static_spec(
          0, static_cast<int>(index), source.backend_name, source.segment_name,
          ingress ? -1 : source.logical_index, ingress ? -1 : source.output_slot,
          ingress ? -1 : source.physical_index, ingress ? 0 : physical->size_bytes, 0, true));
      // A port offset is relative to its bound input: the logical input's
      // byte_offset already places the tensor in its carrier.
      member.inputs.push_back(
          {index, 0, ::simaai::scheduler::client::TensorStagePortSource::Input});
    }
    for (std::size_t index = 0; index < plan.outputs.size(); ++index) {
      const auto& published = plan.published[index];
      const auto& requirement = contract.output_requirements()[index];
      const auto dtype = plan.outputs[index].dtype;
      if (requirement.storage_bytes < dense_bytes(published.shape, dtype))
        throw std::invalid_argument(node_kind + ": checked " + published.name +
                                    " storage is smaller than its published tensor");
      const auto i = static_cast<int>(index);
      runtime.physical_outputs.push_back(specs::build_physical_buffer_static_spec(
          i, i, requirement.storage_bytes, si::DeviceKind::Cpu, published.name, -1, 0,
          requirement.alignment_bytes));
      auto logical = specs::build_logical_output_static_spec(
          i, i, i, i, i, published.shape, dtype_token(dtype), published.layout, published.name,
          published.name, published.name, 0, requirement.storage_bytes,
          static_quant(published.quant));
      logical.stride_bytes = published.stride_bytes;
      // A tiled checked output needs an explicit physical-view authority; its
      // logical shape is unchanged.
      if (published.tiled)
        logical.publication_relation = si::PublicationViewRelation::PackedStorage;
      runtime.logical_outputs.push_back(std::move(logical));
      runtime.output_order.push_back(
          specs::build_output_route_static_spec(i, i, i, published.name, published.name));
      member.outputs.push_back(
          {index, 0, ::simaai::scheduler::client::TensorStagePortSource::Output});
    }
    auto recipe = std::make_shared<si::KernelStageRecipe>();
    recipe->target = plan.target;
    recipe->members.push_back(std::move(member));
    compiled.recipe = std::move(recipe);

    output->node_kind = node_kind;
    output->plugin_kind = "kernel";
    output->element_name = element_name;
    output->logical_stage_id = element_name;
    output->definition = definition;
    output->kernel = std::move(compiled);
    output->renderable = true;
    if (error)
      error->clear();
    return true;
  } catch (const std::exception& exception) {
    if (error)
      *error = exception.what();
    return false;
  }
}

std::string kernel_runner_element_name(int node_index, const std::string& suffix) {
  return "n" + std::to_string(node_index) + "_" + suffix;
}

std::string kernel_runner_fragment(int node_index, const std::string& suffix, const char* node) {
  const auto context = std::string(node) + "::backend_fragment";
  require_element("neatkernelrunner", context.c_str());
  return "neatkernelrunner name=" + kernel_runner_element_name(node_index, suffix);
}

NodeContractDefinition kernel_contract_definition(const std::string& node_kind,
                                                  std::span<const char* const> inputs,
                                                  std::span<const char* const> outputs) {
  NodeContractDefinition result;
  result.node_kind = node_kind;
  result.plugin_kind = "kernel";
  for (const auto* id : inputs) {
    ContractPortSpec port;
    port.port_id = id;
    port.media_type = "application/vnd.simaai.tensor";
    result.inputs.push_back(std::move(port));
  }
  for (const auto* id : outputs) {
    ContractPortSpec port;
    port.port_id = id;
    port.media_type = "application/vnd.simaai.tensor";
    result.outputs.push_back(std::move(port));
  }
  return result;
}

} // namespace simaai::neat::nodes_internal::kernel_requests
