// Core describes these operations through Scheduler's standard admission and
// stopped-stage execution routes. BEV-family semantic projection lives here;
// carrier/binding construction is shared with other kernel nodes.
#include "nodes/sima/internal/LidarBevKernelRequests.h"

#include "pipeline/internal/contract/CompiledNodeContractQuery.h"
#include "pipeline/internal/contract/CompiledKernelTensor.h"
#include "pipeline/internal/contract/ContractFacts.h"
#include "pipeline/internal/TensorBufferEnvelope.h"
#include "pipeline/internal/sima/KernelTensorLayout.h"
#include "pipeline/internal/sima/KernelStageRecipe.h"
#include "pipeline/internal/sima/StaticSpecBuilders.h"
#include "pipeline/internal/sima/TensorSemanticsUtil.h"
#include <simaai/scheduler/client/kernel.h>

#include <algorithm>
#include <optional>
#include <stdexcept>
#include <utility>

namespace simaai::neat {
namespace {
namespace kr = nodes_internal::kernel_requests;
namespace rt = ::simaai::scheduler::client;
namespace si = pipeline_internal::sima;
namespace specs = si::specbuilders;
using kr::checked_tensor_bytes;
using kr::dtype_from_token;
using kr::dtype_token;
using kr::Quantization;
using kr::TensorFact;

std::optional<Quantization> quantization(const std::optional<si::QuantStaticSpec>& quant) {
  if (!quant)
    return std::nullopt;
  if (quant->granularity == si::QuantGranularity::PerAxis) {
    if (quant->axis < 0)
      throw std::invalid_argument("per-axis quantization requires a nonnegative logical axis");
    return Quantization{1.0, 0, quant->axis, quant->scales, quant->zero_points};
  }
  if (quant->granularity != si::QuantGranularity::PerTensor || quant->scales.size() != 1 ||
      quant->zero_points.size() != 1)
    throw std::invalid_argument("per-tensor quantization requires one scale and zero point");
  return Quantization{quant->scales.front(), quant->zero_points.front()};
}

std::optional<si::QuantStaticSpec> static_quant(const std::optional<Quantization>& quant) {
  if (!quant)
    return std::nullopt;
  si::QuantStaticSpec result;
  if (quant->axis >= 0) {
    result.granularity = si::QuantGranularity::PerAxis;
    result.axis = quant->axis;
    result.scales = quant->scales;
    result.zero_points = quant->zero_points;
  } else {
    result.scales = {quant->scale};
    result.zero_points = {quant->zero_point};
  }
  return result;
}

// Resolve current ingress storage, never retained route offsets from an older
// producer. The caller owns optional ingress facts for this cold compilation.
const CompiledRuntimeContract& resolve_inputs(const ContractCompileInput& input,
                                              const std::string& node_kind,
                                              std::optional<CompiledRuntimeContract>& ingress) {
  if (const auto* upstream = compiled_runtime_contract_from_stage(input.immediate_upstream)) {
    if (upstream->logical_outputs.empty())
      throw std::invalid_argument(node_kind + " requires a typed upstream tensor contract");
    return *upstream;
  }
  if (!input.ingress.ingress_sample)
    throw std::invalid_argument(node_kind + " requires a typed upstream tensor contract");
  const auto& sample = *input.ingress.ingress_sample;
  const auto tensors = tensors_from_sample(sample, true);
  for (const auto& tensor : tensors)
    if (!tensor.storage || !tensor.storage->size_bytes || tensor.is_composite() ||
        tensor.semantic.tess)
      throw std::invalid_argument(node_kind + " ingress requires typed single-plane tensors; "
                                              "detessellate image/tile inputs first");
  std::string error;
  const auto holder = pipeline_internal::sample_to_gst_envelope_holder(sample, &error);
  pipeline_internal::TensorBufferView view;
  if (!holder || !pipeline_internal::tensor_buffer_descriptor_from_sample(
                     static_cast<GstSample*>(holder.get()), &view, &error))
    throw std::invalid_argument(node_kind + ": cannot resolve ingress carriers: " + error);
  ingress.emplace();
  if (!specs::build_ingress_runtime_contract(view, tensors, &*ingress, &error))
    throw std::invalid_argument(node_kind + ": " + error);
  return *ingress;
}

// A legacy ProcessCVU producer may describe an output in MLA-tiled storage that
// its logical shape does not reveal. These kernels read dense rows; refuse such
// a producer instead of reinterpreting tile bytes.
bool producer_output_tiled(const CompiledNodeContract* producer, int backend_output_index) {
  if (!producer || backend_output_index < 0)
    return false;
  if (const auto canonical =
          pipeline_internal::compiled_kernel_output_tensor(producer, backend_output_index))
    return si::kernel_tensor_requires_detessellation(*canonical);
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

std::vector<TensorFact> bev_tensor_facts(const CompiledRuntimeContract& upstream,
                                         const CompiledNodeContract* producer,
                                         const std::string& node_kind) {
  std::vector<TensorFact> facts;
  facts.reserve(upstream.logical_outputs.size());
  for (const auto& source : upstream.logical_outputs) {
    if (producer_output_tiled(producer, source.backend_output_index))
      throw std::invalid_argument(node_kind + ": upstream tensor " + source.logical_name +
                                  " is MLA-tiled; detessellate it before this Node");
    const auto normalized_layout = si::tensorsemantics::normalize_layout_token(source.layout);
    facts.push_back({source.shape, dtype_from_token(source.dtype, node_kind.c_str()),
                     source.stride_bytes, source.logical_name, quantization(source.quant),
                     normalized_layout.empty() ? source.layout : normalized_layout});
  }
  return facts;
}

bool compile_bev_contract(const std::string& node_kind, const std::string& element_name,
                          const NodeContractDefinition& definition,
                          const CompiledRuntimeContract& upstream, bool ingress,
                          const kr::KernelRequestPlan& plan, CompiledNodeContract* output,
                          std::string* error) {
  if (!output)
    throw std::invalid_argument(node_kind + ": missing contract output");
  if (plan.inputs.size() != upstream.logical_outputs.size() ||
      plan.outputs.size() != plan.published.size())
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
  ::simaai::scheduler::client::TensorStageMember member;
  member.executable = std::move(executable);
  for (std::size_t index = 0; index < upstream.logical_outputs.size(); ++index) {
    const auto& source = upstream.logical_outputs[index];
    const auto physical = std::find_if(
        upstream.physical_outputs.begin(), upstream.physical_outputs.end(),
        [&](const auto& value) { return value.physical_index == source.physical_index; });
    if (physical == upstream.physical_outputs.end())
      throw std::invalid_argument(node_kind + " input has no authored physical carrier");
    std::string binding_error;
    if (!specs::append_direct_kernel_input(&runtime, source, *physical, index,
                                           contract.input_requirements()[index].storage_bytes,
                                           ingress, &binding_error))
      throw std::invalid_argument(node_kind + ": " + binding_error);
    // A port offset is relative to its bound input: the logical input's
    // byte_offset already places the tensor in its carrier.
    member.inputs.push_back({index, 0, ::simaai::scheduler::client::TensorStagePortSource::Input});
  }
  for (std::size_t index = 0; index < plan.outputs.size(); ++index) {
    const auto& published = plan.published[index];
    const auto& requirement = contract.output_requirements()[index];
    const auto dtype = plan.outputs[index].dtype;
    if (requirement.storage_bytes < checked_tensor_bytes(published.shape, dtype, node_kind.c_str()))
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
}
} // namespace

bool SimaVoxelize::compile_node_contract(const ContractCompileInput& input,
                                         CompiledNodeContract* out, std::string* error) const {
  try {
    std::optional<CompiledRuntimeContract> ingress;
    const auto& upstream = resolve_inputs(input, kind(), ingress);
    const auto facts =
        bev_tensor_facts(upstream, ingress ? nullptr : input.immediate_upstream, kind());
    const auto plan = kr::voxelize_request(opt_, facts);
    return compile_bev_contract(kind(), element_names(input.node_index).front(),
                                contract_definition(), upstream, ingress.has_value(), plan, out,
                                error);
  } catch (const std::exception& exception) {
    if (error)
      *error = exception.what();
    return false;
  }
}

bool SimaPillarScatter::compile_node_contract(const ContractCompileInput& input,
                                              CompiledNodeContract* out, std::string* error) const {
  try {
    std::optional<CompiledRuntimeContract> ingress;
    const auto& upstream = resolve_inputs(input, kind(), ingress);
    const auto facts =
        bev_tensor_facts(upstream, ingress ? nullptr : input.immediate_upstream, kind());
    const auto plan = kr::pillar_scatter_request(opt_, facts);
    return compile_bev_contract(kind(), element_names(input.node_index).front(),
                                contract_definition(), upstream, ingress.has_value(), plan, out,
                                error);
  } catch (const std::exception& exception) {
    if (error)
      *error = exception.what();
    return false;
  }
}

bool SimaBEVPoolPlan::compile_node_contract(const ContractCompileInput& input,
                                            CompiledNodeContract* out, std::string* error) const {
  try {
    std::optional<CompiledRuntimeContract> ingress;
    const auto& upstream = resolve_inputs(input, kind(), ingress);
    const auto facts =
        bev_tensor_facts(upstream, ingress ? nullptr : input.immediate_upstream, kind());
    const auto plan = kr::bev_pool_plan_request(opt_, facts);
    return compile_bev_contract(kind(), element_names(input.node_index).front(),
                                contract_definition(), upstream, ingress.has_value(), plan, out,
                                error);
  } catch (const std::exception& exception) {
    if (error)
      *error = exception.what();
    return false;
  }
}

bool SimaBEVPool::compile_node_contract(const ContractCompileInput& input,
                                        CompiledNodeContract* out, std::string* error) const {
  try {
    std::optional<CompiledRuntimeContract> ingress;
    const auto& upstream = resolve_inputs(input, kind(), ingress);
    const auto facts =
        bev_tensor_facts(upstream, ingress ? nullptr : input.immediate_upstream, kind());
    const auto plan = kr::bev_pool_request(opt_, facts);
    return compile_bev_contract(kind(), element_names(input.node_index).front(),
                                contract_definition(), upstream, ingress.has_value(), plan, out,
                                error);
  } catch (const std::exception& exception) {
    if (error)
      *error = exception.what();
    return false;
  }
}

} // namespace simaai::neat
