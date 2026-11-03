#include "nodes/sima/SimaArgMax.h"

#include "gst/GstHelpers.h"
#include "pipeline/internal/contract/CompiledNodeContractQuery.h"
#include "pipeline/internal/contract/CompiledKernelTensor.h"
#include "pipeline/internal/sima/KernelTensorLayout.h"
#include "pipeline/internal/contract/ContractFacts.h"
#include "pipeline/internal/TensorBufferEnvelope.h"
#include "pipeline/internal/sima/KernelStageRecipe.h"
#include "pipeline/internal/sima/StaticSpecBuilders.h"
#include "pipeline/internal/sima/CanonicalKernelLowering.h"
#include <simaai/scheduler/client/kernel.h>

#include <algorithm>
#include <array>
#include <cctype>
#include <limits>
#include <stdexcept>
#include <utility>

namespace simaai::neat {
namespace {
namespace rt = ::simaai::scheduler::client;
namespace scheduler = ::simaai::scheduler::client;
namespace si = pipeline_internal::sima;
namespace specs = si::specbuilders;
rt::DType kernel_dtype(std::string token) {
  std::transform(token.begin(), token.end(), token.begin(),
                 [](unsigned char c) { return static_cast<char>(std::toupper(c)); });
  if (token == "FP32" || token == "FLOAT32")
    return rt::DType::Float32;
  if (token == "BF16" || token == "BFLOAT16")
    return rt::DType::BFloat16;
  if (token == "INT8")
    return rt::DType::Int8;
  if (token == "UINT8")
    return rt::DType::UInt8;
  if (token == "INT16")
    return rt::DType::Int16;
  if (token == "INT32")
    return rt::DType::Int32;
  throw std::invalid_argument("SimaArgMax: unsupported input dtype " + token);
}
// Resolve actual ingress storage before projecting the shared static contract.
// ArgMax's typed-input policy stays here; carrier identity and view sizing do not.
CompiledRuntimeContract ingress_tensors(const Sample& sample) {
  const auto tensors = tensors_from_sample(sample, true);
  for (const auto& tensor : tensors) {
    if (!tensor.storage || !tensor.storage->size_bytes || tensor.is_composite() ||
        tensor.semantic.tess)
      throw std::invalid_argument("SimaArgMax ingress requires typed single-plane tensors; "
                                  "detessellate image/tile inputs first");
    if (tensor.dtype != TensorDType::UInt8 && tensor.dtype != TensorDType::Int8 &&
        tensor.dtype != TensorDType::BFloat16 && tensor.dtype != TensorDType::Float32)
      throw std::invalid_argument("SimaArgMax ingress supports INT8/UINT8/BF16/FP32");
  }
  std::string error;
  const auto holder = pipeline_internal::sample_to_gst_envelope_holder(sample, &error);
  pipeline_internal::TensorBufferView ingress;
  CompiledRuntimeContract result;
  if (!holder ||
      !pipeline_internal::tensor_buffer_descriptor_from_sample(
          static_cast<GstSample*>(holder.get()), &ingress, &error) ||
      !specs::build_ingress_runtime_contract(ingress, tensors, &result, &error))
    throw std::invalid_argument("SimaArgMax: cannot resolve ingress carriers: " + error);
  return result;
}
} // namespace

SimaArgMax::SimaArgMax(SimaArgMaxOptions opt) : opt_(std::move(opt)) {
  // These properties never existed on the current ArgMax element. Keep the
  // public fields source-compatible, but reject requests that could not run.
  if (!opt_.config_path.empty() || opt_.emit_signals || opt_.transmit ||
      (opt_.sima_allocator_type != 0 && opt_.sima_allocator_type != 2))
    throw std::invalid_argument(
        "SimaArgMax: legacy config/transport/allocator properties are not supported; "
        "use axis and keepdims with a typed tensor input");
}

bool SimaArgMax::has_config_json() const {
  return false;
}
bool SimaArgMax::wire_input_names(const std::vector<std::string>&, const std::string&) {
  return false;
}

std::string SimaArgMax::backend_fragment(int node_index) const {
  require_element("neatkernelrunner", "SimaArgMax::backend_fragment");
  return "neatkernelrunner name=n" + std::to_string(node_index) + "_argmax";
}
std::vector<std::string> SimaArgMax::element_names(int node_index) const {
  return {"n" + std::to_string(node_index) + "_argmax"};
}
OutputSpec SimaArgMax::output_spec(const OutputSpec& input) const {
  OutputSpec out;
  out.media_type = "application/vnd.simaai.tensor";
  out.format = "ARGMAX";
  out.memory = input.memory;
  out.certainty = SpecCertainty::Hint;
  out.note = "neatkernelrunner";
  return out;
}
NodeContractDefinition SimaArgMax::contract_definition() const {
  NodeContractDefinition result;
  result.node_kind = kind();
  result.plugin_kind = "kernel";
  ContractPortSpec input;
  input.port_id = "input_tensor";
  input.media_type = "application/vnd.simaai.tensor";
  result.inputs.push_back(input);
  input.port_id = "output_tensor";
  result.outputs.push_back(std::move(input));
  return result;
}
bool SimaArgMax::compile_node_contract(const ContractCompileInput& input,
                                       CompiledNodeContract* output, std::string* error) const {
  try {
    if (!output)
      throw std::invalid_argument("SimaArgMax: missing contract output");
    const auto* upstream = compiled_runtime_contract_from_stage(input.immediate_upstream);
    std::optional<CompiledRuntimeContract> ingress;
    if (!upstream && input.ingress.ingress_sample) {
      ingress = ingress_tensors(*input.ingress.ingress_sample);
      upstream = &*ingress;
    }
    if (!upstream || upstream->logical_outputs.empty())
      throw std::invalid_argument("SimaArgMax requires a typed upstream tensor contract");
    CompiledKernelContract compiled;
    auto& runtime = compiled.runtime_contract;
    runtime.plugin_kind = "kernel";
    auto recipe = std::make_shared<si::KernelStageRecipe>();
    for (std::size_t index = 0; index < upstream->logical_outputs.size(); ++index) {
      const auto& source = upstream->logical_outputs[index];
      if (source.shape.empty() ||
          source.shape.size() > std::size(sima_ev_tensor_desc{}.shape.sizes) ||
          source.byte_offset < 0)
        throw std::invalid_argument("SimaArgMax requires a positive tensor rank of at most six");
      const auto rank = static_cast<int>(source.shape.size());
      const int axis = opt_.axis < 0 ? opt_.axis + rank : opt_.axis;
      if (axis < 0 || axis >= rank)
        throw std::invalid_argument("SimaArgMax axis is outside input rank");
      const auto physical = std::find_if(
          upstream->physical_outputs.begin(), upstream->physical_outputs.end(),
          [&](const auto& value) { return value.physical_index == source.physical_index; });
      if (physical == upstream->physical_outputs.end())
        throw std::invalid_argument("SimaArgMax input has no authored physical carrier");
      rt::TensorSpec in{source.shape, kernel_dtype(source.dtype)};
      if (!source.stride_bytes.empty()) {
        if (source.stride_bytes.size() != source.shape.size())
          throw std::invalid_argument("SimaArgMax input strides do not match rank");
        ::simaai::scheduler::client::Strided strided;
        std::copy(source.stride_bytes.begin(), source.stride_bytes.end(),
                  strided.strides_bytes.begin());
        in.layout = strided;
      }
      // A logical HWC/CHW label is not a physical storage layout. Reuse
      // the producer's exact checked tensor description whenever it exists.
      const si::ProcessCvuStagePayload* legacy = nullptr;
      if (input.immediate_upstream) {
        const auto& producer = *input.immediate_upstream;
        if (producer.processcvu)
          legacy = &producer.processcvu->payload;
        else if (producer.transport && producer.transport->processcvu_payload)
          legacy = &*producer.transport->processcvu_payload;
      }
      if (legacy && source.backend_output_index >= 0 &&
          static_cast<std::size_t>(source.backend_output_index) < legacy->output_tensors.size()) {
        const auto& descriptor = legacy->output_tensors[source.backend_output_index];
        if (descriptor.descriptor().layout_kind == SIMA_EV_LAYOUT_TILED) {
          auto native = si::kernel_lowering::tensor_spec_from_model_descriptor(descriptor);
          if (auto* failure = std::get_if<rt::Error>(&native))
            throw std::invalid_argument("SimaArgMax input: " + failure->message);
          in = std::get<rt::TensorSpec>(std::move(native));
        }
      }
      if (const auto canonical = pipeline_internal::compiled_kernel_output_tensor(
              input.immediate_upstream, source.backend_output_index);
          canonical && si::kernel_tensor_requires_detessellation(*canonical))
        in = *canonical;
      // Dense execution may publish a legal singleton-axis projection; its
      // authored logical shape/strides stay authoritative. Only a non-dense
      // physical layout needs the checked producer specification above.
      if (!(in.shape == source.shape) || in.dtype != kernel_dtype(source.dtype))
        throw std::invalid_argument(
            "SimaArgMax logical view disagrees with producer tensor storage");
      auto shape = source.shape;
      if (opt_.keepdims)
        shape[axis] = 1;
      else
        shape.erase(shape.begin() + axis);
      rt::TensorSpec out{shape, rt::DType::Int32};
      const std::array parameters{
          ::simaai::scheduler::client::Parameter{"axis", opt_.axis},
          ::simaai::scheduler::client::Parameter{"keepdims", opt_.keepdims}};
      auto checked =
          scheduler::check_kernel({"argmax", rt::Hardware::A65, {&in, 1}, {&out, 1}, parameters});
      if (auto* failure = std::get_if<rt::Error>(&checked))
        throw std::invalid_argument("SimaArgMax: " + std::string(failure->field) + ": " +
                                    failure->message);
      std::uint64_t bytes = sizeof(std::int32_t);
      for (const auto dimension : shape) {
        if (dimension <= 0 || static_cast<std::uint64_t>(dimension) >
                                  std::numeric_limits<std::uint64_t>::max() / bytes)
          throw std::invalid_argument("SimaArgMax output byte size overflows");
        bytes *= static_cast<std::uint64_t>(dimension);
      }
      auto inspected = scheduler::inspect_kernel(std::get<rt::KernelExecutable>(checked));
      if (const auto* failure = std::get_if<rt::Error>(&inspected))
        throw std::invalid_argument("SimaArgMax: " + failure->message);
      const auto contract = std::get<rt::KernelInspection>(std::move(inspected));
      if (contract.output_requirements().front().storage_bytes != bytes)
        throw std::invalid_argument("SimaArgMax output projection differs from checked storage");
      std::string binding_error;
      if (!specs::append_direct_kernel_input(&runtime, source, *physical, index,
                                             contract.input_requirements().front().storage_bytes,
                                             ingress.has_value(), &binding_error))
        throw std::invalid_argument("SimaArgMax input: " + binding_error);
      const auto name = "argmax_" + std::to_string(index);
      runtime.physical_outputs.push_back(specs::build_physical_buffer_static_spec(
          static_cast<int>(index), static_cast<int>(index), bytes, si::DeviceKind::Cpu, name));
      runtime.logical_outputs.push_back(specs::build_logical_output_static_spec(
          static_cast<int>(index), static_cast<int>(index), static_cast<int>(index),
          static_cast<int>(index), static_cast<int>(index), shape, "INT32",
          opt_.keepdims ? source.layout : "", name, name, name, 0, bytes));
      runtime.output_order.push_back(specs::build_output_route_static_spec(
          static_cast<int>(index), static_cast<int>(index), static_cast<int>(index), name, name));
      recipe->members.push_back(
          {std::get<rt::KernelExecutable>(std::move(checked)), {{index, 0}}, {{index, 0}}, {}});
    }
    compiled.recipe = std::move(recipe);
    output->node_kind = kind();
    output->plugin_kind = "kernel";
    output->element_name = element_names(input.node_index).front();
    output->logical_stage_id = output->element_name;
    output->definition = contract_definition();
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
} // namespace simaai::neat

namespace simaai::neat::nodes {
std::shared_ptr<simaai::neat::Node> SimaArgMax(SimaArgMaxOptions opt) {
  return std::make_shared<simaai::neat::SimaArgMax>(std::move(opt));
}
} // namespace simaai::neat::nodes
