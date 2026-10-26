#include "pipeline/internal/contract/ContractCompiler.h"

#include "builder/CompiledChildStageProvider.h"
#include "model/internal/ModelPack.h"
#include "pipeline/internal/sima/StageMemoryPolicy.h"
#include "builder/NodeContractProvider.h"
#include "builder/OutputSpec.h"

#include <algorithm>
#include <exception>
#include <stdexcept>
#include "nodes/io/Input.h"
#include "nodes/common/Output.h"
#include "pipeline/internal/contract/CompiledNodeContractQuery.h"

namespace simaai::neat {
namespace {

bool node_kind_requires_contract_provider(const std::string& kind) {
  return kind == "Preproc" || kind == "Quant" || kind == "Tess" || kind == "QuantTess" ||
         kind == "Dequant" || kind == "Detess" || kind == "DetessDequant" || kind == "Cast" ||
         kind == "SimaBoxDecode" || kind == "ModelFragment";
}

std::string default_logical_stage_id(const Node& node, const std::string& element_name) {
  const std::string user_label = node.user_label();
  return user_label.empty() ? element_name : user_label;
}

// Only the new image publication path needs strict temporal authority. The
// public best-effort OutputSpec walk deliberately keeps its legacy behavior.
bool project_image_temporal_contract(std::span<const std::shared_ptr<Node>> upstream,
                                     const ContractCompileInput& input, CompiledNodeContract* stage,
                                     std::string* err) {
  if (!stage->processcvu || !stage->processcvu->payload.image_transform)
    return true;
  auto boundary = input.ingress.ingress_spec.value_or(OutputSpec{});
  try {
    for (const auto& node : upstream) {
      if (!node)
        continue;
      if (const auto* provider = dynamic_cast<const OutputSpecProvider*>(node.get())) {
        boundary = provider->output_spec(boundary);
      } else {
        // Static/Dynamic caps behavior is not a temporal preservation promise.
        // Preserve spatial facts, but do not carry a rate across an opaque node.
        boundary.fps_num = 0;
        boundary.fps_den = 1;
      }
    }
  } catch (const std::exception& error) {
    if (err)
      *err = std::string("image framerate projection: ") + error.what();
    return false;
  }
  if (boundary.fps_num > 0 && boundary.fps_den <= 0) {
    if (err)
      *err = "image framerate: a known rate requires a positive denominator";
    return false;
  }
  auto& payload = stage->processcvu->payload;
  payload.image_fps_num = boundary.fps_num > 0 ? boundary.fps_num : 0;
  payload.image_fps_den = boundary.fps_num > 0 ? boundary.fps_den : 1;
  return true;
}

bool compile_child_stage_contracts(const Node& node, const NodeContractDefinition& definition,
                                   const CompiledChildStageProvider& provider,
                                   const ContractCompileInput& input, CompiledNodeContract* out,
                                   std::string* err) {
  if (!out) {
    if (err) {
      *err = node.kind() + " contract compile: output is null";
    }
    return false;
  }
  out->node_kind = node.kind();
  out->plugin_kind = definition.plugin_kind;
  const auto names = node.element_names(input.node_index);
  out->element_name = names.empty() ? std::string() : names.front();
  out->logical_stage_id = default_logical_stage_id(node, out->element_name);
  out->definition = definition;
  if (!provider.compile_child_stage_contracts(&out->child_stages, err)) {
    return false;
  }
  out->renderable = true;
  return true;
}

bool compile_known_node_contract(const std::shared_ptr<Node>& node,
                                 const NodeContractDefinition& definition,
                                 const ContractCompileInput& input, CompiledNodeContract* out,
                                 std::string* err, bool* handled) {
  if (const auto* child_stage_provider =
          dynamic_cast<const CompiledChildStageProvider*>(node.get())) {
    if (handled) {
      *handled = true;
    }
    return compile_child_stage_contracts(*node, definition, *child_stage_provider, input, out, err);
  }
  if (handled) {
    *handled = false;
  }
  return false;
}

// Standalone Cast may initially coalesce logical tensors into one carrier.
// An MLA's ordered input contract can require separate, independently aligned
// ports. Finish that semantic port contract before checked recipe preparation;
// storage packing and rendering never choose this representation.
void reconcile_standalone_cast_mla_ports(std::vector<CompiledNodeContract>& contracts) {
  std::vector<CompiledNodeContract*> leaves;
  const auto collect = [&](const auto& self, CompiledNodeContract& stage) -> void {
    if (!stage.child_stages.empty()) {
      for (auto& child : stage.child_stages)
        self(self, child);
    } else if (stage.kernel || stage.processcvu || stage.processmla || stage.boxdecode ||
               stage.dequant || stage.transport) {
      leaves.push_back(&stage);
    }
  };
  for (auto& stage : contracts)
    collect(collect, stage);
  for (std::size_t i = 1; i < leaves.size(); ++i) {
    auto& producer = *leaves[i - 1];
    const auto& consumer = *leaves[i];
    if (producer.node_kind != "Cast" || !producer.transport || !consumer.processmla)
      continue;
    auto& output = producer.transport->runtime_contract;
    const auto& input = consumer.processmla->runtime_contract;
    if (!input.consumer_keeps_distinct_physical_inputs || input.physical_inputs.size() < 2 ||
        output.logical_outputs.size() != input.physical_inputs.size())
      continue;
    std::vector<std::size_t> logical_for_port;
    for (const auto& port : input.physical_inputs) {
      const auto logical = std::find_if(
          output.logical_outputs.begin(), output.logical_outputs.end(), [&](const auto& tensor) {
            return !port.segment_name.empty() && (tensor.segment_name == port.segment_name ||
                                                  tensor.backend_name == port.segment_name ||
                                                  tensor.logical_name == port.segment_name);
          });
      if (logical == output.logical_outputs.end() || !port.size_bytes ||
          (logical->size_bytes && logical->size_bytes != port.size_bytes)) {
        logical_for_port.clear();
        break;
      }
      const auto index = static_cast<std::size_t>(logical - output.logical_outputs.begin());
      if (std::find(logical_for_port.begin(), logical_for_port.end(), index) !=
          logical_for_port.end()) {
        logical_for_port.clear();
        break;
      }
      logical_for_port.push_back(index);
    }
    if (logical_for_port.size() != input.physical_inputs.size())
      continue;
    const auto device = output.physical_outputs.empty()
                            ? pipeline_internal::sima::DeviceKind::Cpu
                            : output.physical_outputs.front().device_kind;
    output.physical_outputs.clear();
    for (std::size_t port = 0; port < logical_for_port.size(); ++port) {
      const auto& input_port = input.physical_inputs[port];
      pipeline_internal::sima::PhysicalBufferStaticSpec physical;
      physical.physical_index = physical.allocator_index = static_cast<int>(port);
      physical.source_physical_index = static_cast<int>(port);
      physical.size_bytes = input_port.size_bytes;
      physical.required_alignment_bytes = input_port.required_alignment_bytes;
      physical.device_kind = device;
      physical.segment_name = input_port.segment_name;
      output.physical_outputs.push_back(std::move(physical));
      auto& logical = output.logical_outputs[logical_for_port[port]];
      logical.physical_index = static_cast<int>(port);
      logical.byte_offset = 0;
      logical.segment_name = input_port.segment_name;
    }
    for (auto& route : output.output_order) {
      const auto logical = std::find_if(
          output.logical_outputs.begin(), output.logical_outputs.end(),
          [&](const auto& tensor) { return tensor.logical_index == route.logical_output_index; });
      if (logical != output.logical_outputs.end())
        route.segment_name = logical->segment_name;
    }
    output.consumer_keeps_distinct_physical_inputs = true;
  }
}

} // namespace

CompiledPipelineContracts
compile_node_contracts(const std::vector<std::shared_ptr<Node>>& nodes,
                       const ContractCompileInput& input,
                       pipeline_internal::sima::ManifestBuildDiagnostics* diagnostics) {
  CompiledPipelineContracts compiled;
  compiled.fully_renderable = true;
  const CompiledNodeContract* immediate_upstream = nullptr;
  const bool use_node_indices = !input.node_indices.empty();
  if (use_node_indices && input.node_indices.size() != nodes.size()) {
    compiled.fully_renderable = false;
    if (diagnostics) {
      diagnostics->errors.push_back("contract compiler: node_indices size does not match nodes");
    }
    return compiled;
  }

  const auto model_fragments = internal::make_model_fragment_semantic_context(nodes);
  std::vector<std::size_t> compiled_node_indices(nodes.size(), nodes.size());
  compiled.stages.reserve(nodes.size());

  for (std::size_t node_index = 0; node_index < nodes.size(); ++node_index) {
    const auto& node = nodes[node_index];
    if (!node) {
      compiled.fully_renderable = false;
      if (diagnostics) {
        diagnostics->errors.push_back("contract compiler: null node");
      }
      continue;
    }

    if (node->kind() == "Input" || node->kind() == "Output") {
      continue;
    }

    CompiledNodeContract stage;
    stage.node_kind = node->kind();
    ContractCompileInput stage_input = input;
    stage_input.model_fragments = model_fragments;
    stage_input.node_occurrence_index = node_index;
    stage_input.node_index =
        use_node_indices ? input.node_indices[node_index] : static_cast<int>(node_index);
    stage_input.immediate_upstream = immediate_upstream;
    stage_input.upstream_preprocess =
        last_preprocess_contract(compiled.stages, input.upstream_preprocess);

    const auto* provider = dynamic_cast<const NodeContractProvider*>(node.get());
    if (!provider) {
      if (node_kind_requires_contract_provider(node->kind())) {
        compiled.fully_renderable = false;
        if (diagnostics) {
          diagnostics->warnings.push_back(
              "contract compiler: no provider for semantic node kind '" + node->kind() + "'");
        }
      }
      continue;
    }

    stage.definition = provider->contract_definition();

    // Preserve the provider's failure reason. Historically the fallback path
    // below reused the same `err` string and then unconditionally overwrote it
    // with a generic "no direct compiler path registered" message, masking the
    // actual reason the provider rejected the node (e.g. "SimaBoxDecode:
    // inferred standalone contract requires …"). Separate per-stage scratch
    // strings so the diagnostic that lands in `diagnostics->errors` is the
    // most specific one we have.
    std::string provider_err;
    if (provider->compile_node_contract(stage_input, &stage, &provider_err)) {
      if (!project_image_temporal_contract(
              std::span<const std::shared_ptr<Node>>(nodes.data(), node_index), input, &stage,
              &provider_err)) {
        stage.renderable = false;
        compiled.fully_renderable = false;
        if (diagnostics) {
          diagnostics->errors.push_back("contract compiler: " + provider_err);
        }
        compiled_node_indices[node_index] = compiled.stages.size();
        compiled.stages.push_back(std::move(stage));
        continue;
      }
      compiled.fully_renderable = compiled.fully_renderable && stage.renderable;
      compiled_node_indices[node_index] = compiled.stages.size();
      compiled.stages.push_back(std::move(stage));
      immediate_upstream = last_effective_child_stage(&compiled.stages.back());
      continue;
    }

    bool handled_directly = false;
    std::string direct_err;
    const bool direct_ok = compile_known_node_contract(node, stage.definition, stage_input, &stage,
                                                       &direct_err, &handled_directly);
    bool compile_ok = direct_ok;
    std::string err;
    if (!handled_directly) {
      // No fallback compiler path applies. The provider already failed; surface
      // its message (it knows the specific contract / wiring requirement that
      // didn't hold) rather than the generic "no direct compiler" stub.
      err = !provider_err.empty()
                ? provider_err
                : "no direct compiler path registered for node kind '" + node->kind() + "'";
      compile_ok = false;
    } else if (!direct_ok) {
      // Fallback path tried but also rejected. Prefer its message if set, else
      // fall back to the provider's.
      err = !direct_err.empty() ? direct_err : provider_err;
      if (err.empty()) {
        err = "no direct compiler path registered for node kind '" + node->kind() + "'";
      }
    }
    if (!compile_ok) {
      stage.renderable = false;
      compiled.fully_renderable = false;
      if (diagnostics) {
        diagnostics->errors.push_back("contract compiler: node kind '" + node->kind() +
                                      "' failed: " + err);
      }
      compiled_node_indices[node_index] = compiled.stages.size();
      compiled.stages.push_back(std::move(stage));
      continue;
    }

    compiled.fully_renderable = compiled.fully_renderable && stage.renderable;
    compiled_node_indices[node_index] = compiled.stages.size();
    compiled.stages.push_back(std::move(stage));
    immediate_upstream = last_effective_child_stage(&compiled.stages.back());
  }

  try {
    reconcile_standalone_cast_mla_ports(compiled.stages);
    for (auto& stage : compiled.stages)
      pipeline_internal::sima::resolve_compiled_execution_targets(stage, input);
    compiled.model_storage_scopes =
        internal::make_model_storage_scopes(nodes, compiled.stages, compiled_node_indices);
  } catch (const std::exception& error) {
    compiled.fully_renderable = false;
    if (diagnostics)
      diagnostics->errors.push_back(std::string("contract compiler: ") + error.what());
  }

  return compiled;
}

} // namespace simaai::neat
