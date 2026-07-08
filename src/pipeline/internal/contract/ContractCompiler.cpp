#include "pipeline/internal/contract/ContractCompiler.h"

#include "builder/CompiledChildStageProvider.h"
#include "builder/NodeContractProvider.h"
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
    stage_input.node_index =
        use_node_indices ? input.node_indices[node_index] : static_cast<int>(node_index);
    stage_input.immediate_upstream = immediate_upstream;

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
      compiled.fully_renderable = compiled.fully_renderable && stage.renderable;
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
      compiled.stages.push_back(std::move(stage));
      continue;
    }

    compiled.fully_renderable = compiled.fully_renderable && stage.renderable;
    compiled.stages.push_back(std::move(stage));
    immediate_upstream = last_effective_child_stage(&compiled.stages.back());
  }

  return compiled;
}

} // namespace simaai::neat
