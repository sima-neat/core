#include "builder/internal/InputSpecSpecialization.h"

#include <stdexcept>

namespace simaai::neat::internal {

SpecializedNodeSequence specialize_nodes_for_input(std::span<const std::shared_ptr<Node>> nodes,
                                                   const OutputSpec& input,
                                                   const InputSpecSpecializationContext& context) {
  SpecializedNodeSequence result;
  result.nodes.reserve(nodes.size());
  result.output_spec = input;

  for (const auto& node : nodes) {
    if (!node) {
      throw std::invalid_argument("specialize_nodes_for_input: node is null");
    }

    std::shared_ptr<Node> selected = node;
    if (const auto* specializer = dynamic_cast<const InputSpecSpecializer*>(node.get())) {
      selected = specializer->specialize_for_input(result.output_spec, context);
      if (!selected) {
        throw std::runtime_error("specialize_nodes_for_input: '" + node->kind() +
                                 "' returned a null specialization");
      }
    }

    // A semantic Node may render a different number of backend elements, but
    // it always remains one Node for provenance, metrics, and serialization.
    result.nodes.push_back(selected);
    if (const auto* provider = dynamic_cast<const OutputSpecProvider*>(selected.get())) {
      result.output_spec = provider->output_spec(result.output_spec);
    }
  }

  return result;
}

} // namespace simaai::neat::internal
