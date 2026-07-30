/**
 * @file
 * @brief Private, immutable specialization of pipeline Nodes from stable input specs.
 */
#pragma once

#include "builder/Node.h"
#include "builder/OutputSpec.h"

#include <memory>
#include <span>
#include <string>
#include <string_view>
#include <unordered_map>
#include <vector>

namespace simaai::neat::internal {

/**
 * Build-local capabilities discovered by the orchestration layer.
 *
 * Nodes consume named facts without depending on the subsystem (GStreamer,
 * hardware inventory, etc.) that discovered them.
 */
class InputSpecSpecializationContext {
public:
  void set_capability(std::string_view name, bool enabled) {
    capabilities_.insert_or_assign(std::string(name), enabled);
  }

  bool capability(std::string_view name) const {
    const auto it = capabilities_.find(std::string(name));
    return it != capabilities_.end() && it->second;
  }

private:
  std::unordered_map<std::string, bool> capabilities_;
};

/**
 * A private compiler hook for semantic Nodes whose backend fragment can be made
 * more precise once the upstream contract is known.
 *
 * Implementations return a new Node.  Keeping specialization immutable avoids
 * leaking build-local decisions into a public Graph that may be compiled again
 * with a different input contract.  The replacement must preserve the semantic
 * Node's output contract; only its backend topology may change.  This invariant
 * keeps already-propagated facts on later segment boundaries valid.
 */
class InputSpecSpecializer {
public:
  virtual ~InputSpecSpecializer() = default;

  virtual std::shared_ptr<Node>
  specialize_for_input(const OutputSpec& input,
                       const InputSpecSpecializationContext& context) const = 0;
};

struct SpecializedNodeSequence {
  std::vector<std::shared_ptr<Node>> nodes;
  OutputSpec output_spec;
};

/**
 * Clone only semantic Nodes that opt into input-spec specialization, preserving
 * the sequence length and sharing all immutable, non-specializing Nodes.
 */
SpecializedNodeSequence
specialize_nodes_for_input(std::span<const std::shared_ptr<Node>> nodes,
                           const OutputSpec& input = {},
                           const InputSpecSpecializationContext& context = {});

} // namespace simaai::neat::internal
