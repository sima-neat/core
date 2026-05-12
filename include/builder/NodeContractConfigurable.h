/**
 * @file
 * @ingroup builder
 * @brief Mixin for Nodes that accept a fully compiled node-level contract.
 *
 * After the Builder has compiled a Node's contract definition (resolving
 * shapes, dtypes, segment names, etc.), Nodes implementing
 * `NodeContractConfigurable` receive that compiled contract and use it to
 * specialize their backend (gst fragment, plugin JSON, internal state).
 */
#pragma once

#include <string>

namespace simaai::neat {

struct CompiledNodeContract;

/**
 * @brief Mixin interface implemented by Nodes that accept a compiled contract.
 *
 * Distinct from `NodeContractProvider`: the Provider exposes the contract
 * *definition* (what fields exist, what their sources are); the Configurable
 * receives the *compiled* contract with concrete values resolved.
 *
 * @ingroup builder
 * @see NodeContractProvider
 * @see NodeContractDefinition
 */
class NodeContractConfigurable {
public:
  virtual ~NodeContractConfigurable() = default;

  /**
   * @brief Apply a compiled contract to this Node.
   *
   * @param contract Compiled contract with concrete shapes/dtypes/etc.
   * @param err On failure, populated with a human-readable diagnostic.
   */
  virtual void apply_compiled_contract(const CompiledNodeContract& contract, std::string* err) = 0;
};

} // namespace simaai::neat
