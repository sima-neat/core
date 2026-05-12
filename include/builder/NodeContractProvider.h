/**
 * @file
 * @ingroup builder
 * @brief Mixin for Nodes that own and compile their own contract definition.
 *
 * A Node implementing `NodeContractProvider` exposes its full contract
 * definition (input/output ports, fields, override policy) and optionally
 * compiles it from a `ContractCompileInput` snapshot. The Builder uses these
 * hooks during contract negotiation; if a Node lacks an implementation of
 * `compile_node_contract()`, the Builder falls back to the central registry.
 */
#pragma once

#include "contracts/NodeContractDefinition.h"

#include <string>

namespace simaai::neat {

struct ContractCompileInput;
struct CompiledNodeContract;

/**
 * @brief Mixin interface implemented by Nodes that own a contract definition.
 *
 * Use this when the Node's contract depends on per-instance state (e.g., a
 * model file) rather than just its `kind()`. The default
 * `compile_node_contract()` implementation reports failure so callers know
 * to fall back to the registry.
 *
 * @ingroup builder
 * @see NodeContractDefinition
 * @see NodeContractConfigurable
 * @see ContractRegistry
 */
class NodeContractProvider {
public:
  virtual ~NodeContractProvider() = default;

  /// @brief Return this Node's contract definition (purely structural metadata).
  virtual NodeContractDefinition contract_definition() const = 0;

  /**
   * @brief Compile this Node's contract from a runtime input snapshot.
   *
   * Default implementation returns false with `*err` set, indicating the
   * Node does not own contract compilation; the Builder should use the
   * central `ContractRegistry` path instead.
   *
   * @param input Concrete inputs (upstream shapes, builder options, model facts).
   * @param out   Output compiled contract.
   * @param err   On failure, populated with a human-readable diagnostic.
   * @return True on successful compilation.
   */
  virtual bool compile_node_contract(const ContractCompileInput& input,
                                     CompiledNodeContract* out,
                                     std::string* err) const {
    (void)input;
    (void)out;
    if (err) {
      *err = "node-owned contract compilation is not available";
    }
    return false;
  }
};

} // namespace simaai::neat
