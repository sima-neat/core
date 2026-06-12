/**
 * @file
 * @ingroup builder
 * @brief Mixin for Nodes that expose compiled child-stage contracts.
 *
 * A composite Node (e.g., a preprocess Node that internally chains resize +
 * normalize + quant + tess sub-stages) implements `CompiledChildStageProvider`
 * to surface the compiled contracts of those sub-stages. The Builder calls into
 * this during contract resolution so each child stage participates in
 * downstream contract negotiation as if it were a top-level Node.
 */
#pragma once

#include <string>
#include <vector>

namespace simaai::neat {

struct CompiledNodeContract;

/**
 * @brief Mixin interface implemented by Nodes that wrap multiple child stages.
 *
 * The framework calls `compile_child_stage_contracts()` after the parent Node
 * has resolved its own contract; the child contracts it returns are then
 * threaded through the rest of the contract pipeline. Concrete Nodes that wrap
 * a sub-pipeline (e.g., Preproc, multi-step transforms) implement this interface.
 *
 * @ingroup builder
 * @see NodeContractProvider
 * @see NodeContractDefinition
 */
class CompiledChildStageProvider {
public:
  virtual ~CompiledChildStageProvider() = default;

  /**
   * @brief Emit compiled contracts for this Node's internal sub-stages.
   *
   * @param out Output vector populated with one entry per child stage, in execution order.
   * @param err On failure, populated with a human-readable diagnostic.
   * @return True on success; false (and `*err` populated) if contracts could not be compiled.
   */
  virtual bool compile_child_stage_contracts(std::vector<CompiledNodeContract>* out,
                                             std::string* err) const = 0;
};

} // namespace simaai::neat
