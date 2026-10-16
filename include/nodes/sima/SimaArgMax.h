/**
 * @file
 * @ingroup nodes_sima
 * @brief `SimaArgMax` Node — postprocess argmax over a tensor (classification).
 *
 * Runs on A65 through the generic kernel runner. Reads a classification tensor (after dequant/cast) and emits the
 * top class index per sample. Place after `Detess`/`DetessDequant`/`DetessCast` at the
 * tail of a classification pipeline.
 */
#pragma once

#include "builder/Node.h"
#include "builder/NodeContractProvider.h"
#include "builder/OutputSpec.h"

#include <memory>
#include <string>
#include <vector>

namespace simaai::neat {

/**
 * @brief Construction options for a `SimaArgMax` Node.
 *
 * @ingroup nodes_sima
 */
struct SimaArgMaxOptions {
  std::string config_path;     ///< Retired JSON configuration; must be empty.
  int sima_allocator_type = 2; ///< Compatibility field: use 0 or 2; allocation belongs to Runtime.
  bool silent = true;          ///< Retained source-compatible diagnostic option.
  bool emit_signals = false;   ///< Retired transport option; must remain false.
  bool transmit = false; ///< Retired transport option; must remain false.
  int axis = -1; ///< Reduction axis; negative values count from the trailing axis.
  bool keepdims = false; ///< Retain the reduced axis with extent one.
};

/**
 * @brief A65 postprocess Node that emits the top-class index from a classification tensor.
 *
 * @ingroup nodes_sima
 */
class SimaArgMax final : public Node, public OutputSpecProvider, public NodeContractProvider {
public:
  /// Construct with optional `SimaArgMaxOptions`.
  explicit SimaArgMax(SimaArgMaxOptions opt = {});

  /// Type label for this Node kind.
  std::string kind() const override {
    return "SimaArgMax";
  }
  /// Whether the Node negotiates static or dynamic caps.
  NodeCapsBehavior caps_behavior() const override {
    return NodeCapsBehavior::Static;
  }
  /// Whether this Node is driven by a JSON config.
  bool has_config_json() const override;
  /// Legacy hook for node-local JSON wiring.
  bool wire_input_names(const std::vector<std::string>& upstream_names,
                        const std::string& tag) override;
  /// GStreamer fragment this Node emits.
  std::string backend_fragment(int node_index) const override;
  /// Deterministic element names this Node will create.
  std::vector<std::string> element_names(int node_index) const override;
  /// Negotiated downstream caps produced by this Node.
  OutputSpec output_spec(const OutputSpec& input) const override;

  NodeContractDefinition contract_definition() const override;
  bool compile_node_contract(const ContractCompileInput&, CompiledNodeContract*,
                             std::string* error) const override;

  /// Inspect the Node's options.
  const SimaArgMaxOptions& options() const {
    return opt_;
  }

private:
  SimaArgMaxOptions opt_;
};

} // namespace simaai::neat

namespace simaai::neat::nodes {
/// Convenience factory for a `SimaArgMax` Node with optional `SimaArgMaxOptions`.
std::shared_ptr<simaai::neat::Node> SimaArgMax(SimaArgMaxOptions opt = {});
} // namespace simaai::neat::nodes
