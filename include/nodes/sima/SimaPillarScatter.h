/**
 * @file
 * @ingroup nodes_sima
 * @brief `SimaPillarScatter` Node — pillar features to the dense PointPillars BEV canvas.
 *
 * Runs the kernels-library `pillar_scatter` operation on A65 or EV74. It fills the
 * canvas with the consumer's zero point and copies each pillar's feature bytes into its
 * cell, without requantization.
 */
#pragma once

#include "builder/Node.h"
#include "builder/NodeContractProvider.h"
#include "builder/OutputSpec.h"
#include "nodes/sima/SimaKernelTypes.h"

#include <memory>
#include <string>
#include <vector>

namespace simaai::neat {

/**
 * @brief Construction options for a `SimaPillarScatter` Node.
 *
 * The input Sample holds three tensors, in order: INT8 or BF16 `features` `[M,C]`, INT32
 * `coordinates` `[M,2]` holding each pillar's (row, column), and INT32 `count` `[1]`.
 * `features` and `coordinates` may add singleton axes before their last axis; for
 * example `[1,M,1,C]` from a pillar-feature MLA graph. `C` must be a multiple of 16.
 *
 * The output `canvas` `[1,rows,columns,C]` has the features' element type and is stored
 * as compact full-channel MLA cell tiles. An INT8 canvas is byte-identical to dense
 * NHWC. A BF16 canvas uses the MLA-tiled BF16 payload: each cell stores the low bytes
 * of all channels followed by their high bytes. `semantic.tess` retains its checked
 * tile shape, axes, format and storage extent. Dense copying and DLPack export reject
 * this layout; use `copy_payload_bytes()` for stored bytes, or a layout-aware consumer
 * such as SimaArgMax. Cloning/Owned output preserves the layout without conversion.
 * Feature values are copied unchanged, and an INT8 canvas keeps the features' `semantic.quant`.
 *
 * @ingroup nodes_sima
 */
struct SimaPillarScatterOptions {
  int rows = 0;       ///< Canvas height `H`.
  int columns = 0;    ///< Canvas width `W`.
  int zero_point = 0; ///< INT8 code stored in every unoccupied cell; 0 (+0.0) for BF16.
  /// Execution target; no fallback.
  SimaKernelTarget target = SimaKernelTarget::A65;
};

/**
 * @brief Kernel Node that scatters pillar features onto a dense BEV canvas.
 *
 * @ingroup nodes_sima
 */
class SimaPillarScatter final : public Node,
                                public OutputSpecProvider,
                                public NodeContractProvider {
public:
  /// Construct with `SimaPillarScatterOptions`; throws `std::invalid_argument` on an invalid
  /// canvas.
  explicit SimaPillarScatter(SimaPillarScatterOptions opt);

  /// Type label for this Node kind.
  std::string kind() const override {
    return "SimaPillarScatter";
  }
  /// Whether the Node negotiates static or dynamic caps.
  NodeCapsBehavior caps_behavior() const override {
    return NodeCapsBehavior::Static;
  }
  /// GStreamer fragment this Node emits.
  std::string backend_fragment(int node_index) const override;
  /// Deterministic element names this Node will create.
  std::vector<std::string> element_names(int node_index) const override;
  /// Negotiated downstream caps produced by this Node.
  OutputSpec output_spec(const OutputSpec& input) const override;

  /// Structural contract definition for this Node.
  NodeContractDefinition contract_definition() const override;
  /// Prepare the checked kernel request from the upstream tensor contract.
  bool compile_node_contract(const ContractCompileInput& input, CompiledNodeContract* out,
                             std::string* error) const override;

  /// Inspect the Node's options.
  const SimaPillarScatterOptions& options() const {
    return opt_;
  }

private:
  SimaPillarScatterOptions opt_;
};

} // namespace simaai::neat

namespace simaai::neat::nodes {
/// Convenience factory for a `SimaPillarScatter` Node.
std::shared_ptr<simaai::neat::Node> SimaPillarScatter(SimaPillarScatterOptions opt);
} // namespace simaai::neat::nodes
