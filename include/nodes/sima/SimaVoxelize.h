/**
 * @file
 * @ingroup nodes_sima
 * @brief `SimaVoxelize` Node — LiDAR points to the quantized PointPillars pillar tensor.
 *
 * Runs the kernels-library `voxelize` operation on A65 or EV74. It assigns each point to
 * a pillar, decorates it with cluster- and pillar-relative offsets, and quantizes the
 * result for the MLA segment that consumes it.
 */
#pragma once

#include "builder/Node.h"
#include "builder/NodeContractProvider.h"
#include "builder/OutputSpec.h"
#include "nodes/sima/SimaKernelTypes.h"
#include "pipeline/TensorTypes.h"

#include <array>
#include <memory>
#include <string>
#include <vector>

namespace simaai::neat {

/**
 * @brief Construction options for a `SimaVoxelize` Node.
 *
 * The input Sample holds two tensors, in order: FP32 `points` `[N,4]` (x, y, z and the
 * sweep time offset) and INT32 `point_count` `[1]`. Only the leading `point_count`
 * points are read. The output Sample holds `decorated` `[1,M,T,10]` (INT8 or BF16, per
 * `output_dtype`), INT32 `coordinates` `[M,2]` (row, column) and INT32 `count` `[1]`.
 *
 * A BF16 `decorated` tensor is not quantized: each binary32 channel is rounded to
 * nearest-even BF16, NaN stores the canonical `0x7FC0`, and `quantization` must stay
 * the identity (multiplier 1, zero point 0, nearest-even).
 *
 * @ingroup nodes_sima
 */
struct SimaVoxelizeOptions {
  /// Metres per cell in x, y and z. z must span the whole range: one vertical cell.
  std::array<float, 3> voxel_size{};
  /// `x_lo, y_lo, z_lo, x_hi, y_hi, z_hi` in metres. Points outside are dropped.
  std::array<float, 6> point_cloud_range{};
  /// Quantization of the decorated FP32 features on store (INT8 output only).
  SimaKernelQuantization quantization;
  /// Element type of `decorated`: `TensorDType::Int8` or `TensorDType::BFloat16`.
  TensorDType output_dtype = TensorDType::Int8;
  /// Pillar capacity `M`. Pillars beyond it are dropped.
  int max_pillars = 0;
  /// Points-per-pillar capacity `T`. Points beyond it are dropped.
  int max_points_per_pillar = 0;
  /// Execution target; no fallback.
  SimaKernelTarget target = SimaKernelTarget::A65;
};

/**
 * @brief Kernel Node that voxelizes, decorates and quantizes one LiDAR sweep.
 *
 * @ingroup nodes_sima
 */
class SimaVoxelize final : public Node, public OutputSpecProvider, public NodeContractProvider {
public:
  /// Construct with `SimaVoxelizeOptions`; throws `std::invalid_argument` on invalid capacities.
  explicit SimaVoxelize(SimaVoxelizeOptions opt);

  /// Type label for this Node kind.
  std::string kind() const override {
    return "SimaVoxelize";
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
  const SimaVoxelizeOptions& options() const {
    return opt_;
  }

private:
  SimaVoxelizeOptions opt_;
};

} // namespace simaai::neat

namespace simaai::neat::nodes {
/// Convenience factory for a `SimaVoxelize` Node.
std::shared_ptr<simaai::neat::Node> SimaVoxelize(SimaVoxelizeOptions opt);
} // namespace simaai::neat::nodes
