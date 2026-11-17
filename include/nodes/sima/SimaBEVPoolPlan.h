/**
 * @file
 * @ingroup nodes_sima
 * @brief `SimaBEVPoolPlan` Node — camera calibration to the SimaBEVPool plan.
 *
 * Runs the kernels-library `bev_pool_plan` operation on A65. It computes which camera
 * frustum points land in which BEV cell (BEVFusion's `get_geometry` and `bev_pool`
 * index) and writes the UINT8 plan tensor that `SimaBEVPool` reads.
 */
#pragma once

#include "builder/Node.h"
#include "builder/NodeContractProvider.h"
#include "builder/OutputSpec.h"
#include "nodes/sima/SimaKernelTypes.h"
#include "pipeline/TensorTypes.h"

#include <array>
#include <cstdint>
#include <memory>
#include <string>
#include <vector>

namespace simaai::neat {

/**
 * @brief Construction options for a `SimaBEVPoolPlan` Node.
 *
 * The input Sample holds four FP32 tensors, in order: `camera2lidar` `[N,4,4]`,
 * `camera_intrinsics` `[N,4,4]`, `img_aug_matrix` `[N,4,4]` and `lidar_aug_matrix`
 * `[4,4]`. The output Sample holds one UINT8 `plan` tensor `[P]`.
 *
 * Copy `xs`, `ys`, `ds`, `bx`, `dx` and `nx` from the checkpoint's view transform:
 * `xs = frustum[0,0,:,0]`, `ys = frustum[0,:,0,1]` and `ds = frustum[:,0,0,2]`.
 *
 * @ingroup nodes_sima
 */
struct SimaBEVPoolPlanOptions {
  std::vector<float> xs;     ///< `fW` image-column samples, in input pixels.
  std::vector<float> ys;     ///< `fH` image-row samples, in input pixels.
  std::vector<float> ds;     ///< `D` depth bins, in metres.
  std::array<float, 3> bx{}; ///< BEV grid cell centre of cell 0, per axis.
  std::array<float, 3> dx{}; ///< BEV cell size, per axis.
  std::array<int, 3> nx{};   ///< BEV cell counts `nX`, `nY`, `nZ`.
  SimaBEVPoolPlanFormat format = SimaBEVPoolPlanFormat::V1; ///< Plan layout.
  int context_channels = 0; ///< SimaBEVPool's `C`; required for `V2`.
  /// Element type of SimaBEVPool's depth and context: `Int8` or `BFloat16`. A `V2` plan
  /// budgets its caches in bytes, so BF16 halves them; `V1` does not depend on it.
  TensorDType input_dtype = TensorDType::Int8;
  int workers = 4;                      ///< `V2`: worker streams, `1..16`.
  int tile_h = 8;                       ///< `V2`: work tile height in cells.
  int tile_w = 8;                       ///< `V2`: work tile width in cells.
  bool cache_plan = false;              ///< Copy the last plan when the calibration repeats.
  std::int64_t plan_capacity_bytes = 0; ///< `P`; 0 selects the worst-case plan size.
  /// Execution target. The kernels library implements SimaBEVPoolPlan on A65 only.
  SimaKernelTarget target = SimaKernelTarget::A65;
};

/**
 * @brief Kernel Node that builds SimaBEVPool's plan from one frame's camera calibration.
 *
 * @ingroup nodes_sima
 */
class SimaBEVPoolPlan final : public Node, public OutputSpecProvider, public NodeContractProvider {
public:
  /// Construct with `SimaBEVPoolPlanOptions`; throws `std::invalid_argument` on an invalid grid.
  explicit SimaBEVPoolPlan(SimaBEVPoolPlanOptions opt);

  /// Type label for this Node kind.
  std::string kind() const override {
    return "SimaBEVPoolPlan";
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
  const SimaBEVPoolPlanOptions& options() const {
    return opt_;
  }

private:
  SimaBEVPoolPlanOptions opt_;
};

} // namespace simaai::neat

namespace simaai::neat::nodes {
/// Convenience factory for a `SimaBEVPoolPlan` Node.
std::shared_ptr<simaai::neat::Node> SimaBEVPoolPlan(SimaBEVPoolPlanOptions opt);
} // namespace simaai::neat::nodes
