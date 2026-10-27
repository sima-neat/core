/**
 * @file
 * @ingroup nodes_sima
 * @brief `SimaBEVPool` Node — lift-splat pooling of camera depth and context into a BEV grid.
 *
 * Runs the kernels-library `bev_pool` operation on A65 or EV74. For each BEV cell it sums
 * `depth x context` over the frustum points that a `SimaBEVPoolPlan` plan assigns to it, then
 * quantizes the sum for the next MLA segment.
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
 * @brief Byte order of a SimaBEVPool depth, context or BEV tensor.
 *
 * @ingroup nodes_sima
 */
enum class SimaBEVPoolStorage {
  /// Channels innermost. Inputs arrive as `[N,fH,fW,D]` and `[N,fH,fW,C]`, the camera
  /// graph's NHWC outputs. The output is published as logical `[1,nZ*C,nX,nY]` with
  /// channels-last byte strides, so its bytes are NHWC `[1,nX,nY,nZ*C]`.
  ChannelsLast,
  /// Logical order. Inputs arrive as `[N,D,fH,fW]` and `[N,C,fH,fW]`; the output is a
  /// dense `[1,nZ*C,nX,nY]` tensor.
  Contiguous,
};

/**
 * @brief Construction options for a `SimaBEVPool` Node.
 *
 * The input Sample holds three tensors, in order: `depth`, `context` (both INT8 or both
 * BF16, layout per `input_storage`) and the UINT8 `plan` `[P]` produced by
 * `SimaBEVPoolPlan` for the same frustum and grid. The output Sample holds one `bev` tensor.
 *
 * BF16 inputs need zero points of 0. Their sums are binary32, in the plan's point order.
 * A BF16 `bev` stores each sum rounded to nearest-even (NaN as `0x7FC0`) and needs the
 * identity quantization; an INT8 `bev` from BF16 inputs quantizes the sum with
 * `multiplier = 1 / output_scale`. INT8 inputs cannot produce a BF16 `bev`.
 *
 * @ingroup nodes_sima
 */
struct SimaBEVPoolOptions {
  std::array<int, 3> cells{}; ///< BEV cell counts `nX`, `nY`, `nZ`; must match the plan.
  int depth_zero_point = 0;   ///< INT8 zero point of `depth`; 0 for BF16.
  int context_zero_point = 0; ///< INT8 zero point of `context`; 0 for BF16.
  /// Layout of the bound plan: the producing SimaBEVPoolPlan's `format`.
  SimaBEVPoolPlanFormat plan_format = SimaBEVPoolPlanFormat::V1;
  /// Applied to each cell's binary32 sum for an INT8 `bev`. For INT8 inputs set
  /// `multiplier = depth_scale * context_scale / output_scale`; for BF16 inputs,
  /// `1 / output_scale`. A BF16 `bev` needs the identity.
  SimaKernelQuantization quantization;
  SimaBEVPoolStorage input_storage = SimaBEVPoolStorage::ChannelsLast;  ///< depth/context order.
  SimaBEVPoolStorage output_storage = SimaBEVPoolStorage::ChannelsLast; ///< bev byte order.
  TensorDType output_dtype = TensorDType::Int8;                         ///< bev element type.
  /// Execution target; no fallback.
  SimaKernelTarget target = SimaKernelTarget::A65;
};

/**
 * @brief Kernel Node that pools camera depth and context features into a BEV grid.
 *
 * @ingroup nodes_sima
 */
class SimaBEVPool final : public Node, public OutputSpecProvider, public NodeContractProvider {
public:
  /// Construct with `SimaBEVPoolOptions`; throws `std::invalid_argument` on an invalid grid.
  explicit SimaBEVPool(SimaBEVPoolOptions opt);

  /// Type label for this Node kind.
  std::string kind() const override {
    return "SimaBEVPool";
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
  const SimaBEVPoolOptions& options() const {
    return opt_;
  }

private:
  SimaBEVPoolOptions opt_;
};

} // namespace simaai::neat

namespace simaai::neat::nodes {
/// Convenience factory for a `SimaBEVPool` Node.
std::shared_ptr<simaai::neat::Node> SimaBEVPool(SimaBEVPoolOptions opt);
} // namespace simaai::neat::nodes
