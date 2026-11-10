/**
 * @file
 * @ingroup nodes_sima
 * @brief `SimaCenterHeadDecode` Node — mmdetection3d CenterHead outputs to 3D boxes.
 *
 * Runs the kernels-library `centerhead_decode` operation on A65: sigmoid scores, the top
 * candidates per class and per head task, box regression and NMS per head task.
 */
#pragma once

#include "builder/Node.h"
#include "builder/NodeContractProvider.h"
#include "builder/OutputSpec.h"
#include "nodes/sima/SimaKernelTypes.h"

#include <array>
#include <memory>
#include <string>
#include <vector>

namespace simaai::neat {

/**
 * @brief NMS of one `SimaCenterHeadDecode` head task (mmdetection3d `nms_type`).
 *
 * @ingroup nodes_sima
 */
enum class SimaCenterHeadNms {
  Unspecified, ///< Rejected: each head task states its NMS.
  Circle,      ///< Compares a squared centre distance with `nms_radius`.
  Rotate,      ///< Compares the bird's-eye-view IoU of rotated boxes with `nms_iou_threshold`.
};

/**
 * @brief One head task: `classes` consecutive heatmap channels and their NMS.
 *
 * @ingroup nodes_sima
 */
struct SimaCenterHeadTask {
  int classes = 0;                                        ///< Heatmap channels of this head task.
  SimaCenterHeadNms nms = SimaCenterHeadNms::Unspecified; ///< NMS of this head task.
  float nms_radius = 0.0f;        ///< Circle: mmdetection3d `min_radius`; rotate: stays 0.
  float nms_iou_threshold = 0.0f; ///< Rotate: mmdetection3d `nms_thr`; circle: stays 0.
};

/**
 * @brief Construction options for a `SimaCenterHeadDecode` Node.
 *
 * The input Sample holds the head maps on two to six ports, as `heatmap_input` to `vel_input`
 * place them. Every port is an FP32 or BF16 channels-last map `[H,W,C]` or `[1,H,W,C]` with the
 * same H and W: dense, or BF16 with gaps after cells and rows, as the MLA of an SDK 3.0 package
 * writes it after `model.inference()`. The heatmap has a port of its own, with one channel per
 * class. Every other port holds box heads: reg (2 channels: x and y offsets), height (1), dim (3:
 * the logarithms of dx, dy, dz), rot (2: sine, cosine) and, with `velocity`, vel (2).
 *
 * The output Sample holds, with `K = tasks.size() * post_max_per_task`: FP32 `boxes` `[K,9]` (x,
 * y, z of the bottom centre, dx, dy, dz, yaw, vx, vy) or `[K,7]` without `velocity`, FP32
 * `scores` `[K]`, INT32 `labels` `[K]` (heatmap channels) and INT32 `count` `[1]`. Rows from
 * `count` on are zero.
 *
 * @ingroup nodes_sima
 */
struct SimaCenterHeadDecodeOptions {
  /// Head tasks, in heatmap channel order: at most 256, of at most 4096 classes each.
  std::vector<SimaCenterHeadTask> tasks;
  /// Voxels per head cell.
  int out_size_factor = 0;
  /// Metres per voxel in x and y; positive.
  std::array<float, 2> voxel_size{};
  /// Point-cloud range minimum x and y, in metres. Required: it places every box.
  std::array<float, 2> range_min{};
  /// Kept scores are above it.
  float score_threshold = 0.0f;
  /// Candidates per class and per head task (mmdetection3d `max_num`).
  int top_k = 0;
  /// Boxes kept per head task after NMS.
  int post_max_per_task = 0;
  /// x, y, z minimum, then maximum, in metres; inclusive. Must span x and y.
  std::array<float, 6> post_center_range{};
  /// The vel head is placed and rows carry vx, vy.
  bool velocity = true;
  /// Input port of each head. Heads sharing a port are stored head task after head task, each
  /// head task's heads in the order reg, height, dim, rot, vel. The defaults are three maps: the
  /// heatmap; reg, height, dim and vel; rot.
  int heatmap_input = 0;
  int reg_input = 1;    ///< See `heatmap_input`.
  int height_input = 1; ///< See `heatmap_input`.
  int dim_input = 1;    ///< See `heatmap_input`.
  int rot_input = 2;    ///< See `heatmap_input`.
  int vel_input = 1;    ///< See `heatmap_input`; only used with `velocity`.
  /// Empty, or one positive factor per heatmap channel: rotate NMS multiplies a box's dx and dy by
  /// its class's factor (mmdetection3d `nms_rescale_factor`). Classes of circle head tasks take 1.
  std::vector<float> nms_size_scale;
  /// Execution target; no fallback. The operation has an A65 implementation only.
  SimaKernelTarget target = SimaKernelTarget::A65;
};

/**
 * @brief Kernel Node that decodes CenterHead outputs into 3D boxes.
 *
 * @ingroup nodes_sima
 */
class SimaCenterHeadDecode final : public Node,
                                   public OutputSpecProvider,
                                   public NodeContractProvider {
public:
  /// Construct with `SimaCenterHeadDecodeOptions`; throws `std::invalid_argument` on invalid
  /// options.
  explicit SimaCenterHeadDecode(SimaCenterHeadDecodeOptions opt);

  /// Type label for this Node kind.
  std::string kind() const override {
    return "SimaCenterHeadDecode";
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
  const SimaCenterHeadDecodeOptions& options() const {
    return opt_;
  }

private:
  SimaCenterHeadDecodeOptions opt_;
};

} // namespace simaai::neat

namespace simaai::neat::nodes {
/// Convenience factory for a `SimaCenterHeadDecode` Node.
std::shared_ptr<simaai::neat::Node> SimaCenterHeadDecode(SimaCenterHeadDecodeOptions opt);
} // namespace simaai::neat::nodes
