/**
 * @file
 * @ingroup nodes_sima
 * @brief `ResizeArgMax` Node — per-class scores to a resized semantic-segmentation class map.
 *
 * Runs the kernels-library `resize_argmax` operation on the EV74. It bilinearly resizes
 * channels-last class scores to the output size and returns the first class with the largest
 * score at each pixel: SegFormer's `interpolate(mode="bilinear")` followed by `argmax`, without
 * storing the resized scores. There is no A65 implementation and no fallback.
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
 * @brief Construction options for a `ResizeArgMax` Node.
 *
 * The input Sample holds one `scores` tensor, `[h,w,C]` or `[N,h,w,C]` with the classes on the
 * last axis: INT8, UINT8, BF16 or FP32, dense or strided. A padded MLA output, such as the
 * `[1,128,128,150]` view over 160 stored channels that a model's slice stage publishes, is read
 * in place. The output Sample holds one dense `class_map` tensor, `[H,W]` or `[N,H,W]`: UINT8
 * for at most 256 classes, otherwise INT32.
 *
 * INT8 and UINT8 scores are compared without dequantization, which selects the same classes as
 * the dequantized scores when one positive scale covers every class. Scores with per-axis
 * quantization, a non-positive scale, MLA-tiled storage or a CHW/NCHW layout are rejected when
 * the graph is built.
 *
 * @ingroup nodes_sima
 */
struct ResizeArgMaxOptions {
  int output_width = 0;  ///< Class-map width `W`; must be positive.
  int output_height = 0; ///< Class-map height `H`; must be positive.
  /// False (default): half-pixel centers, as PyTorch and Hugging Face use for SegFormer.
  /// True: corner-aligned source coordinates.
  bool align_corners = false;
};

/**
 * @brief Kernel Node that decodes per-class scores into a resized class map on the EV74.
 *
 * @ingroup nodes_sima
 */
class ResizeArgMax final : public Node, public OutputSpecProvider, public NodeContractProvider {
public:
  /// Construct with `ResizeArgMaxOptions`; throws `std::invalid_argument` on an empty size.
  explicit ResizeArgMax(ResizeArgMaxOptions opt);

  /// Type label for this Node kind.
  std::string kind() const override {
    return "ResizeArgMax";
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

  NodeContractDefinition contract_definition() const override;
  bool compile_node_contract(const ContractCompileInput&, CompiledNodeContract*,
                             std::string* error) const override;

  /// Inspect the Node's options.
  const ResizeArgMaxOptions& options() const {
    return opt_;
  }

private:
  ResizeArgMaxOptions opt_;
};

} // namespace simaai::neat

namespace simaai::neat::nodes {
/// Convenience factory for a `ResizeArgMax` Node.
std::shared_ptr<simaai::neat::Node> ResizeArgMax(ResizeArgMaxOptions opt);
} // namespace simaai::neat::nodes
