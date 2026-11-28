/**
 * @file
 * @ingroup nodes_sima
 * @brief Bilinear sampling of a feature map with a dynamic coordinate grid on A65.
 */
#pragma once

#include "builder/Node.h"
#include "builder/NodeContractProvider.h"
#include "builder/OutputSpec.h"

#include <memory>
#include <string>
#include <vector>

namespace simaai::neat {

/// Padding outside the feature map. Reflection padding is unsupported.
enum class SimaGridSample2DPadding { Zeros, Border };

/**
 * @brief Options for A65 NEON GridSample2D.
 *
 * An input Sample contains two dense tensors, in order: features `[N,H,W,C]`
 * (FP32 or BF16), and grid `[N,Ho,Wo,2]` (FP32). Grid components are normalized
 * `(x,y)` coordinates. The output Sample contains `sampled` `[N,Ho,Wo,C]`, with
 * the feature dtype and NHWC layout. Dimensions must be positive. NCHW, tiled
 * storage and quantized values require explicit conversion before this Node.
 *
 * Grid values may change on every submission without rebuilding the graph;
 * tensor shapes and dtypes remain fixed for a prepared Run. Computation uses
 * FP32 coordinates, weights and accumulation, with one final BF16 rounding.
 *
 * @ingroup nodes_sima
 */
struct SimaGridSample2DOptions {
  SimaGridSample2DPadding padding = SimaGridSample2DPadding::Zeros;
  /// False: half-pixel centers. True: normalized -1/+1 name corner pixel centers.
  bool align_corners = false;
};

/// Dynamic-grid bilinear feature sampling on A65; no target fallback.
class SimaGridSample2D final : public Node, public OutputSpecProvider, public NodeContractProvider {
public:
  explicit SimaGridSample2D(SimaGridSample2DOptions opt = {});
  std::string kind() const override {
    return "SimaGridSample2D";
  }
  NodeCapsBehavior caps_behavior() const override {
    return NodeCapsBehavior::Static;
  }
  std::string backend_fragment(int node_index) const override;
  std::vector<std::string> element_names(int node_index) const override;
  OutputSpec output_spec(const OutputSpec& input) const override;
  NodeContractDefinition contract_definition() const override;
  bool compile_node_contract(const ContractCompileInput&, CompiledNodeContract*,
                             std::string* error) const override;
  const SimaGridSample2DOptions& options() const {
    return opt_;
  }

private:
  SimaGridSample2DOptions opt_;
};

} // namespace simaai::neat

namespace simaai::neat::nodes {
/// Construct an A65 GridSample2D Node.
std::shared_ptr<simaai::neat::Node> SimaGridSample2D(SimaGridSample2DOptions opt = {});
} // namespace simaai::neat::nodes
