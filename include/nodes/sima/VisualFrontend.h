/**
 * @file
 * @ingroup nodes_sima
 * @brief EV74 visual-frontend processcvu Nodes (`FeatureHistogram`, `GriderFast`,
 *        `TrackDescriptor`, `TrackKLT`).
 */
#pragma once

#include "builder/Node.h"
#include "builder/NodeContractConfigurable.h"
#include "builder/NodeContractProvider.h"
#include "builder/OutputSpec.h"

#include <cstdint>
#include <memory>
#include <string>
#include <vector>

namespace simaai::neat {

/**
 * @brief Shared options for EV74 visual-frontend tensor Nodes.
 *
 * The public tensor shape is always logical and batch-aware: grayscale images
 * are `[batch_size, height, width]`.  The implementation may pack the batch for
 * the EV74 graph ABI, but callers should not pre-pack dimensions themselves.
 */
struct VisualFrontendCommonOptions {
  /// Input image width in pixels. Must be positive and within the graph-specific EV envelope.
  int width = 0;
  /// Input image height in pixels. Must be positive and within the graph-specific EV envelope.
  int height = 0;
  /// Number of packed grayscale images in one dispatch. Public tensor shape uses this as `N`.
  int batch_size = 1;
  /// EV graph debug level. Current native visual graphs accept values in `[0,2]`.
  int debug = 0;
  /// Optional processcvu queue/buffer override. `0` keeps the plugin/runtime default.
  int num_buffers = 0;
  /// Optional GStreamer/processcvu element name. Empty means Neat generates a stable name.
  std::string element_name;
};

/**
 * @brief 8-bit grayscale histogram Node options.
 *
 * Public ABI:
 * - input:  `input_name`, UInt8 `[batch_size,height,width]`
 * - output: `output_name`, Int32 `[batch_size,256]`
 */
struct FeatureHistogramOptions : public VisualFrontendCommonOptions {
  /// Input grayscale image tensor name.
  std::string input_name = "input_image";
  /// Output histogram tensor name.
  std::string output_name = "output_hist";

  /// Human-readable, non-throwing summary for logs, diagnostics, and Python `repr`.
  std::string summary() const;
};

/**
 * @brief Grid-distributed FAST feature detector options.
 *
 * `GriderFast` follows the OpenVINS-style `Grider_FAST` naming while using
 * normal Neat/PascalCase spelling.  Feature records are published as Int32
 * `[count, x0, y0, score0, x1, y1, score1, ...]` per batch item.
 *
 * Public ABI:
 * - input:  `input_name`, UInt8 `[batch_size,height,width]`
 * - output: `output_name`, Int32 `[batch_size,1 + max_features*3]`
 */
struct GriderFastOptions : public VisualFrontendCommonOptions {
  /// FAST detector threshold in `[0,255]`.
  int threshold = 30;
  /// Maximum features emitted per batch item.
  int max_features = 500;
  /// Horizontal grid cells for feature distribution.
  int grid_x = 8;
  /// Vertical grid cells for feature distribution.
  int grid_y = 6;
  /// Minimum pixel distance between accepted features. Must be non-negative.
  int min_px_dist = 10;
  /// Input grayscale image tensor name.
  std::string input_name = "input_image";
  /// Output feature-list tensor name.
  std::string output_name = "output_features";

  /// Human-readable, non-throwing summary for logs, diagnostics, and Python `repr`.
  std::string summary() const;
};

/**
 * @brief FAST + BRIEF-style descriptor frontend options.
 *
 * Public ABI:
 * - input:       `input_name`, UInt8 `[batch_size,height,width]`
 * - features:    `features_output_name`, Int32 `[batch_size,1 + max_features*3]`
 * - descriptors: `descriptors_output_name`, Int32 `[batch_size,max_features,8]`
 *
 * The current EV74 graph ABI fixes `descriptor_words == 8`; changing it is an
 * ABI change and is rejected before processcvu dispatch.
 */
struct TrackDescriptorOptions : public VisualFrontendCommonOptions {
  /// FAST detector threshold in `[0,255]`.
  int threshold = 30;
  /// Maximum features/descriptors emitted per batch item.
  int max_features = 500;
  /// Horizontal grid cells for feature distribution.
  int grid_x = 8;
  /// Vertical grid cells for feature distribution.
  int grid_y = 6;
  /// Minimum pixel distance between accepted features. Must be non-negative.
  int min_px_dist = 10;
  /// Descriptor words per feature. Must remain `8` for the current EV74 ABI.
  int descriptor_words = 8;
  /// Input grayscale image tensor name.
  std::string input_name = "input_image";
  /// Output feature-list tensor name.
  std::string features_output_name = "output_features";
  /// Output descriptor tensor name.
  std::string descriptors_output_name = "output_descriptors";

  /// Human-readable, non-throwing summary for logs, diagnostics, and Python `repr`.
  std::string summary() const;
};

/**
 * @brief Pyramidal Lucas-Kanade / KLT tracker options.
 *
 * Public ABI:
 * - inputs:
 *   - `prev_image_name`, UInt8 `[batch_size,height,width]`
 *   - `cur_image_name`, UInt8 `[batch_size,height,width]`
 *   - `input_points_name`, Int32 `[batch_size,num_points,2]`
 * - outputs when `detect_new_features == 0`:
 *   - `output_points_name`, Float32 `[batch_size,num_points,2]`
 *   - `output_status_name`, Int32 `[batch_size,num_points,1]`
 * - outputs when `detect_new_features == 1` additionally include:
 *   - `output_features_name`, Int32 `[batch_size,1 + max_features*3]`
 *
 * The EV ABI always allocates the internal detected-features output; Neat only
 * publishes it when `detect_new_features != 0`.
 */
struct TrackKLTOptions {
  /// Input image width in pixels.
  int width = 0;
  /// Input image height in pixels.
  int height = 0;
  /// Number of packed image/point sets in one dispatch. Public tensor shape uses this as `N`.
  int batch_size = 1;
  /// Number of input points tracked per batch item.
  int num_points = 0;
  /// Half-window radius for LK tracking. Must fit the EV74 graph envelope.
  int win_half = 10;
  /// Maximum LK solver iterations per pyramid level. Must be positive.
  int max_iters = 30;
  /// Maximum pyramid level. Must fit the EV74 graph envelope.
  int max_level = 3;
  /// When non-zero, publish detected replacement/new features as a third output.
  int detect_new_features = 0;
  /// FAST threshold used only by the detect-new-features path.
  int fast_threshold = 30;
  /// Maximum detected replacement/new features per batch item.
  int max_features = 500;
  /// Horizontal grid cells for detect-new-features distribution.
  int grid_x = 8;
  /// Vertical grid cells for detect-new-features distribution.
  int grid_y = 6;
  /// Minimum pixel distance for detect-new-features. Must be non-negative.
  int min_px_dist = 10;
  /// EV graph debug level. Current native visual graphs accept values in `[0,2]`.
  int debug = 0;
  /// Optional processcvu queue/buffer override. `0` keeps the plugin/runtime default.
  int num_buffers = 0;
  /// Optional GStreamer/processcvu element name. Empty means Neat generates a stable name.
  std::string element_name;
  /// Previous grayscale image tensor name.
  std::string prev_image_name = "prev_image";
  /// Current grayscale image tensor name.
  std::string cur_image_name = "cur_image";
  /// Input point tensor name.
  std::string input_points_name = "input_points";
  /// Output tracked-point tensor name.
  std::string output_points_name = "output_points";
  /// Output point-status tensor name.
  std::string output_status_name = "output_status";
  /// Optional published feature-list tensor name when `detect_new_features != 0`.
  std::string output_features_name = "output_features";

  /// Human-readable, non-throwing summary for logs, diagnostics, and Python `repr`.
  std::string summary() const;
};

class FeatureHistogram final : public Node,
                               public OutputSpecProvider,
                               public NodeContractProvider,
                               public NodeContractConfigurable {
public:
  explicit FeatureHistogram(FeatureHistogramOptions opt = {});
  std::string kind() const override {
    return "FeatureHistogram";
  }
  NodeCapsBehavior caps_behavior() const override {
    return NodeCapsBehavior::Static;
  }
  NodeContractDefinition contract_definition() const override;
  bool compile_node_contract(const ContractCompileInput& input, CompiledNodeContract* out,
                             std::string* err) const override;
  void apply_compiled_contract(const CompiledNodeContract& contract, std::string* err) override;
  std::string backend_fragment(int node_index) const override;
  std::vector<std::string> element_names(int node_index) const override;
  OutputSpec output_spec(const OutputSpec& input) const override;
  const FeatureHistogramOptions& options() const {
    return opt_;
  }

private:
  FeatureHistogramOptions opt_;
};

class GriderFast final : public Node,
                         public OutputSpecProvider,
                         public NodeContractProvider,
                         public NodeContractConfigurable {
public:
  explicit GriderFast(GriderFastOptions opt = {});
  std::string kind() const override {
    return "GriderFast";
  }
  NodeCapsBehavior caps_behavior() const override {
    return NodeCapsBehavior::Static;
  }
  NodeContractDefinition contract_definition() const override;
  bool compile_node_contract(const ContractCompileInput& input, CompiledNodeContract* out,
                             std::string* err) const override;
  void apply_compiled_contract(const CompiledNodeContract& contract, std::string* err) override;
  std::string backend_fragment(int node_index) const override;
  std::vector<std::string> element_names(int node_index) const override;
  OutputSpec output_spec(const OutputSpec& input) const override;
  const GriderFastOptions& options() const {
    return opt_;
  }

private:
  GriderFastOptions opt_;
};

class TrackDescriptor final : public Node,
                              public OutputSpecProvider,
                              public NodeContractProvider,
                              public NodeContractConfigurable {
public:
  explicit TrackDescriptor(TrackDescriptorOptions opt = {});
  std::string kind() const override {
    return "TrackDescriptor";
  }
  NodeCapsBehavior caps_behavior() const override {
    return NodeCapsBehavior::Static;
  }
  NodeContractDefinition contract_definition() const override;
  bool compile_node_contract(const ContractCompileInput& input, CompiledNodeContract* out,
                             std::string* err) const override;
  void apply_compiled_contract(const CompiledNodeContract& contract, std::string* err) override;
  std::string backend_fragment(int node_index) const override;
  std::vector<std::string> element_names(int node_index) const override;
  OutputSpec output_spec(const OutputSpec& input) const override;
  const TrackDescriptorOptions& options() const {
    return opt_;
  }

private:
  TrackDescriptorOptions opt_;
};

class TrackKLT final : public Node,
                       public OutputSpecProvider,
                       public NodeContractProvider,
                       public NodeContractConfigurable {
public:
  explicit TrackKLT(TrackKLTOptions opt = {});
  std::string kind() const override {
    return "TrackKLT";
  }
  NodeCapsBehavior caps_behavior() const override {
    return NodeCapsBehavior::Static;
  }
  NodeContractDefinition contract_definition() const override;
  bool compile_node_contract(const ContractCompileInput& input, CompiledNodeContract* out,
                             std::string* err) const override;
  void apply_compiled_contract(const CompiledNodeContract& contract, std::string* err) override;
  std::string backend_fragment(int node_index) const override;
  std::vector<std::string> element_names(int node_index) const override;
  OutputSpec output_spec(const OutputSpec& input) const override;
  const TrackKLTOptions& options() const {
    return opt_;
  }

private:
  TrackKLTOptions opt_;
};

} // namespace simaai::neat

namespace simaai::neat::nodes {
std::shared_ptr<simaai::neat::Node> FeatureHistogram(FeatureHistogramOptions opt = {});
std::shared_ptr<simaai::neat::Node> GriderFast(GriderFastOptions opt = {});
std::shared_ptr<simaai::neat::Node> TrackDescriptor(TrackDescriptorOptions opt = {});
std::shared_ptr<simaai::neat::Node> TrackKLT(TrackKLTOptions opt = {});
} // namespace simaai::neat::nodes
