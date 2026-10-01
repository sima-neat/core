/**
 * @file
 * @ingroup nodes_sima
 * @brief EV74 visual-frontend processcvu Nodes (`FeatureHistogram`, `GriderFast`,
 *        `TrackDescriptor`, `TrackKLT`, `MetoakDepth`).
 */
#pragma once

#include "builder/Node.h"
#include "builder/NodeContractConfigurable.h"
#include "builder/NodeContractProvider.h"
#include "builder/OutputSpec.h"

#include <cstdint>
#include <memory>
#include <limits>
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

/**
 * @brief Metoak S315 SIMOR depth-map Node: I420->RGB + disparity->metric-depth/point-cloud,
 *        offloaded to the native `simor_depth_map` EV74 kernel (graph id 20).
 *
 * Fixed six-input/three-output contract (batch 1; see the "Metoak EV74 support in Neat"
 * migration notes for the full rationale). Unlike the other native visual graphs above, this
 * one is not a feature/tracking kernel: `bf_mm_name`/`proj_name` are per-frame scalar
 * calibration inputs (not batched images), and it has three heterogeneous outputs rather than
 * one primary tensor.
 *
 * Tensor names currently must retain their canonical defaults; unsupported aliases fail
 * during contract compilation, before any device access.
 *
 * Public ABI (fixed geometry from `width`/`height`; disparity subpixel scale 32 is baked into
 * the kernel, not a parameter here):
 * - inputs:
 *   - `y_name`,      UInt8   `[height,width]`      -- I420 Y
 *   - `u_name`,      UInt8   `[height/2,width/2]`  -- I420 U
 *   - `v_name`,      UInt8   `[height/2,width/2]`  -- I420 V
 *   - `disp_name`,   UInt16  `[height,width]`      -- raw disparity
 *   - `bf_mm_name`,  Float32 `[1]`                 -- selected calibration BF (mm)
 *   - `proj_name`,   Float32 `[3]`                 -- {fx_fy, cx, cy}
 * - outputs:
 *   - `rgb_output_name`,    UInt8   `[height,width,3]` -- interleaved RGB
 *   - `depth_output_name`,  UInt16  `[height,width]`   -- millimeters, 0 = invalid (primary)
 *   - `points_output_name`, Float32 `[height,width,3]` -- interleaved XYZ meters, NaN = invalid
 *
 * `depth_output_name` is the primary output for `output_spec()`'s single-boundary description
 * only; all three outputs are always published together -- selecting one as primary must not
 * be read as hiding the other two.
 */
struct MetoakDepthOptions {
  /// Even input width in [8, 2048]; use native sensor geometry (640 for S315).
  int width = 0;
  /// Even input height in [8, 1536]; use native sensor geometry (360 for S315).
  int height = 0;
  /// EV graph debug level. Current native visual graphs accept values in `[0,2]`.
  int debug = 0;
  /// Optional processcvu buffer override. `0` resolves to the async four-buffer default.
  /// Sync builds apply their normal pool clamp; async builds require four buffers.
  int num_buffers = 0;
  /// Optional GStreamer/processcvu element name. Empty means Neat generates a stable name.
  std::string element_name;

  /// I420 Y input tensor name.
  std::string y_name = "y_src";
  /// I420 U input tensor name.
  std::string u_name = "u_src";
  /// I420 V input tensor name.
  std::string v_name = "v_src";
  /// Raw disparity input tensor name.
  std::string disp_name = "disp_src";
  /// Per-frame calibrated BF (base * focal length, mm) input tensor name.
  std::string bf_mm_name = "bf_mm_src";
  /// Per-frame projection `{fx_fy, cx, cy}` input tensor name.
  std::string proj_name = "proj_src";
  /// RGB output tensor name.
  std::string rgb_output_name = "rgb_dst";
  /// Metric depth output tensor name.
  std::string depth_output_name = "depth_dst";
  /// XYZ point-cloud output tensor name.
  std::string points_output_name = "points_dst";

  /// Human-readable, non-throwing summary for logs, diagnostics, and Python `repr`.
  std::string summary() const;
};

/**
 * Explicit S315 packed-wire ingress for MetoakDepth. Capture is UInt8 [360,1920];
 * decoding and calibration-header interpretation run on EV74, not the host.
 * Principal point must come from calibration for the native 640x360 camera mode.
 * Separate options preserve the existing planar-input public ABI.
 */
struct MetoakRawInputOptions {
  float cx = std::numeric_limits<float>::quiet_NaN();
  float cy = std::numeric_limits<float>::quiet_NaN();
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

class MetoakDepth final : public Node,
                          public OutputSpecProvider,
                          public NodeContractProvider,
                          public NodeContractConfigurable {
public:
  explicit MetoakDepth(MetoakDepthOptions opt = {});
  std::string kind() const override {
    return "MetoakDepth";
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
  const MetoakDepthOptions& options() const {
    return opt_;
  }

private:
  MetoakDepthOptions opt_;
};

} // namespace simaai::neat

namespace simaai::neat::nodes {
std::shared_ptr<simaai::neat::Node> FeatureHistogram(FeatureHistogramOptions opt = {});
std::shared_ptr<simaai::neat::Node> GriderFast(GriderFastOptions opt = {});
std::shared_ptr<simaai::neat::Node> TrackDescriptor(TrackDescriptorOptions opt = {});
std::shared_ptr<simaai::neat::Node> TrackKLT(TrackKLTOptions opt = {});
std::shared_ptr<simaai::neat::Node> MetoakDepth(MetoakDepthOptions opt = {});
/// Capture-driven raw mode; rejects ordinary images and unsupported wire profiles.
std::shared_ptr<simaai::neat::Node> MetoakDepth(MetoakDepthOptions opt, MetoakRawInputOptions raw);
} // namespace simaai::neat::nodes
