/**
 * @file
 * @ingroup nodes_sima
 * @brief `Preproc` Node — fused CVU preprocessing kernel (resize + colorconvert + normalize).
 *
 * One-shot CVU pre-processor that runs upstream of the MLA: resizes the input image,
 * converts color space, applies per-channel mean/stddev normalization, and (optionally)
 * tessellates the result into MLA-tile geometry. Implements the "Preproc"
 * `PreprocessGraphFamily` (BF16 path with MLA-side tess); used as the front end of
 * vision pipelines on the BF16 input path.
 */
#pragma once

#include "builder/InputContractConfigurable.h"
#include "builder/NodeContractConfigurable.h"
#include "builder/NodeContractProvider.h"
#include "builder/Node.h"
#include "builder/OutputSpec.h"
#ifdef SIMA_NEAT_INTERNAL
#include "model/internal/ModelRouteRetarget.h"
#endif

#include <nlohmann/json.hpp>

#include <cstdint>
#include <memory>
#include <optional>
#include <string>
#include <utility>
#include <vector>

namespace simaai::neat {
class Model;

/**
 * @brief Construction options for a `Preproc` Node.
 *
 * Most fields are derived from the bound `Model` when the constructor that takes a
 * `Model` is used; explicit setters are provided for graphs that don't have a model
 * yet at Node-construction time.
 *
 * @ingroup nodes_sima
 */
struct PreprocOptions {
  /// Default-construct with framework-default values; tune fields after construction.
  PreprocOptions() = default;
  /// Initialize options from a loaded `Model` (pulls input/output shapes, dtype, scale/zp).
  explicit PreprocOptions(const simaai::neat::Model& model);

  std::vector<int> input_shape;  ///< Input image shape (H, W, C).
  std::vector<int> output_shape; ///< Output tensor shape after resize/normalize.
  std::vector<int> slice_shape;  ///< Optional slice/tile shape used for batched processing.

  int scaled_width = 0;  ///< Intermediate scaled width (before crop/pad), pixels.
  int scaled_height = 0; ///< Intermediate scaled height (before crop/pad), pixels.

  int batch_size = 1; ///< Batch size processed per invocation.

  bool normalize = true;             ///< Apply mean/stddev normalization.
  bool aspect_ratio = true;          ///< Preserve aspect ratio during resize (letterbox).
  bool tessellate = true;            ///< Tessellate output into MLA tile geometry.
  bool dynamic_input_dims = true;    ///< Allow dynamic input dimensions at runtime.
  bool single_output_handoff = true; ///< Hand off a single output buffer per cycle (vs. ping-pong).

  int input_offset = 0;  ///< Byte offset into the input buffer.
  int input_stride = 1;  ///< Element stride for input addressing.
  int output_stride = 1; ///< Element stride for output addressing.

  std::optional<std::int64_t> q_zp; ///< Output quantization zero-point (when emitting INT8/INT16).
  std::optional<double> q_scale;    ///< Output quantization scale (when emitting INT8/INT16).

  std::vector<float> channel_mean = {0.0f, 0.0f, 0.0f};   ///< Per-channel mean used by normalize.
  std::vector<float> channel_stddev = {1.0f, 1.0f, 1.0f}; ///< Per-channel stddev used by normalize.

  std::string input_img_type;            ///< Input pixel format (e.g. `"NV12"`, `"RGB"`).
  std::string output_img_type = "RGB";   ///< Output color space.
  std::string output_dtype = "INT16";    ///< Output element dtype (e.g. `"INT16"`, `"BF16"`).
  std::string scaling_type = "BILINEAR"; ///< Resize interpolation (`"BILINEAR"`, `"NEAREST"`).
  std::string padding_type = "CENTER";   ///< Letterbox padding mode (`"CENTER"`, `"TOPLEFT"`).
  int pad_value = 0;                     ///< Raw image-space letterbox pad fill value.

  std::string graph_name = "preproc";      ///< CVU graph name in the kernel config.
  std::string node_name = "preproc";       ///< CVU node name in the kernel config.
  std::string element_name;                ///< Optional GStreamer element name.
  std::string cpu = "CVU";                 ///< CPU/accelerator this stage runs on.
  std::string next_cpu = "CVU";            ///< CPU/accelerator the downstream stage runs on.
  std::string debug = "EVXX_DBG_DISABLED"; ///< Debug-output flag passed to the CVU kernel.

  std::string upstream_name = "decoder";        ///< Name of the upstream element for tag wiring.
  std::string graph_input_name = "input_image"; ///< Input tensor name within the CVU graph.

  int num_buffers = 0;                 ///< Override for the element's buffer pool size.
  int num_buffers_model = 0;           ///< Buffer count derived from the bound model.
  bool num_buffers_locked = false;     ///< If true, planner won't override `num_buffers`.
  bool model_managed_contract = false; ///< If true, the model owns the node contract resolution.
#ifdef SIMA_NEAT_INTERNAL
  std::shared_ptr<const simaai::neat::internal::ModelLineageBinding> model_lineage;
#endif

  /// Replace the input image shape (H, W, C).
  void set_input_shape(std::vector<int> shape) {
    input_shape = std::move(shape);
  }

  /// Replace the output tensor shape.
  void set_output_shape(std::vector<int> shape) {
    output_shape = std::move(shape);
  }

  /// Replace the slice/tile shape used for batched processing.
  void set_slice_shape(std::vector<int> shape) {
    slice_shape = std::move(shape);
  }

  /// Safe accessor: return `shape[index]` or `0` if out of bounds.
  static int shape_dim(const std::vector<int>& shape, std::size_t index) {
    return shape.size() > index ? shape[index] : 0;
  }

  /// Last-axis channel count for an HWC-style shape, or `0` if rank < 3.
  static int shape_channels(const std::vector<int>& shape) {
    return shape.size() >= 3 ? shape.back() : 0;
  }

  /// Input image height (axis 0 of `input_shape`).
  int input_height() const {
    return shape_dim(input_shape, 0);
  }

  /// Input image width (axis 1 of `input_shape`).
  int input_width() const {
    return shape_dim(input_shape, 1);
  }

  /// Input channel count.
  int input_channels() const {
    return shape_channels(input_shape);
  }

  /// Output tensor height.
  int output_height() const {
    return shape_dim(output_shape, 0);
  }

  /// Output tensor width.
  int output_width() const {
    return shape_dim(output_shape, 1);
  }

  /// Output channel count.
  int output_channels() const {
    return shape_channels(output_shape);
  }

  /// Slice height.
  int slice_height() const {
    return shape_dim(slice_shape, 0);
  }

  /// Slice width.
  int slice_width() const {
    return shape_dim(slice_shape, 1);
  }

  /// Slice channel count.
  int slice_channels() const {
    return shape_channels(slice_shape);
  }

  /// True if `input_shape` is fully populated.
  bool has_input_shape() const {
    return input_height() > 0 && input_width() > 0 && input_channels() > 0;
  }

  /// True if `output_shape` is fully populated.
  bool has_output_shape() const {
    return output_height() > 0 && output_width() > 0 && output_channels() > 0;
  }

  /// True if `slice_shape` is fully populated.
  bool has_slice_shape() const {
    return slice_height() > 0 && slice_width() > 0 && slice_channels() > 0;
  }
};

/**
 * @brief Fused CVU preprocessing Node — resize + color-convert + normalize (+ optional tess).
 *
 * Implements the "Preproc" `PreprocessGraphFamily`. Place upstream of the MLA on the
 * BF16 input path; pair with `Quant`/`QuantTess` instead when the model expects INT8 input.
 *
 * @ingroup nodes_sima
 */
class Preproc final : public Node,
                      public OutputSpecProvider,
                      public InputContractConfigurable,
                      public NodeContractProvider,
                      public NodeContractConfigurable {
public:
  /// Construct with optional `PreprocOptions`.
  explicit Preproc(PreprocOptions opt = {});

  /// Type label for this Node kind.
  std::string kind() const override {
    return "Preproc";
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
  /// Apply an input contract from upstream.
  void apply_input_contract(const InputContract& contract, std::string* err) override;
  /// Structural contract definition for this Node.
  NodeContractDefinition contract_definition() const override;
  /// Compile this Node's contract from the given input.
  bool compile_node_contract(const ContractCompileInput& input, CompiledNodeContract* out,
                             std::string* err) const override;
  /// Apply a compiled contract back into this Node.
  void apply_compiled_contract(const CompiledNodeContract& contract, std::string* err) override;

  /// Inspect the Node's options.
  const PreprocOptions& options() const {
    return opt_;
  }
  /// Path to the kernel config JSON, if one was loaded from disk.
  const std::string& config_path() const {
    return config_path_;
  }
  /// Path to the snapshotted config JSON written during compilation.
  const std::string& config_snapshot_path() const {
    return config_path_;
  }
  /// Resolved kernel config JSON, or null if no config was supplied/loaded.
  const nlohmann::json* config_json() const;

private:
  struct PreprocConfigHolder;

  void materialize_config_from_input_contract(const InputContract& contract);
  PreprocOptions opt_;
  std::shared_ptr<PreprocConfigHolder> config_holder_;
  std::string config_path_;
};

} // namespace simaai::neat

namespace simaai::neat::nodes {
/// Convenience factory for a `Preproc` Node with optional `PreprocOptions`.
std::shared_ptr<simaai::neat::Node> Preproc(PreprocOptions opt = {});
} // namespace simaai::neat::nodes
