/**
 * @file
 * @ingroup nodes_io
 * @brief `Input` Node — push-mode source. Lets the application feed samples via `Run::push()`.
 *
 * Wraps an `appsrc`-style element. Use this when the application owns frame production
 * (e.g. capturing from a custom source, replaying a buffer, or feeding test data) and
 * needs to deliver samples to the pipeline by hand. A Graph that begins with an
 * `Input` Node is built with `Graph::build()` and driven by `Run::push()` rather
 * than `Run::run()`.
 */
#pragma once

#include "builder/Node.h"
#include "builder/OutputSpec.h"
#include "pipeline/FormatSpec.h"
#include "pipeline/PayloadType.h"
#include "pipeline/TensorCore.h"

#include <memory>
#include <cstdint>
#include <optional>
#include <string>
#include <string_view>
#include <vector>

namespace simaai::neat {

/**
 * @brief Preprocess metadata template attached to ingress buffers.
 *
 * Annotates `GstSimaMeta` with affine/geometry context so downstream plugins
 * (e.g. detection-box decoders) can map model outputs back to original frame
 * coordinates.
 *
 * @ingroup nodes_io
 */
struct PreprocessMetaTemplate {
  bool enabled = false;             ///< If false, no preprocess metadata is emitted.
  int target_width = 0;             ///< Model input width, in pixels.
  int target_height = 0;            ///< Model input height, in pixels.
  int scaled_width = 0;             ///< Width of the scaled content within the target frame.
  int scaled_height = 0;            ///< Height of the scaled content within the target frame.
  std::string resize_mode = "none"; ///< Resize/letterbox strategy label (`none`, `fit`, etc.).
  int pad_value = 0;                ///< Pad fill value used by letterboxing.

  std::string color_in;  ///< Input color space label (e.g. `RGB`, `BGR`).
  std::string color_out; ///< Output color space label after preprocess.
  /// Axis permutation applied by preprocess `layout_convert`, if any.
  std::vector<int> axis_perm;

  bool normalize = false;  ///< True if preprocess applies normalization.
  bool quantize = false;   ///< True if preprocess applies quantization.
  bool tessellate = false; ///< True if preprocess applies tessellation.

  bool roi_list_enabled = false;   ///< True when runtime ROI-list metadata is emitted.
  std::vector<PreprocessRoi> rois; ///< Runtime ROI windows in output-slot order.
  int roi_input_batch_size = 0;    ///< Number of source images available to ROI batch indices.
  int roi_source_width = 0;        ///< Source image width for runtime ROI extraction.
  int roi_source_height = 0;       ///< Source image height for runtime ROI extraction.
  int roi_source_stride_bytes = 0; ///< Source row stride in bytes for runtime ROI extraction.
  int roi_pad_value = 0;           ///< Pad value used for out-of-frame ROI pixels.
};

/**
 * @brief Where ingress buffers should be allocated when the application pushes samples.
 *
 * `Auto` lets Core infer or choose the default allocation target; the explicit
 * values force a specific memory target.
 *
 * @ingroup nodes_io
 */
enum class InputMemoryPolicy {
  Auto = 0,     ///< Let Core infer or choose the default allocation target.
  Ev74,         ///< Allocate in EV74-visible memory.
  Dms0,         ///< Allocate in DMS0 memory.
  SystemMemory, ///< Allocate in plain system memory.
};

/**
 * @brief Caps, buffering, and pool options for the `Input` Node.
 *
 * @ingroup nodes_io
 */
struct InputOptions {
  PayloadType payload_type = PayloadType::Auto; ///< Public payload family.
  FormatSpec format;                            ///< Pixel/tensor format descriptor.
  int width = -1;                               ///< Frame width, in pixels. `-1` = unspecified.
  int height = -1;                              ///< Frame height, in pixels. `-1` = unspecified.
  int depth = -1; ///< Frame depth (e.g. tensor channels). `-1` = unspecified.
  /// Optional dynamic-input limits used for validation/pool sizing.
  /// These do not constrain negotiated caps and are only checked at push time.
  int max_width = -1;  ///< Max width accepted at push time. `-1` = no cap.
  int max_height = -1; ///< Max height accepted at push time. `-1` = no cap.
  int max_depth = -1;  ///< Max depth accepted at push time. `-1` = no cap.
  /// Optional fixed framerate for caps (0/1 means "unspecified").
  int fps_n = 0; ///< Framerate numerator.
  int fps_d = 1; ///< Framerate denominator.
  /// Optional full caps string override (used for multi-tensor caps).
  std::string caps_override;

  bool is_live = true;         ///< Mark the source as live (`appsrc` `is-live`).
  bool do_timestamp = true;    ///< Let `appsrc` stamp PTS on push.
  bool block = true;           ///< Block on pool exhaustion instead of dropping.
  int stream_type = 0;         ///< `GST_APP_STREAM_TYPE_STREAM` by default.
  std::uint64_t max_bytes = 0; ///< `appsrc` `max-bytes` back-pressure threshold. `0` = unlimited.

  /// @deprecated Accepted for source compatibility. Use `memory_policy` instead:
  /// `false` maps to `InputMemoryPolicy::SystemMemory`; default SiMa allocation
  /// behavior maps to `InputMemoryPolicy::Auto`; explicit targets use
  /// `InputMemoryPolicy::Ev74` or `InputMemoryPolicy::Dms0`.
  /// Older Graph JSON using this field is still accepted. New saved Graph JSON
  /// writes `memory_policy`.
  bool use_simaai_pool = true;
  int pool_min_buffers = 1; ///< Minimum buffers held by the pool.
  int pool_max_buffers = 2; ///< Maximum buffers held by the pool.
  /// Ingress allocation target policy used by appsrc input buffer allocation.
  /// `Auto` lets graph lowering infer from downstream nodes where possible.
  InputMemoryPolicy memory_policy = InputMemoryPolicy::Auto;

  /// Optional `GstSimaMeta` buffer name override. Leave empty to avoid forcing a legacy default.
  std::string buffer_name;

  /// Optional preprocess runtime metadata template used to annotate `GstSimaMeta`
  /// with geometry/affine context for downstream plugins (e.g. box decode).
  std::optional<PreprocessMetaTemplate> preprocess_meta;
};

/// Resolve the internal caps media type requested by public `InputOptions`.
inline std::string resolve_input_media_type(const InputOptions& opt) {
  switch (opt.payload_type) {
  case PayloadType::Image:
    return "video/x-raw";
  case PayloadType::Tensor:
    return "application/vnd.simaai.tensor";
  case PayloadType::Encoded:
    if (opt.format.tag == FormatTag::H264) {
      return "video/x-h264";
    }
    return "";
  case PayloadType::Auto:
  default:
    return "";
  }
}

/**
 * @brief Push-mode source Node. The application feeds samples via `Run::push()`.
 *
 * Add this Node when the application owns frame production. Because it carries
 * `InputRole::Push`, the Graph must be driven through `Graph::build()` plus
 * `Run::push()` per sample (not `Run::run()`, which is for source-role pipelines).
 *
 * @ingroup nodes_io
 */
class Input final : public Node, public OutputSpecProvider {
public:
  /// Construct with caps / pool / memory options.
  explicit Input(InputOptions opt);
  /// Construct with a public Graph input endpoint name.
  explicit Input(std::string name);
  /// Construct with a public Graph input endpoint name and caps / pool / memory options.
  Input(std::string name, InputOptions opt);

  /// Type label for this Node kind.
  std::string kind() const override {
    return "Input";
  }
  /// User-facing label for this Node.
  std::string user_label() const override {
    if (!endpoint_name_.empty()) {
      return endpoint_name_;
    }
    return "mysrc";
  }
  /// Explicit public endpoint name, empty when unnamed.
  const std::string& endpoint_name() const noexcept {
    return endpoint_name_;
  }
  /// Role this Node plays as a stream source.
  InputRole input_role() const override {
    return InputRole::Push;
  }
  /// Whether the Node negotiates static or dynamic caps.
  NodeCapsBehavior caps_behavior() const override {
    return NodeCapsBehavior::Static;
  }
  /// Optional buffer-name hint used during wiring.
  std::string buffer_name_hint(int node_index) const override;

  /// GStreamer fragment this Node emits.
  std::string backend_fragment(int node_index) const override;
  /// Deterministic element names this Node will create.
  std::vector<std::string> element_names(int node_index) const override;
  /// Negotiated downstream caps produced by this Node.
  OutputSpec output_spec(const OutputSpec& input) const override;

  /// Inspect the Node's options.
  const InputOptions& options() const {
    return opt_;
  }
  /// Render the negotiated caps as a GStreamer caps string.
  std::string caps_string() const;

private:
  InputOptions opt_;
  std::string endpoint_name_;
};

} // namespace simaai::neat

namespace simaai::neat::nodes {
/// Convenience factory for an `Input` Node with optional `InputOptions`.
std::shared_ptr<simaai::neat::Node> Input(InputOptions opt = {});
/// Convenience factory for a named public Graph input endpoint.
std::shared_ptr<simaai::neat::Node> Input(std::string name, InputOptions opt = {});
} // namespace simaai::neat::nodes
