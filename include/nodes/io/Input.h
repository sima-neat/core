/**
 * @file
 * @ingroup nodes_io
 * @brief `Input` Node — push-mode source. Lets the application feed samples via `Run::push()`.
 *
 * Wraps an `appsrc`-style element. Use this when the application owns frame production
 * (e.g. capturing from a custom source, replaying a buffer, or feeding test data) and
 * needs to deliver samples to the pipeline by hand. A Session that begins with an
 * `Input` Node is built with `Session::build()` and driven by `Run::push()` rather
 * than `Run::run()`.
 */
#pragma once

#include "builder/Node.h"
#include "builder/OutputSpec.h"
#include "pipeline/FormatSpec.h"

#include <memory>
#include <cstdint>
#include <optional>
#include <string>
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
  bool enabled = false;            ///< If false, no preprocess metadata is emitted.
  int target_width = 0;            ///< Model input width, in pixels.
  int target_height = 0;           ///< Model input height, in pixels.
  int scaled_width = 0;            ///< Width of the scaled content within the target frame.
  int scaled_height = 0;           ///< Height of the scaled content within the target frame.
  std::string resize_mode = "none";///< Resize/letterbox strategy label (`none`, `fit`, etc.).
  int pad_value = 0;               ///< Pad fill value used by letterboxing.

  std::string color_in;            ///< Input color space label (e.g. `RGB`, `BGR`).
  std::string color_out;           ///< Output color space label after preprocess.
  /// Axis permutation applied by preprocess `layout_convert`, if any.
  std::vector<int> axis_perm;

  bool normalize = false;          ///< True if preprocess applies normalization.
  bool quantize = false;           ///< True if preprocess applies quantization.
  bool tessellate = false;         ///< True if preprocess applies tessellation.
};

/**
 * @brief Where ingress buffers should be allocated when the application pushes samples.
 *
 * `Auto` defers to legacy / session-level selection; the explicit values force a
 * specific memory target.
 *
 * @ingroup nodes_io
 */
enum class InputMemoryPolicy {
  Auto = 0,        ///< Defer to legacy / session-level target selection.
  Ev74,            ///< Allocate in EV74-visible memory.
  Dms0,            ///< Allocate in DMS0 memory.
  SystemMemory,    ///< Allocate in plain system memory.
};

/**
 * @brief Caps, buffering, and pool options for the `Input` Node.
 *
 * @ingroup nodes_io
 */
struct InputOptions {
  std::string media_type = "video/x-raw"; ///< Top-level GStreamer media type for the negotiated caps.
  FormatSpec format;                      ///< Pixel/tensor format descriptor.
  int width = -1;                         ///< Frame width, in pixels. `-1` = unspecified.
  int height = -1;                        ///< Frame height, in pixels. `-1` = unspecified.
  int depth = -1;                         ///< Frame depth (e.g. tensor channels). `-1` = unspecified.
  /// Optional dynamic-input limits used for validation/pool sizing.
  /// These do not constrain negotiated caps and are only checked at push time.
  int max_width = -1;                     ///< Max width accepted at push time. `-1` = no cap.
  int max_height = -1;                    ///< Max height accepted at push time. `-1` = no cap.
  int max_depth = -1;                     ///< Max depth accepted at push time. `-1` = no cap.
  /// Optional fixed framerate for caps (0/1 means "unspecified").
  int fps_n = 0;                          ///< Framerate numerator.
  int fps_d = 1;                          ///< Framerate denominator.
  /// Optional full caps string override (used for multi-tensor caps).
  std::string caps_override;

  bool is_live = true;                    ///< Mark the source as live (`appsrc` `is-live`).
  bool do_timestamp = true;               ///< Let `appsrc` stamp PTS on push.
  bool block = true;                      ///< Block on pool exhaustion instead of dropping.
  int stream_type = 0;                    ///< `GST_APP_STREAM_TYPE_STREAM` by default.
  std::uint64_t max_bytes = 0;            ///< `appsrc` `max-bytes` back-pressure threshold. `0` = unlimited.

  bool use_simaai_pool = true;            ///< Allocate from the SiMa-aware buffer pool.
  int pool_min_buffers = 1;               ///< Minimum buffers held by the pool.
  int pool_max_buffers = 2;               ///< Maximum buffers held by the pool.
  /// Ingress allocation target policy used by appsrc input buffer allocation.
  /// `Auto` keeps legacy target selection unless caller/model/session overrides.
  InputMemoryPolicy memory_policy = InputMemoryPolicy::Auto;

  /// Optional `GstSimaMeta` buffer name override. Leave empty to avoid forcing a legacy default.
  std::string buffer_name;

  /// Optional preprocess runtime metadata template used to annotate `GstSimaMeta`
  /// with geometry/affine context for downstream plugins (e.g. box decode).
  std::optional<PreprocessMetaTemplate> preprocess_meta;

};

/**
 * @brief Push-mode source Node. The application feeds samples via `Run::push()`.
 *
 * Add this Node when the application owns frame production. Because it carries
 * `InputRole::Push`, the Session must be driven through `Session::build()` plus
 * `Run::push()` per sample (not `Run::run()`, which is for source-role pipelines).
 *
 * @ingroup nodes_io
 */
class Input final : public Node, public OutputSpecProvider {
public:
  /// Construct with caps / pool / memory options.
  explicit Input(InputOptions opt);

  /// Type label for this Node kind.
  std::string kind() const override {
    return "Input";
  }
  /// User-facing label for this Node.
  std::string user_label() const override {
    return "mysrc";
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
};

} // namespace simaai::neat

namespace simaai::neat::nodes {
/// Convenience factory for an `Input` Node with optional `InputOptions`.
std::shared_ptr<simaai::neat::Node> Input(InputOptions opt = {});
} // namespace simaai::neat::nodes
