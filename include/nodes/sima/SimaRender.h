/**
 * @file
 * @ingroup nodes_sima
 * @brief `SimaRender` Node — renders bounding-box overlays onto a video frame.
 *
 * Consumes the boxes emitted by `SimaBoxDecode` and the source frame, draws labelled
 * box overlays in-place, and forwards the annotated frame downstream. Place after
 * `SimaBoxDecode` in detection-display pipelines that show results on screen or stream
 * them out via `H264EncodeSima` + `H264Packetize`.
 */
#pragma once

#include "builder/Node.h"
#include "builder/OutputSpec.h"

#include <memory>
#include <string>
#include <vector>

namespace simaai::neat {

/**
 * @brief Construction options for a `SimaRender` Node.
 *
 * @ingroup nodes_sima
 */
struct SimaRenderOptions {
  std::string config_path;        ///< Optional path to an external render config JSON.
  int sima_allocator_type = 2;    ///< SiMa DMA allocator selector (board-specific; default `2`).
  bool silent = true;             ///< If true, suppress element-level diagnostic logging.
  bool emit_signals = false;      ///< Enable GStreamer signal emission for buffer hand-off.
  bool transmit = false;          ///< If true, transmit results downstream via the element's transport.
};

/**
 * @brief Node that renders bounding-box overlays onto a video frame.
 *
 * @ingroup nodes_sima
 */
class SimaRender final : public Node, public OutputSpecProvider {
public:
  /// Construct with optional `SimaRenderOptions`.
  explicit SimaRender(SimaRenderOptions opt = {});

  /// Type label for this Node kind.
  std::string kind() const override {
    return "SimaRender";
  }
  /// Whether the Node negotiates static or dynamic caps.
  NodeCapsBehavior caps_behavior() const override {
    return NodeCapsBehavior::Static;
  }
  /// Whether this Node is driven by a JSON config.
  bool has_config_json() const override;
  /// Legacy hook for node-local JSON wiring.
  bool wire_input_names(const std::vector<std::string>& upstream_names,
                        const std::string& tag) override;
  /// GStreamer fragment this Node emits.
  std::string backend_fragment(int node_index) const override;
  /// Deterministic element names this Node will create.
  std::vector<std::string> element_names(int node_index) const override;
  /// Negotiated downstream caps produced by this Node.
  OutputSpec output_spec(const OutputSpec& input) const override;

  /// Inspect the Node's options.
  const SimaRenderOptions& options() const {
    return opt_;
  }

private:
  struct ConfigHolder;

  SimaRenderOptions opt_;
  std::shared_ptr<ConfigHolder> config_holder_;
  std::string config_path_;
};

} // namespace simaai::neat

namespace simaai::neat::nodes {
/// Convenience factory for a `SimaRender` Node with optional `SimaRenderOptions`.
std::shared_ptr<simaai::neat::Node> SimaRender(SimaRenderOptions opt = {});
} // namespace simaai::neat::nodes
