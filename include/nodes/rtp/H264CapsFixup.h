/**
 * @file
 * @ingroup nodes_rtp
 * @brief `H264CapsFixup` Node — normalizes ambiguous H.264 RTP caps before depayload.
 *
 * Some upstream RTP sources negotiate caps that omit framerate or frame size, which
 * trips up downstream stages that need them. This Node patches in fallback values
 * so the rest of the pipeline can negotiate cleanly. Insert between an RTP source
 * and `H264Depacketize` when caps from the source are unreliable.
 */
#pragma once

#include "builder/Node.h"

#include <memory>
#include <string>
#include <vector>

namespace simaai::neat {

/**
 * @brief Helper Node that fills in missing H.264 RTP caps fields with fallbacks.
 *
 * Place between an RTP source and `H264Depacketize` when the source emits caps
 * that lack framerate or geometry. The fallback values are only applied to fields
 * the upstream caps don't already specify.
 *
 * @ingroup nodes_rtp
 */
class H264CapsFixup final : public Node {
public:
  /// Construct with fallback framerate and frame geometry to inject when missing.
  H264CapsFixup(int fallback_fps, int fallback_width, int fallback_height);

  /// Type label for this Node kind.
  std::string kind() const override {
    return "H264CapsFixup";
  }
  /// Whether the Node negotiates static or dynamic caps.
  NodeCapsBehavior caps_behavior() const override {
    return NodeCapsBehavior::Static;
  }
  /// GStreamer fragment this Node emits.
  std::string backend_fragment(int node_index) const override;
  /// Deterministic element names this Node will create.
  std::vector<std::string> element_names(int node_index) const override;

  /// Fallback framerate, in frames per second.
  int fallback_fps() const {
    return fallback_fps_;
  }
  /// Fallback frame width, in pixels.
  int fallback_width() const {
    return fallback_width_;
  }
  /// Fallback frame height, in pixels.
  int fallback_height() const {
    return fallback_height_;
  }

private:
  int fallback_fps_ = 30;
  int fallback_width_ = 1280;
  int fallback_height_ = 720;
};

} // namespace simaai::neat

namespace simaai::neat::nodes {

/// Convenience factory for an `H264CapsFixup` Node.
std::shared_ptr<simaai::neat::Node> H264CapsFixup(int fallback_fps = 30, int fallback_width = 1280,
                                                  int fallback_height = 720);

} // namespace simaai::neat::nodes
