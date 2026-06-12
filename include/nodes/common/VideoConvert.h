/**
 * @file
 * @ingroup nodes_common
 * @brief `VideoConvert` Node — color-format conversion (NV12/I420/RGB/BGR/…).
 */
#pragma once

#include "builder/Node.h"

#include <memory>
#include <vector>

namespace simaai::neat {

/**
 * @brief Wraps GStreamer's `videoconvert`. Pick the colorspace via a downstream caps filter.
 *
 * Usage:
 * @code
 * sess.add(sima::nodes::VideoConvert());
 * sess.add(sima::nodes::CapsRaw("RGB"));
 * @endcode
 *
 * @ingroup nodes_common
 */
class VideoConvert final : public Node {
public:
  /// Type label for this Node kind.
  std::string kind() const override {
    return "VideoConvert";
  }
  /// Whether the Node negotiates static or dynamic caps.
  NodeCapsBehavior caps_behavior() const override {
    return NodeCapsBehavior::Dynamic;
  }
  /// GStreamer fragment this Node emits.
  std::string backend_fragment(int node_index) const override;
  /// Deterministic element names this Node will create.
  std::vector<std::string> element_names(int node_index) const override;
};

} // namespace simaai::neat

namespace simaai::neat::nodes {
/// Convenience factory for a `VideoConvert` Node.
std::shared_ptr<simaai::neat::Node> VideoConvert();
} // namespace simaai::neat::nodes
