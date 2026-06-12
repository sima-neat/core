/**
 * @file
 * @ingroup nodes_common
 * @brief `VideoScale` Node — software resize via GStreamer's `videoscale`.
 *
 * For preprocess-pipeline resizing, prefer the planner-inserted CVU resize kernel
 * (configured via `PreprocessOptions::resize`). This Node is for explicit user-space
 * resizes outside the model's preprocess.
 */
#pragma once

#include "builder/Node.h"

#include <memory>
#include <vector>

namespace simaai::neat {

/**
 * @brief Wraps GStreamer's `videoscale`. Pin target dimensions via a downstream caps filter.
 *
 * @ingroup nodes_common
 */
class VideoScale final : public Node {
public:
  /// Type label for this Node kind.
  std::string kind() const override {
    return "VideoScale";
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
/// Convenience factory for a `VideoScale` Node.
std::shared_ptr<simaai::neat::Node> VideoScale();
} // namespace simaai::neat::nodes
