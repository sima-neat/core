/**
 * @file
 * @ingroup nodes_common
 * @brief `VideoTrackSelect` Node — instantiate `qtdemux` and pick one video pad.
 *
 * Bundles the demuxer and pad selection into one Node so an MP4 reader pipeline reads
 * cleanly.
 */
#pragma once

#include "builder/Node.h"

#include <memory>
#include <vector>

namespace simaai::neat {

/**
 * @brief Creates a `qtdemux` element and selects a specific video pad (e.g., `demux.video_0`).
 *
 * Use after `FileInput` for MP4 / MOV containers.
 *
 * @ingroup nodes_common
 */
class VideoTrackSelect final : public Node {
public:
  /// Construct selecting video pad `video_pad_index` (default `0`).
  explicit VideoTrackSelect(int video_pad_index = 0);

  /// Type label for this Node kind.
  std::string kind() const override {
    return "VideoTrackSelect";
  }
  /// Whether the Node negotiates static or dynamic caps.
  NodeCapsBehavior caps_behavior() const override {
    return NodeCapsBehavior::Dynamic;
  }

  /// GStreamer fragment this Node emits.
  std::string backend_fragment(int node_index) const override;
  /// Deterministic element names this Node will create.
  std::vector<std::string> element_names(int node_index) const override;

  /// Which video pad index this Node selects.
  int video_pad_index() const {
    return idx_;
  }

private:
  int idx_ = 0;
};

} // namespace simaai::neat

namespace simaai::neat::nodes {
/// Convenience factory for a `VideoTrackSelect` Node.
std::shared_ptr<simaai::neat::Node> VideoTrackSelect(int video_pad_index = 0);
} // namespace simaai::neat::nodes
