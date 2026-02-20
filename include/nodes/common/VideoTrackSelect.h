/**
 * @file
 * @ingroup nodes_common
 * @brief Helper for selecting video pads from qtdemux.
 */
#pragma once

#include "builder/Node.h"

#include <memory>
#include <vector>

namespace simaai::neat {

// qtdemux exposes pads like demux.video_0, demux.audio_0, etc.
// This node both creates the demux element AND selects a video pad.
class VideoTrackSelect final : public Node {
public:
  explicit VideoTrackSelect(int video_pad_index = 0);

  std::string kind() const override {
    return "VideoTrackSelect";
  }
  NodeCapsBehavior caps_behavior() const override {
    return NodeCapsBehavior::Dynamic;
  }

  std::string backend_fragment(int node_index) const override;
  std::vector<std::string> element_names(int node_index) const override;

  int video_pad_index() const {
    return idx_;
  }

private:
  int idx_ = 0;
};

} // namespace simaai::neat

namespace simaai::neat::nodes {
std::shared_ptr<simaai::neat::Node> VideoTrackSelect(int video_pad_index = 0);
} // namespace simaai::neat::nodes
