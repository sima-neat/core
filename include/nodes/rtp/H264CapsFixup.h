/**
 * @file
 * @ingroup nodes_rtp
 * @brief H264 caps fixup helper node.
 */
#pragma once

#include "builder/Node.h"

#include <memory>
#include <string>
#include <vector>

namespace simaai::neat {

class H264CapsFixup final : public Node {
public:
  H264CapsFixup(int fallback_fps, int fallback_width, int fallback_height);

  std::string kind() const override {
    return "H264CapsFixup";
  }
  NodeCapsBehavior caps_behavior() const override {
    return NodeCapsBehavior::Static;
  }
  std::string backend_fragment(int node_index) const override;
  std::vector<std::string> element_names(int node_index) const override;

  int fallback_fps() const {
    return fallback_fps_;
  }
  int fallback_width() const {
    return fallback_width_;
  }
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

std::shared_ptr<simaai::neat::Node> H264CapsFixup(int fallback_fps = 30, int fallback_width = 1280,
                                                  int fallback_height = 720);

} // namespace simaai::neat::nodes
