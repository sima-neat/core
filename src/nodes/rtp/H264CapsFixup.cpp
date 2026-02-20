#include "nodes/rtp/H264CapsFixup.h"

#include <memory>
#include <string>
#include <vector>

namespace simaai::neat {

H264CapsFixup::H264CapsFixup(int fallback_fps, int fallback_width, int fallback_height)
    : fallback_fps_(fallback_fps), fallback_width_(fallback_width),
      fallback_height_(fallback_height) {}

std::string H264CapsFixup::backend_fragment(int node_index) const {
  return "identity name=n" + std::to_string(node_index) + "_h264_capsfix silent=true";
}

std::vector<std::string> H264CapsFixup::element_names(int node_index) const {
  return {"n" + std::to_string(node_index) + "_h264_capsfix"};
}

} // namespace simaai::neat

namespace simaai::neat::nodes {

std::shared_ptr<simaai::neat::Node> H264CapsFixup(int fallback_fps, int fallback_width,
                                                  int fallback_height) {
  return std::make_shared<simaai::neat::H264CapsFixup>(fallback_fps, fallback_width,
                                                       fallback_height);
}

} // namespace simaai::neat::nodes
