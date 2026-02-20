#include "nodes/common/VideoTrackSelect.h"

#include <memory>
#include <sstream>
#include <string>
#include <utility>
#include <vector>

namespace simaai::neat {

VideoTrackSelect::VideoTrackSelect(int video_pad_index) : idx_(video_pad_index) {}

std::string VideoTrackSelect::backend_fragment(int node_index) const {
  const std::string base = "n" + std::to_string(node_index) + "_demux";
  std::ostringstream ss;
  ss << "qtdemux name=" << base << " " << base << ".video_" << idx_;
  return ss.str();
}

std::vector<std::string> VideoTrackSelect::element_names(int node_index) const {
  return {"n" + std::to_string(node_index) + "_demux"};
}

} // namespace simaai::neat

namespace simaai::neat::nodes {

std::shared_ptr<simaai::neat::Node> VideoTrackSelect(int video_pad_index) {
  return std::make_shared<simaai::neat::VideoTrackSelect>(video_pad_index);
}

} // namespace simaai::neat::nodes
