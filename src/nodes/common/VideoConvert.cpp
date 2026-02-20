#include "nodes/common/VideoConvert.h"

#include <memory>
#include <string>
#include <vector>

namespace simaai::neat {

std::string VideoConvert::backend_fragment(int node_index) const {
  return "videoconvert name=n" + std::to_string(node_index) + "_videoconvert";
}

std::vector<std::string> VideoConvert::element_names(int node_index) const {
  return {"n" + std::to_string(node_index) + "_videoconvert"};
}

} // namespace simaai::neat

namespace simaai::neat::nodes {

std::shared_ptr<simaai::neat::Node> VideoConvert() {
  return std::make_shared<simaai::neat::VideoConvert>();
}

} // namespace simaai::neat::nodes
