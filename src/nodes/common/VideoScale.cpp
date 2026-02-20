#include "nodes/common/VideoScale.h"

#include <memory>
#include <string>
#include <vector>

namespace simaai::neat {

std::string VideoScale::backend_fragment(int node_index) const {
  return "videoscale name=n" + std::to_string(node_index) + "_videoscale";
}

std::vector<std::string> VideoScale::element_names(int node_index) const {
  return {"n" + std::to_string(node_index) + "_videoscale"};
}

} // namespace simaai::neat

namespace simaai::neat::nodes {

std::shared_ptr<simaai::neat::Node> VideoScale() {
  return std::make_shared<simaai::neat::VideoScale>();
}

} // namespace simaai::neat::nodes
