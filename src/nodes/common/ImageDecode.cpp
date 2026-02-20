#include "nodes/common/ImageDecode.h"

#include <memory>
#include <string>
#include <vector>

namespace simaai::neat {

std::string ImageDecode::backend_fragment(int node_index) const {
  return "decodebin name=n" + std::to_string(node_index) + "_decodebin";
}

std::vector<std::string> ImageDecode::element_names(int node_index) const {
  return {std::string("n") + std::to_string(node_index) + "_decodebin"};
}

} // namespace simaai::neat

namespace simaai::neat::nodes {

std::shared_ptr<simaai::neat::Node> ImageDecode() {
  return std::make_shared<simaai::neat::ImageDecode>();
}

} // namespace simaai::neat::nodes
