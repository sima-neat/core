#include "nodes/common/JpegDecode.h"

#include <memory>
#include <string>
#include <vector>

namespace simaai::neat {

std::string JpegDecode::backend_fragment(int node_index) const {
  return "jpegdec name=n" + std::to_string(node_index) + "_jpegdec";
}

std::vector<std::string> JpegDecode::element_names(int node_index) const {
  return {"n" + std::to_string(node_index) + "_jpegdec"};
}

} // namespace simaai::neat

namespace simaai::neat::nodes {

std::shared_ptr<simaai::neat::Node> JpegDecode() {
  return std::make_shared<simaai::neat::JpegDecode>();
}

} // namespace simaai::neat::nodes
