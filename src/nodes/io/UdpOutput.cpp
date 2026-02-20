#include "nodes/io/UdpOutput.h"

#include <memory>
#include <sstream>
#include <string>
#include <vector>

namespace simaai::neat {

std::string UdpOutput::backend_fragment(int /*node_index*/) const {
  std::ostringstream ss;
  ss << "udpsink host=" << opt_.host << " port=" << opt_.port
     << " sync=" << (opt_.sync ? "true" : "false") << " async=" << (opt_.async ? "true" : "false");
  return ss.str();
}

std::vector<std::string> UdpOutput::element_names(int /*node_index*/) const {
  return {};
}

} // namespace simaai::neat

namespace simaai::neat::nodes {

std::shared_ptr<simaai::neat::Node> UdpOutput(UdpOutputOptions opt) {
  return std::make_shared<simaai::neat::UdpOutput>(std::move(opt));
}

} // namespace simaai::neat::nodes
