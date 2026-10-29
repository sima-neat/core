#include "nodes/io/UdpOutput.h"

#include <memory>
#include <sstream>
#include <string>
#include <vector>

namespace simaai::neat {

std::string UdpOutput::backend_fragment(int /*node_index*/) const {
  std::ostringstream ss;
  // A retained last packet shares its frame's memory (RTP payloaders slice the input), so the
  // frame would return to its capture pool still referenced instead of being reused.
  ss << "udpsink host=" << opt_.host << " port=" << opt_.port
     << " sync=" << (opt_.sync ? "true" : "false") << " async=" << (opt_.async ? "true" : "false")
     << " enable-last-sample=false";
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
