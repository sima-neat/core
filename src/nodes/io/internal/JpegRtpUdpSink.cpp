#include "nodes/io/internal/JpegRtpUdpSink.h"

#include <memory>
#include <sstream>
#include <string>
#include <vector>

namespace simaai::neat::internal {

std::string JpegRtpUdpSink::backend_fragment(int node_index) const {
  const std::string name = "n" + std::to_string(node_index);
  std::ostringstream ss;
  if (input_ == Input::ParseDimensions) {
    ss << "jpegparse name=" << name << "_jpegparse disable-passthrough=true ! ";
  }
  // Like UdpOutput: a retained last sample would keep the frame's capture buffer referenced.
  ss << kJpegRtpUdpSinkFactory << " name=" << name << "_jpegudp host=" << opt_.host
     << " port=" << opt_.port << " pt=" << opt_.payload_type << " mtu=" << opt_.mtu
     << " timestamp-offset=0 sync=" << (opt_.sync ? "true" : "false")
     << " async=" << (opt_.async ? "true" : "false") << " enable-last-sample=false";
  return ss.str();
}

std::vector<std::string> JpegRtpUdpSink::element_names(int node_index) const {
  const std::string name = "n" + std::to_string(node_index);
  if (input_ == Input::ParseDimensions) {
    return {name + "_jpegparse", name + "_jpegudp"};
  }
  return {name + "_jpegudp"};
}

std::shared_ptr<Node>
JpegRtpUdpSink::specialize_for_input(const OutputSpec& input,
                                     const InputSpecSpecializationContext&) const {
  const bool sized = input.width > 0 && input.height > 0;
  return std::make_shared<JpegRtpUdpSink>(opt_, sized ? Input::SizedCaps : Input::ParseDimensions);
}

} // namespace simaai::neat::internal
