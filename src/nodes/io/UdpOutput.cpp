#include "nodes/io/UdpOutput.h"

#include "pipeline/ErrorCodes.h"
#include "pipeline/internal/ErrorUtil.h"

#include <memory>
#include <netdb.h>
#include <sstream>
#include <string>
#include <vector>

namespace simaai::neat {

UdpOutput::UdpOutput(UdpOutputOptions opt) : opt_(std::move(opt)) {
  using pipeline_internal::error_util::throw_session_error;
  if (opt_.port < 1 || opt_.port > 65535) {
    throw_session_error(error_codes::kOptionOutOfRange, "UdpOutput.port must be in [1, 65535]");
  }
  addrinfo hints{};
  hints.ai_socktype = SOCK_DGRAM;
  addrinfo* address = nullptr;
  const int status = getaddrinfo(opt_.host.c_str(), nullptr, &hints, &address);
  if (status != 0) {
    throw_session_error(error_codes::kOptionOutOfRange,
                        "UdpOutput.host '" + opt_.host +
                            "' must resolve to a UDP destination: " + gai_strerror(status));
  }
  freeaddrinfo(address);
}

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
