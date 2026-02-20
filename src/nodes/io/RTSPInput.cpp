#include "nodes/io/RTSPInput.h"

#include <memory>
#include <sstream>
#include <string>
#include <utility>
#include <vector>

namespace simaai::neat {

RTSPInput::RTSPInput(std::string url, int latency_ms, bool tcp, bool drop_on_latency,
                     std::string buffer_mode)
    : url_(std::move(url)), latency_ms_(latency_ms), tcp_(tcp), drop_on_latency_(drop_on_latency),
      buffer_mode_(std::move(buffer_mode)) {}

std::string RTSPInput::backend_fragment(int node_index) const {
  const std::string el = "n" + std::to_string(node_index) + "_rtspsrc";
  std::ostringstream ss;
  ss << "rtspsrc name=" << el << " location=\"" << url_ << "\" "
     << "latency=" << latency_ms_ << " "
     << "protocols=" << (tcp_ ? "tcp" : "udp");
  if (drop_on_latency_) {
    ss << " drop-on-latency=true";
  }
  if (!buffer_mode_.empty()) {
    ss << " buffer-mode=" << buffer_mode_;
  }
  return ss.str();
}

std::vector<std::string> RTSPInput::element_names(int node_index) const {
  return {"n" + std::to_string(node_index) + "_rtspsrc"};
}

} // namespace simaai::neat

namespace simaai::neat::nodes {

std::shared_ptr<simaai::neat::Node> RTSPInput(std::string url, int latency_ms, bool tcp,
                                              bool drop_on_latency, std::string buffer_mode) {
  return std::make_shared<simaai::neat::RTSPInput>(std::move(url), latency_ms, tcp, drop_on_latency,
                                                   std::move(buffer_mode));
}

} // namespace simaai::neat::nodes
