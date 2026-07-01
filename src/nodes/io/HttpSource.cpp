#include "nodes/io/HttpSource.h"

#include <memory>
#include <sstream>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace simaai::neat {

HttpSource::HttpSource(HttpSourceOptions opt) : opt_(std::move(opt)) {
  if (opt_.location.empty()) {
    throw std::invalid_argument("HttpSource: location must not be empty");
  }
}

std::string HttpSource::backend_fragment(int node_index) const {
  const std::string el = "n" + std::to_string(node_index) + "_souphttpsrc";
  std::ostringstream ss;
  ss << "souphttpsrc name=" << el << " location=\"" << opt_.location << "\""
     << " timeout=" << opt_.timeout_seconds << " retries=" << opt_.retries
     << " is-live=" << (opt_.is_live ? "true" : "false")
     << " do-timestamp=" << (opt_.do_timestamp ? "true" : "false");
  if (!opt_.user_agent.empty()) {
    ss << " user-agent=\"" << opt_.user_agent << "\"";
  }
  return ss.str();
}

std::vector<std::string> HttpSource::element_names(int node_index) const {
  return {"n" + std::to_string(node_index) + "_souphttpsrc"};
}

} // namespace simaai::neat

namespace simaai::neat::nodes {

std::shared_ptr<simaai::neat::Node> HttpSource(HttpSourceOptions opt) {
  return std::make_shared<simaai::neat::HttpSource>(std::move(opt));
}

} // namespace simaai::neat::nodes
