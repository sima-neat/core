#include "nodes/common/MultipartJpegDemux.h"

#include <memory>
#include <sstream>
#include <string>
#include <utility>
#include <vector>

namespace simaai::neat {

MultipartJpegDemux::MultipartJpegDemux(MultipartJpegDemuxOptions opt) : opt_(std::move(opt)) {}

std::string MultipartJpegDemux::backend_fragment(int node_index) const {
  const std::string el = "n" + std::to_string(node_index) + "_multipartdemux";
  std::ostringstream ss;
  ss << "multipartdemux name=" << el;
  if (!opt_.boundary.empty()) {
    ss << " boundary=\"" << opt_.boundary << "\"";
  }
  if (opt_.single_stream) {
    ss << " single-stream=true";
  }
  return ss.str();
}

std::vector<std::string> MultipartJpegDemux::element_names(int node_index) const {
  return {"n" + std::to_string(node_index) + "_multipartdemux"};
}

} // namespace simaai::neat

namespace simaai::neat::nodes {

std::shared_ptr<simaai::neat::Node> MultipartJpegDemux(MultipartJpegDemuxOptions opt) {
  return std::make_shared<simaai::neat::MultipartJpegDemux>(std::move(opt));
}

} // namespace simaai::neat::nodes
