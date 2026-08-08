#include "nodes/common/MultipartJpegDemux.h"

#include "nodes/common/internal/MultipartHeaderCapture.h"

#include <memory>
#include <sstream>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace simaai::neat {

std::vector<std::string>
normalize_multipart_header_capture(const MultipartHeaderCaptureOptions& opt) {
  return multipart_internal::normalize_capture_names(opt.headers);
}

MultipartJpegDemux::MultipartJpegDemux(MultipartJpegDemuxOptions opt) : opt_(std::move(opt)) {
  // Validate eagerly so a bad allowlist fails at construction, not at graph build.
  capture_ = normalize_multipart_header_capture(opt_.header_capture);
}

std::string MultipartJpegDemux::backend_fragment(int node_index) const {
  std::ostringstream ss;
  if (capture_.empty()) {
    const std::string el = "n" + std::to_string(node_index) + "_multipartdemux";
    ss << "multipartdemux name=" << el;
    if (!opt_.boundary.empty()) {
      ss << " boundary=\"" << opt_.boundary << "\"";
    }
    if (opt_.single_stream) {
      ss << " single-stream=true";
    }
    return ss.str();
  }

  const std::string el = "n" + std::to_string(node_index) + "_neatmultipartjpegdemux";
  ss << "neatmultipartjpegdemux name=" << el;
  if (!opt_.boundary.empty()) {
    ss << " boundary=\"" << opt_.boundary << "\"";
  }
  ss << " capture-headers=\"" << multipart_internal::join_capture_names(capture_) << "\"";
  return ss.str();
}

std::vector<std::string> MultipartJpegDemux::element_names(int node_index) const {
  if (capture_.empty()) {
    return {"n" + std::to_string(node_index) + "_multipartdemux"};
  }
  return {"n" + std::to_string(node_index) + "_neatmultipartjpegdemux"};
}

} // namespace simaai::neat

namespace simaai::neat::nodes {

std::shared_ptr<simaai::neat::Node> MultipartJpegDemux(MultipartJpegDemuxOptions opt) {
  return std::make_shared<simaai::neat::MultipartJpegDemux>(std::move(opt));
}

} // namespace simaai::neat::nodes
