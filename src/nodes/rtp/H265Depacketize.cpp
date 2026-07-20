#include "nodes/rtp/H265Depacketize.h"

#include <memory>
#include <sstream>
#include <string>
#include <vector>

namespace simaai::neat {

H265Depacketize::H265Depacketize(int payload_type, int source_fps)
    : payload_type_(payload_type), source_fps_(source_fps) {}

std::string H265Depacketize::backend_fragment(int node_index) const {
  const std::string rtp = "n" + std::to_string(node_index) + "_rtp_caps";
  const std::string depay = "n" + std::to_string(node_index) + "_depay";
  const std::string parser = "n" + std::to_string(node_index) + "_h265parse";
  const std::string caps = "n" + std::to_string(node_index) + "_h265_caps";

  std::ostringstream ss;
  ss << "capsfilter name=" << rtp << " caps=\"application/x-rtp,media=video,encoding-name=H265";
  if (payload_type_ > 0) {
    ss << ",payload=" << payload_type_;
  }
  ss << "\" ! rtph265depay name=" << depay << " ! h265parse name=" << parser
     << " disable-passthrough=true ! capsfilter name=" << caps
     << " caps=\"video/x-h265,parsed=true,stream-format=(string)byte-stream,"
        "alignment=(string)au";
  if (source_fps_ > 0) {
    ss << ",framerate=(fraction)" << source_fps_ << "/1";
  }
  ss << "\"";
  return ss.str();
}

std::vector<std::string> H265Depacketize::element_names(int node_index) const {
  return {
      "n" + std::to_string(node_index) + "_rtp_caps",
      "n" + std::to_string(node_index) + "_depay",
      "n" + std::to_string(node_index) + "_h265parse",
      "n" + std::to_string(node_index) + "_h265_caps",
  };
}

OutputSpec H265Depacketize::output_spec(const OutputSpec& /*input*/) const {
  OutputSpec out;
  out.media_type = "video/x-h265";
  out.format = "H265";
  if (source_fps_ > 0) {
    out.fps_num = source_fps_;
    out.fps_den = 1;
  }
  out.certainty = SpecCertainty::Hint;
  out.note = "RTP depay -> H265";
  return out;
}

} // namespace simaai::neat

namespace simaai::neat::nodes {

std::shared_ptr<simaai::neat::Node> H265Depacketize(int payload_type, int source_fps) {
  return std::make_shared<simaai::neat::H265Depacketize>(payload_type, source_fps);
}

} // namespace simaai::neat::nodes
