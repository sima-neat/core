#include "nodes/rtp/RTPJpegDepacketize.h"

#include "pipeline/PayloadType.h"

#include <memory>
#include <sstream>
#include <string>
#include <vector>

namespace simaai::neat {

RTPJpegDepacketize::RTPJpegDepacketize(int payload_type) : payload_type_(payload_type) {}

std::string RTPJpegDepacketize::backend_fragment(int node_index) const {
  const std::string rtp = "n" + std::to_string(node_index) + "_rtp_jpeg_caps";
  const std::string dep = "n" + std::to_string(node_index) + "_rtpjpegdepay";

  std::ostringstream ss;
  ss << "capsfilter name=" << rtp
     << " caps=\"application/x-rtp,media=video,encoding-name=JPEG,clock-rate=90000";
  if (payload_type_ > 0) {
    ss << ",payload=" << payload_type_;
  }
  ss << "\" ! rtpjpegdepay name=" << dep;
  return ss.str();
}

std::vector<std::string> RTPJpegDepacketize::element_names(int node_index) const {
  return {
      "n" + std::to_string(node_index) + "_rtp_jpeg_caps",
      "n" + std::to_string(node_index) + "_rtpjpegdepay",
  };
}

OutputSpec RTPJpegDepacketize::output_spec(const OutputSpec& input) const {
  OutputSpec out;
  out.payload_type = PayloadType::Encoded;
  out.media_type = "image/jpeg";
  out.format = "JPEG";
  out.width = input.width;
  out.height = input.height;
  out.fps_num = input.fps_num;
  out.fps_den = input.fps_den;
  out.certainty = SpecCertainty::Hint;
  out.note = "RTP JPEG depay";
  return out;
}

} // namespace simaai::neat

namespace simaai::neat::nodes {

std::shared_ptr<simaai::neat::Node> RTPJpegDepacketize(int payload_type) {
  return std::make_shared<simaai::neat::RTPJpegDepacketize>(payload_type);
}

} // namespace simaai::neat::nodes
