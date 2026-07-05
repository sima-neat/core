#include "nodes/rtp/H264Depacketize.h"

#include <memory>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace simaai::neat {

H264Depacketize::H264Depacketize(int payload_type, int h264_parse_config_interval, int h264_fps,
                                 int h264_width, int h264_height, bool enforce_h264_caps)
    : payload_type_(payload_type), h264_parse_config_interval_(h264_parse_config_interval),
      h264_fps_(h264_fps), h264_width_(h264_width), h264_height_(h264_height),
      enforce_h264_caps_(enforce_h264_caps) {}

std::string H264Depacketize::backend_fragment(int node_index) const {
  const std::string rtp = "n" + std::to_string(node_index) + "_rtp_caps";
  const std::string dep = "n" + std::to_string(node_index) + "_depay";
  const std::string hseg = "n" + std::to_string(node_index) + "_h264_segment";
  const std::string par = "n" + std::to_string(node_index) + "_h264parse";
  const std::string hcc = "n" + std::to_string(node_index) + "_h264_caps";

  std::ostringstream ss;
  ss << "capsfilter name=" << rtp << " caps=\"application/x-rtp,media=video,encoding-name=H264";
  if (payload_type_ > 0) {
    ss << ",payload=" << payload_type_;
  }
  ss << "\" " << "! rtph264depay name=" << dep << " wait-for-keyframe=true "
     << "! identity name=" << hseg << " silent=true single-segment=true "
     << "! h264parse name=" << par << " disable-passthrough=true ";
  if (h264_parse_config_interval_ >= 0) {
    ss << "config-interval=" << h264_parse_config_interval_ << " ";
  }
  ss << "! capsfilter name=" << hcc
     << " caps=\"video/x-h264,parsed=true,stream-format=(string)byte-stream,alignment=(string)au";
  if (enforce_h264_caps_) {
    const bool has_any_caps = h264_width_ > 0 || h264_height_ > 0 || h264_fps_ > 0;
    const bool has_all_caps = h264_width_ > 0 && h264_height_ > 0 && h264_fps_ > 0;
    if (has_any_caps && !has_all_caps) {
      throw std::runtime_error(
          "H264Depacketize: enforced H.264 caps require width, height, and fps");
    }
    if (has_all_caps) {
      ss << ",width=(int)" << h264_width_ << ",height=(int)" << h264_height_;
      ss << ",framerate=(fraction)" << h264_fps_ << "/1";
    }
  }
  ss << "\"";
  return ss.str();
}

std::vector<std::string> H264Depacketize::element_names(int node_index) const {
  return {
      "n" + std::to_string(node_index) + "_rtp_caps",
      "n" + std::to_string(node_index) + "_depay",
      "n" + std::to_string(node_index) + "_h264_segment",
      "n" + std::to_string(node_index) + "_h264parse",
      "n" + std::to_string(node_index) + "_h264_caps",
  };
}

OutputSpec H264Depacketize::output_spec(const OutputSpec& /*input*/) const {
  OutputSpec out;
  out.media_type = "video/x-h264";
  out.format = "H264";
  if (enforce_h264_caps_) {
    const bool has_any_caps = h264_width_ > 0 || h264_height_ > 0 || h264_fps_ > 0;
    const bool has_all_caps = h264_width_ > 0 && h264_height_ > 0 && h264_fps_ > 0;
    if (has_any_caps && !has_all_caps) {
      throw std::runtime_error(
          "H264Depacketize: enforced H.264 caps require width, height, and fps");
    }
    if (has_all_caps) {
      out.width = h264_width_;
      out.height = h264_height_;
      out.fps_num = h264_fps_;
      out.fps_den = 1;
    }
  }
  out.certainty = SpecCertainty::Hint;
  out.note = "RTP depay -> H264";
  return out;
}

} // namespace simaai::neat

namespace simaai::neat::nodes {

std::shared_ptr<simaai::neat::Node> H264Depacketize(int payload_type,
                                                    int h264_parse_config_interval, int h264_fps,
                                                    int h264_width, int h264_height,
                                                    bool enforce_h264_caps) {
  return std::make_shared<simaai::neat::H264Depacketize>(payload_type, h264_parse_config_interval,
                                                         h264_fps, h264_width, h264_height,
                                                         enforce_h264_caps);
}

} // namespace simaai::neat::nodes
