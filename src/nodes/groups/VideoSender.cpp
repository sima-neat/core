#include "nodes/groups/VideoSender.h"

#include "nodes/common/Caps.h"
#include "nodes/common/VideoConvert.h"
#include "nodes/io/UdpOutput.h"
#include "nodes/sima/H264EncodeSima.h"
#include "nodes/sima/H264Packetize.h"
#include "nodes/sima/H264Parse.h"

#include <memory>
#include <sstream>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace simaai::neat::nodes::groups {
namespace {

class H265ParseNode final : public simaai::neat::Node, public simaai::neat::OutputSpecProvider {
public:
  explicit H265ParseNode(int config_interval) : config_interval_(config_interval) {}

  std::string kind() const override {
    return "H265Parse";
  }
  NodeCapsBehavior caps_behavior() const override {
    return NodeCapsBehavior::Dynamic;
  }
  std::string backend_fragment(int node_index) const override {
    return "h265parse name=n" + std::to_string(node_index) +
           "_h265parse disable-passthrough=true config-interval=" +
           std::to_string(config_interval_);
  }
  std::vector<std::string> element_names(int node_index) const override {
    return {"n" + std::to_string(node_index) + "_h265parse"};
  }
  simaai::neat::OutputSpec output_spec(const simaai::neat::OutputSpec&) const override {
    simaai::neat::OutputSpec out;
    out.media_type = "video/x-h265";
    out.format = "H265";
    out.certainty = simaai::neat::SpecCertainty::Hint;
    out.note = "H265 parse output";
    return out;
  }

private:
  int config_interval_ = 1;
};

class H265PacketizeNode final : public simaai::neat::Node, public simaai::neat::OutputSpecProvider {
public:
  H265PacketizeNode(int payload_type, int config_interval)
      : payload_type_(payload_type), config_interval_(config_interval) {}

  std::string kind() const override {
    return "H265Packetize";
  }
  NodeCapsBehavior caps_behavior() const override {
    return NodeCapsBehavior::Dynamic;
  }
  std::string backend_fragment(int) const override {
    std::ostringstream fragment;
    fragment << "rtph265pay name=pay0 pt=" << payload_type_
             << " config-interval=" << config_interval_ << " timestamp-offset=0";
    return fragment.str();
  }
  std::vector<std::string> element_names(int) const override {
    return {"pay0"};
  }
  simaai::neat::OutputSpec output_spec(const simaai::neat::OutputSpec&) const override {
    simaai::neat::OutputSpec out;
    out.media_type = "application/x-rtp";
    out.format = "H265";
    out.certainty = simaai::neat::SpecCertainty::Hint;
    out.note = "RTP H265 payload";
    return out;
  }

private:
  int payload_type_ = 98;
  int config_interval_ = 1;
};

void require_positive(int value, const char* name) {
  if (value <= 0) {
    throw std::invalid_argument(std::string("VideoSenderOptions: ") + name + " must be > 0");
  }
}

void append_h264_rtp_udp_nodes(std::vector<std::shared_ptr<simaai::neat::Node>>& nodes,
                               const VideoSenderOptions& opt) {
  nodes.push_back(nodes::H264Parse(opt.rtp.config_interval));
  nodes.push_back(
      nodes::H264Packetize(simaai::neat::H264Packetize::PayloadType(opt.rtp.payload_type),
                           simaai::neat::H264Packetize::ConfigInterval(opt.rtp.config_interval)));

  simaai::neat::UdpOutputOptions udp_opt;
  udp_opt.host = opt.host;
  udp_opt.port = opt.video_port();
  udp_opt.sync = opt.sync;
  udp_opt.async = opt.async;
  nodes.push_back(nodes::UdpOutput(udp_opt));
}

void append_h265_rtp_udp_nodes(std::vector<std::shared_ptr<simaai::neat::Node>>& nodes,
                               const VideoSenderOptions& opt) {
  nodes.push_back(std::make_shared<H265ParseNode>(opt.rtp.config_interval));
  nodes.push_back(
      std::make_shared<H265PacketizeNode>(opt.rtp.payload_type, opt.rtp.config_interval));

  simaai::neat::UdpOutputOptions udp_opt;
  udp_opt.host = opt.host;
  udp_opt.port = opt.video_port();
  udp_opt.sync = opt.sync;
  udp_opt.async = opt.async;
  nodes.push_back(nodes::UdpOutput(udp_opt));
}

} // namespace

VideoSenderOptions VideoSenderOptions::H264RtpUdpFromRaw(int width, int height, int fps) {
  require_positive(width, "width");
  require_positive(height, "height");
  require_positive(fps, "fps");

  VideoSenderOptions opt;
  opt.input_kind_ = InputKind::Raw;
  opt.width_ = width;
  opt.height_ = height;
  opt.fps_ = fps;
  return opt;
}

VideoSenderOptions VideoSenderOptions::H264RtpUdpFromEncoded() {
  VideoSenderOptions opt;
  opt.input_kind_ = InputKind::EncodedH264;
  return opt;
}

VideoSenderOptions VideoSenderOptions::H265RtpUdpFromEncoded() {
  VideoSenderOptions opt;
  opt.input_kind_ = InputKind::EncodedH265;
  opt.rtp.payload_type = 98;
  return opt;
}

simaai::neat::Graph VideoSender(const VideoSenderOptions& opt) {
  std::vector<std::shared_ptr<simaai::neat::Node>> nodes;
  nodes.reserve(opt.is_raw_input() ? 7 : 3);

  if (opt.is_raw_input()) {
    nodes.push_back(
        nodes::CapsRaw("", opt.width(), opt.height(), opt.fps(), simaai::neat::CapsMemory::Any));
    nodes.push_back(nodes::VideoConvert());
    nodes.push_back(nodes::CapsRaw("NV12", opt.width(), opt.height(), opt.fps(),
                                   simaai::neat::CapsMemory::Any));
    nodes.push_back(nodes::H264EncodeSima(opt.width(), opt.height(), opt.fps(),
                                          opt.encoder.bitrate_kbps, opt.encoder.profile,
                                          opt.encoder.level));
  }

  if (opt.input_kind_ == VideoSenderOptions::InputKind::EncodedH265) {
    append_h265_rtp_udp_nodes(nodes, opt);
  } else {
    append_h264_rtp_udp_nodes(nodes, opt);
  }

  simaai::neat::Graph graph("video_sender");
  for (auto& node : nodes) {
    graph.add(std::move(node));
  }
  return graph;
}

} // namespace simaai::neat::nodes::groups
