#include "nodes/groups/VideoSender.h"

#include "nodes/common/Caps.h"
#include "nodes/common/VideoConvert.h"
#include "nodes/io/UdpOutput.h"
#include "nodes/sima/H264EncodeSima.h"
#include "nodes/sima/H264Packetize.h"
#include "nodes/sima/H264Parse.h"

#include <memory>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace simaai::neat::nodes::groups {
namespace {

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

simaai::neat::Graph VideoSender(const VideoSenderOptions& opt) {
  std::vector<std::shared_ptr<simaai::neat::Node>> nodes;
  nodes.reserve(opt.is_raw_input() ? 6 : 3);

  if (opt.is_raw_input()) {
    nodes.push_back(nodes::VideoConvert());
    nodes.push_back(
        nodes::CapsRaw("NV12", opt.width(), opt.height(), -1, simaai::neat::CapsMemory::Any));
    nodes.push_back(nodes::H264EncodeSima(opt.width(), opt.height(), opt.fps(),
                                          opt.encoder.bitrate_kbps, opt.encoder.profile,
                                          opt.encoder.level));
  }

  append_h264_rtp_udp_nodes(nodes, opt);

  simaai::neat::Graph graph("video_sender");
  for (auto& node : nodes) {
    graph.add(std::move(node));
  }
  return graph;
}

} // namespace simaai::neat::nodes::groups
