#include "nodes/groups/VideoSender.h"

#include "nodes/groups/internal/VideoSenderRawIngress.h"
#include "nodes/io/UdpOutput.h"
#include "nodes/sima/internal/SimaEncode.h"
#include "nodes/common/JpegParse.h"
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

// Bounds synchronized multi-stream IDR bursts without changing the board TX queue.
constexpr int kH265RtpPacketPacingUs = 250;

class H265ParseNode final : public simaai::neat::Node, public simaai::neat::OutputSpecProvider {
public:
  explicit H265ParseNode(int config_interval) : config_interval_(config_interval) {}

  // The kind() strings below are matched by name in ExecutionGraphCompiler.cpp
  // (encoded_video_sender_codec) and GraphBuildSource.cpp
  // (append_fused_node_fragment) to recognize this sender for encoded-output
  // fusion. Renaming either string silently disables fusion rather than failing
  // to compile; unit_fused_realtime_fast_path_options_test guards that.
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
  std::string backend_fragment(int node_index) const override {
    std::ostringstream fragment;
    fragment << "rtph265pay name=pay0 pt=" << payload_type_
             << " config-interval=" << config_interval_ << " timestamp-offset=0"
             << " ! identity name=n" << node_index
             << "_h265_packet_pacer silent=true sleep-time=" << kH265RtpPacketPacingUs;
    return fragment.str();
  }
  std::vector<std::string> element_names(int node_index) const override {
    return {"pay0", "n" + std::to_string(node_index) + "_h265_packet_pacer"};
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

class JpegPacketizeNode final : public Node, public OutputSpecProvider {
public:
  explicit JpegPacketizeNode(int payload_type) : payload_type_(payload_type) {}
  std::string kind() const override {
    return "JpegPacketize";
  }
  NodeCapsBehavior caps_behavior() const override {
    return NodeCapsBehavior::Dynamic;
  }
  std::string backend_fragment(int node_index) const override {
    return "rtpjpegpay name=n" + std::to_string(node_index) +
           "_jpegpay pt=" + std::to_string(payload_type_) + " timestamp-offset=0";
  }
  std::vector<std::string> element_names(int node_index) const override {
    return {"n" + std::to_string(node_index) + "_jpegpay"};
  }
  OutputSpec output_spec(const OutputSpec& input) const override {
    OutputSpec out = input;
    out.payload_type = PayloadType::Encoded;
    out.media_type = "application/x-rtp";
    out.format = "JPEG";
    out.byte_size = 0;
    out.certainty = SpecCertainty::Derived;
    out.note = "RTP JPEG payload";
    return out;
  }

private:
  int payload_type_;
};

void require_positive(int value, const char* name) {
  if (value <= 0) {
    throw std::invalid_argument(std::string("VideoSenderOptions: ") + name + " must be > 0");
  }
}

} // namespace

VideoSenderOptions VideoSenderOptions::FromRaw(SimaEncodeOptions encode) {
  simaai::neat::internal::validate_encode_options(encode);
  const auto codec = encode.type == SimaEncodeType::MJPEG  ? RtspCodec::MJPEG
                     : encode.type == SimaEncodeType::H265 ? RtspCodec::H265
                                                           : RtspCodec::H264;
  auto opt = Passthrough(codec);
  opt.input_kind_ = InputKind::Raw;
  opt.width_ = encode.width;
  opt.height_ = encode.height;
  opt.fps_ = encode.fps;
  if (codec != RtspCodec::MJPEG) {
    opt.encoder.bitrate_kbps = encode.bitrate_kbps.value_or(4000);
    opt.encoder.profile = encode.profile.value_or(codec == RtspCodec::H265 ? "main" : "baseline");
    opt.encoder.level = encode.level.value_or("4.0");
  }
  opt.rate_control_ = std::move(encode.rate_control);
  opt.gop_length_ = encode.gop_length;
  opt.idr_interval_ = encode.idr_interval;
  opt.quality_ = encode.quality;
  opt.num_buffers_ = encode.num_buffers;
  return opt;
}

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
  return Passthrough(RtspCodec::H264);
}

VideoSenderOptions VideoSenderOptions::Passthrough(RtspCodec codec) {
  if (codec != RtspCodec::H264 && codec != RtspCodec::H265 && codec != RtspCodec::MJPEG) {
    throw std::invalid_argument("VideoSenderOptions: codec not supported by Passthrough; only "
                                "H264, H265 and MJPEG can be forwarded as RTP");
  }

  VideoSenderOptions opt;
  opt.input_kind_ = InputKind::Encoded;
  opt.codec_ = codec;
  // Transmit-side defaults, deliberately independent of the RTSP input-side
  // defaults: Core sends H.265 as 98 while expecting to receive it as 96.
  opt.rtp.payload_type = codec == RtspCodec::MJPEG ? 26 : codec == RtspCodec::H265 ? 98 : 96;
  return opt;
}

simaai::neat::Graph VideoSender(const VideoSenderOptions& opt) {
  std::vector<std::shared_ptr<simaai::neat::Node>> nodes;
  nodes.reserve(opt.is_raw_input() ? 5 : 3);

  if (opt.rtp.payload_type < 0 || opt.rtp.payload_type > 127) {
    throw std::invalid_argument("VideoSender: RTP payload_type must be within 0..127");
  }
  if (opt.is_raw_input()) {
    SimaEncodeOptions encode;
    encode.type = opt.codec_ == RtspCodec::MJPEG  ? SimaEncodeType::MJPEG
                  : opt.codec_ == RtspCodec::H265 ? SimaEncodeType::H265
                                                  : SimaEncodeType::H264;
    encode.width = opt.width();
    encode.height = opt.height();
    encode.fps = opt.fps();
    encode.rate_control = opt.rate_control_;
    encode.gop_length = opt.gop_length_;
    encode.idr_interval = opt.idr_interval_;
    encode.quality = opt.quality_;
    encode.num_buffers = opt.num_buffers_;
    if (opt.codec_ == RtspCodec::MJPEG) {
      const VideoSenderEncoderOptions defaults;
      if (opt.encoder.bitrate_kbps != defaults.bitrate_kbps ||
          opt.encoder.profile != defaults.profile || opt.encoder.level != defaults.level) {
        throw std::invalid_argument(
            "VideoSender: MJPEG does not accept legacy video encoder overrides");
      }
    } else {
      encode.bitrate_kbps = opt.encoder.bitrate_kbps;
      encode.profile = opt.encoder.profile;
      // Native legacy callers may use the uppercase profile spellings.
      if (*encode.profile == "BASELINE")
        encode.profile = "baseline";
      else if (*encode.profile == "MAIN")
        encode.profile = "main";
      else if (*encode.profile == "HIGH")
        encode.profile = "high";
      encode.level = opt.encoder.level;
    }
    nodes.push_back(internal::VideoSenderRawIngress(opt.width(), opt.height(), opt.fps()));
    nodes.push_back(simaai::neat::internal::SimaEncodeAccess::prepared_input(std::move(encode)));
  }

  if (opt.codec_ == RtspCodec::MJPEG) {
    nodes.push_back(nodes::JpegParse());
    nodes.push_back(std::make_shared<JpegPacketizeNode>(opt.rtp.payload_type));
  } else if (opt.codec_ == RtspCodec::H265) {
    nodes.push_back(std::make_shared<H265ParseNode>(opt.rtp.config_interval));
    nodes.push_back(
        std::make_shared<H265PacketizeNode>(opt.rtp.payload_type, opt.rtp.config_interval));
  } else {
    nodes.push_back(nodes::H264Parse(opt.rtp.config_interval));
    nodes.push_back(
        nodes::H264Packetize(simaai::neat::H264Packetize::PayloadType(opt.rtp.payload_type),
                             simaai::neat::H264Packetize::ConfigInterval(opt.rtp.config_interval)));
  }

  simaai::neat::UdpOutputOptions udp_opt;
  udp_opt.host = opt.host;
  udp_opt.port = opt.video_port();
  udp_opt.sync = opt.sync;
  udp_opt.async = opt.async;
  nodes.push_back(nodes::UdpOutput(udp_opt));

  simaai::neat::Graph graph("video_sender");
  for (auto& node : nodes) {
    graph.add(std::move(node));
  }
  return graph;
}

} // namespace simaai::neat::nodes::groups
