#include "nodes/groups/VideoSender.h"
#include "test_main.h"

#include <functional>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

void require_invalid_argument(const std::function<void()>& fn, const std::string& msg) {
  try {
    fn();
  } catch (const std::invalid_argument&) {
    return;
  } catch (const std::exception& ex) {
    throw std::runtime_error(msg + " (unexpected exception: " + ex.what() + ")");
  }
  throw std::runtime_error(msg + " (no exception)");
}

void require_in_order(const std::string& text, const std::vector<std::string>& needles,
                      const std::string& msg) {
  std::size_t pos = 0;
  for (const auto& needle : needles) {
    const std::size_t found = text.find(needle, pos);
    if (found == std::string::npos) {
      throw std::runtime_error(msg + " (missing or out of order: " + needle + ")");
    }
    pos = found + needle.size();
  }
}

// The only intentional use of the deprecated factory. Isolated here because a
// diagnostic pragma cannot live inside the RUN_TEST macro argument.
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wdeprecated-declarations"
simaai::neat::nodes::groups::VideoSenderOptions deprecated_h264_encoded_options() {
  return simaai::neat::nodes::groups::VideoSenderOptions::H264RtpUdpFromEncoded();
}
#pragma GCC diagnostic pop

} // namespace

RUN_TEST(
    "unit_video_sender_test", ([] {
      using simaai::neat::nodes::groups::RtspCodec;
      using simaai::neat::nodes::groups::VideoSender;
      using simaai::neat::nodes::groups::VideoSenderOptions;

      {
        auto opt = VideoSenderOptions::H264RtpUdpFromRaw(1280, 720, 30);
        opt.host = "10.0.0.5";
        opt.channel = 3;
        opt.video_port_base = 8997;
        opt.rtp.payload_type = 97;
        opt.rtp.config_interval = 2;
        opt.encoder.bitrate_kbps = 2500;
        opt.encoder.profile = "main";
        opt.encoder.level = "4.1";

        const auto graph = VideoSender(opt);
        require_in_order(graph.describe(),
                         {"VideoSenderRawIngress[convert_to_nv12]", "H264EncodeSima", "H264Parse",
                          "H264Packetize", "UdpOutput"},
                         "standalone VideoSender should retain its safe raw-ingress fallback");

        const std::string backend = graph.describe_backend();
        require(backend.find("videoconvert") != std::string::npos,
                "standalone VideoSender should retain one safe format converter");
        require_contains(backend, "caps=\"video/x-raw,width=1280,height=720,framerate=30/1\"",
                         "VideoSender input raw caps mismatch");
        require_contains(backend,
                         "caps=\"video/x-raw,format=NV12,width=1280,height=720,framerate=30/1\"",
                         "VideoSender encoder raw caps mismatch");
        require_contains(backend, "enc-width=1280", "VideoSender encoder width mismatch");
        require_contains(backend, "enc-height=720", "VideoSender encoder height mismatch");
        require_contains(backend, "enc-frame-rate=30", "VideoSender encoder fps mismatch");
        require_contains(backend, "enc-bitrate=2500", "VideoSender encoder bitrate mismatch");
        require_contains(backend, "enc-profile=main", "VideoSender encoder profile mismatch");
        require_contains(backend, "enc-level=4.1", "VideoSender encoder level mismatch");

        require_contains(backend, "pt=97", "VideoSender RTP payload type mismatch");
        require_contains(backend, "config-interval=2", "VideoSender RTP config interval mismatch");

        require_contains(backend, "host=10.0.0.5", "VideoSender UDP host mismatch");
        require_contains(backend, "port=9000", "VideoSender UDP port mismatch");
        require(opt.video_port() == 9000, "VideoSender computed video port mismatch");
      }

      {
        auto opt = VideoSenderOptions::Passthrough(RtspCodec::H264);
        require(opt.rtp.payload_type == 96, "VideoSender H264 payload type default mismatch");
        opt.channel = 1;
        opt.video_port_base = 9000;
        opt.rtp.payload_type = 98;

        const auto graph = VideoSender(opt);
        require_in_order(graph.describe(), {"H264Parse", "H264Packetize", "UdpOutput"},
                         "VideoSender encoded path should include parse, pay, udp only");

        const std::string backend = graph.describe_backend();
        require_contains(backend, "pt=98", "VideoSender encoded RTP payload type mismatch");
        require_contains(backend, "port=9001", "VideoSender encoded UDP port mismatch");
      }

      {
        // The deprecated factory must stay a pure alias, so a caller that has
        // not migrated yet keeps byte-identical pipelines.
        const auto legacy = deprecated_h264_encoded_options();
        const auto migrated = VideoSenderOptions::Passthrough(RtspCodec::H264);
        require(legacy.rtp.payload_type == migrated.rtp.payload_type,
                "deprecated H264 encoded factory payload type drifted from Passthrough");
        // Element names embed a process-wide instance counter, so two separately
        // built graphs never stringify identically. Compare the node sequence and
        // the wire-visible properties instead, which is what "pure alias" means.
        require(VideoSender(legacy).describe() == VideoSender(migrated).describe(),
                "deprecated H264 encoded factory node sequence drifted from Passthrough");
        const std::string legacy_backend = VideoSender(legacy).describe_backend();
        require_contains(legacy_backend, "pt=96",
                         "deprecated H264 encoded factory payload type drifted from Passthrough");
        require_contains(legacy_backend, "port=9000",
                         "deprecated H264 encoded factory UDP port drifted from Passthrough");
      }

      {
        auto opt = VideoSenderOptions::Passthrough(RtspCodec::H265);
        opt.channel = 2;

        require(opt.rtp.payload_type == 98, "VideoSender H265 payload type default mismatch");
        require(!opt.is_raw_input() && opt.is_encoded_input(),
                "VideoSender H265 input kind mismatch");

        const auto graph = VideoSender(opt);
        require_in_order(graph.describe(), {"H265Parse", "H265Packetize", "UdpOutput"},
                         "VideoSender H265 path should include parse, pay, udp only");

        const std::string backend = graph.describe_backend();
        require_contains(backend, "h265parse", "VideoSender H265 parser missing");
        require_contains(backend, "rtph265pay", "VideoSender H265 packetizer missing");
        require_contains(backend, "pt=98", "VideoSender H265 RTP payload type mismatch");
        require_contains(backend, "sleep-time=250", "VideoSender H265 packet pacing missing");
        require_contains(backend, "port=9002", "VideoSender H265 UDP port mismatch");
        require(backend.find("neatencoder") == std::string::npos,
                "VideoSender H265 must not encode raw video");
        require(backend.find("h264parse") == std::string::npos,
                "VideoSender H265 must not use H264 parser");
      }

      require_invalid_argument([] { (void)VideoSenderOptions::H264RtpUdpFromRaw(0, 720, 30); },
                               "VideoSender should reject invalid raw width");
      require_invalid_argument([] { (void)VideoSenderOptions::H264RtpUdpFromRaw(1280, 0, 30); },
                               "VideoSender should reject invalid raw height");
      require_invalid_argument([] { (void)VideoSenderOptions::H264RtpUdpFromRaw(1280, 720, 0); },
                               "VideoSender should reject invalid raw fps");
      require_invalid_argument([] { (void)VideoSenderOptions::Passthrough(RtspCodec::MJPEG); },
                               "VideoSender should reject MJPEG passthrough");
    }));
