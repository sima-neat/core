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

} // namespace

RUN_TEST(
    "unit_video_sender_test", ([] {
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
        require_in_order(
            graph.describe(),
            {"VideoConvert", "CapsRaw", "H264EncodeSima", "H264Parse", "H264Packetize",
             "UdpOutput"},
            "VideoSender raw path should include convert, caps, encode, parse, pay, udp");

        const std::string backend = graph.describe_backend();
        require_contains(backend,
                         "caps=\"video/x-raw,format=NV12,width=1280,height=720,"
                         "framerate=30/1\"",
                         "VideoSender raw caps mismatch");
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
        auto opt = VideoSenderOptions::H264RtpUdpFromEncoded();
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

      require_invalid_argument([] { (void)VideoSenderOptions::H264RtpUdpFromRaw(0, 720, 30); },
                               "VideoSender should reject invalid raw width");
      require_invalid_argument([] { (void)VideoSenderOptions::H264RtpUdpFromRaw(1280, 0, 30); },
                               "VideoSender should reject invalid raw height");
      require_invalid_argument([] { (void)VideoSenderOptions::H264RtpUdpFromRaw(1280, 720, 0); },
                               "VideoSender should reject invalid raw fps");
    }));
