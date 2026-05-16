#include "nodes/groups/VideoSender.h"
#include "test_main.h"

#include <functional>
#include <stdexcept>
#include <string>

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

        const auto group = VideoSender(opt);
        const auto& nodes = group.nodes();
        require(nodes.size() == 5,
                "VideoSender raw path should include convert, encode, parse, pay, udp");
        require(nodes[0]->kind() == "VideoConvert", "VideoSender raw node[0] mismatch");
        require(nodes[1]->kind() == "H264EncodeSima", "VideoSender raw node[1] mismatch");
        require(nodes[2]->kind() == "H264Parse", "VideoSender raw node[2] mismatch");
        require(nodes[3]->kind() == "H264Packetize", "VideoSender raw node[3] mismatch");
        require(nodes[4]->kind() == "UdpOutput", "VideoSender raw node[4] mismatch");

        const std::string encoder_fragment = nodes[1]->backend_fragment(1);
        require_contains(encoder_fragment, "enc-width=1280", "VideoSender encoder width mismatch");
        require_contains(encoder_fragment, "enc-height=720", "VideoSender encoder height mismatch");
        require_contains(encoder_fragment, "enc-frame-rate=30", "VideoSender encoder fps mismatch");
        require_contains(encoder_fragment, "enc-bitrate=2500",
                         "VideoSender encoder bitrate mismatch");
        require_contains(encoder_fragment, "enc-profile=main",
                         "VideoSender encoder profile mismatch");
        require_contains(encoder_fragment, "enc-level=4.1", "VideoSender encoder level mismatch");

        const std::string pay_fragment = nodes[3]->backend_fragment(3);
        require_contains(pay_fragment, "pt=97", "VideoSender RTP payload type mismatch");
        require_contains(pay_fragment, "config-interval=2",
                         "VideoSender RTP config interval mismatch");

        const std::string udp_fragment = nodes[4]->backend_fragment(4);
        require_contains(udp_fragment, "host=10.0.0.5", "VideoSender UDP host mismatch");
        require_contains(udp_fragment, "port=9000", "VideoSender UDP port mismatch");
        require(opt.video_port() == 9000, "VideoSender computed video port mismatch");
      }

      {
        auto opt = VideoSenderOptions::H264RtpUdpFromEncoded();
        opt.channel = 1;
        opt.video_port_base = 9000;
        opt.rtp.payload_type = 98;

        const auto group = VideoSender(opt);
        const auto& nodes = group.nodes();
        require(nodes.size() == 3, "VideoSender encoded path should include parse, pay, udp only");
        require(nodes[0]->kind() == "H264Parse", "VideoSender encoded node[0] mismatch");
        require(nodes[1]->kind() == "H264Packetize", "VideoSender encoded node[1] mismatch");
        require(nodes[2]->kind() == "UdpOutput", "VideoSender encoded node[2] mismatch");

        const std::string pay_fragment = nodes[1]->backend_fragment(1);
        require_contains(pay_fragment, "pt=98", "VideoSender encoded RTP payload type mismatch");
        const std::string udp_fragment = nodes[2]->backend_fragment(2);
        require_contains(udp_fragment, "port=9001", "VideoSender encoded UDP port mismatch");
      }

      require_invalid_argument([] { (void)VideoSenderOptions::H264RtpUdpFromRaw(0, 720, 30); },
                               "VideoSender should reject invalid raw width");
      require_invalid_argument([] { (void)VideoSenderOptions::H264RtpUdpFromRaw(1280, 0, 30); },
                               "VideoSender should reject invalid raw height");
      require_invalid_argument([] { (void)VideoSenderOptions::H264RtpUdpFromRaw(1280, 720, 0); },
                               "VideoSender should reject invalid raw fps");
    }));
