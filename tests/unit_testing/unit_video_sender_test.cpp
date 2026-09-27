#include "nodes/groups/VideoSender.h"
#include "test_main.h"

#include <functional>
#include <stdexcept>
#include <string>
#include <utility>
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
        require(backend.find("neatencoderinput") != std::string::npos,
                "standalone VideoSender should retain one safe format converter");
        require_contains(backend, "caps=\"video/x-raw,width=1280,height=720,framerate=30/1\"",
                         "VideoSender input raw caps mismatch");
        require_contains(backend,
                         "caps=\"video/x-raw,format=NV12,width=1280,height=720,framerate=30/1\"",
                         "VideoSender encoder raw caps mismatch");
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
        auto legacy = VideoSenderOptions::H264RtpUdpFromRaw(640, 360, 30);
        legacy.encoder.profile = "MAIN";
        legacy.encoder.level = "3.1";
        const auto backend = VideoSender(legacy).describe_backend();
        require_contains(backend, "enc-profile=MAIN", "legacy profile must pass through");
        require_contains(backend, "enc-level=3.1", "legacy level must pass through");

        simaai::neat::SimaEncodeOptions encode;
        encode.level = "3.1";
        require_invalid_argument([&] { (void)VideoSenderOptions::FromRaw(encode); },
                                 "new sender factory must still validate encoder levels");
        encode.level = "4.0";
        auto sender = VideoSenderOptions::FromRaw(encode);
        sender.encoder.level = "3.1";
        require_invalid_argument([&] { (void)VideoSender(sender); },
                                 "new sender must validate overrides at graph construction");
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

      const simaai::neat::SimaEncode defaults;
      require(defaults.options().type == simaai::neat::SimaEncodeType::H264 &&
                  !defaults.output_spec({}).has_shape() &&
                  defaults.caps_behavior() == simaai::neat::NodeCapsBehavior::Static,
              "default encoder must accept input geometry at runtime");
      require(simaai::neat::nodes::SimaEncode()->kind() == "SimaEncode",
              "default encoder factory must be usable");

      // Standalone and sender options resolve the same codec-specific settings.
      static_assert(simaai::neat::SimaEncodeType::AVC == simaai::neat::SimaEncodeType::H264);
      static_assert(simaai::neat::SimaEncodeType::HEVC == simaai::neat::SimaEncodeType::H265);
      for (auto type : {simaai::neat::SimaEncodeType::H264, simaai::neat::SimaEncodeType::H265,
                        simaai::neat::SimaEncodeType::MJPEG}) {
        simaai::neat::SimaEncodeOptions encode;
        encode.type = type;
        encode.fps = 30;
        encode.num_buffers = 4;
        const bool jpeg = type == simaai::neat::SimaEncodeType::MJPEG;
        if (jpeg) {
          encode.quality = 73;
        } else {
          encode.rate_control = "cbr";
          encode.bitrate_kbps = 2500;
          encode.gop_length = 10;
          encode.idr_interval = 30;
        }
        simaai::neat::SimaEncode node(encode);
        simaai::neat::OutputSpec input;
        input.certainty = simaai::neat::SpecCertainty::Derived;
        for (const auto& size : {std::pair{258, 130}, std::pair{640, 360}}) {
          input.width = size.first;
          input.height = size.second;
          input.layout = "Planar";
          input.dtype = "UInt8";
          input.byte_size = size.first * size.second * 3 / 2;
          const auto output = node.output_spec(input);
          require(output.payload_type == simaai::neat::PayloadType::Encoded &&
                      output.width == input.width && output.height == input.height &&
                      output.fps_num == 30 && output.fps_den == 1 && output.byte_size == 0 &&
                      output.layout.empty() && output.dtype.empty(),
                  "reusable encoder must follow input resolution without copying raw layout");
        }
        input.certainty = simaai::neat::SpecCertainty::Hint;
        const auto output = node.output_spec(input);
        require(output.certainty == simaai::neat::SpecCertainty::Hint,
                "encoder must not make tentative input dimensions authoritative");
        require(node.options().type == type, "encoder options accessor lost codec");
        const auto fragment = node.backend_fragment(7);
        require(fragment.find("enc-level=") == std::string::npos,
                "unset encoder level must be left to the backend");
        require_contains(fragment, "num-output-buffers=4", "output count override missing");
        require(fragment.find("enc-width=") == std::string::npos &&
                    fragment.find("enc-height=") == std::string::npos,
                "encoder must negotiate resolution from the input");
        const auto first_adapter = fragment.find("neatencoderinput");
        require(first_adapter != std::string::npos &&
                    fragment.find("neatencoderinput", first_adapter + 1) == std::string::npos,
                "standalone encoder must prepare input exactly once");
        if (jpeg) {
          require(output.media_type == "image/jpeg", "JPEG output media mismatch");
          require(fragment.find("enc-bitrate=") == std::string::npos,
                  "JPEG must not inherit video defaults");
          require_contains(fragment, "enc-quality=73", "JPEG quality missing");
        } else {
          require_contains(fragment, "enc-bitrate-mode=cbr", "rate control missing");
          require_contains(fragment, "enc-gop-length=10", "GOP missing");
          require_contains(fragment, "enc-idr-interval=30", "IDR must be independent of GOP");
        }
        auto sender = VideoSenderOptions::FromRaw(encode);
        encode.fps = 60;
        require(sender.width() == 0 && sender.height() == 0 && sender.fps() == 30 &&
                    sender.is_raw_input(),
                "sender must own encoding options and infer input dimensions");
        const auto sender_fragment = VideoSender(sender).describe_backend();
        require(sender_fragment.find("enc-level=") == std::string::npos,
                "unset sender level must be left to the backend");
        const auto sender_adapter = sender_fragment.find("neatencoderinput");
        require(sender_adapter != std::string::npos &&
                    sender_fragment.find("neatencoderinput", sender_adapter + 1) ==
                        std::string::npos,
                "sender must prepare input exactly once");
        if (!jpeg) {
          auto explicit_encode = encode;
          explicit_encode.level = "4.2";
          require_contains(simaai::neat::SimaEncode(explicit_encode).backend_fragment(0),
                           "enc-level=4.2", "explicit encoder level missing");
          require_contains(
              VideoSender(VideoSenderOptions::FromRaw(explicit_encode)).describe_backend(),
              "enc-level=4.2", "explicit sender level missing");
          sender.encoder = {.bitrate_kbps = 3000, .profile = "main", .level = "4.1"};
          const auto changed = VideoSender(sender).describe_backend();
          require_contains(changed, "enc-bitrate=3000", "legacy bitrate override lost");
          require_contains(changed, "enc-profile=main", "legacy profile override lost");
          require_contains(changed, "enc-level=4.1", "legacy level override lost");
        } else {
          sender.encoder.bitrate_kbps = 3000;
          require_invalid_argument([&] { (void)VideoSender(sender); },
                                   "JPEG must reject video-only overrides");
        }
      }
      {
        simaai::neat::SimaEncodeOptions base;
        for (int buffers : {0, -2, 21}) {
          auto invalid = base;
          invalid.num_buffers = buffers;
          require_invalid_argument([&] { (void)simaai::neat::SimaEncode(invalid); },
                                   "invalid output count accepted");
        }
        auto invalid = base;
        invalid.fps = 0;
        require_invalid_argument([&] { (void)simaai::neat::SimaEncode(invalid); },
                                 "invalid encoder fps accepted");
        invalid = base;
        invalid.quality = 80;
        require_invalid_argument([&] { (void)simaai::neat::SimaEncode(invalid); },
                                 "video codec accepted JPEG quality");
        invalid = base;
        invalid.type = simaai::neat::SimaEncodeType::MJPEG;
        invalid.gop_length = 0;
        require_invalid_argument([&] { (void)VideoSenderOptions::FromRaw(invalid); },
                                 "JPEG accepted explicitly supplied GOP zero");
        invalid = base;
        invalid.type = simaai::neat::SimaEncodeType::H265;
        invalid.profile = "baseline";
        require_invalid_argument([&] { (void)simaai::neat::SimaEncode(invalid); },
                                 "H265 accepted H264-only profile");
      }
      for (const auto& size : {std::pair{641, 360}, std::pair{3842, 720}, std::pair{640, 2162}}) {
        require_invalid_argument(
            [&] {
              (void)VideoSender(VideoSenderOptions::H264RtpUdpFromRaw(size.first, size.second, 30));
            },
            "legacy sender must retain its fixed geometry validation");
      }
      require_invalid_argument([] { (void)VideoSenderOptions::H264RtpUdpFromRaw(0, 720, 30); },
                               "VideoSender should reject invalid raw width");
      require_invalid_argument([] { (void)VideoSenderOptions::H264RtpUdpFromRaw(1280, 0, 30); },
                               "VideoSender should reject invalid raw height");
      require_invalid_argument([] { (void)VideoSenderOptions::H264RtpUdpFromRaw(1280, 720, 0); },
                               "VideoSender should reject invalid raw fps");
      require_invalid_argument(
          [] { (void)VideoSenderOptions::Passthrough(static_cast<RtspCodec>(-1)); },
          "VideoSender should reject an unsupported codec");
    }));
