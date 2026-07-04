#include "nodes/groups/RtspDecodedInput.h"
#include "nodes/groups/RtspEncodedInput.h"
#include "nodes/groups/GroupOutputSpec.h"

#include "nodes/common/EncodedCapsFixup.h"
#include "gst/GstHelpers.h"
#include "nodes/common/JpegParse.h"
#include "nodes/common/Queue.h"
#include "nodes/io/RTSPInput.h"
#include "nodes/rtp/H264CapsFixup.h"
#include "nodes/rtp/H264Depacketize.h"
#include "nodes/rtp/RTPJpegDepacketize.h"
#include "nodes/sima/SimaDecode.h"
#include "pipeline/Graph.h"
#include "pipeline/PayloadType.h"
#include "pipeline/graph/internal/GraphTestHooks.h"

#include "test_utils.h"

#include <functional>
#include <iostream>
#include <memory>
#include <optional>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <utility>
#include <vector>

namespace {

using simaai::neat::Graph;

static_assert(std::is_same_v<
              decltype(simaai::neat::nodes::groups::RtspEncodedInput(
                  std::declval<const simaai::neat::nodes::groups::RtspEncodedInputOptions&>())),
              Graph>);

Graph graph_from_nodes(std::vector<std::shared_ptr<simaai::neat::Node>> nodes) {
  Graph graph;
  for (auto& node : nodes) {
    graph.add(std::move(node));
  }
  return graph;
}

void compare_graph_fragments(const Graph& actual, const Graph& expected, const std::string& label) {
  const std::string actual_text = actual.describe();
  const std::string expected_text = expected.describe();
  require(actual_text == expected_text,
          label + " mismatch\nactual:\n" + actual_text + "\nexpected:\n" + expected_text);
}

std::size_t count_substrings(const std::string& text, const std::string& needle) {
  std::size_t count = 0;
  std::size_t pos = 0;
  while ((pos = text.find(needle, pos)) != std::string::npos) {
    ++count;
    pos += needle.size();
  }
  return count;
}

void require_not_contains(const std::string& haystack, const std::string& needle,
                          const std::string& msg) {
  if (haystack.find(needle) != std::string::npos) {
    throw std::runtime_error(msg + " (unexpected: " + needle + ")");
  }
}

void require_throws_with(const std::function<void()>& fn, const std::string& needle,
                         const std::string& label) {
  try {
    fn();
  } catch (const std::exception& e) {
    require_contains(e.what(), needle, label);
    return;
  }
  throw std::runtime_error(label + ": expected exception");
}

std::optional<std::string> describe_backend_if_available(const Graph& graph,
                                                         const std::string& label) {
  try {
    return graph.describe_backend(false);
  } catch (const std::exception& e) {
    std::cout << "[INFO] " << label << ": backend checks skipped: " << e.what() << "\n";
    return std::nullopt;
  }
}

simaai::neat::nodes::groups::RtspEncodedInputOptions make_h264_encoded_options() {
  simaai::neat::nodes::groups::RtspEncodedInputOptions opt;
  opt.url = "rtsp://example.local/h264";
  opt.latency_ms = 123;
  opt.tcp = false;
  opt.drop_on_latency = true;
  opt.buffer_mode = "none";
  opt.h264_payload_type = 97;
  opt.h264_parse_config_interval = 1;
  opt.h264_fps = 30;
  opt.h264_width = 640;
  opt.h264_height = 480;
  return opt;
}

simaai::neat::nodes::groups::RtspEncodedInputOptions make_mjpeg_encoded_options() {
  simaai::neat::nodes::groups::RtspEncodedInputOptions opt;
  opt.url = "rtsp://example.local/mjpeg";
  opt.codec = simaai::neat::nodes::groups::RtspCodec::MJPEG;
  opt.latency_ms = 80;
  opt.mjpeg_payload_type = 26;
  return opt;
}

void check_rtp_jpeg_depacketize_node() {
  auto node = simaai::neat::nodes::RTPJpegDepacketize(26);
  const std::string fragment = node->backend_fragment(7);
  require_contains(fragment, "application/x-rtp", "RTP JPEG caps missing");
  require_contains(fragment, "clock-rate=90000", "RTP JPEG clock-rate missing");
  require_contains(fragment, "payload=26", "RTP JPEG payload type missing");
  require_not_contains(fragment, "encoding-name=JPEG",
                       "static RTP JPEG payload 26 should not require encoding-name");
  require_contains(fragment, "rtpjpegdepay", "RTP JPEG depay element missing");
  require(node->element_names(7).size() == 2U, "RTPJpegDepacketize element_names size mismatch");

  auto dynamic_node = simaai::neat::nodes::RTPJpegDepacketize(96);
  require_contains(dynamic_node->backend_fragment(8), "encoding-name=JPEG",
                   "dynamic RTP JPEG payload should require encoding-name");

  auto unfiltered_node = simaai::neat::nodes::RTPJpegDepacketize(0);
  require_contains(unfiltered_node->backend_fragment(9), "encoding-name=JPEG",
                   "unfiltered RTP JPEG payload should still require JPEG encoding");
  require_not_contains(unfiltered_node->backend_fragment(9),
                       "payload=", "unfiltered RTP JPEG payload should not filter by payload");

  const auto* provider = dynamic_cast<simaai::neat::OutputSpecProvider*>(node.get());
  require(provider != nullptr, "RTPJpegDepacketize should provide output spec");
  const simaai::neat::OutputSpec spec = provider->output_spec({});
  require(spec.payload_type == simaai::neat::PayloadType::Encoded,
          "RTPJpegDepacketize should output encoded payload");
  require(spec.media_type == "image/jpeg", "RTPJpegDepacketize media type mismatch");
  require(spec.format == "JPEG", "RTPJpegDepacketize format mismatch");
}

void check_h264_encoded_group() {
  const auto opt = make_h264_encoded_options();
  const Graph group = simaai::neat::nodes::groups::RtspEncodedInput(opt);

  std::vector<std::shared_ptr<simaai::neat::Node>> manual;
  manual.push_back(simaai::neat::nodes::RTSPInput(opt.url, opt.latency_ms, opt.tcp,
                                                  opt.drop_on_latency, opt.buffer_mode));
  manual.push_back(simaai::neat::nodes::Queue());
  manual.push_back(
      simaai::neat::nodes::H264Depacketize(opt.h264_payload_type, opt.h264_parse_config_interval,
                                           opt.h264_fps, opt.h264_width, opt.h264_height));
  manual.push_back(simaai::neat::nodes::Queue());
  compare_graph_fragments(group, graph_from_nodes(std::move(manual)), "H264 encoded topology");

  if (const auto backend = describe_backend_if_available(group, "H264 encoded backend")) {
    require_contains(*backend, "rtspsrc", "H264 encoded backend should contain rtspsrc");
    require_contains(*backend, "drop-on-latency=true",
                     "H264 encoded backend should forward drop-on-latency");
    require_contains(*backend, "buffer-mode=none",
                     "H264 encoded backend should forward buffer-mode");
    require_contains(*backend, "rtph264depay", "H264 encoded backend should contain rtph264depay");
    require_contains(*backend, "h264parse", "H264 encoded backend should contain h264parse");
    require_contains(*backend, "payload=97", "H264 encoded backend should use H264 payload type");
  }

  const auto spec = simaai::neat::nodes::groups::RtspEncodedInputOutputSpec(opt);
  require(spec.payload_type == simaai::neat::PayloadType::Encoded,
          "H264 encoded group should advertise encoded payload");
  require(spec.media_type == "video/x-h264", "H264 encoded group media type mismatch");
  require(spec.format == "H264", "H264 encoded group format mismatch");
  require(spec.width == 640 && spec.height == 480, "H264 encoded group shape mismatch");
  require(spec.fps_num == 30 && spec.fps_den == 1, "H264 encoded group fps mismatch");
}

void check_h264_auto_caps_fixup() {
  auto opt = make_h264_encoded_options();
  opt.h264_width = -1;
  opt.h264_height = -1;
  opt.h264_fps = -1;
  opt.fallback_h264_width = 1280;
  opt.fallback_h264_height = 720;
  opt.fallback_h264_fps = 30;
  const Graph group = simaai::neat::nodes::groups::RtspEncodedInput(opt);
  require_contains(group.describe(), "H264CapsFixup", "H264 auto caps should add H264CapsFixup");
  if (const auto backend = describe_backend_if_available(group, "H264 auto caps backend")) {
    require_contains(*backend, "h264_capsfix",
                     "H264 auto caps backend should include caps fixup identity");
  }
}

void check_h264_explicit_fallback_caps_without_probe() {
  auto opt = make_h264_encoded_options();
  opt.auto_caps_from_stream = false;
  opt.h264_width = -1;
  opt.h264_height = -1;
  opt.h264_fps = -1;
  opt.fallback_h264_width = 1280;
  opt.fallback_h264_height = 720;
  opt.fallback_h264_fps = 25;

  const Graph group = simaai::neat::nodes::groups::RtspEncodedInput(opt);
  const std::string backend = group.describe_backend(false);
  require_contains(backend, "width=(int)1280,height=(int)720",
                   "H264 explicit fallback backend should enforce geometry");
  require_contains(backend, "framerate=(fraction)25/1",
                   "H264 explicit fallback backend should enforce fps");
  require_not_contains(backend, "h264_capsfix",
                       "H264 explicit fallback backend should not add auto caps fixup");

  const auto spec = simaai::neat::nodes::groups::RtspEncodedInputOutputSpec(opt);
  require(spec.width == 1280 && spec.height == 720,
          "H264 explicit fallback output spec shape mismatch");
  require(spec.fps_num == 25 && spec.fps_den == 1,
          "H264 explicit fallback output spec fps mismatch");

  opt.fallback_h264_height = -1;
  require_throws_with([&]() { (void)simaai::neat::nodes::groups::RtspEncodedInput(opt); },
                      "H.264 explicit caps require width, height, and fps",
                      "H264 explicit fallback should reject incomplete caps");
}

void check_mjpeg_encoded_group() {
  const auto opt = make_mjpeg_encoded_options();
  const Graph group = simaai::neat::nodes::groups::RtspEncodedInput(opt);

  std::vector<std::shared_ptr<simaai::neat::Node>> manual;
  manual.push_back(simaai::neat::nodes::RTSPInput(opt.url, opt.latency_ms, opt.tcp,
                                                  opt.drop_on_latency, opt.buffer_mode));
  manual.push_back(simaai::neat::nodes::Queue());
  manual.push_back(simaai::neat::nodes::RTPJpegDepacketize(opt.mjpeg_payload_type));
  manual.push_back(simaai::neat::nodes::JpegParse());
  manual.push_back(simaai::neat::nodes::Queue());
  compare_graph_fragments(group, graph_from_nodes(std::move(manual)), "MJPEG encoded topology");

  if (const auto backend = describe_backend_if_available(group, "MJPEG encoded backend")) {
    require_contains(*backend, "rtpjpegdepay", "MJPEG encoded backend should contain rtpjpegdepay");
    require_contains(*backend, "jpegparse", "MJPEG encoded backend should contain jpegparse");
    require_not_contains(*backend, "encoding-name=JPEG",
                         "static RTP JPEG payload 26 should not require encoding-name");
    require_contains(*backend, "payload=26", "MJPEG encoded backend should use MJPEG payload type");
    require_not_contains(*backend, "h264parse",
                         "MJPEG encoded backend should not contain h264parse");
  }

  const auto spec = simaai::neat::nodes::groups::RtspEncodedInputOutputSpec(opt);
  require(spec.payload_type == simaai::neat::PayloadType::Encoded,
          "MJPEG encoded group should advertise encoded payload");
  require(spec.media_type == "image/jpeg", "MJPEG encoded group media type mismatch");
  require(spec.format == "JPEG", "MJPEG encoded group format mismatch");
}

void check_no_queue_mode() {
  auto opt = make_mjpeg_encoded_options();
  opt.insert_queue = false;
  const Graph group = simaai::neat::nodes::groups::RtspEncodedInput(opt);
  require(count_substrings(group.describe(), "Queue") == 0,
          "insert_queue=false encoded topology should not include queues");
  if (const auto backend = describe_backend_if_available(group, "no-queue encoded backend")) {
    require_not_contains(
        *backend, "queue name=", "insert_queue=false encoded backend should not contain queues");
  }
}

void check_decoded_h264_group() {
  simaai::neat::nodes::groups::RtspDecodedInputOptions opt;
  opt.url = "rtsp://example.local/h264";
  opt.latency_ms = 90;
  opt.payload_type = 98;
  opt.h264_width = 640;
  opt.h264_height = 480;
  opt.h264_fps = 30;
  opt.decoder_name = "rtsp_h264_decoder";
  opt.num_buffers = 6;

  const Graph group = simaai::neat::nodes::groups::RtspDecodedInput(opt);
  require_contains(group.describe(), "SimaDecode", "H264 decoded graph should contain SimaDecode");
  if (const auto backend = describe_backend_if_available(group, "H264 decoded backend")) {
    require_contains(*backend, "rtph264depay", "H264 decoded backend should contain H264 depay");
    require_contains(*backend, "neatdecoder", "H264 decoded backend should contain neatdecoder");
    require_contains(*backend, "dec-type=h264", "H264 decoded backend should use SimaDecode(H264)");
    require_contains(*backend, "num-buffers=6", "H264 decoded backend should forward num_buffers");
    require_not_contains(*backend, "rtpjpegdepay",
                         "H264 decoded backend should not contain RTP JPEG");
  }

  const auto spec = simaai::neat::nodes::groups::RtspDecodedInputOutputSpec(opt);
  require(spec.media_type == "video/x-raw", "H264 decoded group media type mismatch");
  require(spec.format == "NV12", "H264 decoded group format mismatch");
  require(spec.width == 640 && spec.height == 480, "H264 decoded group shape mismatch");
  require(spec.fps_num == 30 && spec.memory == "SimaAI",
          "H264 decoded group should advertise decoder-native SimaAI output");
}

void check_decoded_mjpeg_group() {
  simaai::neat::nodes::groups::RtspDecodedInputOptions opt;
  opt.url = "rtsp://example.local/mjpeg";
  opt.codec = simaai::neat::nodes::groups::RtspCodec::MJPEG;
  opt.mjpeg_payload_type = 26;
  opt.decoder_name = "rtsp_mjpeg_decoder";
  opt.dec_width = 800;
  opt.dec_height = 600;
  opt.dec_fps = 25;

  const Graph group = simaai::neat::nodes::groups::RtspDecodedInput(opt);
  require_contains(group.describe(), "SimaDecode", "MJPEG decoded graph should contain SimaDecode");
  require_contains(group.describe(), "EncodedCapsFixup",
                   "MJPEG decoded graph with dec_fps should fix encoded caps");
  if (const auto backend = describe_backend_if_available(group, "MJPEG decoded backend")) {
    require_contains(*backend, "rtpjpegdepay",
                     "MJPEG decoded backend should contain RTP JPEG depay");
    require_contains(*backend, "jpegparse", "MJPEG decoded backend should contain jpegparse");
    require_contains(*backend, "neatdecoder", "MJPEG decoded backend should contain neatdecoder");
    require_contains(*backend, "dec-type=mjpeg",
                     "MJPEG decoded backend should use SimaDecode(MJPEG)");
    require_contains(*backend, "dec-width=800", "MJPEG decoded backend should forward dec_width");
    require_contains(*backend, "encoded_capsfix",
                     "MJPEG decoded backend should include encoded caps fixup");
    require_not_contains(*backend, "rtph264depay",
                         "MJPEG decoded backend should not contain RTP H264 depay");
  }

  const auto spec = simaai::neat::nodes::groups::RtspDecodedInputOutputSpec(opt);
  require(spec.media_type == "video/x-raw", "MJPEG decoded group media type mismatch");
  require(spec.format == "NV12", "MJPEG decoded group format mismatch");
  require(spec.width == 800 && spec.height == 600, "MJPEG decoded group shape mismatch");
  require(spec.fps_num == 25 && spec.memory == "SimaAI",
          "MJPEG decoded group should advertise decoder-native SimaAI output");

  opt.dec_fps = -1;
  const Graph auto_fps_group = simaai::neat::nodes::groups::RtspDecodedInput(opt);
  require_contains(auto_fps_group.describe(), "EncodedCapsFixup",
                   "MJPEG decoded graph with auto caps should fix encoded caps");
  if (const auto backend =
          describe_backend_if_available(auto_fps_group, "MJPEG auto-fps decoded backend")) {
    require_contains(*backend, "encoded_capsfix",
                     "MJPEG auto-fps backend should include encoded caps fixup");
  }

  opt.auto_caps_from_stream = false;
  const Graph no_fps_group = simaai::neat::nodes::groups::RtspDecodedInput(opt);
  require_not_contains(
      no_fps_group.describe(), "EncodedCapsFixup",
      "MJPEG decoded graph without auto caps or dec_fps should not insert caps fixup");
}

void check_mjpeg_sdp_fps_matches_selected_payload() {
  constexpr const char* kMixedSdp = "v=0\r\n"
                                    "o=- 0 0 IN IP4 127.0.0.1\r\n"
                                    "s=Mixed codec stream\r\n"
                                    "t=0 0\r\n"
                                    "m=video 0 RTP/AVP 96\r\n"
                                    "a=rtpmap:96 H264/90000\r\n"
                                    "a=framerate:30\r\n"
                                    "m=video 0 RTP/AVP 26\r\n"
                                    "a=rtpmap:26 JPEG/90000\r\n"
                                    "a=framerate:15\r\n";

  const int mjpeg_fps =
      simaai::neat::session_test::parse_sdp_fps_for_rtp_payload_for_test(kMixedSdp, 26, "JPEG");
  require(mjpeg_fps == 15, "MJPEG SDP FPS should come from selected RTP/JPEG payload");

  const int h264_payload_as_jpeg_fps =
      simaai::neat::session_test::parse_sdp_fps_for_rtp_payload_for_test(kMixedSdp, 96, "JPEG");
  require(h264_payload_as_jpeg_fps == 0,
          "MJPEG SDP FPS should not use H264 RTP media with the same payload filter");

  const int unfiltered_mjpeg_fps =
      simaai::neat::session_test::parse_sdp_fps_for_rtp_payload_for_test(kMixedSdp, -1, "JPEG");
  require(unfiltered_mjpeg_fps == 15,
          "MJPEG SDP FPS should match JPEG media when payload filtering is disabled");

  constexpr const char* kStaticJpegSdp = "v=0\r\n"
                                         "o=- 0 0 IN IP4 127.0.0.1\r\n"
                                         "s=Static JPEG payload stream\r\n"
                                         "t=0 0\r\n"
                                         "m=video 0 RTP/AVP 96\r\n"
                                         "a=rtpmap:96 H264/90000\r\n"
                                         "a=framerate:30\r\n"
                                         "m=video 0 RTP/AVP 26\r\n"
                                         "a=framerate:15\r\n";

  const int static_jpeg_fps = simaai::neat::session_test::parse_sdp_fps_for_rtp_payload_for_test(
      kStaticJpegSdp, -1, "JPEG");
  require(static_jpeg_fps == 15,
          "MJPEG SDP FPS should recognize static RTP JPEG payload without rtpmap");

  constexpr const char* kSessionFpsSdp = "v=0\r\n"
                                         "o=- 0 0 IN IP4 127.0.0.1\r\n"
                                         "s=Session FPS stream\r\n"
                                         "t=0 0\r\n"
                                         "a=framerate:24\r\n"
                                         "m=video 0 RTP/AVP 26\r\n"
                                         "a=rtpmap:26 JPEG/90000\r\n";

  const int session_fps = simaai::neat::session_test::parse_sdp_fps_for_rtp_payload_for_test(
      kSessionFpsSdp, 26, "JPEG");
  require(session_fps == 24, "MJPEG SDP FPS should fall back to session-level framerate");
}

void check_invalid_codec_errors() {
  auto encoded = make_h264_encoded_options();
  encoded.codec = static_cast<simaai::neat::nodes::groups::RtspCodec>(999);
  require_throws_with([&]() { (void)simaai::neat::nodes::groups::RtspEncodedInput(encoded); },
                      "unsupported codec", "encoded group invalid codec");
  require_throws_with(
      [&]() { (void)simaai::neat::nodes::groups::RtspEncodedInputOutputSpec(encoded); },
      "unsupported codec", "encoded output spec invalid codec");

  simaai::neat::nodes::groups::RtspDecodedInputOptions decoded;
  decoded.url = "rtsp://example.local/invalid";
  decoded.codec = static_cast<simaai::neat::nodes::groups::RtspCodec>(999);
  require_throws_with([&]() { (void)simaai::neat::nodes::groups::RtspDecodedInput(decoded); },
                      "unsupported codec", "decoded group invalid codec");
  require_throws_with(
      [&]() { (void)simaai::neat::nodes::groups::RtspDecodedInputOutputSpec(decoded); },
      "unsupported codec", "decoded output spec invalid codec");
}

} // namespace

int main() {
  try {
    check_rtp_jpeg_depacketize_node();
    check_h264_encoded_group();
    check_h264_auto_caps_fixup();
    check_h264_explicit_fallback_caps_without_probe();
    check_mjpeg_encoded_group();
    check_no_queue_mode();
    check_decoded_h264_group();
    check_decoded_mjpeg_group();
    check_mjpeg_sdp_fps_matches_selected_payload();
    check_invalid_codec_errors();
    std::cout << "[OK] unit_rtsp_encoded_input_group_test passed\n";
    return 0;
  } catch (const std::exception& e) {
    std::cerr << "[FAIL] " << e.what() << "\n";
    return 1;
  }
}
