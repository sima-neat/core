#include "nodes/groups/RtspDecodedInput.h"
#include "nodes/groups/RtspEncodedInput.h"
#include "nodes/groups/GroupOutputSpec.h"

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

#include "test_utils.h"

#include <functional>
#include <iostream>
#include <memory>
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

bool decoder_backend_available() {
  try {
    return simaai::neat::element_exists("neatdecoder");
  } catch (const std::exception& e) {
    std::cout << "[INFO] decoder backend checks skipped: " << e.what() << "\n";
    return false;
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
  require_contains(fragment, "encoding-name=JPEG", "RTP JPEG encoding-name missing");
  require_contains(fragment, "clock-rate=90000", "RTP JPEG clock-rate missing");
  require_contains(fragment, "payload=26", "RTP JPEG payload type missing");
  require_contains(fragment, "rtpjpegdepay", "RTP JPEG depay element missing");
  require(node->element_names(7).size() == 2U, "RTPJpegDepacketize element_names size mismatch");

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

  const std::string backend = group.describe_backend(false);
  require_contains(backend, "rtspsrc", "H264 encoded backend should contain rtspsrc");
  require_contains(backend, "drop-on-latency=true",
                   "H264 encoded backend should forward drop-on-latency");
  require_contains(backend, "buffer-mode=none", "H264 encoded backend should forward buffer-mode");
  require_contains(backend, "rtph264depay", "H264 encoded backend should contain rtph264depay");
  require_contains(backend, "h264parse", "H264 encoded backend should contain h264parse");
  require_contains(backend, "payload=97", "H264 encoded backend should use H264 payload type");

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
  require_contains(group.describe_backend(false), "h264_capsfix",
                   "H264 auto caps backend should include caps fixup identity");
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

  const std::string backend = group.describe_backend(false);
  require_contains(backend, "rtpjpegdepay", "MJPEG encoded backend should contain rtpjpegdepay");
  require_contains(backend, "jpegparse", "MJPEG encoded backend should contain jpegparse");
  require_contains(backend, "encoding-name=JPEG",
                   "MJPEG encoded backend should request RTP JPEG caps");
  require_contains(backend, "payload=26", "MJPEG encoded backend should use MJPEG payload type");
  require_not_contains(backend, "h264parse", "MJPEG encoded backend should not contain h264parse");

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
  require_not_contains(group.describe_backend(false), "queue name=",
                       "insert_queue=false encoded backend should not contain queues");
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
  if (decoder_backend_available()) {
    const std::string backend = group.describe_backend(false);
    require_contains(backend, "rtph264depay", "H264 decoded backend should contain H264 depay");
    require_contains(backend, "neatdecoder", "H264 decoded backend should contain neatdecoder");
    require_contains(backend, "dec-type=h264", "H264 decoded backend should use SimaDecode(H264)");
    require_contains(backend, "num-buffers=6", "H264 decoded backend should forward num_buffers");
    require_not_contains(backend, "rtpjpegdepay",
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
  if (decoder_backend_available()) {
    const std::string backend = group.describe_backend(false);
    require_contains(backend, "rtpjpegdepay",
                     "MJPEG decoded backend should contain RTP JPEG depay");
    require_contains(backend, "jpegparse", "MJPEG decoded backend should contain jpegparse");
    require_contains(backend, "neatdecoder", "MJPEG decoded backend should contain neatdecoder");
    require_contains(backend, "dec-type=mjpeg",
                     "MJPEG decoded backend should use SimaDecode(MJPEG)");
    require_contains(backend, "dec-width=800", "MJPEG decoded backend should forward dec_width");
    require_not_contains(backend, "rtph264depay",
                         "MJPEG decoded backend should not contain RTP H264 depay");
  }

  const auto spec = simaai::neat::nodes::groups::RtspDecodedInputOutputSpec(opt);
  require(spec.media_type == "video/x-raw", "MJPEG decoded group media type mismatch");
  require(spec.format == "NV12", "MJPEG decoded group format mismatch");
  require(spec.width == 800 && spec.height == 600, "MJPEG decoded group shape mismatch");
  require(spec.fps_num == 25 && spec.memory == "SimaAI",
          "MJPEG decoded group should advertise decoder-native SimaAI output");
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
    check_mjpeg_encoded_group();
    check_no_queue_mode();
    check_decoded_h264_group();
    check_decoded_mjpeg_group();
    check_invalid_codec_errors();
    std::cout << "[OK] unit_rtsp_encoded_input_group_test passed\n";
    return 0;
  } catch (const std::exception& e) {
    std::cerr << "[FAIL] " << e.what() << "\n";
    return 1;
  }
}
