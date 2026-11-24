#include "asset_utils.h"
#include "gst/GstHelpers.h"
#include "gst/GstInit.h"
#include "nodes/common/Output.h"
#include "nodes/io/Input.h"
#include "nodes/io/RTSPInput.h"
#include "nodes/rtp/H264Depacketize.h"
#include "nodes/rtp/H265Depacketize.h"
#include "nodes/rtp/RTPJpegDepacketize.h"
#include "pipeline/Graph.h"
#include "pipeline/ErrorCodes.h"
#include "test_utils.h"

#include <gst/rtsp-server/rtsp-server.h>
#include <chrono>
#include <cstdlib>
#include <iostream>
#include <string>
#include <thread>
#include <vector>

namespace {

std::string trim_copy(const std::string& value) {
  const std::size_t start = value.find_first_not_of(" \t\r\n");
  if (start == std::string::npos) {
    return {};
  }
  const std::size_t end = value.find_last_not_of(" \t\r\n");
  return value.substr(start, end - start + 1);
}

std::vector<std::string> split_rtsp_urls(const std::string& value) {
  std::vector<std::string> out;
  std::string current;
  for (char c : value) {
    if (c == ',' || c == ';' || c == ' ' || c == '\t' || c == '\n' || c == '\r') {
      const std::string trimmed = trim_copy(current);
      if (!trimmed.empty()) {
        out.push_back(trimmed);
      }
      current.clear();
      continue;
    }
    current.push_back(c);
  }
  const std::string trimmed = trim_copy(current);
  if (!trimmed.empty()) {
    out.push_back(trimmed);
  }
  return out;
}

std::string first_rtsp_url_from_env() {
  if (const char* single = std::getenv("SIMANEAT_TEST_RTSP_H264_URL"); single && *single) {
    return trim_copy(single);
  }
  if (const char* many = std::getenv("SIMANEAT_TEST_RTSP_H264_URLS"); many && *many) {
    std::vector<std::string> urls = split_rtsp_urls(many);
    if (!urls.empty()) {
      return urls.front();
    }
  }
  return {};
}

simaai::neat::Sample pull_one_encoded_sample(simaai::neat::Run& run, int timeout_ms) {
  const auto deadline = std::chrono::steady_clock::now() + std::chrono::milliseconds(timeout_ms);
  simaai::neat::PullError err;
  while (std::chrono::steady_clock::now() < deadline) {
    simaai::neat::Sample sample;
    const simaai::neat::PullStatus status = run.pull(250, sample, &err);
    if (status == simaai::neat::PullStatus::Ok) {
      return sample;
    }
    if (status == simaai::neat::PullStatus::Timeout) {
      continue;
    }
    if (status == simaai::neat::PullStatus::Closed) {
      throw std::runtime_error("RTSP encoded source closed before producing a sample");
    }
    std::string message = err.message.empty() ? "RTSP encoded source pull failed" : err.message;
    if (!err.code.empty()) {
      message += " (code=" + err.code + ")";
    }
    throw std::runtime_error(message);
  }
  throw std::runtime_error("RTSP encoded source did not produce a sample before timeout");
}

// Only the H.263 offer is synthetic. Both H.264 and JPEG carry playable media.
struct VideoOfferMedia {
  GstRTSPMedia parent;
};
using VideoOfferMediaClass = GstRTSPMediaClass;
G_DEFINE_TYPE(VideoOfferMedia, video_offer_media, GST_TYPE_RTSP_MEDIA)

void video_offer_media_init(VideoOfferMedia*) {}

void video_offer_media_class_init(VideoOfferMediaClass* klass) {
  klass->setup_sdp = [](GstRTSPMedia* self, GstSDPMessage* sdp, GstSDPInfo* info) -> gboolean {
    if (!GST_RTSP_MEDIA_CLASS(video_offer_media_parent_class)->setup_sdp(self, sdp, info))
      return FALSE;
    auto* media = &g_array_index(sdp->medias, GstSDPMedia, 0);
    gst_sdp_media_set_media(media, "Video");
    const bool jpeg = g_str_equal(gst_sdp_media_get_format(media, 0), "26");
    if (g_object_get_data(G_OBJECT(self), "static-h263"))
      gst_sdp_media_replace_format(media, 0, "34");
    else if (!jpeg)
      return TRUE;
    for (guint i = gst_sdp_media_attributes_len(media); i > 0; --i) {
      const auto* attr = gst_sdp_media_get_attribute(media, i - 1);
      if (g_str_equal(attr->key, "rtpmap") || g_str_equal(attr->key, "fmtp"))
        gst_sdp_media_remove_attribute(media, i - 1);
    }
    if (jpeg)
      gst_sdp_media_add_attribute(media, "rtpmap", "26 MJPEG/90000");
    return TRUE;
  };
}

void check_loopback_offers() {
  using namespace simaai::neat;
  g_autoptr(GMainContext) context = g_main_context_new();
  g_autoptr(GstRTSPServer) server = gst_rtsp_server_new();
  gst_rtsp_server_set_address(server, "127.0.0.1");
  gst_rtsp_server_set_service(server, "0");
  g_autoptr(GstRTSPMountPoints) mounts = gst_rtsp_server_get_mount_points(server);
  auto* factory = gst_rtsp_media_factory_new();
  gst_rtsp_media_factory_set_launch(
      factory, "( audiotestsrc is-live=true ! alawenc ! rtppcmapay name=pay0 pt=8 )");
  gst_rtsp_mount_points_add_factory(mounts, "/audio", factory);
  auto* video = gst_rtsp_media_factory_new();
  const auto launch = "( filesrc location=\"" + sima_test::test_decoder_fixture_path().string() +
                      "\" ! h264parse ! rtph264pay config-interval=1 name=pay0 pt=96 )";
  gst_rtsp_media_factory_set_media_gtype(video, video_offer_media_get_type());
  gst_rtsp_media_factory_set_launch(video, launch.c_str());
  gst_rtsp_mount_points_add_factory(mounts, "/video", video);
  auto* static_video = gst_rtsp_media_factory_new();
  gst_rtsp_media_factory_set_media_gtype(static_video, video_offer_media_get_type());
  gst_rtsp_media_factory_set_launch(static_video, launch.c_str());
  g_signal_connect(static_video, "media-configure",
                   G_CALLBACK(+[](GstRTSPMediaFactory*, GstRTSPMedia* media, gpointer) {
                     g_object_set_data(G_OBJECT(media), "static-h263", GINT_TO_POINTER(1));
                   }),
                   nullptr);
  gst_rtsp_mount_points_add_factory(mounts, "/static", static_video);
  auto* jpeg = gst_rtsp_media_factory_new();
  gst_rtsp_media_factory_set_media_gtype(jpeg, video_offer_media_get_type());
  gst_rtsp_media_factory_set_launch(jpeg, "( videotestsrc is-live=true ! video/x-raw,format=I420 "
                                          "! jpegenc ! rtpjpegpay name=pay0 pt=26 )");
  gst_rtsp_mount_points_add_factory(mounts, "/jpeg", jpeg);
  require(gst_rtsp_server_attach(server, context) != 0, "could not bind audio RTSP fixture");
  std::jthread worker([&](std::stop_token stop) {
    std::stop_callback wake(stop, [&] { g_main_context_wakeup(context); });
    while (!stop.stop_requested())
      g_main_context_iteration(context, TRUE);
  });

  Graph graph;
  const auto base_url =
      "rtsp://127.0.0.1:" + std::to_string(gst_rtsp_server_get_bound_port(server));
  graph.add(nodes::RTSPInput(base_url + "/audio", 200, true));
  graph.add(nodes::H264Depacketize(96, 1));
  graph.add(nodes::Output());
  require_neat_error([&] { (void)graph.build(); }, error_codes::kMediaCaps, "H264 payload_type=96",
                     "Offered: no video streams");
  const auto rtsp_url = base_url + "/video";
  for (int payload : {1, 128}) {
    using namespace simaai::neat;
    Graph mismatched;
    mismatched.add(nodes::RTSPInput(rtsp_url, 200, true));
    mismatched.add(nodes::H264Depacketize(payload));
    mismatched.add(nodes::Output());
    require_neat_error([&] { (void)mismatched.build(); },
                       payload == 1 ? error_codes::kMediaCaps : error_codes::kOptionOutOfRange,
                       "payload_type", payload == 1 ? "H264 payload_type=96" : "128");
  }

  Graph wrong_codec;
  wrong_codec.add(nodes::RTSPInput(rtsp_url, 200, true));
  wrong_codec.add(nodes::H265Depacketize(96));
  wrong_codec.add(nodes::Output());
  require_neat_error([&] { (void)wrong_codec.build(); }, error_codes::kMediaCaps, "H265", "H264");

  Graph static_mismatch;
  static_mismatch.add(nodes::RTSPInput(base_url + "/static", 200, true));
  static_mismatch.add(nodes::H264Depacketize(96));
  static_mismatch.add(nodes::Output());
  require_neat_error([&] { (void)static_mismatch.build(); }, error_codes::kMediaCaps,
                     "H264 payload_type=96", "H263 payload_type=34");

  Graph matching;
  matching.add(nodes::RTSPInput(rtsp_url, 200, true));
  matching.add(nodes::H264Depacketize(96));
  matching.add(nodes::Output());
  auto run = matching.build();
  require_contains(pull_one_encoded_sample(run, 8000).caps_string, "video/x-h264", "matching SDP");

  Graph jpeg_graph;
  jpeg_graph.add(nodes::RTSPInput(base_url + "/jpeg", 200, true));
  jpeg_graph.add(nodes::RTPJpegDepacketize(26));
  jpeg_graph.add(nodes::Output());
  auto jpeg_run = jpeg_graph.build();
  require_contains(pull_one_encoded_sample(jpeg_run, 8000).caps_string, "image/jpeg",
                   "static JPEG advertised as MJPEG");
}

} // namespace

int main(int argc, char** argv) {
  try {
    simaai::neat::gst_init_once();
    if (argc == 2 && std::string(argv[1]) == "--loopback") {
      check_loopback_offers();
      return 0;
    }
    const std::string rtsp_url = first_rtsp_url_from_env();
    if (rtsp_url.empty()) {
      std::cout << "[SKIP] set SIMANEAT_TEST_RTSP_H264_URL or "
                   "SIMANEAT_TEST_RTSP_H264_URLS\n";
      return 77;
    }

    require(simaai::neat::element_exists("rtspsrc"), "missing GStreamer element: rtspsrc");
    require(simaai::neat::element_exists("rtph264depay"),
            "missing GStreamer element: rtph264depay");
    require(simaai::neat::element_exists("h264parse"), "missing GStreamer element: h264parse");
    require(simaai::neat::element_exists("appsrc"), "missing GStreamer element: appsrc");
    require(simaai::neat::element_exists("appsink"), "missing GStreamer element: appsink");

    simaai::neat::Graph source;
    source.add(simaai::neat::nodes::RTSPInput(rtsp_url, /*latency_ms=*/200, /*tcp=*/true));
    source.add(simaai::neat::nodes::H264Depacketize(/*payload_type=*/96,
                                                    /*h264_parse_config_interval=*/1));
    source.add(simaai::neat::nodes::Output(simaai::neat::OutputOptions::EveryFrame(4)));

    simaai::neat::RunOptions source_opt;
    source_opt.output_memory = simaai::neat::OutputMemory::ZeroCopy;
    simaai::neat::Run source_run = source.build(source_opt);
    simaai::neat::Sample encoded = pull_one_encoded_sample(source_run, /*timeout_ms=*/8000);

    require(encoded.payload_type == simaai::neat::PayloadType::Encoded,
            "RTSP parsed output should be an encoded Sample");
    require_contains(encoded.caps_string, "video/x-h264",
                     "RTSP parsed output should carry H264 caps");
    require(encoded.caps_string.find("video/x-raw,format=ENCODED") == std::string::npos,
            "RTSP parsed output should not carry raw ENCODED caps");
    require(encoded.tensors.size() == 1U, "RTSP parsed output should carry one tensor");
    require(encoded.tensors.front().storage != nullptr, "RTSP parsed tensor missing storage");
    require(encoded.tensors.front().storage->kind == simaai::neat::StorageKind::GstSample,
            "zero-copy RTSP parsed output should preserve the GstSample holder");

    simaai::neat::Graph downstream;
    simaai::neat::InputOptions input_opt;
    input_opt.payload_type = simaai::neat::PayloadType::Auto;
    input_opt.memory_policy = simaai::neat::InputMemoryPolicy::SystemMemory;
    downstream.add(simaai::neat::nodes::Input(input_opt));
    downstream.custom("identity name=encoded_boundary_probe silent=true ! "
                      "fakesink name=encoded_boundary_sink sync=false");

    simaai::neat::RunOptions downstream_opt;
    simaai::neat::Run downstream_run =
        downstream.build(simaai::neat::Sample{encoded}, downstream_opt);
    const std::string pipeline = downstream.last_pipeline();
    require_contains(pipeline, "video/x-h264",
                     "encoded boundary appsrc should preserve upstream H264 caps");
    require(pipeline.find("video/x-raw,format=ENCODED") == std::string::npos,
            "encoded boundary appsrc must not infer raw ENCODED caps");
    require(downstream_run.push(simaai::neat::Sample{encoded}),
            "encoded boundary downstream push failed");
    std::this_thread::sleep_for(std::chrono::milliseconds(100));
    require(downstream_run.last_error().empty(),
            "encoded boundary downstream run reported an error");

    std::cout << "[OK] rtsp_encoded_boundary_caps_test passed\n";
    return 0;
  } catch (const std::exception& e) {
    std::cerr << "[FAIL] " << e.what() << "\n";
    return 1;
  }
}
