#include "builder/GraphPrinter.h"
#include "nodes/groups/UdpH264OutputGroup.h"
#include "nodes/groups/UdpOutputGroupG.h"
#include "nodes/io/UdpOutput.h"
#include "nodes/io/internal/JpegRtpUdpSink.h"
#include "nodes/rtp/H264CapsFixup.h"
#include "nodes/sima/QuantTess.h"
#include "jpeg_rtp_udp_sink_fixture.h"
#include "test_main.h"
#include "test_utils.h"

#include <gst/gst.h>

#include <nlohmann/json.hpp>

#include <functional>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace {

std::string graph_snapshot(const simaai::neat::Graph& graph) {
  simaai::neat::GraphPrinter::Options opt;
  opt.show_backend_fragment = true;
  return graph.describe(opt);
}

bool throws_with(const std::function<void()>& fn, const std::string& needle) {
  try {
    fn();
  } catch (const std::exception& e) {
    if (needle.empty())
      return true;
    return std::string(e.what()).find(needle) != std::string::npos;
  }
  return false;
}

} // namespace

RUN_TEST(
    "unit_transport_output_nodes_test", ([] {
      using nlohmann::json;

      // UdpOutput fragment snapshot.
      {
        simaai::neat::UdpOutputOptions opt;
        opt.host = "10.0.0.5";
        opt.port = 5500;
        opt.sync = true;
        opt.async = false;

        auto node = simaai::neat::nodes::UdpOutput(opt);
        require(node->kind() == "UdpOutput", "UdpOutput kind mismatch");
        const std::string fragment = node->backend_fragment(7);
        require_contains(fragment, "udpsink", "UdpOutput fragment should use udpsink");
        require_contains(fragment, "host=10.0.0.5", "UdpOutput host mismatch");
        require_contains(fragment, "port=5500", "UdpOutput port mismatch");
        require_contains(fragment, "sync=true", "UdpOutput sync mismatch");
        require_contains(fragment, "async=false", "UdpOutput async mismatch");
        require_contains(fragment, "enable-last-sample=false",
                         "UdpOutput must not retain its last packet: it pins the frame's memory");
        require(node->element_names(7).empty(), "UdpOutput element_names contract should be empty");
      }

      // JpegRtpUdpSink fragment snapshot, element names, specialization, and parse against
      // the element.
      {
        using simaai::neat::internal::JpegRtpUdpSink;
        using simaai::neat::internal::JpegRtpUdpSinkOptions;
        const JpegRtpUdpSink defaults{JpegRtpUdpSinkOptions{}};
        require(defaults.kind() == "JpegRtpUdpSink",
                "JpegRtpUdpSink kind is matched by name and must stay stable");
        // Unspecialized, the node keeps a parser so an input without sized caps still
        // negotiates.
        require(defaults.input() == JpegRtpUdpSink::Input::ParseDimensions,
                "JpegRtpUdpSink must default to parsing dimensions");
        require(defaults.backend_fragment(3) ==
                    "jpegparse name=n3_jpegparse disable-passthrough=true ! "
                    "capsfilter name=n3_rtp_jpeg_dimensions "
                    "caps=\"image/jpeg,width=(int)[8,2040,8],height=(int)[8,2040,8]\" ! "
                    "neatjpegudpsink name=n3_jpegudp host=127.0.0.1 port=5000 pt=26 mtu=1400 "
                    "timestamp-offset=0 sync=false async=false enable-last-sample=false",
                "JpegRtpUdpSink default fragment mismatch");
        require(defaults.element_names(3) == std::vector<std::string>{"n3_jpegparse",
                                                                      "n3_rtp_jpeg_dimensions",
                                                                      "n3_jpegudp"},
                "JpegRtpUdpSink parsing element_names mismatch");

        JpegRtpUdpSinkOptions opt;
        opt.host = "10.0.0.5";
        opt.port = 5600;
        opt.payload_type = 27;
        opt.mtu = 1200;
        opt.sync = true;
        opt.async = true;
        const JpegRtpUdpSink node(opt, JpegRtpUdpSink::Input::SizedCaps);
        const std::string fragment = node.backend_fragment(7);
        require(fragment == "capsfilter name=n7_rtp_jpeg_dimensions "
                            "caps=\"image/jpeg,width=(int)[8,2040,8],height=(int)[8,2040,8]\" ! "
                            "neatjpegudpsink name=n7_jpegudp host=10.0.0.5 port=5600 pt=27 "
                            "mtu=1200 timestamp-offset=0 sync=true async=true "
                            "enable-last-sample=false",
                "JpegRtpUdpSink sized fragment mismatch");
        require(fragment ==
                    JpegRtpUdpSink(opt, JpegRtpUdpSink::Input::SizedCaps).backend_fragment(7),
                "JpegRtpUdpSink fragment must be deterministic");
        require(node.element_names(7) ==
                    std::vector<std::string>{"n7_rtp_jpeg_dimensions", "n7_jpegudp"},
                "JpegRtpUdpSink sized element_names must name the fragment's element");

        // Specialization: an input with known dimensions drops the parser; anything else
        // keeps it. The node's options survive either way.
        simaai::neat::OutputSpec sized;
        sized.media_type = "image/jpeg";
        sized.width = 1920;
        sized.height = 1080;
        sized.certainty = simaai::neat::SpecCertainty::Hint;
        simaai::neat::OutputSpec unsized = sized;
        unsized.width = 0;
        unsized.height = 0;
        const simaai::neat::internal::InputSpecSpecializationContext context;
        for (const auto* source : {&defaults, &node}) {
          for (const auto& [input, parse] : {std::pair{&sized, false}, std::pair{&unsized, true}}) {
            const auto selected = source->specialize_for_input(*input, context);
            const auto* sink = dynamic_cast<const JpegRtpUdpSink*>(selected.get());
            require(sink != nullptr && sink->kind() == "JpegRtpUdpSink",
                    "specialization must return a JpegRtpUdpSink");
            require((sink->input() == JpegRtpUdpSink::Input::ParseDimensions) == parse,
                    "specialization must parse exactly when the input has no dimensions");
            require(sink->options().host == source->options().host &&
                        sink->options().port == source->options().port &&
                        sink->options().async == source->options().async,
                    "specialization must keep the sink's options");
            require((sink->backend_fragment(2).find("jpegparse") != std::string::npos) == parse,
                    "specialized fragment must match its variant");
          }
        }
        // RTP/JPEG carries sizes in multiples of 8 up to 2040: a sized input outside that fails
        // the build with a clear error instead of a generic caps negotiation failure.
        for (const auto& [width, height] : {std::pair{8, 8}, std::pair{2040, 2040}}) {
          simaai::neat::OutputSpec edge = sized;
          edge.width = width;
          edge.height = height;
          const auto* sink = dynamic_cast<const JpegRtpUdpSink*>(
              defaults.specialize_for_input(edge, context).get());
          require(sink != nullptr && sink->input() == JpegRtpUdpSink::Input::SizedCaps,
                  "RTP/JPEG-representable sizes must specialize to the bare sink");
        }
        for (const auto& [width, height] : {std::pair{320, 180}, std::pair{2048, 1536}}) {
          simaai::neat::OutputSpec bad = sized;
          bad.width = width;
          bad.height = height;
          std::string message;
          try {
            (void)defaults.specialize_for_input(bad, context);
          } catch (const std::invalid_argument& e) {
            message = e.what();
          }
          require(message == "VideoSender MJPEG: RTP/JPEG carries widths and heights in "
                             "multiples of 8 up to 2040 (got " +
                                 std::to_string(width) + "x" + std::to_string(height) +
                                 "); choose another camera mode.",
                  "unrepresentable RTP/JPEG size must throw a clear invalid_argument, got: " +
                      message);
        }

        const jpeg_rtp_udp_sink_fixture::RestoreRegistry restore_registry;
        jpeg_rtp_udp_sink_fixture::set_factory_present(true);
        for (const auto* variant : {&defaults, &node}) {
          const std::string launch = variant->backend_fragment(7);
          GError* error = nullptr;
          GstElement* bin = gst_parse_launch(launch.c_str(), &error);
          require(bin != nullptr && error == nullptr,
                  std::string("JpegRtpUdpSink fragment must parse: ") +
                      (error ? error->message : "no element"));
          GstElement* sink = GST_IS_BIN(bin) ? gst_bin_get_by_name(GST_BIN(bin), "n7_jpegudp")
                                             : GST_ELEMENT(gst_object_ref(bin));
          require(sink != nullptr && std::string(GST_OBJECT_NAME(sink)) == "n7_jpegudp",
                  "parsed JpegRtpUdpSink element must carry its element name");
          gboolean last_sample = TRUE;
          g_object_get(sink, "enable-last-sample", &last_sample, nullptr);
          require(!last_sample, "JpegRtpUdpSink must not retain the frame's buffer");
          for (const auto& name : variant->element_names(7)) {
            GstElement* element =
                GST_IS_BIN(bin) ? gst_bin_get_by_name(GST_BIN(bin), name.c_str()) : nullptr;
            require(element != nullptr || (!GST_IS_BIN(bin) && name == "n7_jpegudp"),
                    "element_names must name an element of the parsed fragment: " + name);
            if (element) {
              gst_object_unref(element);
            }
          }
          gst_object_unref(sink);
          gst_object_unref(bin);
        }
      }

      // H264CapsFixup snapshot.
      {
        auto node = simaai::neat::nodes::H264CapsFixup(25, 1280, 720);
        require(node->kind() == "H264CapsFixup", "H264CapsFixup kind mismatch");
        require_contains(node->backend_fragment(3), "identity name=n3_h264_capsfix",
                         "H264CapsFixup fragment mismatch");
        const auto names = node->element_names(3);
        require(names.size() == 1 && names[0] == "n3_h264_capsfix",
                "H264CapsFixup element name mismatch");
        auto* raw = dynamic_cast<simaai::neat::H264CapsFixup*>(node.get());
        require(raw != nullptr, "H264CapsFixup dynamic_cast failed");
        require(raw->fallback_fps() == 25, "H264CapsFixup fallback_fps getter mismatch");
        require(raw->fallback_width() == 1280, "H264CapsFixup fallback_width getter mismatch");
        require(raw->fallback_height() == 720, "H264CapsFixup fallback_height getter mismatch");
      }

      // UdpH264OutputGroup snapshot + caps escaping behavior.
      {
        simaai::neat::nodes::groups::UdpH264OutputGroupOptions opt;
        opt.h264_caps = "video/x-h264,profile=\"high\"";
        opt.payload_type = 97;
        opt.config_interval = 2;
        opt.udp_host = "127.0.0.1";
        opt.udp_port = 5600;

        auto group = simaai::neat::nodes::groups::UdpH264OutputGroup(opt);
        const std::string snapshot = graph_snapshot(group);
        require_contains(snapshot, "0) H264Parse",
                         "UdpH264OutputGroup node[0] should be H264Parse");
        require_contains(snapshot, "1) CustomNode",
                         "UdpH264OutputGroup node[1] should be caps custom node");
        require_contains(snapshot, "2) H264Packetize",
                         "UdpH264OutputGroup node[2] should be H264Packetize");
        require_contains(snapshot, "3) UdpOutput",
                         "UdpH264OutputGroup node[3] should be UdpOutput");
        require_contains(snapshot, "capsfilter caps=\"video/x-h264,profile=\\\"high\\\"\"",
                         "UdpH264OutputGroup should escape caps with quotes");
        require_contains(snapshot, "port=5600", "UdpH264OutputGroup UDP sink port mismatch");
      }

      // UdpH264OutputGroup without explicit caps should omit capsfilter node.
      {
        simaai::neat::nodes::groups::UdpH264OutputGroupOptions opt;
        opt.h264_caps.clear();

        auto group = simaai::neat::nodes::groups::UdpH264OutputGroup(opt);
        const std::string snapshot = graph_snapshot(group);
        require_contains(snapshot, "0) H264Parse", "UdpH264OutputGroup no-caps node[0] mismatch");
        require_contains(snapshot, "1) H264Packetize",
                         "UdpH264OutputGroup no-caps node[1] mismatch");
        require_contains(snapshot, "2) UdpOutput", "UdpH264OutputGroup no-caps node[2] mismatch");
        require(snapshot.find("CustomNode") == std::string::npos,
                "UdpH264OutputGroup without caps should skip custom caps node");
      }

      // UdpOutputGroupG snapshot behavior.
      {
        simaai::neat::nodes::groups::UdpOutputGroupGOptions opt;
        opt.render_config = "/tmp/render_config.json";
        opt.width = 1280;
        opt.height = 720;
        opt.fps = 25;
        opt.bitrate_kbps = 2500;
        opt.payload_type = 98;
        opt.config_interval = 3;
        opt.udp_host = "127.0.0.1";
        opt.udp_port = 5700;
        opt.udp_sync = true;
        opt.udp_async = false;

        auto group = simaai::neat::nodes::groups::UdpOutputGroupG(opt);
        const std::string snapshot = graph_snapshot(group);
        require_contains(snapshot, "0) CustomNode",
                         "UdpOutputGroupG node[0] should be render custom node");
        require_contains(snapshot, "1) VideoSenderRawIngress", "raw ingress missing");
        require_contains(snapshot, "width=1280,height=720", "legacy geometry constraint lost");
        require_contains(snapshot, "2) SimaEncode", "UdpOutputGroupG encoder missing");
        require_contains(snapshot, "3) H264Parse", "UdpOutputGroupG parser missing");
        require_contains(snapshot, "4) H264Packetize", "UdpOutputGroupG payloader missing");
        require_contains(snapshot, "5) UdpOutput", "UdpOutputGroupG output missing");
        require_contains(snapshot, "simaai_sampledemux",
                         "UdpOutputGroupG render fragment missing sample demux");
        require_contains(snapshot, R"(config="/tmp/render_config.json")",
                         "UdpOutputGroupG render fragment missing render config");
        require_contains(snapshot, "pt=98", "UdpOutputGroupG payload_type mismatch");
        require_contains(snapshot, "config-interval=3", "UdpOutputGroupG config_interval mismatch");
        require_contains(snapshot, "port=5700", "UdpOutputGroupG udp sink port mismatch");
        require_contains(snapshot, "host=127.0.0.1", "UdpOutputGroupG udp sink host mismatch");
        require_contains(snapshot, "sync=true", "UdpOutputGroupG udp sink sync mismatch");
        require_contains(snapshot, "async=false", "UdpOutputGroupG udp sink async mismatch");
      }

      // QuantTess: fragment contract and failure paths.
      {
        simaai::neat::QuantTessOptions opt;
        opt.config_json = json{{"node_name", "quant0"},
                               {"input_buffers", json::array({json{{"name", "decoder"}}})}};
        opt.element_name = "quant_test";
        opt.num_buffers = 4;

        simaai::neat::QuantTess quant(opt);

        require(quant.kind() == "QuantTess", "QuantTess kind mismatch");
        const std::string fragment = quant.backend_fragment(6);
        require_contains(fragment, "neatkernelrunner name=quant_test",
                         "QuantTess fragment name mismatch");
        require(fragment.find("stage-id=") == std::string::npos &&
                    fragment.find("num-buffers=") == std::string::npos,
                "QuantTess must keep placement and depth in its prepared stage");
        require(fragment.find("config=\"") == std::string::npos,
                "QuantTess should not emit legacy config path in fragment");

        const auto names = quant.element_names(6);
        require(names.size() == 1 && names[0] == "quant_test", "QuantTess element names mismatch");

        const json* cfg = quant.config_json();
        require(cfg != nullptr, "QuantTess config_json should be available");
        require((*cfg)["node_name"].get<std::string>() == "quant0",
                "QuantTess config node_name mismatch");

        require(quant.config_path().empty(),
                "QuantTess should keep config in-memory under compiled-contract ownership");

        require(throws_with(
                    []() {
                      simaai::neat::QuantTessOptions bad;
                      bad.num_buffers_locked = true;
                      bad.num_buffers_model = 4;
                      bad.num_buffers = 2;
                      (void)simaai::neat::QuantTess(bad);
                    },
                    "override is not allowed"),
                "QuantTess should reject locked num_buffers overrides");

        require(throws_with(
                    []() {
                      simaai::neat::QuantTessOptions bad;
                      bad.num_buffers_locked = true;
                      bad.num_buffers_model = 5;
                      bad.num_buffers = 5;
                      (void)simaai::neat::QuantTess(bad);
                    },
                    "must be 4 (async) or 1 (sync)"),
                "QuantTess should enforce allowed locked num_buffers values");
      }
    }));
