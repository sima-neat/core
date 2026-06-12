#include "builder/GraphPrinter.h"
#include "nodes/groups/UdpH264OutputGroup.h"
#include "nodes/groups/UdpOutputGroupG.h"
#include "nodes/io/UdpOutput.h"
#include "nodes/rtp/H264CapsFixup.h"
#include "nodes/sima/QuantTess.h"
#include "test_main.h"
#include "test_utils.h"

#include <nlohmann/json.hpp>

#include <functional>
#include <string>

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
        require_contains(snapshot, "1) H264EncodeSima",
                         "UdpOutputGroupG node[1] should be H264EncodeSima");
        require_contains(snapshot, "2) H264Parse", "UdpOutputGroupG node[2] should be H264Parse");
        require_contains(snapshot, "3) H264Packetize",
                         "UdpOutputGroupG node[3] should be H264Packetize");
        require_contains(snapshot, "4) UdpOutput", "UdpOutputGroupG node[4] should be UdpOutput");
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
        require_contains(fragment, "neatprocesscvu name=quant_test",
                         "QuantTess fragment name mismatch");
        require_contains(fragment, "stage-id=quant_test", "QuantTess stage-id mismatch");
        require_contains(fragment, "num-buffers=4", "QuantTess num-buffers mismatch");
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
