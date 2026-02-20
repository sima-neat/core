#include "nodes/groups/UdpOutputGroupG.h"
#include "test_main.h"

RUN_TEST("unit_group_udp_output_group_g_fragment_test", ([] {
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

           const auto group = simaai::neat::nodes::groups::UdpOutputGroupG(opt);
           const auto& nodes = group.nodes();

           require(nodes.size() == 5,
                   "UdpOutputGroupG should include render, encode, parse, pay, and udp nodes");
           require(nodes[0]->kind() == "CustomNode", "UdpOutputGroupG node[0] kind mismatch");
           require(nodes[1]->kind() == "H264EncodeSima", "UdpOutputGroupG node[1] kind mismatch");
           require(nodes[2]->kind() == "H264Parse", "UdpOutputGroupG node[2] kind mismatch");
           require(nodes[3]->kind() == "H264Packetize", "UdpOutputGroupG node[3] kind mismatch");
           require(nodes[4]->kind() == "UdpOutput", "UdpOutputGroupG node[4] kind mismatch");

           const std::string render_fragment = nodes[0]->backend_fragment(0);
           require_contains(render_fragment, "simaai_sampledemux",
                            "UdpOutputGroupG render fragment missing sample demux");
           require_contains(render_fragment, "config=\"/tmp/render_config.json\"",
                            "UdpOutputGroupG render config propagation mismatch");

           const std::string packetize_fragment = nodes[3]->backend_fragment(3);
           require_contains(packetize_fragment, "pt=98",
                            "UdpOutputGroupG payload_type propagation mismatch");
           require_contains(packetize_fragment, "config-interval=3",
                            "UdpOutputGroupG config_interval propagation mismatch");

           const std::string udp_fragment = nodes[4]->backend_fragment(4);
           require_contains(udp_fragment, "host=127.0.0.1",
                            "UdpOutputGroupG udp host propagation mismatch");
           require_contains(udp_fragment, "port=5700",
                            "UdpOutputGroupG udp port propagation mismatch");
           require_contains(udp_fragment, "sync=true",
                            "UdpOutputGroupG udp sync propagation mismatch");
           require_contains(udp_fragment, "async=false",
                            "UdpOutputGroupG udp async propagation mismatch");
         }));
