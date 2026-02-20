#include "nodes/groups/UdpH264OutputGroup.h"
#include "test_main.h"

RUN_TEST("unit_group_udp_h264_output_group_fragment_test", ([] {
           simaai::neat::nodes::groups::UdpH264OutputGroupOptions opt;
           opt.h264_caps = "video/x-h264,profile=\"high\"";
           opt.payload_type = 97;
           opt.config_interval = 2;
           opt.udp_host = "127.0.0.1";
           opt.udp_port = 5600;
           opt.udp_sync = false;
           opt.udp_async = false;

           const auto group = simaai::neat::nodes::groups::UdpH264OutputGroup(opt);
           const auto& nodes = group.nodes();

           require(nodes.size() == 4,
                   "UdpH264OutputGroup should include parse, caps(optional), pay, and udp nodes");
           require(nodes[0]->kind() == "H264Parse", "UdpH264OutputGroup node[0] kind mismatch");
           require(nodes[1]->kind() == "CustomNode", "UdpH264OutputGroup node[1] kind mismatch");
           require(nodes[2]->kind() == "H264Packetize", "UdpH264OutputGroup node[2] kind mismatch");
           require(nodes[3]->kind() == "UdpOutput", "UdpH264OutputGroup node[3] kind mismatch");

           const std::string caps_fragment = nodes[1]->backend_fragment(1);
           require_contains(caps_fragment, "capsfilter caps=\"video/x-h264,profile=\\\"high\\\"\"",
                            "UdpH264OutputGroup caps escaping mismatch");

           const std::string udp_fragment = nodes[3]->backend_fragment(3);
           require_contains(udp_fragment, "port=5600",
                            "UdpH264OutputGroup UDP port propagation mismatch");

           // Without explicit caps, the custom capsfilter node should not be inserted.
           simaai::neat::nodes::groups::UdpH264OutputGroupOptions no_caps = opt;
           no_caps.h264_caps.clear();
           const auto no_caps_group = simaai::neat::nodes::groups::UdpH264OutputGroup(no_caps);
           const auto& no_caps_nodes = no_caps_group.nodes();
           require(no_caps_nodes.size() == 3,
                   "UdpH264OutputGroup should skip caps node when h264_caps is empty");
           require(no_caps_nodes[0]->kind() == "H264Parse",
                   "UdpH264OutputGroup no-caps node[0] mismatch");
           require(no_caps_nodes[1]->kind() == "H264Packetize",
                   "UdpH264OutputGroup no-caps node[1] mismatch");
           require(no_caps_nodes[2]->kind() == "UdpOutput",
                   "UdpH264OutputGroup no-caps node[2] mismatch");
         }));
