#include "nodes/io/UdpOutput.h"
#include "test_main.h"

RUN_TEST("unit_node_udp_output_fragment_test", ([] {
           simaai::neat::UdpOutputOptions opt;
           opt.host = "10.0.0.9";
           opt.port = 5050;
           opt.sync = true;
           opt.async = false;

           auto node = simaai::neat::nodes::UdpOutput(opt);
           require(node->kind() == "UdpOutput", "UdpOutput kind mismatch");

           const std::string fragment = node->backend_fragment(3);
           require_contains(fragment, "udpsink", "UdpOutput fragment should use udpsink");
           require_contains(fragment, "host=10.0.0.9", "UdpOutput host mismatch");
           require_contains(fragment, "port=5050", "UdpOutput port mismatch");
           require_contains(fragment, "sync=true", "UdpOutput sync mismatch");
           require_contains(fragment, "async=false", "UdpOutput async mismatch");

           const auto names = node->element_names(3);
           require(names.empty(), "UdpOutput element_names contract should be empty");
         }));
