#include "nodes/groups/UdpOutputGroup.h"
#include "test_main.h"

#include <string>

RUN_TEST("unit_udp_output_group_test", ([] {
           using simaai::neat::nodes::groups::UdpOutputNodeGroup;
           using simaai::neat::nodes::groups::UdpOutputNodeGroupOptions;

           UdpOutputNodeGroup group;
           UdpOutputNodeGroupOptions bad;
           bad.h264_caps.clear();

           std::string err;
           require(!group.init(bad, 1, &err), "UdpOutputNodeGroup should reject empty h264_caps");
           require_contains(err, "missing h264_caps",
                            "UdpOutputNodeGroup h264_caps error mismatch");

           err.clear();
           bad.h264_caps = "video/x-h264";
           require(!group.init(bad, 0, &err), "UdpOutputNodeGroup should reject zero streams");
           require_contains(err, "streams must be > 0",
                            "UdpOutputNodeGroup streams error mismatch");
         }));
