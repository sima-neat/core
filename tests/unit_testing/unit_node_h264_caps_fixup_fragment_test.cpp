#include "nodes/rtp/H264CapsFixup.h"
#include "test_main.h"

RUN_TEST("unit_node_h264_caps_fixup_fragment_test", ([] {
           auto node = simaai::neat::nodes::H264CapsFixup(25, 1280, 720);
           require(node->kind() == "H264CapsFixup", "H264CapsFixup kind mismatch");

           const std::string fragment = node->backend_fragment(4);
           require_contains(fragment, "identity name=n4_h264_capsfix",
                            "H264CapsFixup fragment naming mismatch");

           const auto names = node->element_names(4);
           require(names.size() == 1, "H264CapsFixup should expose exactly one element name");
           require(names[0] == "n4_h264_capsfix", "H264CapsFixup element name mismatch");

           auto* raw = dynamic_cast<simaai::neat::H264CapsFixup*>(node.get());
           require(raw != nullptr, "H264CapsFixup dynamic_cast failed");
           require(raw->fallback_fps() == 25, "H264CapsFixup fallback_fps getter mismatch");
           require(raw->fallback_width() == 1280, "H264CapsFixup fallback_width getter mismatch");
           require(raw->fallback_height() == 720, "H264CapsFixup fallback_height getter mismatch");
         }));
