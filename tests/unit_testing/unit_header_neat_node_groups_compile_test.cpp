#include "neat/node_groups.h"
#include "test_main.h"

RUN_TEST("unit_header_neat_node_groups_compile_test", ([] {
           simaai::neat::nodes::groups::ImageInputGroupOptions image_opt;
           image_opt.path = "test.jpg";
           auto group = simaai::neat::nodes::groups::ImageInputGroup(image_opt);
           (void)group;
         }));
