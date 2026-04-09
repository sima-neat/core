#include "neat/nodes.h"
#include "test_main.h"

RUN_TEST("unit_header_neat_nodes_compile_test", ([] {
           auto in = simaai::neat::nodes::Input();
           auto out = simaai::neat::nodes::Output();
           auto v4l2 = simaai::neat::nodes::V4L2Input();
           (void)in;
           (void)out;
           (void)v4l2;
         }));
