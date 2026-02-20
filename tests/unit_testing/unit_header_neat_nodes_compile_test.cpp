#include "neat/nodes.h"
#include "test_main.h"

RUN_TEST("unit_header_neat_nodes_compile_test", ([] {
           auto in = simaai::neat::nodes::Input();
           auto out = simaai::neat::nodes::Output();
           (void)in;
           (void)out;
         }));
