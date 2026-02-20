#include "neat/session.h"
#include "neat/models.h"
#include "neat/graph.h"
#include "neat/nodes.h"
#include "neat/node_groups.h"
#include "test_main.h"

RUN_TEST("unit_header_neat_split_compile_test", ([] {
           simaai::neat::RunOptions run_opt;
           simaai::neat::Model::SessionOptions model_pipe_opt;
           simaai::neat::nodes::groups::ImageInputGroupOptions image_opt;
           simaai::neat::graph::GraphRunOptions graph_opt;
           (void)run_opt;
           (void)model_pipe_opt;
           (void)image_opt;
           (void)graph_opt;
         }));
