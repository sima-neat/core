#include "graph/GraphSession.h"
#include "model/Model.h"
#include "neat.h"
#include "neat/graph.h"
#include "neat/models.h"
#include "neat/node_groups.h"
#include "neat/nodes.h"
#include "neat/session.h"
#include "nodes/groups/ImageInputGroup.h"
#include "pipeline/Session.h"
#include "test_main.h"

RUN_TEST("unit_header_neat_compile_surface_test", ([] {
           simaai::neat::Session session;
           simaai::neat::RunOptions run_opt;
           simaai::neat::Sample sample;
           simaai::neat::Model::Options model_opt;
           simaai::neat::Model::SessionOptions model_session_opt;
           simaai::neat::VerboseOptions quiet = simaai::neat::VerboseOptions::quiet();
           simaai::neat::VerboseOptions production = simaai::neat::VerboseOptions::production();
           simaai::neat::VerboseOptions debug_plugins =
               simaai::neat::VerboseOptions::debug_plugins();
           simaai::neat::VerboseOptions debug_all = simaai::neat::VerboseOptions::debug_all();
           simaai::neat::stages::BoxDecodeOptions decode_opt(simaai::neat::BoxDecodeType::YoloV8);
           simaai::neat::graph::Graph graph;
           simaai::neat::graph::GraphRunOptions graph_run_opt;
           simaai::neat::graph::GraphSession* graph_session = nullptr;
           auto in = simaai::neat::nodes::Input();
           auto out = simaai::neat::nodes::Output();
           simaai::neat::nodes::groups::ImageInputGroupOptions image_opt;
           image_opt.path = "test.jpg";
           auto group = simaai::neat::nodes::groups::ImageInputGroup(image_opt);

           (void)session;
           (void)run_opt;
           (void)sample;
           (void)model_opt;
           (void)model_session_opt;
           (void)quiet;
           (void)production;
           (void)debug_plugins;
           (void)debug_all;
           (void)decode_opt;
           (void)graph;
           (void)graph_run_opt;
           (void)graph_session;
           (void)in;
           (void)out;
           (void)group;
         }));
