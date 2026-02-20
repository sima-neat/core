#include "pipeline/Session.h"
#include "model/Model.h"
#include "graph/GraphSession.h"
#include "nodes/groups/ImageInputGroup.h"
#include "test_main.h"

RUN_TEST("unit_header_neat_leaf_compile_test", ([] {
           simaai::neat::Session session;
           simaai::neat::Model::Options model_opt;
           simaai::neat::nodes::groups::ImageInputGroupOptions image_opt;
           simaai::neat::graph::GraphSession* graph_session = nullptr;
           (void)session;
           (void)model_opt;
           (void)image_opt;
           (void)graph_session;
         }));
