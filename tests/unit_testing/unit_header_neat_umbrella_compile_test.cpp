#include "neat.h"
#include "test_main.h"

RUN_TEST("unit_header_neat_umbrella_compile_test", ([] {
           simaai::neat::Session session;
           simaai::neat::Model::Options model_opt;
           simaai::neat::graph::Graph graph;
           (void)session;
           (void)model_opt;
           (void)graph;
         }));
