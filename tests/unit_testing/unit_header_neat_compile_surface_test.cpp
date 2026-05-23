#include "graph/GraphBuild.h"
#include "model/Model.h"
#include "neat.h"
#include "neat/graph.h"
#include "neat/models.h"
#include "neat/node_groups.h"
#include "neat/nodes.h"
#include "neat/runtime.h"
#include "nodes/groups/ImageInputGroup.h"
#include "pipeline/Graph.h"
#include "pipeline/GraphOptions.h"
#include "pipeline/GraphReport.h"
#include "pipeline/NeatError.h"
#include "test_main.h"

#include <stdexcept>
#include <type_traits>
#include <utility>

RUN_TEST(
    "unit_header_neat_compile_surface_test", ([] {
      static_assert(std::is_constructible_v<simaai::neat::Graph>);
      static_assert(std::is_constructible_v<simaai::neat::Graph, simaai::neat::GraphOptions>);
      static_assert(std::is_same_v<decltype(std::declval<simaai::neat::Graph&>().validate()),
                                   simaai::neat::GraphReport>);
      static_assert(std::is_base_of_v<std::runtime_error, simaai::neat::NeatError>);
      static_assert(std::is_same_v<decltype(std::declval<const simaai::neat::Model&>().graph()),
                                   simaai::neat::Graph>);
      static_assert(
          std::is_same_v<decltype(std::declval<const simaai::neat::Model&>().preprocess()),
                         simaai::neat::Graph>);
      static_assert(std::is_same_v<decltype(std::declval<const simaai::neat::Model&>().inference()),
                                   simaai::neat::Graph>);
      static_assert(
          std::is_same_v<decltype(std::declval<const simaai::neat::Model&>().postprocess()),
                         simaai::neat::Graph>);
      static_assert(std::is_same_v<decltype(std::declval<const simaai::neat::Model&>().graph()),
                                   simaai::neat::Graph>);
      static_assert(std::is_same_v<decltype(simaai::neat::graph::build(
                                       std::declval<simaai::neat::graph::Graph>())),
                                   simaai::neat::graph::GraphRun>);

      simaai::neat::Graph graph_pipeline;
      simaai::neat::GraphOptions graph_opt;
      simaai::neat::GraphReport graph_report;
      simaai::neat::RunOptions run_opt;
      simaai::neat::Sample sample;
      simaai::neat::Model::Options model_opt;
      simaai::neat::Model::RouteOptions model_route_opt;
      simaai::neat::VerboseOptions quiet = simaai::neat::VerboseOptions::quiet();
      simaai::neat::VerboseOptions production = simaai::neat::VerboseOptions::production();
      simaai::neat::VerboseOptions debug_plugins = simaai::neat::VerboseOptions::debug_plugins();
      simaai::neat::VerboseOptions debug_all = simaai::neat::VerboseOptions::debug_all();
      simaai::neat::stages::BoxDecodeOptions decode_opt(simaai::neat::BoxDecodeType::YoloV8);
      simaai::neat::graph::Graph graph;
      simaai::neat::graph::GraphRunOptions graph_run_opt;
      auto in = simaai::neat::nodes::Input();
      auto out = simaai::neat::nodes::Output();
      simaai::neat::nodes::groups::ImageInputGroupOptions image_opt;
      image_opt.path = "test.jpg";
      auto group = simaai::neat::nodes::groups::ImageInputGroup(image_opt);

      (void)graph_pipeline;
      (void)graph_opt;
      (void)graph_report;
      (void)run_opt;
      (void)sample;
      (void)model_opt;
      (void)model_route_opt;
      (void)quiet;
      (void)production;
      (void)debug_plugins;
      (void)debug_all;
      (void)decode_opt;
      (void)graph;
      (void)graph_run_opt;
      (void)in;
      (void)out;
      (void)group;
    }));
