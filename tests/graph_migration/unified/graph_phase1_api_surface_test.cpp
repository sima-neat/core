#include "pipeline/Graph.h"
#include "model/Model.h"
#include "nodes/common/Output.h"
#include "nodes/io/Input.h"
#include "test_main.h"

#include <type_traits>
#include <utility>

RUN_TEST("graph_migration_phase1_graph_api_surface_test", [] {
  static_assert(std::is_same_v<simaai::neat::GraphOptions, simaai::neat::GraphOptions>);
  static_assert(std::is_same_v<simaai::neat::Graph, simaai::neat::Graph>);
  static_assert(std::is_same_v<decltype(std::declval<simaai::neat::Graph&>().add(
                                   std::declval<const simaai::neat::Model&>())),
                               simaai::neat::Graph&>);
  static_assert(std::is_same_v<decltype(std::declval<const simaai::neat::Model&>().graph()),
                               simaai::neat::Graph>);
  static_assert(std::is_same_v<decltype(std::declval<const simaai::neat::Model&>().preprocess()),
                               simaai::neat::Graph>);
  static_assert(std::is_same_v<decltype(std::declval<const simaai::neat::Model&>().inference()),
                               simaai::neat::Graph>);
  static_assert(std::is_same_v<decltype(std::declval<const simaai::neat::Model&>().postprocess()),
                               simaai::neat::Graph>);

  simaai::neat::Graph graph;
  auto& graph_ref = graph.add(simaai::neat::nodes::Input()).add(simaai::neat::nodes::Output());
  require(&graph_ref == &graph, "Graph::add should keep Graph-style chaining");
  require(!graph.describe().empty(), "Graph::describe should use the Graph implementation");

  simaai::neat::Graph custom_graph;
  auto& custom_ref = custom_graph.custom("identity");
  require(&custom_ref == &custom_graph, "Graph::custom should preserve chaining");
});
