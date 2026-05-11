#include "graph/Compiler.h"
#include "graph/Graph.h"
#include "graph/nodes/PipelineNode.h"
#include "graph_test_utils.h"
#include "nodes/common/Output.h"
#include "nodes/common/VideoConvert.h"
#include "nodes/io/Input.h"
#include "test_main.h"

#include <memory>
#include <vector>

namespace {

void require_compile_is_repeatable(simaai::neat::graph::Compiler& compiler,
                                   const simaai::neat::graph::Graph& graph,
                                   const std::string& label) {
  const auto first = compiler.compile(graph);
  const auto second = compiler.compile(graph);

  require(first.pipelines.size() == second.pipelines.size(),
          label + ": pipeline cardinality should be deterministic");
  require(first.stages.size() == second.stages.size(),
          label + ": stage cardinality should be deterministic");
  require(first.edges.size() == second.edges.size(),
          label + ": edge cardinality should be deterministic");

  for (size_t i = 0; i < first.pipelines.size(); ++i) {
    require(first.pipelines[i].id == second.pipelines[i].id,
            label + ": pipeline id should be deterministic");
    require(first.pipelines[i].node_ids == second.pipelines[i].node_ids,
            label + ": pipeline node ordering should be deterministic");
  }
}

} // namespace

RUN_TEST(
    "unit_graph_compiler_determinism_test", ([] {
      using simaai::neat::graph::Compiler;
      using simaai::neat::graph::Graph;
      using simaai::neat::graph::nodes::PipelineNode;

      Compiler compiler;

      // Deterministic segmenting/routing for a mixed pipeline-stage graph.
      {
        Graph g;
        const auto src = g.add(std::make_shared<PipelineNode>(simaai::neat::nodes::Input(), "src"));
        const auto convert =
            g.add(std::make_shared<PipelineNode>(simaai::neat::nodes::VideoConvert(), "convert"));
        const auto stage = g.add(sima_test::make_stage_passthrough("stage"));
        const auto sink = g.add(std::make_shared<PipelineNode>(simaai::neat::nodes::Output(), "sink"));
        g.connect(src, convert);
        g.connect(convert, stage);
        g.connect(stage, sink);

        const auto compiled = compiler.compile(g);
        require(compiled.pipelines.size() == 2,
                "mixed graph: compiler should produce two pipeline segments");
        require(compiled.stages.size() == 1, "mixed graph: compiler should produce one stage");
        require(compiled.edges.size() == 3, "mixed graph: compiler should preserve edge count");
        require(compiled.pipelines[0].node_ids.size() == 2,
                "mixed graph: first pipeline should contain src+convert");
        require(compiled.pipelines[1].node_ids.size() == 1,
                "mixed graph: second pipeline should contain sink");

        require_compile_is_repeatable(compiler, g, "mixed graph");
      }

      // Deterministic behavior across a different insertion order.
      {
        Graph g;
        const auto stage = g.add(sima_test::make_stage_passthrough("stage"));
        const auto sink = g.add(std::make_shared<PipelineNode>(simaai::neat::nodes::Output(), "sink"));
        const auto src = g.add(std::make_shared<PipelineNode>(simaai::neat::nodes::Input(), "src"));
        const auto convert =
            g.add(std::make_shared<PipelineNode>(simaai::neat::nodes::VideoConvert(), "convert"));
        g.connect(src, convert);
        g.connect(convert, stage);
        g.connect(stage, sink);

        const auto compiled = compiler.compile(g);
        require(compiled.pipelines.size() == 2,
                "permuted insertion: compiler should still produce two pipeline segments");
        require(compiled.stages.size() == 1,
                "permuted insertion: compiler should still produce one stage");
        require(compiled.edges.size() == 3,
                "permuted insertion: compiler should preserve edge count");

        require_compile_is_repeatable(compiler, g, "permuted insertion");
      }
    }));
