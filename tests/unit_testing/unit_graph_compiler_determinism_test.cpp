#include "graph/Compiler.h"
#include "graph/Graph.h"
#include "graph/StageExecutor.h"
#include "graph/nodes/PipelineNode.h"
#include "graph/nodes/StageNode.h"
#include "nodes/common/Output.h"
#include "nodes/common/VideoConvert.h"
#include "nodes/io/Input.h"
#include "test_main.h"
#include "test_utils.h"

#include <functional>
#include <memory>
#include <string>
#include <vector>

namespace {

class NoopStage final : public simaai::neat::graph::StageExecutor {
public:
  void on_input(simaai::neat::graph::StageMsg&&,
                std::vector<simaai::neat::graph::StageOutMsg>&) override {}
};

std::shared_ptr<simaai::neat::graph::Node>
make_stage_node(const std::string& kind, std::vector<simaai::neat::graph::PortDesc> inputs,
                std::vector<simaai::neat::graph::PortDesc> outputs,
                simaai::neat::graph::nodes::StageNode::OutputSpecFn out_fn = {}) {
  using simaai::neat::graph::nodes::StageNode;
  StageNode::StageExecutorFactory factory = []() { return std::make_unique<NoopStage>(); };
  return std::make_shared<StageNode>(kind, std::move(factory), std::move(inputs),
                                     std::move(outputs), kind, std::move(out_fn));
}

std::shared_ptr<simaai::neat::graph::Node> make_stage_source(const std::string& kind,
                                                             const simaai::neat::OutputSpec& spec) {
  using simaai::neat::graph::PortDesc;
  const auto out_fn = [spec](const std::vector<simaai::neat::OutputSpec>&,
                             simaai::neat::graph::PortId) { return spec; };
  std::vector<PortDesc> inputs;
  std::vector<PortDesc> outputs = {PortDesc{.name = "out", .spec = spec}};
  return make_stage_node(kind, std::move(inputs), std::move(outputs), out_fn);
}

std::shared_ptr<simaai::neat::graph::Node> make_stage_passthrough(const std::string& kind,
                                                                  int max_in_edges = 1) {
  using simaai::neat::graph::PortDesc;
  const auto out_fn = [](const std::vector<simaai::neat::OutputSpec>& in,
                         simaai::neat::graph::PortId) {
    if (!in.empty())
      return in.front();
    return simaai::neat::OutputSpec{};
  };
  std::vector<PortDesc> inputs = {
      PortDesc{.name = "in", .spec = simaai::neat::OutputSpec{}, .max_in_edges = max_in_edges}};
  std::vector<PortDesc> outputs = {PortDesc{.name = "out", .spec = simaai::neat::OutputSpec{}}};
  return make_stage_node(kind, std::move(inputs), std::move(outputs), out_fn);
}

bool throws_with(const std::function<void()>& fn, const std::string& needle) {
  try {
    fn();
  } catch (const std::exception& e) {
    if (needle.empty())
      return true;
    return std::string(e.what()).find(needle) != std::string::npos;
  }
  return false;
}

} // namespace

RUN_TEST(
    "unit_graph_compiler_determinism_test", ([] {
      using simaai::neat::OutputSpec;
      using simaai::neat::graph::Compiler;
      using simaai::neat::graph::Graph;
      using simaai::neat::graph::PortDesc;
      using simaai::neat::graph::nodes::PipelineNode;

      Compiler compiler;

      // Deterministic segmenting and routing/cardinality for a mixed pipeline/stage graph.
      Graph graph;
      const auto src =
          graph.add(std::make_shared<PipelineNode>(simaai::neat::nodes::Input(), "src"));
      const auto convert =
          graph.add(std::make_shared<PipelineNode>(simaai::neat::nodes::VideoConvert(), "convert"));
      const auto stage = graph.add(make_stage_passthrough("stage"));
      const auto sink =
          graph.add(std::make_shared<PipelineNode>(simaai::neat::nodes::Output(), "sink"));

      graph.connect(src, convert);
      graph.connect(convert, stage);
      graph.connect(stage, sink);

      const auto first = compiler.compile(graph);
      const auto second = compiler.compile(graph);

      require(first.pipelines.size() == 2, "compiler should produce two pipeline segments");
      require(first.stages.size() == 1, "compiler should produce one stage node");
      require(first.edges.size() == 3, "compiler should preserve graph edge cardinality");

      require(first.pipelines[0].node_ids.size() == 2,
              "first pipeline segment should contain contiguous source+convert nodes");
      require(first.pipelines[1].node_ids.size() == 1,
              "second pipeline segment should contain sink node only");
      require(first.pipelines.size() == second.pipelines.size(),
              "compiler output pipeline cardinality must be deterministic");
      for (size_t i = 0; i < first.pipelines.size(); ++i) {
        require(first.pipelines[i].node_ids == second.pipelines[i].node_ids,
                "compiler pipeline segmentation order must be deterministic");
        require(first.pipelines[i].id == second.pipelines[i].id,
                "compiler segment ids must be deterministic");
      }

      // Topology validation: invalid DAG.
      {
        Graph cycle;
        const auto a = cycle.add(make_stage_passthrough("a"));
        const auto b = cycle.add(make_stage_passthrough("b"));
        cycle.connect(a, b);
        cycle.connect(b, a);

        require(throws_with([&]() { (void)compiler.compile(cycle); }, "graph must be a DAG"),
                "compiler should reject cyclic graphs");
      }

      // Topology validation: unknown input port wiring.
      {
        Graph bad_port;
        const auto src_stage = bad_port.add(make_stage_source("src", OutputSpec{}));
        const auto sink_stage = bad_port.add(make_stage_passthrough("sink"));
        bad_port.connect(src_stage, sink_stage, "out", "missing_port");

        require(throws_with([&]() { (void)compiler.compile(bad_port); }, "unknown input port"),
                "compiler should reject edges wired to undeclared input ports");
      }

      // Topology validation: duplicate input port declaration on a node.
      {
        Graph duplicate_ports;
        std::vector<PortDesc> inputs = {PortDesc{.name = "in", .spec = OutputSpec{}},
                                        PortDesc{.name = "in", .spec = OutputSpec{}}};
        std::vector<PortDesc> outputs = {PortDesc{.name = "out", .spec = OutputSpec{}}};
        duplicate_ports.add(make_stage_node("dup_inputs", std::move(inputs), std::move(outputs)));

        require(
            throws_with([&]() { (void)compiler.compile(duplicate_ports); }, "duplicate port name"),
            "compiler should reject duplicate input port names");
      }

      // Topology validation: max_in_edges contract.
      {
        Graph fanin_limited;
        const OutputSpec spec = OutputSpec{
            .media_type = "video/x-raw", .format = "RGB", .width = 32, .height = 24, .depth = 3};
        const auto src_a = fanin_limited.add(make_stage_source("src_a", spec));
        const auto src_b = fanin_limited.add(make_stage_source("src_b", spec));
        const auto sink_stage = fanin_limited.add(make_stage_passthrough("sink", 1));

        fanin_limited.connect(src_a, sink_stage, "out", "in");
        fanin_limited.connect(src_b, sink_stage, "out", "in");

        require(
            throws_with([&]() { (void)compiler.compile(fanin_limited); }, "exceeds max_in_edges"),
            "compiler should enforce max_in_edges contracts");
      }

      // Spec consistency validation on multi-edge merge.
      {
        Graph spec_mismatch;
        const auto rgb_src = spec_mismatch.add(make_stage_source(
            "rgb_src",
            OutputSpec{.media_type = "video/x-raw", .format = "RGB", .width = 64, .height = 48}));
        const auto nv12_src = spec_mismatch.add(make_stage_source(
            "nv12_src",
            OutputSpec{.media_type = "video/x-raw", .format = "NV12", .width = 64, .height = 48}));
        const auto merge = spec_mismatch.add(make_stage_passthrough("merge", 0));

        spec_mismatch.connect(rgb_src, merge, "out", "in");
        spec_mismatch.connect(nv12_src, merge, "out", "in");

        require(
            throws_with([&]() { (void)compiler.compile(spec_mismatch); }, "input spec mismatch"),
            "compiler should reject conflicting merged input specs");
      }
    }));
