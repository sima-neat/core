#include "graph/Graph.h"
#include "graph/GraphSession.h"
#include "graph/StageExecutor.h"
#include "graph/nodes/PipelineNode.h"
#include "graph/nodes/StageNode.h"
#include "nodes/common/Output.h"
#include "test_main.h"
#include "test_utils.h"

namespace {

class BundleStage final : public simaai::neat::graph::StageExecutor {
public:
  void set_ports(const simaai::neat::graph::StagePorts& ports) override {
    const simaai::neat::graph::PortId only = ports.only_output();
    if (only != simaai::neat::graph::kInvalidPort) {
      out_port_ = only;
    }
  }

  void on_input(simaai::neat::graph::StageMsg&& msg,
                std::vector<simaai::neat::graph::StageOutMsg>& out) override {
    simaai::neat::Sample base = std::move(msg.sample);

    simaai::neat::Sample field_a = base;
    field_a.port_name = "encoded";

    simaai::neat::Sample field_b = base;
    field_b.port_name = "meta";

    simaai::neat::Sample bundle;
    bundle.kind = simaai::neat::SampleKind::Bundle;
    bundle.frame_id = base.frame_id;
    bundle.stream_id = base.stream_id;
    bundle.fields = {std::move(field_a), std::move(field_b)};

    out.push_back(simaai::neat::graph::StageOutMsg{
        .out_port = (out_port_ == simaai::neat::graph::kInvalidPort)
                        ? simaai::neat::graph::kInvalidPort
                        : out_port_,
        .sample = std::move(bundle)});
  }

private:
  simaai::neat::graph::PortId out_port_ = simaai::neat::graph::kInvalidPort;
};

std::shared_ptr<simaai::neat::graph::Node> make_bundle_node() {
  using simaai::neat::graph::PortDesc;
  using simaai::neat::graph::nodes::StageNode;
  StageNode::StageExecutorFactory factory = []() { return std::make_unique<BundleStage>(); };
  std::vector<PortDesc> inputs = {PortDesc{.name = "in", .spec = simaai::neat::OutputSpec{}}};
  std::vector<PortDesc> outputs = {PortDesc{.name = "out", .spec = simaai::neat::OutputSpec{}}};
  return std::make_shared<StageNode>("BundleStage", std::move(factory), std::move(inputs),
                                     std::move(outputs), "bundle");
}

} // namespace

RUN_TEST("hybrid_graph_bundle_test", [] {
  simaai::neat::graph::Graph g;

  auto stage = g.add(make_bundle_node());
  auto sink = g.add(std::make_shared<simaai::neat::graph::nodes::PipelineNode>(
      simaai::neat::nodes::Output(), "sink"));

  g.connect(stage, sink);

  simaai::neat::graph::GraphSession session(std::move(g));
  simaai::neat::graph::GraphRun run = session.build();

  simaai::neat::Sample input;
  input.kind = simaai::neat::SampleKind::Tensor;
  input.tensor = make_color_tensor(32, 24, simaai::neat::ImageSpec::PixelFormat::RGB);
  input.frame_id = 7;
  input.stream_id = "bundle";

  require(run.push(stage, input), "GraphRun::push failed");

  auto out = run.pull(sink, 15000);
  require(out.has_value(), "GraphRun::pull timed out");
  require(out->kind == simaai::neat::SampleKind::Bundle, "Expected bundle output");
  require(out->fields.size() == 2, "Bundle fields size mismatch");
  require(out->fields[0].port_name == "encoded", "Bundle field[0] name mismatch");
  require(out->fields[1].port_name == "meta", "Bundle field[1] name mismatch");
  require(out->frame_id == input.frame_id, "Bundle frame_id mismatch");
  require(out->stream_id == input.stream_id, "Bundle stream_id mismatch");

  run.stop();
});
