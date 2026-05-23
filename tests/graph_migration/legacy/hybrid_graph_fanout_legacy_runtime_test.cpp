#include "graph/Graph.h"
#include "graph/GraphBuild.h"
#include "graph/StageExecutor.h"
#include "graph/nodes/StageNode.h"
#include "test_main.h"
#include "test_utils.h"

namespace {

class PassThroughStage final : public simaai::neat::graph::StageExecutor {
public:
  void set_ports(const simaai::neat::graph::StagePorts& ports) override {
    const simaai::neat::graph::PortId only = ports.only_output();
    if (only != simaai::neat::graph::kInvalidPort) {
      out_port_ = only;
    }
  }

  void on_input(simaai::neat::graph::StageMsg&& msg,
                std::vector<simaai::neat::graph::StageOutMsg>& out) override {
    const simaai::neat::graph::PortId out_port = (out_port_ == simaai::neat::graph::kInvalidPort)
                                                     ? simaai::neat::graph::kInvalidPort
                                                     : out_port_;
    out.push_back(
        simaai::neat::graph::StageOutMsg{.out_port = out_port, .sample = std::move(msg.sample)});
  }

private:
  simaai::neat::graph::PortId out_port_ = simaai::neat::graph::kInvalidPort;
};

class FanOutStage final : public simaai::neat::graph::StageExecutor {
public:
  void set_ports(const simaai::neat::graph::StagePorts& ports) override {
    left_ = ports.out_port("left");
    right_ = ports.out_port("right");
  }

  void on_input(simaai::neat::graph::StageMsg&& msg,
                std::vector<simaai::neat::graph::StageOutMsg>& out) override {
    simaai::neat::Sample sample = std::move(msg.sample);
    if (left_ != simaai::neat::graph::kInvalidPort) {
      out.push_back(simaai::neat::graph::StageOutMsg{.out_port = left_, .sample = sample});
    }
    if (right_ != simaai::neat::graph::kInvalidPort) {
      out.push_back(
          simaai::neat::graph::StageOutMsg{.out_port = right_, .sample = std::move(sample)});
    }
  }

private:
  simaai::neat::graph::PortId left_ = simaai::neat::graph::kInvalidPort;
  simaai::neat::graph::PortId right_ = simaai::neat::graph::kInvalidPort;
};

std::shared_ptr<simaai::neat::graph::Node> make_pass_node(const std::string& label) {
  using simaai::neat::graph::PortDesc;
  using simaai::neat::graph::nodes::StageNode;
  StageNode::StageExecutorFactory factory = []() { return std::make_unique<PassThroughStage>(); };
  std::vector<PortDesc> inputs = {PortDesc{.name = "in", .spec = simaai::neat::OutputSpec{}}};
  std::vector<PortDesc> outputs = {PortDesc{.name = "out", .spec = simaai::neat::OutputSpec{}}};
  return std::make_shared<StageNode>("PassThrough", std::move(factory), std::move(inputs),
                                     std::move(outputs), label);
}

std::shared_ptr<simaai::neat::graph::Node> make_fanout_node() {
  using simaai::neat::graph::PortDesc;
  using simaai::neat::graph::nodes::StageNode;
  StageNode::StageExecutorFactory factory = []() { return std::make_unique<FanOutStage>(); };
  std::vector<PortDesc> inputs = {PortDesc{.name = "in", .spec = simaai::neat::OutputSpec{}}};
  std::vector<PortDesc> outputs = {PortDesc{.name = "left", .spec = simaai::neat::OutputSpec{}},
                                   PortDesc{.name = "right", .spec = simaai::neat::OutputSpec{}}};
  return std::make_shared<StageNode>("FanOut", std::move(factory), std::move(inputs),
                                     std::move(outputs), "fanout");
}

} // namespace

RUN_TEST("graph_migration_legacy_hybrid_graph_fanout_test", [] {
  simaai::neat::graph::Graph g;

  auto fan = g.add(make_fanout_node());
  auto sink_a = g.add(make_pass_node("sink_a"));
  auto sink_b = g.add(make_pass_node("sink_b"));

  g.connect(fan, sink_a, "left", "in");
  g.connect(fan, sink_b, "right", "in");
  simaai::neat::graph::GraphRun run = simaai::neat::graph::build(std::move(g));

  const int total = 5;
  for (int i = 0; i < total; ++i) {
    simaai::neat::Sample sample;
    sample.kind = simaai::neat::SampleKind::Tensor;
    sample.tensor = make_color_tensor(16, 12, simaai::neat::ImageSpec::PixelFormat::RGB);
    sample.frame_id = i;
    sample.stream_id = "fanout";
    require(run.push(fan, simaai::neat::Sample{sample}), "GraphRun::push failed");
  }

  for (int i = 0; i < total; ++i) {
    auto out_a = run.pull(sink_a, 2000);
    auto out_b = run.pull(sink_b, 2000);
    require(out_a.has_value(), "sink_a pull timed out");
    require(out_b.has_value(), "sink_b pull timed out");
    require(out_a->frame_id == i, "sink_a frame_id mismatch");
    require(out_b->frame_id == i, "sink_b frame_id mismatch");
  }

  run.stop();
});
