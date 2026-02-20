#include "graph/Graph.h"
#include "graph/GraphSession.h"
#include "graph/StageExecutor.h"
#include "graph/nodes/StageNode.h"
#include "test_main.h"
#include "test_utils.h"

#include <chrono>
#include <memory>
#include <string>
#include <vector>

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
    out.push_back(simaai::neat::graph::StageOutMsg{
        .out_port = (out_port_ == simaai::neat::graph::kInvalidPort)
                        ? simaai::neat::graph::kInvalidPort
                        : out_port_,
        .sample = std::move(msg.sample)});
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
  StageNode::StageExecutorFactory factory = [] { return std::make_unique<PassThroughStage>(); };
  std::vector<PortDesc> inputs = {PortDesc{.name = "in", .spec = simaai::neat::OutputSpec{}}};
  std::vector<PortDesc> outputs = {PortDesc{.name = "out", .spec = simaai::neat::OutputSpec{}}};
  return std::make_shared<StageNode>("PassThrough", std::move(factory), std::move(inputs),
                                     std::move(outputs), label);
}

std::shared_ptr<simaai::neat::graph::Node> make_fanout_node() {
  using simaai::neat::graph::PortDesc;
  using simaai::neat::graph::nodes::StageNode;
  StageNode::StageExecutorFactory factory = [] { return std::make_unique<FanOutStage>(); };
  std::vector<PortDesc> inputs = {PortDesc{.name = "in", .spec = simaai::neat::OutputSpec{}}};
  std::vector<PortDesc> outputs = {PortDesc{.name = "left", .spec = simaai::neat::OutputSpec{}},
                                   PortDesc{.name = "right", .spec = simaai::neat::OutputSpec{}}};
  return std::make_shared<StageNode>("FanOut", std::move(factory), std::move(inputs),
                                     std::move(outputs), "fanout");
}

} // namespace

RUN_TEST("unit_graph_backpressure_timeout_test", [] {
  simaai::neat::graph::Graph g;
  auto fan = g.add(make_fanout_node());
  auto sink_a = g.add(make_pass_node("sink_a"));
  auto sink_b = g.add(make_pass_node("sink_b"));
  g.connect(fan, sink_a, "left", "in");
  g.connect(fan, sink_b, "right", "in");

  simaai::neat::graph::GraphSession session(std::move(g));
  simaai::neat::graph::GraphRunOptions run_opt;
  run_opt.edge_queue = 1;
  run_opt.push_timeout_ms = 100;
  run_opt.pull_timeout_ms = 20;
  simaai::neat::graph::GraphRun run = session.build(run_opt);

  simaai::neat::Sample sample;
  sample.kind = simaai::neat::SampleKind::Tensor;
  sample.tensor = make_color_tensor(24, 16, simaai::neat::ImageSpec::PixelFormat::RGB, 0x2A);
  sample.stream_id = "backpressure-timeout";

  const auto start = std::chrono::steady_clock::now();
  bool saw_timeout = false;
  for (int i = 0; i < 200; ++i) {
    sample.frame_id = i;
    if (!run.push(fan, sample)) {
      saw_timeout = true;
      break;
    }
  }
  const auto elapsed_ms = std::chrono::duration_cast<std::chrono::milliseconds>(
                              std::chrono::steady_clock::now() - start)
                              .count();

  require(saw_timeout, "expected graph push to fail fast under backpressure");
  require(elapsed_ms < 5000, "graph push backpressure timeout took too long");
  const std::string err = run.last_error();
  require_contains(err, "backpressure timeout", "expected actionable backpressure timeout error");
  run.stop();
});
