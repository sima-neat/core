#include "graph/Graph.h"
#include "graph/GraphSession.h"
#include "graph/StageExecutor.h"
#include "graph/nodes/StageNode.h"
#include "test_main.h"
#include "test_utils.h"

#include <memory>
#include <stdexcept>
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

std::shared_ptr<simaai::neat::graph::Node> make_pass_node(const std::string& label) {
  using simaai::neat::graph::PortDesc;
  using simaai::neat::graph::nodes::StageNode;
  StageNode::StageExecutorFactory factory = [] { return std::make_unique<PassThroughStage>(); };
  std::vector<PortDesc> inputs = {PortDesc{.name = "in", .spec = simaai::neat::OutputSpec{}}};
  std::vector<PortDesc> outputs = {PortDesc{.name = "out", .spec = simaai::neat::OutputSpec{}}};
  return std::make_shared<StageNode>("PassThrough", std::move(factory), std::move(inputs),
                                     std::move(outputs), label);
}

} // namespace

RUN_TEST("unit_graph_run_option_validation_test", [] {
  simaai::neat::graph::Graph g;
  auto pass = g.add(make_pass_node("only"));
  (void)pass;
  simaai::neat::graph::GraphSession session(std::move(g));

  simaai::neat::graph::GraphRunOptions run_opt;
  run_opt.push_timeout_ms = -1;

  bool threw = false;
  try {
    (void)session.build(run_opt);
  } catch (const std::invalid_argument& e) {
    threw = true;
    require_contains(std::string(e.what()), "push_timeout_ms",
                     "expected invalid timeout option error");
  }
  require(threw, "expected GraphSession::build to reject negative push_timeout_ms");
});
