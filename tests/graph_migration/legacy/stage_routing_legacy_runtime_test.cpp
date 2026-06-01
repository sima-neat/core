#include "graph/Graph.h"
#include "graph/GraphBuild.h"
#include "graph/StageExecutor.h"
#include "graph/nodes/StageNode.h"
#include "test_main.h"
#include "test_utils.h"

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

class RoutingStage final : public simaai::neat::graph::StageExecutor {
public:
  void set_ports(const simaai::neat::graph::StagePorts& ports) override {
    cpu_ = ports.out_port("cpu");
    mla_ = ports.out_port("mla");
  }

  void on_input(simaai::neat::graph::StageMsg&& msg,
                std::vector<simaai::neat::graph::StageOutMsg>& out) override {
    const bool to_mla = msg.sample.port_name == "mla" || msg.sample.stream_id.find("mla") == 0;
    const simaai::neat::graph::PortId target = to_mla ? mla_ : cpu_;
    if (target == simaai::neat::graph::kInvalidPort) {
      throw std::runtime_error("routing stage missing output port");
    }
    out.push_back(
        simaai::neat::graph::StageOutMsg{.out_port = target, .sample = std::move(msg.sample)});
  }

private:
  simaai::neat::graph::PortId cpu_ = simaai::neat::graph::kInvalidPort;
  simaai::neat::graph::PortId mla_ = simaai::neat::graph::kInvalidPort;
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

std::shared_ptr<simaai::neat::graph::Node> make_routing_node() {
  using simaai::neat::graph::PortDesc;
  using simaai::neat::graph::nodes::StageNode;
  StageNode::StageExecutorFactory factory = []() { return std::make_unique<RoutingStage>(); };
  std::vector<PortDesc> inputs = {PortDesc{.name = "in", .spec = simaai::neat::OutputSpec{}}};
  std::vector<PortDesc> outputs = {PortDesc{.name = "cpu", .spec = simaai::neat::OutputSpec{}},
                                   PortDesc{.name = "mla", .spec = simaai::neat::OutputSpec{}}};
  return std::make_shared<StageNode>("RoutingStage", std::move(factory), std::move(inputs),
                                     std::move(outputs), "router");
}

struct RouteCase {
  std::string stream_id;
  std::string port_name;
  bool to_mla = false;
};

} // namespace

RUN_TEST("graph_migration_legacy_stage_routing_regression_test", ([] {
           simaai::neat::graph::Graph g;
           const auto router = g.add(make_routing_node());
           const auto cpu_sink = g.add(make_pass_node("cpu_sink"));
           const auto mla_sink = g.add(make_pass_node("mla_sink"));

           g.connect(router, cpu_sink, "cpu", "in");
           g.connect(router, mla_sink, "mla", "in");
           simaai::neat::graph::GraphRun run = simaai::neat::graph::build(std::move(g));

           const std::vector<RouteCase> route_cases = {
               RouteCase{.stream_id = "cpu-main", .port_name = "cpu", .to_mla = false},
               RouteCase{.stream_id = "mla-main", .port_name = "mla", .to_mla = true},
               RouteCase{.stream_id = "cpu-side", .port_name = "", .to_mla = false},
               RouteCase{.stream_id = "mla-side", .port_name = "", .to_mla = true},
               RouteCase{.stream_id = "cpu-main", .port_name = "cpu", .to_mla = false},
               RouteCase{.stream_id = "mla-main", .port_name = "mla", .to_mla = true},
               RouteCase{.stream_id = "cpu-fallback", .port_name = "unknown", .to_mla = false},
               RouteCase{.stream_id = "mla-fallback", .port_name = "mla", .to_mla = true},
           };

           int frame = 0;
           for (const auto& c : route_cases) {
             simaai::neat::Sample sample;
             sample.kind = simaai::neat::SampleKind::Tensor;
             sample.tensor = make_color_tensor(20, 16, simaai::neat::ImageSpec::PixelFormat::RGB,
                                               static_cast<uint8_t>(frame & 0xFF));
             sample.frame_id = frame;
             sample.stream_id = c.stream_id;
             sample.port_name = c.port_name;

             require(run.push(router, simaai::neat::Sample{sample}), "routing test push failed");

             const auto expected_sink = c.to_mla ? mla_sink : cpu_sink;
             const auto other_sink = c.to_mla ? cpu_sink : mla_sink;

             auto out = run.pull(expected_sink, 2000);
             require(out.has_value(), "routing test pull timed out");
             require(out->frame_id == frame, "routing output frame mismatch");
             require(out->stream_id == c.stream_id, "routing output stream mismatch");

             auto wrong = run.pull(other_sink, 25);
             require(!wrong.has_value(), "routing drifted to wrong sink");

             ++frame;
           }

           run.stop();
         }));
