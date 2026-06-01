#include "graph/Graph.h"
#include "graph/GraphBuild.h"
#include "graph/StageExecutor.h"
#include "graph_test_utils.h"
#include "graph/nodes/StageNode.h"
#include "test_main.h"

#include <memory>
#include <string>
#include <utility>
#include <vector>

namespace {

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

std::shared_ptr<simaai::neat::graph::Node> make_router_node() {
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

struct RoutedFrames {
  std::vector<int64_t> cpu;
  std::vector<int64_t> mla;
};

RoutedFrames run_once(const std::vector<RouteCase>& cases) {
  simaai::neat::graph::Graph g;
  const auto router = g.add(make_router_node());
  const auto cpu_sink = g.add(sima_test::make_pass_stage_node("CpuSink"));
  const auto mla_sink = g.add(sima_test::make_pass_stage_node("MlaSink"));

  g.connect(router, cpu_sink, "cpu", "in");
  g.connect(router, mla_sink, "mla", "in");
  simaai::neat::graph::GraphRun run = simaai::neat::graph::build(std::move(g));

  RoutedFrames out;
  int64_t frame = 0;
  for (const auto& c : cases) {
    simaai::neat::Sample sample = sima_test::make_tensor_sample(
        static_cast<int>(frame), c.stream_id, -1, static_cast<uint8_t>(frame & 0xFF));
    sample.port_name = c.port_name;

    require(run.push(router, simaai::neat::Sample{sample}), "routing regression push failed");

    const auto expected_sink = c.to_mla ? mla_sink : cpu_sink;
    const auto other_sink = c.to_mla ? cpu_sink : mla_sink;

    auto hit = run.pull(expected_sink, 2000);
    require(hit.has_value(), "routing regression pull timed out");
    require(hit->frame_id == frame, "routing regression frame_id mismatch");
    require(hit->stream_id == c.stream_id, "routing regression stream_id mismatch");

    if (c.to_mla) {
      out.mla.push_back(hit->frame_id);
    } else {
      out.cpu.push_back(hit->frame_id);
    }

    auto miss = run.pull(other_sink, 30);
    require(!miss.has_value(), "routing regression drifted to wrong sink");

    ++frame;
  }

  run.stop();
  return out;
}

} // namespace

RUN_TEST("graph_deterministic_routing_regression_test", ([] {
           const std::vector<RouteCase> cases = {
               RouteCase{.stream_id = "cpu-main", .port_name = "cpu", .to_mla = false},
               RouteCase{.stream_id = "mla-main", .port_name = "mla", .to_mla = true},
               RouteCase{.stream_id = "cpu-side", .port_name = "", .to_mla = false},
               RouteCase{.stream_id = "mla-side", .port_name = "", .to_mla = true},
               RouteCase{.stream_id = "cpu-fallback", .port_name = "unknown", .to_mla = false},
               RouteCase{.stream_id = "mla-fallback", .port_name = "mla", .to_mla = true},
               RouteCase{.stream_id = "cpu-main", .port_name = "cpu", .to_mla = false},
               RouteCase{.stream_id = "mla-main", .port_name = "mla", .to_mla = true},
           };

           const auto first = run_once(cases);
           const auto second = run_once(cases);

           require(first.cpu == second.cpu,
                   "deterministic routing regression: CPU route order changed across runs");
           require(first.mla == second.mla,
                   "deterministic routing regression: MLA route order changed across runs");

           require(first.cpu.size() == 4, "deterministic routing regression: CPU count mismatch");
           require(first.mla.size() == 4, "deterministic routing regression: MLA count mismatch");
         }));
