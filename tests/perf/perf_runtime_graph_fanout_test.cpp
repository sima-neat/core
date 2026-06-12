#include "graph/Graph.h"
#include "graph/GraphBuild.h"
#include "graph/StageExecutor.h"
#include "graph/nodes/FanOut.h"
#include "graph/nodes/StageNode.h"
#include "perf_metrics_common.h"
#include "test_utils.h"

#include <cstdint>
#include <iostream>
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
    out.push_back(
        simaai::neat::graph::StageOutMsg{.out_port = out_port_, .sample = std::move(msg.sample)});
  }

private:
  simaai::neat::graph::PortId out_port_ = simaai::neat::graph::kInvalidPort;
};

std::shared_ptr<simaai::neat::graph::Node>
make_pass_node(const std::string& label, const std::string& in_name, const std::string& out_name) {
  using simaai::neat::graph::PortDesc;
  using simaai::neat::graph::nodes::StageNode;
  StageNode::StageExecutorFactory factory = []() { return std::make_unique<PassThroughStage>(); };
  std::vector<PortDesc> inputs = {PortDesc{.name = in_name, .spec = simaai::neat::OutputSpec{}}};
  std::vector<PortDesc> outputs = {PortDesc{.name = out_name, .spec = simaai::neat::OutputSpec{}}};
  return std::make_shared<StageNode>("PassThrough", std::move(factory), std::move(inputs),
                                     std::move(outputs), label);
}

simaai::neat::Sample make_sample(int64_t frame_id, uint8_t fill = 0x66) {
  simaai::neat::Sample sample;
  sample.kind = simaai::neat::SampleKind::Tensor;
  sample.tensor = make_color_tensor(48, 32, simaai::neat::ImageSpec::PixelFormat::RGB, fill);
  sample.frame_id = frame_id;
  sample.stream_id = "runtime_graph_fanout";
  return sample;
}

} // namespace

int main() {
  try {
    using namespace simaai::neat::graph;

    const int iterations = sima_perf::env_int("SIMA_PERF_ITERS", 300);

    Graph graph;
    const auto fan = graph.add(nodes::FanOutNode({"left", "right"}, "fan", "in"));
    const auto sink_left = graph.add(make_pass_node("sink_left", "in", "out"));
    const auto sink_right = graph.add(make_pass_node("sink_right", "in", "out"));

    graph.connect(fan, sink_left, "left", "in");
    graph.connect(fan, sink_right, "right", "in");

    GraphRunOptions run_opt;
    run_opt.edge_queue = 128;
    run_opt.pull_timeout_ms = 5000;

    const auto startup_t0 = sima_perf::Clock::now();
    GraphRun run = simaai::neat::graph::build(std::move(graph), run_opt);
    const auto startup_t1 = sima_perf::Clock::now();

    std::vector<sima_perf::Clock::time_point> push_timestamps(static_cast<std::size_t>(iterations),
                                                              sima_perf::Clock::now());

    simaai::neat::PowerMonitor power_monitor(sima_perf::power_options_from_env());
    power_monitor.start();
    const auto run_t0 = sima_perf::Clock::now();
    for (int i = 0; i < iterations; ++i) {
      push_timestamps[static_cast<std::size_t>(i)] = sima_perf::Clock::now();
      const simaai::neat::Sample input =
          make_sample(static_cast<int64_t>(i), static_cast<uint8_t>(0x30 + (i % 64)));
      if (!run.push(fan, simaai::neat::Sample{input})) {
        throw std::runtime_error("graph fanout push failed");
      }
    }

    std::vector<double> latencies_ms;
    latencies_ms.reserve(static_cast<std::size_t>(iterations * 2));

    int outputs = 0;
    for (int i = 0; i < iterations; ++i) {
      auto out_left = run.pull(sink_left, 5000);
      auto out_right = run.pull(sink_right, 5000);
      if (!out_left.has_value() || !out_right.has_value()) {
        throw std::runtime_error("graph fanout pull timed out");
      }
      if (out_left->frame_id != i || out_right->frame_id != i) {
        throw std::runtime_error("graph fanout frame_id mismatch");
      }

      const auto now = sima_perf::Clock::now();
      const auto start = push_timestamps[static_cast<std::size_t>(i)];
      latencies_ms.push_back(sima_perf::elapsed_ms(start, now));
      latencies_ms.push_back(sima_perf::elapsed_ms(start, now));
      outputs += 2;
    }
    const auto run_t1 = sima_perf::Clock::now();
    power_monitor.stop();

    run.stop();

    sima_perf::PerfMetrics metrics;
    const double total_s = sima_perf::elapsed_seconds(run_t0, run_t1);
    metrics.throughput = (total_s > 0.0) ? (static_cast<double>(outputs) / total_s) : 0.0;
    metrics.p50 = sima_perf::percentile(latencies_ms, 50.0);
    metrics.p95 = sima_perf::percentile(latencies_ms, 95.0);
    metrics.startup = sima_perf::elapsed_ms(startup_t0, startup_t1);
    metrics.rss_peak_kb = sima_perf::rss_peak_kb();

    const auto power_summary = power_monitor.summary();
    sima_perf::emit_metrics_json("runtime_graph_fanout", iterations, metrics, "graph_fanout",
                                 &power_summary);
    return 0;
  } catch (const std::exception& e) {
    std::cerr << "perf_runtime_graph_fanout_test exception: " << e.what() << "\n";
    return 1;
  }
}
