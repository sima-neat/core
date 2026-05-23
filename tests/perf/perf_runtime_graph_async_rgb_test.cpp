#include "nodes/common/Output.h"
#include "nodes/io/Input.h"
#include "perf_metrics_common.h"
#include "pipeline/Graph.h"
#include "test_utils.h"

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

simaai::neat::Sample make_sample(int64_t frame_id, uint8_t fill = 0x44) {
  simaai::neat::Sample sample;
  sample.kind = simaai::neat::SampleKind::Tensor;
  sample.tensor = make_color_tensor(96, 96, simaai::neat::ImageSpec::PixelFormat::RGB, fill);
  sample.frame_id = frame_id;
  sample.stream_id = "runtime_session_async_rgb";
  return sample;
}

} // namespace

int main() {
  try {
    using namespace simaai::neat;

    const int iterations = sima_perf::env_int("SIMA_PERF_ITERS", 400);

    Graph graph;

    InputOptions input_opt;
    input_opt.payload_type = simaai::neat::PayloadType::Image;
    input_opt.format = simaai::neat::FormatTag::RGB;
    input_opt.width = 96;
    input_opt.height = 96;
    input_opt.depth = 3;
    input_opt.use_simaai_pool = false;
    graph.add(nodes::Input(input_opt));

    OutputOptions output_opt;
    output_opt.max_buffers = std::max(256, iterations);
    output_opt.drop = false;
    output_opt.sync = false;
    graph.add(nodes::Output(output_opt));

    RunOptions run_opt;
    run_opt.overflow_policy = OverflowPolicy::Block;
    run_opt.queue_depth = std::max(32, iterations / 2);
    const Sample seed = make_sample(0);
    const auto startup_t0 = sima_perf::Clock::now();
    Run run = graph.build(Sample{seed}, RunMode::Async, run_opt);
    const auto startup_t1 = sima_perf::Clock::now();

    std::vector<sima_perf::Clock::time_point> push_timestamps(static_cast<std::size_t>(iterations),
                                                              sima_perf::Clock::now());

    simaai::neat::PowerMonitor power_monitor(sima_perf::power_options_from_env());
    power_monitor.start();
    const auto run_t0 = sima_perf::Clock::now();
    for (int i = 0; i < iterations; ++i) {
      push_timestamps[static_cast<std::size_t>(i)] = sima_perf::Clock::now();
      const Sample input =
          make_sample(static_cast<int64_t>(i), static_cast<uint8_t>(0x40 + (i % 32)));
      if (!run.push(Sample{input})) {
        throw std::runtime_error("async push failed");
      }
    }

    run.close_input();

    std::vector<double> latencies_ms;
    latencies_ms.reserve(static_cast<std::size_t>(iterations));

    int outputs = 0;
    while (outputs < iterations) {
      auto out = run.pull(5000);
      if (!out.has_value()) {
        throw std::runtime_error("async pull timed out before receiving all outputs");
      }
      const int64_t frame_id = out->frame_id;
      if (frame_id >= 0 && frame_id < iterations) {
        const auto end = sima_perf::Clock::now();
        const auto start = push_timestamps[static_cast<std::size_t>(frame_id)];
        latencies_ms.push_back(sima_perf::elapsed_ms(start, end));
      }
      ++outputs;
    }
    const auto run_t1 = sima_perf::Clock::now();
    power_monitor.stop();

    const RunStats stats = run.stats();
    run.stop();

    sima_perf::PerfMetrics metrics;
    const double total_s = sima_perf::elapsed_seconds(run_t0, run_t1);
    metrics.throughput = (total_s > 0.0) ? (static_cast<double>(outputs) / total_s) : 0.0;
    metrics.p50 = sima_perf::percentile(latencies_ms, 50.0);
    metrics.p95 = sima_perf::percentile(latencies_ms, 95.0);
    metrics.startup = sima_perf::elapsed_ms(startup_t0, startup_t1);
    metrics.rss_peak_kb = sima_perf::rss_peak_kb();
    metrics.input_drop_count = stats.inputs_dropped;
    metrics.output_drop_count = stats.outputs_dropped;

    const auto power_summary = power_monitor.summary();
    sima_perf::emit_metrics_json("runtime_session_async_rgb", iterations, metrics, "async",
                                 &power_summary);
    return 0;
  } catch (const std::exception& e) {
    std::cerr << "perf_runtime_graph_async_rgb_test exception: " << e.what() << "\n";
    return 1;
  }
}
