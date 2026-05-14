#include "nodes/common/Output.h"
#include "nodes/io/Input.h"
#include "perf_metrics_common.h"
#include "pipeline/Session.h"
#include "test_utils.h"

#include <cstdint>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

simaai::neat::Sample make_sample(int64_t frame_id, uint8_t fill = 0x2A) {
  simaai::neat::Sample sample;
  sample.kind = simaai::neat::SampleKind::Tensor;
  sample.tensor = make_color_tensor(96, 96, simaai::neat::ImageSpec::PixelFormat::RGB, fill);
  sample.frame_id = frame_id;
  sample.stream_id = "runtime_session_sync_rgb";
  return sample;
}

} // namespace

int main() {
  try {
    using namespace simaai::neat;

    const int iterations = sima_perf::env_int("SIMA_PERF_ITERS", 400);

    Session session;

    InputOptions input_opt;
    input_opt.media_type = "video/x-raw";
    input_opt.format = simaai::neat::FormatTag::RGB;
    input_opt.width = 96;
    input_opt.height = 96;
    input_opt.depth = 3;
    input_opt.use_simaai_pool = false;
    session.add(nodes::Input(input_opt));

    OutputOptions output_opt;
    output_opt.max_buffers = 128;
    output_opt.drop = false;
    output_opt.sync = false;
    session.add(nodes::Output(output_opt));

    RunOptions run_opt;
    run_opt.overflow_policy = OverflowPolicy::Block;
    run_opt.queue_depth = 8;
    const Sample seed = make_sample(0);
    const auto startup_t0 = sima_perf::Clock::now();
    Run run = session.build(SampleList{seed}, RunMode::Sync, run_opt);
    const auto startup_t1 = sima_perf::Clock::now();

    std::vector<double> latencies_ms;
    latencies_ms.reserve(static_cast<std::size_t>(iterations));

    simaai::neat::PowerMonitor power_monitor(sima_perf::power_options_from_env());
    power_monitor.start();
    const auto run_t0 = sima_perf::Clock::now();
    for (int i = 0; i < iterations; ++i) {
      const Sample input =
          make_sample(static_cast<int64_t>(i), static_cast<uint8_t>(0x10 + (i % 32)));
      const auto t0 = sima_perf::Clock::now();
      const SampleList outs = run.run(SampleList{input}, 5000);
      const auto t1 = sima_perf::Clock::now();
      if (outs.empty()) {
        throw std::runtime_error("sync run returned no outputs");
      }
      if (outs.front().frame_id != input.frame_id) {
        throw std::runtime_error("sync frame_id mismatch");
      }
      latencies_ms.push_back(sima_perf::elapsed_ms(t0, t1));
    }
    const auto run_t1 = sima_perf::Clock::now();
    power_monitor.stop();

    const RunStats stats = run.stats();
    run.stop();

    sima_perf::PerfMetrics metrics;
    const double total_s = sima_perf::elapsed_seconds(run_t0, run_t1);
    metrics.throughput = (total_s > 0.0) ? (static_cast<double>(iterations) / total_s) : 0.0;
    metrics.p50 = sima_perf::percentile(latencies_ms, 50.0);
    metrics.p95 = sima_perf::percentile(latencies_ms, 95.0);
    metrics.startup = sima_perf::elapsed_ms(startup_t0, startup_t1);
    metrics.rss_peak_kb = sima_perf::rss_peak_kb();
    metrics.input_drop_count = stats.inputs_dropped;
    metrics.output_drop_count = stats.outputs_dropped;

    const auto power_summary = power_monitor.summary();
    sima_perf::emit_metrics_json("runtime_session_sync_rgb", iterations, metrics, "sync",
                                 &power_summary);
    return 0;
  } catch (const std::exception& e) {
    std::cerr << "perf_runtime_session_sync_rgb_test exception: " << e.what() << "\n";
    return 1;
  }
}
