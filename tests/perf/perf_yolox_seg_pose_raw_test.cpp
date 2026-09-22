#include "perf_metrics_common.h"
#include "boxdecode_yolox_seg_pose_raw_fixture.h"

int main() {
  try {
    auto* metrics_output = std::cout.rdbuf(std::cerr.rdbuf());
    yolox_test::RawHeads fixture;
    fixture.decode();
    fixture.verify_reference();
    const int iterations = sima_perf::env_int("SIMA_PERF_ITERS", 500);
    for (int i = 0; i < 50; ++i)
      fixture.decode();
    std::vector<double> samples;
    samples.reserve(iterations);
    const auto start = sima_perf::Clock::now();
    for (int i = 0; i < iterations; ++i) {
      const auto before = sima_perf::Clock::now();
      fixture.decode();
      samples.push_back(sima_perf::elapsed_ms(before, sima_perf::Clock::now()));
    }
    sima_perf::PerfMetrics metrics;
    metrics.throughput = iterations / sima_perf::elapsed_seconds(start, sima_perf::Clock::now());
    metrics.p50 = sima_perf::percentile(samples, 50.0);
    metrics.p95 = sima_perf::percentile(samples, 95.0);
    metrics.rss_peak_kb = sima_perf::rss_peak_kb();
    std::cout.rdbuf(metrics_output);
    sima_perf::emit_metrics_json("yolox_seg_pose_raw", iterations, metrics, "sync");
    std::cout.flush();
    std::cout.rdbuf(std::cerr.rdbuf());
    return 0;
  } catch (const std::exception& error) {
    std::cerr << error.what() << '\n';
    return 1;
  }
}
