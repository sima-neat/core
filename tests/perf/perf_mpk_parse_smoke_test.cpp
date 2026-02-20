#include "mpk/MpKLoader.h"
#include "mpk/PipelineSequence.h"
#include "perf_metrics_common.h"

#include <filesystem>
#include <iostream>
#include <string>
#include <vector>

namespace {

namespace fs = std::filesystem;

fs::path find_repo_root() {
  std::error_code ec;
  fs::path path = fs::current_path(ec);
  if (ec) {
    return fs::current_path();
  }
  while (!path.empty()) {
    if (fs::exists(path / "tests" / "assets" / "mpk" / "valid" / "basic_valid.mpk", ec) && !ec) {
      return path;
    }
    const fs::path parent = path.parent_path();
    if (parent == path) {
      break;
    }
    path = parent;
  }
  return fs::current_path();
}

} // namespace

int main() {
  try {
    const int iterations = sima_perf::env_int("SIMA_PERF_ITERS", 400);

    const fs::path repo_root = find_repo_root();
    const fs::path fixture = repo_root / "tests" / "assets" / "mpk" / "valid" / "basic_valid.mpk";
    if (!fs::exists(fixture)) {
      std::cerr << "perf_mpk_parse_smoke_test: missing fixture " << fixture << "\n";
      return 2;
    }

    const fs::path out_root = fs::temp_directory_path() / "sima_neat_perf_mpk_parse";
    std::error_code ec;
    fs::create_directories(out_root, ec);

    const auto startup_t0 = sima_perf::Clock::now();
    const auto extracted =
        simaai::neat::mpk::MpKLoader::extract(fixture.string(), out_root.string());
    const auto startup_t1 = sima_perf::Clock::now();

    std::vector<double> latencies_ms;
    latencies_ms.reserve(static_cast<std::size_t>(iterations));

    const auto run_t0 = sima_perf::Clock::now();
    for (int i = 0; i < iterations; ++i) {
      const auto t0 = sima_perf::Clock::now();
      const auto seq = simaai::neat::mpk::load_pipeline_sequence(extracted.etc_dir);
      (void)seq;
      const auto t1 = sima_perf::Clock::now();
      latencies_ms.push_back(sima_perf::elapsed_ms(t0, t1));
    }
    const auto run_t1 = sima_perf::Clock::now();

    sima_perf::PerfMetrics metrics;
    const double total_s = sima_perf::elapsed_seconds(run_t0, run_t1);
    metrics.throughput = (total_s > 0.0) ? (static_cast<double>(iterations) / total_s) : 0.0;
    metrics.p50 = sima_perf::percentile(latencies_ms, 50.0);
    metrics.p95 = sima_perf::percentile(latencies_ms, 95.0);
    metrics.startup = sima_perf::elapsed_ms(startup_t0, startup_t1);
    metrics.rss_peak_kb = sima_perf::rss_peak_kb();

    sima_perf::emit_metrics_json("mpk_parse_smoke", iterations, metrics, "mpk_parse");
    return 0;
  } catch (const std::exception& e) {
    std::cerr << "perf_mpk_parse_smoke_test exception: " << e.what() << "\n";
    return 1;
  }
}
