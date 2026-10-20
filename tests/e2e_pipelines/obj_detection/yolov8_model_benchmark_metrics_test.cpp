#include "model/Model.h"

#include "e2e_pipelines/obj_detection/yolov8_test_utils.h"
#include "test_utils.h"

#include <cmath>
#include <filesystem>
#include <iostream>
#include <string>
#include <system_error>

namespace fs = std::filesystem;

namespace {

constexpr int kBenchmarkSamples = 100;
constexpr double kDefaultMinDurationMs = 10000.0;
constexpr double kSentinelSampleIntervalMs = 2000.0;
// Two thirds of the samples Sentinel publishes during the default window, allowing for the
// window edges: 10 s at 2 s per sample is about 5 samples.
constexpr int kMinSentinelSamples =
    static_cast<int>(kDefaultMinDurationMs / kSentinelSampleIntervalMs * 2.0 / 3.0);

} // namespace

int main(int argc, char** argv) {
  try {
    const fs::path root = (argc > 1) ? fs::path(argv[1]) : fs::current_path();
    std::error_code ec;
    fs::create_directories(root / "tmp", ec);
    fs::current_path(root, ec);

    const std::string tar_gz = sima_yolov8_test::resolve_yolov8s_tar_or_skip(root);
    simaai::neat::Model model(tar_gz);

    const auto specs = model.input_specs();
    require(!specs.empty(), "yolov8 benchmark: model.input_specs() returned no inputs");
    const int compiled_batch_size = model.compiled_batch_size();
    require(compiled_batch_size > 0, "yolov8 benchmark: compiled batch size must be positive");
    std::cout << "[e2e_benchmark] compiled_batch_size=" << compiled_batch_size << "\n";

    const simaai::neat::BenchmarkReport report = model.benchmark(kBenchmarkSamples);
    const simaai::neat::Telemetry& telemetry = report.telemetry;
    std::cout << "[e2e_benchmark] latency_ms=" << report.latency_ms << "\n";
    std::cout << "[e2e_benchmark] fps=" << report.fps << "\n";
    std::cout << "[e2e_benchmark] throughput_samples=" << report.throughput_samples << "\n";
    std::cout << "[e2e_benchmark] throughput_duration_ms=" << report.throughput_duration_ms << "\n";
    std::cout << "[e2e_benchmark] avg_power_watts=" << report.avg_power_watts << "\n";
    std::cout << "[e2e_benchmark] peak_power_watts=" << report.peak_power_watts << "\n";
    std::cout << "[e2e_benchmark] energy_joules=" << report.energy_joules << "\n";
    std::cout << "[e2e_benchmark] telemetry_status=" << telemetry.status << "\n";
    std::cout << "[e2e_benchmark] telemetry_samples=" << telemetry.samples << "\n";

    require(std::isfinite(report.latency_ms), "yolov8 benchmark: latency must be finite");
    require(std::isfinite(report.fps), "yolov8 benchmark: FPS must be finite");
    require(report.latency_ms > 0.0, "yolov8 benchmark: latency must be positive");
    require(report.fps > 0.0, "yolov8 benchmark: FPS must be positive");
    require(report.throughput_samples >= kBenchmarkSamples,
            "yolov8 benchmark: throughput pass must measure at least num_samples inputs");
    require(report.throughput_duration_ms >= kDefaultMinDurationMs,
            "yolov8 benchmark: throughput pass must last at least the default minimum duration");

    // Board telemetry comes from the Sentinel daemon that the board installer provisions.
    require(telemetry.status == "ok",
            "yolov8 benchmark: Sentinel telemetry unavailable: " + telemetry.status);
    require(telemetry.samples >= kMinSentinelSamples,
            "yolov8 benchmark: expected >= " + std::to_string(kMinSentinelSamples) +
                " Sentinel samples, got " + std::to_string(telemetry.samples));
    require(report.avg_power_watts > 0.0, "yolov8 benchmark: power must be positive");
    require(report.peak_power_watts >= report.avg_power_watts,
            "yolov8 benchmark: peak power must be at least the average");
    require(report.energy_joules > 0.0, "yolov8 benchmark: energy must be positive");
    require(telemetry.die_temperature_max_c.samples > 0,
            "yolov8 benchmark: die temperature must be measured");
    require(telemetry.cpu_usage_pct.samples > 0, "yolov8 benchmark: CPU must be measured");
    require(telemetry.mla_memory_mib.maximum > 0.0,
            "yolov8 benchmark: the loaded model must hold MLA memory");

    // A second run with no minimum duration keeps the count-based timing. Its trace starting at
    // all also proves the first run stopped its own trace.
    simaai::neat::BenchmarkOptions short_options;
    short_options.num_samples = kBenchmarkSamples;
    short_options.min_duration_ms = 0;
    const simaai::neat::BenchmarkReport short_report = model.benchmark(short_options);
    require(short_report.throughput_samples == kBenchmarkSamples,
            "yolov8 benchmark: min_duration_ms=0 must measure exactly num_samples inputs");
    require(short_report.telemetry.status.find("HTTP 409") == std::string::npos,
            "yolov8 benchmark: the first run left its Sentinel trace active");
    // This E2E test intentionally does not compare Model::benchmark() against a separate
    // same-device single-flight or burst "oracle" run.  Those are independent wall-clock windows
    // and can fail under scheduler, thermal, or device-load variance even when the benchmark
    // contract is correct.  Unit tests cover the same-window MeasureReport arithmetic and
    // Model::benchmark source contract; this device test is a real-model smoke/health check.
    if (compiled_batch_size == 1) {
      std::cout << "[e2e_benchmark] note: this YOLO fixture is batch=1, so it cannot expose a "
                   "batches/s-vs-inferences/s FPS bug by numeric comparison.\n";
    }

    std::cout << "[OK] yolov8_model_benchmark_metrics_test passed\n";
    return 0;
  } catch (const SkipTest& e) {
    std::cout << "[SKIP] " << e.what() << "\n";
    return skip_long_test(e.what());
  } catch (const std::exception& e) {
    if (is_dispatcher_unavailable(e.what())) {
      return skip_long_test("dispatcher unavailable");
    }
    std::cerr << "[ERR] " << e.what() << "\n";
    return 1;
  }
}
