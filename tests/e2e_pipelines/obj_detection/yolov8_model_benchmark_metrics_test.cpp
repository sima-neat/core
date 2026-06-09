#include "model/Model.h"

#include "e2e_pipelines/obj_detection/yolov8_test_utils.h"
#include "test_utils.h"

#include <filesystem>
#include <iostream>
#include <string>
#include <system_error>

namespace fs = std::filesystem;

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

    const simaai::neat::BenchmarkReport report = model.benchmark(10);
    std::cout << "[e2e_benchmark] sync_latency_ms=" << report.sync_latency_ms << "\n";
    std::cout << "[e2e_benchmark] sync_fps=" << report.sync_fps << "\n";
    std::cout << "[e2e_benchmark] async_fps=" << report.async_fps << "\n";
    std::cout << "[e2e_benchmark] avg_power_watts=" << report.avg_power_watts << "\n";
    std::cout << "[e2e_benchmark] energy_joules=" << report.energy_joules << "\n";

    require(report.sync_latency_ms > 0.0, "yolov8 benchmark: sync latency must be positive");
    require(report.sync_fps > 0.0, "yolov8 benchmark: sync FPS must be positive");
    require(report.async_fps > 0.0, "yolov8 benchmark: async FPS must be positive");
    require(report.avg_power_watts >= 0.0, "yolov8 benchmark: power must be non-negative");
    require(report.energy_joules >= 0.0, "yolov8 benchmark: energy must be non-negative");

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
