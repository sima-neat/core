// Benchmark a compiled model with deterministic synthetic inputs.
//
// Usage:
//   tutorial_003_benchmark_your_model --model /path/to/model.tar.gz [--samples 100]

#include "neat.h"

#include <iostream>
#include <stdexcept>
#include <string>

namespace {

bool get_arg(int argc, char** argv, const std::string& key, std::string& out) {
  for (int i = 1; i + 1 < argc; ++i) {
    if (key == argv[i]) {
      out = argv[i + 1];
      return true;
    }
  }
  return false;
}

int parse_int_arg(int argc, char** argv, const std::string& key, int def) {
  std::string value;
  if (!get_arg(argc, argv, key, value))
    return def;
  return std::stoi(value);
}

} // namespace

int main(int argc, char** argv) {
  try {
    std::string model_path;
    if (!get_arg(argc, argv, "--model", model_path)) {
      std::cerr << "Usage: tutorial_003_benchmark_your_model --model <path> [--samples <n>]\n";
      return 1;
    }
    const int samples = parse_int_arg(argc, argv, "--samples", 100);

    // CORE LOGIC
    // STEP load-model
    simaai::neat::Model model(model_path);
    // END STEP

    // STEP run-benchmark
    simaai::neat::BenchmarkReport report = model.benchmark(samples);
    // END STEP

    // STEP read-report
    std::cout << "report_latency_ms=" << report.latency_ms << "\n";
    std::cout << "report_fps=" << report.fps << "\n";
    std::cout << "report_avg_power_watts=" << report.avg_power_watts << "\n";
    std::cout << "report_energy_joules=" << report.energy_joules << "\n";
    // END STEP
    // END CORE LOGIC

    std::cout << "[OK] 003_benchmark_your_model\n";
    return 0;
  } catch (const std::exception& e) {
    std::cerr << "[FAIL] " << e.what() << "\n";
    return 1;
  }
}
