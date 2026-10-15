#include "model/Model.h"

#include "e2e_pipelines/e2e_utils.h"
#include "test_utils.h"

#include <algorithm>
#include <cmath>
#include <filesystem>
#include <iostream>
#include <string>
#include <utility>
#include <vector>

namespace {
using namespace simaai::neat;

// Use the same public tensor-input path and YOLOv8s fixture as model_execution_plan_test.
TensorList make_input(const Model& model) {
  const auto options = model.input_appsrc_options(true);
  const int height = options.height > 0 ? options.height : options.max_height;
  const int width = options.width > 0 ? options.width : options.max_width;
  const int depth = options.depth > 0 ? options.depth : options.max_depth;
  require(height > 0 && width > 0 && depth > 0, "YOLOv8s needs concrete input dimensions");
  const std::vector<float> zeros(static_cast<std::size_t>(height) * width * depth, 0.0F);
  return {Tensor::from_vector(zeros, {height, width, depth}, TensorMemory::EV74)};
}

void run_inference(Model& model, const TensorList& inputs) {
  auto runner = model.build(inputs);
  for (int i = 0; i < 3; ++i) {
    const auto outputs = runner.run(inputs, 30000);
    require(!outputs.empty(), "ordinary inference after profiling produced no outputs");
    for (const auto& tensor : outputs)
      require(!tensor.shape.empty(), "ordinary inference returned an empty tensor shape");
  }
  runner.close();
}

void require_headlines(const BenchmarkReport& report) {
  require(std::isfinite(report.latency_ms) && report.latency_ms > 0,
          "YOLOv8s benchmark latency must be finite and positive");
  require(std::isfinite(report.fps) && report.fps > 0,
          "YOLOv8s benchmark throughput must be finite and positive");
}
} // namespace

int main(int argc, char** argv) {
  std::cout.setf(std::ios::unitbuf);
  try {
    const auto root = argc > 1 ? std::filesystem::path(argv[1]) : std::filesystem::current_path();
    const auto archive = sima_e2e::resolve_yolov8s_tar(root);
    // Strict CI coverage: missing model, runtime, or profiling files must fail, never skip.
    require(!archive.empty(), "Failed to resolve the existing CI YOLOv8s model archive");
    Model model(archive);
    const auto inputs = make_input(model);
    BenchmarkOptions options;
    options.num_samples = 20;
    options.min_duration_ms = 0;
    require(!options.include_layer_stats, "layer collection must default to false");
    const auto baseline = model.benchmark(options);
    require_headlines(baseline);
    require(baseline.layer_stats.empty(), "ordinary benchmarks must not collect layer stats");

    std::vector<std::pair<std::size_t, std::string>> expected_layers;
    for (int round = 0; round < 3; ++round) {
      options.include_layer_stats = true;
      const auto profiled = model.benchmark(options);
      require_headlines(profiled);
      require(!profiled.layer_stats.empty(), "YOLOv8s profiling returned no layer records");
      require(std::any_of(profiled.layer_stats.begin(), profiled.layer_stats.end(),
                          [](const auto& layer) { return layer.total_execution_time_us > 0; }),
              "YOLOv8s profiling returned no hardware timings");
      std::vector<std::pair<std::size_t, std::string>> layers;
      for (const auto& layer : profiled.layer_stats) {
        require(!layer.name.empty(), "profiled layer names must not be empty");
        require(layer.stage_index == 0, "the CI YOLOv8s fixture has one executable stage");
        require(std::isfinite(layer.total_execution_time_us) && layer.total_execution_time_us >= 0,
                "layer timings must be finite and nonnegative, including real zero-time no-ops");
        layers.emplace_back(layer.stage_index, layer.name);
      }
      if (round == 0)
        expected_layers = layers;
      require(layers == expected_layers, "repeated profiling changed layer identity or order");

      // Exercise the public inference API immediately after MIP profiling. No reset or sleep:
      // MLA-RT must hand the hardware back to the kernel driver before benchmark() returns.
      run_inference(model, inputs);
      options.include_layer_stats = false;
      const auto ordinary = model.benchmark(options);
      require_headlines(ordinary);
      require(ordinary.layer_stats.empty(), "profiling must not leak into later ordinary reports");
      require(ordinary.throughput_samples == options.num_samples,
              "ordinary benchmark lost inputs after profiling");
      std::cerr << "[layer_stats] round=" << round + 1 << " layers=" << layers.size()
                << " normal_inference=ok normal_benchmark=ok\n";
    }
    std::cerr << "[OK] yolov8_model_layer_stats_test passed\n";
    return 0;
  } catch (const std::exception& error) {
    std::cerr << "[ERR] " << error.what() << '\n';
    return 1;
  }
}
