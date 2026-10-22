#include "model/Model.h"

#include "e2e_pipelines/e2e_utils.h"
#include "test_utils.h"

#include <algorithm>
#include <cmath>
#include <filesystem>
#include <iostream>
#include <string>
#include <tuple>
#include <utility>
#include <vector>

namespace {
using namespace simaai::neat;

// Match the CI YOLOv8s tensor input.
TensorList make_input(const Model& model, int phase) {
  const auto options = model.input_appsrc_options(true);
  const int height = options.height > 0 ? options.height : options.max_height;
  const int width = options.width > 0 ? options.width : options.max_width;
  const int depth = options.depth > 0 ? options.depth : options.max_depth;
  require(height > 0 && width > 0 && depth > 0, "YOLOv8s needs concrete input dimensions");
  std::vector<float> values(static_cast<std::size_t>(height) * width * depth);
  for (std::size_t i = 0; i < values.size(); ++i)
    values[i] = static_cast<float>((i * (phase ? 17 : 1) + phase * 13) % 251);
  return {Tensor::from_vector(values, {height, width, depth}, TensorMemory::EV74)};
}

using OutputSnapshot =
    std::vector<std::tuple<TensorDType, std::vector<int64_t>, std::vector<std::uint8_t>>>;

OutputSnapshot snapshot(const TensorList& outputs) {
  require(!outputs.empty(), "inference produced no outputs");
  OutputSnapshot result;
  for (const auto& tensor : outputs) {
    const auto contiguous = tensor.contiguous();
    const auto mapping = contiguous.view_read();
    require(mapping.data && mapping.size_bytes > 0, "inference returned empty tensor data");
    const auto* bytes = static_cast<const std::uint8_t*>(mapping.data);
    result.emplace_back(tensor.dtype, tensor.shape,
                        std::vector<std::uint8_t>(bytes, bytes + mapping.size_bytes));
  }
  return result;
}

void require_stage_stats(const BenchmarkReport& report) {
  require(report.mla_stage_stats_status == "ok",
          "MLA stage stats not collected: " + report.mla_stage_stats_status);
  require(report.mla_stage_stats.size() == 1 && report.mla_stage_stats[0].stage_index == 0,
          "the CI YOLOv8s fixture has one MLA stage");
  require(std::isfinite(report.mla_stage_stats[0].latency_us) &&
              report.mla_stage_stats[0].latency_us > 0,
          "MLA stage latency must be finite and positive");
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
    // Missing CI prerequisites must fail, never skip.
    require(!archive.empty(), "Failed to resolve the existing CI YOLOv8s model archive");
    Model model(archive);
    const std::vector<TensorList> inputs{make_input(model, 0), make_input(model, 1)};
    // Copy reusable output buffers while retaining Model::run()'s cached runner.
    const std::vector<OutputSnapshot> expected_outputs{snapshot(model.run(inputs[0], 30000)),
                                                       snapshot(model.run(inputs[1], 30000))};
    require(expected_outputs[0] != expected_outputs[1],
            "the two inputs must produce different outputs to detect stale results");
    BenchmarkOptions options;
    options.num_samples = 20;
    options.min_duration_ms = 0;
    require(!options.include_layer_stats, "layer collection must default to false");
    require(!options.include_mla_stage_stats, "MLA stage collection must default to false");
    const auto baseline = model.benchmark(options);
    require_headlines(baseline);
    require(baseline.layer_stats.empty() && baseline.layer_stats_status == "not requested",
            "ordinary benchmarks must not collect layer stats");
    require(baseline.mla_stage_stats.empty() && baseline.mla_stage_stats_status == "not requested",
            "ordinary benchmarks must not collect MLA stage stats");

    options.include_mla_stage_stats = true;
    const auto stage_only = model.benchmark(options);
    require_headlines(stage_only);
    require_stage_stats(stage_only);
    require(stage_only.layer_stats_status == "not requested",
            "MLA stage stats must not imply layer stats");
    options.include_mla_stage_stats = false;

    std::vector<std::pair<std::size_t, std::string>> expected_layers;
    for (int round = 0; round < 3; ++round) {
      options.include_layer_stats = true;
      options.include_mla_stage_stats = true;
      const auto profiled = model.benchmark(options);
      require_headlines(profiled);
      require_stage_stats(profiled);
      require(profiled.layer_stats_status == "ok",
              "layer stats not collected: " + profiled.layer_stats_status);
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

      // Verify the MIP and kernel profiling handoffs without a reset or delay.
      for (std::size_t i = 0; i < inputs.size(); ++i)
        require(snapshot(model.run(inputs[i], 30000)) == expected_outputs[i],
                "cached inference output changed after profiling");
      {
        auto runner = model.build(inputs.front());
        for (std::size_t i = 0; i < inputs.size(); ++i)
          require(snapshot(runner.run(inputs[i], 30000)) == expected_outputs[i],
                  "fresh inference output changed after profiling");
        runner.close();
      }
      options.include_layer_stats = false;
      options.include_mla_stage_stats = false;
      const auto ordinary = model.benchmark(options);
      require_headlines(ordinary);
      require(ordinary.layer_stats.empty() && ordinary.mla_stage_stats.empty(),
              "profiling must not leak into later ordinary reports");
      require(ordinary.throughput_samples == options.num_samples,
              "ordinary benchmark lost inputs after profiling");
      std::cerr << "[layer_stats] round=" << round + 1 << " layers=" << layers.size()
                << " stage_latency_us=" << profiled.mla_stage_stats[0].latency_us
                << " cached_inference=correct fresh_inference=correct normal_benchmark=ok\n";
    }
    std::cerr << "[OK] yolov8_model_layer_stats_test passed\n";
    return 0;
  } catch (const std::exception& error) {
    std::cerr << "[ERR] " << error.what() << '\n';
    return 1;
  }
}
