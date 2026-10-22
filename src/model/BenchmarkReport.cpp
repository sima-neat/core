#include "model/Model.h"
#include "pipeline/internal/ReportFile.h"

#include <cmath>
#include <stdexcept>
#include <string>

namespace simaai::neat {
namespace {

constexpr const char* kContext = "BenchmarkReport::save";

void require_nonnegative(double value, const char* field) {
  if (!std::isfinite(value) || value < 0)
    throw std::invalid_argument(std::string(kContext) + ": " + field +
                                " must be finite and nonnegative");
}

} // namespace

std::filesystem::path BenchmarkReport::save(const std::filesystem::path& file_path) const {
  auto document = pipeline_internal::ReportDocument::object();
  const auto headline = [&](const char* key, double value) {
    require_nonnegative(value, key);
    document[key] = value;
  };
  headline("latency_ms", latency_ms);
  headline("latency_p50_ms", latency_p50_ms);
  headline("latency_p95_ms", latency_p95_ms);
  headline("latency_p99_ms", latency_p99_ms);
  headline("fps", fps);
  headline("avg_power_watts", avg_power_watts);
  headline("energy_joules", energy_joules);
  headline("peak_power_watts", peak_power_watts);
  headline("throughput_duration_ms", throughput_duration_ms);
  if (throughput_samples < 0)
    throw std::invalid_argument(std::string(kContext) + ": throughput_samples must be nonnegative");
  document["throughput_samples"] = throughput_samples;
  document["telemetry"] = pipeline_internal::telemetry_document(telemetry, kContext);
  auto& layers = document["mla_layer_stats"] = pipeline_internal::ReportDocument::array();
  for (const auto& layer : mla_layer_stats) {
    require_nonnegative(layer.latency_us, "latency_us");
    layers.push_back({{"stage_index", layer.stage_index},
                      {"name", layer.name},
                      {"latency_us", layer.latency_us}});
  }
  document["mla_layer_stats_status"] = mla_layer_stats_status;
  auto& stages = document["mla_stage_stats"] = pipeline_internal::ReportDocument::array();
  for (const auto& stage : mla_stage_stats) {
    require_nonnegative(stage.latency_us, "latency_us");
    stages.push_back({{"stage_index", stage.stage_index}, {"latency_us", stage.latency_us}});
  }
  document["mla_stage_stats_status"] = mla_stage_stats_status;
  pipeline_internal::save_report(
      file_path, document,
      "# NEAT benchmark report. Latency: ms; throughput: inferences/s; power: W; energy: J.\n"
      "# MLA layer latency: hardware run_time in us of one profiling pass per layer, not "
      "headline latency.\n"
      "# MLA stage latency: mean us of 200 uninterrupted mla-rt runs per stage.\n",
      kContext);
  return file_path;
}

} // namespace simaai::neat
