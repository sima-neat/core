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
  auto& layers = document["layer_stats"] = pipeline_internal::ReportDocument::array();
  for (const auto& layer : layer_stats) {
    require_nonnegative(layer.total_execution_time_us, "total_execution_time_us");
    layers.push_back({{"stage_index", layer.stage_index},
                      {"name", layer.name},
                      {"total_execution_time_us", layer.total_execution_time_us}});
  }
  pipeline_internal::save_report(
      file_path, document,
      "# NEAT benchmark report. Latency: ms; throughput: inferences/s; power: W; energy: J.\n"
      "# Layer times: hardware run_time in us from separate profiling passes, not headline "
      "latency.\n",
      kContext);
  return file_path;
}

} // namespace simaai::neat
