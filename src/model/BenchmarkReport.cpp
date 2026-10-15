#include "model/Model.h"

#include <glib.h>
#include <yaml-cpp/yaml.h>

#include <charconv>
#include <cmath>
#include <stdexcept>

namespace simaai::neat {
namespace {

void require_nonnegative(double value, const char* field) {
  if (!std::isfinite(value) || value < 0)
    throw std::invalid_argument(std::string("BenchmarkReport::save: ") + field +
                                " must be finite and nonnegative");
}

// PyYAML requires a decimal mantissa in scientific notation, e.g. 1.0e-09.
std::string yaml_number(double value) {
  char buffer[64];
  const auto result = std::to_chars(buffer, buffer + sizeof(buffer), value);
  if (result.ec != std::errc{})
    throw std::runtime_error("BenchmarkReport::save: could not format number");
  std::string number(buffer, result.ptr);
  const auto exponent = number.find_first_of("eE");
  if (exponent != std::string::npos && number.find('.') == std::string::npos)
    number.insert(exponent, ".0");
  return number;
}

template <typename T> void field(YAML::Emitter& yaml, const char* key, const T& value) {
  yaml << YAML::Key << key << YAML::Value << value;
}
void field(YAML::Emitter& yaml, const char* key, const std::string& value) {
  yaml << YAML::Key << key << YAML::Value << YAML::DoubleQuoted << value;
}
void field(YAML::Emitter& yaml, const char* key, double value) {
  yaml << YAML::Key << key << YAML::Value << yaml_number(value);
}

void write_metric(YAML::Emitter& yaml, const BenchmarkMetric& metric) {
  if (metric.samples < 0 || !std::isfinite(metric.mean) || !std::isfinite(metric.minimum) ||
      !std::isfinite(metric.maximum))
    throw std::invalid_argument("BenchmarkReport::save: invalid telemetry metric: " + metric.key);
  yaml << YAML::BeginMap;
  field(yaml, "key", metric.key);
  field(yaml, "label", metric.label);
  field(yaml, "unit", metric.unit);
  field(yaml, "samples", metric.samples);
  field(yaml, "mean", metric.mean);
  field(yaml, "minimum", metric.minimum);
  field(yaml, "maximum", metric.maximum);
  yaml << YAML::EndMap;
}

void write_telemetry(YAML::Emitter& yaml, const BenchmarkTelemetry& telemetry) {
  if (telemetry.samples < 0 || telemetry.sample_interval_ms < 0)
    throw std::invalid_argument("BenchmarkReport::save: invalid telemetry sample counts");
  yaml << YAML::BeginMap;
  field(yaml, "status", telemetry.status);
  field(yaml, "trace_id", telemetry.trace_id);
  field(yaml, "sentinel_version", telemetry.sentinel_version);
  field(yaml, "sample_interval_ms", telemetry.sample_interval_ms);
  field(yaml, "samples", telemetry.samples);
  const auto metric = [&](const char* key, const BenchmarkMetric& value) {
    yaml << YAML::Key << key << YAML::Value;
    write_metric(yaml, value);
  };
  const auto metrics = [&](const char* key, const std::vector<BenchmarkMetric>& values) {
    yaml << YAML::Key << key << YAML::Value << YAML::BeginSeq;
    for (const auto& value : values)
      write_metric(yaml, value);
    yaml << YAML::EndSeq;
  };
  metric("power_som_watts", telemetry.power_som_watts);
  metric("power_mla_watts", telemetry.power_mla_watts);
  metrics("power_rails_watts", telemetry.power_rails_watts);
  metrics("die_temperatures_c", telemetry.die_temperatures_c);
  metric("die_temperature_max_c", telemetry.die_temperature_max_c);
  metrics("board_temperatures_c", telemetry.board_temperatures_c);
  metric("cpu_usage_pct", telemetry.cpu_usage_pct);
  metrics("cpu_cores_usage_pct", telemetry.cpu_cores_usage_pct);
  metric("mla_memory_mib", telemetry.mla_memory_mib);
  metric("cma_memory_mib", telemetry.cma_memory_mib);
  metric("linux_memory_used_mib", telemetry.linux_memory_used_mib);
  yaml << YAML::EndMap;
}

} // namespace

std::filesystem::path BenchmarkReport::save(const std::filesystem::path& file_path) const {
  const auto path = file_path.string();
  if (path.empty() || file_path.filename().empty() || path.find('\0') != std::string::npos)
    throw std::invalid_argument("BenchmarkReport::save: provide a destination file path");
  YAML::Emitter yaml;
  yaml.SetOutputCharset(YAML::EscapeNonAscii);
  yaml << YAML::BeginMap;
  const auto headline = [&](const char* key, double value) {
    require_nonnegative(value, key);
    field(yaml, key, value);
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
    throw std::invalid_argument("BenchmarkReport::save: throughput_samples must be nonnegative");
  field(yaml, "throughput_samples", throughput_samples);
  yaml << YAML::Key << "telemetry" << YAML::Value;
  write_telemetry(yaml, telemetry);
  yaml << YAML::Key << "layer_stats" << YAML::Value << YAML::BeginSeq;
  for (const auto& layer : layer_stats) {
    require_nonnegative(layer.total_execution_time_us, "total_execution_time_us");
    yaml << YAML::BeginMap;
    field(yaml, "stage_index", layer.stage_index);
    field(yaml, "name", layer.name);
    field(yaml, "total_execution_time_us", layer.total_execution_time_us);
    yaml << YAML::EndMap;
  }
  yaml << YAML::EndSeq << YAML::EndMap;
  if (!yaml.good())
    throw std::runtime_error("BenchmarkReport::save: " + yaml.GetLastError());
  const std::string contents =
      "# NEAT benchmark report. Latency: ms; throughput: inferences/s; power: W; energy: J.\n"
      "# Layer times: hardware run_time in us from separate profiling passes, not headline "
      "latency.\n" +
      std::string(yaml.c_str(), yaml.size()) + '\n';
  if (!file_path.parent_path().empty())
    std::filesystem::create_directories(file_path.parent_path());
  GError* error = nullptr;
  if (!g_file_set_contents_full(path.c_str(), contents.data(), contents.size(),
                                static_cast<GFileSetContentsFlags>(G_FILE_SET_CONTENTS_CONSISTENT |
                                                                   G_FILE_SET_CONTENTS_DURABLE),
                                0666, &error)) {
    const std::string detail = error ? error->message : "unknown write error";
    g_clear_error(&error);
    throw std::runtime_error("BenchmarkReport::save: cannot write '" + path + "': " + detail);
  }
  return file_path;
}

} // namespace simaai::neat
