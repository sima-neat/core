#include "model/Model.h"

#include <glib.h>
#include <nlohmann/json.hpp>
#include <yaml-cpp/yaml.h>

#include <charconv>
#include <cmath>
#include <cstdint>
#include <stdexcept>

namespace simaai::neat {
namespace {

using ReportDocument = nlohmann::ordered_json;

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

void write_yaml(YAML::Emitter& yaml, const ReportDocument& value) {
  if (value.is_object()) {
    yaml << YAML::BeginMap;
    for (const auto& [key, child] : value.items()) {
      yaml << YAML::Key << key << YAML::Value;
      write_yaml(yaml, child);
    }
    yaml << YAML::EndMap;
  } else if (value.is_array()) {
    yaml << YAML::BeginSeq;
    for (const auto& child : value)
      write_yaml(yaml, child);
    yaml << YAML::EndSeq;
  } else if (value.is_string()) {
    yaml << YAML::DoubleQuoted << value.get<std::string>();
  } else if (value.is_number_float()) {
    yaml << yaml_number(value.get<double>());
  } else if (value.is_number_unsigned()) {
    yaml << value.get<std::uint64_t>();
  } else if (value.is_number_integer()) {
    yaml << value.get<std::int64_t>();
  } else {
    throw std::runtime_error("BenchmarkReport::save: unsupported report value");
  }
}

ReportDocument metric_document(const BenchmarkMetric& metric) {
  if (metric.samples < 0 || !std::isfinite(metric.mean) || !std::isfinite(metric.minimum) ||
      !std::isfinite(metric.maximum))
    throw std::invalid_argument("BenchmarkReport::save: invalid telemetry metric: " + metric.key);
  return {{"key", metric.key},         {"label", metric.label}, {"unit", metric.unit},
          {"samples", metric.samples}, {"mean", metric.mean},   {"minimum", metric.minimum},
          {"maximum", metric.maximum}};
}

ReportDocument telemetry_document(const BenchmarkTelemetry& telemetry) {
  if (telemetry.samples < 0 || telemetry.sample_interval_ms < 0)
    throw std::invalid_argument("BenchmarkReport::save: invalid telemetry sample counts");
  const auto metrics = [](const std::vector<BenchmarkMetric>& values) {
    auto result = ReportDocument::array();
    for (const auto& value : values)
      result.push_back(metric_document(value));
    return result;
  };
  return {{"status", telemetry.status},
          {"trace_id", telemetry.trace_id},
          {"sentinel_version", telemetry.sentinel_version},
          {"sample_interval_ms", telemetry.sample_interval_ms},
          {"samples", telemetry.samples},
          {"power_som_watts", metric_document(telemetry.power_som_watts)},
          {"power_mla_watts", metric_document(telemetry.power_mla_watts)},
          {"power_rails_watts", metrics(telemetry.power_rails_watts)},
          {"die_temperatures_c", metrics(telemetry.die_temperatures_c)},
          {"die_temperature_max_c", metric_document(telemetry.die_temperature_max_c)},
          {"board_temperatures_c", metrics(telemetry.board_temperatures_c)},
          {"cpu_usage_pct", metric_document(telemetry.cpu_usage_pct)},
          {"cpu_cores_usage_pct", metrics(telemetry.cpu_cores_usage_pct)},
          {"mla_memory_mib", metric_document(telemetry.mla_memory_mib)},
          {"cma_memory_mib", metric_document(telemetry.cma_memory_mib)},
          {"linux_memory_used_mib", metric_document(telemetry.linux_memory_used_mib)}};
}

} // namespace

std::filesystem::path BenchmarkReport::save(const std::filesystem::path& file_path) const {
  const auto path = file_path.string();
  if (path.empty() || file_path.filename().empty() || path.find('\0') != std::string::npos)
    throw std::invalid_argument("BenchmarkReport::save: provide a destination file path");
  auto document = ReportDocument::object();
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
    throw std::invalid_argument("BenchmarkReport::save: throughput_samples must be nonnegative");
  document["throughput_samples"] = throughput_samples;
  document["telemetry"] = telemetry_document(telemetry);
  auto& layers = document["layer_stats"] = ReportDocument::array();
  for (const auto& layer : layer_stats) {
    require_nonnegative(layer.total_execution_time_us, "total_execution_time_us");
    layers.push_back({{"stage_index", layer.stage_index},
                      {"name", layer.name},
                      {"total_execution_time_us", layer.total_execution_time_us}});
  }
  std::string contents;
  if (file_path.filename().string().ends_with(".json")) {
    contents = document.dump(2, ' ', true) + '\n';
  } else {
    YAML::Emitter yaml;
    yaml.SetOutputCharset(YAML::EscapeNonAscii);
    write_yaml(yaml, document);
    if (!yaml.good())
      throw std::runtime_error("BenchmarkReport::save: " + yaml.GetLastError());
    contents =
        "# NEAT benchmark report. Latency: ms; throughput: inferences/s; power: W; energy: J.\n"
        "# Layer times: hardware run_time in us from separate profiling passes, not headline "
        "latency.\n" +
        std::string(yaml.c_str(), yaml.size()) + '\n';
  }
  if (!file_path.parent_path().empty())
    std::filesystem::create_directories(file_path.parent_path());
  GError* error = nullptr;
  if (!g_file_set_contents_full(path.c_str(), contents.data(), contents.size(),
                                static_cast<GFileSetContentsFlags>(G_FILE_SET_CONTENTS_CONSISTENT |
                                                                   G_FILE_SET_CONTENTS_DURABLE),
                                0666, &error)) {
    const std::string detail = error ? error->message : "unknown error";
    g_clear_error(&error);
    throw std::runtime_error("BenchmarkReport::save: cannot write '" + path + "': " + detail);
  }
  return file_path;
}

} // namespace simaai::neat
