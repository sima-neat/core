#include "pipeline/internal/ReportFile.h"

#include <glib.h>
#include <yaml-cpp/yaml.h>

#include <charconv>
#include <cmath>
#include <cstdint>
#include <stdexcept>
#include <vector>

namespace simaai::neat::pipeline_internal {
namespace {

// PyYAML requires a decimal mantissa in scientific notation, e.g. 1.0e-09.
std::string yaml_number(double value, const std::string& context) {
  char buffer[64];
  const auto result = std::to_chars(buffer, buffer + sizeof(buffer), value);
  if (result.ec != std::errc{})
    throw std::runtime_error(context + ": could not format number");
  std::string number(buffer, result.ptr);
  const auto exponent = number.find_first_of("eE");
  if (exponent != std::string::npos && number.find('.') == std::string::npos)
    number.insert(exponent, ".0");
  return number;
}

void write_yaml(YAML::Emitter& yaml, const ReportDocument& value, const std::string& context) {
  if (value.is_object()) {
    yaml << YAML::BeginMap;
    for (const auto& [key, child] : value.items()) {
      yaml << YAML::Key << key << YAML::Value;
      write_yaml(yaml, child, context);
    }
    yaml << YAML::EndMap;
  } else if (value.is_array()) {
    yaml << YAML::BeginSeq;
    for (const auto& child : value)
      write_yaml(yaml, child, context);
    yaml << YAML::EndSeq;
  } else if (value.is_string()) {
    yaml << YAML::DoubleQuoted << value.get<std::string>();
  } else if (value.is_number_float()) {
    yaml << yaml_number(value.get<double>(), context);
  } else if (value.is_number_unsigned()) {
    yaml << value.get<std::uint64_t>();
  } else if (value.is_number_integer()) {
    yaml << value.get<std::int64_t>();
  } else {
    throw std::runtime_error(context + ": unsupported report value");
  }
}

ReportDocument stats_document(const TelemetryStats& stats, const std::string& context) {
  if (stats.samples < 0 || !std::isfinite(stats.mean) || !std::isfinite(stats.minimum) ||
      !std::isfinite(stats.maximum) || !std::isfinite(stats.last))
    throw std::invalid_argument(context + ": invalid telemetry metric: " + stats.key);
  return {{"key", stats.key},         {"label", stats.label}, {"unit", stats.unit},
          {"samples", stats.samples}, {"mean", stats.mean},   {"minimum", stats.minimum},
          {"maximum", stats.maximum}, {"last", stats.last}};
}

} // namespace

ReportDocument telemetry_document(const Telemetry& telemetry, const std::string& context) {
  if (telemetry.samples < 0 || telemetry.sample_interval_ms < 0 ||
      !std::isfinite(telemetry.duration_ms) || telemetry.duration_ms < 0)
    throw std::invalid_argument(context + ": invalid telemetry sample counts or duration");
  const auto stats = [&](const TelemetryStats& value) { return stats_document(value, context); };
  const auto stats_list = [&](const std::vector<TelemetryStats>& values) {
    auto result = ReportDocument::array();
    for (const auto& value : values)
      result.push_back(stats(value));
    return result;
  };
  return {{"status", telemetry.status},
          {"sentinel_version", telemetry.sentinel_version},
          {"sample_interval_ms", telemetry.sample_interval_ms},
          {"samples", telemetry.samples},
          {"duration_ms", telemetry.duration_ms},
          {"power_som_watts", stats(telemetry.power_som_watts)},
          {"power_mla_watts", stats(telemetry.power_mla_watts)},
          {"power_rails_watts", stats_list(telemetry.power_rails_watts)},
          {"die_temperatures_c", stats_list(telemetry.die_temperatures_c)},
          {"die_temperature_max_c", stats(telemetry.die_temperature_max_c)},
          {"board_temperatures_c", stats_list(telemetry.board_temperatures_c)},
          {"cpu_usage_pct", stats(telemetry.cpu_usage_pct)},
          {"cpu_cores_usage_pct", stats_list(telemetry.cpu_cores_usage_pct)},
          {"mla_memory_mib", stats(telemetry.mla_memory_mib)},
          {"ev74_memory_mib", stats(telemetry.ev74_memory_mib)},
          {"linux_memory_used_mib", stats(telemetry.linux_memory_used_mib)}};
}

std::string report_yaml(const ReportDocument& document, const std::string& header,
                        const std::string& context) {
  YAML::Emitter yaml;
  yaml.SetOutputCharset(YAML::EscapeNonAscii);
  write_yaml(yaml, document, context);
  if (!yaml.good())
    throw std::runtime_error(context + ": " + yaml.GetLastError());
  return header + std::string(yaml.c_str(), yaml.size()) + '\n';
}

void save_report(const std::filesystem::path& file_path, const ReportDocument& document,
                 const std::string& yaml_header, const std::string& context) {
  const auto path = file_path.string();
  if (path.empty() || file_path.filename().empty() || path.find('\0') != std::string::npos)
    throw std::invalid_argument(context + ": provide a destination file path");
  const std::string contents = file_path.filename().string().ends_with(".json")
                                   ? document.dump(2, ' ', true) + '\n'
                                   : report_yaml(document, yaml_header, context);
  if (!file_path.parent_path().empty())
    std::filesystem::create_directories(file_path.parent_path());
  GError* error = nullptr;
  if (!g_file_set_contents_full(path.c_str(), contents.data(), contents.size(),
                                static_cast<GFileSetContentsFlags>(G_FILE_SET_CONTENTS_CONSISTENT |
                                                                   G_FILE_SET_CONTENTS_DURABLE),
                                0666, &error)) {
    const std::string detail = error ? error->message : "unknown error";
    g_clear_error(&error);
    throw std::runtime_error(context + ": cannot write '" + path + "': " + detail);
  }
}

} // namespace simaai::neat::pipeline_internal
