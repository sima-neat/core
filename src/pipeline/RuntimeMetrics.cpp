#include "pipeline/RuntimeMetrics.h"

#include <algorithm>
#include <iomanip>
#include <sstream>

namespace simaai::neat {
namespace {

std::string json_escape(const std::string& input) {
  std::ostringstream oss;
  for (const char ch : input) {
    switch (ch) {
    case '\\':
      oss << "\\\\";
      break;
    case '"':
      oss << "\\\"";
      break;
    case '\n':
      oss << "\\n";
      break;
    case '\r':
      oss << "\\r";
      break;
    case '\t':
      oss << "\\t";
      break;
    default:
      if (static_cast<unsigned char>(ch) < 0x20) {
        oss << "\\u" << std::hex << std::setw(4) << std::setfill('0')
            << static_cast<int>(static_cast<unsigned char>(ch));
      } else {
        oss << ch;
      }
    }
  }
  return oss.str();
}

std::string spaces(int count) {
  return std::string(static_cast<std::size_t>(std::max(0, count)), ' ');
}

void append_metric_value_json(std::ostringstream& oss, const RuntimeMetricValue& value,
                              int indent) {
  const std::string pad = spaces(indent);
  oss << pad << "{"
      << "\"name\": \"" << json_escape(value.name) << "\", "
      << "\"value\": " << value.value << ", "
      << "\"unit\": \"" << json_escape(value.unit) << "\""
      << "}";
}

} // namespace

std::string runtime_metrics_to_json(const RuntimeMetrics& metrics, int indent) {
  const std::string pad = spaces(indent);
  const std::string p2 = spaces(indent + 2);
  const std::string p4 = spaces(indent + 4);
  const std::string p6 = spaces(indent + 6);
  std::ostringstream oss;
  oss << std::fixed << std::setprecision(6);
  oss << pad << "{\n";
  oss << p2 << "\"source_kind\": \"" << json_escape(metrics.source_kind) << "\",\n";
  oss << p2 << "\"source_name\": \"" << json_escape(metrics.source_name) << "\",\n";
  oss << p2 << "\"elapsed_seconds\": " << metrics.elapsed_seconds << ",\n";
  oss << p2 << "\"throughput_fps\": " << metrics.throughput_fps << ",\n";
  oss << p2 << "\"latency\": {\n";
  oss << p4 << "\"avg_ms\": " << metrics.latency.avg_ms << ",\n";
  oss << p4 << "\"min_ms\": " << metrics.latency.min_ms << ",\n";
  oss << p4 << "\"max_ms\": " << metrics.latency.max_ms << ",\n";
  oss << p4 << "\"p50_ms\": " << metrics.latency.p50_ms << ",\n";
  oss << p4 << "\"p95_ms\": " << metrics.latency.p95_ms << ",\n";
  oss << p4 << "\"has_percentiles\": "
      << (metrics.latency.has_percentiles ? "true" : "false") << "\n";
  oss << p2 << "},\n";
  oss << p2 << "\"counters\": {\n";
  oss << p4 << "\"inputs_enqueued\": " << metrics.counters.inputs_enqueued << ",\n";
  oss << p4 << "\"inputs_dropped\": " << metrics.counters.inputs_dropped << ",\n";
  oss << p4 << "\"inputs_pushed\": " << metrics.counters.inputs_pushed << ",\n";
  oss << p4 << "\"outputs_ready\": " << metrics.counters.outputs_ready << ",\n";
  oss << p4 << "\"outputs_pulled\": " << metrics.counters.outputs_pulled << ",\n";
  oss << p4 << "\"outputs_dropped\": " << metrics.counters.outputs_dropped << "\n";
  oss << p2 << "},\n";
  oss << p2 << "\"power\": " << power_summary_to_json(metrics.power, indent + 2) << ",\n";
  oss << p2 << "\"metadata\": {";
  if (!metrics.metadata.empty())
    oss << "\n";
  for (std::size_t i = 0; i < metrics.metadata.size(); ++i) {
    const auto& item = metrics.metadata[i];
    oss << p4 << "\"" << json_escape(item.first) << "\": \"" << json_escape(item.second) << "\"";
    if (i + 1 != metrics.metadata.size())
      oss << ",";
    oss << "\n";
  }
  if (!metrics.metadata.empty())
    oss << p2;
  oss << "},\n";
  oss << p2 << "\"groups\": [";
  if (!metrics.groups.empty())
    oss << "\n";
  for (std::size_t i = 0; i < metrics.groups.size(); ++i) {
    const auto& group = metrics.groups[i];
    oss << p4 << "{\n";
    oss << p6 << "\"name\": \"" << json_escape(group.name) << "\",\n";
    oss << p6 << "\"values\": [";
    if (!group.values.empty())
      oss << "\n";
    for (std::size_t j = 0; j < group.values.size(); ++j) {
      append_metric_value_json(oss, group.values[j], indent + 8);
      if (j + 1 != group.values.size())
        oss << ",";
      oss << "\n";
    }
    if (!group.values.empty())
      oss << p6;
    oss << "]\n" << p4 << "}";
    if (i + 1 != metrics.groups.size())
      oss << ",";
    oss << "\n";
  }
  if (!metrics.groups.empty())
    oss << p2;
  oss << "]\n";
  oss << pad << "}";
  return oss.str();
}

std::string runtime_metrics_to_compact_text(const RuntimeMetrics& metrics) {
  std::ostringstream oss;
  oss << std::fixed << std::setprecision(3);
  oss << "RuntimeMetrics";
  if (!metrics.source_kind.empty())
    oss << " source=" << metrics.source_kind;
  if (!metrics.source_name.empty())
    oss << "/" << metrics.source_name;
  oss << " elapsed_s=" << metrics.elapsed_seconds
      << " throughput_fps=" << metrics.throughput_fps
      << " latency_avg_ms=" << metrics.latency.avg_ms
      << " latency_min_ms=" << metrics.latency.min_ms
      << " latency_max_ms=" << metrics.latency.max_ms
      << " outputs_pulled=" << metrics.counters.outputs_pulled;
  if (metrics.power.enabled) {
    oss << " power_avg_w=" << metrics.power.total_avg_watts
        << " energy_j=" << metrics.power.energy_joules;
  }
  return oss.str();
}

std::string runtime_metrics_to_text(const RuntimeMetrics& metrics) {
  std::ostringstream oss;
  oss << std::fixed << std::setprecision(3);
  oss << "[METRICS]";
  if (!metrics.source_kind.empty())
    oss << " source=" << metrics.source_kind;
  if (!metrics.source_name.empty())
    oss << " name=" << metrics.source_name;
  oss << "\n";
  oss << "Summary: elapsed_s=" << metrics.elapsed_seconds
      << " throughput_fps=" << metrics.throughput_fps
      << " outputs_pulled=" << metrics.counters.outputs_pulled << "\n";
  oss << "Latency: avg_ms=" << metrics.latency.avg_ms
      << " min_ms=" << metrics.latency.min_ms
      << " max_ms=" << metrics.latency.max_ms;
  if (metrics.latency.has_percentiles) {
    oss << " p50_ms=" << metrics.latency.p50_ms
        << " p95_ms=" << metrics.latency.p95_ms;
  }
  oss << "\n";
  oss << "Counters: inputs_enqueued=" << metrics.counters.inputs_enqueued
      << " inputs_dropped=" << metrics.counters.inputs_dropped
      << " inputs_pushed=" << metrics.counters.inputs_pushed
      << " outputs_ready=" << metrics.counters.outputs_ready
      << " outputs_dropped=" << metrics.counters.outputs_dropped << "\n";
  if (metrics.power.enabled) {
    oss << format_power_summary(metrics.power);
  }
  for (const auto& group : metrics.groups) {
    oss << "Group: " << group.name;
    for (const auto& value : group.values) {
      oss << " " << value.name << "=" << value.value;
      if (!value.unit.empty())
        oss << value.unit;
    }
    oss << "\n";
  }
  return oss.str();
}

std::string format_runtime_metrics(const RuntimeMetrics& metrics, RuntimeMetricsFormat format) {
  switch (format) {
  case RuntimeMetricsFormat::Json:
    return runtime_metrics_to_json(metrics);
  case RuntimeMetricsFormat::CompactText:
    return runtime_metrics_to_compact_text(metrics);
  case RuntimeMetricsFormat::Text:
  default:
    return runtime_metrics_to_text(metrics);
  }
}

} // namespace simaai::neat
