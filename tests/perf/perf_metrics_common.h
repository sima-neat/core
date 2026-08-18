#ifndef SIMA_NEAT_INTERNAL
#define SIMA_NEAT_INTERNAL 1
#endif
#pragma once

#include "pipeline/PowerTelemetry.h"
#include "pipeline/Run.h"

#include <algorithm>
#include <chrono>
#include <cctype>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <memory>
#include <ostream>
#include <string>
#include <string_view>
#include <sys/resource.h>
#include <vector>

namespace sima_perf {

using Clock = std::chrono::steady_clock;

struct PerfMetrics {
  double throughput = 0.0;
  double p50 = 0.0;
  double p95 = 0.0;
  double startup = 0.0;
  double rss_peak_kb = 0.0;
  std::uint64_t input_drop_count = 0;
  std::uint64_t output_drop_count = 0;
};

struct ComponentLatencySelection {
  std::string component_id;
  std::vector<const simaai::neat::MeasurePluginLatencyWithPercentiles*> exact_rows;
};

inline void emit_json_string(std::ostream& out, std::string_view value) {
  out << '"';
  for (const unsigned char ch : value) {
    switch (ch) {
    case '"':
      out << "\\\"";
      break;
    case '\\':
      out << "\\\\";
      break;
    case '\b':
      out << "\\b";
      break;
    case '\f':
      out << "\\f";
      break;
    case '\n':
      out << "\\n";
      break;
    case '\r':
      out << "\\r";
      break;
    case '\t':
      out << "\\t";
      break;
    default:
      if (ch < 0x20U) {
        const char hex[] = "0123456789abcdef";
        out << "\\u00" << hex[(ch >> 4U) & 0x0fU] << hex[ch & 0x0fU];
      } else {
        out << static_cast<char>(ch);
      }
      break;
    }
  }
  out << '"';
}

inline void
emit_component_latency_row(const simaai::neat::MeasurePluginLatencyWithPercentiles& measured) {
  const auto& row = measured.metric;
  const auto& percentiles = measured.percentiles;
  std::cout << "{";
  std::cout << "\"backend\":";
  emit_json_string(std::cout, row.backend);
  std::cout << ",\"phase\":";
  emit_json_string(std::cout, row.phase);
  std::cout << ",\"kernel_name\":";
  emit_json_string(std::cout, row.kernel_name);
  std::cout << ",\"stage_name\":";
  emit_json_string(std::cout, row.stage_name);
  std::cout << ",\"plugin_instance_id\":";
  emit_json_string(std::cout, row.plugin_instance_id);
  std::cout << ",\"samples\":" << row.calls << ",\"reliable\":" << (row.reliable ? "true" : "false")
            << ",\"percentiles_available\":" << (percentiles.available ? "true" : "false")
            << ",\"avg_ms\":" << row.avg_ms << ",\"p50_ms\":" << percentiles.p50_ms
            << ",\"p95_ms\":" << percentiles.p95_ms << ",\"p99_ms\":" << percentiles.p99_ms
            << ",\"max_ms\":" << row.max_ms << "}";
}

inline void emit_component_latency_json(const std::vector<ComponentLatencySelection>& components) {
  std::cout << ",\n  \"component_latency\": {";
  for (std::size_t component_index = 0; component_index < components.size(); ++component_index) {
    const auto& component = components[component_index];
    std::cout << (component_index == 0 ? "\n    " : ",\n    ");
    emit_json_string(std::cout, component.component_id);
    std::cout << ": [";
    bool first_row = true;
    for (const auto* row : component.exact_rows) {
      if (!row) {
        continue;
      }
      std::cout << (first_row ? "" : ",");
      emit_component_latency_row(*row);
      first_row = false;
    }
    std::cout << "]";
  }
  std::cout << "\n  }";
}

inline int env_int(const char* key, int fallback, int minimum = 1) {
  const char* value = std::getenv(key);
  if (!value || !*value) {
    return fallback;
  }
  try {
    const int parsed = std::stoi(value);
    return std::max(minimum, parsed);
  } catch (...) {
    return fallback;
  }
}

inline bool env_bool(const char* key, bool fallback = false) {
  const char* value = std::getenv(key);
  if (!value || !*value) {
    return fallback;
  }
  std::string text(value);
  std::transform(text.begin(), text.end(), text.begin(),
                 [](unsigned char c) { return static_cast<char>(std::tolower(c)); });
  return text == "1" || text == "true" || text == "yes" || text == "on";
}

inline simaai::neat::PowerMonitorOptions power_options_from_env() {
  simaai::neat::PowerMonitorOptions options;
  options.enabled = env_bool("SIMA_PERF_POWER", false);
  options.sample_interval_ms = env_int("SIMA_PERF_POWER_INTERVAL_MS", 100);
  if (options.enabled) {
    options = simaai::neat::board_power_monitor_options(options.sample_interval_ms);
  }
  return options;
}

inline double percentile(std::vector<double> values, double p) {
  if (values.empty()) {
    return 0.0;
  }
  std::sort(values.begin(), values.end());
  const double scaled = (p / 100.0) * static_cast<double>(values.size() - 1);
  const std::size_t lo = static_cast<std::size_t>(scaled);
  const std::size_t hi = std::min(values.size() - 1, lo + 1);
  const double frac = scaled - static_cast<double>(lo);
  return values[lo] + (values[hi] - values[lo]) * frac;
}

inline double rss_peak_kb() {
  struct rusage usage {};
  if (::getrusage(RUSAGE_SELF, &usage) != 0) {
    return 0.0;
  }
  // Linux reports ru_maxrss in kilobytes.
  return static_cast<double>(usage.ru_maxrss);
}

inline void emit_metrics_json(const std::string& scenario_id, int iterations,
                              const PerfMetrics& metrics, const std::string& run_mode,
                              const simaai::neat::PowerSummary* power = nullptr,
                              const simaai::neat::MeasureReport* report = nullptr,
                              const std::vector<ComponentLatencySelection>* components = nullptr) {
  std::cout << std::fixed << std::setprecision(6) << "{\n"
            << "  \"scenario_id\": \"" << scenario_id << "\",\n"
            << "  \"iterations\": " << iterations << ",\n"
            << "  \"run_mode\": \"" << run_mode << "\",\n"
            << "  \"throughput\": " << metrics.throughput << ",\n"
            << "  \"p50\": " << metrics.p50 << ",\n"
            << "  \"p95\": " << metrics.p95 << ",\n"
            << "  \"startup\": " << metrics.startup << ",\n"
            << "  \"rss_peak_kb\": " << metrics.rss_peak_kb << ",\n"
            << "  \"input_drop_count\": " << metrics.input_drop_count << ",\n"
            << "  \"output_drop_count\": " << metrics.output_drop_count;
  if (report) {
    std::cout << ",\n  \"measure_report\": " << report->to_json(2);
  }
  if (components) {
    emit_component_latency_json(*components);
  }
  if (power && power->enabled) {
    std::cout << ",\n  \"power\": " << simaai::neat::power_summary_to_json(*power, 2) << "\n";
  } else {
    std::cout << "\n";
  }
  std::cout << "}\n";
  std::cout.flush();
}

inline double elapsed_seconds(Clock::time_point start, Clock::time_point end) {
  return std::chrono::duration_cast<std::chrono::duration<double>>(end - start).count();
}

inline double elapsed_ms(Clock::time_point start, Clock::time_point end) {
  return std::chrono::duration_cast<std::chrono::duration<double, std::milli>>(end - start).count();
}

} // namespace sima_perf
