#pragma once

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <string>
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
  struct rusage usage{};
  if (::getrusage(RUSAGE_SELF, &usage) != 0) {
    return 0.0;
  }
  // Linux reports ru_maxrss in kilobytes.
  return static_cast<double>(usage.ru_maxrss);
}

inline void emit_metrics_json(const std::string& scenario_id, int iterations,
                              const PerfMetrics& metrics, const std::string& run_mode) {
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
            << "  \"output_drop_count\": " << metrics.output_drop_count << "\n"
            << "}\n";
}

inline double elapsed_seconds(Clock::time_point start, Clock::time_point end) {
  return std::chrono::duration_cast<std::chrono::duration<double>>(end - start).count();
}

inline double elapsed_ms(Clock::time_point start, Clock::time_point end) {
  return std::chrono::duration_cast<std::chrono::duration<double, std::milli>>(end - start).count();
}

} // namespace sima_perf
