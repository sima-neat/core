#include "model/internal/BenchmarkTelemetry.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <ctime>
#include <limits>
#include <numeric>
#include <optional>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

namespace simaai::neat::internal {
namespace {

using nlohmann::json;

constexpr int kDieSensorCount = 7; // rtsn_0..6; rtsn_7..13 are trip channels for the same sites.

/// `neat-benchmark-<UTC time>`, unique per benchmark and sortable by start time.
std::string benchmark_trace_name() {
  const auto now = std::chrono::system_clock::now();
  const std::time_t seconds = std::chrono::system_clock::to_time_t(now);
  const auto millis =
      std::chrono::duration_cast<std::chrono::milliseconds>(now.time_since_epoch()).count() % 1000;
  std::tm utc{};
  ::gmtime_r(&seconds, &utc);
  char stamp[32];
  std::strftime(stamp, sizeof(stamp), "%Y%m%dT%H%M%S", &utc);
  char name[64];
  std::snprintf(name, sizeof(name), "neat-benchmark-%s.%03dZ", stamp, static_cast<int>(millis));
  return name;
}

/// The member `key` of `object` if it has `type`; otherwise an empty value of that type.
const json& member(const json& object, const char* key, json::value_t type) {
  static const json kEmptyArray = json::array();
  static const json kEmptyObject = json::object();
  const auto it = object.is_object() ? object.find(key) : object.end();
  if (it != object.end() && it->type() == type) {
    return *it;
  }
  return type == json::value_t::array ? kEmptyArray : kEmptyObject;
}

const json& samples(const json& run) {
  return member(run, "samples", json::value_t::array);
}

/// Sentinel serializes chrono::DateTime<Utc> as RFC3339 UTC (Z or +00:00),
/// optionally with fractional seconds. Do not infer epochs from numeric values,
/// accept local-time strings, or map samples by their position in the trace.
std::optional<std::chrono::system_clock::time_point> sample_timestamp(const json& sample) {
  const auto it = sample.is_object() ? sample.find("timestamp") : sample.end();
  if (it == sample.end() || !it->is_string()) {
    return std::nullopt;
  }
  std::string_view value = it->get_ref<const std::string&>();
  if (value.ends_with("Z")) {
    value.remove_suffix(1);
  } else if (value.ends_with("+00:00")) {
    value.remove_suffix(6);
  } else {
    return std::nullopt;
  }
  if (value.size() < 19 || value[4] != '-' || value[7] != '-' || value[10] != 'T' ||
      value[13] != ':' || value[16] != ':') {
    return std::nullopt;
  }
  const auto number = [&](std::size_t offset, std::size_t length) -> int {
    int out = 0;
    for (const char digit : value.substr(offset, length)) {
      if (digit < '0' || digit > '9') {
        return -1;
      }
      out = out * 10 + digit - '0';
    }
    return out;
  };
  const int year = number(0, 4);
  const int month = number(5, 2);
  const int day = number(8, 2);
  const int hour = number(11, 2);
  const int minute = number(14, 2);
  const int second = number(17, 2);
  const std::chrono::year_month_day date{std::chrono::year{year},
                                         std::chrono::month{static_cast<unsigned>(month)},
                                         std::chrono::day{static_cast<unsigned>(day)}};
  if (year < 0 || !date.ok() || hour < 0 || hour > 23 || minute < 0 || minute > 59 || second < 0 ||
      second > 59) {
    return std::nullopt;
  }
  std::int64_t fraction = 0;
  if (value.size() != 19) {
    if (value[19] != '.' || value.size() < 21 || value.size() > 29) {
      return std::nullopt;
    }
    fraction = number(20, value.size() - 20);
    if (fraction < 0) {
      return std::nullopt;
    }
    for (std::size_t i = value.size() - 20; i < 9; ++i) {
      fraction *= 10;
    }
  }
  const auto seconds = std::chrono::duration_cast<std::chrono::seconds>(
                           std::chrono::sys_days{date}.time_since_epoch())
                           .count() +
                       hour * 3600 + minute * 60 + second;
  constexpr std::int64_t kNanosecondsPerSecond = 1000000000;
  if (seconds < std::numeric_limits<std::int64_t>::min() / kNanosecondsPerSecond ||
      seconds > (std::numeric_limits<std::int64_t>::max() - fraction) / kNanosecondsPerSecond) {
    return std::nullopt;
  }
  return std::chrono::system_clock::time_point{
      std::chrono::duration_cast<std::chrono::system_clock::duration>(
          std::chrono::nanoseconds{seconds * kNanosecondsPerSecond + fraction})};
}

bool valid_window(const BenchmarkTelemetryWindow& window) {
  if (window.begin.before > window.begin.after || window.end.before > window.end.after ||
      window.end.before < window.begin.after || window.end.utc < window.begin.utc) {
    return false;
  }
  const auto elapsed = window.end.utc - window.begin.utc;
  // The UTC reads occurred somewhere within the steady brackets. No guessed
  // offset or arbitrary clock-step tolerance is needed for this local provider.
  return elapsed >= window.end.before - window.begin.after &&
         elapsed <= window.end.after - window.begin.before;
}

std::optional<double> sample_value(const json& sample, const std::string& key) {
  const json& values = member(sample, "values", json::value_t::object);
  const auto it = values.find(key);
  if (it == values.end() || !it->is_number() || !std::isfinite(it->get<double>())) {
    return std::nullopt;
  }
  return it->get<double>();
}

/// The `label` or `unit` of a metric from the run's metric definitions.
std::string metric_attribute(const json& run, const std::string& key, const char* attribute) {
  for (const json& definition : member(run, "metrics", json::value_t::array)) {
    if (definition.value("key", "") == key) {
      return definition.value(attribute, "");
    }
  }
  return {};
}

BenchmarkMetric summarize(std::string key, std::string label, std::string unit,
                          const std::vector<double>& values) {
  BenchmarkMetric metric;
  metric.key = std::move(key);
  metric.label = std::move(label);
  metric.unit = std::move(unit);
  if (!values.empty()) {
    metric.samples = static_cast<int>(values.size());
    metric.mean = std::accumulate(values.begin(), values.end(), 0.0) / values.size();
    metric.minimum = *std::min_element(values.begin(), values.end());
    metric.maximum = *std::max_element(values.begin(), values.end());
  }
  return metric;
}

BenchmarkMetric metric(const json& run, const std::string& key) {
  std::vector<double> values;
  for (const json& sample : samples(run)) {
    if (const auto value = sample_value(sample, key)) {
      values.push_back(*value);
    }
  }
  const std::string label = metric_attribute(run, key, "label");
  return summarize(key, label.empty() ? key : label, metric_attribute(run, key, "unit"), values);
}

std::vector<BenchmarkMetric> metrics(const json& run, const std::vector<std::string>& keys) {
  std::vector<BenchmarkMetric> out;
  for (const std::string& key : keys) {
    out.push_back(metric(run, key));
  }
  return out;
}

/// Every metric key that starts with `prefix` and ends with `suffix`.
std::vector<std::string> metric_keys(const json& run, const std::string& prefix,
                                     const std::string& suffix) {
  std::vector<std::string> keys;
  for (const json& definition : member(run, "metrics", json::value_t::array)) {
    const std::string key = definition.value("key", "");
    if (key.size() > prefix.size() + suffix.size() && key.starts_with(prefix) &&
        key.ends_with(suffix)) {
      keys.push_back(key);
    }
  }
  return keys;
}

/// The hottest die sensor in each sample.
BenchmarkMetric die_max(const json& run, const std::vector<std::string>& sensors) {
  std::vector<double> values;
  for (const json& sample : samples(run)) {
    std::optional<double> hottest;
    for (const std::string& sensor : sensors) {
      if (const auto value = sample_value(sample, sensor)) {
        hottest = std::max(hottest.value_or(*value), *value);
      }
    }
    if (hottest) {
      values.push_back(*hottest);
    }
  }
  return summarize("die_max", "Die max", metric_attribute(run, sensors.front(), "unit"), values);
}

/// Sentinel converts memory to MiB but labels it "MB".
BenchmarkMetric memory_mib(const json& run, const std::string& key) {
  BenchmarkMetric memory = metric(run, key);
  memory.unit = "MiB";
  return memory;
}

} // namespace

BenchmarkTelemetry summarize_benchmark_telemetry(const json& saved_run,
                                                 const BenchmarkTelemetryWindow& window) {
  const json& saved_metadata = member(saved_run, "metadata", json::value_t::object);
  const auto unavailable = [&](const std::string& reason) {
    BenchmarkTelemetry telemetry;
    telemetry.status = reason;
    telemetry.trace_id = saved_metadata.value("id", "");
    telemetry.sentinel_version = saved_metadata.value("sentinel_version", "");
    telemetry.sample_interval_ms = saved_metadata.value("sample_interval_ms", 0);
    return telemetry;
  };
  if (!valid_window(window)) {
    return unavailable("Sentinel throughput window has invalid or discontinuous clocks");
  }
  json run = saved_run;
  run["samples"] = json::array();
  std::optional<std::chrono::system_clock::time_point> previous;
  for (const json& sample : samples(saved_run)) {
    const auto timestamp = sample_timestamp(sample);
    if (!timestamp || (previous && *timestamp <= *previous)) {
      return unavailable("Sentinel sample timestamps are invalid or not strictly increasing");
    }
    previous = timestamp;
    if (*timestamp >= window.begin.utc && *timestamp <= window.end.utc) {
      run["samples"].push_back(sample);
    }
  }
  const json& metadata = member(run, "metadata", json::value_t::object);
  std::vector<std::string> die_sensors;
  for (int i = 0; i < kDieSensorCount; ++i) {
    die_sensors.push_back("rtsn_" + std::to_string(i));
  }

  BenchmarkTelemetry telemetry;
  telemetry.samples = static_cast<int>(samples(run).size());
  telemetry.status = telemetry.samples > 0
                         ? "ok"
                         : "Sentinel recorded no samples during the throughput window; "
                           "increase min_duration_ms";
  telemetry.trace_id = metadata.value("id", "");
  telemetry.sentinel_version = metadata.value("sentinel_version", "");
  telemetry.sample_interval_ms = metadata.value("sample_interval_ms", 0);
  telemetry.power_som_watts = metric(run, "power_current_watts");
  telemetry.power_mla_watts = metric(run, "power_rail_mla_watts");
  telemetry.power_rails_watts = metrics(run, metric_keys(run, "power_rail_", "_watts"));
  telemetry.die_temperatures_c = metrics(run, die_sensors);
  telemetry.die_temperature_max_c = die_max(run, die_sensors);
  telemetry.board_temperatures_c = metrics(run, {"lm96163_temp1", "lm96163_temp2"});
  telemetry.cpu_usage_pct = metric(run, "cpu_usage_pct");
  telemetry.cpu_cores_usage_pct = metrics(run, metric_keys(run, "cpu_core_", "_usage_pct"));
  telemetry.mla_memory_mib = memory_mib(run, "mla_mem_allocated_mb");
  telemetry.cma_memory_mib = memory_mib(run, "ev74_cma_used_mb");
  telemetry.linux_memory_used_mib = memory_mib(run, "linux_mem_used_mb");
  return telemetry;
}

BenchmarkTelemetryBoundary BenchmarkTelemetryBoundary::now() noexcept {
  BenchmarkTelemetryBoundary boundary;
  boundary.before = std::chrono::steady_clock::now();
  boundary.utc = std::chrono::system_clock::now();
  boundary.after = std::chrono::steady_clock::now();
  return boundary;
}

SentinelTrace::SentinelTrace(const std::string& note, pipeline_internal::SentinelClient client)
    : client_(std::move(client)) {
  try {
    trace_id_ = client_.start_trace(benchmark_trace_name(), note, {"neat-benchmark"});
    recording_ = true;
  } catch (const std::exception& e) {
    remember_failure(e.what());
  } catch (...) {
    remember_failure("Sentinel could not start a trace (unknown exception)");
  }
}

SentinelTrace::~SentinelTrace() {
  stop();
}

void SentinelTrace::remember_failure(const char* message) noexcept {
  if (failed_) {
    return;
  }
  failed_ = true;
  try {
    failure_ = message;
  } catch (...) {
    // Preserve the failure even when recording its diagnostic cannot allocate.
  }
}

void SentinelTrace::stop() noexcept {
  if (!recording_) {
    return;
  }
  recording_ = false;
  try {
    client_.stop_trace(trace_id_);
  } catch (const std::exception& e) {
    remember_failure(e.what());
  } catch (...) {
    remember_failure("Sentinel could not stop a trace (unknown exception)");
  }
}

BenchmarkTelemetry SentinelTrace::finish(const BenchmarkTelemetryWindow& window) {
  stop();
  if (!failed_) {
    try {
      return summarize_benchmark_telemetry(client_.read_run(trace_id_), window);
    } catch (const std::exception& e) {
      remember_failure(e.what());
    } catch (...) {
      remember_failure("Sentinel could not read a trace (unknown exception)");
    }
  }
  BenchmarkTelemetry telemetry;
  telemetry.status = failure_.empty() ? "Sentinel trace failed (diagnostic unavailable)" : failure_;
  telemetry.trace_id = trace_id_;
  return telemetry;
}

} // namespace simaai::neat::internal
