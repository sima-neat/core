#include "model/internal/BenchmarkTelemetry.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <ctime>
#include <numeric>
#include <optional>
#include <string>
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

BenchmarkTelemetry summarize_benchmark_telemetry(const json& run) {
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
  telemetry.mla_memory_mib = memory_mib(run, "mla_mem_allocated_mb");
  telemetry.cma_memory_mib = memory_mib(run, "ev74_cma_used_mb");
  telemetry.linux_memory_used_mib = memory_mib(run, "linux_mem_used_mb");
  return telemetry;
}

SentinelTrace::SentinelTrace(const std::string& note, pipeline_internal::SentinelClient client)
    : client_(std::move(client)) {
  try {
    trace_id_ = client_.start_trace(benchmark_trace_name(), note, {"neat-benchmark"});
    recording_ = true;
  } catch (const std::exception& e) {
    failure_ = e.what();
  }
}

SentinelTrace::~SentinelTrace() {
  stop();
}

void SentinelTrace::stop() noexcept {
  if (!recording_) {
    return;
  }
  recording_ = false;
  try {
    client_.stop_trace(trace_id_);
  } catch (const std::exception& e) {
    failure_ = e.what();
  }
}

BenchmarkTelemetry SentinelTrace::finish() {
  stop();
  if (failure_.empty()) {
    try {
      return summarize_benchmark_telemetry(client_.read_run(trace_id_));
    } catch (const std::exception& e) {
      failure_ = e.what();
    }
  }
  BenchmarkTelemetry telemetry;
  telemetry.status = failure_;
  telemetry.trace_id = trace_id_;
  return telemetry;
}

} // namespace simaai::neat::internal
