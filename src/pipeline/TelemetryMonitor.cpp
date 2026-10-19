#include "pipeline/TelemetryMonitor.h"

#include "pipeline/internal/ReportFile.h"
#include "pipeline/internal/SentinelClient.h"
#include "pipeline/internal/TelemetryMonitorInternal.h"

#include <algorithm>
#include <atomic>
#include <cctype>
#include <chrono>
#include <cmath>
#include <condition_variable>
#include <cstdint>
#include <cstdio>
#include <ctime>
#include <iostream>
#include <mutex>
#include <optional>
#include <ostream>
#include <set>
#include <string>
#include <thread>
#include <utility>
#include <vector>

namespace simaai::neat {
namespace {

using nlohmann::json;
using Clock = std::chrono::steady_clock;
using std::chrono::milliseconds;

constexpr int kDieSensorCount = 7; // rtsn_0..6; rtsn_7..13 are trip channels for the same sites.
constexpr milliseconds kDefaultInterval{2000}; // Sentinel's interval in its systemd unit.
constexpr milliseconds kLongestInterval{60000};
constexpr std::chrono::seconds kRequestTimeout{1};

constexpr const char* kNotStarted = "not started";
constexpr const char* kWaiting = "waiting for the first Sentinel sample";
constexpr const char* kNoSample =
    "Sentinel took no new sample during the window; make the window longer than the sample "
    "interval";
constexpr const char* kYamlHeader =
    "# NEAT board telemetry from Sentinel. Each measurement names its unit; duration: ms.\n";

/// The string member `key` of `object`, or an empty string.
std::string text(const json& object, const char* key) {
  const auto it = object.find(key);
  return it != object.end() && it->is_string() ? it->get<std::string>() : std::string();
}

/// Stats for `key`, labelled from its Sentinel metric definition, with no samples.
TelemetryStats unmeasured(const json& metrics, const std::string& key) {
  TelemetryStats stats;
  stats.key = key;
  for (const json& definition : metrics) {
    if (text(definition, "key") == key) {
      stats.label = text(definition, "label");
      stats.unit = text(definition, "unit");
      break;
    }
  }
  if (stats.label.empty()) {
    stats.label = key;
  }
  return stats;
}

std::vector<TelemetryStats> unmeasured_each(const json& metrics,
                                            const std::vector<std::string>& keys) {
  std::vector<TelemetryStats> out;
  for (const std::string& key : keys) {
    out.push_back(unmeasured(metrics, key));
  }
  return out;
}

/// Every metric key that starts with `prefix` and ends with `suffix`, in definition order.
std::vector<std::string> keys_like(const json& metrics, const std::string& prefix,
                                   const std::string& suffix) {
  std::vector<std::string> keys;
  for (const json& definition : metrics) {
    const std::string key = text(definition, "key");
    if (key.size() > prefix.size() + suffix.size() && key.starts_with(prefix) &&
        key.ends_with(suffix)) {
      keys.push_back(key);
    }
  }
  return keys;
}

/// Sentinel converts memory to MiB but labels it "MB".
TelemetryStats unmeasured_memory(const json& metrics, const std::string& key) {
  TelemetryStats memory = unmeasured(metrics, key);
  memory.unit = "MiB";
  return memory;
}

/// Calls `fn` on every summary that holds one Sentinel metric, so all but the die maximum.
template <typename Fn> void for_each_metric(Telemetry& telemetry, Fn&& fn) {
  fn(telemetry.power_som_watts);
  fn(telemetry.power_mla_watts);
  std::for_each(telemetry.power_rails_watts.begin(), telemetry.power_rails_watts.end(), fn);
  std::for_each(telemetry.die_temperatures_c.begin(), telemetry.die_temperatures_c.end(), fn);
  std::for_each(telemetry.board_temperatures_c.begin(), telemetry.board_temperatures_c.end(), fn);
  fn(telemetry.cpu_usage_pct);
  std::for_each(telemetry.cpu_cores_usage_pct.begin(), telemetry.cpu_cores_usage_pct.end(), fn);
  fn(telemetry.mla_memory_mib);
  fn(telemetry.cma_memory_mib);
  fn(telemetry.linux_memory_used_mib);
}

void add(TelemetryStats& stats, double value) {
  ++stats.samples;
  stats.minimum = stats.samples == 1 ? value : std::min(stats.minimum, value);
  stats.maximum = stats.samples == 1 ? value : std::max(stats.maximum, value);
  stats.mean += (value - stats.mean) / stats.samples;
  stats.last = value;
}

/// The value of `key` in a sample, unless Sentinel could not read it.
std::optional<double> reading(const json& values, const std::string& key) {
  const auto it = values.find(key);
  if (it == values.end() || !it->is_number() || !std::isfinite(it->get<double>())) {
    return std::nullopt;
  }
  return it->get<double>();
}

/// Milliseconds since the epoch of a Sentinel timestamp such as `2026-10-19T08:00:02.000123Z`.
std::optional<std::int64_t> timestamp_ms(const std::string& timestamp) {
  int year = 0, month = 0, day = 0, hour = 0, minute = 0, second = 0, end = 0;
  if (std::sscanf(timestamp.c_str(), "%4d-%2d-%2dT%2d:%2d:%2d%n", &year, &month, &day, &hour,
                  &minute, &second, &end) != 6) {
    return std::nullopt;
  }
  std::tm utc{};
  utc.tm_year = year - 1900;
  utc.tm_mon = month - 1;
  utc.tm_mday = day;
  utc.tm_hour = hour;
  utc.tm_min = minute;
  utc.tm_sec = second;
  std::int64_t ms = static_cast<std::int64_t>(::timegm(&utc)) * 1000;
  if (timestamp[end] == '.') {
    int scale = 100;
    for (std::size_t i = end + 1; i < timestamp.size() && std::isdigit(timestamp[i]) && scale > 0;
         ++i, scale /= 10) {
      ms += (timestamp[i] - '0') * scale;
    }
  }
  return ms;
}

/// Time between two consecutive Sentinel samples, when it is plausible.
std::optional<milliseconds> interval_between(const std::string& earlier, const std::string& later) {
  const auto from = timestamp_ms(earlier);
  const auto to = timestamp_ms(later);
  if (!from || !to || *to <= *from || *to - *from > kLongestInterval.count()) {
    return std::nullopt;
  }
  return milliseconds(*to - *from);
}

} // namespace

Telemetry internal::telemetry_layout(const json& metrics) {
  std::vector<std::string> die_sensors;
  for (int i = 0; i < kDieSensorCount; ++i) {
    die_sensors.push_back("rtsn_" + std::to_string(i));
  }

  Telemetry telemetry;
  telemetry.power_som_watts = unmeasured(metrics, "power_current_watts");
  telemetry.power_mla_watts = unmeasured(metrics, "power_rail_mla_watts");
  telemetry.power_rails_watts =
      unmeasured_each(metrics, keys_like(metrics, "power_rail_", "_watts"));
  telemetry.die_temperatures_c = unmeasured_each(metrics, die_sensors);
  telemetry.die_temperature_max_c.key = "die_max";
  telemetry.die_temperature_max_c.label = "Die max";
  telemetry.die_temperature_max_c.unit = telemetry.die_temperatures_c.front().unit;
  telemetry.board_temperatures_c = unmeasured_each(metrics, {"lm96163_temp1", "lm96163_temp2"});
  telemetry.cpu_usage_pct = unmeasured(metrics, "cpu_usage_pct");
  telemetry.cpu_cores_usage_pct =
      unmeasured_each(metrics, keys_like(metrics, "cpu_core_", "_usage_pct"));
  telemetry.mla_memory_mib = unmeasured_memory(metrics, "mla_mem_allocated_mb");
  telemetry.cma_memory_mib = unmeasured_memory(metrics, "ev74_cma_used_mb");
  telemetry.linux_memory_used_mib = unmeasured_memory(metrics, "linux_mem_used_mb");
  return telemetry;
}

void internal::count_sample(Telemetry& telemetry, const json& sample) {
  const auto found = sample.find("values");
  const json values = found != sample.end() && found->is_object() ? *found : json::object();

  ++telemetry.samples;
  for_each_metric(telemetry, [&](TelemetryStats& stats) {
    if (const auto value = reading(values, stats.key)) {
      add(stats, *value);
    }
  });
  std::optional<double> hottest;
  for (const TelemetryStats& sensor : telemetry.die_temperatures_c) {
    if (const auto value = reading(values, sensor.key)) {
      hottest = std::max(hottest.value_or(*value), *value);
    }
  }
  if (hottest) {
    add(telemetry.die_temperature_max_c, *hottest);
  }
}

std::filesystem::path Telemetry::save(const std::filesystem::path& file_path) const {
  constexpr const char* kContext = "Telemetry::save";
  pipeline_internal::save_report(file_path, pipeline_internal::telemetry_document(*this, kContext),
                                 kYamlHeader, kContext);
  return file_path;
}

std::ostream& operator<<(std::ostream& out, const Telemetry& telemetry) {
  constexpr const char* kContext = "Telemetry";
  return out << pipeline_internal::report_yaml(
             pipeline_internal::telemetry_document(telemetry, kContext), kYamlHeader, kContext);
}

/**
 * Counts each Sentinel sample once. Sentinel samples on a fixed schedule and stamps each sample
 * with its wall-clock time. The worker asks for the latest sample shortly before the next one is
 * due, then every tenth of an interval until it arrives. A new sample is counted directly when
 * its timestamp is one interval after the last counted one. Otherwise a sample may have been
 * missed, or the clock jumped, so the worker counts every sample after the last counted one, by
 * position, from Sentinel's cache of its latest 240 samples.
 */
class TelemetryMonitor::Impl {
public:
  explicit Impl(pipeline_internal::SentinelClient sentinel) : sentinel_(std::move(sentinel)) {
    summary_ = internal::telemetry_layout(json::array());
    summary_.status = kNotStarted;
  }

  ~Impl() {
    stop();
  }

  void start() {
    std::lock_guard<std::mutex> control(control_mu_);
    if (running_) {
      return;
    }
    {
      std::lock_guard<std::mutex> lock(mu_);
      summary_ = internal::telemetry_layout(json::array());
      summary_.status = kWaiting;
      start_time_ = Clock::now();
      end_time_ = {};
      stop_requested_ = false;
    }
    connected_ = false;
    last_timestamp_.clear();
    interval_ = kDefaultInterval;
    interval_known_ = false;
    warned_.clear();
    const Clock::time_point first_poll = connect() ? next_due() : Clock::now() + interval_;
    running_ = true;
    worker_ = std::thread([this, first_poll] { run(first_poll); });
  }

  Telemetry read() const {
    std::lock_guard<std::mutex> lock(mu_);
    Telemetry out = summary_;
    if (start_time_ != Clock::time_point{}) {
      const Clock::time_point end = end_time_ == Clock::time_point{} ? Clock::now() : end_time_;
      out.duration_ms = std::chrono::duration<double, std::milli>(end - start_time_).count();
    }
    return out;
  }

  Telemetry stop() {
    std::lock_guard<std::mutex> control(control_mu_);
    if (running_) {
      {
        std::lock_guard<std::mutex> lock(mu_);
        end_time_ = Clock::now();
        stop_requested_ = true;
      }
      wake_.notify_all();
      worker_.join();
      poll();
      running_ = false;
      std::lock_guard<std::mutex> lock(mu_);
      if (summary_.status == kWaiting) {
        summary_.status = kNoSample;
      }
    }
    return read();
  }

  bool running() const {
    return running_;
  }

private:
  void run(Clock::time_point next) {
    for (;;) {
      {
        std::unique_lock<std::mutex> lock(mu_);
        if (wake_.wait_until(lock, next, [this] { return stop_requested_; })) {
          return;
        }
      }
      next = poll();
    }
  }

  /// Read Sentinel's latest sample and count it when it is new. Returns when to ask again.
  Clock::time_point poll() {
    try {
      if (!connected_) {
        return connect() ? next_due() : Clock::now() + interval_;
      }
      const json sample = sentinel_.latest_sample();
      const Clock::time_point now = Clock::now();
      const std::string timestamp = text(sample, "timestamp");
      if (timestamp == last_timestamp_) {
        if (now - last_seen_ > 3 * interval_) {
          warn("Sentinel has taken no new sample for three sample intervals; counting resumes "
               "when it does");
        }
        return now + interval_ / 10;
      }
      const auto gap = interval_between(last_timestamp_, timestamp);
      if (interval_known_ && gap && *gap < interval_ * 3 / 2) {
        count(sample);
      } else {
        count_since_last(sample);
      }
      last_seen_ = now;
      return next_due();
    } catch (const std::exception& e) {
      fail(e.what());
      return Clock::now() + interval_;
    }
  }

  /// Shortly before Sentinel's next sample is due.
  Clock::time_point next_due() const {
    return last_seen_ + interval_ * 9 / 10;
  }

  /// Read the cache for the metric definitions, version, and sample interval. Its newest
  /// sample is the baseline: it predates the window and is not counted.
  bool connect() {
    json cache;
    try {
      cache = sentinel_.cache();
    } catch (const std::exception& e) {
      fail(e.what());
      return false;
    }
    const json& samples = cache.at("samples");
    if (!samples.empty()) {
      last_timestamp_ = text(samples.back(), "timestamp");
    }
    if (samples.size() >= 2) {
      if (const auto interval =
              interval_between(text(samples[samples.size() - 2], "timestamp"), last_timestamp_)) {
        interval_ = *interval;
        interval_known_ = true;
      }
    }
    last_seen_ = Clock::now();
    connected_ = true;
    std::lock_guard<std::mutex> lock(mu_);
    summary_ = internal::telemetry_layout(cache.at("metrics"));
    summary_.status = kWaiting;
    summary_.sentinel_version = text(cache, "version");
    summary_.sample_interval_ms = static_cast<int>(interval_.count());
    return true;
  }

  /// Count every cached sample after the last counted one. When the cache no longer holds that
  /// sample, the samples in between are lost and only `latest` is counted.
  void count_since_last(const json& latest) {
    json cache;
    try {
      cache = sentinel_.cache();
    } catch (const std::exception& e) {
      fail(e.what());
      count(latest);
      return;
    }
    {
      std::lock_guard<std::mutex> lock(mu_);
      summary_.sentinel_version = text(cache, "version");
    }
    const json& samples = cache.at("samples");
    auto first_new = samples.begin();
    if (!last_timestamp_.empty()) {
      const auto last = std::find_if(samples.rbegin(), samples.rend(), [&](const json& sample) {
        return text(sample, "timestamp") == last_timestamp_;
      });
      if (last == samples.rend()) {
        warn("Sentinel's cache no longer holds the last counted sample, so the samples since "
             "then are missing");
        last_timestamp_.clear(); // Not consecutive: `latest` must not set the interval.
        count(latest);
        return;
      }
      first_new = last.base();
    }
    std::for_each(first_new, samples.end(), [this](const json& sample) { count(sample); });
  }

  void count(const json& sample) {
    const std::string timestamp = text(sample, "timestamp");
    if (!interval_known_ && !last_timestamp_.empty()) {
      if (const auto interval = interval_between(last_timestamp_, timestamp)) {
        interval_ = *interval;
        interval_known_ = true;
      }
    }
    last_timestamp_ = timestamp;
    std::lock_guard<std::mutex> lock(mu_);
    internal::count_sample(summary_, sample);
    summary_.status = "ok";
    summary_.sample_interval_ms = static_cast<int>(interval_.count());
  }

  /// Warn about a Sentinel failure; until a sample is counted, it is also the status.
  void fail(const std::string& reason) {
    warn(reason);
    std::lock_guard<std::mutex> lock(mu_);
    if (summary_.samples == 0) {
      summary_.status = reason;
    }
  }

  /// Print each distinct warning once per window.
  void warn(const std::string& message) {
    if (warned_.insert(message).second) {
      std::cerr << "[WARN] TelemetryMonitor: " << message << "\n";
    }
  }

  pipeline_internal::SentinelClient sentinel_;

  std::mutex control_mu_; // Serializes start() and stop().
  std::atomic<bool> running_{false};
  std::thread worker_;

  // Shared with read() and stop().
  mutable std::mutex mu_;
  std::condition_variable wake_;
  bool stop_requested_ = false;
  Telemetry summary_;
  Clock::time_point start_time_{};
  Clock::time_point end_time_{};

  // Owned by whichever of start(), the worker, and stop() runs; they never overlap.
  bool connected_ = false;
  std::string last_timestamp_;
  Clock::time_point last_seen_{};
  milliseconds interval_ = kDefaultInterval;
  bool interval_known_ = false;
  std::set<std::string> warned_;
};

TelemetryMonitor::TelemetryMonitor()
    : impl_(std::make_unique<Impl>(pipeline_internal::SentinelClient(
          pipeline_internal::kSentinelSocketPath, kRequestTimeout))) {}

TelemetryMonitor::~TelemetryMonitor() = default;
TelemetryMonitor::TelemetryMonitor(TelemetryMonitor&&) noexcept = default;
TelemetryMonitor& TelemetryMonitor::operator=(TelemetryMonitor&&) noexcept = default;

void TelemetryMonitor::start() {
  impl_->start();
}

Telemetry TelemetryMonitor::read() const {
  return impl_->read();
}

Telemetry TelemetryMonitor::stop() {
  return impl_->stop();
}

bool TelemetryMonitor::running() const {
  return impl_->running();
}

TelemetryMonitor internal::TelemetryMonitorAccess::make(std::string sentinel_socket_path) {
  TelemetryMonitor monitor;
  monitor.impl_ = std::make_unique<TelemetryMonitor::Impl>(
      pipeline_internal::SentinelClient(std::move(sentinel_socket_path), kRequestTimeout));
  return monitor;
}

} // namespace simaai::neat
