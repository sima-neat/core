#include "pipeline/TelemetryMonitor.h"
#include "pipeline/internal/TelemetryMonitorInternal.h"
#include "test_main.h"
#include "test_utils.h"

#include <glib.h>
#include <httplib.h>
#include <nlohmann/json.hpp>
#include <sys/socket.h>
#include <unistd.h>
#include <yaml-cpp/yaml.h>

#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <deque>
#include <filesystem>
#include <fstream>
#include <functional>
#include <iostream>
#include <limits>
#include <map>
#include <mutex>
#include <sstream>
#include <stdexcept>
#include <string>
#include <thread>
#include <utility>
#include <vector>

namespace {

using nlohmann::json;
using simaai::neat::Telemetry;
using simaai::neat::TelemetryMonitor;
using simaai::neat::TelemetryStats;
using simaai::neat::internal::count_sample;
using simaai::neat::internal::telemetry_layout;
using simaai::neat::internal::TelemetryMonitorAccess;
using Clock = std::chrono::steady_clock;
using namespace std::chrono_literals;

constexpr int kIntervalMs = 50; // The fake Sentinel's sample interval.

void require_near(double actual, double expected, const std::string& label) {
  require(std::abs(actual - expected) <= 1e-9,
          label + ": expected " + std::to_string(expected) + ", got " + std::to_string(actual));
}

json metric_definition(const std::string& key, const std::string& label, const std::string& unit) {
  return {{"key", key},      {"label", label},    {"short", key},    {"unit", unit},
          {"group", "test"}, {"description", ""}, {"warn", nullptr}, {"critical", nullptr}};
}

/// Sentinel's metric definitions on a Modalix SoM with two CPU cores.
json modalix_metrics() {
  json metrics = json::array({
      metric_definition("power_current_watts", "SoM power", "W"),
      metric_definition("power_average_watts", "Average power", "W"),
      metric_definition("power_rail_mla_watts", "MLA", "W"),
      metric_definition("power_rail_soc_ddr_vdd_watts", "DDR VDD", "W"),
      metric_definition("cpu_usage_pct", "CPU", "%"),
      metric_definition("cpu_core_0_usage_pct", "CPU core 0", "%"),
      metric_definition("cpu_core_1_usage_pct", "CPU core 1", "%"),
      metric_definition("lm96163_temp1", "SoM top", "C"),
      metric_definition("lm96163_temp2", "SoM bottom", "C"),
      metric_definition("mla_mem_allocated_mb", "MLA memory", "MB"),
      metric_definition("ev74_cma_used_mb", "CMA used", "MB"),
      metric_definition("linux_mem_used_mb", "Linux used", "MB"),
  });
  for (int i = 0; i < 14; ++i) {
    metrics.push_back(
        metric_definition("rtsn_" + std::to_string(i), "Sensor " + std::to_string(i), "C"));
  }
  return metrics;
}

json modalix_sample(double power, double mla, double hot_site, json cpu, double mla_mem) {
  json values = {{"power_current_watts", power},
                 {"power_average_watts", 100.0},
                 {"power_rail_mla_watts", mla},
                 {"power_rail_soc_ddr_vdd_watts", 0.5},
                 {"cpu_usage_pct", cpu},
                 {"cpu_core_0_usage_pct", 100.0},
                 {"cpu_core_1_usage_pct", std::move(cpu)},
                 {"lm96163_temp1", 40.0},
                 {"lm96163_temp2", 42.0},
                 {"mla_mem_allocated_mb", mla_mem},
                 {"ev74_cma_used_mb", 300.0 + mla_mem},
                 {"linux_mem_used_mb", 900.0}};
  for (int i = 0; i < 7; ++i) {
    values["rtsn_" + std::to_string(i)] = 50.0 + i;
  }
  values["rtsn_4"] = hot_site;
  values["rtsn_9"] = 200.0; // Trip channel: must not reach any summary.
  return json{{"timestamp", "2026-10-19T08:00:00Z"}, {"values", std::move(values)}};
}

Telemetry modalix_telemetry() {
  Telemetry t = telemetry_layout(modalix_metrics());
  count_sample(t, modalix_sample(9.0, 3.0, 60.0, 40.0, 100.0));
  count_sample(t, modalix_sample(10.0, 4.0, 52.0, nullptr, 180.0));
  count_sample(t, modalix_sample(11.0, 5.0, 70.0, 80.0, 160.0));
  return t;
}

// --- Summary --------------------------------------------------------------------------------

void summary_maps_sentinel_metrics() {
  const Telemetry t = modalix_telemetry();
  require(t.samples == 3, "every counted sample is a sample");

  const TelemetryStats& som = t.power_som_watts;
  require(som.key == "power_current_watts" && som.label == "SoM power" && som.unit == "W",
          "SoM power uses the Sentinel definition");
  require(som.samples == 3, "SoM power samples");
  require_near(som.mean, 10.0, "SoM power mean");
  require_near(som.minimum, 9.0, "SoM power minimum");
  require_near(som.maximum, 11.0, "SoM power maximum");
  require_near(som.last, 11.0, "SoM power last");
  require_near(t.power_mla_watts.mean, 4.0, "MLA rail mean");

  require(t.power_rails_watts.size() == 2, "one entry per reported power rail");
  require(t.power_rails_watts[0].key == "power_rail_mla_watts" &&
              t.power_rails_watts[1].key == "power_rail_soc_ddr_vdd_watts",
          "rails follow the Sentinel definition order");

  require(t.die_temperatures_c.size() == 7, "die temperatures cover rtsn_0..6 only");
  require(t.die_temperatures_c[4].label == "Sensor 4", "die labels come from Sentinel");
  require(t.die_temperature_max_c.key == "die_max" && t.die_temperature_max_c.unit == "C",
          "die max is a derived metric");
  require_near(t.die_temperature_max_c.minimum, 56.0, "die max per sample uses the hottest site");
  require_near(t.die_temperature_max_c.maximum, 70.0, "die max peak");

  require(t.board_temperatures_c.size() == 2 && t.board_temperatures_c[1].label == "SoM bottom",
          "board temperatures are the two SoM sensors");

  require(t.cpu_usage_pct.samples == 2, "null CPU readings are skipped");
  require_near(t.cpu_usage_pct.last, 80.0, "last skips nothing it should keep");
  require(t.cpu_cores_usage_pct.size() == 2 &&
              t.cpu_cores_usage_pct[0].key == "cpu_core_0_usage_pct" &&
              t.cpu_cores_usage_pct[1].label == "CPU core 1",
          "one entry per CPU core, in Sentinel definition order");

  const TelemetryStats& mla = t.mla_memory_mib;
  require(mla.key == "mla_mem_allocated_mb" && mla.unit == "MiB" && mla.samples == 3,
          "memory is relabelled MiB");
  require_near(t.cma_memory_mib.maximum, 480.0, "CMA maximum");
  require_near(t.linux_memory_used_mib.mean, 900.0, "Linux memory mean");
}

void summary_skips_unreadable_values() {
  Telemetry t = telemetry_layout(modalix_metrics());
  json sample = modalix_sample(9.0, 3.0, 60.0, 40.0, 100.0);
  sample["values"]["power_current_watts"] = std::numeric_limits<double>::infinity();
  sample["values"]["cpu_usage_pct"] = std::nan("");
  sample["values"]["lm96163_temp1"] = "n/a";
  count_sample(t, sample);
  require(t.samples == 1, "a sample with unreadable values is still a sample");
  require(t.power_som_watts.samples == 0 && t.cpu_usage_pct.samples == 0 &&
              t.board_temperatures_c[0].samples == 0,
          "non-finite and non-numeric values are not counted");
  require(t.power_mla_watts.samples == 1, "readable values in the same sample are counted");
}

void summary_without_mla_rail() {
  json metrics = modalix_metrics();
  metrics.erase(2); // power_rail_mla_watts
  Telemetry t = telemetry_layout(metrics);
  json sample = modalix_sample(9.0, 3.0, 60.0, 40.0, 100.0);
  sample["values"].erase("power_rail_mla_watts");
  count_sample(t, sample);
  require(t.power_mla_watts.samples == 0 && t.power_mla_watts.label == "power_rail_mla_watts",
          "a board without an MLA rail reports it as not measured, labelled by key");
  require(t.power_rails_watts.size() == 1, "rails list only the rails the board reports");
}

void summary_size_does_not_grow_with_the_window() {
  // 20 hours of Sentinel samples at 2 s.
  constexpr int kSamples = 36000;
  Telemetry t = telemetry_layout(modalix_metrics());
  const auto measurements = [](const Telemetry& value) {
    return value.power_rails_watts.size() + value.die_temperatures_c.size() +
           value.board_temperatures_c.size() + value.cpu_cores_usage_pct.size();
  };
  const std::size_t before = measurements(t);
  for (int i = 0; i < kSamples; ++i) {
    count_sample(t, modalix_sample(5.0 + i % 10, 3.0, 60.0, 40.0, 100.0));
  }
  require(t.samples == kSamples && t.power_som_watts.samples == kSamples, "every sample counted");
  require(measurements(t) == before, "the summary keeps one entry per measurement");
  require(std::abs(t.power_som_watts.mean - 9.5) < 1e-9, "the running mean stays exact");
  require_near(t.power_som_watts.minimum, 5.0, "minimum over 20 hours");
  require_near(t.power_som_watts.maximum, 14.0, "maximum over 20 hours");
  require_near(t.power_som_watts.last, 14.0, "last of 20 hours");
}

// --- Saving and printing --------------------------------------------------------------------

struct TemporaryDirectory {
  std::filesystem::path path;
  TemporaryDirectory() {
    GError* error = nullptr;
    gchar* value = g_dir_make_tmp("neat-telemetry-XXXXXX", &error);
    if (!value) {
      const std::string detail = error ? error->message : "unknown error";
      g_clear_error(&error);
      throw std::runtime_error(detail);
    }
    path = value;
    g_free(value);
  }
  ~TemporaryDirectory() {
    std::error_code ignored;
    std::filesystem::remove_all(path, ignored);
  }
};

std::string read_file(const std::filesystem::path& path) {
  std::ifstream file(path);
  std::ostringstream contents;
  contents << file.rdbuf();
  return contents.str();
}

void telemetry_saves_and_prints_the_same_yaml() {
  TemporaryDirectory dir;
  Telemetry t = modalix_telemetry();
  t.die_temperature_max_c.unit = "°C"; // Sentinel text may hold non-ASCII.

  const auto yaml_path = dir.path / "nested" / "telemetry.yaml";
  require(t.save(yaml_path) == yaml_path, "save returns the caller's path");
  std::ostringstream printed;
  printed << t;
  require(printed.str() == read_file(yaml_path), "printing writes the text save() writes");
  require(printed.str().starts_with("# NEAT board telemetry"), "YAML starts with its header");

  const YAML::Node yaml = YAML::LoadFile(yaml_path.string());
  require(yaml["samples"].as<int>() == 3 && yaml["power_som_watts"]["last"].as<double>() == 11.0,
          "YAML holds the summary");
  require(yaml["die_temperature_max_c"]["unit"].as<std::string>() == "°C",
          "non-ASCII units survive the escaped YAML");
  require(!yaml["trace_id"], "telemetry has no trace");

  const auto json_path = dir.path / "telemetry.json";
  t.save(json_path);
  const json saved = json::parse(read_file(json_path));
  require(saved["samples"] == 3 && saved["duration_ms"] == 0.0 &&
              saved["cpu_cores_usage_pct"][1]["label"] == "CPU core 1" &&
              saved["power_som_watts"]["mean"] == 10.0,
          "JSON holds the same summary");

  Telemetry invalid = t;
  invalid.power_som_watts.last = std::numeric_limits<double>::infinity();
  bool threw = false;
  try {
    invalid.save(json_path);
  } catch (const std::invalid_argument&) {
    threw = true;
  }
  require(threw && json::parse(read_file(json_path)) == saved,
          "an invalid value throws and keeps the existing file");
}

// --- Monitor ------------------------------------------------------------------------------

/// Sentinel's timestamp `ms` after 08:00 UTC on the test day, such as
/// `2026-10-19T08:00:00.050000Z`.
std::string timestamp_at(std::int64_t ms) {
  const std::int64_t of_day = 8 * 3600 * 1000 + ms;
  char text[40];
  std::snprintf(text, sizeof(text), "2026-10-19T%02d:%02d:%02d.%06dZ",
                static_cast<int>(of_day / 3600000), static_cast<int>(of_day / 60000 % 60),
                static_cast<int>(of_day / 1000 % 60), static_cast<int>(of_day % 1000 * 1000));
  return text;
}

/// Sentinel on a Unix socket. The test takes its samples; each is `kIntervalMs` after the
/// previous one by timestamp.
class FakeSentinel {
public:
  explicit FakeSentinel(std::size_t cache_size = 240) : cache_size_(cache_size) {
    char dir_template[] = "/tmp/neat-sentinel-XXXXXX";
    if (::mkdtemp(dir_template) == nullptr) {
      throw std::runtime_error("mkdtemp failed");
    }
    dir_ = dir_template;
    path_ = dir_ + "/api.sock";
    server_.Get("/v1/samples/latest", [this](const httplib::Request&, httplib::Response& res) {
      answer(res, "/v1/samples/latest", [this] {
        return json{{"schema", 1},
                    {"version", "0.1.0"},
                    {"sample", samples_.empty() ? json() : samples_.back()}};
      });
    });
    server_.Get("/v1/cache", [this](const httplib::Request&, httplib::Response& res) {
      answer(res, "/v1/cache", [this] {
        return json{
            {"schema", 1},
            {"version", "0.1.0"},
            {"metrics", json::array({metric_definition("power_current_watts", "SoM power", "W")})},
            {"samples", json(samples_)}};
      });
    });
    server_.set_address_family(AF_UNIX);
  }

  ~FakeSentinel() {
    if (thread_.joinable()) {
      server_.stop();
      thread_.join();
    }
    ::unlink(path_.c_str());
    ::rmdir(dir_.c_str());
  }

  /// Start answering, like Sentinel starting on the board.
  void start() {
    thread_ = std::thread([this] { server_.listen(path_, 80); });
    server_.wait_until_ready();
  }

  /// Take a sample with this SoM power, one interval after the previous one or at `timestamp`.
  void take_sample(double power, std::string timestamp = {}) {
    std::lock_guard<std::mutex> lock(mu_);
    append(power, std::move(timestamp));
  }

  /// Take these samples at once, faster than the monitor asks for them.
  void take_burst(const std::vector<double>& powers) {
    std::lock_guard<std::mutex> lock(mu_);
    for (double power : powers) {
      append(power, {});
    }
  }

  /// Answer every request with this HTTP status; 200 serves samples again.
  void answer_with(int status) {
    std::lock_guard<std::mutex> lock(mu_);
    status_ = status;
  }

  /// Wait this long before each answer.
  void delay_answers(std::chrono::milliseconds delay) {
    std::lock_guard<std::mutex> lock(mu_);
    delay_ = delay;
  }

  int requests(const std::string& path) {
    std::lock_guard<std::mutex> lock(mu_);
    return requests_[path];
  }

  TelemetryMonitor monitor() const {
    return TelemetryMonitorAccess::make(path_);
  }

private:
  void append(double power, std::string timestamp) {
    next_ms_ += kIntervalMs;
    samples_.push_back({{"timestamp", timestamp.empty() ? timestamp_at(next_ms_) : timestamp},
                        {"values", {{"power_current_watts", power}}}});
    if (samples_.size() > cache_size_) {
      samples_.pop_front();
    }
  }

  void answer(httplib::Response& res, const std::string& path, const std::function<json()>& body) {
    std::chrono::milliseconds delay{};
    {
      std::lock_guard<std::mutex> lock(mu_);
      ++requests_[path];
      delay = delay_;
    }
    std::this_thread::sleep_for(delay);
    std::lock_guard<std::mutex> lock(mu_);
    res.status = status_;
    res.set_content(status_ == 200 ? body().dump() : json{{"error", "busy"}}.dump(),
                    "application/json");
  }

  std::size_t cache_size_;
  std::deque<json> samples_;
  std::int64_t next_ms_ = 0;
  int status_ = 200;
  std::chrono::milliseconds delay_{};
  std::map<std::string, int> requests_;
  std::mutex mu_;
  std::string dir_;
  std::string path_;
  httplib::Server server_;
  std::thread thread_;
};

/// A running fake Sentinel that already holds two samples, so the monitor learns its interval.
struct RunningSentinel : FakeSentinel {
  explicit RunningSentinel(std::size_t cache_size = 240) : FakeSentinel(cache_size) {
    take_sample(100.0);
    take_sample(100.0);
    start();
  }
};

/// Captures what the monitor writes to stderr.
class CapturedStderr {
public:
  CapturedStderr() : previous_(std::cerr.rdbuf(text_.rdbuf())) {}
  ~CapturedStderr() {
    std::cerr.rdbuf(previous_);
  }
  int warnings() const {
    int count = 0;
    const std::string text = text_.str();
    for (auto at = text.find("[WARN] TelemetryMonitor: "); at != std::string::npos;
         at = text.find("[WARN] TelemetryMonitor: ", at + 1)) {
      ++count;
    }
    return count;
  }
  std::string text() const {
    return text_.str();
  }

private:
  std::ostringstream text_;
  std::streambuf* previous_;
};

/// Wait until the monitor has counted `samples` samples.
void wait_for_samples(const TelemetryMonitor& monitor, int samples) {
  const auto deadline = Clock::now() + 10s;
  while (monitor.read().samples < samples && Clock::now() < deadline) {
    std::this_thread::sleep_for(5ms);
  }
  require(monitor.read().samples == samples, "expected " + std::to_string(samples) +
                                                 " samples, counted " +
                                                 std::to_string(monitor.read().samples));
}

/// Take samples with these powers, one Sentinel interval apart in real time.
void take_samples_over_time(FakeSentinel& sentinel, const std::vector<double>& powers) {
  for (double power : powers) {
    std::this_thread::sleep_for(std::chrono::milliseconds(kIntervalMs));
    sentinel.take_sample(power);
  }
}

void monitor_counts_each_new_sample_once() {
  RunningSentinel sentinel;
  TelemetryMonitor monitor = sentinel.monitor();
  monitor.start();
  require(monitor.running(), "a started monitor is running");
  take_samples_over_time(sentinel, {1.0, 2.0, 3.0, 4.0, 5.0});
  wait_for_samples(monitor, 5);
  std::this_thread::sleep_for(std::chrono::milliseconds(2 * kIntervalMs));

  const Telemetry t = monitor.stop();
  require(!monitor.running(), "a stopped monitor is not running");
  require(t.status == "ok", "status is ok once a sample is counted; got " + t.status);
  require(t.samples == 5, "re-reading the same sample does not count it again");
  require(t.sentinel_version == "0.1.0" && t.sample_interval_ms == kIntervalMs,
          "version and interval come from Sentinel's cache");
  require(t.power_som_watts.maximum == 5.0, "samples taken before start() are not counted");
  require_near(t.power_som_watts.mean, 3.0, "mean of the counted samples");
  require_near(t.power_som_watts.last, 5.0, "last counted sample");
  require(t.duration_ms > 0.0, "the window has a duration");
}

void monitor_counts_a_burst_of_samples_from_the_cache() {
  RunningSentinel sentinel;
  TelemetryMonitor monitor = sentinel.monitor();
  monitor.start();
  sentinel.take_burst({1.0, 2.0, 3.0, 4.0});
  wait_for_samples(monitor, 4);
  const Telemetry t = monitor.stop();
  require(t.samples == 4 && t.power_som_watts.last == 4.0,
          "samples the monitor did not see one by one are counted from the cache, in order");
}

void monitor_catches_up_after_sentinel_errors() {
  RunningSentinel sentinel;
  CapturedStderr stderr_text;
  TelemetryMonitor monitor = sentinel.monitor();
  monitor.start();
  sentinel.answer_with(500);
  take_samples_over_time(sentinel, {1.0, 2.0, 3.0});
  std::this_thread::sleep_for(std::chrono::milliseconds(2 * kIntervalMs));
  require(monitor.read().status.find("HTTP 500") != std::string::npos,
          "before any sample, status is the error; got " + monitor.read().status);
  sentinel.answer_with(200);
  take_samples_over_time(sentinel, {4.0});
  wait_for_samples(monitor, 4);
  const Telemetry t = monitor.stop();
  require(t.status == "ok" && t.power_som_watts.last == 4.0, "counting resumes in order");
  require(stderr_text.warnings() == 1, "a repeated error warns once:\n" + stderr_text.text());
}

void monitor_counts_only_the_newest_after_a_gap_longer_than_the_cache() {
  RunningSentinel sentinel(/*cache_size=*/3);
  CapturedStderr stderr_text;
  TelemetryMonitor monitor = sentinel.monitor();
  monitor.start();
  sentinel.take_burst({1.0, 2.0, 3.0, 4.0, 5.0, 6.0});
  wait_for_samples(monitor, 1);
  const Telemetry t = monitor.stop();
  require(t.samples == 1 && t.power_som_watts.last == 6.0,
          "only the newest sample is counted when the cache lost the rest");
  require(stderr_text.text().find("samples since then are missing") != std::string::npos,
          "the lost samples are reported:\n" + stderr_text.text());
}

void a_sentinel_that_stops_sampling_warns_once() {
  RunningSentinel sentinel;
  CapturedStderr stderr_text;
  TelemetryMonitor monitor = sentinel.monitor();
  monitor.start();
  take_samples_over_time(sentinel, {1.0});
  wait_for_samples(monitor, 1);
  std::this_thread::sleep_for(std::chrono::milliseconds(8 * kIntervalMs)); // Sentinel stalls.
  take_samples_over_time(sentinel, {2.0});
  wait_for_samples(monitor, 2);
  const Telemetry t = monitor.stop();
  require(t.status == "ok" && t.power_som_watts.last == 2.0,
          "counting resumes when Sentinel samples again");
  require(stderr_text.warnings() == 1 &&
              stderr_text.text().find("no new sample for three sample intervals") !=
                  std::string::npos,
          "a stall warns once:\n" + stderr_text.text());
}

void monitor_counts_a_sample_after_the_clock_moves_back() {
  RunningSentinel sentinel;
  TelemetryMonitor monitor = sentinel.monitor();
  monitor.start();
  take_samples_over_time(sentinel, {1.0});
  wait_for_samples(monitor, 1);
  sentinel.take_sample(2.0, timestamp_at(-60000)); // An NTP step back by a minute.
  wait_for_samples(monitor, 2);
  require(monitor.stop().power_som_watts.last == 2.0, "an earlier timestamp is a new sample");
}

void monitor_restart_begins_a_new_window() {
  RunningSentinel sentinel;
  TelemetryMonitor monitor = sentinel.monitor();
  monitor.start();
  take_samples_over_time(sentinel, {1.0, 2.0});
  wait_for_samples(monitor, 2);
  monitor.stop();

  monitor.start();
  take_samples_over_time(sentinel, {7.0});
  wait_for_samples(monitor, 1);
  const Telemetry t = monitor.stop();
  require(t.samples == 1 && t.power_som_watts.mean == 7.0, "a restart counts from zero");
}

void monitor_lifecycle() {
  RunningSentinel sentinel;
  TelemetryMonitor monitor = sentinel.monitor();
  const Telemetry before = monitor.read();
  require(before.status == "not started" && before.samples == 0 && !monitor.running(),
          "read() before start() reports a monitor that has not started");
  require(sentinel.requests("/v1/cache") == 0, "constructing a monitor contacts nothing");

  monitor.start();
  monitor.start(); // Already running: no new window.
  require(sentinel.requests("/v1/cache") == 1, "start() while running does nothing");
  const Telemetry first = monitor.stop();
  require(first.status.find("no new sample") != std::string::npos,
          "a window without a sample says why; got " + first.status);
  std::this_thread::sleep_for(20ms);
  const Telemetry second = monitor.stop();
  require(second.duration_ms == first.duration_ms && second.status == first.status,
          "stop() again returns the same result");
  require(monitor.read().duration_ms == first.duration_ms, "read() after stop() is frozen");

  TelemetryMonitor moved = std::move(monitor);
  moved.start();
  require(moved.running(), "a moved monitor works");
}

void destroying_a_running_monitor_stops_it() {
  RunningSentinel sentinel;
  {
    TelemetryMonitor monitor = sentinel.monitor();
    monitor.start();
    take_samples_over_time(sentinel, {1.0});
    wait_for_samples(monitor, 1);
  }
  const int requests = sentinel.requests("/v1/samples/latest");
  std::this_thread::sleep_for(std::chrono::milliseconds(4 * kIntervalMs));
  require(sentinel.requests("/v1/samples/latest") == requests, "the destructor stops the worker");
}

void a_hung_sentinel_cannot_block_stop() {
  RunningSentinel sentinel;
  CapturedStderr stderr_text;
  TelemetryMonitor monitor = sentinel.monitor();
  monitor.start();
  sentinel.delay_answers(1500ms); // Longer than the monitor's 1 s request timeout.
  std::this_thread::sleep_for(std::chrono::milliseconds(2 * kIntervalMs));
  const auto stop_started = Clock::now();
  monitor.stop();
  require(Clock::now() - stop_started < 3s, "stop() waits at most for its request timeouts");
  sentinel.delay_answers(0ms);
}

void missing_sentinel_warns_once_and_never_throws() {
  CapturedStderr stderr_text;
  TelemetryMonitor monitor = TelemetryMonitorAccess::make("/tmp/neat-no-such-sentinel.sock");
  monitor.start();
  std::this_thread::sleep_for(50ms);
  const Telemetry t = monitor.stop();
  require(t.samples == 0, "no telemetry without Sentinel");
  require(t.status.find("Sentinel is not running on this board") != std::string::npos &&
              t.status.find("sima-cli neat install sentinel") != std::string::npos &&
              t.status.find("sudo systemctl start simaai-sentinel") != std::string::npos,
          "status says how to install or start Sentinel; got " + t.status);
  require(stderr_text.warnings() == 1, "one warning for the whole window:\n" + stderr_text.text());
}

void sentinel_that_starts_later_is_counted() {
  FakeSentinel sentinel;
  sentinel.take_sample(100.0);
  sentinel.take_sample(100.0);
  CapturedStderr stderr_text;
  TelemetryMonitor monitor = sentinel.monitor();
  monitor.start(); // Sentinel is not answering yet.
  require(monitor.read().status.find("not running") != std::string::npos,
          "the status names the missing Sentinel");
  sentinel.start();
  // The monitor retries after its 2 s default interval, then counts the samples after that.
  std::this_thread::sleep_for(2500ms);
  take_samples_over_time(sentinel, {1.0, 2.0});
  wait_for_samples(monitor, 2);
  require(monitor.stop().status == "ok", "counting starts once Sentinel answers");
}

void old_sentinel_gets_an_upgrade_hint() {
  RunningSentinel sentinel;
  sentinel.answer_with(404);
  CapturedStderr stderr_text;
  TelemetryMonitor monitor = sentinel.monitor();
  monitor.start();
  const Telemetry t = monitor.stop();
  require(t.status.find("Sentinel is too old") != std::string::npos &&
              t.status.find("sima-cli neat install sentinel") != std::string::npos,
          "a missing endpoint asks for an upgrade; got " + t.status);
}

void interval_falls_back_until_sentinel_has_two_samples() {
  FakeSentinel sentinel;
  sentinel.take_sample(100.0);
  sentinel.start();
  TelemetryMonitor monitor = sentinel.monitor();
  monitor.start();
  require(monitor.read().sample_interval_ms == 2000,
          "with one cached sample the interval is Sentinel's 2 s default");
  take_samples_over_time(sentinel, {1.0, 2.0});
  wait_for_samples(monitor, 2);
  const Telemetry t = monitor.stop();
  require(t.sample_interval_ms == kIntervalMs, "the interval comes from counted samples");
}

} // namespace

RUN_TEST("unit_telemetry_monitor_test", ([] {
           std::vector<std::string> failures;
           const auto run_case = [&](std::string name, const std::function<void()>& fn) {
             try {
               fn();
             } catch (const std::exception& e) {
               failures.push_back(std::move(name) + ": " + e.what());
             }
           };

           run_case("summary_maps_sentinel_metrics", summary_maps_sentinel_metrics);
           run_case("summary_skips_unreadable_values", summary_skips_unreadable_values);
           run_case("summary_without_mla_rail", summary_without_mla_rail);
           run_case("summary_size", summary_size_does_not_grow_with_the_window);
           run_case("save_and_print", telemetry_saves_and_prints_the_same_yaml);
           run_case("counts_each_sample_once", monitor_counts_each_new_sample_once);
           run_case("burst_from_cache", monitor_counts_a_burst_of_samples_from_the_cache);
           run_case("catches_up_after_errors", monitor_catches_up_after_sentinel_errors);
           run_case("gap_past_cache",
                    monitor_counts_only_the_newest_after_a_gap_longer_than_the_cache);
           run_case("stall_warns_once", a_sentinel_that_stops_sampling_warns_once);
           run_case("clock_moves_back", monitor_counts_a_sample_after_the_clock_moves_back);
           run_case("restart", monitor_restart_begins_a_new_window);
           run_case("lifecycle", monitor_lifecycle);
           run_case("destructor_stops", destroying_a_running_monitor_stops_it);
           run_case("hung_sentinel", a_hung_sentinel_cannot_block_stop);
           run_case("missing_sentinel", missing_sentinel_warns_once_and_never_throws);
           run_case("sentinel_starts_later", sentinel_that_starts_later_is_counted);
           run_case("old_sentinel", old_sentinel_gets_an_upgrade_hint);
           run_case("interval_fallback", interval_falls_back_until_sentinel_has_two_samples);

           std::string message =
               std::to_string(failures.size()) + " telemetry monitor check(s) failed:";
           for (const auto& failure : failures) {
             message += "\n  - " + failure;
           }
           require(failures.empty(), message);
         }));
