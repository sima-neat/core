// Covers the deprecated PowerMonitor API, which keeps working until it is removed.
#pragma GCC diagnostic ignored "-Wdeprecated-declarations"

#include "pipeline/PowerTelemetry.h"
#include "pipeline/internal/PowerTelemetryInternal.h"
#include "test_main.h"
#include "test_utils.h"

#include <httplib.h>
#include <nlohmann/json.hpp>
#include <sys/socket.h>
#include <unistd.h>

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <mutex>
#include <stdexcept>
#include <string>
#include <thread>
#include <utility>
#include <vector>

namespace {

using nlohmann::json;
using simaai::neat::PowerMonitor;
using simaai::neat::PowerMonitorOptions;
using simaai::neat::PowerSummary;
using simaai::neat::internal::PowerMonitorAccess;

void require_near(double actual, double expected, const std::string& label) {
  require(std::abs(actual - expected) < 1e-9,
          label + " expected=" + std::to_string(expected) + " actual=" + std::to_string(actual));
}

/// Reply of Sentinel's `GET /v1/samples/latest`.
json latest_sample(const std::string& timestamp, json mla_watts, json ddr_watts) {
  return {{"schema", 1},
          {"sample",
           {{"timestamp", timestamp},
            {"values",
             {{"power_current_watts", 99.0},
              {"power_rail_mla_watts", std::move(mla_watts)},
              {"power_rail_soc_ddr_vdd_watts", std::move(ddr_watts)},
              {"cpu_usage_pct", 10.0}}}}}};
}

/// Sentinel on a Unix socket that answers each request with the next reply, then the last.
class FakeSentinel {
public:
  explicit FakeSentinel(std::vector<json> replies) : replies_(std::move(replies)) {
    char dir_template[] = "/tmp/neat-sentinel-XXXXXX";
    if (::mkdtemp(dir_template) == nullptr) {
      throw std::runtime_error("mkdtemp failed");
    }
    dir_ = dir_template;
    path_ = dir_ + "/api.sock";
    server_.Get(".*", [this](const httplib::Request& request, httplib::Response& response) {
      std::lock_guard<std::mutex> lock(mu_);
      requests_.push_back(request.path);
      const json& reply = replies_[std::min(requests_.size(), replies_.size()) - 1];
      response.set_content(reply.dump(), "application/json");
    });
    server_.set_address_family(AF_UNIX);
    thread_ = std::thread([this] { server_.listen(path_, 80); });
    server_.wait_until_ready();
  }

  ~FakeSentinel() {
    server_.stop();
    thread_.join();
    ::unlink(path_.c_str());
    ::rmdir(dir_.c_str());
  }

  PowerMonitor monitor(bool enabled = true) const {
    PowerMonitorOptions options;
    options.enabled = enabled;
    return PowerMonitorAccess::make(options, path_);
  }

  std::vector<std::string> requests() {
    std::lock_guard<std::mutex> lock(mu_);
    return requests_;
  }

private:
  std::vector<json> replies_;
  std::vector<std::string> requests_;
  std::mutex mu_;
  std::string dir_;
  std::string path_;
  httplib::Server server_;
  std::thread thread_;
};

void samples_become_rail_summaries() {
  FakeSentinel sentinel({latest_sample("t1", 3.0, 0.5), latest_sample("t1", 3.0, 0.5),
                         latest_sample("t2", 5.0, nullptr)});
  PowerMonitor monitor = sentinel.monitor();
  monitor.sample_once();
  monitor.sample_once();
  monitor.sample_once();
  const PowerSummary summary = monitor.summary();

  require(summary.samples == 2, "a repeated Sentinel sample is added once");
  require(summary.rails.size() == 2, "one rail per power_rail_*_watts metric");
  require(summary.rails[0].config.name == "mla", "rail name comes from the metric key");
  require_near(summary.rails[0].power_w.avg, 4.0, "MLA rail average");
  require(summary.rails[1].config.name == "soc_ddr_vdd" && summary.rails[1].power_w.samples == 1 &&
              summary.rails[1].power_w.errors == 1,
          "a missing rail reading counts as an error");
  require_near(summary.total_avg_watts, 4.25, "total is the sum of the rails with a reading");
  require_near(summary.total_max_watts, 5.0, "total maximum");
  require(summary.rails[0].voltage_v.samples == 0 && summary.rails[0].current_a.samples == 0,
          "Sentinel reports no rail voltage or current");
}

void interval_keeps_only_later_samples() {
  FakeSentinel sentinel({latest_sample("before", 9.0, 9.0), latest_sample("during", 2.0, 1.0)});
  PowerMonitor monitor = sentinel.monitor();
  monitor.start();
  monitor.stop();
  const PowerSummary summary = monitor.summary();

  require(summary.samples == 1, "the sample taken before start() is excluded");
  require_near(summary.total_avg_watts, 3.0, "the sample taken before stop() is kept");
  require(summary.duration_seconds > 0.0, "the interval has a duration");
}

void missing_sentinel_collects_nothing() {
  PowerMonitorOptions options;
  options.enabled = true;
  PowerMonitor monitor = PowerMonitorAccess::make(options, "/tmp/neat-no-such-sentinel.sock");
  monitor.start();
  monitor.stop();
  const PowerSummary summary = monitor.summary();
  require(summary.enabled && summary.samples == 0 && summary.rails.empty(),
          "without Sentinel the monitor collects no samples");
}

void disabled_monitor_reads_nothing() {
  FakeSentinel sentinel({latest_sample("t1", 1.0, 1.0)});
  PowerMonitor monitor = sentinel.monitor(/*enabled=*/false);
  monitor.start();
  monitor.sample_once();
  monitor.stop();
  require(!monitor.summary().enabled && sentinel.requests().empty(),
          "a disabled monitor never reads Sentinel");
}

void summaries_format() {
  PowerSummary summary;
  summary.enabled = true;
  summary.samples = 1;
  summary.duration_seconds = 0.25;
  summary.total_avg_watts = 2.0;
  summary.total_min_watts = 2.0;
  summary.total_max_watts = 2.0;
  summary.energy_joules = 0.5;
  require_contains(format_power_summary(summary), "PowerStats:", "text power report");
  require_contains(power_summary_to_json(summary), "\"total_avg_watts\": 2.000000",
                   "json power report");
  require(format_power_summary(PowerSummary{}).empty(), "disabled power summary is omitted");
  require(simaai::neat::board_power_monitor_options().enabled, "board power options enable it");
}

} // namespace

RUN_TEST("unit_power_telemetry_test", ([] {
           samples_become_rail_summaries();
           interval_keeps_only_later_samples();
           missing_sentinel_collects_nothing();
           disabled_monitor_reads_nothing();
           summaries_format();
         }));
