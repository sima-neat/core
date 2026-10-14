#ifndef SIMA_NEAT_INTERNAL
#define SIMA_NEAT_INTERNAL 1
#endif

#include "model/internal/BenchmarkTelemetry.h"
#include "pipeline/internal/SentinelClient.h"
#include "test_main.h"
#include "test_utils.h"

#include <httplib.h>
#include <nlohmann/json.hpp>
#include <sys/socket.h>
#include <unistd.h>

#include <cmath>
#include <cstdlib>
#include <functional>
#include <mutex>
#include <stdexcept>
#include <string>
#include <thread>
#include <utility>
#include <vector>

namespace {

using nlohmann::json;
using simaai::neat::BenchmarkMetric;
using simaai::neat::BenchmarkTelemetry;
using simaai::neat::internal::SentinelTrace;
using simaai::neat::internal::summarize_benchmark_telemetry;
using simaai::neat::pipeline_internal::SentinelClient;

void require_near(double actual, double expected, const std::string& label) {
  require(std::abs(actual - expected) <= 1e-9,
          label + ": expected " + std::to_string(expected) + ", got " + std::to_string(actual));
}

json metric_definition(const std::string& key, const std::string& label, const std::string& unit) {
  return {{"key", key},      {"label", label},    {"short", key},    {"unit", unit},
          {"group", "test"}, {"description", ""}, {"warn", nullptr}, {"critical", nullptr}};
}

json modalix_run() {
  json metrics = json::array({
      metric_definition("power_current_watts", "SoM power", "W"),
      metric_definition("power_average_watts", "Average power", "W"),
      metric_definition("power_rail_mla_watts", "MLA", "W"),
      metric_definition("power_rail_soc_ddr_vdd_watts", "DDR VDD", "W"),
      metric_definition("cpu_usage_pct", "CPU", "%"),
      metric_definition("lm96163_temp1", "SoM top", "°C"),
      metric_definition("lm96163_temp2", "SoM bottom", "°C"),
      metric_definition("mla_mem_allocated_mb", "MLA memory", "MB"),
      metric_definition("ev74_cma_used_mb", "CMA used", "MB"),
      metric_definition("linux_mem_used_mb", "Linux used", "MB"),
  });
  for (int i = 0; i < 14; ++i) {
    metrics.push_back(
        metric_definition("rtsn_" + std::to_string(i), "Sensor " + std::to_string(i), "°C"));
  }

  const auto sample = [](double power, double mla, double hot_site, json cpu, double mla_mem) {
    json values = {{"power_current_watts", power},
                   {"power_average_watts", 100.0},
                   {"power_rail_mla_watts", mla},
                   {"power_rail_soc_ddr_vdd_watts", 0.5},
                   {"cpu_usage_pct", std::move(cpu)},
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
    return json{{"timestamp", "2026-10-14T10:00:00Z"}, {"values", std::move(values)}};
  };

  return {{"schema", 1},
          {"metadata",
           {{"id", "20261014T100000.000Z-neat-benchmark"},
            {"name", "neat-benchmark-20261014T100000.000Z"},
            {"sample_interval_ms", 2000},
            {"sentinel_version", "0.1.0"}}},
          {"metrics", std::move(metrics)},
          {"samples", json::array({sample(9.0, 3.0, 60.0, 40.0, 100.0),
                                   sample(10.0, 4.0, 52.0, nullptr, 180.0),
                                   sample(11.0, 5.0, 70.0, 80.0, 160.0)})}};
}

void summary_maps_sentinel_metrics() {
  const BenchmarkTelemetry t = summarize_benchmark_telemetry(modalix_run());

  require(t.status == "ok" && t.samples == 3, "a run with samples is ok");
  require(t.trace_id == "20261014T100000.000Z-neat-benchmark", "trace id comes from metadata");
  require(t.sentinel_version == "0.1.0" && t.sample_interval_ms == 2000, "trace metadata");

  const BenchmarkMetric& som = t.power_som_watts;
  require(som.key == "power_current_watts" && som.label == "SoM power" && som.unit == "W",
          "SoM power uses the Sentinel definition");
  require(som.samples == 3, "SoM power samples");
  require_near(som.mean, 10.0, "SoM power mean");
  require_near(som.minimum, 9.0, "SoM power minimum");
  require_near(som.maximum, 11.0, "SoM power maximum");
  require_near(t.power_mla_watts.mean, 4.0, "MLA rail mean");

  require(t.power_rails_watts.size() == 2, "one entry per reported power rail");
  require(t.power_rails_watts[0].key == "power_rail_mla_watts" &&
              t.power_rails_watts[1].key == "power_rail_soc_ddr_vdd_watts",
          "rails follow the Sentinel definition order");
  require_near(t.power_rails_watts[0].mean, t.power_mla_watts.mean, "MLA rail matches the list");

  require(t.die_temperatures_c.size() == 7, "die temperatures cover rtsn_0..6 only");
  require(t.die_temperatures_c[4].label == "Sensor 4", "die labels come from Sentinel");
  require(t.die_temperature_max_c.key == "die_max" && t.die_temperature_max_c.unit == "°C",
          "die max is a derived metric");
  require_near(t.die_temperature_max_c.minimum, 56.0, "die max per sample uses the hottest site");
  require_near(t.die_temperature_max_c.maximum, 70.0, "die max peak");

  require(t.board_temperatures_c.size() == 2 && t.board_temperatures_c[1].label == "SoM bottom",
          "board temperatures are the two SoM sensors");

  require(t.cpu_usage_pct.samples == 2, "null CPU readings are skipped");
  require_near(t.cpu_usage_pct.maximum, 80.0, "CPU maximum");

  const BenchmarkMetric& mla = t.mla_memory_mib;
  require(mla.key == "mla_mem_allocated_mb" && mla.unit == "MiB" && mla.samples == 3,
          "MLA memory uses the Sentinel definition");
  require_near(mla.minimum, 100.0, "MLA memory minimum");
  require_near(mla.maximum, 180.0, "MLA memory maximum");
  require_near(t.cma_memory_mib.maximum, 480.0, "CMA maximum");
  require_near(t.linux_memory_used_mib.mean, 900.0, "Linux memory mean");
}

void summary_without_mla_rail_or_samples() {
  json dvt = modalix_run();
  for (json& sample : dvt["samples"]) {
    sample["values"].erase("power_rail_mla_watts");
  }
  json& metrics = dvt["metrics"];
  for (auto it = metrics.begin(); it != metrics.end(); ++it) {
    if ((*it)["key"] == "power_rail_mla_watts") {
      metrics.erase(it);
      break;
    }
  }
  const BenchmarkTelemetry t = summarize_benchmark_telemetry(dvt);
  require(t.status == "ok" && t.power_mla_watts.samples == 0,
          "a board without an MLA rail reports MLA power as not measured");
  require(t.power_rails_watts.size() == 1, "rails list only the rails the board reports");

  json empty = modalix_run();
  empty["samples"] = json::array();
  const BenchmarkTelemetry none = summarize_benchmark_telemetry(empty);
  require(none.samples == 0 && none.status.find("no samples") != std::string::npos,
          "a run without samples reports why");
  require(none.power_som_watts.samples == 0 && none.mla_memory_mib.samples == 0,
          "a run without samples measures nothing");
}

/// Answers requests on a Unix socket with canned replies, in order, and records each request.
class FakeSentinel {
public:
  explicit FakeSentinel(std::vector<std::pair<int, json>> replies) : replies_(std::move(replies)) {
    char dir_template[] = "/tmp/neat-sentinel-XXXXXX";
    if (::mkdtemp(dir_template) == nullptr) {
      throw std::runtime_error("mkdtemp failed");
    }
    dir_ = dir_template;
    path_ = dir_ + "/api.sock";
    const auto answer = [this](const httplib::Request& request, httplib::Response& response) {
      std::lock_guard<std::mutex> lock(mu_);
      const std::size_t index = requests_.size();
      requests_.push_back(request.method + " " + request.path);
      const auto& [status, body] =
          index < replies_.size() ? replies_[index] : std::pair<int, json>{599, json::object()};
      response.status = status;
      response.set_content(body.dump(), "application/json");
    };
    server_.Get(".*", answer);
    server_.Post(".*", answer);
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

  SentinelClient client() const {
    return SentinelClient(path_);
  }

  std::vector<std::string> requests() {
    std::lock_guard<std::mutex> lock(mu_);
    return requests_;
  }

private:
  std::vector<std::pair<int, json>> replies_;
  std::vector<std::string> requests_;
  std::mutex mu_;
  std::string dir_;
  std::string path_;
  httplib::Server server_;
  std::thread thread_;
};

const json kTrace = {{"schema", 1}, {"trace", {{"id", "trace-1"}}}};

void trace_records_one_run() {
  FakeSentinel sentinel({{200, kTrace}, {200, kTrace}, {200, modalix_run()}});
  SentinelTrace trace("model.tar.gz", sentinel.client());
  const BenchmarkTelemetry t = trace.finish();
  require(t.status == "ok" && t.samples == 3, "a recorded trace is summarized; got " + t.status);
  const std::vector<std::string> expected = {"POST /v1/traces", "POST /v1/traces/trace-1/stop",
                                             "GET /v1/runs/trace-1"};
  require(sentinel.requests() == expected, "the trace must be stopped once, by ID");
}

void trace_without_sentinel_reports_reason() {
  SentinelTrace trace("model.tar.gz", SentinelClient("/tmp/neat-no-such-sentinel.sock"));
  const BenchmarkTelemetry t = trace.finish();
  require(t.status.find("not reachable") != std::string::npos,
          "a missing Sentinel socket reports why; got " + t.status);
  require(t.samples == 0 && t.power_som_watts.samples == 0, "no telemetry without Sentinel");
}

void trace_with_active_trace_reports_conflict() {
  FakeSentinel sentinel(
      {{409, {{"error", "checkpoint 'user' is already recording; stop it before starting"}}}});
  SentinelTrace trace("model.tar.gz", sentinel.client());
  const BenchmarkTelemetry t = trace.finish();
  require(t.status.find("HTTP 409") != std::string::npos &&
              t.status.find("already recording") != std::string::npos,
          "a busy Sentinel reports its own reason; got " + t.status);
  const std::vector<std::string> expected = {"POST /v1/traces"};
  require(sentinel.requests() == expected, "the user's trace must not be stopped");
}

void trace_stops_when_benchmark_throws() {
  FakeSentinel sentinel({{200, kTrace}, {200, kTrace}});
  try {
    SentinelTrace trace("model.tar.gz", sentinel.client());
    throw std::runtime_error("pull timed out");
  } catch (const std::runtime_error&) {
  }
  const std::vector<std::string> expected = {"POST /v1/traces", "POST /v1/traces/trace-1/stop"};
  require(sentinel.requests() == expected, "the trace must be stopped when the benchmark throws");
}

void client_supports_sentinel_without_stop_by_id() {
  FakeSentinel sentinel({{200, kTrace},
                         {404, {{"error", "unknown Sentinel API endpoint"}}},
                         {200, kTrace},
                         {200, modalix_run()}});
  SentinelTrace trace("model.tar.gz", sentinel.client());
  const BenchmarkTelemetry t = trace.finish();
  require(t.status == "ok", "Sentinel 0.1.0 still records the benchmark; got " + t.status);
  require(sentinel.requests()[2] == "POST /v1/traces/stop",
          "Sentinel without stop-by-ID falls back to stopping the active trace");
}

} // namespace

RUN_TEST("unit_benchmark_telemetry_test", ([] {
           std::vector<std::string> failures;
           const auto run_case = [&](std::string name, const std::function<void()>& fn) {
             try {
               fn();
             } catch (const std::exception& e) {
               failures.push_back(std::move(name) + ": " + e.what());
             }
           };

           run_case("summary_maps_sentinel_metrics", summary_maps_sentinel_metrics);
           run_case("summary_without_mla_rail_or_samples", summary_without_mla_rail_or_samples);
           run_case("trace_records_one_run", trace_records_one_run);
           run_case("trace_without_sentinel", trace_without_sentinel_reports_reason);
           run_case("trace_with_active_trace", trace_with_active_trace_reports_conflict);
           run_case("trace_stops_on_throw", trace_stops_when_benchmark_throws);
           run_case("client_without_stop_by_id", client_supports_sentinel_without_stop_by_id);

           std::string message =
               std::to_string(failures.size()) + " benchmark telemetry check(s) failed:";
           for (const auto& failure : failures) {
             message += "\n  - " + failure;
           }
           require(failures.empty(), message);
         }));
