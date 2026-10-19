#ifndef SIMA_NEAT_INTERNAL
#define SIMA_NEAT_INTERNAL 1
#endif

#include "model/internal/BenchmarkTelemetry.h"
#include "model/internal/ModelInternal.h"
#include "pipeline/internal/SentinelClient.h"
#include "test_main.h"
#include "test_utils.h"

#include <httplib.h>
#include <nlohmann/json.hpp>
#include <sys/socket.h>
#include <unistd.h>

#include <atomic>
#include <cmath>
#include <cstdlib>
#include <functional>
#include <mutex>
#include <new>
#include <stdexcept>
#include <string>
#include <thread>
#include <utility>
#include <vector>

// Executable-local allocation faults, following unit_manifest_context_exception_test.
// No production fault hook: fail a real stop-path allocation, then its diagnostic copy.
namespace allocation_fault {
thread_local int stage = 0;
thread_local int failures = 0;
thread_local bool unknown_exception = false;
struct StopFailure final : std::exception {
  const char* what() const noexcept override {
    return "injected Sentinel stop failure with a diagnostic longer than small-string storage";
  }
};
} // namespace allocation_fault

void* operator new(std::size_t bytes) {
  if (allocation_fault::stage == 1) {
    allocation_fault::stage = 2;
    ++allocation_fault::failures;
    if (allocation_fault::unknown_exception) {
      throw 17;
    }
    throw allocation_fault::StopFailure{};
  }
  if (allocation_fault::stage == 2) {
    allocation_fault::stage = 0;
    ++allocation_fault::failures;
    throw std::bad_alloc{};
  }
  if (void* memory = std::malloc(bytes ? bytes : 1)) {
    return memory;
  }
  throw std::bad_alloc{};
}
void* operator new[](std::size_t bytes) {
  return ::operator new(bytes);
}
void* operator new(std::size_t bytes, const std::nothrow_t&) noexcept {
  try {
    return ::operator new(bytes);
  } catch (...) {
    return nullptr;
  }
}
void* operator new[](std::size_t bytes, const std::nothrow_t&) noexcept {
  try {
    return ::operator new[](bytes);
  } catch (...) {
    return nullptr;
  }
}
void operator delete(void* memory) noexcept {
  std::free(memory);
}
void operator delete[](void* memory) noexcept {
  std::free(memory);
}
void operator delete(void* memory, std::size_t) noexcept {
  std::free(memory);
}
void operator delete[](void* memory, std::size_t) noexcept {
  std::free(memory);
}
void operator delete(void* memory, const std::nothrow_t&) noexcept {
  std::free(memory);
}
void operator delete[](void* memory, const std::nothrow_t&) noexcept {
  std::free(memory);
}

namespace {

using nlohmann::json;
using simaai::neat::BenchmarkMetric;
using simaai::neat::BenchmarkTelemetry;
using simaai::neat::internal::BenchmarkTelemetryBoundary;
using simaai::neat::internal::BenchmarkTelemetryWindow;
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

BenchmarkTelemetryBoundary boundary(int seconds) {
  using namespace std::chrono;
  const auto utc = sys_days{year{2026} / October / 14} + hours{10} + std::chrono::seconds{seconds};
  const auto monotonic = steady_clock::time_point{std::chrono::seconds{100 + seconds}};
  return {utc, monotonic, monotonic};
}

BenchmarkTelemetryWindow window() {
  return {boundary(0), boundary(4)};
}

json modalix_run() {
  json metrics = json::array({
      metric_definition("power_current_watts", "SoM power", "W"),
      metric_definition("power_average_watts", "Average power", "W"),
      metric_definition("power_rail_mla_watts", "MLA", "W"),
      metric_definition("power_rail_soc_ddr_vdd_watts", "DDR VDD", "W"),
      metric_definition("cpu_usage_pct", "CPU", "%"),
      metric_definition("cpu_core_0_usage_pct", "CPU core 0", "%"),
      metric_definition("cpu_core_1_usage_pct", "CPU core 1", "%"),
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

  const auto sample = [](const char* timestamp, double power, double mla, double hot_site, json cpu,
                         double mla_mem) {
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
    return json{{"timestamp", timestamp}, {"values", std::move(values)}};
  };

  return {{"schema", 1},
          {"metadata",
           {{"id", "20261014T100000.000Z-neat-benchmark"},
            {"name", "neat-benchmark-20261014T100000.000Z"},
            {"sample_interval_ms", 2000},
            {"sentinel_version", "0.1.0"}}},
          {"metrics", std::move(metrics)},
          {"samples", json::array({sample("2026-10-14T10:00:00Z", 9.0, 3.0, 60.0, 40.0, 100.0),
                                   sample("2026-10-14T10:00:02Z", 10.0, 4.0, 52.0, nullptr, 180.0),
                                   sample("2026-10-14T10:00:04Z", 11.0, 5.0, 70.0, 80.0, 160.0)})}};
}

void summary_maps_sentinel_metrics() {
  const BenchmarkTelemetry t = summarize_benchmark_telemetry(modalix_run(), window());

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
  require(t.cpu_cores_usage_pct.size() == 2 &&
              t.cpu_cores_usage_pct[0].key == "cpu_core_0_usage_pct" &&
              t.cpu_cores_usage_pct[1].label == "CPU core 1",
          "one entry per CPU core, in Sentinel definition order");
  require_near(t.cpu_cores_usage_pct[0].mean, 100.0, "a busy core keeps its own reading");
  require(t.cpu_cores_usage_pct[1].samples == 2, "null core readings are skipped");

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
  const BenchmarkTelemetry t = summarize_benchmark_telemetry(dvt, window());
  require(t.status == "ok" && t.power_mla_watts.samples == 0,
          "a board without an MLA rail reports MLA power as not measured");
  require(t.power_rails_watts.size() == 1, "rails list only the rails the board reports");

  json empty = modalix_run();
  empty["samples"] = json::array();
  const BenchmarkTelemetry none = summarize_benchmark_telemetry(empty, window());
  require(none.samples == 0 && none.status.find("no samples") != std::string::npos,
          "a run without samples reports why");
  require(none.power_som_watts.samples == 0 && none.mla_memory_mib.samples == 0,
          "a run without samples measures nothing");
}

json run_with_power_tails() {
  json run = modalix_run();
  json before = run["samples"].front();
  before["timestamp"] = "2026-10-14T09:59:59.999999999Z";
  before["values"]["power_current_watts"] = 1000.0;
  before["values"]["cpu_core_0_usage_pct"] = 0.0;
  run["samples"].insert(run["samples"].begin(), before);
  json after = before;
  after["timestamp"] = "2026-10-14T10:00:04.000000001+00:00";
  after["values"]["power_current_watts"] = 2000.0;
  run["samples"].push_back(after);
  return run;
}

void summary_selects_only_timestamped_work_window() {
  const BenchmarkTelemetry t = summarize_benchmark_telemetry(run_with_power_tails(), window());
  require(t.status == "ok" && t.samples == 3, "both exact endpoints are included, tails excluded");
  require_near(t.power_som_watts.mean, 10.0, "selected power: (9+10+11)/3");
  require_near(t.power_som_watts.maximum, 11.0, "power peak excludes 1000/2000 W tails");
  require_near(t.cpu_cores_usage_pct[0].mean, 100.0, "CPU uses the same selected samples");

  const BenchmarkTelemetry one =
      summarize_benchmark_telemetry(run_with_power_tails(), {boundary(1), boundary(3)});
  require(one.status == "ok" && one.samples == 1, "a single in-window sample is valid");
  require_near(one.power_som_watts.mean, 10.0, "single-sample summary");
  require(one.cpu_usage_pct.samples == 0, "missing in-window CPU is not borrowed from tails");

  const BenchmarkTelemetry none =
      summarize_benchmark_telemetry(modalix_run(), {boundary(20), boundary(22)});
  require(none.status.find("no samples") != std::string::npos && none.samples == 0,
          "timestamp mismatch is unavailable, not an unbounded summary");
}

void summary_rejects_invalid_timestamps_and_clock_boundaries() {
  for (const json& timestamp : json::array({nullptr, 1234, "2026-10-14T10:00:02",
                                            "2026-02-30T10:00:02Z", "2026-10-14T10:00:02+05:00",
                                            "2026-10-14T10:00:02.xyzZ", "9999-10-14T10:00:02Z"})) {
    json run = modalix_run();
    run["samples"][1]["timestamp"] = timestamp;
    const BenchmarkTelemetry t = summarize_benchmark_telemetry(run, window());
    require(t.status.find("timestamps") != std::string::npos && t.samples == 0 &&
                t.power_som_watts.samples == 0,
            "unusable timestamps invalidate telemetry, not just one metric");
  }
  json missing = modalix_run();
  missing["samples"][0].erase("timestamp");
  require(summarize_benchmark_telemetry(missing, window()).samples == 0,
          "a missing timestamp must not be assigned a guessed position");
  json duplicate = modalix_run();
  duplicate["samples"][1]["timestamp"] = duplicate["samples"][0]["timestamp"];
  require(summarize_benchmark_telemetry(duplicate, window()).status.find("timestamps") !=
              std::string::npos,
          "duplicate timestamps do not double-weight one sample");
  json reversed = modalix_run();
  reversed["samples"][1]["timestamp"] = "2026-10-14T09:59:59Z";
  require(summarize_benchmark_telemetry(reversed, window()).samples == 0,
          "nonmonotonic provider samples are unavailable");

  for (const int wall_step_seconds : {-1, 1}) {
    auto changed = window();
    changed.end.utc += std::chrono::seconds{wall_step_seconds};
    const BenchmarkTelemetry t = summarize_benchmark_telemetry(modalix_run(), changed);
    require(t.status.find("clocks") != std::string::npos && t.samples == 0,
            "both forward and backward detected clock steps invalidate telemetry");
  }
  auto invalid = window();
  std::swap(invalid.begin, invalid.end);
  require(summarize_benchmark_telemetry(modalix_run(), invalid).samples == 0,
          "reversed measurement boundaries are unavailable");
  auto bracketed = window();
  bracketed.begin.before -= std::chrono::nanoseconds{20};
  bracketed.begin.after += std::chrono::nanoseconds{30};
  bracketed.end.before -= std::chrono::nanoseconds{40};
  bracketed.end.after += std::chrono::nanoseconds{50};
  require(summarize_benchmark_telemetry(modalix_run(), bracketed).samples == 3,
          "paired capture uncertainty does not require an epoch-offset guess");
}

void local_clock_captures_remain_compatible() {
  // Exercise the real local UTC/steady pairing as well as the deterministic
  // fake-clock oracles. This is test-only sampling, never warmed inference work.
  const json empty = {{"samples", json::array()}};
  for (int i = 0; i < 64; ++i) {
    const auto begin = BenchmarkTelemetryBoundary::now();
    if (i == 0) {
      std::this_thread::sleep_for(std::chrono::milliseconds{10});
    }
    const auto end = BenchmarkTelemetryBoundary::now();
    const auto telemetry = summarize_benchmark_telemetry(empty, {begin, end});
    require(telemetry.status.find("no samples") != std::string::npos,
            "local short/sleep UTC and steady captures must not spuriously reject the clock: " +
                telemetry.status);
  }
}

/// Answers requests on a Unix socket with canned replies, in order, and records each request.
class FakeSentinel {
public:
  explicit FakeSentinel(std::vector<std::pair<int, json>> replies,
                        std::function<void(std::size_t)> before_reply = {})
      : replies_(std::move(replies)), before_reply_(std::move(before_reply)) {
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
      if (before_reply_) {
        before_reply_(index);
      }
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
  std::function<void(std::size_t)> before_reply_;
  std::vector<std::string> requests_;
  std::mutex mu_;
  std::string dir_;
  std::string path_;
  httplib::Server server_;
  std::thread thread_;
};

const json kTrace = {{"schema", 1}, {"trace", {{"id", "trace-1"}}}};

simaai::neat::MeasureReport completed_measurement() {
  simaai::neat::MeasureReport measured;
  measured.elapsed_s = 4.0;
  measured.outputs = 8;
  measured.options.logical_batch_size = 1;
  measured.counters.inputs_pushed = 8;
  measured.counters.outputs_ready = 8;
  measured.counters.outputs_pulled = 8;
  measured.latency_samples_collected = true;
  measured.end_to_end.count = 8;
  measured.end_to_end.avg_ms = 3.0;
  measured.end_to_end.p50_ms = 2.0;
  measured.end_to_end.p95_ms = 5.0;
  measured.end_to_end.p99_ms = 6.0;
  measured.throughput_batches_per_s = 2.0;
  measured.throughput_inferences_per_s = 2.0;
  return measured;
}

void delayed_stop_cannot_extend_window_or_contaminate_report() {
  // A fake clock makes a ten-second stop delay deterministic: the HTTP handler
  // advances it only after all model outputs and measurement are complete.
  std::atomic<int> clock_seconds{0};
  std::atomic<bool> outputs_complete{false};
  std::atomic<bool> timer_stopped{false};
  FakeSentinel sentinel({{200, kTrace}, {200, kTrace}, {200, run_with_power_tails()}},
                        [&](std::size_t request) {
                          if (request == 1) {
                            require(outputs_complete.load() && timer_stopped.load(),
                                    "Sentinel stop must follow output completion and timer stop");
                            clock_seconds.store(14);
                          }
                        });
  SentinelTrace trace("model.tar.gz", sentinel.client());
  const auto begin = boundary(clock_seconds.load());
  clock_seconds.store(4);
  outputs_complete.store(true);
  const auto end = boundary(clock_seconds.load());
  const auto measured = completed_measurement();
  timer_stopped.store(true);
  const BenchmarkTelemetry telemetry = trace.finish({begin, end});
  require(clock_seconds.load() == 14, "the deterministic stop actually advanced fake time");
  const auto report = simaai::neat::internal::build_benchmark_report_from_measurements(
      measured, measured, 8, 8, 8, telemetry);
  require_near(report.throughput_duration_ms, 4000.0, "model timer excludes ten-second drain");
  require_near(report.fps, 2.0, "FPS comes from eight completed outputs in four seconds");
  require_near(report.avg_power_watts, 10.0, "full report excludes power tails");
  require_near(report.peak_power_watts, 11.0, "full report peak excludes power tails");
  require_near(report.energy_joules, 40.0, "energy: 10 W * eight inputs / two inputs per second");
  require(report.telemetry.samples == 3, "full report keeps only measured-window samples");
}

void trace_stop_failure_preserves_model_report_without_power() {
  FakeSentinel sentinel({{200, kTrace}, {500, {{"error", "stop failed"}}}});
  SentinelTrace trace("model.tar.gz", sentinel.client());
  const auto telemetry = trace.finish(window());
  require(telemetry.status.find("stop failed") != std::string::npos && telemetry.samples == 0,
          "stop failure reports unavailable telemetry");
  const auto measured = completed_measurement();
  const auto report = simaai::neat::internal::build_benchmark_report_from_measurements(
      measured, measured, 8, 8, 8, telemetry);
  require_near(report.fps, 2.0, "optional telemetry failure cannot hide model FPS");
  require_near(report.throughput_duration_ms, 4000.0, "optional failure preserves model duration");
  require(report.telemetry.status != "ok" && report.avg_power_watts == 0.0 &&
              report.energy_joules == 0.0,
          "unavailable power is not a zero-energy success");
  require(sentinel.requests().size() == 2, "failed stop must not summarize an active run");
}

void cleanup_failure_survives_diagnostic_allocation_failure() {
  for (const bool unknown : {false, true}) {
    FakeSentinel sentinel({{200, kTrace}});
    SentinelTrace trace("model.tar.gz", sentinel.client());
    allocation_fault::failures = 0;
    allocation_fault::unknown_exception = unknown;
    allocation_fault::stage = 1;
    const auto telemetry = trace.finish(window());
    require(allocation_fault::stage == 0 && allocation_fault::failures == 2,
            "real stop allocation and subsequent failure-string allocation both failed");
    require(telemetry.status.find("diagnostic unavailable") != std::string::npos &&
                telemetry.samples == 0,
            "a nonallocating failure flag prevents failed cleanup becoming success");
  }
  FakeSentinel sentinel({{200, kTrace}});
  allocation_fault::failures = 0;
  try {
    SentinelTrace trace("model.tar.gz", sentinel.client());
    allocation_fault::unknown_exception = true;
    allocation_fault::stage = 1;
    throw 7;
  } catch (int original) {
    require(original == 7, "nonthrowing cleanup preserves the original unwinding exception");
  }
  require(allocation_fault::stage == 0 && allocation_fault::failures == 2,
          "destructor cleanup survives both failures during unwinding");
}

void trace_records_one_run() {
  FakeSentinel sentinel({{200, kTrace}, {200, kTrace}, {200, modalix_run()}});
  SentinelTrace trace("model.tar.gz", sentinel.client());
  const BenchmarkTelemetry t = trace.finish(window());
  require(t.status == "ok" && t.samples == 3, "a recorded trace is summarized; got " + t.status);
  const std::vector<std::string> expected = {"POST /v1/traces", "POST /v1/traces/trace-1/stop",
                                             "GET /v1/runs/trace-1"};
  require(sentinel.requests() == expected, "the trace must be stopped once, by ID");
}

void trace_without_sentinel_reports_reason() {
  SentinelTrace trace("model.tar.gz", SentinelClient("/tmp/neat-no-such-sentinel.sock"));
  const BenchmarkTelemetry t = trace.finish(window());
  require(t.status.find("not reachable") != std::string::npos,
          "a missing Sentinel socket reports why; got " + t.status);
  require(t.samples == 0 && t.power_som_watts.samples == 0, "no telemetry without Sentinel");
}

void trace_with_active_trace_reports_conflict() {
  FakeSentinel sentinel(
      {{409, {{"error", "checkpoint 'user' is already recording; stop it before starting"}}}});
  SentinelTrace trace("model.tar.gz", sentinel.client());
  const BenchmarkTelemetry t = trace.finish(window());
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
  const BenchmarkTelemetry t = trace.finish(window());
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
           run_case("timestamp_work_window", summary_selects_only_timestamped_work_window);
           run_case("invalid_timestamp_or_clock",
                    summary_rejects_invalid_timestamps_and_clock_boundaries);
           run_case("local_clock_captures", local_clock_captures_remain_compatible);
           run_case("delayed_stop_power_tail",
                    delayed_stop_cannot_extend_window_or_contaminate_report);
           run_case("stop_failure_model_report",
                    trace_stop_failure_preserves_model_report_without_power);
           run_case("cleanup_diagnostic_allocation_failure",
                    cleanup_failure_survives_diagnostic_allocation_failure);
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
