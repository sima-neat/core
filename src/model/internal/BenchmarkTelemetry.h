/**
 * @file
 * @brief Sentinel telemetry for `Model::benchmark()` (not part of the public API).
 */
#pragma once
#ifndef SIMA_NEAT_INTERNAL
#error "Internal header. Not part of the public API."
#endif

#include "model/Model.h"
#include "pipeline/internal/SentinelClient.h"

#include <nlohmann/json.hpp>

#include <chrono>
#include <string>

namespace simaai::neat::internal {

/// Sentinel's local Unix-socket daemon timestamps samples with UTC wall time.
/// Bracket that clock read with steady time so a detected wall-clock step makes
/// telemetry unavailable rather than silently selecting a different interval.
struct BenchmarkTelemetryBoundary {
  std::chrono::system_clock::time_point utc;
  std::chrono::steady_clock::time_point before;
  std::chrono::steady_clock::time_point after;

  static BenchmarkTelemetryBoundary now() noexcept;
};

/// Inclusive sample-timestamp selection, contained within the model timer:
/// begin after measurement starts; end after every output completes and before
/// measurement stops. Provider interval-averaged values are not interpolated.
struct BenchmarkTelemetryWindow {
  BenchmarkTelemetryBoundary begin;
  BenchmarkTelemetryBoundary end;
};

/**
 * @brief Records a Sentinel trace over the benchmark throughput window.
 *
 * The trace starts at construction. `finish(window)` stops it and summarizes only samples
 * timestamped inside the explicit measured-work window, excluding setup and stop tails. Service
 * failures become `BenchmarkTelemetry::status`, so a missing or busy Sentinel does not fail the
 * benchmark. Construction and result creation can still throw on resource exhaustion. The
 * nonthrowing destructor stops a trace that is still recording, including during unwinding.
 */
class SentinelTrace {
public:
  explicit SentinelTrace(const std::string& note, pipeline_internal::SentinelClient client =
                                                      pipeline_internal::SentinelClient());
  ~SentinelTrace();
  SentinelTrace(const SentinelTrace&) = delete;
  SentinelTrace& operator=(const SentinelTrace&) = delete;

  /// Stop the trace after the model timer has stopped, then select the captured window.
  BenchmarkTelemetry finish(const BenchmarkTelemetryWindow& window);

private:
  void stop() noexcept;
  void remember_failure(const char* message) noexcept;

  pipeline_internal::SentinelClient client_;
  std::string trace_id_;
  std::string failure_;
  bool failed_ = false;
  bool recording_ = false;
};

/// Summarize a saved Sentinel run. Invalid timestamps/clock boundaries produce
/// unavailable telemetry; they never fall back to an unbounded trace summary.
BenchmarkTelemetry summarize_benchmark_telemetry(const nlohmann::json& run,
                                                 const BenchmarkTelemetryWindow& window);

} // namespace simaai::neat::internal
