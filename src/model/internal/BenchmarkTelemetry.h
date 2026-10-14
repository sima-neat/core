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

#include <string>

namespace simaai::neat::internal {

/**
 * @brief Records a Sentinel trace over the benchmark throughput window.
 *
 * The trace starts at construction. `finish()` stops it and summarizes its samples. Nothing
 * throws: the first failure becomes `BenchmarkTelemetry::status`, so a missing or busy Sentinel
 * never fails the benchmark. The destructor stops a trace that is still recording, which covers
 * exceptions thrown during the window.
 */
class SentinelTrace {
public:
  explicit SentinelTrace(const std::string& note, pipeline_internal::SentinelClient client =
                                                      pipeline_internal::SentinelClient());
  ~SentinelTrace();
  SentinelTrace(const SentinelTrace&) = delete;
  SentinelTrace& operator=(const SentinelTrace&) = delete;

  /// Stop the trace. Call it as soon as the measured window ends; `finish()` stops it otherwise.
  void stop() noexcept;

  /// Stop the trace and summarize the samples it recorded.
  BenchmarkTelemetry finish();

private:
  pipeline_internal::SentinelClient client_;
  std::string trace_id_;
  std::string failure_;
  bool recording_ = false;
};

/// Summarize a saved Sentinel run (`SentinelClient::read_run`).
BenchmarkTelemetry summarize_benchmark_telemetry(const nlohmann::json& run);

} // namespace simaai::neat::internal
