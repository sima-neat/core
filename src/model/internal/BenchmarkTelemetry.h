/**
 * @file
 * @brief Sentinel telemetry for `Model::benchmark()` (not part of the public API).
 */
#pragma once
#ifndef SIMA_NEAT_INTERNAL
#error "Internal header. Not part of the public API."
#endif

#include "model/Model.h"

#include <nlohmann/json.hpp>

#include <string>

namespace simaai::neat::internal {

inline constexpr const char* kSentinelSocketPath = "/run/simaai-sentinel/api.sock";

/**
 * @brief Records a Sentinel trace around the benchmark throughput window.
 *
 * Every step is best effort: the first failure is kept as the telemetry status and later steps
 * are skipped, so a missing or busy Sentinel never fails the benchmark. The destructor stops a
 * trace that is still recording, which covers exceptions thrown during the window.
 */
class BenchmarkTelemetryRecorder {
public:
  explicit BenchmarkTelemetryRecorder(std::string note,
                                      std::string socket_path = kSentinelSocketPath);
  ~BenchmarkTelemetryRecorder();
  BenchmarkTelemetryRecorder(const BenchmarkTelemetryRecorder&) = delete;
  BenchmarkTelemetryRecorder& operator=(const BenchmarkTelemetryRecorder&) = delete;

  void start();                 ///< Start the trace.
  void stop() noexcept;         ///< Stop the trace; safe to call more than once.
  BenchmarkTelemetry collect(); ///< Stop the trace and summarize its samples.

private:
  template <typename Step> void attempt(Step&& step);

  std::string note_;
  std::string socket_path_;
  std::string failure_;
  std::string trace_id_;
  bool recording_ = false;
};

/// Summarize a saved Sentinel run (`GET /v1/runs/{id}`).
BenchmarkTelemetry summarize_benchmark_telemetry(const nlohmann::json& run);

} // namespace simaai::neat::internal
