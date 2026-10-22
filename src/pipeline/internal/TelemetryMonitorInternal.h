/**
 * @file
 * @brief TelemetryMonitor internals exposed for tests (not part of the public API).
 */
#pragma once
#ifndef SIMA_NEAT_INTERNAL
#error "Internal header. Not part of the public API."
#endif

#include "pipeline/TelemetryMonitor.h"

#include <nlohmann/json.hpp>

#include <chrono>
#include <optional>
#include <string>

namespace simaai::neat::internal {

/// Brackets a local UTC read with steady time; benchmark-only attribution rejects
/// clock discontinuities instead of guessing remote clock offsets.
struct BenchmarkTelemetryBoundary {
  std::chrono::system_clock::time_point utc;
  std::chrono::steady_clock::time_point before;
  std::chrono::steady_clock::time_point after;
  static BenchmarkTelemetryBoundary now() noexcept;
};

bool valid_benchmark_telemetry_window(const BenchmarkTelemetryBoundary& begin,
                                      const BenchmarkTelemetryBoundary& end) noexcept;
std::optional<std::chrono::system_clock::time_point>
benchmark_sample_timestamp(const nlohmann::json& sample);

/// Builds a `TelemetryMonitor` that reads Sentinel at a chosen socket, for tests.
struct TelemetryMonitorAccess {
  static TelemetryMonitor make(std::string sentinel_socket_path);

  // Cold prepare may allocate/connect/start a waiting worker. The benchmark owns
  // this monitor exclusively. Arm and seal do not lock, allocate, join or perform
  // I/O. Arm follows measurement start; seal precedes measurement stop; ordinary
  // stop() then joins/final-reads. Public monitors retain cache-order semantics.
  static void prepare_benchmark(TelemetryMonitor& monitor);
  static void arm_benchmark(TelemetryMonitor& monitor) noexcept;
  static void seal_benchmark(TelemetryMonitor& monitor) noexcept;
};

/// Telemetry with every measurement the monitor summarizes, labelled from Sentinel's metric
/// definitions, and no samples.
Telemetry telemetry_layout(const nlohmann::json& metrics);

/// Count one Sentinel sample (`timestamp` and `values`) into `telemetry`.
void count_sample(Telemetry& telemetry, const nlohmann::json& sample);

} // namespace simaai::neat::internal
