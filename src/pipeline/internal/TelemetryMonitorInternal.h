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

#include <string>

namespace simaai::neat::internal {

/// Builds a `TelemetryMonitor` that reads Sentinel at a chosen socket, for tests.
struct TelemetryMonitorAccess {
  static TelemetryMonitor make(std::string sentinel_socket_path);
};

/// Telemetry with every measurement the monitor summarizes, labelled from Sentinel's metric
/// definitions, and no samples.
Telemetry telemetry_layout(const nlohmann::json& metrics);

/// Count one Sentinel sample (`timestamp` and `values`) into `telemetry`.
void count_sample(Telemetry& telemetry, const nlohmann::json& sample);

} // namespace simaai::neat::internal
