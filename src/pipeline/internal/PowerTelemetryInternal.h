#pragma once

#include "pipeline/PowerTelemetry.h"

#include <string>

namespace simaai::neat::internal {

/// Builds a `PowerMonitor` that reads Sentinel at a chosen socket, for tests.
struct PowerMonitorAccess {
  static PowerMonitor make(PowerMonitorOptions options, std::string sentinel_socket_path);
};

} // namespace simaai::neat::internal
