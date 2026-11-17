#pragma once

#include "pipeline/PowerTelemetry.h"

#include <string>

namespace simaai::neat::internal {

#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wdeprecated-declarations"
/// Builds a `PowerMonitor` that reads Sentinel at a chosen socket, for tests.
struct PowerMonitorAccess {
  static PowerMonitor make(PowerMonitorOptions options, std::string sentinel_socket_path);
};
#pragma GCC diagnostic pop

} // namespace simaai::neat::internal
