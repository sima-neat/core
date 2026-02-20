/**
 * @file
 * @ingroup diagnostics
 * @brief Canonical framework error codes used by SessionReport and PullError.
 */
#pragma once

#include <string_view>

namespace simaai::neat::error_codes {

// Naming rule:
//   - domain.reason
//   - lowercase tokens
//   - snake_case inside each token
// Example: misconfig.input_shape

// Misconfiguration classes
inline constexpr const char* kPipelineShape = "misconfig.pipeline_shape";
inline constexpr const char* kCaps = "misconfig.caps";
inline constexpr const char* kInputShape = "misconfig.input_shape";

// Build/runtime classes
inline constexpr const char* kParseLaunch = "build.parse_launch";
inline constexpr const char* kRuntimePull = "runtime.pull";

// IO classes
inline constexpr const char* kIoParse = "io.parse";
inline constexpr const char* kIoOpen = "io.open";

// Infra classes
inline constexpr const char* kDispatcherUnavailable = "infra.dispatcher_unavailable";
// Legacy dispatcher value kept for compatibility with older reports.
inline constexpr const char* kDispatcherUnavailableLegacy = "DispatcherUnavailable";

inline bool is_dispatcher_unavailable(std::string_view code) {
  return code == kDispatcherUnavailable || code == kDispatcherUnavailableLegacy;
}

} // namespace simaai::neat::error_codes
