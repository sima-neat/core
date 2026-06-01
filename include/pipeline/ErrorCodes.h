/**
 * @file
 * @ingroup diagnostics
 * @brief Canonical framework error code constants used in `GraphReport::error_code` and
 * `PullError::code`.
 *
 * Error codes follow a `domain.reason` taxonomy: the domain identifies the failure category
 * (`misconfig`, `build`, `runtime`, `io`, `infra`); the reason is a snake_case token. This
 * lets triage tools, dashboards, and CI bucket failures by `domain` while keeping per-failure
 * granularity. Use the constants in this file rather than hard-coding strings, so the codes
 * stay consistent across the framework.
 */
#pragma once

#include <string_view>

namespace simaai::neat::error_codes {

// Naming rule:
//   - domain.reason
//   - lowercase tokens
//   - snake_case inside each token
// Example: misconfig.input_shape

// ── Misconfiguration classes ──────────────────────────────────────────────────────────────
/// Pipeline graph geometry mismatch (e.g., wrong number of sinks, cycles, missing terminal Output).
inline constexpr const char* kPipelineShape = "misconfig.pipeline_shape";
/// Caps/format negotiation failed between adjacent elements (resolution, format, framerate,
/// layout).
inline constexpr const char* kCaps = "misconfig.caps";
/// Input tensor shape violates the model's contract (rank, spatial dims, channel count).
inline constexpr const char* kInputShape = "misconfig.input_shape";
/// Framework/runtime plugin ABI mismatch, usually from mixed PyNEAT and runtime artifacts.
inline constexpr const char* kRuntimeAbiMismatch = "misconfig.runtime_abi_mismatch";

// ── Build / runtime classes ──────────────────────────────────────────────────────────────
/// `gst_parse_launch` failed to parse the generated GStreamer pipeline string.
inline constexpr const char* kParseLaunch = "build.parse_launch";
/// `Run::pull()` encountered a runtime-side error (downstream EOS, bus error, or appsink failure).
inline constexpr const char* kRuntimePull = "runtime.pull";

// ── I/O classes ──────────────────────────────────────────────────────────────────────────
/// JSON or config parsing error (typically from the MPK contract or a per-stage config).
inline constexpr const char* kIoParse = "io.parse";
/// Failed to open a file or device path (file missing, permission denied, kernel device absent).
inline constexpr const char* kIoOpen = "io.open";

// ── Infra classes ────────────────────────────────────────────────────────────────────────
/**
 * @brief Dispatcher resource unavailable.
 *
 * Returned when an MLA/EV74/A65 dispatcher can't be acquired. Common causes: EV74 firmware
 * not loaded (`/dev/rpmsg*` missing), MLA license missing, hardware fault. The framework
 * deliberately does not fall back to CPU — see "no host fallback" in §16.
 */
inline constexpr const char* kDispatcherUnavailable = "infra.dispatcher_unavailable";
/// Legacy spelling kept for compatibility with older reports. Prefer `kDispatcherUnavailable`.
inline constexpr const char* kDispatcherUnavailableLegacy = "DispatcherUnavailable";

/// Returns `true` if `code` matches either the canonical or legacy dispatcher-unavailable code.
inline bool is_dispatcher_unavailable(std::string_view code) {
  return code == kDispatcherUnavailable || code == kDispatcherUnavailableLegacy;
}

} // namespace simaai::neat::error_codes
