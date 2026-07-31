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
/// Input tensor violates the expected contract (rank, shape, layout, or data type).
inline constexpr const char* kInputShape = "misconfig.input_shape";
/// Framework/runtime plugin ABI mismatch, usually from mixed PyNEAT and runtime artifacts.
inline constexpr const char* kRuntimeAbiMismatch = "misconfig.runtime_abi_mismatch";
/// A custom graph fragment contains an element that cannot be attributed to a stable Node name.
inline constexpr const char* kGraphElementName = "misconfig.graph_element_name";
/// Adjacent media stages require incompatible caps.
inline constexpr const char* kMediaCaps = "misconfig.media_caps";
/// Adjacent media stages require incompatible formats.
inline constexpr const char* kMediaFormat = "misconfig.media_format";
/// An image is larger than the configured preprocessing input capacity.
inline constexpr const char* kInputCapacity = "misconfig.input_capacity";
/// A tensor contract omitted its dtype/format.
inline constexpr const char* kTensorDtypeMissing = "misconfig.tensor_dtype_missing";
/// A user-facing option is outside the range supported by the active input contract.
inline constexpr const char* kOptionOutOfRange = "misconfig.option_out_of_range";

// ── Build / runtime classes ──────────────────────────────────────────────────────────────
/// `gst_parse_launch` failed to parse the generated GStreamer pipeline string.
inline constexpr const char* kParseLaunch = "build.parse_launch";
/// A custom GStreamer launch fragment contains invalid syntax.
inline constexpr const char* kPipelineSyntax = "build.pipeline_syntax";
/// A required GStreamer element/plugin is not installed.
inline constexpr const char* kPluginMissing = "build.plugin_missing";
/// A GStreamer element property is unknown or invalid.
inline constexpr const char* kPropertyInvalid = "build.property_invalid";
/// `Run::pull()` encountered a runtime-side error (downstream EOS, bus error, or appsink failure).
inline constexpr const char* kRuntimePull = "runtime.pull";
/// A pipeline element stopped processing for a reason that has no more-specific classification.
inline constexpr const char* kRuntimeElementFailed = "runtime.element_failed";
/// No output arrived before the configured wait expired.
inline constexpr const char* kOutputTimeout = "runtime.output_timeout";
/// The pipeline reached EOS before a required output was produced.
inline constexpr const char* kUnexpectedEos = "runtime.unexpected_eos";

// ── I/O classes ──────────────────────────────────────────────────────────────────────────
/// JSON or config parsing error (typically from the MPK contract or a per-stage config).
inline constexpr const char* kIoParse = "io.parse";
/// Failed to open a file or device path (file missing, permission denied, kernel device absent).
inline constexpr const char* kIoOpen = "io.open";
/// An input file does not exist.
inline constexpr const char* kFileNotFound = "io.file_not_found";
/// A file/device exists but cannot be opened with the required permissions.
inline constexpr const char* kPermissionDenied = "io.permission_denied";
/// An RTSP source could not be contacted or opened.
inline constexpr const char* kRtspConnectionFailed = "io.rtsp_connection_failed";
/// The requested camera name is not available.
inline constexpr const char* kCameraNotFound = "io.camera_not_found";
/// A model archive path does not exist.
inline constexpr const char* kModelNotFound = "io.model_not_found";
/// An input source ended normally and has no more data.
inline constexpr const char* kSourceEnded = "io.source_ended";

// ── Codec classes ─────────────────────────────────────────────────────────────────────────
/// The input did not contain a valid H.264 access unit before EOS.
inline constexpr const char* kInvalidH264Stream = "codec.invalid_h264_stream";
/// A decoder failed after accepting the encoded stream.
inline constexpr const char* kDecodeFailed = "codec.decode_failed";
/// An encoder failed while producing an encoded stream.
inline constexpr const char* kEncodeFailed = "codec.encode_failed";

// ── Resource classes ──────────────────────────────────────────────────────────────────────
/// A required memory allocation failed without identifying a device-specific allocator.
inline constexpr const char* kMemoryAllocationFailed = "resource.memory_allocation_failed";
/// Contiguous device DMA/CMA memory is exhausted.
inline constexpr const char* kDeviceMemoryExhausted = "resource.device_memory_exhausted";
/// A plugin could not acquire an output buffer from its pool.
inline constexpr const char* kOutputPoolExhausted = "resource.output_pool_exhausted";
/// A supplied or allocated buffer is smaller than the required payload.
inline constexpr const char* kBufferTooSmall = "resource.buffer_too_small";
/// A file/device write failed because its storage is full.
inline constexpr const char* kDiskFull = "resource.disk_full";

// ── Infra classes ────────────────────────────────────────────────────────────────────────
/**
 * @brief Dispatcher resource unavailable.
 *
 * Returned when an MLA/EV74/A65 dispatcher can't be acquired. Common causes: EV74 firmware
 * not loaded (`/dev/rpmsg*` missing), MLA license missing, hardware fault. The framework
 * deliberately does not fall back to CPU — see "no host fallback" in §16.
 */
inline constexpr const char* kDispatcherUnavailable = "infra.dispatcher_unavailable";
/// The dispatcher was available, but accelerator execution failed.
inline constexpr const char* kAcceleratorExecutionFailed = "infra.accelerator_execution_failed";
/// Legacy spelling kept for compatibility with older reports. Prefer `kDispatcherUnavailable`.
inline constexpr const char* kDispatcherUnavailableLegacy = "DispatcherUnavailable";

// ── Internal classes ──────────────────────────────────────────────────────────────────────
/// A Neat plugin failed unexpectedly without a user-actionable classification.
inline constexpr const char* kInternalPluginFailure = "internal.plugin_failure";

/// Returns `true` if `code` matches either the canonical or legacy dispatcher-unavailable code.
inline bool is_dispatcher_unavailable(std::string_view code) {
  return code == kDispatcherUnavailable || code == kDispatcherUnavailableLegacy;
}

} // namespace simaai::neat::error_codes
