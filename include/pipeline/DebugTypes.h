/**
 * @file
 * @ingroup pipeline
 * @brief Developer-facing debug types used by introspective dump helpers.
 *
 * The `simaai::neat::debug` namespace exposes types used by debug-mode dump
 * APIs that snapshot or stream individual pipeline outputs. `DebugOptions`
 * tunes the behaviour (timeout, strictness, memory placement); `DebugOutput`
 * carries the result of a one-shot dump; `DebugStream` returns a callable
 * iterator over outputs from a long-running pipeline. Tag types are used as
 * overload selectors for the dump entry points.
 *
 * @see builder::OutputSpec for the spec/observed contract types.
 */
#pragma once

#include "builder/OutputSpec.h"
#include "pipeline/TensorCore.h"

#include <functional>
#include <memory>
#include <optional>
#include <string>
#include <vector>

namespace simaai::neat::debug {

/**
 * @brief Tunables for debug-mode pipeline dumps.
 *
 * @ingroup pipeline
 */
struct DebugOptions {
  int timeout_ms = 10000;          ///< Per-output wait timeout, in milliseconds.
  bool strict = false;             ///< If true, mismatches against expected specs throw.
  bool force_system_memory = true; ///< If true, request system memory for outputs (eases dumping).
};

/**
 * @brief One captured pipeline output (single-shot dump result).
 *
 * Carries the expected spec, the spec actually observed, the caps string of the
 * payload, and either a `Tensor` (if the payload is tensorizable) or a raw
 * byte buffer fallback. `warnings` holds any non-fatal issues collected during
 * capture.
 *
 * @ingroup pipeline
 */
struct DebugOutput {
  OutputSpec expected;               ///< Expected output spec.
  OutputSpec observed;               ///< Spec actually observed at runtime.
  std::string caps_string;           ///< Payload caps string.
  bool tensorizable = false;         ///< True iff the payload could be wrapped as a Tensor.
  bool unknown = false;              ///< True iff the payload type was not recognised.
  std::vector<std::string> warnings; ///< Non-fatal issues collected during capture.
  std::optional<simaai::neat::Tensor> tensor; ///< Owning tensor (when available).
  std::vector<uint8_t> bytes;                 ///< Raw bytes for non-tensorizable outputs.
};

/**
 * @brief Streaming iterator over debug-mode pipeline outputs.
 *
 * `next(timeout_ms)` returns the next captured output (or `std::nullopt` on
 * timeout / end-of-stream). `close()` tears the underlying stream down. The
 * `state` shared pointer keeps any backing pipeline alive for as long as the
 * stream is held. The boolean conversion reports whether the stream is
 * usable (i.e., whether `next` is bound).
 *
 * @ingroup pipeline
 */
struct DebugStream {
  std::function<std::optional<DebugOutput>(int)> next; ///< Pull next output; arg is timeout_ms.
  std::function<void()> close;                         ///< Tear down the underlying stream.
  OutputSpec expected;                                 ///< Expected output spec.
  OutputSpec observed;                                 ///< Spec observed at runtime.
  std::string caps_string;                             ///< Payload caps string.
  bool tensorizable = false;                           ///< True iff outputs are tensorizable.
  bool unknown = false;                                ///< True iff payload type was unrecognised.
  std::shared_ptr<void> state; ///< Keeps the underlying pipeline/stream alive.

  /// @brief True iff the stream is bound and usable.
  explicit operator bool() const {
    return static_cast<bool>(next);
  }
};

/// @brief Tag type selecting the single-output dump overload.
struct OutputTag {};
/// @brief Tag type selecting the streaming-output dump overload.
struct StreamTag {};

/// @brief Tag instance selecting single-output dump overload.
inline constexpr OutputTag output{};
/// @brief Tag instance selecting streaming-output dump overload.
inline constexpr StreamTag stream{};

} // namespace simaai::neat::debug
