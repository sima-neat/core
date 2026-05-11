/**
 * @file
 * @ingroup internal_sima
 * @brief **Framework internal — reach-through tier.** Defensive guard at the kernel-200
 *        envelope boundary.
 *
 * Kernel-200 style envelopes carry a buffer of compiled-time `max_w / max_h / max_stride /
 * allocated_bytes` and a runtime `actual_w / actual_h / actual_stride / required_bytes`.
 * `validate_kernel200_envelope` checks the actual values fit inside the declared envelope
 * limits before submission, returning a structured violation when they don't.
 */
#pragma once
#ifndef SIMA_NEAT_INTERNAL
#error "Internal header. Not part of the public API."
#endif

#include <cstddef>
#include <string>

namespace simaai::neat::pipeline_internal::sima {

/**
 * @brief Compile-time envelope limits declared by a kernel-200 stage.
 *
 * Captures the upper-bound geometry the buffer was allocated for. Any actual frame at runtime
 * must fit within these limits or the guard rejects it.
 */
struct Kernel200EnvelopeLimits {
  int max_w = 0;             ///< Maximum width (pixels) the envelope was sized for.
  int max_h = 0;             ///< Maximum height (rows) the envelope was sized for.
  int max_stride = 0;        ///< Maximum row stride (bytes) the envelope was sized for.
  std::size_t allocated_bytes = 0;  ///< Total bytes allocated for the envelope buffer.
};

/**
 * @brief Runtime actuals to validate against the declared envelope.
 *
 * Captures the actual geometry of the frame being submitted; compared against
 * `Kernel200EnvelopeLimits` by `validate_kernel200_envelope`.
 */
struct Kernel200EnvelopeActual {
  int actual_w = 0;          ///< Actual width of the frame being submitted.
  int actual_h = 0;          ///< Actual height of the frame being submitted.
  int actual_stride = 0;     ///< Actual row stride of the frame being submitted.
  std::size_t required_bytes = 0;  ///< Bytes the actual frame requires.
};

/// Structured violation returned when actuals don't fit the envelope limits.
struct Kernel200EnvelopeViolation {
  std::string code;     ///< Short stable code identifying the failure (e.g., `"width_overflow"`).
  std::string message;  ///< Human-readable description of the violation.
};

/**
 * @brief Validate that runtime actuals fit within the declared envelope limits.
 *
 * @param limits     Compile-time envelope upper bounds.
 * @param actual     Runtime actual geometry/size.
 * @param violation  Optional out-parameter; populated with details when the check fails.
 * @return `true` when actuals are within limits; `false` otherwise (and `violation` is filled).
 */
bool validate_kernel200_envelope(const Kernel200EnvelopeLimits& limits,
                                 const Kernel200EnvelopeActual& actual,
                                 Kernel200EnvelopeViolation* violation = nullptr);

} // namespace simaai::neat::pipeline_internal::sima

