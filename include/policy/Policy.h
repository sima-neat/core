/**
 * @file
 * @ingroup builder
 * @brief Base policy vocabulary shared across the policy subsystem.
 *
 * Defines the minimal Allow/Deny decision contract returned by every policy check
 * (decoder, encoder, RTSP, memory). Concrete policies — see DefaultPolicy — bundle
 * individual policy structs and call back into the framework with these `Evaluation`
 * results to validate user-supplied parameters before a Graph is built.
 *
 * @see DefaultPolicy
 * @see DecoderPolicy
 * @see EncoderPolicy
 * @see RtspPolicy
 * @see MemoryPolicy
 */
#pragma once

#include <string>

namespace simaai::neat::policy {

/**
 * @brief Outcome of a single policy check.
 * @ingroup builder
 */
enum class Decision {
  Allow, ///< Parameter is acceptable; the framework may proceed.
  Deny,  ///< Parameter is rejected; build/run should fail with `reason`.
};

/**
 * @brief Result of evaluating a parameter against a policy.
 *
 * Returned by every `validate_*` method on the per-domain policy structs. When `decision`
 * is `Decision::Deny`, `reason` carries a human-readable message suitable for a NeatError.
 * @ingroup builder
 */
struct Evaluation {
  Decision decision = Decision::Allow; ///< Allow or deny outcome; defaults to Allow.
  std::string reason;                  ///< Human-readable diagnostic when denied.

  /// True if the policy allowed the parameter; false if it denied.
  bool ok() const noexcept {
    return decision == Decision::Allow;
  }
};

} // namespace simaai::neat::policy
