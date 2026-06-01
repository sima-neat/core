/**
 * @file
 * @ingroup builder
 * @brief Encoder-side policy: bitrate bounds for outgoing video.
 *
 * Defines the legal range of encoder bitrates accepted by the framework. Encoder-bearing
 * nodes (e.g., the H.264 encoder feeding RTSP/UDP outputs) consult this policy at build
 * time and reject configurations outside the range with a NeatError.
 *
 * @see Policy.h
 * @see DefaultPolicy
 */
#pragma once

#include "policy/Policy.h"

namespace simaai::neat::policy {

/**
 * @brief Encoder-side policy parameters and validation.
 *
 * Defaults span 100 kbps to 100 Mbps — wide enough for both low-bandwidth IP cameras
 * and high-quality 4K streams. Tighten the range if the application has stricter caps.
 * @ingroup builder
 */
struct EncoderPolicy {
  int min_bitrate_kbps = 100;    ///< Lower bound on encoder bitrate (kbit/s).
  int max_bitrate_kbps = 100000; ///< Upper bound on encoder bitrate (kbit/s).

  /**
   * @brief Validate a requested encoder bitrate against the configured range.
   * @param bitrate_kbps Requested bitrate in kbit/s.
   * @return `Evaluation` allowing the bitrate or denying with a reason.
   */
  Evaluation validate_bitrate(int bitrate_kbps) const;
};

} // namespace simaai::neat::policy
