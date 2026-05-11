/**
 * @file
 * @ingroup builder
 * @brief Decoder-side policy: resolution caps and caps-negotiation rules.
 *
 * Holds the limits the framework enforces on decoder-bearing nodes (file readers, RTSP
 * input, image decoders) — typically a maximum frame resolution plus a flag controlling
 * whether dynamic GStreamer caps are accepted. Used by builder-side validation when a
 * Session is composed.
 *
 * @see Policy.h
 * @see DefaultPolicy
 */
#pragma once

#include "policy/Policy.h"

namespace simaai::neat::policy {

/**
 * @brief Decoder-side policy parameters and validation.
 *
 * Defaults are tuned for 4K (3840x2160) streams with dynamic caps enabled. Adjust
 * fields directly on a `DefaultPolicy::decoder` instance if your application needs
 * tighter or looser limits.
 * @ingroup builder
 */
struct DecoderPolicy {
  int max_width = 3840;          ///< Maximum accepted frame width in pixels.
  int max_height = 2160;         ///< Maximum accepted frame height in pixels.
  bool allow_dynamic_caps = true; ///< If false, decoder caps must be statically known.

  /**
   * @brief Validate a decoder's resolution against the configured caps.
   * @param width Decoded frame width in pixels.
   * @param height Decoded frame height in pixels.
   * @return `Evaluation` allowing the resolution or denying with a reason.
   */
  Evaluation validate_resolution(int width, int height) const;
};

} // namespace simaai::neat::policy
