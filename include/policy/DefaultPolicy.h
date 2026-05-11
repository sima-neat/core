/**
 * @file
 * @ingroup builder
 * @brief Opinionated default policy bundle.
 *
 * Aggregates the four per-domain policies (decoder, encoder, memory, RTSP) into a single
 * struct that the framework instantiates when the application doesn't supply its own.
 * `make_default_policy()` returns a value with the framework's recommended limits — what
 * "just works" for the common Modalix DevKit pipelines.
 *
 * @see Policy.h for the underlying Decision/Evaluation contract
 */
#pragma once

#include "policy/DecoderPolicy.h"
#include "policy/EncoderPolicy.h"
#include "policy/MemoryPolicy.h"
#include "policy/RtspPolicy.h"

namespace simaai::neat::policy {

/**
 * @brief Bundle of all per-domain policies the framework consults during build/run.
 *
 * Each member has its own defaults (see the individual policy headers). Applications can
 * either accept the framework defaults via `make_default_policy()` or tweak fields directly.
 * @ingroup builder
 */
struct DefaultPolicy {
  DecoderPolicy decoder{};  ///< Decoder-side limits (resolution caps, etc.).
  EncoderPolicy encoder{};  ///< Encoder-side limits (bitrate range, etc.).
  MemoryPolicy memory{};    ///< Memory-pool limits.
  RtspPolicy rtsp{};        ///< RTSP-specific limits (port range, etc.).
};

/// Returns a `DefaultPolicy` populated with the framework's recommended defaults.
DefaultPolicy make_default_policy();

} // namespace simaai::neat::policy
