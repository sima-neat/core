#pragma once
#ifndef SIMA_NEAT_INTERNAL
#error "Internal header. Not part of the public API."
#endif

#include <cstdint>

namespace simaai::neat {

struct InputStreamStats {
  std::uint64_t push_count = 0;            ///< Total successful push calls.
  std::uint64_t push_failures = 0;         ///< Pushes that failed (queue full, EOS, error).
  std::uint64_t pull_count = 0;            ///< Total successful pull calls (output side).
  std::uint64_t poll_count = 0;            ///< Total pull calls (including timeouts).
  std::uint64_t dropped_frames = 0;        ///< Frames dropped due to overflow policy.
  std::uint64_t renegotiations = 0;        ///< Times caps were re-negotiated mid-stream.
  std::uint64_t alloc_grows = 0;           ///< Times the buffer pool grew due to demand.
  std::uint64_t growth_blocked = 0;        ///< Pool-grow attempts that hit a configured cap.
  std::uint64_t renegotiation_blocked = 0; ///< Renegotiation attempts rejected downstream.
  double avg_alloc_us = 0.0;               ///< Average buffer allocation time.
  double avg_map_us = 0.0;                 ///< Average buffer map time.
  double avg_copy_us = 0.0;                ///< Average input copy time.
  double avg_push_us = 0.0;                ///< Average end-to-end push call time.
  double avg_pull_wait_us = 0.0;           ///< Average output wait time.
  double avg_decode_us = 0.0;              ///< Average input-side decode time.
};

} // namespace simaai::neat
