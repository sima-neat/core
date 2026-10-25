#pragma once

#include <cstdint>

namespace simaai::neat::pipeline_internal {

// Diagnostic observation only, not a jitter buffer or packet-loss estimator.
// One instance is used by one serialized pad probe. Reordered/duplicate packets
// do not move the high-water mark backwards; SSRC changes start a new sequence.
struct RtpSequenceObservation {
  std::uint32_t ssrc = 0;
  std::uint16_t sequence = 0;
  bool initialized = false;

  struct Delta {
    unsigned forward_gap = 0;
    bool reordered_or_duplicate = false;
    bool ssrc_changed = false;
  };

  Delta observe(std::uint32_t current_ssrc, std::uint16_t current_sequence) {
    Delta delta;
    if (!initialized || ssrc != current_ssrc) {
      delta.ssrc_changed = initialized;
      initialized = true;
      ssrc = current_ssrc;
      sequence = current_sequence;
      return delta;
    }
    const auto step = static_cast<std::uint16_t>(current_sequence - sequence);
    if (step == 0 || step >= 0x8000U) {
      delta.reordered_or_duplicate = true;
    } else {
      delta.forward_gap = step - 1U;
      sequence = current_sequence;
    }
    return delta;
  }
};

} // namespace simaai::neat::pipeline_internal
