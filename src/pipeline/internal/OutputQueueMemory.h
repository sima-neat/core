#pragma once
#ifndef SIMA_NEAT_INTERNAL
#error "Internal header. Not part of the public API."
#endif

#include "pipeline/internal/SampleUtil.h"

namespace simaai::neat::pipeline_internal {

// Per-selected-payload policy only. No allocation, queue mutation, credit release,
// or promotion of a foreign sibling to framework-owned storage occurs here.
struct OutputQueueMemoryDecision {
  bool has_dma = false;
  bool only_framework = false;
  bool preserve_framework = false;
  bool strict_zero_copy = false;

  bool permits_pressure_copy() const noexcept {
    return !has_dma && !only_framework;
  }
  bool copy_when_latched(bool latched) const noexcept {
    return latched && !(has_dma && preserve_framework) && !strict_zero_copy;
  }
};

inline OutputQueueMemoryDecision output_queue_memory_decision(const Sample& sample,
                                                              bool requested_copy,
                                                              bool requested_preservation,
                                                              bool strict_zero_copy) {
  const bool has_dma = sample_has_dmabuf_memory(sample);
  const bool has_framework = has_dma || sample_has_certified_terminal_host_memory(sample);
  return {has_dma, sample_has_only_framework_output_memory(sample),
          has_framework && (requested_preservation || !requested_copy), strict_zero_copy};
}

} // namespace simaai::neat::pipeline_internal
