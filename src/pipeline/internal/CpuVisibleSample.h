#pragma once
#ifndef SIMA_NEAT_INTERNAL
#error "Internal header. Not part of the public API."
#endif

typedef struct _GstBuffer GstBuffer;

namespace simaai::neat::pipeline_internal {

struct CpuVisibleBufferState {
  bool uses_simaai_backing = false;
  bool cached = false;
  bool has_producer = false;
  unsigned producer = 0;
  bool has_cpu_dirty = false;
  bool cpu_dirty = false;
  bool requires_sync = false;
};

CpuVisibleBufferState cpu_visible_buffer_state(GstBuffer* buffer);

} // namespace simaai::neat::pipeline_internal
