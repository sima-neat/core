#pragma once
#ifndef SIMA_NEAT_INTERNAL
#error "Internal header. Not part of the public API."
#endif

#include "pipeline/GraphOptions.h"
#include "pipeline/Tensor.h"

#include <cstddef>

typedef struct _GstBuffer GstBuffer;
typedef struct _GstSample GstSample;

namespace simaai::neat::pipeline_internal {

struct CpuVisiblePrepareResult {
  std::size_t buffers_seen = 0;
  std::size_t buffers_synced = 0;
  std::size_t segments_synced = 0;

  void add(const CpuVisiblePrepareResult& other) {
    buffers_seen += other.buffers_seen;
    buffers_synced += other.buffers_synced;
    segments_synced += other.segments_synced;
  }
};

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
bool gst_buffer_cpu_read_requires_sync(GstBuffer* buffer);
bool gst_sample_cpu_read_prepared(GstSample* sample);
CpuVisiblePrepareResult prepare_gst_buffer_for_cpu_read(GstBuffer* buffer);
CpuVisiblePrepareResult prepare_gst_sample_for_cpu_read(GstSample* sample);
CpuVisiblePrepareResult prepare_tensor_for_cpu_read(Tensor& tensor);
CpuVisiblePrepareResult prepare_sample_for_cpu_read(Sample& sample);

} // namespace simaai::neat::pipeline_internal
