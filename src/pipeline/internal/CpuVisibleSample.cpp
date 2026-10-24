#include "pipeline/internal/CpuVisibleSample.h"

#include "pipeline/internal/SimaaiMemory.h"

#include <gst/gst.h>

#include <cstdint>
#include <cstring>

namespace simaai::neat::pipeline_internal {
namespace {

bool buffer_memory_uses_segment_allocator(const GstMemory* memory) {
  if (!memory || !memory->allocator || !memory->allocator->mem_type) {
    return false;
  }
  const char* mem_type = memory->allocator->mem_type;
  return std::strcmp(mem_type, "SimaaiSegmentMemory") == 0 ||
         std::strcmp(mem_type, "NeatSimaaiSegmentMemory") == 0;
}

bool buffer_uses_simaai_backing(GstBuffer* buffer) {
  if (!buffer) {
    return false;
  }
  const guint memory_count = gst_buffer_n_memory(buffer);
  for (guint i = 0; i < memory_count; ++i) {
    GstMemory* memory = gst_buffer_peek_memory(buffer, i);
    if (!memory) {
      continue;
    }
    if (buffer_memory_uses_segment_allocator(memory)) {
      return true;
    }
    if (GST_MEMORY_FLAG_IS_SET(memory, GST_SIMAAI_MEMORY_TARGET_EV74) ||
        GST_MEMORY_FLAG_IS_SET(memory, GST_SIMAAI_MEMORY_TARGET_DMS0) ||
        GST_MEMORY_FLAG_IS_SET(memory, GST_SIMAAI_MEMORY_TARGET_DMS1) ||
        GST_MEMORY_FLAG_IS_SET(memory, GST_SIMAAI_MEMORY_TARGET_DMS2) ||
        GST_MEMORY_FLAG_IS_SET(memory, GST_SIMAAI_MEMORY_TARGET_DMS3)) {
      return true;
    }
  }
  return false;
}

bool buffer_has_cached_simaai_backing(GstBuffer* buffer) {
  if (!buffer) {
    return false;
  }
  const guint memory_count = gst_buffer_n_memory(buffer);
  for (guint i = 0; i < memory_count; ++i) {
    GstMemory* memory = gst_buffer_peek_memory(buffer, i);
    if (memory && GST_MEMORY_FLAG_IS_SET(memory, GST_SIMAAI_MEMORY_FLAG_CACHED)) {
      return true;
    }
  }
  return false;
}

GQuark sima_buffer_cpu_dirty_quark() {
  static GQuark q = g_quark_from_static_string("sima-buffer-cpu-dirty-v1");
  return q;
}

GQuark sima_buffer_producer_quark() {
  static GQuark q = g_quark_from_static_string("sima-buffer-producer-v1");
  return q;
}

bool decode_small_uint(gpointer value, unsigned* out) {
  if (!value || !out) {
    return false;
  }
  const intptr_t raw = reinterpret_cast<intptr_t>(value);
  if (raw <= 0) {
    return false;
  }
  *out = static_cast<unsigned>(raw - 1);
  return true;
}

enum class SimaBufferProducer : unsigned {
  Unknown = 0,
  Cpu = 1,
  Device = 2,
};

bool query_buffer_cpu_dirty(GstBuffer* buffer, bool* out_dirty) {
  if (!buffer || !out_dirty) {
    return false;
  }
  gpointer value =
      gst_mini_object_get_qdata(GST_MINI_OBJECT_CAST(buffer), sima_buffer_cpu_dirty_quark());
  if (!value) {
    return false;
  }
  *out_dirty = (reinterpret_cast<intptr_t>(value) == 1);
  return true;
}

bool query_buffer_producer(GstBuffer* buffer, SimaBufferProducer* out_producer) {
  if (!buffer || !out_producer) {
    return false;
  }
  unsigned decoded = 0U;
  if (!decode_small_uint(
          gst_mini_object_get_qdata(GST_MINI_OBJECT_CAST(buffer), sima_buffer_producer_quark()),
          &decoded)) {
    return false;
  }
  *out_producer = static_cast<SimaBufferProducer>(decoded);
  return true;
}

} // namespace

CpuVisibleBufferState cpu_visible_buffer_state(GstBuffer* buffer) {
  CpuVisibleBufferState out;
  out.uses_simaai_backing = buffer_uses_simaai_backing(buffer);
  out.cached = buffer_has_cached_simaai_backing(buffer);
  if (!buffer || !out.uses_simaai_backing || !out.cached) {
    out.requires_sync = false;
    return out;
  }

  // Wrapper hints are diagnostics only: pool reuse does not identify a new
  // device-write generation. CPU visibility is established by the allocation's
  // retained map lease, never by eager synchronization or wrapper qdata.
  SimaBufferProducer producer = SimaBufferProducer::Unknown;
  out.has_producer = query_buffer_producer(buffer, &producer);
  out.producer = static_cast<unsigned>(producer);
  out.has_cpu_dirty = query_buffer_cpu_dirty(buffer, &out.cpu_dirty);

  out.requires_sync = true;
  return out;
}

} // namespace simaai::neat::pipeline_internal
