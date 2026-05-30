#include "pipeline/internal/CpuVisibleSample.h"

#include "pipeline/internal/SimaMemApi.h"
#include "pipeline/internal/SimaaiMemory.h"

#include <gst/gst.h>

#include <algorithm>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <dlfcn.h>
#include <mutex>
#include <string>
#include <unordered_map>
#include <vector>

namespace simaai::neat::pipeline_internal {
namespace {

extern "C" {
gsize gst_simaai_memory_get_segment_count(const GstMemory* memory);
void* gst_simaai_memory_get_segment_at(const GstMemory* memory, gsize index);
}

// SimaMemApi, sima_mem_api(), and simaai_memory_t come from the shared internal
// header pipeline/internal/SimaMemApi.h (single source of truth, one process-wide
// dlsym resolution). Previously duplicated here and in TensorUtil.cpp.

bool buffer_memory_uses_segment_allocator(const GstMemory* memory) {
  if (!memory || !memory->allocator || !memory->allocator->mem_type) {
    return false;
  }
  const std::string mem_type = memory->allocator->mem_type;
  return mem_type == "SimaaiSegmentMemory" || mem_type == "NeatSimaaiSegmentMemory";
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

GQuark sima_sample_cpu_visible_prepared_quark() {
  static GQuark q = g_quark_from_static_string("sima-sample-cpu-visible-prepared-v1");
  return q;
}

gpointer encode_small_uint(unsigned value) {
  return reinterpret_cast<gpointer>(static_cast<intptr_t>(value + 1U));
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

void mark_buffer_cpu_read_clean(GstBuffer* buffer) {
  if (!buffer) {
    return;
  }
  gst_mini_object_set_qdata(GST_MINI_OBJECT_CAST(buffer), sima_buffer_cpu_dirty_quark(),
                            reinterpret_cast<gpointer>(static_cast<intptr_t>(2)), nullptr);
  gst_mini_object_set_qdata(GST_MINI_OBJECT_CAST(buffer), sima_buffer_producer_quark(),
                            encode_small_uint(static_cast<unsigned>(SimaBufferProducer::Unknown)),
                            nullptr);
}

void mark_sample_cpu_read_prepared(GstSample* sample) {
  if (!sample || !GST_IS_SAMPLE(sample)) {
    return;
  }
  gst_mini_object_set_qdata(GST_MINI_OBJECT_CAST(sample), sima_sample_cpu_visible_prepared_quark(),
                            reinterpret_cast<gpointer>(static_cast<intptr_t>(1)), nullptr);
}

uint64_t resolve_buffer_id(GstBuffer* buffer) {
  if (!buffer) {
    return 0;
  }
  GstCustomMeta* meta = gst_buffer_get_custom_meta(buffer, "GstSimaMeta");
  GstStructure* s = meta ? gst_custom_meta_get_structure(meta) : nullptr;
  if (s) {
    gint64 buffer_id = -1;
    if (gst_structure_get_int64(s, "buffer-id", &buffer_id) && buffer_id > 0) {
      return static_cast<uint64_t>(buffer_id);
    }
  }

  SimaMemApi& api = sima_mem_api();
  if (!api.get_phys) {
    return 0;
  }
  const guint n_mems = gst_buffer_n_memory(buffer);
  GstMemory* mem = (n_mems > 0) ? gst_buffer_peek_memory(buffer, n_mems - 1) : nullptr;
  if (!mem) {
    return 0;
  }
  return static_cast<uint64_t>(api.get_phys(mem));
}

simaai_memory_t* get_or_attach_memory(uint64_t buffer_id) {
  SimaMemApi& api = sima_mem_api();
  if (!api.attach || buffer_id == 0) {
    return nullptr;
  }
  static std::mutex mu;
  static std::unordered_map<uint64_t, simaai_memory_t*> cache;
  {
    std::lock_guard<std::mutex> lock(mu);
    auto it = cache.find(buffer_id);
    if (it != cache.end()) {
      return it->second;
    }
  }
  simaai_memory_t* mem = api.attach(buffer_id);
  if (!mem) {
    return nullptr;
  }
  {
    std::lock_guard<std::mutex> lock(mu);
    cache.emplace(buffer_id, mem);
  }
  return mem;
}

std::size_t invalidate_buffer_segment_backing_for_cpu_read(GstBuffer* buffer,
                                                           const SimaMemApi& api) {
  if (!buffer) {
    return 0U;
  }
  // Prefer the allocator's single coherency authority (ownership-skip-gated
  // invalidate over all segments + mark-CPU-clean on the inline per-allocation
  // state), so the framework's CPU-read prepare shares one state and one skip
  // decision with the allocator map(READ) path. If the loaded allocator predates
  // that symbol (api.cpu_read_begin == null), fall back to the legacy
  // unconditional per-segment invalidate — always-invalidate, correct but
  // unoptimised — so the framework stays loadable on any allocator.
  std::size_t prepared = 0U;
  const guint mem_count = gst_buffer_n_memory(buffer);
  for (guint mi = 0; mi < mem_count; ++mi) {
    GstMemory* gst_mem = gst_buffer_peek_memory(buffer, mi);
    if (!buffer_memory_uses_segment_allocator(gst_mem)) {
      continue;
    }
    if (api.cpu_read_begin) {
      api.cpu_read_begin(gst_mem);
    } else if (api.invalidate) {
      const gsize seg_count = gst_simaai_memory_get_segment_count(gst_mem);
      for (gsize si = 0; si < seg_count; ++si) {
        auto* seg =
            reinterpret_cast<simaai_memory_t*>(gst_simaai_memory_get_segment_at(gst_mem, si));
        if (seg) {
          api.invalidate(seg);
        }
      }
    }
    ++prepared;
  }
  return prepared;
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

  // The producer/dirty marks are queried for DIAGNOSTICS ONLY. They live as
  // GQuark qdata on the GstBuffer/GstSample WRAPPER, which GStreamer buffer
  // pools recycle over the same physical segment. Trusting a "Cpu" or "clean"
  // wrapper mark to SKIP the invalidate is the recycling bug: a device write
  // into the recycled segment is then missed and the CPU reads stale cache.
  // Until coherency state is keyed to the ALLOCATION (Tier-A Phase 1: inline
  // SegCoherency on GstSimaaiSegmentMemory + an end_device_access mark on the
  // device-write path), every cached SiMa-backed buffer is synced before a CPU
  // read. This matches the allocator's own map(READ) invalidate and the
  // Tensor::map(Read) conservative fallback; it costs redundant invalidates,
  // never a missed one.
  SimaBufferProducer producer = SimaBufferProducer::Unknown;
  out.has_producer = query_buffer_producer(buffer, &producer);
  out.producer = static_cast<unsigned>(producer);
  out.has_cpu_dirty = query_buffer_cpu_dirty(buffer, &out.cpu_dirty);

  out.requires_sync = true;
  return out;
}

bool gst_buffer_cpu_read_requires_sync(GstBuffer* buffer) {
  return cpu_visible_buffer_state(buffer).requires_sync;
}

bool gst_sample_cpu_read_prepared(GstSample* sample) {
  if (!sample || !GST_IS_SAMPLE(sample)) {
    return false;
  }
  return gst_mini_object_get_qdata(GST_MINI_OBJECT_CAST(sample),
                                   sima_sample_cpu_visible_prepared_quark()) != nullptr;
}

CpuVisiblePrepareResult prepare_gst_buffer_for_cpu_read(GstBuffer* buffer) {
  CpuVisiblePrepareResult result;
  if (!buffer) {
    return result;
  }
  result.buffers_seen = 1;
  if (!gst_buffer_cpu_read_requires_sync(buffer)) {
    return result;
  }

  SimaMemApi& api = sima_mem_api();
  if (!api.invalidate && !api.cpu_read_begin) {
    return result;
  }

  const std::size_t segment_count = invalidate_buffer_segment_backing_for_cpu_read(buffer, api);
  if (segment_count > 0U) {
    result.buffers_synced = 1;
    result.segments_synced = segment_count;
    return result;
  }

  const uint64_t buffer_id = resolve_buffer_id(buffer);
  if (buffer_id == 0) {
    return result;
  }
  simaai_memory_t* mem = get_or_attach_memory(buffer_id);
  if (!mem) {
    return result;
  }
  api.invalidate(mem);
  result.buffers_synced = 1;
  result.segments_synced = 1;
  return result;
}

CpuVisiblePrepareResult prepare_gst_sample_for_cpu_read(GstSample* sample) {
  if (!sample || !GST_IS_SAMPLE(sample)) {
    return {};
  }
  GstBuffer* buffer = gst_sample_get_buffer(sample);
  const bool requires_sync = gst_buffer_cpu_read_requires_sync(buffer);
  CpuVisiblePrepareResult result = prepare_gst_buffer_for_cpu_read(buffer);
  if (!requires_sync || result.buffers_synced > 0U) {
    // Keep this marker diagnostic only for now. Some GStreamer paths recycle
    // sample/buffer wrapper objects across device writes, so Tensor::map(Read)
    // must still use the producer/dirty state as the correctness fallback and
    // must not blindly trust a wrapper-scoped "already prepared" bit.
    mark_sample_cpu_read_prepared(sample);
  }
  return result;
}

CpuVisiblePrepareResult prepare_tensor_for_cpu_read(const Tensor& tensor) {
  if (!tensor.storage || tensor.storage->kind != StorageKind::GstSample ||
      !tensor.storage->holder) {
    return {};
  }
  auto* sample = static_cast<GstSample*>(tensor.storage->holder.get());
  return prepare_gst_sample_for_cpu_read(sample);
}

CpuVisiblePrepareResult prepare_sample_for_cpu_read(Sample& sample) {
  CpuVisiblePrepareResult result;
  if (sample.tensor.has_value()) {
    result.add(prepare_tensor_for_cpu_read(*sample.tensor));
  }
  for (auto& tensor : sample.tensors) {
    result.add(prepare_tensor_for_cpu_read(tensor));
  }
  for (auto& field : sample.fields) {
    result.add(prepare_sample_for_cpu_read(field));
  }
  return result;
}

} // namespace simaai::neat::pipeline_internal
