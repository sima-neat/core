#include "pipeline/internal/CpuVisibleSample.h"

#include "pipeline/internal/SimaaiMemory.h"

#include <gst/gst.h>

#include <algorithm>
#include <cstdint>
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

struct simaai_memory;
using simaai_memory_t = simaai_memory;

struct SimaMemApi {
  using AttachFn = simaai_memory_t* (*)(uint64_t);
  using InvalidateFn = void (*)(simaai_memory_t*);
  using GetPhysFn = guintptr (*)(GstMemory*);

  AttachFn attach = nullptr;
  InvalidateFn invalidate = nullptr;
  GetPhysFn get_phys = nullptr;
  bool tried = false;
};

SimaMemApi& sima_mem_api() {
  static SimaMemApi api;
  if (api.tried) {
    return api;
  }
  api.tried = true;

  void* handle = dlopen("libsimaaimem.so", RTLD_LAZY | RTLD_LOCAL);
  if (!handle) {
    handle = dlopen("libsimaaimem.so.1", RTLD_LAZY | RTLD_LOCAL);
  }
  if (!handle) {
    return api;
  }

  api.attach = reinterpret_cast<SimaMemApi::AttachFn>(dlsym(handle, "simaai_memory_attach"));
  api.invalidate = reinterpret_cast<SimaMemApi::InvalidateFn>(
      dlsym(handle, "simaai_memory_invalidate_cache"));
  api.get_phys = reinterpret_cast<SimaMemApi::GetPhysFn>(
      dlsym(handle, "gst_simaai_memory_get_phys_addr"));
  return api;
}

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
  gpointer value = gst_mini_object_get_qdata(GST_MINI_OBJECT_CAST(buffer),
                                             sima_buffer_cpu_dirty_quark());
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
  if (!decode_small_uint(gst_mini_object_get_qdata(GST_MINI_OBJECT_CAST(buffer),
                                                   sima_buffer_producer_quark()),
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
  gst_mini_object_set_qdata(
      GST_MINI_OBJECT_CAST(buffer), sima_buffer_producer_quark(),
      encode_small_uint(static_cast<unsigned>(SimaBufferProducer::Unknown)), nullptr);
}

void mark_sample_cpu_read_prepared(GstSample* sample) {
  if (!sample || !GST_IS_SAMPLE(sample)) {
    return;
  }
  gst_mini_object_set_qdata(GST_MINI_OBJECT_CAST(sample),
                            sima_sample_cpu_visible_prepared_quark(),
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

bool seen_sima_segment(const std::vector<simaai_memory_t*>& seen, simaai_memory_t* mem) {
  return std::find(seen.begin(), seen.end(), mem) != seen.end();
}

std::size_t invalidate_buffer_segment_backing_for_cpu_read(GstBuffer* buffer,
                                                           const SimaMemApi& api) {
  if (!buffer || !api.invalidate) {
    return 0U;
  }
  std::size_t invalidated = 0U;
  std::vector<simaai_memory_t*> seen;
  const guint mem_count = gst_buffer_n_memory(buffer);
  for (guint mi = 0; mi < mem_count; ++mi) {
    GstMemory* gst_mem = gst_buffer_peek_memory(buffer, mi);
    if (!buffer_memory_uses_segment_allocator(gst_mem)) {
      continue;
    }
    const gsize seg_count = gst_simaai_memory_get_segment_count(gst_mem);
    for (gsize si = 0; si < seg_count; ++si) {
      auto* segment = reinterpret_cast<simaai_memory_t*>(
          gst_simaai_memory_get_segment_at(gst_mem, si));
      if (!segment || seen_sima_segment(seen, segment)) {
        continue;
      }
      seen.push_back(segment);
      api.invalidate(segment);
      ++invalidated;
    }
  }
  return invalidated;
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

  SimaBufferProducer producer = SimaBufferProducer::Unknown;
  out.has_producer = query_buffer_producer(buffer, &producer);
  out.producer = static_cast<unsigned>(producer);
  if (out.has_producer) {
    if (producer == SimaBufferProducer::Device) {
      out.requires_sync = true;
      return out;
    }
    if (producer == SimaBufferProducer::Cpu) {
      out.requires_sync = false;
      return out;
    }
  }

  out.has_cpu_dirty = query_buffer_cpu_dirty(buffer, &out.cpu_dirty);
  if (out.has_cpu_dirty) {
    out.requires_sync = out.cpu_dirty;
    return out;
  }

  // Unknown cached SiMa-backed producer remains conservative. This mirrors the
  // Tensor::map(Read) correctness fallback and is intentionally shared with the
  // async output preparation path.
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
  if (!api.invalidate) {
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

CpuVisiblePrepareResult prepare_tensor_for_cpu_read(Tensor& tensor) {
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
