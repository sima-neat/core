// src/pipeline/TensorUtil.cpp
#include "pipeline/internal/TensorUtil.h"

#include "pipeline/internal/CpuVisibleSample.h"
#include "pipeline/internal/GstDiagnosticsUtil.h"
#include "pipeline/internal/InputStreamUtil.h"
#include "pipeline/internal/EnvUtil.h"
#include "pipeline/internal/SimaaiGstCompat.h"
#include "pipeline/internal/SimaaiMemory.h"

#include <gst/gst.h>

#include <dlfcn.h>

#include <algorithm>
#include <cstring>
#include <atomic>
#include <cstdint>
#include <limits>
#include <memory>
#include <mutex>
#include <stdexcept>
#include <unordered_map>
#include <vector>

namespace simaai::neat::pipeline_internal {

namespace {

struct CompositeHolder {
  std::shared_ptr<void> primary;
  std::shared_ptr<void> keepalive;
};

std::shared_ptr<void> make_preprocess_meta_holder(GstBuffer* source) {
  if (!source || !has_simaai_preprocess_meta(source)) {
    return {};
  }
  GstBuffer* meta_buf = gst_buffer_new();
  if (!meta_buf) {
    return {};
  }

  std::string copy_err;
  if (!copy_simaai_preprocess_meta(meta_buf, source, &copy_err)) {
    gst_buffer_unref(meta_buf);
    return {};
  }

  GstSample* meta_sample = gst_sample_new(meta_buf, nullptr, nullptr, nullptr);
  gst_buffer_unref(meta_buf);
  if (!meta_sample) {
    return {};
  }

  auto holder = std::shared_ptr<void>(
      gst_sample_ref(meta_sample), [](void* p) { gst_sample_unref(static_cast<GstSample*>(p)); });
  gst_sample_unref(meta_sample);
  return holder;
}

extern "C" {
gsize gst_simaai_memory_get_segment_count(const GstMemory* memory);
const gchar* gst_simaai_memory_get_segment_name_at(const GstMemory* memory, gsize index);
gsize gst_simaai_memory_get_segment_size_at(const GstMemory* memory, gsize index);
void* gst_simaai_memory_get_segment(const GstMemory* memory, const gchar* name);
void* gst_simaai_memory_get_segment_at(const GstMemory* memory, gsize index);
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

std::vector<simaai::neat::Segment> extract_runtime_segments_from_buffer(GstBuffer* buffer) {
  std::vector<simaai::neat::Segment> segments;
  if (!buffer) {
    return segments;
  }

  const guint memory_count = gst_buffer_n_memory(buffer);
  if (memory_count == 0U) {
    return segments;
  }

  if (memory_count == 1U) {
    GstMemory* memory = gst_buffer_peek_memory(buffer, 0U);
    if (buffer_memory_uses_segment_allocator(memory)) {
      const gsize segment_count = gst_simaai_memory_get_segment_count(memory);
      segments.reserve(static_cast<std::size_t>(segment_count));
      for (gsize i = 0; i < segment_count; ++i) {
        const gchar* raw_name = gst_simaai_memory_get_segment_name_at(memory, i);
        const gsize raw_size = gst_simaai_memory_get_segment_size_at(memory, i);
        std::string name = raw_name ? std::string(raw_name) : std::string();
        if (name.empty()) {
          name = "memory" + std::to_string(static_cast<std::size_t>(i));
        }
        segments.push_back(
            simaai::neat::Segment{std::move(name), static_cast<std::size_t>(raw_size)});
      }
      if (!segments.empty()) {
        return segments;
      }
    }
  }

  segments.reserve(memory_count);
  for (guint i = 0; i < memory_count; ++i) {
    GstMemory* memory = gst_buffer_peek_memory(buffer, i);
    gsize offset = 0;
    gsize maxsize = 0;
    const gsize size = memory ? gst_memory_get_sizes(memory, &offset, &maxsize) : 0U;
    segments.push_back(simaai::neat::Segment{"memory" + std::to_string(static_cast<std::size_t>(i)),
                                             static_cast<std::size_t>(size)});
  }
  return segments;
}

struct simaai_memory;
using simaai_memory_t = simaai_memory;

struct SimaMemApi {
  using AttachFn = simaai_memory_t* (*)(uint64_t);
  using InvalidateFn = void (*)(simaai_memory_t*);
  using InvalidatePartFn = void (*)(simaai_memory_t*, unsigned int, unsigned int);
  using FlushFn = void (*)(simaai_memory_t*);
  using MapFn = void* (*)(simaai_memory_t*);
  using UnmapFn = void (*)(simaai_memory_t*);
  using GetPhysFn = guintptr (*)(GstMemory*);

  AttachFn attach = nullptr;
  InvalidateFn invalidate = nullptr;
  InvalidatePartFn invalidate_part = nullptr;
  FlushFn flush = nullptr;
  MapFn map = nullptr;
  UnmapFn unmap = nullptr;
  GetPhysFn get_phys = nullptr;
  bool tried = false;
};

SimaMemApi& sima_mem_api() {
  static SimaMemApi api;
  if (api.tried)
    return api;
  api.tried = true;

  void* handle = dlopen("libsimaaimem.so", RTLD_LAZY | RTLD_LOCAL);
  if (!handle) {
    handle = dlopen("libsimaaimem.so.1", RTLD_LAZY | RTLD_LOCAL);
  }
  if (!handle)
    return api;

  api.attach = reinterpret_cast<SimaMemApi::AttachFn>(dlsym(handle, "simaai_memory_attach"));
  api.invalidate =
      reinterpret_cast<SimaMemApi::InvalidateFn>(dlsym(handle, "simaai_memory_invalidate_cache"));
  api.invalidate_part = reinterpret_cast<SimaMemApi::InvalidatePartFn>(
      dlsym(handle, "simaai_memory_invalidate_cache_part"));
  api.flush = reinterpret_cast<SimaMemApi::FlushFn>(dlsym(handle, "simaai_memory_flush_cache"));
  api.map = reinterpret_cast<SimaMemApi::MapFn>(dlsym(handle, "simaai_memory_map"));
  api.unmap = reinterpret_cast<SimaMemApi::UnmapFn>(dlsym(handle, "simaai_memory_unmap"));
  api.get_phys =
      reinterpret_cast<SimaMemApi::GetPhysFn>(dlsym(handle, "gst_simaai_memory_get_phys_addr"));
  return api;
}

uint64_t resolve_buffer_id(GstBuffer* buffer) {
  if (!buffer)
    return 0;
  GstCustomMeta* meta = gst_buffer_get_custom_meta(buffer, "GstSimaMeta");
  GstStructure* s = meta ? gst_custom_meta_get_structure(meta) : nullptr;
  if (s) {
    gint64 buffer_id = -1;
    if (gst_structure_get_int64(s, "buffer-id", &buffer_id) && buffer_id > 0) {
      return static_cast<uint64_t>(buffer_id);
    }
  }

  SimaMemApi& api = sima_mem_api();
  if (!api.get_phys)
    return 0;

  const guint n_mems = gst_buffer_n_memory(buffer);
  GstMemory* mem = (n_mems > 0) ? gst_buffer_peek_memory(buffer, n_mems - 1) : nullptr;
  if (!mem)
    return 0;
  return static_cast<uint64_t>(api.get_phys(mem));
}

simaai_memory_t* get_or_attach_memory(uint64_t buffer_id) {
  SimaMemApi& api = sima_mem_api();
  if (!api.attach || buffer_id == 0)
    return nullptr;
  static std::mutex mu;
  static std::unordered_map<uint64_t, simaai_memory_t*> cache;
  {
    std::lock_guard<std::mutex> lock(mu);
    auto it = cache.find(buffer_id);
    if (it != cache.end())
      return it->second;
  }
  simaai_memory_t* mem = api.attach(buffer_id);
  if (!mem)
    return nullptr;
  {
    std::lock_guard<std::mutex> lock(mu);
    cache.emplace(buffer_id, mem);
  }
  return mem;
}

GQuark sima_buffer_cpu_dirty_quark_local() {
  static GQuark q = g_quark_from_static_string("sima-buffer-cpu-dirty-v1");
  return q;
}

GQuark sima_buffer_producer_quark_local() {
  static GQuark q = g_quark_from_static_string("sima-buffer-producer-v1");
  return q;
}

gpointer encode_small_uint_local(unsigned value) {
  // qdata uses nullptr as absent, so store value + 1.
  return reinterpret_cast<gpointer>(static_cast<intptr_t>(value + 1U));
}

bool decode_small_uint_local(gpointer value, unsigned* out) {
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

bool query_buffer_cpu_dirty(GstBuffer* buffer, bool* out_dirty) {
  if (!buffer || !out_dirty) {
    return false;
  }
  gpointer value =
      gst_mini_object_get_qdata(GST_MINI_OBJECT_CAST(buffer), sima_buffer_cpu_dirty_quark_local());
  if (!value) {
    return false;
  }
  *out_dirty = (reinterpret_cast<intptr_t>(value) == 1);
  return true;
}

enum class LocalSimaBufferProducer : unsigned {
  Unknown = 0,
  Cpu = 1,
  Device = 2,
};

bool query_buffer_producer(GstBuffer* buffer, LocalSimaBufferProducer* out_producer) {
  if (!buffer || !out_producer) {
    return false;
  }
  unsigned decoded = 0U;
  if (!decode_small_uint_local(gst_mini_object_get_qdata(GST_MINI_OBJECT_CAST(buffer),
                                                         sima_buffer_producer_quark_local()),
                               &decoded)) {
    return false;
  }
  *out_producer = static_cast<LocalSimaBufferProducer>(decoded);
  return true;
}

void mark_buffer_cpu_read_clean(GstBuffer* buffer) {
  if (!buffer) {
    return;
  }
  gst_mini_object_set_qdata(GST_MINI_OBJECT_CAST(buffer), sima_buffer_cpu_dirty_quark_local(),
                            reinterpret_cast<gpointer>(static_cast<intptr_t>(2)), nullptr);
  gst_mini_object_set_qdata(
      GST_MINI_OBJECT_CAST(buffer), sima_buffer_producer_quark_local(),
      encode_small_uint_local(static_cast<unsigned>(LocalSimaBufferProducer::Unknown)), nullptr);
}

bool buffer_cpu_read_requires_invalidate(GstBuffer* buffer) {
  return gst_buffer_cpu_read_requires_sync(buffer);
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
      auto* segment =
          reinterpret_cast<simaai_memory_t*>(gst_simaai_memory_get_segment_at(gst_mem, si));
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

void invalidate_cached_buffer(GstBuffer* buffer) {
  (void)prepare_gst_buffer_for_cpu_read(buffer);
}

std::size_t resolve_buffer_offset(GstBuffer* buffer) {
  if (!buffer)
    return 0;
  const guint n_mems = gst_buffer_n_memory(buffer);
  GstMemory* mem0 = (n_mems > 0) ? gst_buffer_peek_memory(buffer, n_mems - 1) : nullptr;
  if (mem0) {
    gsize mem_offset = 0;
    gsize mem_max = 0;
    gst_memory_get_sizes(mem0, &mem_offset, &mem_max);
    if (mem_offset > 0)
      return static_cast<std::size_t>(mem_offset);
  }
  GstCustomMeta* meta = gst_buffer_get_custom_meta(buffer, "GstSimaMeta");
  GstStructure* s = meta ? gst_custom_meta_get_structure(meta) : nullptr;
  if (s) {
    gint64 buffer_offset = 0;
    if (gst_structure_get_int64(s, "buffer-offset", &buffer_offset) && buffer_offset > 0) {
      return static_cast<std::size_t>(buffer_offset);
    }
  }
  return 0;
}

std::size_t tensor_dtype_bytes(TensorDType dtype) {
  switch (dtype) {
  case TensorDType::UInt8:
    return 1;
  case TensorDType::Int8:
    return 1;
  case TensorDType::UInt16:
    return 2;
  case TensorDType::Int16:
    return 2;
  case TensorDType::Int32:
    return 4;
  case TensorDType::BFloat16:
    return 2;
  case TensorDType::Float32:
    return 4;
  case TensorDType::Float64:
    return 8;
  }
  return 1;
}

simaai::neat::Mapping map_tensor_storage_raw(const simaai::neat::Tensor& tensor,
                                             simaai::neat::MapMode mode) {
  if (!tensor.storage) {
    return {};
  }
  simaai::neat::Mapping base = tensor.storage->map(mode);
  if (!base.data) {
    return base;
  }
  simaai::neat::Mapping out = std::move(base);
  if (!out.keepalive && tensor.storage) {
    out.keepalive = std::static_pointer_cast<void>(tensor.storage);
  }
  if (tensor.byte_offset != 0) {
    out.data = static_cast<uint8_t*>(out.data) + tensor.byte_offset;
    if (out.size_bytes > static_cast<std::size_t>(tensor.byte_offset)) {
      out.size_bytes -= static_cast<std::size_t>(tensor.byte_offset);
    } else {
      out.size_bytes = 0U;
    }
  }
  return out;
}

bool holder_debug_enabled() {
  return env_bool("SIMA_INPUTSTREAM_HOLDER_DEBUG", false);
}

bool tensor_map_debug_enabled() {
  return env_bool("SIMA_TENSOR_MAP_DEBUG", false);
}

const char* storage_kind_name_local(simaai::neat::StorageKind kind) {
  switch (kind) {
  case simaai::neat::StorageKind::CpuOwned:
    return "CpuOwned";
  case simaai::neat::StorageKind::CpuExternal:
    return "CpuExternal";
  case simaai::neat::StorageKind::GstSample:
    return "GstSample";
  }
  return "Unknown";
}

void log_holder(const char* tag, GstSample* sample, GstBuffer* buffer) {
  if (!holder_debug_enabled())
    return;
  const int buf_ref = buffer ? GST_MINI_OBJECT_REFCOUNT_VALUE(buffer) : -1;
  std::fprintf(stderr, "[HOLDER] %s sample=%p is_sample=%d buffer=%p buf_ref=%d\n",
               tag ? tag : "event", static_cast<void*>(sample), sample && GST_IS_SAMPLE(sample),
               static_cast<void*>(buffer), buf_ref);
}

GstSample* sample_from_tensor(const simaai::neat::Tensor& tensor) {
  if (!tensor.storage)
    return nullptr;
  if (tensor.storage->kind != simaai::neat::StorageKind::GstSample) {
    if (holder_debug_enabled()) {
      std::fprintf(stderr,
                   "[HOLDER] sample_from_tensor: non-sample storage kind=%s holder=%p shape=%s\n",
                   storage_kind_name_local(tensor.storage->kind),
                   tensor.storage->holder ? tensor.storage->holder.get() : nullptr,
                   tensor.debug_string().c_str());
    }
    return nullptr;
  }
  if (!tensor.storage->holder) {
    if (holder_debug_enabled()) {
      std::fprintf(stderr, "[HOLDER] sample_from_tensor: missing holder for GstSample storage\n");
    }
    return nullptr;
  }
  return static_cast<GstSample*>(tensor.storage->holder.get());
}

struct SimaaiMemoryInfo {
  std::uint64_t target_flags = 0;
  std::uint64_t mem_flags = 0;
  simaai::neat::Device device{simaai::neat::DeviceType::CPU, 0};
};

struct MemorySelection {
  guint index = 0;
  bool auto_selected = false;
};

std::atomic<std::uint64_t> g_tensor_copy_count{0};
std::atomic<std::uint64_t> g_tensor_copy_bytes{0};
std::atomic<std::uint64_t> g_tensor_view_count{0};
std::atomic<std::uint64_t> g_gst_memory_map_calls{0};
std::atomic<std::uint64_t> g_holder_fast_path_hits{0};
std::atomic<std::uint64_t> g_bundle_projection_count{0};
std::atomic<std::uint64_t> g_packed_view_reuse_hits{0};
std::atomic<std::uint64_t> g_packed_view_reuse_opportunities{0};

void update_mem_flags(GstMemory* mem, SimaaiMemoryInfo* info) {
  if (!mem || !info)
    return;
  if (GST_MEMORY_FLAG_IS_SET(mem, GST_SIMAAI_MEMORY_TARGET_EV74)) {
    info->target_flags |= GST_SIMAAI_MEMORY_TARGET_EV74;
  }
  if (GST_MEMORY_FLAG_IS_SET(mem, GST_SIMAAI_MEMORY_TARGET_DMS0)) {
    info->target_flags |= GST_SIMAAI_MEMORY_TARGET_DMS0;
  }
  if (GST_MEMORY_FLAG_IS_SET(mem, GST_SIMAAI_MEMORY_TARGET_DMS1)) {
    info->target_flags |= GST_SIMAAI_MEMORY_TARGET_DMS1;
  }
  if (GST_MEMORY_FLAG_IS_SET(mem, GST_SIMAAI_MEMORY_TARGET_DMS2)) {
    info->target_flags |= GST_SIMAAI_MEMORY_TARGET_DMS2;
  }
  if (GST_MEMORY_FLAG_IS_SET(mem, GST_SIMAAI_MEMORY_TARGET_DMS3)) {
    info->target_flags |= GST_SIMAAI_MEMORY_TARGET_DMS3;
  }
  if (GST_MEMORY_FLAG_IS_SET(mem, GST_SIMAAI_MEMORY_FLAG_CACHED)) {
    info->mem_flags |= GST_SIMAAI_MEMORY_FLAG_CACHED;
  }
  if (GST_MEMORY_FLAG_IS_SET(mem, GST_SIMAAI_MEMORY_FLAG_RDONLY)) {
    info->mem_flags |= GST_SIMAAI_MEMORY_FLAG_RDONLY;
  }
}

std::size_t dense_tensor_physical_span_bytes(const simaai::neat::Tensor& tensor) {
  if (!tensor.is_dense()) {
    return 0U;
  }
  const std::size_t elem_bytes = tensor_dtype_bytes(tensor.dtype);
  if (elem_bytes == 0U) {
    return 0U;
  }
  if (tensor.shape.empty()) {
    return 0U;
  }
  if (tensor.strides_bytes.size() != tensor.shape.size()) {
    return tensor.dense_bytes_tight();
  }

  std::size_t max_offset = 0U;
  for (std::size_t i = 0; i < tensor.shape.size(); ++i) {
    const int64_t dim = tensor.shape[i];
    const int64_t stride = tensor.strides_bytes[i];
    if (dim <= 0 || stride <= 0) {
      return 0U;
    }
    const std::size_t dim_minus_one = static_cast<std::size_t>(dim - 1);
    const std::size_t stride_bytes = static_cast<std::size_t>(stride);
    if (dim_minus_one > 0U &&
        stride_bytes > (std::numeric_limits<std::size_t>::max() / dim_minus_one)) {
      return 0U;
    }
    const std::size_t term = dim_minus_one * stride_bytes;
    if (term > (std::numeric_limits<std::size_t>::max() - max_offset)) {
      return 0U;
    }
    max_offset += term;
  }
  if (elem_bytes > (std::numeric_limits<std::size_t>::max() - max_offset)) {
    return 0U;
  }
  return max_offset + elem_bytes;
}

SimaaiMemoryInfo infer_simaai_memory_info(GstBuffer* buffer) {
  SimaaiMemoryInfo info;
  if (!buffer)
    return info;

  const guint n_mems = gst_buffer_n_memory(buffer);
  for (guint i = 0; i < n_mems; ++i) {
    GstMemory* mem = gst_buffer_peek_memory(buffer, i);
    update_mem_flags(mem, &info);
  }

  if (info.target_flags & GST_SIMAAI_MEMORY_TARGET_EV74) {
    info.device = {simaai::neat::DeviceType::SIMA_CVU, 0};
  } else if (info.target_flags & GST_SIMAAI_MEMORY_TARGET_DMS0) {
    info.device = {simaai::neat::DeviceType::SIMA_MLA, 0};
  } else if (info.target_flags & GST_SIMAAI_MEMORY_TARGET_DMS1) {
    info.device = {simaai::neat::DeviceType::SIMA_MLA, 1};
  } else if (info.target_flags & GST_SIMAAI_MEMORY_TARGET_DMS2) {
    info.device = {simaai::neat::DeviceType::SIMA_MLA, 2};
  } else if (info.target_flags & GST_SIMAAI_MEMORY_TARGET_DMS3) {
    info.device = {simaai::neat::DeviceType::SIMA_MLA, 3};
  }

  return info;
}

MemorySelection select_sample_memory_index(GstBuffer* buffer, int memory_index) {
  MemorySelection selection;
  selection.auto_selected = (memory_index < 0);
  if (!buffer) {
    return selection;
  }
  const guint n_mems = gst_buffer_n_memory(buffer);
  if (n_mems == 0) {
    return selection;
  }

  guint index = static_cast<guint>(memory_index);
  if (selection.auto_selected || index >= n_mems) {
    index = 0;
  }
  selection.index = index;
  return selection;
}

std::shared_ptr<void> make_sample_holder(GstSample* sample) {
  if (!sample)
    return {};
  const bool dbg = holder_debug_enabled();
  return std::shared_ptr<void>(gst_sample_ref(sample), [dbg](void* p) {
    GstSample* s = static_cast<GstSample*>(p);
    GstBuffer* b = s ? gst_sample_get_buffer(s) : nullptr;
    const int buf_ref = b ? GST_MINI_OBJECT_REFCOUNT_VALUE(b) : -1;
    void* buf_pool = b ? static_cast<void*>(b->pool) : nullptr;
    if (dbg) {
      std::fprintf(stderr, "[HOLDER] release sample=%p buffer=%p buf_ref=%d pool=%p\n", p,
                   static_cast<void*>(b), buf_ref, buf_pool);
    }
    gst_sample_unref(s);
  });
}

std::shared_ptr<simaai::neat::Storage>
make_gst_sample_memory_storage(GstSample* sample, guint memory_index, int route_memory_index,
                               std::shared_ptr<void> shared_holder = {}) {
  if (!sample)
    return {};
  GstBuffer* buffer = gst_sample_get_buffer(sample);
  if (!buffer)
    return {};
  GstMemory* mem = gst_buffer_peek_memory(buffer, memory_index);
  if (!mem)
    return {};

  auto storage = std::make_shared<simaai::neat::Storage>();
  storage->kind = simaai::neat::StorageKind::GstSample;
  storage->holder = shared_holder ? std::move(shared_holder) : make_sample_holder(sample);
  storage->data = nullptr;

  gsize offset = 0;
  gsize maxsize = 0;
  storage->size_bytes = gst_memory_get_sizes(mem, &offset, &maxsize);
  SimaaiMemoryInfo mem_info;
  update_mem_flags(mem, &mem_info);
  storage->device = mem_info.device;
  storage->sima_mem_target_flags = mem_info.target_flags;
  storage->sima_mem_flags = mem_info.mem_flags;
  storage->sima_segments = extract_runtime_segments_from_buffer(buffer);

  const bool use_segment_view =
      gst_buffer_n_memory(buffer) == 1U && buffer_memory_uses_segment_allocator(mem) &&
      route_memory_index >= 0 &&
      static_cast<std::size_t>(route_memory_index) < storage->sima_segments.size();
  if (use_segment_view) {
    const auto& segment = storage->sima_segments[static_cast<std::size_t>(route_memory_index)];
    SimaMemApi& api = sima_mem_api();
    auto* segment_mem = static_cast<simaai_memory_t*>(
        gst_simaai_memory_get_segment(mem, segment.name.empty() ? nullptr : segment.name.c_str()));
    if (segment_mem && api.map && api.unmap) {
      struct SegmentMapState {
        bool mapped = false;
        std::mutex mu;
      };
      auto map_state = std::make_shared<SegmentMapState>();
      auto holder = storage->holder;
      const std::size_t segment_size = segment.size_bytes;
      storage->size_bytes = segment_size > 0U ? segment_size : storage->size_bytes;
      storage->map_fn = [holder, map_state, segment_mem, segment_size, invalidate = api.invalidate,
                         map = api.map, unmap = api.unmap](simaai::neat::MapMode mode) {
        if (mode != simaai::neat::MapMode::Read && mode != simaai::neat::MapMode::ReadWrite &&
            mode != simaai::neat::MapMode::Write) {
          return simaai::neat::Mapping{};
        }
        std::lock_guard<std::mutex> lock(map_state->mu);
        if (map_state->mapped) {
          return simaai::neat::Mapping{};
        }
        bool needs_cpu_invalidate = false;
        if (mode == simaai::neat::MapMode::Read || mode == simaai::neat::MapMode::ReadWrite) {
          GstSample* sample = static_cast<GstSample*>(holder.get());
          GstBuffer* buffer =
              (sample && GST_IS_SAMPLE(sample)) ? gst_sample_get_buffer(sample) : nullptr;
          needs_cpu_invalidate = buffer_cpu_read_requires_invalidate(buffer);
        }
        if (needs_cpu_invalidate && invalidate) {
          invalidate(segment_mem);
        }
        void* base = map ? map(segment_mem) : nullptr;
        if (!base) {
          return simaai::neat::Mapping{};
        }
        map_state->mapped = true;
        simaai::neat::Mapping mapping;
        mapping.data = base;
        mapping.size_bytes = segment_size;
        mapping.keepalive = holder;
        mapping.unmap = [map_state, segment_mem, unmap]() {
          std::lock_guard<std::mutex> lock(map_state->mu);
          if (!map_state->mapped) {
            return;
          }
          if (unmap) {
            unmap(segment_mem);
          }
          map_state->mapped = false;
        };
        return mapping;
      };
      return storage;
    }
  }

  struct GstMemoryMapState {
    GstMapInfo info{};
    bool mapped = false;
    std::mutex mu;
  };
  auto map_state = std::make_shared<GstMemoryMapState>();
  auto holder = storage->holder;
  storage->map_fn = [holder, map_state, memory_index](simaai::neat::MapMode mode) {
    GstSample* s = static_cast<GstSample*>(holder.get());
    GstBuffer* b = s ? gst_sample_get_buffer(s) : nullptr;
    if (!b)
      return simaai::neat::Mapping{};
    GstMemory* mem = gst_buffer_peek_memory(b, memory_index);
    if (!mem)
      return simaai::neat::Mapping{};

    std::lock_guard<std::mutex> lock(map_state->mu);
    if (map_state->mapped) {
      return simaai::neat::Mapping{};
    }

    GstMapFlags flags = GST_MAP_READ;
    if (mode == simaai::neat::MapMode::Write) {
      flags = GST_MAP_WRITE;
    } else if (mode == simaai::neat::MapMode::ReadWrite) {
      flags = static_cast<GstMapFlags>(GST_MAP_READ | GST_MAP_WRITE);
    }
    if (mode == simaai::neat::MapMode::Read || mode == simaai::neat::MapMode::ReadWrite) {
      invalidate_cached_buffer(b);
    }
    ++g_gst_memory_map_calls;
    if (!gst_memory_map(mem, &map_state->info, flags)) {
      return simaai::neat::Mapping{};
    }
    map_state->mapped = true;
    GstBuffer* buffer_ref = gst_buffer_ref(b);

    simaai::neat::Mapping mapping;
    mapping.data = map_state->info.data;
    mapping.size_bytes = map_state->info.size;
    mapping.keepalive = holder;
    mapping.unmap = [buffer_ref, map_state, memory_index]() {
      std::lock_guard<std::mutex> lock(map_state->mu);
      if (map_state->mapped) {
        GstMemory* mapped_mem = gst_buffer_peek_memory(buffer_ref, memory_index);
        if (mapped_mem) {
          gst_memory_unmap(mapped_mem, &map_state->info);
        }
        map_state->mapped = false;
      }
      gst_buffer_unref(buffer_ref);
    };
    return mapping;
  };
  return storage;
}

} // namespace

const char* storage_kind_name(simaai::neat::StorageKind kind) {
  switch (kind) {
  case simaai::neat::StorageKind::CpuOwned:
    return "CpuOwned";
  case simaai::neat::StorageKind::CpuExternal:
    return "CpuExternal";
  case simaai::neat::StorageKind::GstSample:
    return "GstSample";
  }
  return "Unknown";
}

std::shared_ptr<simaai::neat::Storage>
make_gst_sample_storage_impl(GstSample* sample,
                             const std::vector<simaai::neat::Segment>* cached_segments) {
  if (!sample)
    return {};
  GstBuffer* buffer = gst_sample_get_buffer(sample);
  if (!buffer)
    return {};

  log_holder("make_storage", sample, buffer);
  auto holder = make_sample_holder(sample);
  struct GstMapState {
    GstMapInfo info{};
    bool mapped = false;
    std::mutex mu;
  };
  auto map_state = std::make_shared<GstMapState>();

  auto storage = std::make_shared<simaai::neat::Storage>();
  storage->kind = simaai::neat::StorageKind::GstSample;
  const SimaaiMemoryInfo mem_info = infer_simaai_memory_info(buffer);
  storage->device = mem_info.device;
  storage->size_bytes = gst_buffer_get_size(buffer);
  storage->holder = holder;
  storage->data = nullptr;
  storage->sima_mem_target_flags = mem_info.target_flags;
  storage->sima_mem_flags = mem_info.mem_flags;
  storage->sima_segments = (cached_segments && !cached_segments->empty())
                               ? *cached_segments
                               : extract_runtime_segments_from_buffer(buffer);
  const std::uint64_t mem_flags = mem_info.mem_flags;
  const std::uint64_t target_flags = mem_info.target_flags;
  storage->map_fn = [holder, map_state, mem_flags, target_flags](simaai::neat::MapMode mode) {
    GstSample* s = static_cast<GstSample*>(holder.get());
    GstBuffer* b = s ? gst_sample_get_buffer(s) : nullptr;
    if (!b)
      return simaai::neat::Mapping{};

    std::lock_guard<std::mutex> lock(map_state->mu);
    if (map_state->mapped) {
      return simaai::neat::Mapping{};
    }
    GstMapFlags flags = GST_MAP_READ;
    if (mode == simaai::neat::MapMode::Write) {
      flags = GST_MAP_WRITE;
    } else if (mode == simaai::neat::MapMode::ReadWrite) {
      flags = static_cast<GstMapFlags>(GST_MAP_READ | GST_MAP_WRITE);
    }
    if (mode == simaai::neat::MapMode::Read || mode == simaai::neat::MapMode::ReadWrite) {
      invalidate_cached_buffer(b);
    }
    if (!gst_buffer_map(b, &map_state->info, flags)) {
      return simaai::neat::Mapping{};
    }
    map_state->mapped = true;
    GstBuffer* buffer_ref = gst_buffer_ref(b);

    simaai::neat::Mapping mapping;
    mapping.data = map_state->info.data;
    mapping.size_bytes = map_state->info.size;
    mapping.unmap = [buffer_ref, map_state]() {
      std::lock_guard<std::mutex> lock(map_state->mu);
      if (map_state->mapped) {
        gst_buffer_unmap(buffer_ref, &map_state->info);
        map_state->mapped = false;
      }
      gst_buffer_unref(buffer_ref);
    };
    return mapping;
  };
  return storage;
}

std::shared_ptr<simaai::neat::Storage> make_gst_sample_storage(GstSample* sample) {
  return make_gst_sample_storage_impl(sample, nullptr);
}

std::shared_ptr<simaai::neat::Storage>
make_gst_sample_storage_with_segments(GstSample* sample,
                                      const std::vector<simaai::neat::Segment>& segments) {
  return make_gst_sample_storage_impl(sample, &segments);
}

TensorIoStats snapshot_tensor_io_stats() {
  TensorIoStats stats;
  stats.tensor_copy_count = g_tensor_copy_count.load(std::memory_order_relaxed);
  stats.tensor_copy_bytes = g_tensor_copy_bytes.load(std::memory_order_relaxed);
  stats.tensor_view_count = g_tensor_view_count.load(std::memory_order_relaxed);
  stats.gst_memory_map_calls = g_gst_memory_map_calls.load(std::memory_order_relaxed);
  stats.holder_fast_path_hits = g_holder_fast_path_hits.load(std::memory_order_relaxed);
  stats.bundle_projection_count = g_bundle_projection_count.load(std::memory_order_relaxed);
  stats.packed_view_reuse_hits = g_packed_view_reuse_hits.load(std::memory_order_relaxed);
  stats.packed_view_reuse_opportunities =
      g_packed_view_reuse_opportunities.load(std::memory_order_relaxed);
  return stats;
}

void reset_tensor_io_stats() {
  g_tensor_copy_count.store(0, std::memory_order_relaxed);
  g_tensor_copy_bytes.store(0, std::memory_order_relaxed);
  g_tensor_view_count.store(0, std::memory_order_relaxed);
  g_gst_memory_map_calls.store(0, std::memory_order_relaxed);
  g_holder_fast_path_hits.store(0, std::memory_order_relaxed);
  g_bundle_projection_count.store(0, std::memory_order_relaxed);
  g_packed_view_reuse_hits.store(0, std::memory_order_relaxed);
  g_packed_view_reuse_opportunities.store(0, std::memory_order_relaxed);
}

void record_tensor_holder_fast_path_hit() {
  g_holder_fast_path_hits.fetch_add(1, std::memory_order_relaxed);
}

void record_tensor_bundle_projection(std::size_t field_count) {
  if (field_count == 0U) {
    return;
  }
  g_bundle_projection_count.fetch_add(1, std::memory_order_relaxed);
}

void record_tensor_packed_view_reuse(std::size_t logical_output_count,
                                     std::size_t unique_memory_count, bool materialize_output) {
  if (logical_output_count == 0U || unique_memory_count == 0U ||
      logical_output_count <= unique_memory_count) {
    return;
  }
  const std::uint64_t reused =
      static_cast<std::uint64_t>(logical_output_count - unique_memory_count);
  g_packed_view_reuse_opportunities.fetch_add(reused, std::memory_order_relaxed);
  if (!materialize_output) {
    g_packed_view_reuse_hits.fetch_add(reused, std::memory_order_relaxed);
  }
}

std::string resolve_tensor_runtime_segment_name(const simaai::neat::Tensor& tensor) {
  if (!tensor.storage) {
    return tensor.route.segment_name;
  }
  if (tensor.route.memory_index >= 0 &&
      static_cast<std::size_t>(tensor.route.memory_index) < tensor.storage->sima_segments.size()) {
    const auto& segment =
        tensor.storage->sima_segments[static_cast<std::size_t>(tensor.route.memory_index)];
    if (!segment.name.empty()) {
      return segment.name;
    }
  }
  return tensor.route.segment_name;
}

std::size_t resolve_tensor_runtime_segment_size(const simaai::neat::Tensor& tensor,
                                                const std::string& segment_name) {
  if (!tensor.storage) {
    return 0U;
  }
  if (!segment_name.empty()) {
    const auto it = std::find_if(
        tensor.storage->sima_segments.begin(), tensor.storage->sima_segments.end(),
        [&](const simaai::neat::Segment& segment) { return segment.name == segment_name; });
    if (it != tensor.storage->sima_segments.end()) {
      return it->size_bytes;
    }
  }
  if (tensor.route.memory_index >= 0 &&
      static_cast<std::size_t>(tensor.route.memory_index) < tensor.storage->sima_segments.size()) {
    return tensor.storage->sima_segments[static_cast<std::size_t>(tensor.route.memory_index)]
        .size_bytes;
  }
  return tensor.storage->size_bytes;
}

simaai::neat::Mapping map_tensor_from_buffer_memory(GstBuffer* buffer, guint memory_index,
                                                    std::size_t offset, std::size_t size,
                                                    const std::shared_ptr<void>& keepalive) {
  if (!buffer) {
    return {};
  }
  GstMemory* memory = gst_buffer_peek_memory(buffer, memory_index);
  if (!memory) {
    return {};
  }

  invalidate_cached_buffer(buffer);

  auto map_state = std::make_shared<GstMapInfo>();
  std::memset(map_state.get(), 0, sizeof(GstMapInfo));
  ++g_gst_memory_map_calls;
  if (!gst_memory_map(memory, map_state.get(), GST_MAP_READ)) {
    return {};
  }

  if (offset > map_state->size) {
    gst_memory_unmap(memory, map_state.get());
    return {};
  }

  if (size == 0U || offset + size > map_state->size) {
    size = map_state->size - offset;
  }

  GstBuffer* buffer_ref = gst_buffer_ref(buffer);
  simaai::neat::Mapping mapping;
  mapping.data = static_cast<uint8_t*>(map_state->data) + offset;
  mapping.size_bytes = size;
  mapping.keepalive = keepalive;
  mapping.unmap = [buffer_ref, map_state, memory_index]() {
    GstMemory* mapped_memory = gst_buffer_peek_memory(buffer_ref, memory_index);
    if (mapped_memory) {
      gst_memory_unmap(mapped_memory, map_state.get());
    }
    gst_buffer_unref(buffer_ref);
  };
  return mapping;
}

simaai::neat::Mapping map_tensor_view(const simaai::neat::Tensor& tensor,
                                      simaai::neat::MapMode mode) {
  if (!tensor.storage)
    return {};
  if (mode != simaai::neat::MapMode::Read) {
    return map_tensor_storage_raw(tensor, mode);
  }
  if (tensor.storage->kind != simaai::neat::StorageKind::GstSample) {
    return map_tensor_storage_raw(tensor, mode);
  }

  auto* sample = static_cast<GstSample*>(tensor.storage->holder.get());
  if (!sample || !GST_IS_SAMPLE(sample)) {
    return map_tensor_storage_raw(tensor, mode);
  }
  GstBuffer* buffer = gst_sample_get_buffer(sample);
  if (!buffer) {
    return map_tensor_storage_raw(tensor, mode);
  }
  if (!buffer_uses_simaai_backing(buffer)) {
    return map_tensor_storage_raw(tensor, mode);
  }
  const bool sample_cpu_prepared = gst_sample_cpu_read_prepared(sample);
  if (buffer_cpu_read_requires_invalidate(buffer)) {
    // Correctness fallback: prepare the full SiMa backing before exposing a
    // CPU pointer. Some EV/MLA outputs have a logical tensor segment smaller
    // than the allocation/backing that the cache-maintenance API tracks, so a
    // narrow segment-only invalidate can still leave stale pooled bytes visible
    // on A65. InputStream may do this earlier to hide the cost, but Tensor::map
    // must do it here as the final guard.
    (void)prepare_gst_sample_for_cpu_read(sample);
  }
  if (env_bool("SIMA_TENSOR_DISABLE_DIRECT_SEGMENT_MAP", false)) {
    return map_tensor_storage_raw(tensor, mode);
  }
  if (tensor.storage->map_fn && tensor.storage->size_bytes > 0 &&
      tensor.storage->size_bytes != gst_buffer_get_size(buffer)) {
    return map_tensor_storage_raw(tensor, mode);
  }

  SimaMemApi& api = sima_mem_api();
  if (!api.attach || !api.map || !api.unmap) {
    return map_tensor_storage_raw(tensor, mode);
  }

  if (gst_buffer_n_memory(buffer) == 1U) {
    const GstMemory* root_memory = gst_buffer_peek_memory(buffer, 0U);
    const std::string segment_name = resolve_tensor_runtime_segment_name(tensor);
    if (root_memory && buffer_memory_uses_segment_allocator(root_memory) && !segment_name.empty()) {
      auto* segment_mem = reinterpret_cast<simaai_memory_t*>(
          gst_simaai_memory_get_segment(root_memory, segment_name.c_str()));
      if (segment_mem) {
        std::size_t size = resolve_tensor_runtime_segment_size(tensor, segment_name);
        const std::size_t required_bytes = dense_tensor_physical_span_bytes(tensor);
        std::size_t offset =
            (tensor.byte_offset > 0) ? static_cast<std::size_t>(tensor.byte_offset) : 0U;
        if (size == 0U && tensor.storage) {
          size = tensor.storage->size_bytes;
        }
        if (offset > size) {
          return map_tensor_storage_raw(tensor, mode);
        }
        if (offset > 0U) {
          size -= offset;
        }
        if (required_bytes > 0U && required_bytes < size) {
          size = required_bytes;
        }
        const bool needs_cpu_invalidate = buffer_cpu_read_requires_invalidate(buffer);
        if (tensor_map_debug_enabled() && tensor.route.logical_index == 0) {
          const CpuVisibleBufferState dbg_state = cpu_visible_buffer_state(buffer);
          std::fprintf(
              stderr,
              "[tensor-map][debug] visibility logical=%d needs_invalidate=%d sample_prepared=%d "
              "has_producer=%d producer=%u has_dirty=%d dirty=%d buffer=%p\n",
              tensor.route.logical_index, needs_cpu_invalidate ? 1 : 0, sample_cpu_prepared ? 1 : 0,
              dbg_state.has_producer ? 1 : 0, dbg_state.producer, dbg_state.has_cpu_dirty ? 1 : 0,
              dbg_state.cpu_dirty ? 1 : 0, static_cast<void*>(buffer));
        }
        if (needs_cpu_invalidate && api.invalidate) {
          // Segment-backed CPU reads must invalidate the segment backing, not
          // only the logical tensor span; the allocator/MPK backing may be
          // larger than the exposed binding bytes and cache maintenance is
          // cache-line based.
          api.invalidate(segment_mem);
        }
        void* base = api.map(segment_mem);
        if (base) {
          if (tensor_map_debug_enabled() && tensor.route.logical_index == 0) {
            const auto* bytes = static_cast<const uint8_t*>(base);
            const std::size_t sample_count = std::min<std::size_t>(8U, size);
            std::fprintf(
                stderr,
                "[tensor-map][debug] segment-path logical=%d segment=%s offset=%zu size=%zu head=",
                tensor.route.logical_index, segment_name.c_str(), offset, size);
            for (std::size_t bi = 0; bi < sample_count; ++bi) {
              if (bi > 0U) {
                std::fprintf(stderr, ",");
              }
              std::fprintf(stderr, "%u", static_cast<unsigned>(bytes[offset + bi]));
            }
            std::fprintf(stderr, "\n");
          }
          simaai::neat::Mapping mapping;
          mapping.data = static_cast<uint8_t*>(base) + offset;
          mapping.size_bytes = size;
          mapping.keepalive = tensor.storage->holder
                                  ? tensor.storage->holder
                                  : std::static_pointer_cast<void>(tensor.storage);
          mapping.unmap = [segment_mem, unmap = api.unmap]() {
            if (unmap) {
              unmap(segment_mem);
            }
          };
          return mapping;
        }
      }
      if (tensor_map_debug_enabled() && tensor.route.logical_index == 0) {
        std::fprintf(stderr, "[tensor-map][debug] segment-path-miss logical=%d segment=%s\n",
                     tensor.route.logical_index, segment_name.c_str());
      }
    }
  }

  if (tensor.route.memory_index >= 0) {
    const guint memory_count = gst_buffer_n_memory(buffer);
    if (memory_count > 1U && static_cast<guint>(tensor.route.memory_index) < memory_count) {
      std::size_t size =
          resolve_tensor_runtime_segment_size(tensor, resolve_tensor_runtime_segment_name(tensor));
      const std::size_t required_bytes = dense_tensor_physical_span_bytes(tensor);
      std::size_t offset =
          (tensor.byte_offset > 0) ? static_cast<std::size_t>(tensor.byte_offset) : 0U;
      if (size == 0U && tensor.storage) {
        size = tensor.storage->size_bytes;
      }
      if (offset <= size) {
        size -= offset;
      } else {
        size = 0U;
      }
      if (required_bytes > 0U && required_bytes < size) {
        size = required_bytes;
      }
      return map_tensor_from_buffer_memory(
          buffer, static_cast<guint>(tensor.route.memory_index), offset, size,
          tensor.storage->holder ? tensor.storage->holder
                                 : std::static_pointer_cast<void>(tensor.storage));
    }
  }

  const uint64_t buffer_id = resolve_buffer_id(buffer);
  if (buffer_id == 0) {
    return map_tensor_storage_raw(tensor, mode);
  }

  simaai_memory_t* mem = get_or_attach_memory(buffer_id);
  if (!mem) {
    return map_tensor_storage_raw(tensor, mode);
  }

  std::size_t size = gst_buffer_get_size(buffer);
  const std::size_t required_bytes = dense_tensor_physical_span_bytes(tensor);

  std::size_t offset = (tensor.byte_offset > 0) ? static_cast<std::size_t>(tensor.byte_offset)
                                                : resolve_buffer_offset(buffer);
  if (offset > size) {
    return tensor.map(mode);
  }
  if (offset > 0) {
    size -= offset;
  }
  if (required_bytes > 0U && required_bytes < size) {
    size = required_bytes;
  }

  if (buffer_cpu_read_requires_invalidate(buffer) && api.invalidate) {
    api.invalidate(mem);
  }

  void* base = api.map(mem);
  if (!base) {
    return map_tensor_storage_raw(tensor, mode);
  }
  if (tensor_map_debug_enabled() && tensor.route.logical_index == 0) {
    const auto* bytes = static_cast<const uint8_t*>(base);
    const std::size_t sample_count = std::min<std::size_t>(8U, size);
    std::fprintf(
        stderr,
        "[tensor-map][debug] attach-path logical=%d buffer_id=0x%llx offset=%zu size=%zu head=",
        tensor.route.logical_index, static_cast<unsigned long long>(buffer_id), offset, size);
    for (std::size_t bi = 0; bi < sample_count; ++bi) {
      if (bi > 0U) {
        std::fprintf(stderr, ",");
      }
      std::fprintf(stderr, "%u", static_cast<unsigned>(bytes[offset + bi]));
    }
    std::fprintf(stderr, "\n");
  }

  simaai::neat::Mapping mapping;
  mapping.data = static_cast<uint8_t*>(base) + offset;
  mapping.size_bytes = size;
  mapping.keepalive = tensor.storage->holder ? tensor.storage->holder
                                             : std::static_pointer_cast<void>(tensor.storage);
  mapping.unmap = [mem, unmap = api.unmap]() {
    if (unmap)
      unmap(mem);
  };
  return mapping;
}

GstBuffer* buffer_from_tensor_holder(const std::shared_ptr<void>& holder) {
  if (!holder)
    return nullptr;
  auto* sample = static_cast<GstSample*>(holder.get());
  if (!sample || !GST_IS_SAMPLE(sample)) {
    if (holder_debug_enabled()) {
      std::fprintf(stderr, "[HOLDER] buffer_from_holder invalid sample=%p\n",
                   static_cast<void*>(sample));
    }
    return nullptr;
  }
  GstBuffer* buffer = gst_sample_get_buffer(sample);
  log_holder("buffer_from_holder", sample, buffer);
  if (!buffer)
    return nullptr;
  return gst_buffer_ref(buffer);
}

bool drop_tensor_holder(const simaai::neat::Tensor& tensor) {
  if (!tensor.storage)
    return false;
  if (tensor.storage->kind != simaai::neat::StorageKind::GstSample)
    return false;
  if (tensor.storage.use_count() > 1) {
    if (holder_debug_enabled()) {
      std::fprintf(stderr,
                   "[HOLDER] skip drop shared storage use_count=%ld logical=%d physical=%d\n",
                   static_cast<long>(tensor.storage.use_count()), tensor.route.logical_index,
                   tensor.route.physical_index);
    }
    return false;
  }
  bool cleared = false;
  if (tensor.storage->holder) {
    tensor.storage->holder.reset();
    cleared = true;
  }
  if (tensor.storage->map_fn) {
    tensor.storage->map_fn = {};
    cleared = true;
  }
  return cleared;
}

simaai::neat::Tensor copy_tensor_from_sample_memory(const simaai::neat::Tensor& ref,
                                                    int memory_index, bool keep_holder) {
  GstSample* sample = sample_from_tensor(ref);
  if (!sample) {
    if (holder_debug_enabled()) {
      const auto* st = ref.storage.get();
      std::fprintf(stderr,
                   "[HOLDER] copy_tensor_from_sample_memory: sample missing kind=%s holder=%p "
                   "mem_index=%d tensor=%s\n",
                   st ? storage_kind_name(st->kind) : "<null>",
                   (st && st->holder) ? st->holder.get() : nullptr, memory_index,
                   ref.debug_string().c_str());
    }
    throw std::runtime_error("copy_tensor_from_sample_memory: missing sample holder");
  }

  GstBuffer* buffer = gst_sample_get_buffer(sample);
  if (!buffer) {
    if (holder_debug_enabled()) {
      std::fprintf(stderr,
                   "[HOLDER] copy_tensor_from_sample_memory: sample has no GstBuffer sample=%p "
                   "mem_index=%d\n",
                   static_cast<void*>(sample), memory_index);
    }
    throw std::runtime_error("copy_tensor_from_sample_memory: missing buffer");
  }

  // If this is cached device memory, invalidate before CPU mapping to avoid
  // reading stale contents.
  invalidate_cached_buffer(buffer);

  const guint n_mems = gst_buffer_n_memory(buffer);
  if (n_mems == 0) {
    throw std::runtime_error("copy_tensor_from_sample_memory: buffer has no memories");
  }

  const MemorySelection selection = select_sample_memory_index(buffer, memory_index);
  const guint index = selection.index;

  if (holder_debug_enabled()) {
    std::fprintf(stderr,
                 "[HOLDER] copy_tensor_from_sample_memory: n_mems=%u requested=%d auto=%d "
                 "selected=%u keep_holder=%d\n",
                 static_cast<unsigned>(n_mems), memory_index, selection.auto_selected ? 1 : 0,
                 static_cast<unsigned>(index), keep_holder ? 1 : 0);
    for (guint i = 0; i < n_mems; ++i) {
      GstMemory* mem_i = gst_buffer_peek_memory(buffer, i);
      gsize offset_i = 0;
      gsize max_i = 0;
      const gsize size_i = mem_i ? gst_memory_get_sizes(mem_i, &offset_i, &max_i) : 0;
      std::uint64_t mem_target_flags = 0;
      if (mem_i && GST_MEMORY_FLAG_IS_SET(mem_i, GST_SIMAAI_MEMORY_TARGET_EV74)) {
        mem_target_flags |= GST_SIMAAI_MEMORY_TARGET_EV74;
      }
      if (mem_i && GST_MEMORY_FLAG_IS_SET(mem_i, GST_SIMAAI_MEMORY_TARGET_DMS0)) {
        mem_target_flags |= GST_SIMAAI_MEMORY_TARGET_DMS0;
      }
      if (mem_i && GST_MEMORY_FLAG_IS_SET(mem_i, GST_SIMAAI_MEMORY_TARGET_DMS1)) {
        mem_target_flags |= GST_SIMAAI_MEMORY_TARGET_DMS1;
      }
      if (mem_i && GST_MEMORY_FLAG_IS_SET(mem_i, GST_SIMAAI_MEMORY_TARGET_DMS2)) {
        mem_target_flags |= GST_SIMAAI_MEMORY_TARGET_DMS2;
      }
      if (mem_i && GST_MEMORY_FLAG_IS_SET(mem_i, GST_SIMAAI_MEMORY_TARGET_DMS3)) {
        mem_target_flags |= GST_SIMAAI_MEMORY_TARGET_DMS3;
      }
      std::fprintf(stderr, "[HOLDER]   mem[%u] size=%zu offset=%zu max=%zu target_flags=0x%llx\n",
                   static_cast<unsigned>(i), static_cast<size_t>(size_i),
                   static_cast<size_t>(offset_i), static_cast<size_t>(max_i),
                   static_cast<unsigned long long>(mem_target_flags));
    }
  }

  GstMemory* mem = gst_buffer_peek_memory(buffer, index);
  if (!mem) {
    throw std::runtime_error("copy_tensor_from_sample_memory: missing buffer memory");
  }

  GstMapInfo map{};
  ++g_gst_memory_map_calls;
  if (!gst_memory_map(mem, &map, GST_MAP_READ)) {
    throw std::runtime_error("copy_tensor_from_sample_memory: gst_memory_map failed");
  }

  auto storage = simaai::neat::make_cpu_owned_storage(map.size);
  simaai::neat::Mapping dst_map = storage->map(simaai::neat::MapMode::Write);
  if (map.size > 0 && dst_map.data) {
    std::memcpy(dst_map.data, map.data, map.size);
  }
  ++g_tensor_copy_count;
  g_tensor_copy_bytes.fetch_add(static_cast<std::uint64_t>(map.size), std::memory_order_relaxed);
  gst_memory_unmap(mem, &map);

  simaai::neat::Tensor out;
  out.storage = std::move(storage);
  if (keep_holder && ref.storage && ref.storage->holder) {
    out.storage->holder =
        std::shared_ptr<void>(new CompositeHolder{out.storage->holder, ref.storage->holder},
                              [](void* p) { delete static_cast<CompositeHolder*>(p); });
  } else if (!keep_holder) {
    out.storage->holder = make_preprocess_meta_holder(buffer);
  }
  out.dtype = ref.dtype;
  out.device = {simaai::neat::DeviceType::CPU, 0};
  out.read_only = false;
  out.semantic = ref.semantic;
  out.byte_offset = ref.byte_offset;

  const std::size_t elem = tensor_dtype_bytes(out.dtype);
  if (!ref.shape.empty()) {
    // Preserve logical tensor contract from negotiated caps/meta. Flattening to
    // a 1D byte-derived shape breaks downstream expectations for HWC/CHW tensors.
    out.shape = ref.shape;
    out.strides_bytes = ref.strides_bytes;
  }

  return out;
}

simaai::neat::Tensor tensor_view_from_sample_memory(const simaai::neat::Tensor& ref,
                                                    int memory_index, bool keep_holder) {
  GstSample* sample = sample_from_tensor(ref);
  if (!sample) {
    throw std::runtime_error("tensor_view_from_sample_memory: missing sample holder");
  }
  GstBuffer* buffer = gst_sample_get_buffer(sample);
  if (!buffer) {
    throw std::runtime_error("tensor_view_from_sample_memory: missing buffer");
  }
  const guint n_mems = gst_buffer_n_memory(buffer);
  if (n_mems == 0) {
    throw std::runtime_error("tensor_view_from_sample_memory: buffer has no memories");
  }

  if (!keep_holder) {
    // Zero-copy views require the underlying GstSample lifetime.
    keep_holder = true;
  }
  std::shared_ptr<void> shared_holder;
  if (keep_holder && ref.storage && ref.storage->kind == simaai::neat::StorageKind::GstSample &&
      ref.storage->holder) {
    shared_holder = ref.storage->holder;
  }
  const MemorySelection selection = select_sample_memory_index(buffer, memory_index);
  const guint index = selection.index;
  auto storage =
      make_gst_sample_memory_storage(sample, index, memory_index, std::move(shared_holder));
  if (!storage) {
    throw std::runtime_error("tensor_view_from_sample_memory: missing buffer memory");
  }
  if (ref.storage && !ref.storage->sima_segments.empty()) {
    storage->sima_segments = ref.storage->sima_segments;
  }

  simaai::neat::Tensor out = ref;
  out.storage = std::move(storage);
  out.device = out.storage->device;
  out.read_only = true;
  out.byte_offset = ref.byte_offset;
  std::size_t route_index = static_cast<std::size_t>(index);
  if (memory_index >= 0 &&
      static_cast<std::size_t>(memory_index) < out.storage->sima_segments.size()) {
    route_index = static_cast<std::size_t>(memory_index);
  }
  out.route.memory_index = static_cast<int>(route_index);
  out.route.physical_index = static_cast<int>(route_index);
  if (route_index < out.storage->sima_segments.size()) {
    if (out.storage->sima_segments[route_index].size_bytes > 0U) {
      out.storage->size_bytes = out.storage->sima_segments[route_index].size_bytes;
    }
    if (!out.storage->sima_segments[route_index].name.empty()) {
      out.route.segment_name = out.storage->sima_segments[route_index].name;
    }
  }
  ++g_tensor_view_count;
  record_tensor_holder_fast_path_hit();
  return out;
}

std::shared_ptr<void> holder_from_tensor(const simaai::neat::Tensor& tensor) {
  if (!tensor.storage)
    return nullptr;
  return tensor.storage->holder;
}

bool copy_into_simaai_segment_memory(void* segment, const void* src, std::size_t bytes,
                                     std::string* err) {
  if (!segment || !src || bytes == 0U) {
    if (err) {
      *err = "copy_into_simaai_segment_memory: invalid argument";
    }
    return false;
  }
  SimaMemApi& api = sima_mem_api();
  if (!api.map || !api.unmap) {
    if (err) {
      *err = "copy_into_simaai_segment_memory: simaai map/unmap unavailable";
    }
    return false;
  }
  auto* seg = static_cast<simaai_memory_t*>(segment);
  void* dst = api.map(seg);
  if (!dst) {
    if (err) {
      *err = "copy_into_simaai_segment_memory: simaai_memory_map failed";
    }
    return false;
  }
  std::memcpy(dst, src, bytes);
  if (api.flush) {
    api.flush(seg);
  }
  api.unmap(seg);
  return true;
}

} // namespace simaai::neat::pipeline_internal
