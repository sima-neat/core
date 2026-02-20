// src/pipeline/TensorUtil.cpp
#include "pipeline/internal/TensorUtil.h"

#include "pipeline/internal/GstDiagnosticsUtil.h"
#include "pipeline/internal/SimaaiMemory.h"

#include <gst/gst.h>

#include <dlfcn.h>

#include <cstring>
#include <memory>
#include <mutex>
#include <stdexcept>
#include <unordered_map>

namespace simaai::neat::pipeline_internal {
namespace {

struct simaai_memory;
using simaai_memory_t = simaai_memory;

struct SimaMemApi {
  using AttachFn = simaai_memory_t* (*)(uint64_t);
  using InvalidateFn = void (*)(simaai_memory_t*);
  using InvalidatePartFn = void (*)(simaai_memory_t*, unsigned int, unsigned int);
  using MapFn = void* (*)(simaai_memory_t*);
  using UnmapFn = void (*)(simaai_memory_t*);
  using GetPhysFn = guintptr (*)(GstMemory*);

  AttachFn attach = nullptr;
  InvalidateFn invalidate = nullptr;
  InvalidatePartFn invalidate_part = nullptr;
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

void invalidate_cached_buffer(GstBuffer* buffer) {
  if (!buffer)
    return;
  SimaMemApi& api = sima_mem_api();
  if (!api.invalidate)
    return;

  const uint64_t buffer_id = resolve_buffer_id(buffer);
  if (buffer_id == 0)
    return;
  simaai_memory_t* mem = get_or_attach_memory(buffer_id);
  if (!mem)
    return;
  api.invalidate(mem);
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

bool holder_debug_enabled() {
  const char* v = std::getenv("SIMA_INPUTSTREAM_HOLDER_DEBUG");
  return v && std::strcmp(v, "0") != 0;
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
  if (tensor.storage->kind != simaai::neat::StorageKind::GstSample)
    return nullptr;
  if (!tensor.storage->holder)
    return nullptr;
  return static_cast<GstSample*>(tensor.storage->holder.get());
}

struct SimaaiMemoryInfo {
  std::uint64_t target_flags = 0;
  std::uint64_t mem_flags = 0;
  simaai::neat::Device device{simaai::neat::DeviceType::CPU, 0};
};

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

} // namespace

std::shared_ptr<simaai::neat::Storage> make_gst_sample_storage(GstSample* sample) {
  if (!sample)
    return {};
  GstBuffer* buffer = gst_sample_get_buffer(sample);
  if (!buffer)
    return {};

  log_holder("make_storage", sample, buffer);
  const bool dbg = holder_debug_enabled();
  auto holder = std::shared_ptr<void>(gst_sample_ref(sample), [dbg](void* p) {
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
    // Always attempt to invalidate caches before CPU mapping. This is a no-op
    // when the buffer isn't backed by simaai memory or the API isn't present.
    invalidate_cached_buffer(b);
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

simaai::neat::Mapping map_tensor_view(const simaai::neat::Tensor& tensor,
                                      simaai::neat::MapMode mode) {
  if (!tensor.storage)
    return {};
  if (mode != simaai::neat::MapMode::Read) {
    return tensor.map(mode);
  }
  if (tensor.storage->kind != simaai::neat::StorageKind::GstSample) {
    return tensor.map(mode);
  }

  auto* sample = static_cast<GstSample*>(tensor.storage->holder.get());
  if (!sample || !GST_IS_SAMPLE(sample)) {
    return tensor.map(mode);
  }
  GstBuffer* buffer = gst_sample_get_buffer(sample);
  if (!buffer) {
    return tensor.map(mode);
  }

  SimaMemApi& api = sima_mem_api();
  if (!api.attach || !api.map || !api.unmap) {
    return tensor.map(mode);
  }

  const uint64_t buffer_id = resolve_buffer_id(buffer);
  if (buffer_id == 0) {
    return tensor.map(mode);
  }

  simaai_memory_t* mem = get_or_attach_memory(buffer_id);
  if (!mem) {
    return tensor.map(mode);
  }

  std::size_t size = gst_buffer_get_size(buffer);
  if (tensor.is_dense()) {
    const std::size_t dense_bytes = tensor.dense_bytes_tight();
    if (dense_bytes > 0 && dense_bytes < size) {
      size = dense_bytes;
    }
  }

  std::size_t offset = (tensor.byte_offset > 0) ? static_cast<std::size_t>(tensor.byte_offset)
                                                : resolve_buffer_offset(buffer);
  if (offset > size) {
    return tensor.map(mode);
  }
  if (offset > 0) {
    size -= offset;
  }

  if (api.invalidate_part && size > 0) {
    api.invalidate_part(mem, static_cast<unsigned int>(offset), static_cast<unsigned int>(size));
  } else if (api.invalidate) {
    api.invalidate(mem);
  }

  void* base = api.map(mem);
  if (!base) {
    return tensor.map(mode);
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
    throw std::runtime_error("copy_tensor_from_sample_memory: missing sample holder");
  }

  GstBuffer* buffer = gst_sample_get_buffer(sample);
  if (!buffer) {
    throw std::runtime_error("copy_tensor_from_sample_memory: missing buffer");
  }

  // If this is cached device memory, invalidate before CPU mapping to avoid
  // reading stale contents.
  invalidate_cached_buffer(buffer);

  const guint n_mems = gst_buffer_n_memory(buffer);
  if (n_mems == 0) {
    throw std::runtime_error("copy_tensor_from_sample_memory: buffer has no memories");
  }

  guint index = static_cast<guint>(memory_index);
  if (memory_index < 0 || index >= n_mems) {
    index = 0;
  }
  if (n_mems > 1 && index == 0) {
    guint best_index = index;
    gsize best_size = 0;
    bool best_is_target = false;
    for (guint i = 0; i < n_mems; ++i) {
      GstMemory* mem = gst_buffer_peek_memory(buffer, i);
      if (!mem)
        continue;
      gsize offset = 0;
      gsize maxsize = 0;
      const gsize size = gst_memory_get_sizes(mem, &offset, &maxsize);
      const bool is_target = GST_MEMORY_FLAG_IS_SET(mem, GST_SIMAAI_MEMORY_TARGET_EV74) ||
                             GST_MEMORY_FLAG_IS_SET(mem, GST_SIMAAI_MEMORY_TARGET_DMS0) ||
                             GST_MEMORY_FLAG_IS_SET(mem, GST_SIMAAI_MEMORY_TARGET_DMS1) ||
                             GST_MEMORY_FLAG_IS_SET(mem, GST_SIMAAI_MEMORY_TARGET_DMS2) ||
                             GST_MEMORY_FLAG_IS_SET(mem, GST_SIMAAI_MEMORY_TARGET_DMS3);
      if (is_target && (!best_is_target || size > best_size)) {
        best_is_target = true;
        best_size = size;
        best_index = i;
      } else if (!best_is_target && size > best_size) {
        best_size = size;
        best_index = i;
      }
    }
    index = best_index;
  }

  GstMemory* mem = gst_buffer_peek_memory(buffer, index);
  if (!mem) {
    throw std::runtime_error("copy_tensor_from_sample_memory: missing buffer memory");
  }

  GstMapInfo map{};
  if (!gst_memory_map(mem, &map, GST_MAP_READ)) {
    throw std::runtime_error("copy_tensor_from_sample_memory: gst_memory_map failed");
  }

  auto storage = simaai::neat::make_cpu_owned_storage(map.size);
  simaai::neat::Mapping dst_map = storage->map(simaai::neat::MapMode::Write);
  if (map.size > 0 && dst_map.data) {
    std::memcpy(dst_map.data, map.data, map.size);
  }
  gst_memory_unmap(mem, &map);

  simaai::neat::Tensor out;
  out.storage = std::move(storage);
  if (keep_holder && ref.storage && ref.storage->holder) {
    out.storage->holder = ref.storage->holder;
  }
  out.dtype = ref.dtype;
  out.device = {simaai::neat::DeviceType::CPU, 0};
  out.read_only = false;
  out.semantic = ref.semantic;
  out.byte_offset = 0;

  const std::size_t elem = tensor_dtype_bytes(out.dtype);
  if (elem > 0 && (map.size % elem) == 0) {
    const std::size_t elems = map.size / elem;
    out.shape = {static_cast<int64_t>(elems)};
    out.strides_bytes = {static_cast<int64_t>(elem)};
  }

  return out;
}

std::shared_ptr<void> holder_from_tensor(const simaai::neat::Tensor& tensor) {
  if (!tensor.storage)
    return nullptr;
  return tensor.storage->holder;
}

} // namespace simaai::neat::pipeline_internal
