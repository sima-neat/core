#pragma once
#ifndef SIMA_NEAT_INTERNAL
#error "Internal header. Not part of the public API."
#endif

#include <gst/gst.h>

#include <cstddef>
#include <cstdint>
#include <dlfcn.h>

// Single source of truth for the lazily-resolved SiMa cache-maintenance + coherency
// entry points, shared by CpuVisibleSample.cpp and TensorUtil.cpp (previously
// duplicated). The inline accessor's function-local static is one instance across
// all TUs (ODR), so the dlsym resolution happens exactly once process-wide.
namespace simaai::neat::pipeline_internal {

struct simaai_memory;
using simaai_memory_t = simaai_memory;

struct SimaMemApi {
  using AttachFn = simaai_memory_t* (*)(uint64_t);
  using InvalidateFn = void (*)(simaai_memory_t*);
  using InvalidatePartFn = void (*)(simaai_memory_t*, unsigned int, unsigned int);
  using FlushFn = void (*)(simaai_memory_t*);
  using MapFn = void* (*)(simaai_memory_t*);
  using UnmapFn = void (*)(simaai_memory_t*);
  using GetSizeFn = std::size_t (*)(simaai_memory_t*);
  using GetPhysFn = guintptr (*)(GstMemory*);
  // Allocator's ownership-skip-gated CPU-read authority; resolved at runtime from
  // the global scope (RTLD_DEFAULT) and OPTIONAL — null against an allocator that
  // predates it, in which case callers fall back to a plain invalidate.
  using CpuReadBeginFn = void (*)(GstMemory*);

  AttachFn attach = nullptr;
  InvalidateFn invalidate = nullptr;
  InvalidatePartFn invalidate_part = nullptr;
  FlushFn flush = nullptr;
  MapFn map = nullptr;
  UnmapFn unmap = nullptr;
  GetSizeFn get_size = nullptr;
  GetPhysFn get_phys = nullptr;
  CpuReadBeginFn cpu_read_begin = nullptr;
  bool tried = false;
};

inline SimaMemApi& sima_mem_api() {
  static SimaMemApi api;
  if (api.tried) {
    return api;
  }
  api.tried = true;

  // The coherency authority lives in the loaded allocator plugin, not
  // libsimaaimem — resolve it from the global scope.
  api.cpu_read_begin = reinterpret_cast<SimaMemApi::CpuReadBeginFn>(
      dlsym(RTLD_DEFAULT, "gst_simaai_segment_memory_cpu_read_begin"));

  void* handle = dlopen("libsimaaimem.so", RTLD_LAZY | RTLD_LOCAL);
  if (!handle) {
    handle = dlopen("libsimaaimem.so.1", RTLD_LAZY | RTLD_LOCAL);
  }
  if (!handle) {
    return api;
  }
  api.attach = reinterpret_cast<SimaMemApi::AttachFn>(dlsym(handle, "simaai_memory_attach"));
  api.invalidate =
      reinterpret_cast<SimaMemApi::InvalidateFn>(dlsym(handle, "simaai_memory_invalidate_cache"));
  api.invalidate_part = reinterpret_cast<SimaMemApi::InvalidatePartFn>(
      dlsym(handle, "simaai_memory_invalidate_cache_part"));
  api.flush = reinterpret_cast<SimaMemApi::FlushFn>(dlsym(handle, "simaai_memory_flush_cache"));
  api.map = reinterpret_cast<SimaMemApi::MapFn>(dlsym(handle, "simaai_memory_map"));
  api.unmap = reinterpret_cast<SimaMemApi::UnmapFn>(dlsym(handle, "simaai_memory_unmap"));
  api.get_size =
      reinterpret_cast<SimaMemApi::GetSizeFn>(dlsym(handle, "simaai_memory_get_size"));
  api.get_phys =
      reinterpret_cast<SimaMemApi::GetPhysFn>(dlsym(handle, "gst_simaai_memory_get_phys_addr"));
  return api;
}

}  // namespace simaai::neat::pipeline_internal
