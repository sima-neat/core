#include "test_utils.h"

#include "gst/GstInit.h"

#include "pipeline/internal/SimaaiGstCompat.h"

#include <gst/gst.h>

#include <dlfcn.h>

#include <cstdint>
#include <cstdlib>
#include <iostream>
#include <string>

namespace {

struct SimaaiAllocatorApi {
  void (*memory_init_once)(void) = nullptr;
  GstAllocator* (*memory_get_allocator)(void) = nullptr;
  void (*buffer_memory_init_once)(void) = nullptr;
  GstAllocator* (*get_allocator)(void) = nullptr;
  void (*segment_memory_init_once)(void) = nullptr;
  GstAllocator* (*memory_get_segment_allocator)(void) = nullptr;
  void (*memory_allocation_params_init)(GstSimaaiAllocationParams*) = nullptr;
  gboolean (*memory_allocation_params_add_segment)(GstSimaaiAllocationParams*, gsize,
                                                   const gchar*) = nullptr;

  bool has_contiguous() const {
    return (memory_init_once && memory_get_allocator) || (buffer_memory_init_once && get_allocator);
  }

  bool has_segment() const {
    return segment_memory_init_once && memory_get_segment_allocator &&
           memory_allocation_params_init && memory_allocation_params_add_segment;
  }
};

const SimaaiAllocatorApi& simaai_allocator_api() {
  static const SimaaiAllocatorApi api = []() {
    SimaaiAllocatorApi out;
    auto sym = [](const char* name) -> void* { return dlsym(RTLD_DEFAULT, name); };

    out.memory_init_once = reinterpret_cast<void (*)(void)>(sym("gst_simaai_memory_init_once"));
    out.memory_get_allocator =
        reinterpret_cast<GstAllocator* (*)(void)>(sym("gst_simaai_memory_get_allocator"));
    out.buffer_memory_init_once =
        reinterpret_cast<void (*)(void)>(sym("gst_simaai_buffer_memory_init_once"));
    out.get_allocator = reinterpret_cast<GstAllocator* (*)(void)>(sym("gst_simaai_get_allocator"));
    out.segment_memory_init_once =
        reinterpret_cast<void (*)(void)>(sym("gst_simaai_segment_memory_init_once"));
    out.memory_get_segment_allocator =
        reinterpret_cast<GstAllocator* (*)(void)>(sym("gst_simaai_memory_get_segment_allocator"));
    out.memory_allocation_params_init = reinterpret_cast<void (*)(GstSimaaiAllocationParams*)>(
        sym("gst_simaai_memory_allocation_params_init"));
    out.memory_allocation_params_add_segment =
        reinterpret_cast<gboolean (*)(GstSimaaiAllocationParams*, gsize, const gchar*)>(
            sym("gst_simaai_memory_allocation_params_add_segment"));
    return out;
  }();
  return api;
}

struct AllocatorHolder {
  GstAllocator* alloc = nullptr;
  ~AllocatorHolder() {
    if (alloc) {
      gst_object_unref(alloc);
    }
  }
  explicit operator bool() const {
    return alloc != nullptr;
  }
};

AllocatorHolder get_simaai_allocator_with_share() {
  const auto& api = simaai_allocator_api();
  if (!api.has_contiguous()) {
    return {};
  }
  AllocatorHolder holder;
  if (api.memory_init_once && api.memory_get_allocator) {
    api.memory_init_once();
    holder.alloc = api.memory_get_allocator();
  } else if (api.buffer_memory_init_once && api.get_allocator) {
    api.buffer_memory_init_once();
    holder.alloc = api.get_allocator();
  }

  if (holder.alloc && holder.alloc->mem_share) {
    return holder;
  }
  if (holder.alloc) {
    gst_object_unref(holder.alloc);
    holder.alloc = nullptr;
  }
  return holder;
}

AllocatorHolder get_simaai_segment_allocator_with_share() {
  const auto& api = simaai_allocator_api();
  if (!api.has_segment()) {
    return {};
  }
  api.segment_memory_init_once();
  AllocatorHolder holder;
  holder.alloc = api.memory_get_segment_allocator();
  if (holder.alloc && holder.alloc->mem_share) {
    return holder;
  }
  if (holder.alloc) {
    gst_object_unref(holder.alloc);
    holder.alloc = nullptr;
  }
  return holder;
}

void fill_incrementing(GstMemory* mem) {
  GstMapInfo info;
  require(gst_memory_map(mem, &info, GST_MAP_WRITE), "map write failed");
  for (gsize i = 0; i < info.size; ++i) {
    info.data[i] = static_cast<guint8>(i & 0xff);
  }
  gst_memory_unmap(mem, &info);
}

void require_read_only(GstMemory* mem) {
  GstMapInfo info;
  if (gst_memory_map(mem, &info, GST_MAP_WRITE)) {
    gst_memory_unmap(mem, &info);
    throw std::runtime_error("expected read-only shared memory");
  }
}

void require_map_pattern(GstMemory* mem, gsize expected_size, guint8 expected_first,
                         guint8 expected_last) {
  GstMapInfo info;
  require(gst_memory_map(mem, &info, GST_MAP_READ), "map read failed");
  require(info.size == expected_size, "size mismatch");
  require(info.size > 0, "empty mapped size");
  require(info.data[0] == expected_first, "first byte mismatch");
  require(info.data[info.size - 1] == expected_last, "last byte mismatch");
  gst_memory_unmap(mem, &info);
}

void require_mem_share(GstMemory* mem, const char* label) {
  require(mem != nullptr, "memory missing");
  require(mem->allocator != nullptr, std::string(label) + " allocator missing");
  require(mem->allocator->mem_share != nullptr, std::string(label) + " mem_share missing");
}

bool run_contiguous_share_tests() {
  AllocatorHolder alloc = get_simaai_allocator_with_share();
  if (!alloc) {
    return false;
  }

  constexpr gsize kSize = 128;
  GstAllocationParams params;
  gst_allocation_params_init(&params);
  GstMemory* mem = gst_allocator_alloc(alloc.alloc, kSize, &params);
  if (!mem) {
    return false;
  }
  require_mem_share(mem, "Simaai allocator");

  fill_incrementing(mem);

  GstMemory* full = gst_memory_share(mem, 0, static_cast<gsize>(-1));
  require(full != nullptr, "full share failed");
  require_map_pattern(full, kSize, 0, static_cast<guint8>((kSize - 1) & 0xff));
  gst_memory_unref(full);

  GstMemory* share1 = gst_memory_share(mem, 16, 32);
  require(share1 != nullptr, "share1 failed");
  require_map_pattern(share1, 32, 16, 47);
  require_read_only(share1);

  GstMemory* share2 = gst_memory_share(share1, 8, 8);
  require(share2 != nullptr, "share2 failed");
  require_map_pattern(share2, 8, 24, 31);
  require_read_only(share2);

  require(gst_memory_share(mem, -1, 8) == nullptr, "negative offset share should fail");
  require(gst_memory_share(mem, kSize - 4, 8) == nullptr, "out-of-range share should fail");

  gst_memory_unref(mem);
  require_map_pattern(share1, 32, 16, 47);

  gst_memory_unref(share2);
  gst_memory_unref(share1);
  return true;
}

bool run_segment_share_tests() {
  const auto& api = simaai_allocator_api();
  AllocatorHolder alloc = get_simaai_segment_allocator_with_share();
  if (!alloc) {
    return false;
  }

  constexpr gsize kSegA = 64;
  constexpr gsize kSegB = 64;
  constexpr gsize kTotal = kSegA + kSegB;

  GstSimaaiAllocationParams seg_params;
  api.memory_allocation_params_init(&seg_params);
  seg_params.parent.flags = static_cast<GstMemoryFlags>(GST_SIMAAI_MEMORY_FLAG_DEFAULT);

  if (!api.memory_allocation_params_add_segment(&seg_params, kSegA, "segA") ||
      !api.memory_allocation_params_add_segment(&seg_params, kSegB, "segB")) {
    return false;
  }

  GstMemory* mem =
      gst_allocator_alloc(alloc.alloc, kTotal, reinterpret_cast<GstAllocationParams*>(&seg_params));
  if (!mem) {
    return false;
  }
  require_mem_share(mem, "Simaai segment allocator");

  fill_incrementing(mem);

  GstMemory* share1 = gst_memory_share(mem, 8, 16);
  require(share1 != nullptr, "segment share1 failed");
  require_map_pattern(share1, 16, 8, 23);
  require_read_only(share1);

  GstMemory* share2 = gst_memory_share(share1, 4, 4);
  require(share2 != nullptr, "segment share2 failed");
  require_map_pattern(share2, 4, 12, 15);
  require_read_only(share2);

  require(gst_memory_share(mem, -2, 4) == nullptr, "segment negative offset share should fail");
  require(gst_memory_share(mem, kTotal - 1, 4) == nullptr,
          "segment out-of-range share should fail");

  gst_memory_unref(mem);
  require_map_pattern(share1, 16, 8, 23);

  gst_memory_unref(share2);
  gst_memory_unref(share1);
  return true;
}

} // namespace

int main(int argc, char** argv) {
  try {
    (void)argc;
    (void)argv;
    simaai::neat::gst_init_once();

    const bool ran_contiguous = run_contiguous_share_tests();
    if (!ran_contiguous) {
      std::cout << "[SKIP] Simaai allocator mem_share not available\n";
    }
    const bool ran_segment = run_segment_share_tests();
    if (!ran_segment) {
      std::cout << "[SKIP] segment allocator not available\n";
    }
    if (!ran_contiguous && !ran_segment) {
      std::cout << "[SKIP] Simaai allocator share not available\n";
      return 0;
    }

    std::cout << "[OK] unit_simaai_allocator_share_test passed\n";
    return 0;
  } catch (const std::runtime_error& e) {
    return fail_test(e.what());
  } catch (const std::exception& e) {
    std::cerr << "[FAIL] " << e.what() << "\n";
    return 1;
  }
}
