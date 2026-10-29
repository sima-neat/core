#include "pipeline/internal/SampleUtil.h"
// src/pipeline/TensorUtil.cpp
#include "pipeline/internal/TensorUtil.h"
#include "pipeline/internal/TensorMath.h"
#include "pipeline/GraphOptions.h"

#include <simaai/scheduler/client/memory_access.h>

#include "pipeline/internal/CpuVisibleSample.h"
#include "pipeline/internal/GstDiagnosticsUtil.h"
#include "pipeline/internal/InputStreamUtil.h"
#include "pipeline/internal/EnvUtil.h"
#include "pipeline/internal/SimaaiGstCompat.h"
#include "pipeline/internal/SimaaiMemory.h"
#include <simaai/scheduler/client/dma_memory.h>

#include <gst/gst.h>
#include <gst/allocators/gstdmabuf.h>

#include <dlfcn.h>

#include <algorithm>
#include <cstring>
#include <atomic>
#include <cstdint>
#include <limits>
#include <memory>
#include <stdexcept>
#include <unordered_map>
#include <vector>

namespace simaai::neat::pipeline_internal {
namespace memory_client = ::simaai::scheduler::client;
namespace dma_client = ::simaai::scheduler::client::dma;

namespace {

dma_client::CpuAccess dmabuf_cpu_access(MapMode mode) {
  switch (mode) {
  case MapMode::Read:
    return dma_client::CpuAccess::Read;
  case MapMode::Write:
    // A Tensor can cover only part of a cache line. Preserve neighboring bytes.
    return dma_client::CpuAccess::ReadWrite;
  case MapMode::ReadWrite:
    return dma_client::CpuAccess::ReadWrite;
  }
  return dma_client::CpuAccess::ReadWrite;
}

void finish_segment_access(memory_client::AccessCookie& cookie) noexcept {
  memory_client::MemoryAccessError error;
  if (!memory_client::finish_selected(cookie, error)) {
    g_warning("Tensor DMA-BUF CPU access completion failed: %s (%d)", error.operation, error.code);
  }
}

memory_client::SelectedMemory select_runtime_segment(const GstMemory* memory,
                                                     memory_client::SegmentSelector selector = {}) {
  auto result = memory_client::select_segment(const_cast<GstMemory*>(memory), selector);
  if (auto* selected = std::get_if<memory_client::SelectedMemory>(&result))
    return std::move(*selected);
  return {};
}
memory_client::SelectedMemory select_runtime_segment_at(const GstMemory* memory,
                                                        std::size_t index) {
  return select_runtime_segment(memory, {memory_client::SegmentSelector::Kind::Index, index});
}
memory_client::SelectedMemory select_runtime_segment_named(const GstMemory* memory,
                                                           const char* name) {
  return name && *name
             ? select_runtime_segment(memory, {memory_client::SegmentSelector::Kind::Name, 0, name})
             : select_runtime_segment(memory);
}
memory_client::MemoryDescription
segment_description(const memory_client::SelectedMemory& selected) {
  auto result = memory_client::describe(selected);
  if (const auto* description = std::get_if<memory_client::MemoryDescription>(&result))
    return *description;
  return {};
}
Mapping map_selected_segment(const memory_client::SelectedMemory& selected, MapMode mode,
                             std::size_t offset, std::size_t size, std::shared_ptr<void> holder) {
  memory_client::AccessCookie cookie;
  memory_client::MappedWindow window;
  memory_client::MemoryAccessError error;
  const auto access =
      mode == MapMode::Read ? memory_client::CpuAccess::Read : memory_client::CpuAccess::ReadWrite;
  if (!memory_client::begin_selected(selected, access, {offset, size}, cookie, window, error))
    return {};
  struct Guard {
    memory_client::AccessCookie& cookie;
    ~Guard() {
      finish_segment_access(cookie);
    }
  } guard{cookie};
  Mapping mapping;
  mapping.data = window.data;
  mapping.size_bytes = window.size;
  mapping.keepalive = std::move(holder);
  mapping.unmap = [cookie]() mutable { finish_segment_access(cookie); };
  cookie = {};
  return mapping;
}

struct CompositeHolder {
  std::shared_ptr<void> primary;
  std::shared_ptr<void> keepalive;
};

// Optional range capability for Core-owned storage. Only explicit internal
// logical reads use it; public map/view semantics and TensorBuffer ABI stay
// unchanged. Wrapped callbacks retain their existing mapping contract.
struct DmaBufStorageMapper {
  std::shared_ptr<void> holder;
  guint memory_index;

  Mapping operator()(MapMode mode) const;
  Mapping map_range(MapMode mode, std::size_t offset, std::size_t length) const;
};
std::size_t dense_tensor_physical_span_bytes(const Tensor& tensor);

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
      const gsize buffer_size = gst_buffer_get_size(buffer);
      const gsize segment_count = gst_simaai_memory_get_segment_count(memory);
      segments.reserve(static_cast<std::size_t>(segment_count));
      for (gsize i = 0; i < segment_count; ++i) {
        const gchar* raw_name = gst_simaai_memory_get_segment_name_at(memory, i);
        gsize raw_size = gst_simaai_memory_get_segment_size_at(memory, i);
        if (segment_count == 1U && buffer_size > 0U && raw_size > buffer_size) {
          /*
           * Packed CMA pools expose a single parent DMA-BUF allocation while GstMemory/GstBuffer
           * carries the logical frame view.  Runtime segment metadata must describe the view that
           * will be transported across a graph boundary, not the whole packed parent allocation.
           */
          raw_size = buffer_size;
        }
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
                                             simaai::neat::MapMode mode,
                                             std::size_t logical_read_span) {
  if (!tensor.storage) {
    return {};
  }
  if (tensor.byte_offset < 0) {
    return {};
  }
  Mapping base;
  bool offset_applied = false;
  if (logical_read_span != 0U && mode == MapMode::Read) {
    if (const auto* mapper = tensor.storage->map_fn.target<DmaBufStorageMapper>()) {
      const auto offset = static_cast<std::size_t>(tensor.byte_offset);
      if (offset > tensor.storage->size_bytes ||
          logical_read_span > tensor.storage->size_bytes - offset) {
        return {};
      }
      base = mapper->map_range(mode, offset, logical_read_span);
      offset_applied = true;
    }
  }
  if (!offset_applied) {
    base = tensor.storage->map(mode);
  }
  if (!base.data) {
    return base;
  }
  simaai::neat::Mapping out = std::move(base);
  // A custom mapper may retain the original GstSample, predating the public loan
  // attached to this storage. Keep both guards: Mapping destruction invokes CPU END
  // before releasing keepalive, including the current storage's producer/loan guard.
  out.keepalive = std::make_shared<CompositeHolder>(
      CompositeHolder{std::static_pointer_cast<void>(tensor.storage), out.keepalive});
  if (offset_applied) {
    return out;
  }
  if (static_cast<std::size_t>(tensor.byte_offset) > out.size_bytes) {
    return {};
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
  static const bool enabled = env_bool("SIMA_INPUTSTREAM_HOLDER_DEBUG", false);
  return enabled;
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
  if (tensor.has_tiled_storage())
    return tensor.semantic.tess->storage_bytes;
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

std::size_t find_runtime_segment_index(const std::vector<simaai::neat::Segment>& segments,
                                       const std::string& segment_name) {
  if (segment_name.empty()) {
    return static_cast<std::size_t>(-1);
  }
  for (std::size_t i = 0; i < segments.size(); ++i) {
    if (segments[i].name == segment_name) {
      return i;
    }
  }
  return static_cast<std::size_t>(-1);
}

std::size_t clamp_segment_span_to_live_handle(const memory_client::SelectedMemory& segment_mem,
                                              std::size_t metadata_size) {
  std::size_t span = metadata_size;
  if (segment_mem) {
    const std::size_t live_size = segment_description(segment_mem).authority_bytes;
    if (live_size > 0U && (span == 0U || live_size < span)) {
      span = live_size;
    }
  }
  return span;
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

Mapping DmaBufStorageMapper::operator()(MapMode mode) const {
  // A direct TensorBuffer::map intentionally retains whole-storage semantics.
  return map_range(mode, 0U, 0U);
}

Mapping DmaBufStorageMapper::map_range(MapMode mode, std::size_t offset, std::size_t length) const {
  auto* retained_sample = static_cast<GstSample*>(holder.get());
  GstBuffer* retained_buffer = retained_sample && GST_IS_SAMPLE(retained_sample)
                                   ? gst_sample_get_buffer(retained_sample)
                                   : nullptr;
  if (!retained_buffer || memory_index >= gst_buffer_n_memory(retained_buffer)) {
    return {};
  }
  auto* memory = gst_buffer_peek_memory(retained_buffer, memory_index);
  dma_client::Error error;
  auto view = length != 0U ? dma_client::DmaBufView::fromGstMemory(memory, offset, length, &error)
                           : dma_client::DmaBufView::fromGstMemory(memory, &error);
  if (!view) {
    g_warning("Tensor DMA-BUF view failed: %s (errno=%d offset=%zu bytes=%zu)",
              error.message().c_str(), error.code, offset, length);
    return {};
  }
  // Each public Mapping owns its lower lease. Scheduler/Runner already share
  // compatible readers and exclude writers by allocation identity; a Core-side
  // active-map flag would reject legal aliases and duplicate that authority.
  // Allocate the callback owner before START, so allocation failure cannot
  // strand an admitted access. This is the public compatibility map path, not
  // the allocation-free prepared-stage callback path.
  auto scope = std::make_shared<std::optional<dma_client::CpuMapping>>();
  std::function<void()> finish = [scope]() noexcept {
    if (!scope->has_value())
      return;
    dma_client::Error error;
    if (!scope->value().finish(&error)) {
      g_warning("Tensor DMA-BUF CPU access completion failed: %s (%d)", error.operation.c_str(),
                error.code);
    }
    scope->reset();
  };
  auto mapped = view->map(dmabuf_cpu_access(mode), &error);
  if (!mapped) {
    g_warning("Tensor DMA-BUF CPU mapping failed: %s (errno=%d offset=%zu bytes=%zu)",
              error.message().c_str(), error.code, view->offset(), view->length());
    return {};
  }
  scope->emplace(std::move(*mapped));

  Mapping result;
  result.data = scope->value().data();
  result.size_bytes = scope->value().size();
  result.keepalive = holder;
  result.unmap = std::move(finish);
  return result;
}

void set_dmabuf_map_function(const std::shared_ptr<Storage>& storage, guint memory_index) {
  storage->map_fn = DmaBufStorageMapper{storage->holder, memory_index};
}

std::shared_ptr<simaai::neat::Storage> make_gst_sample_memory_storage(
    GstSample* sample, guint memory_index, int route_memory_index,
    std::shared_ptr<void> shared_holder = {}, const std::string& route_segment_name = {},
    const std::vector<simaai::neat::Segment>* current_segments = nullptr) {
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
  storage->sima_segments = current_segments && !current_segments->empty()
                               ? *current_segments
                               : extract_runtime_segments_from_buffer(buffer);

  if (gst_is_dmabuf_memory(mem)) {
    set_dmabuf_map_function(storage, memory_index);
    return storage;
  }

  std::size_t route_segment_index =
      find_runtime_segment_index(storage->sima_segments, route_segment_name);
  if (route_segment_index == static_cast<std::size_t>(-1) && route_memory_index >= 0 &&
      static_cast<std::size_t>(route_memory_index) < storage->sima_segments.size()) {
    route_segment_index = static_cast<std::size_t>(route_memory_index);
  }
  const bool use_segment_view = gst_buffer_n_memory(buffer) == 1U &&
                                buffer_memory_uses_segment_allocator(mem) &&
                                route_segment_index != static_cast<std::size_t>(-1);
  if (use_segment_view) {
    const auto& segment = storage->sima_segments[route_segment_index];
    const gsize admitted_children = gst_simaai_memory_get_segment_count(mem);
    if (admitted_children > 1U && (offset != 0U || storage->size_bytes != maxsize))
      throw std::runtime_error("tensor buffer binding: a partial GstMemory view of multiple "
                               "Runtime child segments is not supported");
    auto segment_mem = select_runtime_segment_at(mem, route_segment_index);
    if (segment_mem) {
      auto holder = storage->holder;
      const gsize child_count = admitted_children;
      // Segment::mappedAddress already includes the Segment's allocation
      // offset. GstMemory share offsets are additional offsets relative to
      // that address, just as the installed allocator's mem_map_full applies.
      const std::size_t view_offset = child_count == 1U ? offset : 0U;
      const std::size_t live_size = segment_description(segment_mem).authority_bytes;
      if (view_offset > live_size) {
        throw std::runtime_error("tensor buffer binding: GstMemory view exceeds Runtime segment");
      }
      const std::size_t segment_size =
          std::min({segment.size_bytes, live_size - view_offset, storage->size_bytes});
      storage->size_bytes = segment_size;
      storage->map_fn = [holder, segment_mem, segment_size](simaai::neat::MapMode mode) {
        if (mode != simaai::neat::MapMode::Read && mode != simaai::neat::MapMode::ReadWrite &&
            mode != simaai::neat::MapMode::Write) {
          return simaai::neat::Mapping{};
        }
        return map_selected_segment(segment_mem, mode, 0U, segment_size, holder);
      };
      return storage;
    }
  }

  auto holder = storage->holder;
  storage->map_fn = [holder, memory_index](simaai::neat::MapMode mode) {
    GstSample* s = static_cast<GstSample*>(holder.get());
    GstBuffer* b = s ? gst_sample_get_buffer(s) : nullptr;
    if (!b)
      return simaai::neat::Mapping{};
    GstMemory* mem = gst_buffer_peek_memory(b, memory_index);
    if (!mem)
      return simaai::neat::Mapping{};

    GstMapFlags flags = GST_MAP_READ;
    if (mode == simaai::neat::MapMode::Write) {
      flags = GST_MAP_WRITE;
    } else if (mode == simaai::neat::MapMode::ReadWrite) {
      flags = static_cast<GstMapFlags>(GST_MAP_READ | GST_MAP_WRITE);
    }
    // Every Sample copy shares this storage, so fan-out branches map it concurrently. Each
    // mapping owns its GstMapInfo; GStreamer's memory lock refuses conflicting write maps.
    auto info = std::make_shared<GstMapInfo>();
    ++g_gst_memory_map_calls;
    if (!gst_memory_map(mem, info.get(), flags)) {
      return simaai::neat::Mapping{};
    }
    GstBuffer* buffer_ref = gst_buffer_ref(b);

    simaai::neat::Mapping mapping;
    mapping.data = info->data;
    mapping.size_bytes = info->size;
    mapping.keepalive = holder;
    mapping.unmap = [buffer_ref, info, memory_index]() {
      GstMemory* mapped_mem = gst_buffer_peek_memory(buffer_ref, memory_index);
      if (mapped_mem) {
        gst_memory_unmap(mapped_mem, info.get());
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
                             std::vector<simaai::neat::Segment>* current_segments) {
  if (!sample)
    return {};
  GstBuffer* buffer = gst_sample_get_buffer(sample);
  if (!buffer)
    return {};

  log_holder("make_storage", sample, buffer);
  auto holder = make_sample_holder(sample);

  auto storage = std::make_shared<simaai::neat::Storage>();
  storage->kind = simaai::neat::StorageKind::GstSample;
  const SimaaiMemoryInfo mem_info = infer_simaai_memory_info(buffer);
  storage->device = mem_info.device;
  storage->size_bytes = gst_buffer_get_size(buffer);
  storage->holder = holder;
  storage->data = nullptr;
  storage->sima_mem_target_flags = mem_info.target_flags;
  storage->sima_mem_flags = mem_info.mem_flags;
  storage->sima_segments = (current_segments && !current_segments->empty())
                               ? std::move(*current_segments)
                               : extract_runtime_segments_from_buffer(buffer);

  /*
   * Standard DMA-BUF memory is the shared codec/CVU/MLA/A65 transport.  Map it
   * through the existing DmaBufView authority so the actual CPU access is
   * bracketed by DMA_BUF_IOCTL_SYNC START/END.  A raw GstDmaBufAllocator mmap
   * has no ownership transition and can return stale decoder pixels from a
   * recycled pool slot.  Device-only consumers never invoke this map function
   * and therefore pay no CPU synchronization cost.
   */
  if (gst_buffer_n_memory(buffer) == 1U) {
    GstMemory* memory = gst_buffer_peek_memory(buffer, 0U);
    if (memory && gst_is_dmabuf_memory(memory)) {
      set_dmabuf_map_function(storage, 0U);
      return storage;
    }
  }

  const std::uint64_t mem_flags = mem_info.mem_flags;
  const std::uint64_t target_flags = mem_info.target_flags;
  storage->map_fn = [holder, mem_flags, target_flags](simaai::neat::MapMode mode) {
    GstSample* s = static_cast<GstSample*>(holder.get());
    GstBuffer* b = s ? gst_sample_get_buffer(s) : nullptr;
    if (!b)
      return simaai::neat::Mapping{};

    GstMapFlags flags = GST_MAP_READ;
    if (mode == simaai::neat::MapMode::Write) {
      flags = GST_MAP_WRITE;
    } else if (mode == simaai::neat::MapMode::ReadWrite) {
      flags = static_cast<GstMapFlags>(GST_MAP_READ | GST_MAP_WRITE);
    }
    // Concurrent mappings of the shared storage each own their GstMapInfo (see above). A read map
    // refs first: mapping a writable multi-memory buffer merges its memories in place, racing
    // other branches; mapping it read-only merges into a temporary memory instead.
    auto info = std::make_shared<GstMapInfo>();
    const bool read_only = mode == simaai::neat::MapMode::Read;
    GstBuffer* buffer_ref = read_only ? gst_buffer_ref(b) : nullptr;
    if (!gst_buffer_map(b, info.get(), flags)) {
      if (buffer_ref)
        gst_buffer_unref(buffer_ref);
      return simaai::neat::Mapping{};
    }
    if (!buffer_ref)
      buffer_ref = gst_buffer_ref(b);

    simaai::neat::Mapping mapping;
    mapping.data = info->data;
    mapping.size_bytes = info->size;
    mapping.unmap = [buffer_ref, info]() {
      gst_buffer_unmap(buffer_ref, info.get());
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
                                      std::vector<simaai::neat::Segment> segments) {
  return make_gst_sample_storage_impl(sample, &segments);
}

TensorBufferBindingResult tensor_buffer_current_segments(
    GstBuffer* buffer, const std::vector<simaai::neat::Segment>& admitted_segments,
    std::vector<simaai::neat::Segment>* current_segments, std::string* err) {
  const auto fail = [&](const char* message) {
    if (err) {
      *err = message;
    }
    return TensorBufferBindingResult::Invalid;
  };
  if (!buffer || !current_segments) {
    return fail("tensor buffer binding: missing current buffer or segment output");
  }
  const guint memory_count = gst_buffer_n_memory(buffer);
  if (memory_count == 0U) {
    return fail("tensor buffer binding: current buffer has no GstMemory");
  }
  // Runtime may flatten children of multiple GstMemory parents. Core's present
  // storage mapper supports plain multiple memories or one named-child parent;
  // do not confuse a flattened child selector with a GstMemory ordinal.
  for (guint i = 0; i < memory_count; ++i) {
    const auto* memory = gst_buffer_peek_memory(buffer, i);
    if (!memory) {
      return fail("tensor buffer binding: current buffer has a null GstMemory");
    }
    if (memory_count > 1U && buffer_memory_uses_segment_allocator(memory) &&
        gst_simaai_memory_get_segment_count(memory) > 0U) {
      return fail("tensor buffer binding: multiple GstMemory parents with Runtime child segments "
                  "are not supported");
    }
  }

  const GstMemory* parent = gst_buffer_peek_memory(buffer, 0U);
  const gsize child_count = memory_count == 1U && buffer_memory_uses_segment_allocator(parent)
                                ? gst_simaai_memory_get_segment_count(parent)
                                : 0U;
  const std::size_t count = child_count > 0U ? child_count : memory_count;
  if (child_count > 1U && (parent->offset != 0U || parent->size != parent->maxsize)) {
    return fail("tensor buffer binding: a partial GstMemory view of multiple Runtime child "
                "segments is not supported");
  }
  if (admitted_segments.size() != count) {
    if (err) {
      *err = "tensor buffer binding: admitted segment selectors do not match current topology";
    }
    return TensorBufferBindingResult::IncompatibleCarrierCount;
  }
  const bool refresh_in_place = current_segments == &admitted_segments;
  if (!refresh_in_place) {
    current_segments->clear();
    current_segments->reserve(count);
  }
  for (std::size_t i = 0; i < count; ++i) {
    std::size_t size = 0U;
    std::string name;
    if (refresh_in_place) {
      name = std::move((*current_segments)[i].name);
    } else {
      name = admitted_segments[i].name;
    }
    if (child_count > 0U) {
      const gchar* raw_name = gst_simaai_memory_get_segment_name_at(parent, i);
      if (raw_name && *raw_name) {
        name = raw_name;
      }
      size = gst_simaai_memory_get_segment_size_at(parent, i);
      const auto live_segment = select_runtime_segment_at(parent, i);
      if (!live_segment) {
        return fail("tensor buffer binding: current Runtime child handle is missing");
      }
      const std::size_t view_offset = child_count == 1U ? parent->offset : 0U;
      if (view_offset > segment_description(live_segment).authority_bytes) {
        return fail("tensor buffer binding: GstMemory view exceeds Runtime segment");
      }
      size = std::min(size, segment_description(live_segment).authority_bytes - view_offset);
      // Preserve the existing shared/packed single-root visible-span clamp.
      // Multiple independent children have no guessed aggregate offset math.
      if (child_count == 1U) {
        const gsize visible =
            gst_memory_get_sizes(const_cast<GstMemory*>(parent), nullptr, nullptr);
        size = std::min(size, static_cast<std::size_t>(visible));
      }
    } else {
      size = gst_memory_get_sizes(gst_buffer_peek_memory(buffer, static_cast<guint>(i)), nullptr,
                                  nullptr);
    }
    if (name.empty()) {
      name = "memory" + std::to_string(i);
    }
    if (refresh_in_place) {
      (*current_segments)[i] = Segment{std::move(name), size};
    } else {
      current_segments->push_back(Segment{std::move(name), size});
    }
  }
  return TensorBufferBindingResult::Current;
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
  if (!tensor.route.segment_name.empty()) {
    const auto it =
        std::find_if(tensor.storage->sima_segments.begin(), tensor.storage->sima_segments.end(),
                     [&](const simaai::neat::Segment& segment) {
                       return segment.name == tensor.route.segment_name;
                     });
    if (it != tensor.storage->sima_segments.end()) {
      return tensor.route.segment_name;
    }
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

  if (size == 0U || size > map_state->size - offset) {
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

// Both entrypoints share route selection. An optional checked logical span may
// optimize the final storage mapping, never override an earlier named segment.
static simaai::neat::Mapping map_tensor_view_impl(const simaai::neat::Tensor& tensor,
                                                  simaai::neat::MapMode mode,
                                                  std::size_t logical_read_span) {
  if (!tensor.storage || tensor.byte_offset < 0)
    return {};
  if (mode != simaai::neat::MapMode::Read) {
    return map_tensor_storage_raw(tensor, mode, logical_read_span);
  }
  if (tensor.storage->kind != simaai::neat::StorageKind::GstSample) {
    return map_tensor_storage_raw(tensor, mode, logical_read_span);
  }

  auto* sample = static_cast<GstSample*>(tensor.storage->holder.get());
  if (!sample || !GST_IS_SAMPLE(sample)) {
    return map_tensor_storage_raw(tensor, mode, logical_read_span);
  }
  GstBuffer* buffer = gst_sample_get_buffer(sample);
  if (!buffer) {
    return map_tensor_storage_raw(tensor, mode, logical_read_span);
  }
  if (!buffer_uses_simaai_backing(buffer)) {
    return map_tensor_storage_raw(tensor, mode, logical_read_span);
  }
  if (env_bool("SIMA_TENSOR_DISABLE_DIRECT_SEGMENT_MAP", false)) {
    return map_tensor_storage_raw(tensor, mode, logical_read_span);
  }
  if (tensor.storage->map_fn && tensor.storage->size_bytes > 0 &&
      tensor.storage->size_bytes != gst_buffer_get_size(buffer)) {
    return map_tensor_storage_raw(tensor, mode, logical_read_span);
  }

  if (gst_buffer_n_memory(buffer) == 1U) {
    const GstMemory* root_memory = gst_buffer_peek_memory(buffer, 0U);
    if (root_memory && tensor.storage->map_fn &&
        (root_memory->offset != 0U || root_memory->size != root_memory->maxsize) &&
        buffer_memory_uses_segment_allocator(root_memory)) {
      // A visible window has an authored mapper with checked offset/bounds.
      // The direct root shortcut must not replace that current binding.
      return map_tensor_storage_raw(tensor, mode, logical_read_span);
    }
    const std::string segment_name = resolve_tensor_runtime_segment_name(tensor);
    if (root_memory && buffer_memory_uses_segment_allocator(root_memory) && !segment_name.empty()) {
      auto segment_mem = select_runtime_segment_named(root_memory, segment_name.c_str());
      if (segment_mem) {
        std::size_t size = resolve_tensor_runtime_segment_size(tensor, segment_name);
        size = clamp_segment_span_to_live_handle(segment_mem, size);
        const std::size_t required_bytes = dense_tensor_physical_span_bytes(tensor);
        std::size_t offset =
            (tensor.byte_offset > 0) ? static_cast<std::size_t>(tensor.byte_offset) : 0U;
        if (size == 0U && tensor.storage) {
          size = tensor.storage->size_bytes;
        }
        if (offset > size) {
          return map_tensor_storage_raw(tensor, mode, logical_read_span);
        }
        if (offset > 0U) {
          size -= offset;
        }
        if (required_bytes > 0U && required_bytes < size) {
          size = required_bytes;
        }
        if (tensor_map_debug_enabled() && tensor.route.logical_index == 0) {
          const CpuVisibleBufferState dbg_state = cpu_visible_buffer_state(buffer);
          std::fprintf(stderr,
                       "[tensor-map][debug] visibility logical=%d requires_sync=%d "
                       "has_producer=%d producer=%u has_dirty=%d dirty=%d buffer=%p\n",
                       tensor.route.logical_index, dbg_state.requires_sync ? 1 : 0,
                       dbg_state.has_producer ? 1 : 0, dbg_state.producer,
                       dbg_state.has_cpu_dirty ? 1 : 0, dbg_state.cpu_dirty ? 1 : 0,
                       static_cast<void*>(buffer));
        }
        auto keepalive = tensor.storage->holder ? tensor.storage->holder
                                                : std::static_pointer_cast<void>(tensor.storage);
        auto mapping = map_selected_segment(segment_mem, mode, offset, size, std::move(keepalive));
        if (mapping.data && tensor_map_debug_enabled() && tensor.route.logical_index == 0) {
          const auto* bytes = static_cast<const uint8_t*>(mapping.data);
          std::fprintf(
              stderr,
              "[tensor-map][debug] segment-path logical=%d segment=%s offset=%zu size=%zu head=",
              tensor.route.logical_index, segment_name.c_str(), offset, size);
          for (std::size_t bi = 0; bi < std::min<std::size_t>(8U, size); ++bi)
            std::fprintf(stderr, "%s%u", bi ? "," : "", static_cast<unsigned>(bytes[bi]));
          std::fprintf(stderr, "\n");
        }
        return mapping;
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

  // Storage owns the buffer and its checked mapping contract. Never attach a
  // physical address from metadata or retain buffers in a process-global cache.
  return map_tensor_storage_raw(tensor, mode, logical_read_span);
}

simaai::neat::Mapping map_tensor_view(const simaai::neat::Tensor& tensor,
                                      simaai::neat::MapMode mode) {
  return map_tensor_view_impl(tensor, mode, 0U);
}

// Internal shape-bounded consumers request an exact READ without changing the
// public map/view contract. Capability affects maintenance cost, never the
// observable logical extent. No geometry is cached across mutable Tensor views.
simaai::neat::Mapping map_dense_tensor_read(const simaai::neat::Tensor& tensor) {
  if (!tensor.storage || tensor.byte_offset < 0 || !tensor.is_dense() ||
      tensor.has_tiled_storage() ||
      (!tensor.strides_bytes.empty() && tensor.strides_bytes.size() != tensor.shape.size())) {
    return {};
  }
  const auto span = is_bound_dense_scalar(tensor) ? tensor_dtype_bytes(tensor.dtype)
                                                  : dense_tensor_physical_span_bytes(tensor);
  if (span == 0U) {
    return {};
  }
  // Route exactly as public mapping before using any storage-specific range
  // capability. Wrapped/custom callbacks retain their complete authority and
  // cleanup; a wider successful map only narrows this reader's logical view.
  Mapping result = map_tensor_view_impl(tensor, MapMode::Read, span);
  if (!result.data || result.size_bytes < span) {
    return {};
  }
  result.size_bytes = span;
  return result;
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

namespace {

using OwnedBuffer = std::unique_ptr<GstBuffer, decltype(&gst_buffer_unref)>;

struct GstMemoryUnref {
  void operator()(GstMemory* memory) const {
    gst_memory_unref(memory);
  }
};
using OwnedMemory = std::unique_ptr<GstMemory, GstMemoryUnref>;

// Owned outputs are for the application's CPU, so they use the same builtin
// system allocator and page alignment as the Scheduler's terminal HostCpu pool.
// Keeping them out of CMA leaves device memory to the running pipeline.
constexpr gsize kOwnedOutputAlignment = 4096;

OwnedMemory copy_output_storage(const std::shared_ptr<Storage>& source,
                                GstSample* whole_sample = nullptr) {
  GstAllocator* allocator = gst_allocator_find(GST_ALLOCATOR_SYSMEM);
  if (!allocator) {
    throw std::runtime_error("Owned output allocation failed: system allocator unavailable");
  }
  GstAllocationParams params;
  gst_allocation_params_init(&params);
  params.align = kOwnedOutputAlignment - 1;
  // A zero-length logical tensor still needs a valid transport allocation.
  OwnedMemory memory(
      gst_allocator_alloc(allocator, std::max<std::size_t>(source->size_bytes, 1U), &params));
  gst_object_unref(allocator);
  if (!memory) {
    throw std::runtime_error("Owned output allocation failed: " +
                             std::to_string(source->size_bytes) + " bytes of system memory");
  }
  GstMapInfo output = GST_MAP_INFO_INIT;
  if (!gst_memory_map(memory.get(), &output, GST_MAP_WRITE)) {
    throw std::runtime_error("Owned output mapping failed");
  }
  // A whole-buffer view (for example a fragmented encoded access unit) needs
  // contiguous owned storage. Copy fragments directly, without GstBuffer's
  // implicit merge/staging allocation.
  auto* buffer = whole_sample ? gst_sample_get_buffer(whole_sample) : nullptr;
  const guint count = buffer ? gst_buffer_n_memory(buffer) : 1U;
  std::size_t offset = 0;
  try {
    for (guint i = 0; i < count; ++i) {
      auto part = buffer ? make_gst_sample_memory_storage(whole_sample, i, i) : source;
      if (!part || offset > source->size_bytes) {
        throw std::runtime_error("Owned output: invalid source carrier extent");
      }
      auto input = part->map(MapMode::Read);
      if (part->size_bytes > input.size_bytes || part->size_bytes > source->size_bytes - offset ||
          (part->size_bytes && !input.data)) {
        throw std::runtime_error("Owned output: cannot map complete source storage");
      }
      if (part->size_bytes) {
        std::memcpy(output.data + offset, input.data, part->size_bytes);
      }
      offset += part->size_bytes;
    }
  } catch (...) {
    gst_memory_unmap(memory.get(), &output);
    throw;
  }
  gst_memory_unmap(memory.get(), &output);
  if (offset != source->size_bytes) {
    throw std::runtime_error("Owned output: inconsistent source extent");
  }
  ++g_tensor_copy_count;
  g_tensor_copy_bytes.fetch_add(source->size_bytes, std::memory_order_relaxed);
  return memory;
}

// System memory keeps one address for its lifetime, so an Owned tensor can expose
// it through Tensor::data_ptr() while the tensor's sample holds the memory.
void* system_memory_address(GstMemory* memory) {
  GstMapInfo info = GST_MAP_INFO_INIT;
  if (!gst_memory_map(memory, &info, GST_MAP_READ)) {
    throw std::runtime_error("Owned output mapping failed");
  }
  void* address = info.data;
  gst_memory_unmap(memory, &info);
  return address;
}

// A plan belongs to one output operation. It retains source authority until all
// copies complete, then publishes envelopes containing only selected carriers.
// Allocation identity alone is insufficient: aliases reuse a copy only when
// their checked spans match as well.
class OwnedOutputMaterializer {
public:
  explicit OwnedOutputMaterializer(bool preserve_framework)
      : preserve_framework_(preserve_framework) {}

  void apply(Sample& sample) {
    plan(sample);
    for (auto& carrier : carriers_) {
      carrier.destination = copy_output_storage(carrier.source, carrier.whole_sample);
    }
    for (auto& envelope : envelopes_) {
      build_envelope(envelope);
    }
    for (const auto& binding : bindings_) {
      publish(binding);
    }
    set_ownership(sample);
  }

private:
  struct SpanKey {
    const void* live_identity = nullptr;
    std::optional<dma_client::DmaBufIdentity> dma_identity;
    std::size_t offset = 0;
    std::size_t size = 0;

    memory_client::SelectedMemory selected_authority;
    bool operator==(const SpanKey& other) const {
      const bool authority_equal =
          (!selected_authority && !other.selected_authority) ||
          memory_client::same_authority(selected_authority, other.selected_authority);
      return authority_equal && live_identity == other.live_identity &&
             dma_identity == other.dma_identity && offset == other.offset && size == other.size;
    }
  };

  struct Carrier {
    std::vector<SpanKey> spans;
    std::shared_ptr<Storage> source;
    GstSample* whole_sample = nullptr;
    OwnedMemory destination;
  };

  struct EnvelopeSlot {
    std::size_t source_order;
    std::size_t carrier;
    Segment segment;
  };

  struct Envelope {
    const void* identity;
    GstSample* source_sample;
    bool whole_buffer;
    std::vector<EnvelopeSlot> slots;
    std::shared_ptr<void> destination;
    std::vector<Segment> segments;
  };

  struct Binding {
    Tensor* tensor;
    std::size_t envelope;
    std::size_t source_order;
  };

  struct Selection {
    GstSample* sample = nullptr;
    GstBuffer* buffer = nullptr;
    GstMemory* memory = nullptr;
    memory_client::SelectedMemory runtime_segment;
    std::shared_ptr<Storage> source;
    std::size_t source_order = 0;
    bool whole_buffer = false;
    bool dmabuf = false;
  };

  static std::size_t checked_span(const Tensor& tensor, std::size_t element) {
    if (tensor.has_tiled_storage())
      return tensor.semantic.tess->storage_bytes;
    const auto& shape = tensor.shape;
    const auto& strides = tensor.strides_bytes;
    if (shape.empty() && !is_bound_dense_scalar(tensor)) {
      return 0U; // Opaque/encoded storage need not describe a dense tensor.
    }
    if (!strides.empty() && strides.size() != shape.size()) {
      throw std::runtime_error("Owned output: tensor stride rank does not match shape");
    }
    if (element == 0U) {
      throw std::runtime_error("Owned output: tensor element size is zero");
    }
    bool empty = false;
    for (std::size_t i = 0; i < shape.size(); ++i) {
      if (shape[i] < 0 || (!strides.empty() && strides[i] < 0)) {
        throw std::runtime_error("Owned output: negative tensor shape or stride");
      }
      empty = empty || shape[i] == 0;
    }
    if (empty) {
      return 0U;
    }
    std::size_t span = element;
    for (std::size_t i = 0; i < shape.size(); ++i) {
      const auto extent = static_cast<std::size_t>(shape[i]);
      if (strides.empty()) {
        if (extent > std::numeric_limits<std::size_t>::max() / span) {
          throw std::runtime_error("Owned output: tensor span overflow");
        }
        span *= extent;
      } else {
        const auto stride = static_cast<std::size_t>(strides[i]);
        if (extent > 1U &&
            stride > (std::numeric_limits<std::size_t>::max() - span) / (extent - 1U)) {
          throw std::runtime_error("Owned output: tensor span overflow");
        }
        span += (extent - 1U) * stride;
      }
    }
    return span;
  }

  static void validate_binding(const Tensor& tensor, std::size_t size) {
    auto check = [size](int64_t offset, std::size_t span) {
      if (offset < 0 || static_cast<std::size_t>(offset) > size ||
          span > size - static_cast<std::size_t>(offset)) {
        throw std::runtime_error("Owned output: tensor view exceeds selected carrier bounds");
      }
    };
    const std::size_t element = tensor_dtype_bytes(tensor.dtype);
    if (tensor.is_dense()) {
      check(tensor.byte_offset, checked_span(tensor, element));
    } else {
      if (tensor.byte_offset != 0) {
        throw std::runtime_error("Owned output: composite tensor has nonzero byte offset");
      }
      for (const auto& plane : tensor.planes) {
        const std::size_t span =
            plane_span_bytes_checked(plane.shape, plane.strides_bytes, element);
        if (span == 0U) {
          throw std::runtime_error("Owned output: plane bytes could not be computed");
        }
        check(plane.byte_offset, span);
      }
    }
  }

  static Selection resolve(const Tensor& tensor, bool preserve_framework) {
    if (tensor.route.memory_index < -1) {
      throw std::runtime_error("Owned output: invalid negative selected memory index");
    }
    if (!tensor.storage) {
      throw std::runtime_error("Owned output: missing tensor storage");
    }
    Selection selected;
    selected.sample = sample_from_tensor(tensor);
    if (!selected.sample) {
      if (tensor.storage->kind == StorageKind::GstSample) {
        throw std::runtime_error("Owned output: missing GstSample authority");
      }
      selected.source = tensor.storage;
      validate_binding(tensor, selected.source->size_bytes);
      return selected;
    }
    selected.buffer = gst_sample_get_buffer(selected.sample);
    const guint count = selected.buffer ? gst_buffer_n_memory(selected.buffer) : 0U;
    if (count == 0U) {
      throw std::runtime_error("Owned output: missing buffer memory");
    }
    const bool segmented = count == 1U && buffer_memory_uses_segment_allocator(
                                              gst_buffer_peek_memory(selected.buffer, 0U));
    const auto live_segments =
        segmented ? extract_runtime_segments_from_buffer(selected.buffer) : std::vector<Segment>{};
    // An unselected complete storage view denotes the admitted whole payload,
    // including a packed Runtime root with several named logical segments.
    selected.whole_buffer = tensor.route.memory_index < 0 && tensor.route.segment_name.empty() &&
                            (count > 1U || live_segments.size() > 1U) &&
                            tensor.storage->size_bytes == gst_buffer_get_size(selected.buffer);
    if (selected.whole_buffer) {
      selected.source = tensor.storage;
      selected.dmabuf = !segmented;
      for (guint i = 0; i < count; ++i) {
        selected.dmabuf =
            selected.dmabuf && gst_is_dmabuf_memory(gst_buffer_peek_memory(selected.buffer, i));
      }
      if (segmented) {
        selected.memory = gst_buffer_peek_memory(selected.buffer, 0U);
        selected.runtime_segment = select_runtime_segment(selected.memory);
        std::size_t total = 0U;
        for (const auto& segment : live_segments) {
          if (segment.size_bytes > std::numeric_limits<std::size_t>::max() - total) {
            throw std::runtime_error("Owned output: packed Runtime extent overflow");
          }
          total += segment.size_bytes;
        }
        gsize offset = 0U;
        gsize maximum = 0U;
        gst_memory_get_sizes(selected.memory, &offset, &maximum);
        // Runtime's allocator admits this packed root and maps/synchronizes the
        // whole range. Do not invent a layout by concatenating child segments.
        if (!selected.runtime_segment || offset != 0U || total != selected.source->size_bytes ||
            segment_description(selected.runtime_segment).authority_bytes != total) {
          throw std::runtime_error("Owned output: inconsistent packed Runtime whole authority");
        }
      }
      validate_binding(tensor, selected.source->size_bytes);
      return selected;
    }

    const auto& segments = segmented ? live_segments : tensor.storage->sima_segments;
    const auto named = find_runtime_segment_index(segments, tensor.route.segment_name);
    int index = tensor.route.memory_index;
    if (segmented && named != static_cast<std::size_t>(-1)) {
      index = static_cast<int>(named);
    } else if (index < 0 && named < count) {
      index = static_cast<int>(named);
    } else if (index < 0 && count == 1U && (!segmented || live_segments.size() == 1U)) {
      index = 0;
    }
    if (index < 0 || (!segmented && static_cast<guint>(index) >= count) ||
        (segmented && static_cast<std::size_t>(index) >= live_segments.size())) {
      throw std::runtime_error("Owned output: invalid or ambiguous selected memory index");
    }
    selected.source_order = static_cast<std::size_t>(index);
    const guint memory_index = segmented ? 0U : static_cast<guint>(index);
    selected.memory = gst_buffer_peek_memory(selected.buffer, memory_index);
    std::string segment_name;
    if (segmented) {
      segment_name = live_segments[selected.source_order].name;
      selected.runtime_segment =
          select_runtime_segment_named(selected.memory, segment_name.c_str());
      if (!selected.runtime_segment) {
        throw std::runtime_error("Owned output: missing selected Runtime segment");
      }
    }
    selected.dmabuf = gst_is_dmabuf_memory(selected.memory);
    if (preserve_framework &&
        (selected.dmabuf || tensor_has_certified_terminal_host_memory(tensor))) {
      // Use the already admitted output Storage without constructing another
      // map state on the retained hot path. The selected GstMemory, not the
      // enclosing buffer or possibly stale metadata, supplies its live extent.
      selected.source = tensor.storage;
      gsize offset = 0;
      gsize maximum = 0;
      validate_binding(tensor, gst_memory_get_sizes(selected.memory, &offset, &maximum));
    } else {
      selected.source = make_gst_sample_memory_storage(selected.sample, memory_index, index,
                                                       tensor.storage->holder, segment_name);
      if (!selected.source) {
        throw std::runtime_error("Owned output: missing selected memory storage");
      }
      validate_binding(tensor, selected.source->size_bytes);
    }
    return selected;
  }

  static SpanKey memory_key(GstMemory* memory) {
    gsize offset = 0;
    gsize maximum = 0;
    const auto size = static_cast<std::size_t>(gst_memory_get_sizes(memory, &offset, &maximum));
    if (!gst_is_dmabuf_memory(memory)) {
      return {memory, std::nullopt, static_cast<std::size_t>(offset), size};
    }
    dma_client::Error error;
    auto view = dma_client::DmaBufView::fromGstMemory(memory, &error);
    auto identity = view ? view->identity(&error) : std::nullopt;
    if (!identity) {
      throw std::runtime_error("Owned output carrier admission failed: " + error.message());
    }
    return {nullptr, identity, view->offset(), view->length()};
  }

  void plan(Tensor& tensor) {
    const Selection selected = resolve(tensor, preserve_framework_);
    // Retention is per selected payload, not memory zero or its siblings. This
    // path performs no allocation, CPU map, synchronization or identity import.
    if (preserve_framework_ &&
        (selected.dmabuf || tensor_has_certified_terminal_host_memory(tensor))) {
      return;
    }
    std::vector<SpanKey> spans;
    if (selected.whole_buffer && !selected.runtime_segment) {
      for (guint i = 0; i < gst_buffer_n_memory(selected.buffer); ++i) {
        spans.push_back(memory_key(gst_buffer_peek_memory(selected.buffer, i)));
      }
    } else if (selected.runtime_segment) {
      std::size_t offset = segment_description(selected.runtime_segment).allocation_offset;
      if (selected.memory && gst_simaai_memory_get_segment_count(selected.memory) == 1U) {
        gsize view_offset = 0U;
        gst_memory_get_sizes(selected.memory, &view_offset, nullptr);
        if (view_offset > std::numeric_limits<std::size_t>::max() - offset) {
          throw std::runtime_error("Owned output: Runtime view offset overflow");
        }
        offset += view_offset;
      }
      spans.push_back(
          {nullptr, std::nullopt, offset, selected.source->size_bytes, selected.runtime_segment});
    } else if (selected.memory) {
      spans.push_back(memory_key(selected.memory));
    } else {
      spans.push_back({selected.source.get(), std::nullopt, 0U, selected.source->size_bytes});
    }
    auto carrier = std::find_if(carriers_.begin(), carriers_.end(),
                                [&](const Carrier& entry) { return entry.spans == spans; });
    if (carrier == carriers_.end()) {
      carriers_.push_back(
          {std::move(spans),
           selected.source,
           selected.whole_buffer && !selected.runtime_segment ? selected.sample : nullptr,
           {}});
      carrier = std::prev(carriers_.end());
    }
    const std::size_t carrier_index = static_cast<std::size_t>(carrier - carriers_.begin());
    const void* identity = selected.sample ? static_cast<void*>(selected.sample)
                                           : static_cast<void*>(selected.source.get());
    auto envelope = std::find_if(envelopes_.begin(), envelopes_.end(), [&](const Envelope& entry) {
      return entry.identity == identity && entry.whole_buffer == selected.whole_buffer;
    });
    if (envelope == envelopes_.end()) {
      envelopes_.push_back({identity, selected.sample, selected.whole_buffer, {}, {}, {}});
      envelope = std::prev(envelopes_.end());
    }
    // Different source slots may intentionally name aliases of one allocation.
    // Preserve those named slots while sharing their one copied backing memory.
    auto slot = std::find_if(
        envelope->slots.begin(), envelope->slots.end(),
        [&](const EnvelopeSlot& entry) { return entry.source_order == selected.source_order; });
    if (slot == envelope->slots.end()) {
      Segment segment{tensor.route.segment_name, selected.source->size_bytes};
      const auto& segments =
          selected.runtime_segment ? selected.source->sima_segments : tensor.storage->sima_segments;
      if (!selected.whole_buffer && selected.source_order < segments.size()) {
        segment.name = segments[selected.source_order].name;
      }
      if (segment.name.empty()) {
        segment.name = "memory" + std::to_string(selected.source_order);
      }
      envelope->slots.push_back({selected.source_order, carrier_index, std::move(segment)});
    } else if (slot->carrier != carrier_index) {
      throw std::runtime_error("Owned output: inconsistent selected source slot authority");
    }
    bindings_.push_back(
        {&tensor, static_cast<std::size_t>(envelope - envelopes_.begin()), selected.source_order});
  }

  void plan(Sample& sample) {
    if (sample.tensor) {
      plan(*sample.tensor);
    }
    for (auto& tensor : sample.tensors) {
      plan(tensor);
    }
    for (auto& field : sample.fields) {
      plan(field);
    }
  }

  bool set_ownership(Sample& sample) const {
    const auto copied = [&](const Tensor& tensor) {
      return std::any_of(bindings_.begin(), bindings_.end(),
                         [&](const Binding& binding) { return binding.tensor == &tensor; });
    };
    bool owned = !sample.tensor || copied(*sample.tensor);
    for (const auto& tensor : sample.tensors) {
      owned = copied(tensor) && owned;
    }
    for (auto& field : sample.fields) {
      owned = set_ownership(field) && owned;
    }
    sample.owned = owned;
    return owned;
  }

  void build_envelope(Envelope& envelope) {
    std::sort(envelope.slots.begin(), envelope.slots.end(),
              [](const EnvelopeSlot& a, const EnvelopeSlot& b) {
                return a.source_order < b.source_order;
              });
    OwnedBuffer destination(gst_buffer_new(), gst_buffer_unref);
    if (!destination) {
      throw std::bad_alloc();
    }
    for (const auto& slot : envelope.slots) {
      gst_buffer_append_memory(destination.get(),
                               gst_memory_ref(carriers_[slot.carrier].destination.get()));
      envelope.segments.push_back(slot.segment);
    }
    auto* source = envelope.source_sample ? gst_sample_get_buffer(envelope.source_sample) : nullptr;
    if (source) {
      if (!gst_buffer_copy_into(destination.get(), source, GST_BUFFER_COPY_TIMESTAMPS, 0, -1)) {
        throw std::runtime_error("Owned output timing copy failed");
      }
      // Never retain producer parent-buffer/credit metadata. Public tensor facts
      // already live in the projected Sample; preprocess is needed on reentry.
      std::string error;
      if (has_simaai_preprocess_meta(source) &&
          !copy_simaai_preprocess_meta(destination.get(), source, &error)) {
        throw std::runtime_error("Owned output metadata copy failed: " + error);
      }
    }
    auto* independent = gst_sample_new(
        destination.get(),
        envelope.source_sample ? gst_sample_get_caps(envelope.source_sample) : nullptr,
        envelope.source_sample ? gst_sample_get_segment(envelope.source_sample) : nullptr, nullptr);
    if (!independent) {
      throw std::bad_alloc();
    }
    envelope.destination = std::shared_ptr<void>(
        independent, [](void* p) { gst_sample_unref(static_cast<GstSample*>(p)); });
  }

  void publish(const Binding& binding) const {
    Tensor& tensor = *binding.tensor;
    const auto& envelope = envelopes_[binding.envelope];
    const auto slot =
        std::find_if(envelope.slots.begin(), envelope.slots.end(), [&](const EnvelopeSlot& entry) {
          return entry.source_order == binding.source_order;
        });
    const auto index = static_cast<guint>(slot - envelope.slots.begin());
    auto storage = make_gst_sample_memory_storage(
        static_cast<GstSample*>(envelope.destination.get()), index, index, envelope.destination);
    storage->sima_segments = envelope.segments;
    storage->device = {DeviceType::CPU, 0};
    storage->data = system_memory_address(carriers_[slot->carrier].destination.get());
    tensor.storage = std::move(storage);
    tensor.device = tensor.storage->device;
    tensor.read_only = false;
    // Only the envelope-local selector changes. Logical/backend/physical
    // provenance, offsets, padding, planes and numerical semantics do not.
    tensor.route.memory_index = static_cast<int>(index);
  }

  bool preserve_framework_;
  std::vector<Carrier> carriers_;
  std::vector<Envelope> envelopes_;
  std::vector<Binding> bindings_;
};

} // namespace

void materialize_owned_output(Sample& sample, bool preserve_framework) {
  OwnedOutputMaterializer(preserve_framework).apply(sample);
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
  Mapping dma_mapping;
  const bool dmabuf = gst_is_dmabuf_memory(mem);
  if (dmabuf) {
    // Explicit Owned is a CPU read too: use the same START/END authority as a
    // retained tensor view, not GstDmaBufAllocator's unsynchronized mmap.
    auto source_storage = make_gst_sample_memory_storage(sample, index, index, ref.storage->holder);
    dma_mapping = source_storage->map(MapMode::Read);
    if (!dma_mapping.data) {
      throw std::runtime_error("copy_tensor_from_sample_memory: DMA-BUF CPU mapping failed");
    }
    map.data = static_cast<guint8*>(dma_mapping.data);
    map.size = dma_mapping.size_bytes;
  } else {
    ++g_gst_memory_map_calls;
    if (!gst_memory_map(mem, &map, GST_MAP_READ)) {
      throw std::runtime_error("copy_tensor_from_sample_memory: gst_memory_map failed");
    }
  }

  auto storage = simaai::neat::make_cpu_owned_storage(map.size);
  simaai::neat::Mapping dst_map = storage->map(simaai::neat::MapMode::Write);
  if (map.size > 0 && dst_map.data) {
    std::memcpy(dst_map.data, map.data, map.size);
  }
  ++g_tensor_copy_count;
  g_tensor_copy_bytes.fetch_add(static_cast<std::uint64_t>(map.size), std::memory_order_relaxed);
  if (!dmabuf) {
    gst_memory_unmap(mem, &map);
  }

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

simaai::neat::Tensor tensor_view_from_sample_segment(const simaai::neat::Tensor& ref,
                                                     const std::string& segment_name,
                                                     int fallback_memory_index, bool keep_holder) {
  GstSample* sample = sample_from_tensor(ref);
  if (!sample) {
    throw std::runtime_error("tensor_view_from_sample_segment: missing sample holder");
  }
  GstBuffer* buffer = gst_sample_get_buffer(sample);
  if (!buffer) {
    throw std::runtime_error("tensor_view_from_sample_segment: missing buffer");
  }
  const guint n_mems = gst_buffer_n_memory(buffer);
  if (n_mems == 0) {
    throw std::runtime_error("tensor_view_from_sample_segment: buffer has no memories");
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
  int selected_memory_index = fallback_memory_index;
  if (!segment_name.empty() && ref.storage && !ref.storage->sima_segments.empty()) {
    const std::size_t cached_segment_index =
        find_runtime_segment_index(ref.storage->sima_segments, segment_name);
    if (cached_segment_index != static_cast<std::size_t>(-1) &&
        cached_segment_index < static_cast<std::size_t>(n_mems)) {
      selected_memory_index = static_cast<int>(cached_segment_index);
    }
  }
  const MemorySelection selection = select_sample_memory_index(buffer, selected_memory_index);
  const guint index = selection.index;
  auto storage = make_gst_sample_memory_storage(
      sample, index, fallback_memory_index, std::move(shared_holder), segment_name,
      segment_name.empty() && ref.storage ? &ref.storage->sima_segments : nullptr);
  if (!storage) {
    throw std::runtime_error("tensor_view_from_sample_segment: missing buffer memory");
  }
  if (!segment_name.empty() && ref.storage && !ref.storage->sima_segments.empty()) {
    // Generic name-first requests resolve against the live allocator, then
    // retain the existing caller-side route projection. Admitted numeric
    // binding paths already supplied their current table without re-extraction.
    storage->sima_segments = ref.storage->sima_segments;
  }
  simaai::neat::Tensor out = ref;
  out.storage = std::move(storage);
  out.device = out.storage->device;
  out.read_only = true;
  out.byte_offset = ref.byte_offset;
  std::size_t route_index = static_cast<std::size_t>(index);
  const std::size_t named_route_index =
      find_runtime_segment_index(out.storage->sima_segments, segment_name);
  if (named_route_index != static_cast<std::size_t>(-1)) {
    route_index = named_route_index;
  } else if (fallback_memory_index >= 0 &&
             static_cast<std::size_t>(fallback_memory_index) < out.storage->sima_segments.size()) {
    route_index = static_cast<std::size_t>(fallback_memory_index);
  }
  out.route.memory_index = static_cast<int>(route_index);
  out.route.physical_index = static_cast<int>(route_index);
  if (route_index < out.storage->sima_segments.size()) {
    if (out.storage->sima_segments[route_index].size_bytes > 0U) {
      out.storage->size_bytes =
          std::min(out.storage->size_bytes, out.storage->sima_segments[route_index].size_bytes);
    }
    if (!out.storage->sima_segments[route_index].name.empty()) {
      out.route.segment_name = out.storage->sima_segments[route_index].name;
    }
  }
  ++g_tensor_view_count;
  record_tensor_holder_fast_path_hit();
  return out;
}

simaai::neat::Tensor tensor_view_from_sample_memory(const simaai::neat::Tensor& ref,
                                                    int memory_index, bool keep_holder) {
  return tensor_view_from_sample_segment(ref, std::string{}, memory_index, keep_holder);
}

std::shared_ptr<void> holder_from_tensor(const simaai::neat::Tensor& tensor) {
  if (!tensor.storage)
    return nullptr;
  return tensor.storage->holder;
}

} // namespace simaai::neat::pipeline_internal
