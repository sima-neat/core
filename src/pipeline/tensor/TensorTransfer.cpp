/**
 * @file TensorTransfer.cpp
 * @brief CPU <-> SiMa device transfer helpers for Tensor using standard DMA-BUF memory.
 *
 * This file implements:
 *  - transfer_to_device(): copies a Tensor payload into a device-backed GstBuffer,
 *    wraps it in a GstSample, and returns a Tensor that references that storage.
 *  - transfer_to_cpu(): copies payload into CPU-owned storage.
 *
 * Payload semantics (consistent with the rest of the repo):
 *  - Dense tensors:
 *      - Prefer tight packed bytes via dense_bytes_tight()/copy_dense_bytes_tight_to().
 *      - If tight shape is unknown, fall back to copying the mapped region as-is.
 *  - NV12 / I420:
 *      - Always copy into tight packed YUV (no padding) using copy_nv12_contiguous_to()
 *        or copy_i420_contiguous_to(). This is the same logic used elsewhere in the repo.
 *  - Other composite tensors (planes present):
 *      - This implementation *packs* planes tightly into the destination (row-by-row),
 *        and then rewrites plane offsets/strides to match the packed layout.
 *      - This matches the behavior of Tensor::clone() and avoids the inconsistency
 *        of copying raw bytes but advertising packed planes.
 *
 * Segment semantics:
 *  - If required_segments is provided, it is used as the destination segment layout.
 *  - Else allocate one segment for the selected payload: {"tensor", payload_bytes}.
 *
 */

#include "pipeline/internal/TensorTransfer.h"

#include "pipeline/internal/GstDataAdapter.h"
#include "pipeline/internal/InputStreamUtil.h"
#include "pipeline/internal/SimaaiGstCompat.h"
#include "pipeline/internal/TensorUtil.h"
#include "pipeline/internal/TensorMath.h"
#include "gst/GstInit.h"
#include <simaai/scheduler/client/dma_memory.h>

#include <gst/gst.h>
#include <gstsimaaitensorbuffer_meta.h>

#include <algorithm>
#include <array>
#include <cerrno>
#include <cstdint>
#include <cstring>
#include <iterator>
#include <list>
#include <memory>
#include <mutex>
#include <new>
#include <stdexcept>
#include <string>
#include <tuple>
#include <utility>
#include <vector>

#include <unistd.h>

namespace simaai::neat::pipeline_internal {
namespace {

using simaai::neat::Device;
using simaai::neat::DeviceType;
using simaai::neat::MapMode;
using simaai::neat::Mapping;
using simaai::neat::Plane;
using simaai::neat::Segment;
using simaai::neat::Storage;
using simaai::neat::StorageKind;
using simaai::neat::Tensor;

namespace dma = ::simaai::scheduler::client::dma;

// One allocation per pool makes capacity accounting independent of the number
// of outstanding tensors. Evicted pools stop caching, but their borrowed buffer
// remains valid until its final GStreamer reference is released.
class TransferBufferCache {
  static constexpr std::size_t max_bytes = 256U * 1024U * 1024U;
  static constexpr std::size_t max_pools = 128U;
  // Nearby crop sizes share storage without charging six almost-identical
  // allocations against the byte budget. Logical spans stay exact.
  static constexpr std::size_t class_quantum = 64U * 1024U;
  struct Retire {
    void operator()(GstBufferPool* pool) const {
      if (pool) {
        gst_buffer_pool_set_active(pool, FALSE);
        gst_object_unref(pool);
      }
    }
  };
  using Pool = std::unique_ptr<GstBufferPool, Retire>;
  struct Entry {
    dma::HeapKind heap;
    std::size_t capacity;
    Pool pool;
  };
  std::mutex mutex_;
  std::list<Entry> pools_; // Most recently acquired first; shared across heaps.
  std::size_t capacity_ = 0;

  static GstBuffer* take(GstBufferPool* pool) {
    GstBufferPoolAcquireParams params{};
    params.flags = GST_BUFFER_POOL_ACQUIRE_FLAG_DONTWAIT;
    GstBuffer* buffer = nullptr;
    if (gst_buffer_pool_acquire_buffer(pool, &buffer, &params) != GST_FLOW_OK)
      gst_clear_buffer(&buffer);
    return buffer;
  }

  GstBuffer* reuse(dma::HeapKind heap, std::size_t capacity) {
    struct Unref {
      void operator()(GstBufferPool* pool) const {
        gst_object_unref(pool);
      }
    };
    // Borrow GObject references, not cache owners: eviction can still deactivate
    // and drain these pools immediately while another transfer inspects them.
    std::array<std::unique_ptr<GstBufferPool, Unref>, max_pools> candidates;
    std::size_t count = 0;
    {
      std::lock_guard lock(mutex_);
      for (const auto& entry : pools_)
        if (entry.heap == heap && entry.capacity == capacity)
          candidates[count++].reset(GST_BUFFER_POOL(gst_object_ref(entry.pool.get())));
    }
    for (std::size_t i = 0; i < count; ++i) {
      // Never wait for a caller to release an earlier tensor. Even a replacement
      // allocation after Scheduler retires a slot runs outside the cache lock.
      if (auto* buffer = take(candidates[i].get())) {
        std::lock_guard lock(mutex_);
        const auto it = std::find_if(pools_.begin(), pools_.end(), [&](const Entry& entry) {
          return entry.pool.get() == candidates[i].get();
        });
        if (it != pools_.end())
          pools_.splice(pools_.begin(), pools_, it);
        return buffer;
      }
    }
    return nullptr;
  }

  bool clear() {
    std::list<Entry> retired;
    {
      std::lock_guard lock(mutex_);
      retired.splice(retired.begin(), pools_);
      capacity_ = 0;
    }
    // Deactivate/free outside the cache lock. Each pool has at most one buffer:
    // either idle and released here, or borrowed and released on final return.
    return !retired.empty();
  }

public:
  GstBuffer* acquire(dma::HeapKind heap, std::size_t bytes, dma::Error* error) {
    const auto page = static_cast<std::size_t>(::getpagesize());
    const auto quantum = bytes <= class_quantum ? page : std::max(page, class_quantum);
    const bool cacheable = bytes > 0 && bytes <= max_bytes;
    const auto capacity = cacheable ? ((bytes + quantum - 1) / quantum) * quantum : 0;
    if (cacheable) {
      if (auto* buffer = reuse(heap, capacity)) {
        *error = {};
        return buffer;
      }
    }
    Pool created;
    GstBuffer* buffer = nullptr;
    for (int attempt = 0; attempt < 2; ++attempt) {
      if (cacheable) {
        // Allocate outside the cache lock; finite one-slot pools also work with
        // Scheduler versions that do not support max_buffers=0.
        created = Pool(dma::createDmaBufPool(heap, capacity, 1U, 1U, {}, error));
        if (created) {
          buffer = take(created.get());
          if (!buffer)
            *error = {EIO, "acquire transfer buffer"};
        }
      } else {
        buffer = dma::allocateDmaBufBuffer(heap, bytes, {}, error);
      }
      if (buffer)
        break;
      // A racing transfer may have published a usable slot while allocation
      // failed. Preserve that fast recovery before draining the idle cache.
      if (auto* available = reuse(heap, capacity)) {
        *error = {};
        return available;
      }
      if (attempt != 0 || error->code != ENOMEM || !clear())
        return nullptr;
    }
    if (!buffer)
      return nullptr;
    std::unique_ptr<GstBuffer, decltype(&gst_buffer_unref)> lease(buffer, gst_buffer_unref);
    if (created) {
      std::list<Entry> staged;
      try {
        staged.push_front({heap, capacity, std::move(created)});
      } catch (const std::bad_alloc&) {
        // Cache admission is optional: the borrowed slot remains valid after
        // its pool is retired, and is freed on final return instead of cached.
        *error = {};
        return lease.release();
      }
      std::list<Entry> retired;
      {
        std::lock_guard lock(mutex_);
        // Publish only after borrowing the new slot, so another thread cannot
        // take it first. Count loaned slots too: this conservatively bounds all
        // retained cache capacity, including the eventual concurrency high-water.
        pools_.splice(pools_.begin(), staged);
        capacity_ += capacity;
        while (capacity_ > max_bytes || pools_.size() > max_pools) {
          auto oldest = std::prev(pools_.end());
          capacity_ -= oldest->capacity;
          retired.splice(retired.end(), pools_, oldest);
        }
      }
    }
    *error = {};
    return lease.release();
  }
};

GstBuffer* acquire_transfer_buffer(dma::HeapKind heap, std::size_t bytes, dma::Error* error) {
  static TransferBufferCache cache;
  return cache.acquire(heap, bytes, error);
}

//==============================================================================
// Composite packing helpers
//==============================================================================

/**
 * Compute packed byte size for a composite tensor:
 *   sum_i (plane_i.h * plane_i.w * elem_bytes)
 * using plane.shape[0]=h, shape[1]=w.
 *
 * Throws if plane metadata is incomplete/invalid.
 */
std::size_t packed_composite_bytes_or_throw(const Tensor& src, std::size_t elem) {
  if (src.planes.empty())
    return 0;
  if (elem == 0)
    throw std::runtime_error("transfer: unknown element size");

  std::size_t total = 0;
  for (const auto& plane : src.planes) {
    if (plane.shape.size() < 2) {
      throw std::runtime_error("transfer: composite plane missing shape");
    }
    const int64_t h = plane.shape[0];
    const int64_t w = plane.shape[1];
    if (h <= 0 || w <= 0) {
      throw std::runtime_error("transfer: invalid plane dimensions");
    }

    std::size_t row_bytes = 0;
    if (!safe_mul(static_cast<std::size_t>(w), elem, &row_bytes)) {
      throw std::runtime_error("transfer: plane row_bytes overflow");
    }
    std::size_t plane_bytes = 0;
    if (!safe_mul(row_bytes, static_cast<std::size_t>(h), &plane_bytes)) {
      throw std::runtime_error("transfer: plane_bytes overflow");
    }
    if (!safe_add(total, plane_bytes, &total)) {
      throw std::runtime_error("transfer: total composite bytes overflow");
    }
  }
  return total;
}

/**
 * Copy a composite tensor into a packed layout in `dst`:
 *  - Each plane is copied row-by-row into consecutive regions.
 *  - Destination plane layout matches contiguous_planes() below.
 */
bool copy_composite_packed(const Tensor& src, std::size_t elem, uint8_t* dst,
                           std::size_t dst_size) {
  if (!dst)
    return false;
  if (src.planes.empty())
    return false;
  if (elem == 0)
    return false;

  const std::size_t required = packed_composite_bytes_or_throw(src, elem);
  if (required == 0 || dst_size < required)
    return false;

  Mapping mapping = src.map_read();
  if (!mapping.data)
    return false;

  const uint8_t* base = static_cast<const uint8_t*>(mapping.data);
  const std::size_t total = mapping.size_bytes;

  std::size_t dst_off = 0;
  for (const auto& plane : src.planes) {
    if (plane.shape.size() < 2)
      return false;

    const int64_t h = plane.shape[0];
    const int64_t w = plane.shape[1];
    if (h <= 0 || w <= 0)
      return false;

    std::size_t row_bytes = 0;
    if (!safe_mul(static_cast<std::size_t>(w), elem, &row_bytes))
      return false;

    // Source row stride (bytes). Default to tight row_bytes if missing.
    int64_t src_stride = static_cast<int64_t>(row_bytes);
    if (!plane.strides_bytes.empty())
      src_stride = plane.strides_bytes[0];

    // If stride[1] exists, require it matches elem_bytes (repo convention).
    if (plane.strides_bytes.size() > 1 && plane.strides_bytes[1] != static_cast<int64_t>(elem)) {
      throw std::runtime_error("transfer: plane stride[1] mismatch");
    }

    if (src_stride <= 0 || static_cast<std::size_t>(src_stride) < row_bytes)
      return false;
    if (plane.byte_offset < 0)
      return false;

    // Bounds check source plane region (when mapping size is known).
    if (total > 0) {
      // end = byte_offset + stride*(h-1) + row_bytes
      std::size_t end = 0;
      const std::size_t off = static_cast<std::size_t>(plane.byte_offset);
      const std::size_t stride = static_cast<std::size_t>(src_stride);
      const std::size_t uh = static_cast<std::size_t>(h);

      if (uh == 0)
        return false;

      std::size_t term = 0;
      if (!safe_mul((uh - 1), stride, &term))
        return false;
      if (!safe_add(off, term, &end))
        return false;
      if (!safe_add(end, row_bytes, &end))
        return false;
      if (end > total)
        return false;
    }

    // Copy rows.
    const uint8_t* src_ptr = base + static_cast<std::size_t>(plane.byte_offset);
    uint8_t* dst_ptr = dst + dst_off;
    for (int64_t y = 0; y < h; ++y) {
      std::memcpy(dst_ptr + static_cast<std::size_t>(y) * row_bytes,
                  src_ptr + static_cast<std::size_t>(y) * static_cast<std::size_t>(src_stride),
                  row_bytes);
    }

    std::size_t plane_bytes = 0;
    if (!safe_mul(row_bytes, static_cast<std::size_t>(h), &plane_bytes))
      return false;
    dst_off += plane_bytes;
  }

  return true;
}

/**
 * Produce packed (contiguous) plane metadata corresponding to copy_composite_packed().
 * Sets:
 *  - byte_offset packed from 0
 *  - strides_bytes = {row_bytes, elem_bytes}
 */
std::vector<Plane> contiguous_planes(const Tensor& src, std::size_t elem) {
  if (src.planes.empty())
    return {};
  if (elem == 0)
    throw std::runtime_error("transfer: unknown element size");

  std::vector<Plane> out_planes;
  out_planes.reserve(src.planes.size());

  std::size_t offset = 0;
  for (const auto& plane : src.planes) {
    if (plane.shape.size() < 2) {
      throw std::runtime_error("transfer: invalid plane shape");
    }
    const int64_t h = plane.shape[0];
    const int64_t w = plane.shape[1];
    if (h <= 0 || w <= 0) {
      throw std::runtime_error("transfer: invalid plane dimensions");
    }

    std::size_t row_bytes = 0;
    if (!safe_mul(static_cast<std::size_t>(w), elem, &row_bytes)) {
      throw std::runtime_error("transfer: plane row_bytes overflow");
    }

    Plane out = plane;
    out.byte_offset = static_cast<int64_t>(offset);
    out.strides_bytes = {static_cast<int64_t>(row_bytes), static_cast<int64_t>(elem)};
    out_planes.push_back(std::move(out));

    std::size_t plane_bytes = 0;
    if (!safe_mul(row_bytes, static_cast<std::size_t>(h), &plane_bytes)) {
      throw std::runtime_error("transfer: plane_bytes overflow");
    }
    if (!safe_add(offset, plane_bytes, &offset)) {
      throw std::runtime_error("transfer: total plane bytes overflow");
    }
  }

  return out_planes;
}

//==============================================================================
// Payload sizing and copying
//==============================================================================

/**
 * Returns a tight “payload byte size” suitable for allocating a destination buffer
 * when we intend to copy/pack into a tight representation.
 *
 * For dense tensors, returns dense_bytes_tight() if known (>0), else 0.
 * For NV12/I420, returns required bytes if known (>0), else 0.
 * For other composite tensors, returns packed plane bytes (throws if metadata invalid).
 */
std::size_t compute_payload_bytes_strict(const Tensor& src) {
  if (src.has_tiled_storage())
    return src.semantic.tess->storage_bytes;
  // Prefer explicit NV12/I420 requirements (do not rely on plane shapes if not present).
  if (src.is_nv12()) {
    const std::size_t bytes = src.nv12_required_bytes();
    if (bytes > 0)
      return bytes;
  }
  if (src.is_i420()) {
    const std::size_t bytes = src.i420_required_bytes();
    if (bytes > 0)
      return bytes;
  }

  if (src.is_composite()) {
    const std::size_t elem = dtype_bytes(src.dtype);
    return packed_composite_bytes_or_throw(src, elem);
  }

  if (src.is_dense()) {
    const std::size_t bytes = src.dense_bytes_tight();
    if (bytes > 0)
      return bytes;
  }

  return 0;
}

/**
 * Copy src payload into dst. The destination buffer is assumed to be at least dst_size bytes.
 *
 * Copy policy:
 *  - NV12 / I420 => tight packed YUV
 *  - dense with known tight size => tight packed bytes
 *  - composite (other) => packed plane bytes (row-by-row) + tight plane layout
 *  - fallback => raw mapped bytes
 */
bool copy_tensor_payload(const Tensor& src, uint8_t* dst, std::size_t dst_size) {
  if (!dst)
    return false;
  if (src.has_tiled_storage())
    return src.copy_payload_bytes_to(dst, dst_size);

  if (src.is_nv12()) {
    return src.copy_nv12_contiguous_to(dst, dst_size);
  }
  if (src.is_i420()) {
    return src.copy_i420_contiguous_to(dst, dst_size);
  }

  if (src.is_dense()) {
    const std::size_t dense_bytes = src.dense_bytes_tight();
    if (dense_bytes > 0) {
      return src.copy_dense_bytes_tight_to(dst, dst_size);
    }
  }

  if (src.is_composite()) {
    const std::size_t elem = dtype_bytes(src.dtype);
    return copy_composite_packed(src, elem, dst, dst_size);
  }

  // Fallback: copy the mapped region as-is.
  return src.copy_payload_bytes_to(dst, dst_size);
}

//==============================================================================
// Segments, flags, and tensor finalization
//==============================================================================

std::size_t segment_total_bytes(const std::vector<Segment>& segments) {
  std::size_t total = 0;
  for (const auto& seg : segments) {
    if (!safe_add(total, seg.size_bytes, &total)) {
      throw std::runtime_error("transfer: segment_total_bytes overflow");
    }
  }
  return total;
}

std::vector<Segment> resolve_segments(const std::vector<Segment>* required_segments,
                                      std::size_t payload_bytes) {
  std::vector<Segment> segments = required_segments ? *required_segments : std::vector<Segment>{};
  if (segments.empty())
    segments.push_back({"tensor", payload_bytes});

  const std::size_t total = segment_total_bytes(segments);
  if (payload_bytes == 0 || total < payload_bytes) {
    throw std::runtime_error("transfer: segment layout too small for payload");
  }

  return segments;
}

std::uint64_t target_flags_from_device(const Device& target) {
  if (target.type == DeviceType::SIMA_CVU) {
    return static_cast<std::uint64_t>(GST_SIMAAI_MEMORY_TARGET_EV74);
  }
  if (target.type == DeviceType::SIMA_MLA) {
    switch (target.id) {
    case 0:
      return static_cast<std::uint64_t>(GST_SIMAAI_MEMORY_TARGET_DMS0);
    case 1:
      return static_cast<std::uint64_t>(GST_SIMAAI_MEMORY_TARGET_DMS1);
    case 2:
      return static_cast<std::uint64_t>(GST_SIMAAI_MEMORY_TARGET_DMS2);
    case 3:
      return static_cast<std::uint64_t>(GST_SIMAAI_MEMORY_TARGET_DMS3);
    default:
      break;
    }
  }
  return 0;
}

// A new allocation must not inherit producer bindings, layout or parent-buffer loans.
bool copy_gst_metadata(GstBuffer* dst, GstBuffer* src, std::string* error) {
  const auto flags =
      static_cast<GstBufferCopyFlags>(GST_BUFFER_COPY_TIMESTAMPS | GST_BUFFER_COPY_FLAGS);
  if (!gst_buffer_copy_into(dst, src, flags, 0, -1)) {
    *error = "failed to copy buffer timing";
    return false;
  }
  try {
    using namespace simaai::gst;
    const auto* frame = tensor_buffer_frame_metadata(src, error);
    if (!error->empty()) {
      *error = "compact frame metadata: " + *error;
      return false;
    }
    auto* legacy = gst_buffer_get_custom_meta(src, "GstSimaMeta");
    if (!frame && !legacy)
      return true;
    auto* metadata = gst_buffer_add_custom_meta(dst, "GstSimaMeta");
    if (!metadata) {
      *error = "failed to copy frame metadata";
      return false;
    }
    auto* fields = gst_custom_meta_get_structure(metadata);
    if (frame) {
      gst_structure_set(fields, "frame-id", G_TYPE_INT64, gint64(frame->frame_id), "stream-id",
                        G_TYPE_STRING, frame->stream_id.c_str(), "timestamp", G_TYPE_UINT64,
                        guint64(frame->timestamp), nullptr);
      if (frame->input_sequence)
        gst_structure_set(fields, "input-seq", G_TYPE_INT64, gint64(*frame->input_sequence),
                          nullptr);
      if (frame->original_sequence)
        gst_structure_set(fields, "orig-input-seq", G_TYPE_INT64, gint64(*frame->original_sequence),
                          nullptr);
      tensor_buffer_write_preprocess_facts(fields, frame->preprocess, true);
    } else {
      auto* source_fields = gst_custom_meta_get_structure(legacy);
      for (const auto* name : {"frame-id", "input-seq", "orig-input-seq", "stream-id",
                               "orig-stream-id", "timestamp", "attributes"}) {
        if (const auto* value = gst_structure_get_value(source_fields, name))
          gst_structure_set_value(fields, name, value);
      }
      const auto preprocess = read_simaai_preprocess_meta(src);
      if (preprocess && !write_simaai_preprocess_meta(dst, *preprocess)) {
        *error = "failed to write preprocessing metadata";
        return false;
      }
    }
    // Source timing is independent of the producing stage's frame and allocation IDs.
    TensorBufferSourceValues source;
    if (!tensor_buffer_read_source_values(src, &source, error))
      return false;
    using Values = TensorBufferSourceValues;
    for (const auto& [flag, name] : {std::pair{Values::FrameIdValid, "sample-frame-id-valid"},
                                     {Values::PtsValid, "sample-pts-valid"},
                                     {Values::DtsValid, "sample-dts-valid"},
                                     {Values::DurationValid, "sample-duration-valid"}}) {
      if (source.has(flag))
        gst_structure_set(fields, name, G_TYPE_BOOLEAN, gboolean(source.is_valid(flag)), nullptr);
    }
    if (source.has(Values::FrameId))
      gst_structure_set(fields, "sample-frame-id", G_TYPE_INT64, gint64(source.frame_id), nullptr);
    for (const auto& [flag, name, value] :
         {std::tuple{Values::Pts, "sample-pts-ns", source.pts_ns},
          {Values::Dts, "sample-dts-ns", source.dts_ns},
          {Values::Duration, "sample-duration-ns", source.duration_ns},
          {Values::Offset, "sample-offset", source.offset},
          {Values::OffsetEnd, "sample-offset-end", source.offset_end}}) {
      if (source.has(flag))
        gst_structure_set(fields, name, G_TYPE_UINT64, guint64(value), nullptr);
    }
    return true;
  } catch (const std::exception& exception) {
    *error = exception.what();
  }
  return false;
}

/**
 * Normalize output Tensor to match the payload we just wrote:
 *  - storage/device updated
 *  - read_only=false (destination is writable storage)
 *  - byte_offset=0
 *  - dense: contiguous strides
 *  - composite: packed plane offsets/strides
 */
Tensor finalize_transfer_tensor(const Tensor& src, const std::shared_ptr<Storage>& storage,
                                const std::vector<Segment>& segments) {
  if (!storage) {
    throw std::runtime_error("transfer: missing destination storage");
  }

  const std::size_t elem = dtype_bytes(src.dtype);
  if (elem == 0) {
    throw std::runtime_error("transfer: unknown element size");
  }

  Tensor out = src;
  out.storage = storage;
  out.device = storage->device;
  out.read_only = false;
  out.byte_offset = 0;
  out.route.memory_index = 0;
  out.route.physical_byte_offset = 0;

  // Dense: ensure contiguous strides.
  out.strides_bytes =
      src.has_tiled_storage() ? std::vector<int64_t>{} : contiguous_strides(out.shape, elem);

  // Composite: ensure plane layout matches packed payload copy.
  if (out.is_composite()) {
    out.planes = contiguous_planes(src, elem);
  }

  out.storage->sima_segments = segments;
  return out;
}

std::shared_ptr<Storage> make_driver_dmabuf_storage(GstSample* sample, const Device& target,
                                                    std::uint64_t target_flags) {
  auto storage = make_gst_sample_storage(sample);
  if (!storage || !storage->holder) {
    return {};
  }

  // A standard GstDmaBufAllocator memory deliberately has no legacy SiMa
  // allocator flags. Keep device placement in Tensor storage metadata while
  // preserving the ordinary GstMemory that direct kernel importers require.
  storage->device = target;
  storage->sima_mem_target_flags = target_flags;
  storage->sima_mem_flags = 0U;
  return storage;
}

Tensor transfer_to_driver_dmabuf(const Tensor& src, const Device& target,
                                 std::uint64_t target_flags, const std::vector<Segment>& segments,
                                 std::size_t payload_bytes) {
  using ::simaai::scheduler::client::dma::CpuAccess;
  using ::simaai::scheduler::client::dma::DmaBufView;
  using ::simaai::scheduler::client::dma::Error;
  using ::simaai::scheduler::client::dma::HeapKind;
  using OwnedBuffer = std::unique_ptr<GstBuffer, decltype(&gst_buffer_unref)>;
  using OwnedSample = std::unique_ptr<GstSample, decltype(&gst_sample_unref)>;

  const std::size_t allocation_bytes = segment_total_bytes(segments);
  const HeapKind heap = target.type == DeviceType::SIMA_MLA ? HeapKind::MlaDms : HeapKind::Cma;
  Error error;
  OwnedBuffer dst(acquire_transfer_buffer(heap, allocation_bytes, &error), gst_buffer_unref);
  if (!dst) {
    throw std::runtime_error("transfer: standard DMA-BUF allocation failed: " + error.message());
  }

  auto view = DmaBufView::fromGstMemory(gst_buffer_peek_memory(dst.get(), 0U), 0U,
                                        gst_buffer_get_size(dst.get()), &error);
  if (!view) {
    throw std::runtime_error("transfer: standard DMA-BUF view failed: " + error.message());
  }
  auto mapping = view->map(CpuAccess::Write, &error);
  if (!mapping) {
    throw std::runtime_error("transfer: standard DMA-BUF map failed: " + error.message());
  }

  const bool copied =
      copy_tensor_payload(src, static_cast<std::uint8_t*>(mapping->data()), mapping->size());
  if (!copied) {
    throw std::runtime_error("transfer: standard DMA-BUF payload copy failed");
  }
  // Preserve freshly allocated padding when reusing a larger carrier.
  if (allocation_bytes > payload_bytes) {
    std::memset(static_cast<std::uint8_t*>(mapping->data()) + payload_bytes, 0,
                allocation_bytes - payload_bytes);
  }
  if (!mapping->finish(&error)) {
    throw std::runtime_error("transfer: standard DMA-BUF CPU sync failed: " + error.message());
  }

  // Reuse one write-access plan for the full private capacity, regardless of
  // logical length. Release its owners before resizing so GstMemory stays
  // writable and no child memory/producer lease is substituted.
  mapping.reset();
  view.reset();
  if (gst_buffer_get_size(dst.get()) != allocation_bytes)
    gst_buffer_resize(dst.get(), 0, static_cast<gssize>(allocation_bytes));
  if (gst_buffer_get_size(dst.get()) != allocation_bytes) {
    throw std::runtime_error("transfer: resize DMA-BUF to requested carrier span failed");
  }

  if (src.storage && src.storage->kind == StorageKind::GstSample) {
    std::string holder_error;
    OwnedBuffer src_buffer(buffer_from_holder_if_gstsample(src, &holder_error), gst_buffer_unref);
    if (src_buffer) {
      std::string metadata_error;
      const bool copied_metadata = copy_gst_metadata(dst.get(), src_buffer.get(), &metadata_error);
      if (!copied_metadata) {
        throw std::runtime_error("transfer: " + metadata_error);
      }
    }
  }

  OwnedSample sample(gst_sample_new(dst.get(), nullptr, nullptr, nullptr), gst_sample_unref);
  if (!sample) {
    throw std::runtime_error("transfer: failed to wrap standard DMA-BUF sample");
  }
  dst.reset();
  auto storage = make_driver_dmabuf_storage(sample.get(), target, target_flags);
  sample.reset();
  if (!storage) {
    throw std::runtime_error("transfer: failed to create standard DMA-BUF storage");
  }
  return finalize_transfer_tensor(src, storage, segments);
}

} // namespace

//==============================================================================
// Public API
//==============================================================================

Tensor transfer_to_device(const Tensor& src, const Device& target,
                          const std::vector<Segment>* required_segments) {
  const std::uint64_t target_flags = target_flags_from_device(target);
  if (target_flags == 0) {
    throw std::runtime_error("transfer: unsupported target device");
  }

  simaai::neat::gst_init_once();

  // Determine payload bytes. Prefer tight size; otherwise fall back to mapped size.
  std::size_t payload_bytes = 0;
  try {
    payload_bytes = compute_payload_bytes_strict(src);
  } catch (const std::exception&) {
    // If strict sizing fails for a non-composite tensor, we can still fall back to mapping size.
    // For composite tensors this likely indicates missing plane metadata; surface it as an error.
    if (src.is_composite())
      throw;
  }

  if (payload_bytes == 0) {
    Mapping mapping = src.map_read();
    if (!mapping.data) {
      throw std::runtime_error("transfer: source mapping failed");
    }
    payload_bytes = mapping.size_bytes;
  }
  if (payload_bytes == 0) {
    throw std::runtime_error("transfer: unknown payload size");
  }

  const std::vector<Segment> segments = resolve_segments(required_segments, payload_bytes);

  return transfer_to_driver_dmabuf(src, target, target_flags, segments, payload_bytes);
}

Tensor transfer_to_cpu(const Tensor& src) {
  // Determine payload bytes. Prefer tight size; otherwise fall back to mapped size.
  std::size_t payload_bytes = 0;
  try {
    payload_bytes = compute_payload_bytes_strict(src);
  } catch (const std::exception&) {
    if (src.is_composite())
      throw;
  }

  if (payload_bytes == 0) {
    Mapping mapping = src.map_read();
    if (!mapping.data) {
      throw std::runtime_error("transfer: source mapping failed");
    }
    payload_bytes = mapping.size_bytes;
  }
  if (payload_bytes == 0) {
    throw std::runtime_error("transfer: unknown payload size");
  }

  auto storage = simaai::neat::make_cpu_owned_storage(payload_bytes);
  Mapping dst_map = storage->map(MapMode::Write);
  if (!dst_map.data) {
    throw std::runtime_error("transfer: destination map failed");
  }
  if (!copy_tensor_payload(src, static_cast<uint8_t*>(dst_map.data), dst_map.size_bytes)) {
    throw std::runtime_error("transfer: payload copy failed");
  }

  return finalize_transfer_tensor(src, storage, {});
}

} // namespace simaai::neat::pipeline_internal
