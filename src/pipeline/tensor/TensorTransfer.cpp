/**
 * @file TensorTransfer.cpp
 * @brief CPU <-> SiMa device transfer helpers for Tensor using the SiMa GStreamer
 *        segment allocator + buffer pool (when available).
 *
 * This file implements:
 *  - A small global cache of GstBufferPools keyed by {target_flags, mem_flags, segments}.
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
 *  - Else if required_segment_names is provided, we reuse the segment layout from src.storage
 *    and validate the required names exist.
 *  - Else if src.storage already has segment layout, reuse it.
 *  - Else allocate a single default segment: {"tensor", payload_bytes}.
 *
 * TODO(repo-policy):
 *  - Decide whether destination mem_flags should ever include GST_SIMAAI_MEMORY_FLAG_RDONLY.
 *    We currently clear it because we map/write into the destination buffer.
 *  - Consider pool eviction / max cache size. The current pool cache grows unbounded with new keys.
 */

#include "pipeline/internal/TensorTransfer.h"

#include "pipeline/internal/GstDataAdapter.h"
#include "pipeline/internal/SimaaiMemory.h"
#include "pipeline/internal/SimaaiGstCompat.h"
#include "pipeline/internal/TensorUtil.h"
#include "pipeline/internal/TensorMath.h"
#include "gst/GstInit.h"

#include <gst/gst.h>
#include <gst/gstbufferpool.h>

#include <cstdint>
#include <cstring>
#include <limits>
#include <memory>
#include <mutex>
#include <optional>
#include <sstream>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

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

//==============================================================================
// Small safe-math helpers
//==============================================================================

//==============================================================================
// Pool cache (keyed by flags + segment layout)
//==============================================================================

struct PoolCache {
  std::mutex mu;
  std::unordered_map<std::string, std::shared_ptr<GstBufferPool>> pools;
  std::size_t hits = 0;
  std::size_t misses = 0;
};

PoolCache& pool_cache() {
  static PoolCache cache;
  return cache;
}

/** Build a deterministic pool key from target flags, memory flags, and segment layout. */
std::string pool_key(std::uint64_t target_flags, std::uint64_t mem_flags,
                     const std::vector<Segment>& segments) {
  std::ostringstream oss;
  oss << target_flags << ":" << mem_flags << ":" << segments.size();
  for (const auto& seg : segments) {
    oss << "|" << seg.name << ":" << seg.size_bytes;
  }
  return oss.str();
}

#if SIMA_HAS_SIMAAI_POOL
std::shared_ptr<GstBufferPool> create_pool(std::uint64_t target_flags, std::uint64_t mem_flags,
                                           const std::vector<Segment>& segments) {
  if (segments.empty())
    return {};
  gst_simaai_segment_memory_init_once();

  std::vector<gsize> sizes;
  sizes.reserve(segments.size());
  std::vector<std::string> names_str;
  names_str.reserve(segments.size());

  for (const auto& seg : segments) {
    sizes.push_back(static_cast<gsize>(seg.size_bytes));
    names_str.push_back(seg.name);
  }

  std::vector<const char*> names;
  names.reserve(names_str.size());
  for (const auto& s : names_str) {
    names.push_back(s.c_str());
  }

  const GstMemoryFlags flags = static_cast<GstMemoryFlags>(target_flags | mem_flags);

  GstBufferPool* pool = gst_simaai_allocate_buffer_pool2(
      /*object=*/nullptr, gst_simaai_memory_get_segment_allocator(),
      /*min_buffers=*/2,
      /*max_buffers=*/0, flags, static_cast<gsize>(segments.size()), sizes.data(), names.data());
  if (!pool)
    return {};

  return std::shared_ptr<GstBufferPool>(pool,
                                        [](GstBufferPool* p) { gst_simaai_free_buffer_pool(p); });
}
#endif

/** Get (or create) a pool for this (target_flags, mem_flags, segments) layout. */
std::shared_ptr<GstBufferPool> get_pool(std::uint64_t target_flags, std::uint64_t mem_flags,
                                        const std::vector<Segment>& segments) {
  const std::string key = pool_key(target_flags, mem_flags, segments);

  PoolCache& cache = pool_cache();
  {
    std::lock_guard<std::mutex> lock(cache.mu);
    auto it = cache.pools.find(key);
    if (it != cache.pools.end() && it->second) {
      cache.hits++;
      return it->second;
    }
    cache.misses++;
  }

#if !SIMA_HAS_SIMAAI_POOL
  (void)target_flags;
  (void)mem_flags;
  (void)segments;
  return {};
#else
  std::shared_ptr<GstBufferPool> created = create_pool(target_flags, mem_flags, segments);
  if (!created)
    return {};

  std::lock_guard<std::mutex> lock(cache.mu);
  auto [it, inserted] = cache.pools.emplace(key, created);
  if (!inserted && it->second) {
    return it->second;
  }
  return created;
#endif
}

//==============================================================================
// Tensor math helpers (shared)
//==============================================================================

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

bool segments_contain_names(const std::vector<Segment>& segments,
                            const std::vector<std::string>& names) {
  for (const auto& required : names) {
    bool found = false;
    for (const auto& seg : segments) {
      if (seg.name == required) {
        found = true;
        break;
      }
    }
    if (!found)
      return false;
  }
  return true;
}

std::vector<Segment> resolve_segments(const Tensor& src,
                                      const std::vector<Segment>* required_segments,
                                      const std::vector<std::string>* required_names,
                                      std::size_t payload_bytes) {
  std::vector<Segment> segments;

  if (required_segments && !required_segments->empty()) {
    segments = *required_segments;
  } else if (required_names && !required_names->empty()) {
    if (!src.storage || src.storage->sima_segments.empty()) {
      throw std::runtime_error("transfer: required segment names but no segment layout");
    }
    segments = src.storage->sima_segments;
  } else if (src.storage && !src.storage->sima_segments.empty()) {
    segments = src.storage->sima_segments;
  }

  if (segments.empty()) {
    Segment seg;
    seg.name = "tensor";
    seg.size_bytes = payload_bytes;
    segments.push_back(std::move(seg));
  }

  if (required_names && !required_names->empty()) {
    if (!segments_contain_names(segments, *required_names)) {
      throw std::runtime_error("transfer: missing required segments");
    }
  }

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

/** Copy “lightweight” buffer metadata/timestamps/flags from src to dst when available. */
void copy_gst_metadata(GstBuffer* dst, GstBuffer* src) {
  if (!dst || !src)
    return;
  const GstBufferCopyFlags flags = static_cast<GstBufferCopyFlags>(
      GST_BUFFER_COPY_METADATA | GST_BUFFER_COPY_TIMESTAMPS | GST_BUFFER_COPY_FLAGS);
  gst_buffer_copy_into(dst, src, flags, 0, -1);
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

  // Dense: ensure contiguous strides.
  out.strides_bytes = contiguous_strides(out.shape, elem);

  // Composite: ensure plane layout matches packed payload copy.
  if (out.is_composite()) {
    out.planes = contiguous_planes(src, elem);
  }

  out.storage->sima_segments = segments;
  return out;
}

} // namespace

//==============================================================================
// Public API
//==============================================================================

TransferPoolStats tensor_transfer_pool_stats() {
  PoolCache& cache = pool_cache();
  std::lock_guard<std::mutex> lock(cache.mu);
  TransferPoolStats stats;
  stats.hits = cache.hits;
  stats.misses = cache.misses;
  stats.entries = cache.pools.size();
  return stats;
}

Tensor transfer_to_device(const Tensor& src, const Device& target,
                          const std::vector<Segment>* required_segments,
                          const std::vector<std::string>* required_segment_names) {
  const std::uint64_t target_flags = target_flags_from_device(target);
  if (target_flags == 0) {
    throw std::runtime_error("transfer: unsupported target device");
  }

#if !SIMA_HAS_SIMAAI_POOL
  throw std::runtime_error("transfer: simaai buffer pool unavailable");
#else
  simaai::neat::gst_init_once();

  // Choose destination memory flags.
  std::uint64_t mem_flags = 0;
  if (src.storage) {
    mem_flags = src.storage->sima_mem_flags;
  }
  if (mem_flags == 0) {
    mem_flags = static_cast<std::uint64_t>(GST_SIMAAI_MEMORY_FLAG_CACHED);
  }
  // We will map/write into the destination; do not request read-only memory.
  mem_flags &= ~static_cast<std::uint64_t>(GST_SIMAAI_MEMORY_FLAG_RDONLY);

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

  const std::vector<Segment> segments =
      resolve_segments(src, required_segments, required_segment_names, payload_bytes);

  std::shared_ptr<GstBufferPool> pool = get_pool(target_flags, mem_flags, segments);
  if (!pool) {
    throw std::runtime_error("transfer: buffer pool allocation failed");
  }

  GstBuffer* dst = nullptr;
  if (gst_buffer_pool_acquire_buffer(pool.get(), &dst, nullptr) != GST_FLOW_OK || !dst) {
    throw std::runtime_error("transfer: buffer pool acquire failed");
  }

  // Try to preserve metadata if the source is itself a GstSample-backed tensor.
  if (src.storage && src.storage->kind == StorageKind::GstSample) {
    std::string holder_err;
    GstBuffer* src_buf = buffer_from_holder_if_gstsample(src, &holder_err);
    if (src_buf) {
      copy_gst_metadata(dst, src_buf);
      gst_buffer_unref(src_buf);
    }
  }

  GstMapInfo map{};
  if (!gst_buffer_map(dst, &map, GST_MAP_WRITE)) {
    gst_buffer_unref(dst);
    throw std::runtime_error("transfer: destination map failed");
  }

  bool ok = false;
  try {
    ok = copy_tensor_payload(src, static_cast<uint8_t*>(map.data), map.size);
  } catch (...) {
    gst_buffer_unmap(dst, &map);
    gst_buffer_unref(dst);
    throw;
  }

  gst_buffer_unmap(dst, &map);
  if (!ok) {
    gst_buffer_unref(dst);
    throw std::runtime_error("transfer: payload copy failed");
  }

  // Wrap destination buffer in a sample solely as an ownership / mapping carrier.
  // Caps are intentionally omitted because Tensor carries shape/dtype/semantic itself.
  GstSample* sample = gst_sample_new(dst, nullptr, nullptr, nullptr);
  if (!sample) {
    gst_buffer_unref(dst);
    throw std::runtime_error("transfer: failed to wrap GstSample");
  }
  gst_buffer_unref(dst);

  auto storage = make_gst_sample_storage(sample);
  gst_sample_unref(sample);
  if (!storage) {
    throw std::runtime_error("transfer: failed to create storage");
  }

  return finalize_transfer_tensor(src, storage, segments);
#endif
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
