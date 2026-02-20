/**
 * @file TensorCore.cpp
 * @brief Core utilities for Tensor:
 *        - CPU storage helpers (owned + external)
 *        - Deep clone to CPU (dense + planar)
 *        - Contiguity helpers
 *        - Device transfer wrappers (cpu/cvu/mla/to)
 *        - Payload byte-copy helpers
 *        - Validation + debug formatting
 *
 * Key invariants / semantics (confirmed by TensorCore.h + call sites):
 *  - Tensor::map()/map_read() applies `byte_offset` to the returned pointer and
 *    shrinks `size_bytes` accordingly.
 *  - Composite tensors (planar) must have `byte_offset == 0`; each plane has its own
 *    `byte_offset` relative to the tensor mapping base.
 *  - Plane `strides_bytes[0]` is the row-stride in **bytes**. If `strides_bytes[1]` is present,
 *    it must equal element-size in bytes (see tensor_plane_bytes_tight() usage in InputStream).
 *  - clone() is used for “owning” CPU-safe copies (e.g., Run copy_input/copy_output paths),
 *    so clone() returns CPU-owned storage.
 *  - mla(force=false) prefers CVU (SIMA_CVU) as the default “MLA-capable” working location; use
 *    force=true to require actual SIMA_MLA residency (DMS0 by default).
 */

#include "pipeline/TensorCore.h"
#include "pipeline/internal/TensorMath.h"
#include "pipeline/internal/TensorTransfer.h"
#include "pipeline/internal/TensorUtil.h"

#include <atomic>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <memory>
#include <new>
#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

namespace simaai::neat {

using simaai::neat::pipeline_internal::contiguous_strides_bytes;
using simaai::neat::pipeline_internal::dtype_bytes;
using simaai::neat::pipeline_internal::safe_add;
using simaai::neat::pipeline_internal::safe_mul;

namespace {

//==============================================================================
// Small internal helpers
//==============================================================================

constexpr std::string_view dtype_name(simaai::neat::TensorDType dtype) noexcept {
  switch (dtype) {
  case simaai::neat::TensorDType::UInt8:
    return "UInt8";
  case simaai::neat::TensorDType::Int8:
    return "Int8";
  case simaai::neat::TensorDType::UInt16:
    return "UInt16";
  case simaai::neat::TensorDType::Int16:
    return "Int16";
  case simaai::neat::TensorDType::Int32:
    return "Int32";
  case simaai::neat::TensorDType::BFloat16:
    return "BFloat16";
  case simaai::neat::TensorDType::Float32:
    return "Float32";
  case simaai::neat::TensorDType::Float64:
    return "Float64";
  }
  return "Unknown";
}

constexpr std::string_view device_name(DeviceType type) noexcept {
  switch (type) {
  case DeviceType::CPU:
    return "CPU";
  case DeviceType::SIMA_APU:
    return "SIMA_APU";
  case DeviceType::SIMA_CVU:
    return "SIMA_CVU";
  case DeviceType::SIMA_MLA:
    return "SIMA_MLA";
  case DeviceType::UNKNOWN:
    return "UNKNOWN";
  }
  return "UNKNOWN";
}

constexpr std::string_view layout_name(simaai::neat::TensorLayout layout) noexcept {
  switch (layout) {
  case simaai::neat::TensorLayout::Unknown:
    return "Unknown";
  case simaai::neat::TensorLayout::HWC:
    return "HWC";
  case simaai::neat::TensorLayout::CHW:
    return "CHW";
  case simaai::neat::TensorLayout::HW:
    return "HW";
  case simaai::neat::TensorLayout::Planar:
    return "Planar";
  }
  return "Unknown";
}

constexpr const char* storage_kind_name(StorageKind kind) noexcept {
  switch (kind) {
  case StorageKind::CpuOwned:
    return "CpuOwned";
  case StorageKind::CpuExternal:
    return "CpuExternal";
  case StorageKind::GstSample:
    return "GstSample";
  case StorageKind::DeviceHandle:
    return "DeviceHandle";
  case StorageKind::Unknown:
  default:
    return "Unknown";
  }
}

bool mapfail_debug_enabled() {
  const char* v = std::getenv("SIMA_TENSOR_MAPFAIL_DEBUG");
  return v && std::strcmp(v, "0") != 0;
}

void log_mapfail_once(const char* label, const Tensor& t, const Mapping& mapping) {
  static std::atomic<bool> logged{false};
  if (!mapfail_debug_enabled())
    return;
  if (logged.exchange(true))
    return;

  const Storage* storage = t.storage.get();
  const char* kind = storage ? storage_kind_name(storage->kind) : "None";
  const Device dev = storage ? storage->device : Device{};
  std::fprintf(stderr,
               "[DBG] mapfail-pre label=%s storage=%s device=%s:%d size=%zu "
               "mem_target=0x%llx mem_flags=0x%llx read_only=%s byte_offset=%lld\n",
               label ? label : "unknown", kind, std::string(device_name(dev.type)).c_str(), dev.id,
               storage ? storage->size_bytes : 0,
               static_cast<unsigned long long>(storage ? storage->sima_mem_target_flags : 0),
               static_cast<unsigned long long>(storage ? storage->sima_mem_flags : 0),
               t.read_only ? "true" : "false", static_cast<long long>(t.byte_offset));
  std::fprintf(stderr, "[DBG] mapfail-post label=%s map_data=%p map_size=%zu\n",
               label ? label : "unknown", mapping.data, mapping.size_bytes);
}

/**
 * Compute tight dense byte size = elem_bytes * product(shape[i]).
 * Returns 0 on invalid dims or overflow.
 */
std::size_t dense_size_bytes_checked(const std::vector<int64_t>& shape, std::size_t elem_bytes) {
  if (shape.empty())
    return 0;

  std::size_t total = elem_bytes;
  for (int64_t dim : shape) {
    if (dim <= 0)
      return 0;
    const std::size_t u = static_cast<std::size_t>(dim);
    if (!safe_mul(total, u, &total))
      return 0;
  }
  return total;
}

std::vector<int64_t> normalized_dense_strides_bytes(const std::vector<int64_t>& shape,
                                                    std::size_t elem_bytes,
                                                    const std::vector<int64_t>& maybe_strides) {
  if (maybe_strides.size() == shape.size())
    return maybe_strides;
  return contiguous_strides_bytes(shape, elem_bytes);
}

/**
 * Copy an N-D dense tensor with arbitrary positive byte-strides into a contiguous buffer.
 *
 * Iteration order is row-major (last dimension changes fastest), matching contiguous layout.
 *
 * @param src Base pointer to the tensor payload (already includes Tensor::byte_offset).
 * @param src_bytes Mapping size in bytes (0 => skip bounds check).
 * @param shape Element-shape.
 * @param src_strides_bytes Per-dimension byte-strides (same rank as shape).
 * @param elem_bytes Element size in bytes.
 * @param dst Contiguous destination, sized for elem_bytes * product(shape).
 */
bool copy_dense_strided_to_contiguous(const uint8_t* src, std::size_t src_bytes,
                                      const std::vector<int64_t>& shape,
                                      const std::vector<int64_t>& src_strides_bytes,
                                      std::size_t elem_bytes, uint8_t* dst) {
  if (!src || !dst)
    return false;
  if (shape.empty())
    return true;
  if (src_strides_bytes.size() != shape.size())
    return false;

  // Conservative bounds check: maximum offset touched by any element.
  std::size_t max_offset = 0;
  for (size_t i = 0; i < shape.size(); ++i) {
    const int64_t dim = shape[i];
    const int64_t stride = src_strides_bytes[i];
    if (dim <= 0 || stride <= 0)
      return false;
    std::size_t term = static_cast<std::size_t>(dim - 1) * static_cast<std::size_t>(stride);
    if (!safe_add(max_offset, term, &max_offset))
      return false;
  }
  if (src_bytes > 0 && (max_offset + elem_bytes) > src_bytes)
    return false;

  // Total element count (overflow protected).
  std::size_t total_elems = 1;
  for (int64_t dim : shape) {
    if (dim <= 0)
      return false;
    const std::size_t u = static_cast<std::size_t>(dim);
    if (!safe_mul(total_elems, u, &total_elems))
      return false;
  }

  std::vector<int64_t> idx(shape.size(), 0);
  uint8_t* out = dst;

  for (std::size_t n = 0; n < total_elems; ++n) {
    std::size_t src_offset = 0;
    for (size_t i = 0; i < idx.size(); ++i) {
      std::size_t term =
          static_cast<std::size_t>(idx[i]) * static_cast<std::size_t>(src_strides_bytes[i]);
      if (!safe_add(src_offset, term, &src_offset))
        return false;
    }

    std::memcpy(out, src + src_offset, elem_bytes);
    out += elem_bytes;

    // Increment odometer index (last dim fastest).
    for (int i = static_cast<int>(idx.size()) - 1; i >= 0; --i) {
      idx[static_cast<size_t>(i)]++;
      if (idx[static_cast<size_t>(i)] < shape[static_cast<size_t>(i)])
        break;
      idx[static_cast<size_t>(i)] = 0;
    }
  }

  return true;
}

/**
 * Compute the number of bytes spanned by a plane according to its shape + stride.
 * This matches the access pattern used by map_nv12_read/map_i420_read and by copy loops:
 *   last touched byte ~= byte_offset + stride0*(h-1) + row_bytes.
 *
 * @return 0 if invalid or overflow.
 */
std::size_t plane_span_bytes_checked(const Plane& plane, std::size_t elem_bytes) {
  if (elem_bytes == 0)
    return 0;

  if (plane.shape.size() >= 2) {
    const int64_t h = plane.shape[0];
    const int64_t w = plane.shape[1];
    if (h <= 0 || w <= 0)
      return 0;

    std::size_t row_bytes = 0;
    if (!safe_mul(static_cast<std::size_t>(w), elem_bytes, &row_bytes))
      return 0;

    int64_t stride0 = static_cast<int64_t>(row_bytes);
    if (!plane.strides_bytes.empty())
      stride0 = plane.strides_bytes[0];
    if (stride0 <= 0)
      return 0;
    if (static_cast<std::size_t>(stride0) < row_bytes)
      return 0;

    // If a second stride is present, require it match elem_bytes (InputStream expectation).
    if (plane.strides_bytes.size() > 1) {
      if (plane.strides_bytes[1] != static_cast<int64_t>(elem_bytes))
        return 0;
    }

    const std::size_t uh = static_cast<std::size_t>(h);
    const std::size_t ustride = static_cast<std::size_t>(stride0);

    // span = (h-1)*stride0 + row_bytes
    std::size_t span = 0;
    if (uh == 0)
      return 0;
    std::size_t rows_term = 0;
    if (!safe_mul((uh - 1), ustride, &rows_term))
      return 0;
    if (!safe_add(rows_term, row_bytes, &span))
      return 0;
    return span;
  }

  // 1-D plane fallback (rare): treat as a vector of elements.
  if (plane.shape.size() == 1) {
    const int64_t n = plane.shape[0];
    if (n <= 0)
      return 0;
    std::size_t bytes = 0;
    if (!safe_mul(static_cast<std::size_t>(n), elem_bytes, &bytes))
      return 0;
    return bytes;
  }

  return 0;
}

} // namespace

//==============================================================================
// Storage creation
//==============================================================================

std::shared_ptr<Storage> make_cpu_owned_storage(std::size_t size_bytes) {
  void* ptr = nullptr;
  if (size_bytes > 0) {
    // POSIX aligned allocation; fallback to malloc if needed.
    if (posix_memalign(&ptr, 64, size_bytes) != 0) {
      ptr = std::malloc(size_bytes);
    }
  }
  if (size_bytes > 0 && !ptr)
    throw std::bad_alloc();

  auto buf = std::shared_ptr<uint8_t>(static_cast<uint8_t*>(ptr), [](uint8_t* p) { std::free(p); });
  auto holder = std::shared_ptr<void>(buf, buf.get()); // aliasing control block

  auto storage = std::make_shared<Storage>();
  storage->kind = StorageKind::CpuOwned;
  storage->device = {DeviceType::CPU, 0};
  storage->size_bytes = size_bytes;
  storage->holder = holder;
  storage->data = buf.get();
  storage->map_fn = [holder, size_bytes](MapMode /*mode*/) {
    Mapping mapping;
    mapping.data = holder.get();
    mapping.size_bytes = size_bytes;
    return mapping;
  };
  return storage;
}

/**
 * Wrap an external CPU buffer in Storage.
 *
 * NOTE:
 *  - `read_only` is an intent flag. Write enforcement is done at Tensor::map()
 *    via Tensor::read_only. Storage-level mapping does not currently enforce
 *    read-only, because some call sites may intentionally wrap external memory and
 *    still want to write through it (by setting Tensor::read_only = false).
 */
std::shared_ptr<Storage> make_cpu_external_storage(void* data, std::size_t size_bytes,
                                                   std::shared_ptr<void> holder,
                                                   bool /*read_only*/) {
  auto storage = std::make_shared<Storage>();
  storage->kind = StorageKind::CpuExternal;
  storage->device = {DeviceType::CPU, 0};
  storage->size_bytes = size_bytes;
  storage->holder = std::move(holder);
  storage->data = data;
  storage->map_fn = [data, size_bytes](MapMode /*mode*/) {
    Mapping mapping;
    mapping.data = data;
    mapping.size_bytes = size_bytes;
    return mapping;
  };
  return storage;
}

//==============================================================================
// Tensor: cloning / contiguity
//==============================================================================

Tensor Tensor::clone() const {
  const std::size_t elem = dtype_bytes(dtype);
  if (elem == 0)
    throw std::runtime_error("Tensor::clone: unknown element size");

  // --------------------------
  // Dense tensor clone (tight contiguous copy)
  // --------------------------
  if (is_dense()) {
    const std::size_t bytes = dense_size_bytes_checked(shape, elem);
    if (!shape.empty() && bytes == 0) {
      throw std::runtime_error("Tensor::clone: invalid/overflowing dense shape");
    }

    auto storage_copy = make_cpu_owned_storage(bytes);

    const Mapping src_map = map(MapMode::Read); // includes byte_offset
    if (!src_map.data && bytes != 0) {
      throw std::runtime_error("Tensor::clone: failed to map source");
    }

    const Mapping dst_map = storage_copy->map(MapMode::Write);
    if (!dst_map.data && bytes != 0) {
      throw std::runtime_error("Tensor::clone: failed to map destination");
    }

    if (bytes > 0 && (src_map.size_bytes < bytes || dst_map.size_bytes < bytes)) {
      throw std::runtime_error("Tensor::clone: mapping smaller than dense byte size");
    }

    const auto* src = static_cast<const uint8_t*>(src_map.data);
    auto* dst = static_cast<uint8_t*>(dst_map.data);

    if (bytes > 0) {
      if (is_contiguous() || strides_bytes.empty()) {
        std::memcpy(dst, src, bytes);
      } else {
        const std::vector<int64_t> src_strides =
            normalized_dense_strides_bytes(shape, elem, strides_bytes);
        if (!copy_dense_strided_to_contiguous(src, src_map.size_bytes, shape, src_strides, elem,
                                              dst)) {
          throw std::runtime_error("Tensor::clone: strided dense copy failed");
        }
      }
    }

    Tensor out;
    out.storage = std::move(storage_copy);
    out.dtype = dtype;
    out.layout = layout;
    out.shape = shape;
    out.strides_bytes = contiguous_strides_bytes(shape, elem);
    out.byte_offset = 0;
    out.device = {DeviceType::CPU, 0};
    out.semantic = semantic;
    out.read_only = false;
    return out;
  }

  // --------------------------
  // Composite / planar clone (pack planes tightly)
  // --------------------------
  std::size_t total_bytes = 0;
  std::vector<std::size_t> plane_row_bytes;
  plane_row_bytes.reserve(planes.size());

  for (const auto& plane : planes) {
    if (plane.shape.size() < 2)
      throw std::runtime_error("Tensor::clone: invalid plane shape");
    const int64_t h = plane.shape[0];
    const int64_t w = plane.shape[1];
    if (h <= 0 || w <= 0)
      throw std::runtime_error("Tensor::clone: invalid plane dimensions");

    std::size_t row_bytes = 0;
    if (!safe_mul(static_cast<std::size_t>(w), elem, &row_bytes)) {
      throw std::runtime_error("Tensor::clone: plane row_bytes overflow");
    }
    plane_row_bytes.push_back(row_bytes);

    std::size_t plane_bytes = 0;
    if (!safe_mul(row_bytes, static_cast<std::size_t>(h), &plane_bytes)) {
      throw std::runtime_error("Tensor::clone: plane byte size overflow");
    }
    if (!safe_add(total_bytes, plane_bytes, &total_bytes)) {
      throw std::runtime_error("Tensor::clone: total plane byte size overflow");
    }
  }

  auto storage_copy = make_cpu_owned_storage(total_bytes);

  const Mapping src_map = map(MapMode::Read); // base for planes (composite expects byte_offset=0)
  if (!src_map.data && total_bytes != 0) {
    throw std::runtime_error("Tensor::clone: failed to map source planes");
  }

  const Mapping dst_map = storage_copy->map(MapMode::Write);
  if (!dst_map.data && total_bytes != 0) {
    throw std::runtime_error("Tensor::clone: failed to map destination planes");
  }

  const auto* src = static_cast<const uint8_t*>(src_map.data);
  auto* dst = static_cast<uint8_t*>(dst_map.data);

  std::vector<Plane> out_planes;
  out_planes.reserve(planes.size());

  std::size_t offset = 0;
  for (size_t i = 0; i < planes.size(); ++i) {
    const auto& plane = planes[i];
    const std::size_t row_bytes = plane_row_bytes[i];
    const int64_t h = plane.shape[0];

    if (plane.byte_offset < 0) {
      throw std::runtime_error("Tensor::clone: plane byte_offset is negative");
    }

    // Source stride0 (bytes) defaults to tight row_bytes if missing.
    int64_t src_stride = static_cast<int64_t>(row_bytes);
    if (!plane.strides_bytes.empty())
      src_stride = plane.strides_bytes[0];
    if (src_stride <= 0)
      throw std::runtime_error("Tensor::clone: invalid plane stride");

    // Bounds check against the mapped source span (uses stride*(h-1)+row_bytes).
    const std::size_t span = [&]() -> std::size_t {
      const std::size_t uh = static_cast<std::size_t>(h);
      const std::size_t ustride = static_cast<std::size_t>(src_stride);
      if (uh == 0)
        return 0;
      std::size_t rows_term = 0;
      if (!safe_mul((uh - 1), ustride, &rows_term))
        return 0;
      std::size_t s = 0;
      if (!safe_add(rows_term, row_bytes, &s))
        return 0;
      return s;
    }();
    if (span == 0 && row_bytes != 0) {
      throw std::runtime_error("Tensor::clone: plane span overflow");
    }
    const std::size_t plane_off = static_cast<std::size_t>(plane.byte_offset);
    if (src_map.size_bytes > 0 && (plane_off + span) > src_map.size_bytes) {
      throw std::runtime_error("Tensor::clone: plane exceeds source mapping bounds");
    }

    // Copy row-by-row into packed destination.
    for (int64_t y = 0; y < h; ++y) {
      const auto* src_row =
          src + plane_off + static_cast<std::size_t>(y) * static_cast<std::size_t>(src_stride);
      auto* dst_row = dst + offset + static_cast<std::size_t>(y) * row_bytes;
      std::memcpy(dst_row, src_row, row_bytes);
    }

    Plane out_plane = plane;
    out_plane.byte_offset = static_cast<int64_t>(offset);
    // Contiguous plane: stride0=row_bytes, stride1=elem_bytes (InputStream expects this if
    // present).
    out_plane.strides_bytes = {static_cast<int64_t>(row_bytes), static_cast<int64_t>(elem)};
    out_planes.push_back(std::move(out_plane));

    std::size_t plane_bytes = 0;
    if (!safe_mul(row_bytes, static_cast<std::size_t>(h), &plane_bytes)) {
      throw std::runtime_error("Tensor::clone: plane_bytes overflow");
    }
    offset += plane_bytes;
  }

  Tensor out;
  out.storage = std::move(storage_copy);
  out.dtype = dtype;
  out.layout = layout;
  out.shape = shape;
  out.strides_bytes = contiguous_strides_bytes(shape, elem);
  out.byte_offset = 0;
  out.device = {DeviceType::CPU, 0};
  out.semantic = semantic;
  out.planes = std::move(out_planes);
  out.read_only = false;
  return out;
}

Tensor Tensor::contiguous() const {
  if (is_dense() && is_contiguous())
    return *this;
  return clone();
}

//==============================================================================
// Tensor: device transfers
//==============================================================================

Tensor Tensor::to(Device target) const {
  if (device.type == target.type && device.id == target.id)
    return *this;

  switch (target.type) {
  case DeviceType::CPU:
    return cpu();
  case DeviceType::SIMA_CVU:
    return cvu();
  case DeviceType::SIMA_MLA:
    return simaai::neat::pipeline_internal::transfer_to_device(*this, target, nullptr, nullptr);
  case DeviceType::SIMA_APU:
  case DeviceType::UNKNOWN:
    break;
  }
  throw std::runtime_error("Tensor::to: unsupported target device");
}

Tensor Tensor::cpu() const {
  if (device.type == DeviceType::CPU) {
    if (!storage)
      throw std::runtime_error("Tensor::cpu: missing storage");
    Mapping mapping = map_read();
    if (mapping.data)
      return *this;
  }
  return simaai::neat::pipeline_internal::transfer_to_cpu(*this);
}

Tensor Tensor::cvu() const {
  if (device.type == DeviceType::SIMA_CVU)
    return *this;
  return simaai::neat::pipeline_internal::transfer_to_device(*this, {DeviceType::SIMA_CVU, 0},
                                                             nullptr, nullptr);
}

Tensor Tensor::mla(bool force) const {
  // Policy: prefer CVU unless the caller explicitly forces MLA residency.
  if (device.type == DeviceType::SIMA_MLA)
    return *this;
  if (!force && device.type == DeviceType::SIMA_CVU)
    return *this;

  if (!force) {
    try {
      return simaai::neat::pipeline_internal::transfer_to_device(*this, {DeviceType::SIMA_CVU, 0},
                                                                 nullptr, nullptr);
    } catch (const std::exception&) {
      // Fall back to MLA (DMS0).
    }
  }
  return simaai::neat::pipeline_internal::transfer_to_device(*this, {DeviceType::SIMA_MLA, 0},
                                                             nullptr, nullptr);
}

Tensor Tensor::to_cpu_if_needed() const {
  return cpu();
}

Mapping Tensor::view(MapMode mode) const {
  if (read_only && mode != MapMode::Read) {
    throw std::runtime_error("Tensor::view: tensor is read-only");
  }
  return simaai::neat::pipeline_internal::map_tensor_view(*this, mode);
}

//==============================================================================
// Tensor: copying bytes out
//==============================================================================

std::size_t Tensor::dense_bytes_tight() const {
  if (!is_dense())
    return 0;
  const std::size_t elem = dtype_bytes(dtype);
  if (elem == 0)
    return 0;
  return dense_size_bytes_checked(shape, elem);
}

bool Tensor::copy_dense_bytes_tight_to(uint8_t* dst, std::size_t dst_size) const {
  if (!dst)
    return false;
  if (!is_dense())
    return false;

  const std::size_t elem = dtype_bytes(dtype);
  if (elem == 0)
    return false;

  const std::size_t bytes = dense_size_bytes_checked(shape, elem);
  if (bytes == 0 || dst_size < bytes)
    return false;

  const Mapping mapping = map_read();
  if (!mapping.data || mapping.size_bytes == 0)
    return false;
  if (mapping.size_bytes < bytes)
    return false;

  const auto* src = static_cast<const uint8_t*>(mapping.data);

  if (is_contiguous() || strides_bytes.empty()) {
    std::memcpy(dst, src, bytes);
    return true;
  }

  const std::vector<int64_t> src_strides =
      normalized_dense_strides_bytes(shape, elem, strides_bytes);
  return copy_dense_strided_to_contiguous(src, mapping.size_bytes, shape, src_strides, elem, dst);
}

std::vector<uint8_t> Tensor::copy_dense_bytes_tight() const {
  const std::size_t bytes = dense_bytes_tight();
  if (bytes == 0)
    throw std::runtime_error("copy_dense_bytes_tight: unknown dense size");
  std::vector<uint8_t> out(bytes);
  if (!copy_dense_bytes_tight_to(out.data(), out.size())) {
    throw std::runtime_error("copy_dense_bytes_tight: copy failed");
  }
  return out;
}

/**
 * Copy payload bytes into dst.
 *
 * Semantics:
 *  - Dense tensors: copies tight packed bytes computed from dtype+shape.
 *  - Composite tensors: copies the full mapped region as-is (planes + any padding).
 *
 * If you need packed planar bytes for known formats, prefer the explicit helpers
 * (e.g., copy_nv12_contiguous_to / copy_i420_contiguous_to) used by transfer code.
 */
bool Tensor::copy_payload_bytes_to(uint8_t* dst, std::size_t dst_size) const {
  if (!dst)
    return false;

  if (is_dense()) {
    const std::size_t bytes = dense_bytes_tight();
    if (bytes > 0)
      return copy_dense_bytes_tight_to(dst, dst_size);
  }

  const Mapping mapping = map_read();
  if (!mapping.data || mapping.size_bytes == 0)
    return false;
  if (dst_size < mapping.size_bytes)
    return false;

  std::memcpy(dst, mapping.data, mapping.size_bytes);
  return true;
}

std::vector<uint8_t> Tensor::copy_payload_bytes() const {
  const std::size_t dense_bytes = dense_bytes_tight();
  if (dense_bytes > 0) {
    std::vector<uint8_t> out(dense_bytes);
    if (!copy_dense_bytes_tight_to(out.data(), out.size())) {
      throw std::runtime_error("copy_payload_bytes: dense copy failed");
    }
    return out;
  }

  const Mapping mapping = map_read();
  if (!mapping.data || mapping.size_bytes == 0) {
    log_mapfail_once("copy_payload_bytes", *this, mapping);
    throw std::runtime_error("copy_payload_bytes: mapping failed");
  }

  std::vector<uint8_t> out(mapping.size_bytes);
  std::memcpy(out.data(), mapping.data, mapping.size_bytes);
  return out;
}

//==============================================================================
// Tensor: validation / debug
//==============================================================================

bool Tensor::validate(std::string* err) const {
  auto fail = [&](const std::string& msg) {
    if (err)
      *err = msg;
    return false;
  };

  if (is_composite() && byte_offset != 0) {
    return fail("composite tensor must have byte_offset == 0");
  }

  const std::size_t elem = dtype_bytes(dtype);
  const std::size_t storage_bytes = storage ? storage->size_bytes : 0;

  // Bounds-check planes (when storage size is known).
  for (const auto& plane : planes) {
    if (plane.byte_offset < 0)
      return fail("plane byte_offset is negative");
    if (storage_bytes == 0)
      continue;

    const std::size_t span = plane_span_bytes_checked(plane, elem);
    if (span == 0)
      return fail("plane bytes could not be computed");

    const std::size_t end = static_cast<std::size_t>(plane.byte_offset) + span;
    if (end > storage_bytes)
      return fail("plane exceeds storage bounds");
  }

  // Image semantic constraints.
  if (semantic.image.has_value()) {
    const ImageSpec::PixelFormat fmt = semantic.image->format;

    if (fmt == ImageSpec::PixelFormat::NV12 || fmt == ImageSpec::PixelFormat::I420) {
      if (dtype != simaai::neat::TensorDType::UInt8) {
        return fail("NV12/I420 tensors must use UInt8 dtype");
      }
      const int w = width();
      const int h = height();
      if (w > 0 && h > 0) {
        if ((w % 2) != 0 || (h % 2) != 0) {
          return fail("NV12/I420 tensors require even width/height");
        }
      }
    }

    auto has_role = [&](PlaneRole role) {
      for (const auto& p : planes) {
        if (p.role == role)
          return true;
      }
      return false;
    };

    if (fmt == ImageSpec::PixelFormat::NV12) {
      if (!has_role(PlaneRole::Y) || !has_role(PlaneRole::UV)) {
        return fail("NV12 tensor missing Y/UV planes");
      }
    } else if (fmt == ImageSpec::PixelFormat::I420) {
      if (!has_role(PlaneRole::Y) || !has_role(PlaneRole::U) || !has_role(PlaneRole::V)) {
        return fail("I420 tensor missing Y/U/V planes");
      }
    }
  }

  // Encoded semantic constraints.
  if (semantic.encoded.has_value()) {
    if (semantic.image.has_value() || semantic.tess.has_value()) {
      return fail("encoded tensor cannot also be image/tess");
    }
    if (dtype != simaai::neat::TensorDType::UInt8) {
      return fail("encoded tensor must use UInt8 dtype");
    }
    if (!is_dense()) {
      return fail("encoded tensor must be dense");
    }
    if (!planes.empty()) {
      return fail("encoded tensor must not have planes");
    }
    if (shape.size() != 1 || shape[0] <= 0) {
      return fail("encoded tensor must have shape [num_bytes]");
    }
    if (layout != simaai::neat::TensorLayout::Unknown) {
      return fail("encoded tensor layout must be Unknown");
    }
  }

  return true;
}

std::string Tensor::debug_string() const {
  std::string out = "Tensor{dtype=";
  out += std::string(dtype_name(dtype));

  out += " device=";
  out += std::string(device_name(device.type));
  out += ":" + std::to_string(device.id);

  out += " layout=";
  out += std::string(layout_name(layout));

  out += " shape=";
  for (size_t i = 0; i < shape.size(); ++i) {
    if (i)
      out.push_back('x');
    out += std::to_string(shape[i]);
  }

  out += " strides_bytes=";
  for (size_t i = 0; i < strides_bytes.size(); ++i) {
    if (i)
      out.push_back(',');
    out += std::to_string(strides_bytes[i]);
  }

  out += " byte_offset=" + std::to_string(byte_offset);
  out += " planes=" + std::to_string(planes.size());
  out += "}";
  return out;
}

} // namespace simaai::neat
