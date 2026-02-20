/**
 * @file TensorImage.cpp
 * @brief Image-/YUV-specific helpers for Tensor.
 *
 * This file implements:
 *  - Lightweight width/height/channel helpers
 *  - NV12 / I420 identification
 *  - Safe mapping helpers that expose NV12/I420 planes as typed views
 *  - Helpers to pack NV12/I420 into a tight contiguous buffer (no padding)
 *
 * Semantics (confirmed by TensorCore.h + InputStream usage):
 *  - For composite tensors, plane offsets (`Plane::byte_offset`) are relative to the base
 *    pointer returned by `Tensor::map_read()` (which already includes Tensor::byte_offset).
 *    Composite tensors are expected to have `Tensor::byte_offset == 0`.
 *  - `Plane::strides_bytes[0]` (if present) is the row-stride in bytes.
 *  - For NV12/I420, dtype must be UInt8 and width/height must be even.
 *
 * TODO(repo-policy): If you want `map_nv12_read()` / `map_i420_read()` to call `validate()`
 * unconditionally (for stronger debug guarantees), decide whether the extra cost is acceptable.
 */

#include "pipeline/TensorCore.h"
#include "pipeline/internal/TensorMath.h"

#include <cstddef>
#include <cstdint>
#include <cstring>
#include <limits>
#include <optional>
#include <stdexcept>

namespace simaai::neat {

using simaai::neat::pipeline_internal::safe_add;
using simaai::neat::pipeline_internal::safe_mul;

namespace {

//==============================================================================
// Small safe-math helpers (avoid overflow in bounds computations)
//==============================================================================

/**
 * Prefer Y-plane dimensions for composite image tensors (unambiguous),
 * otherwise fall back to Tensor's shape-based heuristics.
 */
inline void resolve_image_dims_from_y_or_shape(const Tensor& t, const Plane* y_plane, int* w_out,
                                               int* h_out) {
  int w = -1;
  int h = -1;

  if (y_plane && y_plane->shape.size() >= 2) {
    h = static_cast<int>(y_plane->shape[0]);
    w = static_cast<int>(y_plane->shape[1]);
  }

  if (w <= 0 || h <= 0) {
    // Fall back to generic tensor heuristics.
    w = t.width();
    h = t.height();
  }

  if (w_out)
    *w_out = w;
  if (h_out)
    *h_out = h;
}

/** If plane stride[1] is present, require it equals expected (bytes per element). */
inline void require_stride1_if_present(const Plane& p, int64_t expected, const char* what) {
  if (p.strides_bytes.size() > 1 && p.strides_bytes[1] != expected) {
    throw std::runtime_error(std::string(what) + ": plane stride[1] mismatch");
  }
}

/**
 * Compute end offset (exclusive) for a 2D plane inside a mapped buffer.
 * end = byte_offset + stride*(rows-1) + row_bytes
 *
 * @return true if end computed successfully and does not overflow.
 */
inline bool plane_end_offset(std::size_t byte_offset, std::size_t stride, std::size_t rows,
                             std::size_t row_bytes, std::size_t* end_out) noexcept {
  if (!end_out)
    return false;
  if (rows == 0) {
    *end_out = byte_offset;
    return true;
  }

  std::size_t term = 0;
  if (!safe_mul((rows - 1), stride, &term))
    return false;
  std::size_t end = 0;
  if (!safe_add(byte_offset, term, &end))
    return false;
  if (!safe_add(end, row_bytes, &end))
    return false;
  *end_out = end;
  return true;
}

/**
 * Validate that a 2D plane view is within mapping bounds.
 * If mapping_size == 0, bounds check is skipped (unknown size).
 */
inline void require_plane_in_bounds(std::size_t mapping_size, int64_t byte_offset,
                                    int64_t stride_bytes, int rows, int row_bytes,
                                    const char* what) {
  if (byte_offset < 0) {
    throw std::runtime_error(std::string(what) + ": negative plane byte_offset");
  }
  if (stride_bytes <= 0) {
    throw std::runtime_error(std::string(what) + ": invalid plane stride");
  }
  if (rows <= 0 || row_bytes <= 0) {
    throw std::runtime_error(std::string(what) + ": invalid plane geometry");
  }

  const std::size_t off = static_cast<std::size_t>(byte_offset);
  const std::size_t stride = static_cast<std::size_t>(stride_bytes);
  const std::size_t urows = static_cast<std::size_t>(rows);
  const std::size_t urowb = static_cast<std::size_t>(row_bytes);

  std::size_t end = 0;
  if (!plane_end_offset(off, stride, urows, urowb, &end)) {
    throw std::runtime_error(std::string(what) + ": plane bounds overflow");
  }
  if (mapping_size > 0 && end > mapping_size) {
    throw std::runtime_error(std::string(what) + ": plane exceeds buffer bounds");
  }
}

/** Copy `rows` rows of `row_bytes` from src (stride src_stride) to dst (tight packed). */
inline void copy_rows(uint8_t* dst, std::size_t row_bytes, const uint8_t* src, int64_t src_stride,
                      int rows) {
  for (int y = 0; y < rows; ++y) {
    std::memcpy(dst + static_cast<std::size_t>(y) * row_bytes,
                src + static_cast<std::size_t>(y) * static_cast<std::size_t>(src_stride),
                row_bytes);
  }
}

} // namespace

//==============================================================================
// Basic shape helpers
//==============================================================================

/**
 * @return Tensor width if known.
 *
 * Rules:
 *  - For composite tensors, prefer Y-plane shape[1] when available.
 *  - Otherwise:
 *      shape.size() >= 2 => shape[1]
 *      shape.size() == 1 => shape[0]
 *      else => -1
 *
 * NOTE: For non-image 1D tensors, width()==N is a convenience; image mapping code should
 * prefer plane dims (which it does).
 */
int Tensor::width() const {
  if (is_composite()) {
    const Plane* y = try_plane(PlaneRole::Y);
    if (y && y->shape.size() >= 2) {
      return static_cast<int>(y->shape[1]);
    }
  }
  if (shape.size() >= 2)
    return static_cast<int>(shape[1]);
  if (shape.size() == 1)
    return static_cast<int>(shape[0]);
  return -1;
}

/**
 * @return Tensor height if known.
 *
 * Rules:
 *  - For composite tensors, prefer Y-plane shape[0] when available.
 *  - Otherwise:
 *      shape.size() >= 1 => shape[0]
 *      else => -1
 */
int Tensor::height() const {
  if (is_composite()) {
    const Plane* y = try_plane(PlaneRole::Y);
    if (y && y->shape.size() >= 1) {
      return static_cast<int>(y->shape[0]);
    }
  }
  if (shape.size() >= 1)
    return static_cast<int>(shape[0]);
  return -1;
}

/**
 * @return Channel count if inferable, else -1.
 *
 * Behavior:
 *  - If layout == HWC and shape has >=3 dims => shape[2]
 *  - Else for semantic.image:
 *      RGB/BGR => 3
 *      GRAY8 => 1
 *      NV12/I420 => -1 (planar YUV; channels are not represented as a single packed C)
 */
int Tensor::channels() const {
  if (layout == simaai::neat::TensorLayout::HWC && shape.size() >= 3) {
    return static_cast<int>(shape[2]);
  }
  if (semantic.image.has_value()) {
    switch (semantic.image->format) {
    case ImageSpec::PixelFormat::RGB:
      return 3;
    case ImageSpec::PixelFormat::BGR:
      return 3;
    case ImageSpec::PixelFormat::GRAY8:
      return 1;
    case ImageSpec::PixelFormat::NV12:
    case ImageSpec::PixelFormat::I420:
    case ImageSpec::PixelFormat::UNKNOWN:
      break;
    }
  }
  return -1;
}

std::optional<ImageSpec::PixelFormat> Tensor::image_format() const {
  if (!semantic.image.has_value())
    return std::nullopt;
  return semantic.image->format;
}

bool Tensor::is_nv12() const {
  return semantic.image.has_value() && semantic.image->format == ImageSpec::PixelFormat::NV12;
}

bool Tensor::is_i420() const {
  return semantic.image.has_value() && semantic.image->format == ImageSpec::PixelFormat::I420;
}

//==============================================================================
// NV12 mapping / copying
//==============================================================================

/**
 * Map an NV12 tensor for read access and return typed pointers/strides.
 *
 * Requirements:
 *  - semantic.image.format == NV12
 *  - dtype == UInt8
 *  - planes contain Y + UV
 *  - width/height even
 *
 * Returns:
 *  - std::nullopt if not NV12
 *  - Nv12Mapped containing:
 *      - mapping keepalive/unmap
 *      - pointers to Y/UV planes + row strides (bytes)
 */
std::optional<Nv12Mapped> Tensor::map_nv12_read() const {
  if (!is_nv12())
    return std::nullopt;
  if (dtype != simaai::neat::TensorDType::UInt8) {
    throw std::runtime_error("map_nv12_read: NV12 requires UInt8 dtype");
  }

  const Plane* y_plane = try_plane(PlaneRole::Y);
  const Plane* uv_plane = try_plane(PlaneRole::UV);
  if (!y_plane || !uv_plane) {
    throw std::runtime_error("map_nv12_read: missing Y/UV planes");
  }

  int w = -1, h = -1;
  resolve_image_dims_from_y_or_shape(*this, y_plane, &w, &h);
  if (w <= 0 || h <= 0) {
    throw std::runtime_error("map_nv12_read: invalid dimensions");
  }
  if ((w % 2) != 0 || (h % 2) != 0) {
    throw std::runtime_error("map_nv12_read: NV12 requires even dimensions");
  }

  // Optional shape sanity checks (only if provided).
  if (y_plane->shape.size() >= 2) {
    if (y_plane->shape[0] != h || y_plane->shape[1] != w) {
      throw std::runtime_error("map_nv12_read: Y plane shape mismatch");
    }
  }
  if (uv_plane->shape.size() >= 2) {
    if (uv_plane->shape[0] != h / 2 || uv_plane->shape[1] != w) {
      throw std::runtime_error("map_nv12_read: UV plane shape mismatch");
    }
  }

  // Strides are bytes/row. Default to tight width in bytes.
  const int64_t y_stride =
      !y_plane->strides_bytes.empty() ? y_plane->strides_bytes[0] : static_cast<int64_t>(w);
  const int64_t uv_stride =
      !uv_plane->strides_bytes.empty() ? uv_plane->strides_bytes[0] : static_cast<int64_t>(w);

  // NV12 planes are UInt8 => elem_bytes=1, so stride1 (if present) must be 1.
  require_stride1_if_present(*y_plane, 1, "map_nv12_read");
  require_stride1_if_present(*uv_plane, 1, "map_nv12_read");

  if (y_stride < w || uv_stride < w) {
    throw std::runtime_error("map_nv12_read: invalid plane stride");
  }

  Mapping mapping = map_read();
  if (!mapping.data) {
    throw std::runtime_error("map_nv12_read: mapping failed");
  }

  const std::size_t total = mapping.size_bytes;
  require_plane_in_bounds(total, y_plane->byte_offset, y_stride, h, w, "map_nv12_read(Y)");
  require_plane_in_bounds(total, uv_plane->byte_offset, uv_stride, h / 2, w, "map_nv12_read(UV)");

  const uint8_t* base = static_cast<const uint8_t*>(mapping.data);

  Nv12Mapped out;
  out.mapping = std::move(mapping);
  out.view.width = w;
  out.view.height = h;
  out.view.y = base + y_plane->byte_offset;
  out.view.y_stride = y_stride;
  out.view.uv = base + uv_plane->byte_offset;
  out.view.uv_stride = uv_stride;
  return out;
}

/**
 * @return Tight packed NV12 size in bytes (no padding): w*h*3/2.
 * Returns 0 if not NV12 or dims unknown.
 */
std::size_t Tensor::nv12_required_bytes() const {
  if (!is_nv12())
    return 0;
  const int w = width();
  const int h = height();
  if (w <= 0 || h <= 0)
    return 0;
  return static_cast<std::size_t>(w) * static_cast<std::size_t>(h) * 3 / 2;
}

/**
 * Pack NV12 into a tight contiguous buffer:
 *  - copies Y plane rows (w bytes per row) into dst[0 : w*h)
 *  - copies UV plane rows (w bytes per row, h/2 rows) into dst[w*h : w*h*3/2)
 */
bool Tensor::copy_nv12_contiguous_to(uint8_t* dst, std::size_t dst_size) const {
  if (!dst)
    return false;
  const std::size_t required = nv12_required_bytes();
  if (required == 0 || dst_size < required)
    return false;

  auto mapped = map_nv12_read();
  if (!mapped.has_value())
    return false;

  const Nv12View& view = mapped->view;
  const int w = view.width;
  const int h = view.height;
  const int uv_h = h / 2;

  copy_rows(dst, static_cast<std::size_t>(w), view.y, view.y_stride, h);

  uint8_t* dst_uv = dst + static_cast<std::size_t>(w) * static_cast<std::size_t>(h);
  copy_rows(dst_uv, static_cast<std::size_t>(w), view.uv, view.uv_stride, uv_h);

  return true;
}

std::vector<uint8_t> Tensor::copy_nv12_contiguous() const {
  const std::size_t required = nv12_required_bytes();
  if (required == 0) {
    throw std::runtime_error("copy_nv12_contiguous: not an NV12 tensor");
  }
  std::vector<uint8_t> out(required);
  if (!copy_nv12_contiguous_to(out.data(), out.size())) {
    throw std::runtime_error("copy_nv12_contiguous: copy failed");
  }
  return out;
}

//==============================================================================
// I420 mapping / copying
//==============================================================================

/**
 * Map an I420 tensor for read access and return typed pointers/strides.
 *
 * Requirements:
 *  - semantic.image.format == I420
 *  - dtype == UInt8
 *  - planes contain Y + U + V
 *  - width/height even
 *
 * Returns:
 *  - std::nullopt if not I420
 *  - I420Mapped containing:
 *      - mapping keepalive/unmap
 *      - pointers to Y/U/V planes + row strides (bytes)
 */
std::optional<I420Mapped> Tensor::map_i420_read() const {
  if (!is_i420())
    return std::nullopt;
  if (dtype != simaai::neat::TensorDType::UInt8) {
    throw std::runtime_error("map_i420_read: I420 requires UInt8 dtype");
  }

  const Plane* y_plane = try_plane(PlaneRole::Y);
  const Plane* u_plane = try_plane(PlaneRole::U);
  const Plane* v_plane = try_plane(PlaneRole::V);
  if (!y_plane || !u_plane || !v_plane) {
    throw std::runtime_error("map_i420_read: missing Y/U/V planes");
  }

  int w = -1, h = -1;
  resolve_image_dims_from_y_or_shape(*this, y_plane, &w, &h);
  if (w <= 0 || h <= 0) {
    throw std::runtime_error("map_i420_read: invalid dimensions");
  }
  if ((w % 2) != 0 || (h % 2) != 0) {
    throw std::runtime_error("map_i420_read: I420 requires even dimensions");
  }

  // Optional shape sanity checks (only if provided).
  if (y_plane->shape.size() >= 2) {
    if (y_plane->shape[0] != h || y_plane->shape[1] != w) {
      throw std::runtime_error("map_i420_read: Y plane shape mismatch");
    }
  }
  if (u_plane->shape.size() >= 2) {
    if (u_plane->shape[0] != h / 2 || u_plane->shape[1] != w / 2) {
      throw std::runtime_error("map_i420_read: U plane shape mismatch");
    }
  }
  if (v_plane->shape.size() >= 2) {
    if (v_plane->shape[0] != h / 2 || v_plane->shape[1] != w / 2) {
      throw std::runtime_error("map_i420_read: V plane shape mismatch");
    }
  }

  const int64_t y_stride =
      !y_plane->strides_bytes.empty() ? y_plane->strides_bytes[0] : static_cast<int64_t>(w);
  const int64_t u_stride =
      !u_plane->strides_bytes.empty() ? u_plane->strides_bytes[0] : static_cast<int64_t>(w / 2);
  const int64_t v_stride =
      !v_plane->strides_bytes.empty() ? v_plane->strides_bytes[0] : static_cast<int64_t>(w / 2);

  // I420 planes are UInt8 => elem_bytes=1, so stride1 (if present) must be 1.
  require_stride1_if_present(*y_plane, 1, "map_i420_read");
  require_stride1_if_present(*u_plane, 1, "map_i420_read");
  require_stride1_if_present(*v_plane, 1, "map_i420_read");

  if (y_stride < w || u_stride < (w / 2) || v_stride < (w / 2)) {
    throw std::runtime_error("map_i420_read: invalid plane stride");
  }

  Mapping mapping = map_read();
  if (!mapping.data) {
    throw std::runtime_error("map_i420_read: mapping failed");
  }

  const std::size_t total = mapping.size_bytes;
  require_plane_in_bounds(total, y_plane->byte_offset, y_stride, h, w, "map_i420_read(Y)");
  require_plane_in_bounds(total, u_plane->byte_offset, u_stride, h / 2, w / 2, "map_i420_read(U)");
  require_plane_in_bounds(total, v_plane->byte_offset, v_stride, h / 2, w / 2, "map_i420_read(V)");

  const uint8_t* base = static_cast<const uint8_t*>(mapping.data);

  I420Mapped out;
  out.mapping = std::move(mapping);
  out.view.width = w;
  out.view.height = h;
  out.view.y = base + y_plane->byte_offset;
  out.view.y_stride = y_stride;
  out.view.u = base + u_plane->byte_offset;
  out.view.u_stride = u_stride;
  out.view.v = base + v_plane->byte_offset;
  out.view.v_stride = v_stride;
  return out;
}

/**
 * @return Tight packed I420 size in bytes (no padding): w*h*3/2.
 * Returns 0 if not I420 or dims unknown.
 */
std::size_t Tensor::i420_required_bytes() const {
  if (!is_i420())
    return 0;
  const int w = width();
  const int h = height();
  if (w <= 0 || h <= 0)
    return 0;
  return static_cast<std::size_t>(w) * static_cast<std::size_t>(h) * 3 / 2;
}

/**
 * Pack I420 into a tight contiguous buffer:
 *  - copies Y plane rows (w bytes per row) into dst[0 : w*h)
 *  - copies U plane rows (w/2 bytes per row, h/2 rows) into dst[w*h : w*h + (w/2)*(h/2))
 *  - copies V plane rows (w/2 bytes per row, h/2 rows) immediately after U
 */
bool Tensor::copy_i420_contiguous_to(uint8_t* dst, std::size_t dst_size) const {
  if (!dst)
    return false;
  const std::size_t required = i420_required_bytes();
  if (required == 0 || dst_size < required)
    return false;

  auto mapped = map_i420_read();
  if (!mapped.has_value())
    return false;

  const I420View& view = mapped->view;
  const int w = view.width;
  const int h = view.height;
  const int uv_w = w / 2;
  const int uv_h = h / 2;

  copy_rows(dst, static_cast<std::size_t>(w), view.y, view.y_stride, h);

  uint8_t* dst_u = dst + static_cast<std::size_t>(w) * static_cast<std::size_t>(h);
  copy_rows(dst_u, static_cast<std::size_t>(uv_w), view.u, view.u_stride, uv_h);

  uint8_t* dst_v = dst_u + static_cast<std::size_t>(uv_w) * static_cast<std::size_t>(uv_h);
  copy_rows(dst_v, static_cast<std::size_t>(uv_w), view.v, view.v_stride, uv_h);

  return true;
}

std::vector<uint8_t> Tensor::copy_i420_contiguous() const {
  const std::size_t required = i420_required_bytes();
  if (required == 0) {
    throw std::runtime_error("copy_i420_contiguous: not an I420 tensor");
  }
  std::vector<uint8_t> out(required);
  if (!copy_i420_contiguous_to(out.data(), out.size())) {
    throw std::runtime_error("copy_i420_contiguous: copy failed");
  }
  return out;
}

} // namespace simaai::neat