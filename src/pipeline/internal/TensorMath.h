#pragma once
#ifndef SIMA_NEAT_INTERNAL
#error "Internal header. Not part of the public API."
#endif

// Small shared utilities used across simaai::neat::Tensor implementation files.
//
// These are header-only by design:
//  - avoids link-time duplication / ODR issues
//  - keeps behavior consistent across multiple translation units

#include "pipeline/TensorTypes.h"

#include <cctype>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <string>
#include <string_view>
#include <vector>

namespace simaai::neat::pipeline_internal {

//------------------------------------------------------------------------------
// Safe arithmetic helpers
//------------------------------------------------------------------------------

inline bool safe_add(std::size_t a, std::size_t b, std::size_t* out) noexcept {
  if (!out)
    return false;
  if (a > (std::numeric_limits<std::size_t>::max() - b))
    return false;
  *out = a + b;
  return true;
}

inline bool safe_mul(std::size_t a, std::size_t b, std::size_t* out) noexcept {
  if (!out)
    return false;
  if (a == 0 || b == 0) {
    *out = 0;
    return true;
  }
  if (a > (std::numeric_limits<std::size_t>::max() / b))
    return false;
  *out = a * b;
  return true;
}

/**
 * Compute the number of bytes spanned by a plane according to its shape + stride.
 * This matches the access pattern used by map_nv12_read/map_i420_read and by copy loops:
 *   last touched byte ~= byte_offset + stride0*(h-1) + row_bytes.
 *
 * @return 0 if invalid or overflow.
 */
inline std::size_t plane_span_bytes_checked(const std::vector<int64_t>& shape,
                                            const std::vector<int64_t>& strides_bytes,
                                            std::size_t elem_bytes) {
  if (elem_bytes == 0)
    return 0;

  if (shape.size() >= 2) {
    const int64_t h = shape[0];
    const int64_t w = shape[1];
    if (h <= 0 || w <= 0)
      return 0;

    std::size_t row_bytes = 0;
    if (!safe_mul(static_cast<std::size_t>(w), elem_bytes, &row_bytes))
      return 0;

    int64_t stride0 = static_cast<int64_t>(row_bytes);
    if (!strides_bytes.empty())
      stride0 = strides_bytes[0];
    if (stride0 <= 0)
      return 0;
    if (static_cast<std::size_t>(stride0) < row_bytes)
      return 0;

    // If a second stride is present, require it match elem_bytes (InputStream expectation).
    if (strides_bytes.size() > 1) {
      if (strides_bytes[1] != static_cast<int64_t>(elem_bytes))
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
  if (shape.size() == 1) {
    const int64_t n = shape[0];
    if (n <= 0)
      return 0;
    std::size_t bytes = 0;
    if (!safe_mul(static_cast<std::size_t>(n), elem_bytes, &bytes))
      return 0;
    return bytes;
  }

  return 0;
}

//------------------------------------------------------------------------------
// Tensor helpers
//------------------------------------------------------------------------------

/** @return element size in bytes for dtype, or 0 if unknown. */
constexpr std::size_t dtype_bytes(TensorDType dtype) noexcept {
  switch (dtype) {
  case TensorDType::UInt8:
  case TensorDType::Int8:
    return 1;
  case TensorDType::UInt16:
  case TensorDType::Int16:
  case TensorDType::BFloat16:
    return 2;
  case TensorDType::Int32:
  case TensorDType::Float32:
    return 4;
  case TensorDType::Float64:
    return 8;
  }
  return 0;
}

/**
 * Compute row-major contiguous strides in bytes for a dense tensor.
 * Example: shape=[H,W,C], elem_bytes=1 => [W*C, C, 1]
 */
inline std::vector<int64_t> contiguous_strides_bytes(const std::vector<int64_t>& shape,
                                                     std::size_t elem_bytes) {
  if (shape.empty())
    return {};
  std::vector<int64_t> strides(shape.size(), 0);
  int64_t stride = static_cast<int64_t>(elem_bytes);
  for (int i = static_cast<int>(shape.size()) - 1; i >= 0; --i) {
    strides[static_cast<size_t>(i)] = stride;
    stride *= shape[static_cast<size_t>(i)];
  }
  return strides;
}

// Keep the older name used in some files for minimal diffs.
inline std::vector<int64_t> contiguous_strides(const std::vector<int64_t>& shape,
                                               std::size_t elem_bytes) {
  return contiguous_strides_bytes(shape, elem_bytes);
}

/**
 * Normalize a tensor-view stride vector to a projected logical tensor rank.
 *
 * MPK/MLA tensor views can carry strides for a parent tensor that has leading
 * singleton/batch dimensions while the logical consumer shape has already been
 * projected to a lower rank.  This helper preserves the concrete tensor view by
 * dropping only those leading dimensions when the source shape proves they are
 * singleton-compatible.  If the source shape is not available, callers may opt
 * into the suffix fallback, which keeps the last target-rank strides rather than
 * incorrectly treating a strided parent view as dense storage.
 */
inline std::vector<int64_t> normalize_strides_rank_to_shape(
    const std::vector<int64_t>& strides, const std::vector<int64_t>& source_shape,
    const std::vector<int64_t>& target_shape, bool allow_suffix_fallback = false) {
  if (strides.empty() || target_shape.empty() || strides.size() == target_shape.size()) {
    return strides;
  }
  if (source_shape.size() == strides.size() && source_shape.size() >= target_shape.size()) {
    const std::size_t drop = source_shape.size() - target_shape.size();
    bool can_drop = true;
    for (std::size_t i = 0; i < drop; ++i) {
      if (source_shape[i] != 1) {
        can_drop = false;
        break;
      }
    }
    for (std::size_t i = 0; can_drop && i < target_shape.size(); ++i) {
      if (target_shape[i] <= 0 || source_shape[i + drop] < target_shape[i]) {
        can_drop = false;
      }
    }
    if (can_drop) {
      return std::vector<int64_t>(strides.begin() + static_cast<std::ptrdiff_t>(drop),
                                  strides.end());
    }
  }
  if (allow_suffix_fallback && strides.size() > target_shape.size()) {
    return std::vector<int64_t>(strides.end() - static_cast<std::ptrdiff_t>(target_shape.size()),
                                strides.end());
  }
  return strides;
}

//------------------------------------------------------------------------------
// String helpers
//------------------------------------------------------------------------------

/**
 * Uppercase ASCII copy (locale-independent) used for format strings.
 */
inline std::string upper_copy(std::string s) {
  for (char& c : s) {
    c = static_cast<char>(std::toupper(static_cast<unsigned char>(c)));
  }
  return s;
}

inline std::string upper_copy(std::string_view sv) {
  return upper_copy(std::string(sv));
}

inline std::string lower_copy(std::string s) {
  for (char& c : s) {
    c = static_cast<char>(std::tolower(static_cast<unsigned char>(c)));
  }
  return s;
}

inline std::string lower_copy(std::string_view sv) {
  return lower_copy(std::string(sv));
}

inline std::string trim_copy(std::string_view sv) {
  size_t start = 0;
  size_t end = sv.size();
  while (start < end && std::isspace(static_cast<unsigned char>(sv[start]))) {
    ++start;
  }
  while (end > start && std::isspace(static_cast<unsigned char>(sv[end - 1]))) {
    --end;
  }
  return std::string(sv.substr(start, end - start));
}

inline std::string sanitize_name(std::string_view sv, const char* empty_default = "dbg") {
  std::string out;
  out.reserve(sv.size());
  for (char c : sv) {
    const bool ok = (c >= 'a' && c <= 'z') || (c >= 'A' && c <= 'Z') || (c >= '0' && c <= '9') ||
                    (c == '_' || c == '-');
    out.push_back(ok ? c : '_');
  }
  if (out.empty())
    out = empty_default ? empty_default : "dbg";
  if (!out.empty() && (out[0] >= '0' && out[0] <= '9'))
    out = "_" + out;
  return out;
}

} // namespace simaai::neat::pipeline_internal
