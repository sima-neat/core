#pragma once
#ifndef SIMA_NEAT_INTERNAL
#error "Internal header. Not part of the public API."
#endif

#include "pipeline/TensorCore.h"
#include "pipeline/internal/TensorMath.h"

#include <limits>
#include <optional>
#include <string_view>

namespace simaai::neat::pipeline_internal {

// Validate only the explicit storage envelope. Kernels remains the authority
// for tile clipping, packing and padding; Core must not reproduce that layout
// algorithm. A null result means valid, with no mapping, allocation or mutation.
inline const char*
explicit_tiled_storage_error(const Tensor& tensor, const TessSpec& tess,
                             std::optional<std::uint64_t> descriptor_bytes = {}) noexcept {
  if (tess.storage_bytes == 0)
    return nullptr; // Legacy format-only annotations retain their behavior.
  if (tensor.is_composite())
    return "explicit tiled storage cannot describe composite planes";

  const std::string_view format = tess.format;
  bool matching_dtype = false;
  switch (tensor.dtype) {
  case TensorDType::BFloat16:
    matching_dtype = format == "MLA_TILED_BF16_COMPACT" || format == "MLA_TILED_BF16_PAD16";
    break;
  case TensorDType::Int8:
    matching_dtype = format == "MLA_TILED_INT8_COMPACT" || format == "MLA_TILED_INT8_PAD16";
    break;
  case TensorDType::UInt8:
    matching_dtype = format == "MLA_TILED_UINT8_COMPACT" || format == "MLA_TILED_UINT8_PAD16";
    break;
  default:
    break;
  }
  if (!matching_dtype)
    return "explicit tiled storage format does not match tensor dtype";
  if (tess.axes.size() != 4 || tess.slice_shape.size() != 4)
    return "explicit tiled storage requires four axes and tile dimensions";
  for (std::size_t i = 0; i < 4; ++i) {
    const auto axis = tess.axes[i];
    if (tess.slice_shape[i] <= 0 || axis < -1 ||
        (axis >= 0 && static_cast<std::uint64_t>(axis) >= tensor.shape.size()) ||
        (axis == -1 && (i == 3 || tess.slice_shape[i] != 1)))
      return "explicit tiled storage has invalid axes or tile dimensions";
    for (std::size_t j = 0; j < i; ++j)
      if (axis >= 0 && axis == tess.axes[j])
        return "explicit tiled storage axes must be unique";
  }

  if (tensor.shape.empty())
    return "explicit tiled storage requires positive logical dimensions";
  std::size_t logical_bytes = dtype_bytes(tensor.dtype);
  for (const auto dimension : tensor.shape) {
    if (dimension <= 0)
      return "explicit tiled storage requires positive logical dimensions";
    if (static_cast<std::uint64_t>(dimension) > std::numeric_limits<std::size_t>::max() ||
        !safe_mul(logical_bytes, static_cast<std::size_t>(dimension), &logical_bytes))
      return "explicit tiled storage logical byte extent overflows";
  }
  if (tess.storage_bytes < logical_bytes)
    return "explicit tiled storage is smaller than its logical byte extent";
  if (descriptor_bytes && tess.storage_bytes > *descriptor_bytes)
    return "explicit tiled storage exceeds its selected descriptor extent";
  if (tensor.byte_offset < 0)
    return "explicit tiled storage byte_offset is negative";
  // A zero storage size is unknown, as in the existing Tensor plane checks.
  // Avoid adding offset and extent: even individually valid values can wrap.
  if (tensor.storage && tensor.storage->size_bytes != 0 &&
      (static_cast<std::uint64_t>(tensor.byte_offset) > tensor.storage->size_bytes ||
       tess.storage_bytes >
           tensor.storage->size_bytes - static_cast<std::size_t>(tensor.byte_offset)))
    return "explicit tiled storage exceeds selected storage bounds";
  return nullptr;
}

} // namespace simaai::neat::pipeline_internal
