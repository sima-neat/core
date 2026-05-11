/**
 * @file
 * @ingroup internal_sima
 * @brief **Framework internal — reach-through tier.** Tensor-semantics helpers (layout tokens,
 *        axis semantics, `sima_ev_*` descriptor builders).
 *
 * Header-only utilities used throughout the planner and contract assembly to:
 *   - normalize layout tokens (NCHW / CHW / HWC / HW),
 *   - convert between `TensorAxisSemantic` and the `sima_ev_axis_semantic` ABI enum,
 *   - infer axis semantics from a (shape, layout-token) pair,
 *   - build `sima_ev_shape_desc` / `sima_ev_strided_layout_desc` / `sima_ev_tensor_desc`
 *     records (dense or tiled, with or without axis-semantic stamping).
 *
 * All functions are inline so the helpers can sit on hot paths without TU boundaries.
 *
 * @see sima_ev_tensor_abi.h (the ABI types these helpers populate)
 */
#pragma once
#ifndef SIMA_NEAT_INTERNAL
#error "Internal header. Not part of the public API."
#endif

#include "pipeline/TensorTypes.h"
#include "gst/SimaPluginStaticManifestAbi.h"

#include <ev/ev_tensor_abi.h>

#include <algorithm>
#include <cctype>
#include <cstdint>
#include <cstring>
#include <limits>
#include <optional>
#include <string>
#include <string_view>
#include <vector>

namespace simaai::neat::pipeline_internal::sima::tensorsemantics {

/// Returns an ASCII-uppercased copy of `value`.
inline std::string upper_copy_ascii(std::string value) {
  std::transform(value.begin(), value.end(), value.begin(),
                 [](unsigned char c) { return static_cast<char>(std::toupper(c)); });
  return value;
}

/**
 * @brief Canonicalize a layout token to one of `"CHW"`, `"HWC"`, `"HW"`, or empty.
 *
 * Maps `"NCHW"` → `"CHW"` and `"NHWC"` → `"HWC"` (the leading-N is implicit in our model).
 * Returns the empty string for unknown tokens.
 */
inline std::string normalize_layout_token(std::string raw_layout) {
  const std::string layout = upper_copy_ascii(std::move(raw_layout));
  if (layout == "NCHW" || layout == "CHW") {
    return "CHW";
  }
  if (layout == "NHWC" || layout == "HWC") {
    return "HWC";
  }
  if (layout == "HW") {
    return "HW";
  }
  return {};
}

/// Convert a raw `sima_ev_axis_semantic` byte into the public `TensorAxisSemantic` enum.
inline TensorAxisSemantic from_ev_axis(std::uint8_t axis) noexcept {
  switch (static_cast<sima_ev_axis_semantic>(axis)) {
  case SIMA_EV_AXIS_N:
    return TensorAxisSemantic::N;
  case SIMA_EV_AXIS_D:
    return TensorAxisSemantic::D;
  case SIMA_EV_AXIS_H:
    return TensorAxisSemantic::H;
  case SIMA_EV_AXIS_W:
    return TensorAxisSemantic::W;
  case SIMA_EV_AXIS_C:
    return TensorAxisSemantic::C;
  case SIMA_EV_AXIS_UNKNOWN:
  default:
    return TensorAxisSemantic::Unknown;
  }
}

/// Convert a `TensorAxisSemantic` value into the raw `sima_ev_axis_semantic` byte.
inline std::uint8_t to_ev_axis(TensorAxisSemantic axis) noexcept {
  switch (axis) {
  case TensorAxisSemantic::N:
    return SIMA_EV_AXIS_N;
  case TensorAxisSemantic::D:
    return SIMA_EV_AXIS_D;
  case TensorAxisSemantic::H:
    return SIMA_EV_AXIS_H;
  case TensorAxisSemantic::W:
    return SIMA_EV_AXIS_W;
  case TensorAxisSemantic::C:
    return SIMA_EV_AXIS_C;
  case TensorAxisSemantic::Unknown:
  default:
    return SIMA_EV_AXIS_UNKNOWN;
  }
}

/// Lift the per-axis semantics out of a `sima_ev_shape_desc` into a host vector.
inline std::vector<TensorAxisSemantic> host_axis_semantics_from_ev(const sima_ev_shape_desc& shape) {
  const auto rank = std::min<std::uint32_t>(shape.rank, SIMA_EV_MAX_RANK);
  std::vector<TensorAxisSemantic> out;
  out.reserve(rank);
  for (std::uint32_t i = 0; i < rank; ++i) {
    out.push_back(from_ev_axis(shape.axis_semantics[i]));
  }
  return out;
}

/// Lift a raw axis-semantic byte vector into a `TensorAxisSemantic` vector.
inline std::vector<TensorAxisSemantic> host_axis_semantics_from_raw(
    const std::vector<std::uint8_t>& raw_axes) {
  std::vector<TensorAxisSemantic> out;
  out.reserve(raw_axes.size());
  for (const std::uint8_t axis : raw_axes) {
    out.push_back(from_ev_axis(axis));
  }
  return out;
}

/// True when `axes` is empty (unspecified) or has exactly `rank` entries.
inline bool axis_semantics_match_rank(const std::size_t rank,
                                      const std::vector<TensorAxisSemantic>& axes) noexcept {
  return axes.empty() || axes.size() == rank;
}

/// Raw-byte overload of `axis_semantics_match_rank`.
inline bool axis_semantics_match_rank(const std::size_t rank,
                                      const std::vector<std::uint8_t>& raw_axes) noexcept {
  return raw_axes.empty() || raw_axes.size() == rank;
}

/**
 * @brief Derive a layout token (`"HW"` / `"HWC"` / `"CHW"`) from an axis-semantic vector.
 *
 * Recognizes both the bare 2/3-D forms and the leading-N 4-D forms (e.g., `[N,H,W,C]`).
 * Returns the empty string for shapes that don't match a known canonical layout.
 */
inline std::string layout_token_from_axis_semantics(const std::vector<TensorAxisSemantic>& axes) {
  if (axes.empty()) {
    return {};
  }
  const auto is_exact = [&](std::initializer_list<TensorAxisSemantic> expected) {
    return axes.size() == expected.size() &&
           std::equal(axes.begin(), axes.end(), expected.begin(), expected.end());
  };
  if (is_exact({TensorAxisSemantic::H, TensorAxisSemantic::W})) {
    return "HW";
  }
  if (is_exact({TensorAxisSemantic::H, TensorAxisSemantic::W, TensorAxisSemantic::C}) ||
      is_exact({TensorAxisSemantic::N, TensorAxisSemantic::H, TensorAxisSemantic::W,
                TensorAxisSemantic::C})) {
    return "HWC";
  }
  if (is_exact({TensorAxisSemantic::C, TensorAxisSemantic::H, TensorAxisSemantic::W}) ||
      is_exact({TensorAxisSemantic::N, TensorAxisSemantic::C, TensorAxisSemantic::H,
                TensorAxisSemantic::W})) {
    return "CHW";
  }
  return {};
}

/// Derive a layout token from a `sima_ev_shape_desc`.
inline std::string layout_token_from_ev_shape(const sima_ev_shape_desc& shape) {
  return layout_token_from_axis_semantics(host_axis_semantics_from_ev(shape));
}

/// Derive a layout token from a `sima_ev_tensor_desc` (uses its `.shape`).
inline std::string layout_token_from_ev_tensor_desc(const sima_ev_tensor_desc& tensor) {
  return layout_token_from_ev_shape(tensor.shape);
}

/**
 * @brief Check that a raw layout token is consistent with an axis-semantic vector.
 *
 * Returns true when either input is empty (unspecified). Returns false when the layout
 * normalizes to a non-empty token that disagrees with the layout derived from `axes`.
 */
inline bool layout_token_matches_axis_semantics(
    const std::string& raw_layout, const std::vector<TensorAxisSemantic>& axes) {
  if (raw_layout.empty() || axes.empty()) {
    return true;
  }
  const std::string normalized = normalize_layout_token(raw_layout);
  if (normalized.empty()) {
    return false;
  }
  const std::string derived = layout_token_from_axis_semantics(axes);
  return derived.empty() || normalized == derived;
}

/// Raw-byte overload of `layout_token_matches_axis_semantics`.
inline bool layout_token_matches_axis_semantics(
    const std::string& raw_layout, const std::vector<std::uint8_t>& raw_axes) {
  return layout_token_matches_axis_semantics(raw_layout, host_axis_semantics_from_raw(raw_axes));
}

/// Stamp axis-semantic bytes into a `sima_ev_shape_desc` from a host vector.
inline void fill_ev_axis_semantics(const std::vector<TensorAxisSemantic>& in,
                                   sima_ev_shape_desc* out) {
  if (!out) {
    return;
  }
  for (std::uint32_t i = 0; i < SIMA_EV_MAX_RANK; ++i) {
    out->axis_semantics[i] = SIMA_EV_AXIS_UNKNOWN;
  }
  const auto rank = std::min<std::uint32_t>(std::min<std::size_t>(in.size(), out->rank),
                                            static_cast<std::size_t>(SIMA_EV_MAX_RANK));
  for (std::uint32_t i = 0; i < rank; ++i) {
    out->axis_semantics[i] = to_ev_axis(in[i]);
  }
}

/// Index of the first axis with `semantic`, or `nullopt` if none.
inline std::optional<std::size_t> find_axis(const std::vector<TensorAxisSemantic>& axes,
                                            TensorAxisSemantic semantic) {
  const auto it = std::find(axes.begin(), axes.end(), semantic);
  if (it == axes.end()) {
    return std::nullopt;
  }
  return static_cast<std::size_t>(std::distance(axes.begin(), it));
}

/**
 * @brief Stamp axis-semantic bytes from `(shape, layout_token)` into a raw byte buffer.
 *
 * Handles 4-D leading-N tensors (when shape[0] == 1) for both CHW and HWC layouts. Falls back
 * to a generic best-guess (C/W/H/D/N from fastest to slowest) for unrecognized layouts.
 */
template <typename ShapeT>
inline void fill_axis_semantics_from_shape_layout(const std::vector<ShapeT>& shape,
                                                  const std::string& raw_layout,
                                                  std::uint8_t* semantics) {
  if (!semantics) {
    return;
  }
  for (std::uint32_t i = 0; i < SIMA_EV_MAX_RANK; ++i) {
    semantics[i] = SIMA_EV_AXIS_UNKNOWN;
  }
  const std::uint32_t rank = static_cast<std::uint32_t>(shape.size());
  const std::string layout = normalize_layout_token(raw_layout);
  const bool leading_batch =
      rank >= 4U && !shape.empty() && shape.front() == static_cast<ShapeT>(1) &&
      (layout == "CHW" || layout == "HWC");
  if (layout == "CHW") {
    if (leading_batch) {
      semantics[0] = SIMA_EV_AXIS_N;
      if (rank > 1U) semantics[1] = SIMA_EV_AXIS_C;
      if (rank > 2U) semantics[2] = SIMA_EV_AXIS_H;
      if (rank > 3U) semantics[3] = SIMA_EV_AXIS_W;
      return;
    }
    if (rank == 3U) {
      semantics[0] = SIMA_EV_AXIS_C;
      semantics[1] = SIMA_EV_AXIS_H;
      semantics[2] = SIMA_EV_AXIS_W;
      return;
    }
  } else if (layout == "HWC") {
    if (leading_batch) {
      semantics[0] = SIMA_EV_AXIS_N;
      if (rank > 1U) semantics[1] = SIMA_EV_AXIS_H;
      if (rank > 2U) semantics[2] = SIMA_EV_AXIS_W;
      if (rank > 3U) semantics[3] = SIMA_EV_AXIS_C;
      return;
    }
    if (rank == 3U) {
      semantics[0] = SIMA_EV_AXIS_H;
      semantics[1] = SIMA_EV_AXIS_W;
      semantics[2] = SIMA_EV_AXIS_C;
      return;
    }
    if (rank == 2U) {
      semantics[0] = SIMA_EV_AXIS_H;
      semantics[1] = SIMA_EV_AXIS_W;
      return;
    }
  } else if (layout == "HW" && rank == 2U) {
    semantics[0] = SIMA_EV_AXIS_H;
    semantics[1] = SIMA_EV_AXIS_W;
    return;
  }
  std::uint32_t cursor = std::min<std::uint32_t>(rank, SIMA_EV_MAX_RANK);
  if (cursor > 0U) semantics[--cursor] = SIMA_EV_AXIS_C;
  if (cursor > 0U) semantics[--cursor] = SIMA_EV_AXIS_W;
  if (cursor > 0U) semantics[--cursor] = SIMA_EV_AXIS_H;
  if (cursor > 0U) semantics[--cursor] = SIMA_EV_AXIS_D;
  if (cursor > 0U) semantics[--cursor] = SIMA_EV_AXIS_N;
}

/// Set every entry of a `SIMA_EV_MAX_RANK`-sized axis-semantic buffer to `UNKNOWN`.
inline void clear_axis_semantics(std::uint8_t* semantics) {
  if (!semantics) {
    return;
  }
  for (std::uint32_t i = 0; i < SIMA_EV_MAX_RANK; ++i) {
    semantics[i] = SIMA_EV_AXIS_UNKNOWN;
  }
}

/// Position of the first axis in `shape` whose semantic equals `axis`; -1 if none.
inline int find_shape_axis(const sima_ev_shape_desc& shape, sima_ev_axis_semantic axis) {
  const auto rank = std::min<std::uint32_t>(shape.rank, SIMA_EV_MAX_RANK);
  for (std::uint32_t i = 0; i < rank; ++i) {
    if (static_cast<sima_ev_axis_semantic>(shape.axis_semantics[i]) == axis) {
      return static_cast<int>(i);
    }
  }
  return -1;
}

/**
 * @brief Fill a `sima_ev_shape_desc` from a host shape vector + layout token.
 *
 * Stamps axis semantics derived from the layout. Returns false (and writes the supplied
 * descriptive error string into `error_detail`) on missing-output, invalid-rank, or
 * non-positive-dim conditions.
 */
template <typename ShapeT>
inline bool fill_shape_desc(const std::vector<ShapeT>& shape,
                            const std::string& layout_token,
                            sima_ev_shape_desc* out,
                            std::string* error_detail,
                            std::string_view missing_output_msg,
                            std::string_view rank_invalid_msg,
                            std::string_view dim_invalid_msg) {
  if (!out) {
    if (error_detail) {
      *error_detail = std::string(missing_output_msg);
    }
    return false;
  }
  std::memset(out, 0, sizeof(*out));
  if (shape.empty() || shape.size() > SIMA_EV_MAX_RANK) {
    if (error_detail) {
      *error_detail = std::string(rank_invalid_msg);
    }
    return false;
  }
  out->rank = static_cast<std::uint32_t>(shape.size());
  fill_axis_semantics_from_shape_layout(shape, layout_token, out->axis_semantics);
  for (std::size_t i = 0; i < shape.size(); ++i) {
    if (shape[i] <= 0) {
      if (error_detail) {
        *error_detail = std::string(dim_invalid_msg);
      }
      return false;
    }
    out->sizes[i] = static_cast<std::int64_t>(shape[i]);
  }
  return true;
}

/// Same as `fill_shape_desc` but leaves `axis_semantics` empty (UNKNOWN) — used for
/// generic non-image tensors where the canonical CHW/HWC mapping doesn't apply.
template <typename ShapeT>
inline bool fill_shape_desc_without_axis_semantics(const std::vector<ShapeT>& shape,
                                                   sima_ev_shape_desc* out,
                                                   std::string* error_detail,
                                                   std::string_view missing_output_msg,
                                                   std::string_view rank_invalid_msg,
                                                   std::string_view dim_invalid_msg) {
  if (!out) {
    if (error_detail) {
      *error_detail = std::string(missing_output_msg);
    }
    return false;
  }
  std::memset(out, 0, sizeof(*out));
  if (shape.empty() || shape.size() > SIMA_EV_MAX_RANK) {
    if (error_detail) {
      *error_detail = std::string(rank_invalid_msg);
    }
    return false;
  }
  out->rank = static_cast<std::uint32_t>(shape.size());
  clear_axis_semantics(out->axis_semantics);
  for (std::size_t i = 0; i < shape.size(); ++i) {
    if (shape[i] <= 0) {
      if (error_detail) {
        *error_detail = std::string(dim_invalid_msg);
      }
      return false;
    }
    out->sizes[i] = static_cast<std::int64_t>(shape[i]);
  }
  return true;
}

/**
 * @brief Compute dense byte strides for a tensor according to the canonical layout order.
 *
 * For CHW the fastest axis is W → H → C; for HWC it is C → W → H. Other axes (D, N) follow
 * in slower order. Returns false on missing-output or invalid-dtype conditions.
 */
inline bool fill_dense_strides(const sima_ev_shape_desc& shape,
                               const std::string& raw_layout,
                               const std::uint32_t dtype,
                               sima_ev_strided_layout_desc* out,
                               std::string* error_detail,
                               std::string_view missing_output_msg,
                               std::string_view dtype_invalid_msg) {
  if (!out) {
    if (error_detail) {
      *error_detail = std::string(missing_output_msg);
    }
    return false;
  }
  std::memset(out, 0, sizeof(*out));
  const int elem_bytes = sima_ev_elem_size_bytes(dtype);
  if (elem_bytes <= 0) {
    if (error_detail) {
      *error_detail = std::string(dtype_invalid_msg);
    }
    return false;
  }
  const std::string layout = normalize_layout_token(raw_layout);
  const bool channel_first = layout == "CHW";
  std::vector<int> fastest_axes;
  fastest_axes.reserve(shape.rank);
  auto maybe_push_axis = [&](sima_ev_axis_semantic axis) {
    const int idx = find_shape_axis(shape, axis);
    if (idx >= 0 && std::find(fastest_axes.begin(), fastest_axes.end(), idx) == fastest_axes.end()) {
      fastest_axes.push_back(idx);
    }
  };
  if (channel_first) {
    maybe_push_axis(SIMA_EV_AXIS_W);
    maybe_push_axis(SIMA_EV_AXIS_H);
    maybe_push_axis(SIMA_EV_AXIS_C);
  } else {
    maybe_push_axis(SIMA_EV_AXIS_C);
    maybe_push_axis(SIMA_EV_AXIS_W);
    maybe_push_axis(SIMA_EV_AXIS_H);
  }
  maybe_push_axis(SIMA_EV_AXIS_D);
  maybe_push_axis(SIMA_EV_AXIS_N);
  for (std::uint32_t i = 0; i < std::min<std::uint32_t>(shape.rank, SIMA_EV_MAX_RANK); ++i) {
    if (std::find(fastest_axes.begin(), fastest_axes.end(), static_cast<int>(i)) ==
        fastest_axes.end()) {
      fastest_axes.push_back(static_cast<int>(i));
    }
  }
  std::int64_t stride = elem_bytes;
  for (const int axis : fastest_axes) {
    out->strides_bytes[axis] = stride;
    if (shape.sizes[axis] > 0 &&
        stride <= std::numeric_limits<std::int64_t>::max() / shape.sizes[axis]) {
      stride *= shape.sizes[axis];
    }
  }
  return true;
}

/// Compute strides for a fully contiguous (last-dim fastest) tensor — used when no canonical
/// axis-semantic layout applies.
inline bool fill_dense_strides_contiguous(const sima_ev_shape_desc& shape,
                                          const std::uint32_t dtype,
                                          sima_ev_strided_layout_desc* out,
                                          std::string* error_detail,
                                          std::string_view missing_output_msg,
                                          std::string_view dtype_invalid_msg) {
  if (!out) {
    if (error_detail) {
      *error_detail = std::string(missing_output_msg);
    }
    return false;
  }
  std::memset(out, 0, sizeof(*out));
  const int elem_bytes = sima_ev_elem_size_bytes(dtype);
  if (elem_bytes <= 0) {
    if (error_detail) {
      *error_detail = std::string(dtype_invalid_msg);
    }
    return false;
  }
  std::int64_t stride = elem_bytes;
  const auto rank = std::min<std::uint32_t>(shape.rank, SIMA_EV_MAX_RANK);
  for (std::uint32_t rev = 0; rev < rank; ++rev) {
    const std::uint32_t axis = rank - 1U - rev;
    out->strides_bytes[axis] = stride;
    if (shape.sizes[axis] > 0 &&
        stride <= std::numeric_limits<std::int64_t>::max() / shape.sizes[axis]) {
      stride *= shape.sizes[axis];
    }
  }
  return true;
}

/**
 * @brief Map a dtype token (e.g., `"FP32"`, `"BF16"`, `"INT8"`) to its `SIMA_EV_DTYPE_*` value.
 *
 * Recognizes both the bare tokens (`"FP32"`, `"BF16"`, etc.) and the `EVXX_*` prefixed tokens.
 * `UINT8` is mapped to `INT8`. Returns false for unknown tokens.
 */
inline bool dtype_token_to_ev(std::string raw_dtype, std::uint32_t* out_dtype) {
  if (!out_dtype) {
    return false;
  }
  const std::string token = upper_copy_ascii(std::move(raw_dtype));
  if (token == "FP32" || token == "FLOAT32" || token == "EVXX_FLOAT32") {
    *out_dtype = SIMA_EV_DTYPE_FP32;
    return true;
  }
  if (token == "BF16" || token == "BFLOAT16" || token == "EVXX_BFLOAT16") {
    *out_dtype = SIMA_EV_DTYPE_BF16;
    return true;
  }
  if (token == "FP16" || token == "FLOAT16" || token == "EVXX_FLOAT16") {
    *out_dtype = SIMA_EV_DTYPE_FP16;
    return true;
  }
  if (token == "INT32" || token == "EVXX_INT32") {
    *out_dtype = SIMA_EV_DTYPE_INT32;
    return true;
  }
  if (token == "INT16" || token == "EVXX_INT16") {
    *out_dtype = SIMA_EV_DTYPE_INT16;
    return true;
  }
  if (token == "INT8" || token == "EVXX_INT8" || token == "UINT8" || token == "EVXX_UINT8") {
    *out_dtype = SIMA_EV_DTYPE_INT8;
    return true;
  }
  return false;
}

/**
 * @brief Total byte size of a tiled tensor with fixed-size tile slots.
 *
 * Each tile occupies a slot of `align(tile_elems × elem_bytes, tile_align_bytes)`. Returns 0
 * on any inconsistency (rank mismatch, non-positive dim, dtype unknown, overflow).
 */
template <typename ShapeT, typename TileShapeT>
inline std::uint64_t generic_fixed_slot_tiled_size_bytes(
    const std::vector<ShapeT>& shape, const std::vector<TileShapeT>& tile_shape,
    const std::string& dtype_token, const std::uint32_t tile_align_bytes) {
  if (shape.empty() || shape.size() != tile_shape.size()) {
    return 0U;
  }
  std::uint32_t dtype = SIMA_EV_DTYPE_INT8;
  if (!dtype_token_to_ev(dtype_token, &dtype)) {
    return 0U;
  }
  const int elem_bytes_i = sima_ev_elem_size_bytes(dtype);
  if (elem_bytes_i <= 0) {
    return 0U;
  }
  const auto elem_bytes = static_cast<std::uint64_t>(elem_bytes_i);
  std::uint64_t full_tile_elems = 1U;
  std::uint64_t tile_count_product = 1U;
  for (std::size_t i = 0; i < shape.size(); ++i) {
    if (shape[i] <= 0 || tile_shape[i] <= 0 || tile_shape[i] > shape[i]) {
      return 0U;
    }
    const auto dim = static_cast<std::uint64_t>(shape[i]);
    const auto tile = static_cast<std::uint64_t>(tile_shape[i]);
    const auto tile_count = (dim + tile - 1U) / tile;
    if (full_tile_elems > std::numeric_limits<std::uint64_t>::max() / tile ||
        tile_count_product > std::numeric_limits<std::uint64_t>::max() / tile_count) {
      return 0U;
    }
    full_tile_elems *= tile;
    tile_count_product *= tile_count;
  }
  if (full_tile_elems > std::numeric_limits<std::uint64_t>::max() / elem_bytes) {
    return 0U;
  }
  const std::uint64_t full_tile_payload = full_tile_elems * elem_bytes;
  std::uint64_t full_tile_bytes = full_tile_payload;
  if (tile_align_bytes != 0U) {
    const auto align = static_cast<std::uint64_t>(tile_align_bytes);
    full_tile_bytes = ((full_tile_payload + align - 1U) / align) * align;
  }
  if (tile_count_product != 0U &&
      full_tile_bytes > std::numeric_limits<std::uint64_t>::max() / tile_count_product) {
    return 0U;
  }
  return full_tile_bytes * tile_count_product;
}

/**
 * @brief Normalize a raw tile-shape vector to match the rank of `shape`.
 *
 * Trims leading 1s when the tile shape is longer; left-pads with 1s when shorter; rejects any
 * non-leading dimension that is non-positive or larger than its corresponding shape dim.
 */
inline bool normalize_tile_shape(const std::vector<int>& shape,
                                 const std::vector<int>& raw_tile_shape,
                                 std::vector<int>* out,
                                 std::string* error_detail,
                                 std::string_view missing_msg,
                                 std::string_view rank_prefix_invalid_msg,
                                 std::string_view dim_invalid_msg) {
  if (!out) {
    if (error_detail) {
      *error_detail = "tile_shape_output_storage_missing";
    }
    return false;
  }
  out->clear();
  if (shape.empty() || raw_tile_shape.empty()) {
    if (error_detail) {
      *error_detail = std::string(missing_msg);
    }
    return false;
  }
  std::vector<int> normalized = raw_tile_shape;
  if (normalized.size() > shape.size()) {
    const std::size_t extra = normalized.size() - shape.size();
    for (std::size_t i = 0; i < extra; ++i) {
      if (normalized[i] != 1) {
        if (error_detail) {
          *error_detail = std::string(rank_prefix_invalid_msg);
        }
        return false;
      }
    }
    normalized.erase(normalized.begin(), normalized.begin() + static_cast<std::ptrdiff_t>(extra));
  } else if (normalized.size() < shape.size()) {
    normalized.insert(normalized.begin(), shape.size() - normalized.size(), 1);
  }
  for (std::size_t i = 0; i < normalized.size(); ++i) {
    if (normalized[i] <= 0 || normalized[i] > shape[i]) {
      if (error_detail) {
        *error_detail = std::string(dim_invalid_msg);
      }
      return false;
    }
  }
  *out = std::move(normalized);
  return true;
}

/// Build a fully-populated dense `sima_ev_tensor_desc` (shape + axis semantics + strides).
inline bool build_dense_tensor_desc(const std::vector<int>& shape,
                                    const std::string& dtype_token,
                                    const std::string& layout_token,
                                    sima_ev_tensor_desc* out,
                                    std::string* error_detail,
                                    std::string_view missing_output_msg,
                                    std::string_view rank_invalid_msg,
                                    std::string_view dim_invalid_msg,
                                    std::string_view dtype_invalid_msg,
                                    std::string_view stride_output_missing_msg) {
  if (!out) {
    if (error_detail) {
      *error_detail = std::string(missing_output_msg);
    }
    return false;
  }
  std::memset(out, 0, sizeof(*out));
  if (!fill_shape_desc(shape, layout_token, &out->shape, error_detail, missing_output_msg,
                       rank_invalid_msg, dim_invalid_msg)) {
    return false;
  }
  if (!dtype_token_to_ev(dtype_token, &out->dtype)) {
    if (error_detail) {
      *error_detail = std::string(dtype_invalid_msg);
    }
    return false;
  }
  out->layout_kind = SIMA_EV_LAYOUT_STRIDED;
  return fill_dense_strides(out->shape, layout_token, out->dtype, &out->layout.strided,
                            error_detail, stride_output_missing_msg, dtype_invalid_msg);
}

/// Build a generic (no-axis-semantics) dense `sima_ev_tensor_desc` with last-dim-fastest strides.
template <typename ShapeT>
inline bool build_generic_dense_tensor_desc(const std::vector<ShapeT>& shape,
                                            const std::string& dtype_token,
                                            sima_ev_tensor_desc* out,
                                            std::string* error_detail,
                                            std::string_view missing_output_msg,
                                            std::string_view rank_invalid_msg,
                                            std::string_view dim_invalid_msg,
                                            std::string_view dtype_invalid_msg,
                                            std::string_view stride_output_missing_msg) {
  if (!out) {
    if (error_detail) {
      *error_detail = std::string(missing_output_msg);
    }
    return false;
  }
  std::memset(out, 0, sizeof(*out));
  if (!fill_shape_desc_without_axis_semantics(shape, &out->shape, error_detail, missing_output_msg,
                                              rank_invalid_msg, dim_invalid_msg)) {
    return false;
  }
  if (!dtype_token_to_ev(dtype_token, &out->dtype)) {
    if (error_detail) {
      *error_detail = std::string(dtype_invalid_msg);
    }
    return false;
  }
  out->layout_kind = SIMA_EV_LAYOUT_STRIDED;
  return fill_dense_strides_contiguous(out->shape, out->dtype, &out->layout.strided, error_detail,
                                       stride_output_missing_msg, dtype_invalid_msg);
}

/// Forward declaration; defined below.
template <typename ShapeT, typename TileShapeT>
inline bool build_generic_tiled_tensor_desc(const std::vector<ShapeT>& shape,
                                            const std::vector<TileShapeT>& tile_shape,
                                            const std::string& dtype_token,
                                            std::uint32_t tile_align_bytes,
                                            sima_ev_tensor_desc* out,
                                            std::string* error_detail,
                                            std::string_view missing_output_msg,
                                            std::string_view rank_invalid_msg,
                                            std::string_view dim_invalid_msg,
                                            std::string_view dtype_invalid_msg,
                                            std::string_view tile_rank_mismatch_msg,
                                            std::string_view tile_dim_invalid_msg);

/**
 * @brief Build a tiled `sima_ev_tensor_desc` with axis-semantic stamping when possible.
 *
 * If the layout token normalizes to a known canonical layout, stamps the corresponding axis
 * semantics and sets `SIMA_EV_TILED_FLAG_COMPACT_CHANNELS` when a C axis is present. Otherwise
 * delegates to `build_generic_tiled_tensor_desc`.
 */
inline bool build_tiled_tensor_desc(const std::vector<int>& shape,
                                    const std::vector<int>& tile_shape,
                                    const std::string& dtype_token,
                                    const std::string& layout_token,
                                    std::uint32_t tile_align_bytes,
                                    sima_ev_tensor_desc* out,
                                    std::string* error_detail,
                                    std::string_view missing_output_msg,
                                    std::string_view rank_invalid_msg,
                                    std::string_view dim_invalid_msg,
                                    std::string_view dtype_invalid_msg,
                                    std::string_view tile_rank_mismatch_msg,
                                    std::string_view tile_dim_invalid_msg) {
  if (!out) {
    if (error_detail) {
      *error_detail = std::string(missing_output_msg);
    }
    return false;
  }
  std::memset(out, 0, sizeof(*out));
  if (shape.size() != tile_shape.size()) {
    if (error_detail) {
      *error_detail = std::string(tile_rank_mismatch_msg);
    }
    return false;
  }
  const std::string normalized_layout = normalize_layout_token(layout_token);
  if (!layout_token.empty() && normalized_layout.empty()) {
    if (error_detail) {
      *error_detail = std::string(rank_invalid_msg);
    }
    return false;
  }
  if (normalized_layout.empty()) {
    return build_generic_tiled_tensor_desc(shape, tile_shape, dtype_token, tile_align_bytes, out,
                                           error_detail, missing_output_msg, rank_invalid_msg,
                                           dim_invalid_msg, dtype_invalid_msg,
                                           tile_rank_mismatch_msg, tile_dim_invalid_msg);
  }
  if (!fill_shape_desc(shape, normalized_layout, &out->shape, error_detail, missing_output_msg,
                       rank_invalid_msg, dim_invalid_msg)) {
    return false;
  }
  if (!dtype_token_to_ev(dtype_token, &out->dtype)) {
    if (error_detail) {
      *error_detail = std::string(dtype_invalid_msg);
    }
    return false;
  }
  out->layout_kind = SIMA_EV_LAYOUT_TILED;
  for (std::size_t i = 0; i < tile_shape.size(); ++i) {
    if (tile_shape[i] <= 0 || tile_shape[i] > shape[i]) {
      if (error_detail) {
        *error_detail = std::string(tile_dim_invalid_msg);
      }
      return false;
    }
    out->layout.tiled.tile_sizes[i] = static_cast<std::int64_t>(tile_shape[i]);
  }
  out->layout.tiled.tile_align_bytes = tile_align_bytes;
  out->layout.tiled.flags =
      find_shape_axis(out->shape, SIMA_EV_AXIS_C) >= 0 ? SIMA_EV_TILED_FLAG_COMPACT_CHANNELS
                                                       : SIMA_EV_TILED_FLAG_NONE;
  return true;
}

/**
 * @brief Build a tiled `sima_ev_tensor_desc` without axis-semantic stamping.
 *
 * Used for generic non-image tiled tensors. Rejects rank-mismatched, non-positive, or
 * out-of-bound tile dims.
 */
template <typename ShapeT, typename TileShapeT>
inline bool build_generic_tiled_tensor_desc(const std::vector<ShapeT>& shape,
                                            const std::vector<TileShapeT>& tile_shape,
                                            const std::string& dtype_token,
                                            std::uint32_t tile_align_bytes,
                                            sima_ev_tensor_desc* out,
                                            std::string* error_detail,
                                            std::string_view missing_output_msg,
                                            std::string_view rank_invalid_msg,
                                            std::string_view dim_invalid_msg,
                                            std::string_view dtype_invalid_msg,
                                            std::string_view tile_rank_mismatch_msg,
                                            std::string_view tile_dim_invalid_msg) {
  if (!out) {
    if (error_detail) {
      *error_detail = std::string(missing_output_msg);
    }
    return false;
  }
  std::memset(out, 0, sizeof(*out));
  if (shape.size() != tile_shape.size()) {
    if (error_detail) {
      *error_detail = std::string(tile_rank_mismatch_msg);
    }
    return false;
  }
  if (!fill_shape_desc_without_axis_semantics(shape, &out->shape, error_detail,
                                              missing_output_msg, rank_invalid_msg,
                                              dim_invalid_msg)) {
    return false;
  }
  if (!dtype_token_to_ev(dtype_token, &out->dtype)) {
    if (error_detail) {
      *error_detail = std::string(dtype_invalid_msg);
    }
    return false;
  }
  out->layout_kind = SIMA_EV_LAYOUT_TILED;
  for (std::size_t i = 0; i < tile_shape.size(); ++i) {
    if (tile_shape[i] <= 0 || tile_shape[i] > shape[i]) {
      if (error_detail) {
        *error_detail = std::string(tile_dim_invalid_msg);
      }
      return false;
    }
    out->layout.tiled.tile_sizes[i] = static_cast<std::int64_t>(tile_shape[i]);
  }
  out->layout.tiled.tile_align_bytes = tile_align_bytes;
  out->layout.tiled.flags = SIMA_EV_TILED_FLAG_NONE;
  return true;
}

} // namespace simaai::neat::pipeline_internal::sima::tensorsemantics
