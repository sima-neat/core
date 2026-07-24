#include "pipeline/internal/sima/PluginContractSubsets.h"

#include "pipeline/internal/sima/BoxDecodeStaticContractExtractor.h"
#include "pipeline/internal/sima/ProcessCvuFamily.h"
#include "pipeline/internal/sima/StaticSpecBuilders.h"
#include "pipeline/internal/sima/stagesemantics/BoxDecodeStageSemantics.h"
#include "pipeline/internal/EnvUtil.h"
#include "pipeline/internal/sima/TensorSemanticsUtil.h"

#include <algorithm>
#include <array>
#include <cinttypes>
#include <cctype>
#include <cstdio>
#include <cstdint>
#include <cstring>
#include <limits>
#include <numeric>
#include <queue>
#include <sstream>
#include <stdexcept>
#include <type_traits>
#include <unordered_map>
#include <unordered_set>

namespace simaai::neat::pipeline_internal::sima::plugin_contracts {
namespace {

using stagesemantics::CompiledProcessCvuRuntimeConfig;

std::string lower_copy(std::string value) {
  std::transform(value.begin(), value.end(), value.begin(),
                 [](unsigned char c) { return static_cast<char>(std::tolower(c)); });
  return value;
}

std::string upper_copy(std::string value) {
  std::transform(value.begin(), value.end(), value.begin(),
                 [](unsigned char c) { return static_cast<char>(std::toupper(c)); });
  return value;
}

std::string normalize_layout_token_local(std::string value) {
  return tensorsemantics::normalize_layout_token(std::move(value));
}

bool detess_layout_debug_enabled_local() {
  return pipeline_internal::env_bool("SIMA_DETESS_LAYOUT_DEBUG", false);
}

bool tess_segment_debug_enabled_local() {
  return pipeline_internal::env_bool("SIMA_TESS_SEGMENT_DEBUG", false);
}

// Phase-1 (Option A++): contract-side geometric invariants for per-frame
// descriptors. Off by default. When the SIMA_NEAT_CONTRACT_INVARIANTS
// environment variable is set non-empty / non-zero, every contract subset
// extractor that constructs a sima_ev_tensor_desc routes through
// check_per_frame_geometric_invariants(...) before populating the descriptor.
//
// These invariants do NOT enforce a specific rank (3D/5D/etc.). They enforce
// the relationships that must hold for any per-frame rank R in [1, 8]:
//   * input.rank == output.rank        (per-frame symmetry)
//   * slice.rank == input.rank         (only when slice_shape is non-empty)
//   * 1 <= rank <= kMaxPerFrameRank
//   * batch_size >= 1
//
// The helper logs a single structured warning per violation; it never throws
// and never alters control flow. This way Phase 1 is behavior-neutral but
// gives us a CI-grade signal that future regressions show up immediately.
constexpr int kMaxPerFrameRank = 8;

bool contract_invariants_enabled_local() {
  static const int cached = []() {
    const char* raw = std::getenv("SIMA_NEAT_CONTRACT_INVARIANTS");
    return (raw && *raw && std::strcmp(raw, "0") != 0) ? 1 : 0;
  }();
  return cached == 1;
}

// Render a shape vector as `[a,b,c]` for diagnostic logs. Tolerates empty.
std::string shape_dbg_local(const std::vector<std::int64_t>& shape) {
  std::ostringstream out;
  out << "[";
  for (std::size_t i = 0; i < shape.size(); ++i) {
    if (i) {
      out << ",";
    }
    out << shape[i];
  }
  out << "]";
  return out.str();
}

// `slice_shape` may be empty for dense-only families (cast/quant/dequant);
// `expected_batch_size` is informational and may be 0 if the extractor doesn't
// track it yet (Phase 1 never gates on it). `family` and `stage_name` are
// rendered into the warning so the offending extractor is easy to find.
void check_per_frame_geometric_invariants(const char* family, const char* stage_name,
                                          const std::vector<std::int64_t>& input_shape,
                                          const std::vector<std::int64_t>& output_shape,
                                          const std::vector<std::int64_t>& slice_shape,
                                          int expected_batch_size) {
  if (!contract_invariants_enabled_local()) {
    return;
  }
  const auto warn = [&](const char* code, const char* detail) {
    std::fprintf(stderr,
                 "[contract-invariant][warn] family=%s stage=%s code=%s detail=%s "
                 "input_shape=%s output_shape=%s slice_shape=%s expected_batch_size=%d\n",
                 family ? family : "<unknown>", stage_name ? stage_name : "<unknown>", code,
                 detail ? detail : "", shape_dbg_local(input_shape).c_str(),
                 shape_dbg_local(output_shape).c_str(), shape_dbg_local(slice_shape).c_str(),
                 expected_batch_size);
  };
  if (input_shape.empty() || output_shape.empty()) {
    warn("EMPTY_SHAPE", "input or output shape is empty");
    return;
  }
  if (input_shape.size() != output_shape.size()) {
    warn("RANK_ASYMMETRY_IO", "input.rank != output.rank");
  }
  if (!slice_shape.empty() && slice_shape.size() != input_shape.size()) {
    warn("RANK_MISMATCH_SLICE", "slice.rank != input.rank");
  }
  const int rank = static_cast<int>(input_shape.size());
  if (rank < 1 || rank > kMaxPerFrameRank) {
    warn("RANK_OUT_OF_RANGE", "rank not in [1, kMaxPerFrameRank]");
  }
  if (expected_batch_size < 0) {
    warn("BATCH_NEGATIVE", "batch_size cannot be negative");
  }
}

std::string join_i64_debug_local(const std::vector<std::int64_t>& values) {
  std::ostringstream out;
  out << "[";
  for (std::size_t i = 0; i < values.size(); ++i) {
    if (i > 0U) {
      out << "x";
    }
    out << values[i];
  }
  out << "]";
  return out.str();
}

std::string normalize_dtype_token(std::string value) {
  value = upper_copy(std::move(value));
  if (value == "FLOAT" || value == "FLOAT32" || value == "FP32" || value == "EVXX_FLOAT32") {
    return "FP32";
  }
  if (value == "BFLOAT16" || value == "BF16" || value == "EVXX_BFLOAT16") {
    return "BF16";
  }
  if (value == "INT8" || value == "EVXX_INT8") {
    return "INT8";
  }
  if (value == "INT16" || value == "EVXX_INT16") {
    return "INT16";
  }
  if (value == "INT32" || value == "EVXX_INT32") {
    return "INT32";
  }
  return value;
}

// Phase 2 (Option A++): per-frame-rank-aware batch normalization.
//
// `per_frame_rank` is the rank the kernel descriptor must consume (3 for the
// HWC family that dominates today, but generalized up to kMaxPerFrameRank).
// The function strips leading dims until the residual rank == per_frame_rank.
// `per_frame_rank <= 0` is treated as "rank not yet derived" and returns the
// shape unchanged so legacy callers without rank context behave as before.
//
// The strip is repeated (not single-step) so a 6D shape with rank=3 collapses
// in one call: e.g. [B, T, D, H, W, C] with per_frame_rank=3 → [H, W, C], with
// the leading B*T*D folded into batch (counted by inferred_batch_size_from_shape).
//
// Default value of 3 preserves the existing rank-3 contract: callers that
// haven't been migrated yet to derive the rank explicitly still get the
// historical behavior.
std::vector<std::int64_t> semantic_shape_without_batch(std::vector<std::int64_t> shape,
                                                       int per_frame_rank = 3) {
  if (per_frame_rank <= 0 || per_frame_rank > kMaxPerFrameRank) {
    return shape;
  }
  while (static_cast<int>(shape.size()) > per_frame_rank) {
    shape.erase(shape.begin());
  }
  return shape;
}

// Returns the per-frame rank for a stage whose tile geometry / peer-tensor
// shape is available. When the stage has tile semantics, slice_shape_hint is
// authoritative — its rank IS the per-frame rank by definition. Otherwise the
// peer (e.g. output) per-frame shape (already stripped) is used. Returns 0 if
// neither source is usable; callers should treat that as "fallback to default
// per_frame_rank=3" to preserve legacy behavior.
//
// `peer_per_frame_shape` MUST already be batch-stripped if used. If callers
// pass a still-batched shape they'll get its full rank back, which produces a
// correct invariant in current code (in/out ranks match) and also lets bs=1
// callers continue working since their loader-stripped shape happens to match.
int derive_per_frame_rank(const std::vector<std::int64_t>& slice_shape_hint,
                          const std::vector<std::int64_t>& peer_per_frame_shape) {
  if (!slice_shape_hint.empty()) {
    return static_cast<int>(slice_shape_hint.size());
  }
  if (!peer_per_frame_shape.empty()) {
    return static_cast<int>(peer_per_frame_shape.size());
  }
  return 0;
}

std::uint64_t dtype_size_bytes(const std::string& dtype) {
  const std::string normalized = normalize_dtype_token(dtype);
  if (normalized == "FP32" || normalized == "INT32" || normalized == "UINT32") {
    return 4U;
  }
  if (normalized == "BF16" || normalized == "FP16" || normalized == "INT16" ||
      normalized == "UINT16") {
    return 2U;
  }
  return 1U;
}

std::uint32_t resolve_tile_align_bytes_local(int byte_align) {
  if (byte_align <= 0) {
    return 0U;
  }
  return byte_align == 1 ? 16U : static_cast<std::uint32_t>(byte_align);
}

bool build_dense_desc_local(const std::vector<int>& shape, const std::string& dtype,
                            const std::string& layout, sima_ev_tensor_desc* out) {
  std::string error_detail;
  const std::string normalized_layout = normalize_layout_token_local(layout);
  if (!layout.empty() && normalized_layout.empty()) {
    return false;
  }
  if (normalized_layout.empty()) {
    return tensorsemantics::build_generic_dense_tensor_desc(
        shape, dtype, out, &error_detail, "subset_dense_tensor_desc_output_missing",
        "subset_shape_rank_invalid", "subset_shape_dim_invalid", "subset_dtype_invalid",
        "subset_dense_stride_output_missing");
  }
  return tensorsemantics::build_dense_tensor_desc(
      shape, dtype, normalized_layout, out, &error_detail,
      "subset_dense_tensor_desc_output_missing", "subset_shape_rank_invalid",
      "subset_shape_dim_invalid", "subset_dtype_invalid", "subset_dense_stride_output_missing");
}

bool build_tiled_desc_local(const std::vector<int>& shape, const std::vector<int>& tile_shape,
                            const std::string& dtype, const std::string& layout,
                            std::uint32_t tile_align_bytes, sima_ev_tensor_desc* out,
                            std::string* error_detail) {
  std::vector<int> normalized_tile_shape;
  if (!tensorsemantics::normalize_tile_shape(shape, tile_shape, &normalized_tile_shape,
                                             error_detail,
                                             "subset_tiled_tensor_desc_tile_shape_missing",
                                             "subset_tiled_tensor_desc_tile_rank_prefix_invalid",
                                             "subset_tiled_tensor_desc_tile_dim_invalid")) {
    return false;
  }
  return tensorsemantics::build_tiled_tensor_desc(
      shape, normalized_tile_shape, dtype, layout, tile_align_bytes, out, error_detail,
      "subset_tiled_tensor_desc_output_missing", "subset_shape_rank_invalid",
      "subset_shape_dim_invalid", "subset_dtype_invalid",
      "subset_tiled_tensor_desc_shape_rank_mismatch", "subset_tiled_tensor_desc_tile_dim_invalid");
}

bool build_tiled_desc_local(const std::vector<int>& shape, const std::vector<int>& tile_shape,
                            const std::string& dtype, const std::string& layout,
                            std::uint32_t tile_align_bytes, sima_ev_tensor_desc* out) {
  std::string error_detail;
  return build_tiled_desc_local(shape, tile_shape, dtype, layout, tile_align_bytes, out,
                                &error_detail);
}

bool build_generic_tiled_desc_local(const std::vector<int>& shape,
                                    const std::vector<int>& tile_shape, const std::string& dtype,
                                    std::uint32_t tile_align_bytes, sima_ev_tensor_desc* out) {
  std::string error_detail;
  return tensorsemantics::build_generic_tiled_tensor_desc(
      shape, tile_shape, dtype, tile_align_bytes, out, &error_detail,
      "subset_tiled_tensor_desc_output_missing", "subset_shape_rank_invalid",
      "subset_shape_dim_invalid", "subset_dtype_invalid",
      "subset_tiled_tensor_desc_shape_rank_mismatch", "subset_tiled_tensor_desc_tile_dim_invalid");
}

void apply_tensor_axes_local(sima_ev_tensor_desc* desc) {
  if (!desc) {
    return;
  }
  const std::uint32_t rank =
      std::min<std::uint32_t>(desc->shape.rank, static_cast<std::uint32_t>(SIMA_EV_MAX_RANK));
  std::vector<std::int64_t> shape;
  shape.reserve(rank);
  for (std::uint32_t i = 0; i < rank; ++i) {
    shape.push_back(desc->shape.sizes[i]);
  }
  tensorsemantics::fill_axis_semantics_from_shape_layout(shape, "", desc->shape.axis_semantics);
  if (desc->layout_kind == SIMA_EV_LAYOUT_TILED &&
      tensorsemantics::find_shape_axis(desc->shape, SIMA_EV_AXIS_C) >= 0) {
    desc->layout.tiled.flags |= SIMA_EV_TILED_FLAG_COMPACT_CHANNELS;
  }
}

void apply_tiled_channel_storage_policy_local(sima_ev_tensor_desc* desc, bool c16_packed) {
  if (!desc || desc->layout_kind != SIMA_EV_LAYOUT_TILED ||
      tensorsemantics::find_shape_axis(desc->shape, SIMA_EV_AXIS_C) < 0) {
    return;
  }
  if (c16_packed) {
    desc->layout.tiled.flags &= ~static_cast<std::uint32_t>(SIMA_EV_TILED_FLAG_COMPACT_CHANNELS);
  } else {
    desc->layout.tiled.flags |= SIMA_EV_TILED_FLAG_COMPACT_CHANNELS;
  }
}

bool build_tensor_tiled_desc_local(const std::vector<int>& shape,
                                   const std::vector<int>& tile_shape, const std::string& dtype,
                                   std::uint32_t tile_align_bytes, bool c16_packed,
                                   sima_ev_tensor_desc* out, std::string* error_detail) {
  if (!build_tiled_desc_local(shape, tile_shape, dtype, "", tile_align_bytes, out, error_detail)) {
    return false;
  }
  apply_tensor_axes_local(out);
  apply_tiled_channel_storage_policy_local(out, c16_packed);
  return true;
}

bool build_tensor_tiled_desc_local(const std::vector<int>& shape,
                                   const std::vector<int>& tile_shape, const std::string& dtype,
                                   std::uint32_t tile_align_bytes, bool c16_packed,
                                   sima_ev_tensor_desc* out) {
  std::string error_detail;
  return build_tensor_tiled_desc_local(shape, tile_shape, dtype, tile_align_bytes, c16_packed, out,
                                       &error_detail);
}

bool build_tensor_tiled_desc_local(const std::vector<int>& shape,
                                   const std::vector<int>& tile_shape, const std::string& dtype,
                                   std::uint32_t tile_align_bytes, sima_ev_tensor_desc* out) {
  return build_tensor_tiled_desc_local(shape, tile_shape, dtype, tile_align_bytes, false, out);
}

bool build_tensor_dense_desc_local(const std::vector<int>& shape, const std::string& dtype,
                                   sima_ev_tensor_desc* out) {
  if (!build_dense_desc_local(shape, dtype, "", out)) {
    return false;
  }
  apply_tensor_axes_local(out);
  return true;
}

struct ShapeDims {
  int width = 0;
  int height = 0;
  int depth = 0;
  int channels = 0;
  std::string layout;
};

bool canonical_slice_dhwc_from_shape(const std::vector<std::int64_t>& shape, int* out_d, int* out_h,
                                     int* out_w, int* out_c);

ShapeDims dims_from_shape(std::vector<std::int64_t> shape) {
  ShapeDims dims;
  shape = semantic_shape_without_batch(std::move(shape));
  if (shape.empty()) {
    return dims;
  }
  if (shape.size() == 1U) {
    dims.width = static_cast<int>(shape[0]);
    dims.height = 1;
    dims.depth = 1;
    dims.channels = 1;
    return dims;
  }
  if (shape.size() == 2U) {
    dims.height = static_cast<int>(shape[0]);
    dims.width = static_cast<int>(shape[1]);
    dims.depth = 1;
    dims.channels = 1;
    return dims;
  }
  std::uint64_t leading = 1U;
  for (std::size_t i = 0; i + 3U < shape.size(); ++i) {
    leading *= static_cast<std::uint64_t>(std::max<std::int64_t>(shape[i], 1));
  }
  dims.depth = static_cast<int>(leading);
  dims.height = static_cast<int>(shape[shape.size() - 3U]);
  dims.width = static_cast<int>(shape[shape.size() - 2U]);
  dims.channels = static_cast<int>(shape.back());
  return dims;
}

// Phase 2 (Option A++): infer the implicit batch_size from a not-yet-stripped
// shape, given the kernel's per-frame rank. Multiplies all leading dims
// (everything beyond per_frame_rank) into the batch count.
//
// per_frame_rank=3 (default) preserves legacy semantics: shapes of rank ≤ 3
// have batch=1 and ranks 4+ multiply leading dims into batch. For per_frame_rank=5
// (hypothetical 5D-per-frame stage) a 6D shape returns shape[0] as batch.
//
// Returns 0 on overflow or when the shape is empty.
int inferred_batch_size_from_shape(const std::vector<std::int64_t>& shape, int per_frame_rank = 3) {
  if (shape.empty()) {
    return 0;
  }
  if (per_frame_rank <= 0 || per_frame_rank > kMaxPerFrameRank) {
    return 1;
  }
  if (static_cast<int>(shape.size()) <= per_frame_rank) {
    return 1;
  }
  std::int64_t batch = 1;
  for (std::size_t i = 0; static_cast<int>(i) + per_frame_rank < static_cast<int>(shape.size());
       ++i) {
    batch *= std::max<std::int64_t>(shape[i], 1);
    if (batch > std::numeric_limits<int>::max()) {
      return 0;
    }
  }
  if (batch <= 0 || batch > std::numeric_limits<int>::max()) {
    return 0;
  }
  return static_cast<int>(batch);
}

void require_matching_batch_size(const int expected, const int actual,
                                 const std::string_view family) {
  if (expected > 0 && actual > 0 && expected != actual) {
    throw std::invalid_argument("plugin contract subset '" + std::string(family) +
                                "' requires a consistent batch_size across shapes");
  }
}

std::uint64_t round_up_to_multiple_local(const std::uint64_t value, const std::uint64_t multiple) {
  if (multiple == 0U) {
    return value;
  }
  const std::uint64_t remainder = value % multiple;
  if (remainder == 0U) {
    return value;
  }
  return value + (multiple - remainder);
}

struct DetessBoundaryShapeView {
  std::vector<std::int64_t> logical_input_shape;
  std::vector<std::int64_t> transport_shape;
  std::uint64_t transport_size_bytes = 0U;
};

DetessBoundaryShapeView
detess_boundary_shape_view_from_stage(const MpkPluginIoContract& stage,
                                      const MpkTensorContract* transport_tensor,
                                      std::uint64_t transport_size_bytes) {
  if (stage.frame_shape.empty()) {
    throw std::runtime_error("detess boundary contract requires frame_shape for '" + stage.name +
                             "'");
  }
  if (!transport_tensor) {
    throw std::runtime_error("detess boundary contract requires a packed transport tensor for '" +
                             stage.name + "'");
  }
  if (transport_tensor->mpk_shape.empty()) {
    throw std::runtime_error("detess boundary contract requires packed transport shape for '" +
                             stage.name + "'");
  }
  bool have_positive_logical_dims = false;
  for (const auto dim : stage.frame_shape) {
    if (dim > 0) {
      have_positive_logical_dims = true;
      continue;
    }
    throw std::runtime_error("detess boundary contract requires canonical frame geometry for '" +
                             stage.name + "'");
  }
  if (!have_positive_logical_dims) {
    throw std::runtime_error("detess boundary contract requires canonical frame geometry for '" +
                             stage.name + "'");
  }

  if (transport_size_bytes == 0U) {
    throw std::runtime_error("detess boundary contract transport bytes mismatch for '" +
                             stage.name + "'");
  }

  DetessBoundaryShapeView view;
  view.logical_input_shape = stage.frame_shape;
  view.transport_shape = transport_tensor->mpk_shape;
  view.transport_size_bytes = transport_size_bytes;
  if ((stage.has_align_c16 && stage.align_c16) || (stage.has_cblock && stage.cblock)) {
    const std::uint64_t packed_channels =
        static_cast<std::uint64_t>(view.transport_shape.empty() ? 0 : view.transport_shape.back());
    const std::uint64_t elem_bytes = dtype_size_bytes(
        !transport_tensor->dtype.empty() ? transport_tensor->dtype : stage.frame_type);
    const bool byte_granule_aligned =
        packed_channels > 0U && elem_bytes > 0U &&
        packed_channels <= (std::numeric_limits<std::uint64_t>::max() / elem_bytes) &&
        ((packed_channels * elem_bytes) % 16U) == 0U;
    if (!byte_granule_aligned) {
      throw std::runtime_error(
          "detess boundary contract expected 16-byte aligned packed channel storage for '" +
          stage.name + "'");
    }
  }
  return view;
}

bool canonical_slice_dhwc_from_shape(const std::vector<std::int64_t>& shape, int* out_d, int* out_h,
                                     int* out_w, int* out_c) {
  if (!out_d || !out_h || !out_w || !out_c || shape.empty()) {
    return false;
  }
  std::vector<std::int64_t> normalized = shape;
  if (normalized.size() >= 4U && normalized.front() == 1) {
    normalized.erase(normalized.begin());
  }
  if (normalized.size() == 1U) {
    *out_d = 1;
    *out_h = 1;
    *out_w = static_cast<int>(normalized[0]);
    *out_c = 1;
    return true;
  }
  if (normalized.size() == 2U) {
    *out_d = 1;
    *out_h = static_cast<int>(normalized[0]);
    *out_w = static_cast<int>(normalized[1]);
    *out_c = 1;
    return true;
  }
  if (normalized.size() == 3U) {
    *out_d = 1;
    *out_h = static_cast<int>(normalized[0]);
    *out_w = static_cast<int>(normalized[1]);
    *out_c = static_cast<int>(normalized[2]);
    return true;
  }
  std::uint64_t depth = 1U;
  for (std::size_t i = 0; i + 3U < normalized.size(); ++i) {
    depth *= static_cast<std::uint64_t>(std::max<std::int64_t>(normalized[i], 1));
  }
  *out_d = static_cast<int>(depth);
  *out_h = static_cast<int>(normalized[normalized.size() - 3U]);
  *out_w = static_cast<int>(normalized[normalized.size() - 2U]);
  *out_c = static_cast<int>(normalized.back());
  return true;
}

std::vector<int> tensor_desc_tile_shape_from_slice_shape(const std::vector<int>& tensor_shape,
                                                         const std::vector<int>& slice_shape) {
  std::vector<int> out;
  std::string error_detail;
  if (!tensorsemantics::normalize_tile_shape(tensor_shape, slice_shape, &out, &error_detail,
                                             "subset_tiled_tensor_desc_tile_shape_missing",
                                             "subset_tiled_tensor_desc_tile_rank_prefix_invalid",
                                             "subset_tiled_tensor_desc_tile_dim_invalid")) {
    return slice_shape;
  }
  return out;
}

int explicit_stage_batch_size_or_one(const MpkPluginIoContract& stage) {
  return stage.batch_size > 0 ? stage.batch_size : 1;
}

int explicit_pair_batch_size_or_one(const MpkPluginIoContract& lhs, const MpkPluginIoContract& rhs,
                                    std::string_view family) {
  const bool lhs_has_batch = lhs.batch_size > 0;
  const bool rhs_has_batch = rhs.batch_size > 0;
  if (lhs_has_batch && rhs_has_batch && lhs.batch_size != rhs.batch_size) {
    throw std::invalid_argument("plugin contract subset '" + std::string(family) +
                                "' requires matching explicit batch_size across paired stages");
  }
  if (lhs_has_batch) {
    return lhs.batch_size;
  }
  if (rhs_has_batch) {
    return rhs.batch_size;
  }
  return 1;
}

ShapeDims dims_from_detess_shape(const std::vector<std::int64_t>& shape,
                                 const std::string& context) {
  const auto dims = dims_from_shape(shape);
  if (dims.width <= 0 || dims.height <= 0 || dims.depth <= 0 || dims.channels <= 0) {
    throw std::invalid_argument("detess runtime config requires canonical geometry for '" +
                                context + "'");
  }
  return dims;
}

ShapeDims dims_from_detess_transport_shape(const std::vector<std::int64_t>& shape,
                                           const std::string& context) {
  int depth = 0;
  int height = 0;
  int width = 0;
  int channels = 0;
  if (!canonical_slice_dhwc_from_shape(shape, &depth, &height, &width, &channels)) {
    throw std::invalid_argument(
        "detess runtime config requires canonical packed transport geometry for '" + context + "'");
  }
  ShapeDims dims;
  dims.width = width;
  dims.height = height;
  dims.depth = std::max(depth, 1);
  dims.channels = std::max(channels, 1);
  return dims;
}

void validate_detessdequant_head_contract_local(const MpkPluginIoContract& detess,
                                                const MpkPluginIoContract& dequant,
                                                const DetessDequantHeadContractSubset& head) {
  if (detess.input_tensors.empty() || detess.output_tensors.empty() ||
      dequant.output_tensors.empty()) {
    throw std::runtime_error("detessdequant validation requires detess/dequant tensors for '" +
                             detess.name + "'");
  }

  const auto logical_input_dims = dims_from_detess_shape(head.per_head_input_shape, detess.name);
  const auto frame_dims = dims_from_detess_shape(head.frame_shape, detess.name + " frame");

  int slice_d = 0;
  int slice_h = 0;
  int slice_w = 0;
  int slice_c = 0;
  // Align the slice rank to the frame rank before extracting channels. A per-tile slice may omit
  // leading (batch) dims that the frame carries -- e.g. a non-image boxes tensor with slice
  // [300,4] and frame [1,300,4]. canonical_slice_dhwc_from_shape reads a rank-2 shape as
  // (h,w),channels=1 but a rank-3 shape as (h,w,c), so without this the slice would report
  // channels=1 against the frame's channels=4 and spuriously fail the channel-consistency check.
  // Left-padding with 1s preserves image slices (e.g. [1,18,512] vs frame [1,35,35,512]).
  std::vector<std::int64_t> slice_for_geometry = head.slice_shape;
  while (slice_for_geometry.size() < head.frame_shape.size()) {
    slice_for_geometry.insert(slice_for_geometry.begin(), 1);
  }
  if (!canonical_slice_dhwc_from_shape(slice_for_geometry, &slice_d, &slice_h, &slice_w,
                                       &slice_c)) {
    throw std::runtime_error("detessdequant validation requires canonical slice geometry for '" +
                             detess.name + "'");
  }
  const int logical_slice_channels = std::max(slice_c, 1);
  if (std::getenv("NEAT_DUMP_POST_REGIONS") != nullptr) {
    const auto js = [](const std::vector<std::int64_t>& v) {
      std::string s;
      for (auto x : v)
        s += std::to_string(x) + ",";
      return s;
    };
    std::fprintf(
        stderr,
        "[ddq-geom] %s frame_shape={%s} slice_shape={%s} per_head_in={%s} slice_c=%d frame_ch=%d\n",
        detess.name.c_str(), js(head.frame_shape).c_str(), js(head.slice_shape).c_str(),
        js(head.per_head_input_shape).c_str(), slice_c, frame_dims.channels);
  }
  if (logical_slice_channels != frame_dims.channels) {
    throw std::runtime_error(
        "detessdequant validation found mismatched detess frame/slice geometry for '" +
        detess.name + "'");
  }
  if (logical_input_dims.width <= 0 || logical_input_dims.height <= 0 ||
      logical_input_dims.channels <= 0) {
    throw std::runtime_error("detessdequant validation requires logical detess geometry for '" +
                             detess.name + "'");
  }

  if (head.input_transport_shape.empty() || head.input_transport_size_bytes == 0U) {
    throw std::runtime_error("detessdequant validation requires packed transport geometry for '" +
                             detess.name + "'");
  }
  if (std::getenv("NEAT_DUMP_POST_REGIONS") != nullptr) {
    const auto& it0 = detess.input_tensors.front();
    std::string tsh, ish;
    for (auto v : head.input_transport_shape)
      tsh += std::to_string(v) + ",";
    for (auto v : it0.mpk_shape)
      ish += std::to_string(v) + ",";
    std::fprintf(
        stderr,
        "[detessdequant] %s: head.transport_bytes=%llu detess.in.size_bytes=%llu "
        "in_dtype=%s in_logical_dtype=%s transport_shape={%s} in_mpk_shape={%s} n_in=%zu\n",
        detess.name.c_str(), static_cast<unsigned long long>(head.input_transport_size_bytes),
        static_cast<unsigned long long>(it0.size_bytes), it0.dtype.c_str(),
        it0.logical_dtype.c_str(), tsh.c_str(), ish.c_str(), detess.input_tensors.size());
  }
  if (head.input_transport_size_bytes !=
      static_cast<std::uint64_t>(detess.input_tensors.front().size_bytes)) {
    throw std::runtime_error("detessdequant validation found packed input byte mismatch for '" +
                             detess.name + "'");
  }

  const std::string output_dtype = normalize_dtype_token(head.output_dtype);
  const std::uint64_t output_elem_bytes = dtype_size_bytes(output_dtype);
  if (output_elem_bytes == 0U) {
    throw std::runtime_error("detessdequant validation requires normalized output dtype for '" +
                             dequant.name + "'");
  }
  (void)output_elem_bytes;
}

std::uint64_t dense_size_bytes(const std::vector<std::int64_t>& shape, const std::string& dtype) {
  if (shape.empty()) {
    return 0U;
  }
  std::uint64_t total = dtype_size_bytes(dtype);
  for (const auto dim : shape) {
    if (dim <= 0) {
      return 0U;
    }
    total *= static_cast<std::uint64_t>(dim);
  }
  return total;
}

std::uint64_t tessellated_output_size_bytes_local(const ShapeDims& dims, int tile_w, int tile_h,
                                                  int tile_d, int tile_c, const std::string& dtype,
                                                  const bool byte_align) {
  if (dims.width <= 0 || dims.height <= 0 || dims.depth <= 0 || dims.channels <= 0 || tile_w <= 0 ||
      tile_h <= 0 || tile_d <= 0 || tile_c <= 0 || dtype.empty()) {
    return 0U;
  }
  const std::uint64_t elem_bytes = dtype_size_bytes(dtype);
  if (elem_bytes == 0U) {
    return 0U;
  }

  const auto ceil_div = [](const int value, const int step) -> int {
    return (value + step - 1) / step;
  };
  const int num_w_tiles = ceil_div(dims.width, tile_w);
  const int num_h_tiles = ceil_div(dims.height, tile_h);
  const int num_d_tiles = ceil_div(dims.depth, tile_d);
  const int num_c_tiles = ceil_div(dims.channels, tile_c);

  std::uint64_t total = 0U;
  for (int c = 0; c < num_c_tiles; ++c) {
    const int t_channels = (c + 1 < num_c_tiles) ? tile_c : (dims.channels - c * tile_c);
    for (int d = 0; d < num_d_tiles; ++d) {
      const int t_depth = (d + 1 < num_d_tiles) ? tile_d : (dims.depth - d * tile_d);
      for (int h = 0; h < num_h_tiles; ++h) {
        const int t_height = (h + 1 < num_h_tiles) ? tile_h : (dims.height - h * tile_h);
        for (int w = 0; w < num_w_tiles; ++w) {
          const int t_width = (w + 1 < num_w_tiles) ? tile_w : (dims.width - w * tile_w);
          std::uint64_t tile_bytes = static_cast<std::uint64_t>(t_channels) *
                                     static_cast<std::uint64_t>(t_depth) *
                                     static_cast<std::uint64_t>(t_height) *
                                     static_cast<std::uint64_t>(t_width) * elem_bytes;
          if (byte_align) {
            tile_bytes = ((tile_bytes + 15U) / 16U) * 16U;
          }
          total += tile_bytes;
        }
      }
    }
  }
  return total;
}

std::vector<std::int64_t> preferred_shape_from_tensor(const MpkTensorContract& tensor) {
  return !tensor.mpk_shape.empty() ? tensor.mpk_shape : tensor.logical_shape;
}

std::vector<std::int64_t> preferred_stage_input_tensor_shape(const MpkPluginIoContract& stage,
                                                             const MpkTensorContract& tensor) {
  if (!tensor.logical_shape.empty()) {
    return tensor.logical_shape;
  }
  if (!tensor.mpk_shape.empty()) {
    return tensor.mpk_shape;
  }
  return preferred_shape_from_tensor(tensor);
}

std::vector<std::int64_t>
preferred_geometry_shape_from_stage_tensor(const MpkPluginIoContract& stage,
                                           const MpkTensorContract* tensor,
                                           bool prefer_output_shape_fallback) {
  if (tensor) {
    if (!tensor->logical_shape.empty()) {
      return tensor->logical_shape;
    }
    if (tensor->shape_semantics == MpkShapeSemantics::Geometry && !tensor->mpk_shape.empty()) {
      return tensor->mpk_shape;
    }
  }

  if (prefer_output_shape_fallback && !stage.out_shape_raw.empty()) {
    return stage.out_shape_raw;
  }
  return tensor ? preferred_shape_from_tensor(*tensor) : std::vector<std::int64_t>{};
}

void require_single_batch_stage(const MpkPluginIoContract& stage, std::string_view family) {
  if (stage.batch_size > 0 && stage.batch_size != 1) {
    throw std::invalid_argument("plugin contract subset '" + std::string(family) +
                                "' requires batch_size=1 for stage '" + stage.name + "'");
  }
  if (stage.batch_sz_model > 0 && stage.batch_sz_model != 1) {
    throw std::invalid_argument("plugin contract subset '" + std::string(family) +
                                "' requires batch_sz_model=1 for stage '" + stage.name + "'");
  }
}

int normalize_round_off_token(const std::string& raw, const std::string& stage_name) {
  const std::string token = upper_copy(raw);
  if (token.empty()) {
    throw std::invalid_argument("plugin contract subset 'quantize' missing required field "
                                "'round_off' for stage '" +
                                stage_name + "'");
  }
  if (token == "RT_ZERO" || token == "TOZERO") {
    return 0;
  }
  if (token == "RT_EVEN" || token == "TONEAREST") {
    return 1;
  }
  if (token == "RT_POSITVE_INFINITY" || token == "TOPOSITIVEINFINITY") {
    return 2;
  }
  if (token == "RT_NEGATIVE_INFINITY" || token == "TONEGATIVEINFINITY") {
    return 3;
  }
  throw std::invalid_argument("plugin contract subset 'quantize' received unsupported round_off '" +
                              raw + "' for stage '" + stage_name + "'");
}

std::string canonical_stage_family(const MpkPluginIoContract& stage) {
  const std::string raw_kernel = lower_copy(stage.kernel + " " + stage.name);
  if (raw_kernel.find("detessdequant") != std::string::npos ||
      raw_kernel.find("detess_dequant") != std::string::npos) {
    return "detessdequant";
  }
  if (raw_kernel.find("quanttess") != std::string::npos ||
      raw_kernel.find("quant_tess") != std::string::npos) {
    return "quanttess";
  }
  if (raw_kernel.find("detess") != std::string::npos) {
    return "detess";
  }
  if (raw_kernel.find("dequant") != std::string::npos) {
    return "dequant";
  }
  if (raw_kernel.find("tess") != std::string::npos) {
    return "tess";
  }
  if (raw_kernel.find("quant") != std::string::npos) {
    return "quant";
  }
  if (raw_kernel.find("mla") != std::string::npos) {
    return "mla";
  }
  if (raw_kernel.find("cast_") != std::string::npos ||
      raw_kernel.find(" cast") != std::string::npos ||
      raw_kernel.find("cast_transform") != std::string::npos) {
    return "cast";
  }
  return canonical_processcvu_family_from_kernel(stage.kernel);
}

const MpkPluginIoContract* find_first_stage_for_family(const MpkContract& contract,
                                                       std::string_view family) {
  const auto ordered = plugins_in_execution_order(contract);
  for (const auto index : ordered) {
    if (index >= contract.plugins.size()) {
      continue;
    }
    const auto& stage = contract.plugins[index];
    if (canonical_stage_family(stage) == family) {
      return &stage;
    }
  }
  return nullptr;
}

std::vector<const MpkPluginIoContract*> collect_stages_for_family(const MpkContract& contract,
                                                                  std::string_view family) {
  std::vector<const MpkPluginIoContract*> out;
  const auto ordered = plugins_in_execution_order(contract);
  for (const auto index : ordered) {
    if (index >= contract.plugins.size()) {
      continue;
    }
    const auto& stage = contract.plugins[index];
    if (canonical_stage_family(stage) == family) {
      out.push_back(&stage);
    }
  }
  return out;
}

std::uint64_t output_key_local(std::size_t plugin_index, int output_index) {
  return (static_cast<std::uint64_t>(plugin_index) << 32U) |
         static_cast<std::uint64_t>(static_cast<std::uint32_t>(output_index));
}

std::optional<std::size_t> plugin_index_from_stage_ptr_local(const MpkContract& contract,
                                                             const MpkPluginIoContract* stage) {
  if (!stage) {
    return std::nullopt;
  }
  for (std::size_t i = 0; i < contract.plugins.size(); ++i) {
    if (&contract.plugins[i] == stage) {
      return i;
    }
  }
  return std::nullopt;
}

std::unordered_map<std::uint64_t, std::vector<const MpkContractEdge*>>
build_outgoing_edges_local(const MpkContract& contract) {
  std::unordered_map<std::uint64_t, std::vector<const MpkContractEdge*>> outgoing;
  outgoing.reserve(contract.edges.size());
  for (const auto& edge : contract.edges) {
    outgoing[output_key_local(edge.src_plugin_index, edge.src_output_index)].push_back(&edge);
  }
  return outgoing;
}

const MpkPluginIoContract* resolve_dequant_stage_for_detess_local(
    const MpkContract& contract, const MpkPluginIoContract& detess,
    const std::unordered_map<std::uint64_t, std::vector<const MpkContractEdge*>>& outgoing_edges) {
  const auto detess_index = plugin_index_from_stage_ptr_local(contract, &detess);
  if (!detess_index.has_value()) {
    return nullptr;
  }

  std::queue<std::pair<std::size_t, int>> pending;
  std::unordered_set<std::uint64_t> visited_outputs;
  std::unordered_set<std::size_t> found_dequant_indices;
  for (std::size_t oi = 0; oi < detess.output_tensors.size(); ++oi) {
    pending.emplace(*detess_index, static_cast<int>(oi));
  }

  while (!pending.empty()) {
    const auto [plugin_index, output_index] = pending.front();
    pending.pop();
    const auto visit_key = output_key_local(plugin_index, output_index);
    if (!visited_outputs.insert(visit_key).second) {
      continue;
    }

    const auto outgoing_it = outgoing_edges.find(visit_key);
    if (outgoing_it == outgoing_edges.end()) {
      continue;
    }

    for (const auto* edge : outgoing_it->second) {
      if (!edge || edge->dst_plugin_index >= contract.plugins.size()) {
        continue;
      }
      const auto& consumer = contract.plugins[edge->dst_plugin_index];
      const std::string family = canonical_stage_family(consumer);
      if (family == "detessdequant" || family == "dequant") {
        found_dequant_indices.insert(edge->dst_plugin_index);
        continue;
      }
      for (std::size_t oi = 0; oi < consumer.output_tensors.size(); ++oi) {
        pending.emplace(edge->dst_plugin_index, static_cast<int>(oi));
      }
    }
  }

  if (found_dequant_indices.size() != 1U) {
    return nullptr;
  }
  const auto matched = *found_dequant_indices.begin();
  return matched < contract.plugins.size() ? &contract.plugins[matched] : nullptr;
}

constexpr auto kQuantizeRequired = std::array{
    PluginContractFieldKey::QuantParams,
    PluginContractFieldKey::InputShape,
    PluginContractFieldKey::InputDtype,
    PluginContractFieldKey::OutputDtype,
};
constexpr auto kQuantizeOptional = std::array{
    PluginContractFieldKey::RoundOff,
};
constexpr auto kCastRequired = std::array{
    PluginContractFieldKey::InputShape,
    PluginContractFieldKey::InputDtype,
    PluginContractFieldKey::OutputDtype,
};
constexpr auto kCastOptional = std::array<PluginContractFieldKey, 0>{};
constexpr auto kTessellateRequired = std::array{
    PluginContractFieldKey::InputShape,
    PluginContractFieldKey::FrameType,
    PluginContractFieldKey::SliceShape,
};
constexpr auto kTessellateOptional = std::array{
    PluginContractFieldKey::AlignC16,
    PluginContractFieldKey::Cblock,
};
constexpr auto kQuantTessRequired = std::array{
    PluginContractFieldKey::QuantParams, PluginContractFieldKey::InputShape,
    PluginContractFieldKey::InputDtype,  PluginContractFieldKey::OutputDtype,
    PluginContractFieldKey::SliceShape,  PluginContractFieldKey::FrameType,
};
constexpr auto kQuantTessOptional = std::array{
    PluginContractFieldKey::AlignC16,
    PluginContractFieldKey::Cblock,
};
constexpr auto kDetessellateRequired = std::array{
    PluginContractFieldKey::InputShape,
    PluginContractFieldKey::FrameShape,
    PluginContractFieldKey::FrameType,
    PluginContractFieldKey::SliceShape,
};
constexpr auto kDetessellateOptional = std::array{
    PluginContractFieldKey::AlignC16,
    PluginContractFieldKey::Cblock,
};
constexpr auto kDequantizeRequired = std::array{
    PluginContractFieldKey::QuantParams, PluginContractFieldKey::InputShape,
    PluginContractFieldKey::OutputShape, PluginContractFieldKey::InputDtype,
    PluginContractFieldKey::OutputDtype,
};
constexpr auto kDequantizeOptional = std::array<PluginContractFieldKey, 0>{};
constexpr auto kProcessMlaRequired = std::array{
    PluginContractFieldKey::ModelPath,
    PluginContractFieldKey::BatchSize,
    PluginContractFieldKey::BatchSzModel,
    PluginContractFieldKey::DispatcherOutputSizes,
};
constexpr auto kProcessMlaOptional = std::array{
    PluginContractFieldKey::DispatcherOutputNames,
};
constexpr auto kDetessDequantRequired = std::array{
    PluginContractFieldKey::PerHeadInputShape, PluginContractFieldKey::PerHeadQuantParams,
    PluginContractFieldKey::FrameShape,        PluginContractFieldKey::FrameType,
    PluginContractFieldKey::SliceShape,        PluginContractFieldKey::OutputDtype,
};
constexpr auto kDetessDequantOptional = std::array{
    PluginContractFieldKey::AlignC16,
    PluginContractFieldKey::Cblock,
};
constexpr auto kBoxDecodeRequired = std::array{
    PluginContractFieldKey::LogicalInputs,
    PluginContractFieldKey::InputBindings,
    PluginContractFieldKey::SliceGeometry,
};
constexpr auto kBoxDecodeOptional = std::array{
    PluginContractFieldKey::RouteFlags,
    PluginContractFieldKey::DecodeTypeOption,
    PluginContractFieldKey::ScoreActivation,
};

const PluginContractFamilyDeclaration kDeclarations[] = {
    {"quantize", kQuantizeRequired, kQuantizeOptional, false},
    {"cast", kCastRequired, kCastOptional, false},
    {"tessellate", kTessellateRequired, kTessellateOptional, false},
    {"quanttess", kQuantTessRequired, kQuantTessOptional, false},
    {"detessellate", kDetessellateRequired, kDetessellateOptional, false},
    {"dequantize", kDequantizeRequired, kDequantizeOptional, false},
    {"processmla", kProcessMlaRequired, kProcessMlaOptional, false},
    {"detessdequant", kDetessDequantRequired, kDetessDequantOptional, true},
    {"boxdecode", kBoxDecodeRequired, kBoxDecodeOptional, false},
};

void validate_present_fields(std::string_view family,
                             std::span<const PluginContractFieldKey> present,
                             const std::string& stage_name) {
  const auto& declaration = plugin_contract_family_declaration(family);
  const std::unordered_set<PluginContractFieldKey> present_set(present.begin(), present.end());
  for (const auto required : declaration.required_fields) {
    if (present_set.find(required) == present_set.end()) {
      throw std::invalid_argument("plugin contract subset '" + std::string(family) +
                                  "' missing required field '" +
                                  std::string(plugin_contract_field_key_name(required)) +
                                  "' for stage '" + stage_name + "'");
    }
  }
}

template <typename T>
void require_non_empty_value(const T& value, std::string_view family, PluginContractFieldKey field,
                             const std::string& stage_name) {
  if constexpr (requires { value.empty(); }) {
    if (!value.empty()) {
      return;
    }
  } else if constexpr (std::is_same_v<T, int>) {
    if (value > 0) {
      return;
    }
  } else if constexpr (std::is_same_v<T, bool>) {
    return;
  }
  throw std::invalid_argument(
      "plugin contract subset '" + std::string(family) + "' missing required field '" +
      std::string(plugin_contract_field_key_name(field)) + "' for stage '" + stage_name + "'");
}

std::string default_boxdecode_tensor_name(const BoxDecodeStaticContract& contract,
                                          std::size_t index) {
  if (index < contract.tensors.size() && !contract.tensors[index].logical_name.empty()) {
    return contract.tensors[index].logical_name;
  }
  if (index < contract.tensor_names.size() && !contract.tensor_names[index].empty()) {
    return contract.tensor_names[index];
  }
  return "input_tensor_" + std::to_string(index);
}

std::string default_boxdecode_backend_name(const BoxDecodeStaticContract& contract,
                                           std::size_t index) {
  if (index < contract.tensors.size() && !contract.tensors[index].backend_name.empty()) {
    return contract.tensors[index].backend_name;
  }
  return default_boxdecode_tensor_name(contract, index);
}

std::string default_boxdecode_tensorbuffer_segment_name(const BoxDecodeStaticContract& contract,
                                                        std::size_t index) {
  if (index < contract.physical_inputs.size() && !contract.physical_inputs[index].name.empty()) {
    return contract.physical_inputs[index].name;
  }
  if (index < contract.tensors.size() && !contract.tensors[index].source_segment_name.empty()) {
    return contract.tensors[index].source_segment_name;
  }
  return default_boxdecode_tensor_name(contract, index);
}

std::string default_boxdecode_runtime_segment_name(const BoxDecodeStaticContract& contract,
                                                   std::size_t index) {
  if (index < contract.physical_inputs.size() && !contract.physical_inputs[index].name.empty()) {
    return contract.physical_inputs[index].name;
  }
  if (index < contract.tensors.size() && !contract.tensors[index].source_segment_name.empty()) {
    return contract.tensors[index].source_segment_name;
  }
  return default_boxdecode_tensor_name(contract, index);
}

std::string default_boxdecode_layout(const BoxDecodeTensorStaticContract& tensor) {
  if (!tensor.layout.empty()) {
    const std::string normalized = upper_copy(tensor.layout);
    if (normalized == "CHW" || normalized == "NCHW") {
      return "CHW";
    }
    if (normalized == "HW" || normalized == "NHW") {
      return "HW";
    }
    if (normalized == "HWC" || normalized == "NHWC") {
      return "HWC";
    }
  }
  return {};
}

std::vector<std::int64_t> default_boxdecode_shape(const BoxDecodeTensorStaticContract& tensor) {
  // Return the raw input_shape converted to int64_t.
  std::vector<std::int64_t> shape;
  shape.reserve(tensor.input_shape.size());
  for (const int dim : tensor.input_shape) {
    shape.push_back(static_cast<std::int64_t>(dim));
  }
  return shape;
}

LogicalInputStaticSpec
logical_input_from_boxdecode_static_contract(const BoxDecodeStaticContract& contract,
                                             std::size_t index) {
  const auto& tensor = contract.tensors[index];
  auto logical = specbuilders::build_logical_input_static_spec(
      static_cast<int>(index), static_cast<int>(index), tensor.source_physical_index,
      default_boxdecode_shape(tensor),
      !tensor.data_type.empty() ? tensor.data_type : contract.input_dtype,
      default_boxdecode_layout(tensor), default_boxdecode_tensor_name(contract, index),
      default_boxdecode_backend_name(contract, index),
      default_boxdecode_tensorbuffer_segment_name(contract, index), tensor.source_byte_offset);
  logical.size_bytes = tensor.source_size_bytes;
  if (index < contract.dq_scale.size() && index < contract.dq_zp.size()) {
    QuantStaticSpec quant;
    quant.granularity = QuantGranularity::PerTensor;
    quant.axis = -1;
    quant.scales = {contract.dq_scale[index]};
    quant.zero_points = {contract.dq_zp[index]};
    logical.quant = std::move(quant);
  }
  return logical;
}

InputBindingStaticSpec
input_binding_from_boxdecode_static_contract(const BoxDecodeStaticContract& contract,
                                             const LogicalInputStaticSpec& logical,
                                             std::size_t index) {
  auto binding = specbuilders::build_input_binding_static_spec(
      0, logical.logical_index, logical.backend_name,
      default_boxdecode_runtime_segment_name(contract, index));
  if (index < contract.tensors.size()) {
    const auto& tensor = contract.tensors[index];
    binding.src_logical_output_index = tensor.source_logical_output_index;
    binding.src_output_slot = tensor.source_output_slot;
    binding.src_physical_output_index = tensor.source_physical_index;
    binding.src_physical_byte_offset = tensor.source_byte_offset;
  }
  if (index < contract.physical_inputs.size()) {
    if (contract.physical_inputs[index].physical_index >= 0) {
      binding.src_physical_output_index = contract.physical_inputs[index].physical_index;
    }
    binding.src_physical_byte_offset = contract.physical_inputs[index].byte_offset;
    binding.src_physical_size_bytes = contract.physical_inputs[index].size_bytes;
  }
  return binding;
}

} // namespace

// Phase 3a (Option A++): public wrappers around the anon-namespace helpers so
// out-of-translation-unit subset assemblers (notably
// ProcessCvuStageSemantics.cpp) can apply the same per-frame normalization
// without each rebuilding the convention. These delegate verbatim — see the
// in-anon-namespace definitions for documentation.
std::vector<std::int64_t> semantic_shape_without_batch_public(std::vector<std::int64_t> shape,
                                                              int per_frame_rank) {
  return semantic_shape_without_batch(std::move(shape), per_frame_rank);
}

int derive_per_frame_rank_public(const std::vector<std::int64_t>& slice_shape_hint,
                                 const std::vector<std::int64_t>& peer_per_frame_shape) {
  return derive_per_frame_rank(slice_shape_hint, peer_per_frame_shape);
}

int inferred_batch_size_from_shape_public(const std::vector<std::int64_t>& shape,
                                          int per_frame_rank) {
  return inferred_batch_size_from_shape(shape, per_frame_rank);
}

bool unit_axis_shape_alias_public(const std::vector<std::int64_t>& lhs,
                                  const std::vector<std::int64_t>& rhs) {
  auto strip_unit_axes = [](const std::vector<std::int64_t>& in) {
    std::vector<std::int64_t> out;
    out.reserve(in.size());
    for (const auto dim : in) {
      if (dim != 1) {
        out.push_back(dim);
      }
    }
    return out;
  };
  return strip_unit_axes(lhs) == strip_unit_axes(rhs);
}

std::vector<std::int64_t>
canonical_value_transform_shape_public(std::string_view family,
                                       const std::vector<std::int64_t>& input_shape,
                                       const std::vector<std::int64_t>& output_shape,
                                       const std::vector<std::int64_t>& preferred_semantic_shape) {
  auto is_value_transform_family = [](std::string_view raw_family) {
    const std::string family = lower_copy(std::string(raw_family));
    return family == "cast" || family == "quant" || family == "quantize" || family == "dequant" ||
           family == "dequantize" || family == "detess" || family == "detessellate" ||
           family == "detesscast" || family == "detessdequant";
  };

  if (!is_value_transform_family(family)) {
    return !output_shape.empty() ? output_shape : input_shape;
  }

  // Value-transform stages do not semantically flatten or squeeze tensors.
  // Some MPKs author a downstream/public terminal tensor with singleton axes
  // removed (for example ResNet [1,1,1,1000] -> [1,1000]).  That squeezed
  // view is valid for publication, but it must not become the EV runtime ABI
  // descriptor.  Prefer the authoritative full semantic shape when all
  // candidates differ only by size-1 axes; otherwise keep the explicit output
  // shape so genuine reshape-like routes are not silently rewritten.
  if (!preferred_semantic_shape.empty()) {
    const bool preferred_matches_input =
        input_shape.empty() || unit_axis_shape_alias_public(preferred_semantic_shape, input_shape);
    const bool preferred_matches_output =
        output_shape.empty() ||
        unit_axis_shape_alias_public(preferred_semantic_shape, output_shape);
    if (preferred_matches_input && preferred_matches_output) {
      return preferred_semantic_shape;
    }
  }

  if (!input_shape.empty() && !output_shape.empty() &&
      unit_axis_shape_alias_public(input_shape, output_shape)) {
    return input_shape.size() >= output_shape.size() ? input_shape : output_shape;
  }

  return !output_shape.empty() ? output_shape : input_shape;
}

std::string_view plugin_contract_field_key_name(const PluginContractFieldKey key) {
  switch (key) {
  case PluginContractFieldKey::QuantParams:
    return "quant_params";
  case PluginContractFieldKey::InputShape:
    return "input_shape";
  case PluginContractFieldKey::OutputShape:
    return "output_shape";
  case PluginContractFieldKey::InputDtype:
    return "input_dtype";
  case PluginContractFieldKey::OutputDtype:
    return "output_dtype";
  case PluginContractFieldKey::RoundOff:
    return "round_off";
  case PluginContractFieldKey::SliceShape:
    return "slice_shape";
  case PluginContractFieldKey::FrameType:
    return "frame_type";
  case PluginContractFieldKey::AlignC16:
    return "align_c16";
  case PluginContractFieldKey::Cblock:
    return "cblock";
  case PluginContractFieldKey::ModelPath:
    return "model_path";
  case PluginContractFieldKey::BatchSize:
    return "batch_size";
  case PluginContractFieldKey::BatchSzModel:
    return "batch_sz_model";
  case PluginContractFieldKey::DispatcherOutputSizes:
    return "dispatcher_output_sizes";
  case PluginContractFieldKey::DispatcherOutputNames:
    return "dispatcher_output_names";
  case PluginContractFieldKey::PerHeadInputShape:
    return "per_head_input_shape";
  case PluginContractFieldKey::PerHeadQuantParams:
    return "per_head_quant_params";
  case PluginContractFieldKey::FrameShape:
    return "frame_shape";
  case PluginContractFieldKey::LogicalInputs:
    return "logical_inputs";
  case PluginContractFieldKey::InputBindings:
    return "input_bindings";
  case PluginContractFieldKey::SliceGeometry:
    return "slice_geometry";
  case PluginContractFieldKey::RouteFlags:
    return "route_flags";
  case PluginContractFieldKey::DecodeTypeOption:
    return "decode_type_option";
  case PluginContractFieldKey::ScoreActivation:
    return "score_activation";
  }
  return "unknown";
}

std::optional<PluginContractFieldKey> plugin_contract_field_key_from_name(std::string_view name) {
  constexpr auto kAll = std::array<PluginContractFieldKey, 23>{
      PluginContractFieldKey::QuantParams,
      PluginContractFieldKey::InputShape,
      PluginContractFieldKey::InputDtype,
      PluginContractFieldKey::OutputDtype,
      PluginContractFieldKey::RoundOff,
      PluginContractFieldKey::SliceShape,
      PluginContractFieldKey::FrameType,
      PluginContractFieldKey::AlignC16,
      PluginContractFieldKey::Cblock,
      PluginContractFieldKey::ModelPath,
      PluginContractFieldKey::BatchSize,
      PluginContractFieldKey::BatchSzModel,
      PluginContractFieldKey::DispatcherOutputSizes,
      PluginContractFieldKey::DispatcherOutputNames,
      PluginContractFieldKey::PerHeadInputShape,
      PluginContractFieldKey::PerHeadQuantParams,
      PluginContractFieldKey::FrameShape,
      PluginContractFieldKey::LogicalInputs,
      PluginContractFieldKey::InputBindings,
      PluginContractFieldKey::SliceGeometry,
      PluginContractFieldKey::RouteFlags,
      PluginContractFieldKey::DecodeTypeOption,
      PluginContractFieldKey::ScoreActivation,
  };
  for (const auto key : kAll) {
    if (plugin_contract_field_key_name(key) == name) {
      return key;
    }
  }
  return std::nullopt;
}

const PluginContractFamilyDeclaration& plugin_contract_family_declaration(std::string_view family) {
  for (const auto& declaration : kDeclarations) {
    if (declaration.family == family) {
      return declaration;
    }
  }
  throw std::invalid_argument("unknown plugin contract family '" + std::string(family) + "'");
}

QuantizeContractSubset extract_quantize_contract_subset_from_mpk(const MpkContract& contract) {
  const auto* stage = find_first_stage_for_family(contract, "quant");
  if (!stage) {
    throw std::runtime_error("quantize contract subset extraction requires a quant stage");
  }
  return extract_quantize_contract_subset_from_stage(*stage);
}

QuantizeContractSubset
extract_quantize_contract_subset_from_stage(const MpkPluginIoContract& stage) {
  if (canonical_stage_family(stage) != "quant" && canonical_stage_family(stage) != "quanttess") {
    throw std::runtime_error(
        "quantize contract subset extraction requires a quant/quanttess stage");
  }
  if (stage.input_tensors.empty() || stage.output_tensors.empty()) {
    throw std::runtime_error("quantize contract subset extraction missing tensor metadata");
  }
  require_single_batch_stage(stage, "quantize");

  QuantizeContractSubset subset;
  subset.quant_params = stage.quant.value_or(MpkQuantContract{});
  subset.input_shape = preferred_stage_input_tensor_shape(stage, stage.input_tensors.front());
  subset.input_dtype = normalize_dtype_token(!stage.canonical_input_dtype.empty()
                                                 ? stage.canonical_input_dtype
                                                 : stage.input_tensors.front().dtype);
  subset.output_dtype = normalize_dtype_token(!stage.canonical_output_dtype.empty()
                                                  ? stage.canonical_output_dtype
                                                  : stage.output_tensors.front().dtype);
  subset.round_off = normalize_round_off_token(stage.round_off, stage.name);

  const std::array present = {
      PluginContractFieldKey::QuantParams, PluginContractFieldKey::InputShape,
      PluginContractFieldKey::InputDtype,  PluginContractFieldKey::OutputDtype,
      PluginContractFieldKey::RoundOff,
  };
  validate_present_fields("quantize", present, stage.name);
  require_non_empty_value(subset.quant_params.scales, "quantize",
                          PluginContractFieldKey::QuantParams, stage.name);
  require_non_empty_value(subset.quant_params.zero_points, "quantize",
                          PluginContractFieldKey::QuantParams, stage.name);
  require_non_empty_value(subset.input_shape, "quantize", PluginContractFieldKey::InputShape,
                          stage.name);
  require_non_empty_value(subset.input_dtype, "quantize", PluginContractFieldKey::InputDtype,
                          stage.name);
  require_non_empty_value(subset.output_dtype, "quantize", PluginContractFieldKey::OutputDtype,
                          stage.name);
  return subset;
}

std::vector<CastContractSubset>
extract_cast_contract_subsets_from_mpk(const MpkContract& contract) {
  const auto stages = collect_stages_for_family(contract, "cast");
  if (stages.empty()) {
    throw std::runtime_error("cast contract subset extraction requires at least one cast stage");
  }

  std::vector<CastContractSubset> subsets;
  subsets.reserve(stages.size());
  for (const auto* stage : stages) {
    if (!stage) {
      continue;
    }
    subsets.push_back(extract_cast_contract_subset_from_stage(*stage));
  }
  return subsets;
}

CastContractSubset extract_cast_contract_subset_from_stage(const MpkPluginIoContract& stage) {
  if (canonical_stage_family(stage) != "cast") {
    throw std::runtime_error("cast contract subset extraction requires a cast stage");
  }
  if (stage.input_tensors.empty() || stage.output_tensors.empty()) {
    throw std::runtime_error("cast contract subset extraction missing tensor metadata");
  }
  require_single_batch_stage(stage, "cast");

  CastContractSubset subset;
  subset.input_shape =
      preferred_geometry_shape_from_stage_tensor(stage, &stage.input_tensors.front(), false);
  subset.input_dtype = normalize_dtype_token(!stage.canonical_input_dtype.empty()
                                                 ? stage.canonical_input_dtype
                                                 : stage.input_tensors.front().dtype);
  subset.output_dtype = normalize_dtype_token(!stage.canonical_output_dtype.empty()
                                                  ? stage.canonical_output_dtype
                                                  : stage.output_tensors.front().dtype);

  const std::array present = {
      PluginContractFieldKey::InputShape,
      PluginContractFieldKey::InputDtype,
      PluginContractFieldKey::OutputDtype,
  };
  validate_present_fields("cast", present, stage.name);
  require_non_empty_value(subset.input_shape, "cast", PluginContractFieldKey::InputShape,
                          stage.name);
  require_non_empty_value(subset.input_dtype, "cast", PluginContractFieldKey::InputDtype,
                          stage.name);
  require_non_empty_value(subset.output_dtype, "cast", PluginContractFieldKey::OutputDtype,
                          stage.name);
  return subset;
}

TessellateContractSubset extract_tessellate_contract_subset_from_mpk(const MpkContract& contract) {
  const auto* stage = find_first_stage_for_family(contract, "tess");
  if (!stage) {
    throw std::runtime_error("tessellate contract subset extraction requires a tess stage");
  }
  return extract_tessellate_contract_subset_from_stage(*stage);
}

TessellateContractSubset
extract_tessellate_contract_subset_from_stage(const MpkPluginIoContract& stage) {
  if (canonical_stage_family(stage) != "tess" && canonical_stage_family(stage) != "quanttess") {
    throw std::runtime_error(
        "tessellate contract subset extraction requires a tess/quanttess stage");
  }
  if (stage.input_tensors.empty()) {
    throw std::runtime_error("tessellate contract subset extraction missing input tensor");
  }
  require_single_batch_stage(stage, "tessellate");

  TessellateContractSubset subset;
  // Pre-MLA tessellate consumes the canonical tensor contract directly. When an
  // exact-stage MPK omits per-node input_shapes, mpk_shape is empty here even
  // though legacy stage geometry may still be partially authored. Falling
  // back to logical_shape would silently drop the batch axis and drift the
  // processcvu payload away from the MPK contract.
  subset.input_shape = preferred_stage_input_tensor_shape(stage, stage.input_tensors.front());
  subset.frame_type = normalize_dtype_token(stage.frame_type);
  subset.slice_shape = stage.slice_shape;
  subset.align_c16 = stage.has_align_c16 && stage.align_c16;
  subset.cblock = stage.has_cblock && stage.cblock;
  subset.batch_size = explicit_stage_batch_size_or_one(stage);
  for (const auto& tensor : stage.output_tensors) {
    subset.output_size_bytes += static_cast<std::uint64_t>(tensor.size_bytes);
  }

  const std::array present = {
      PluginContractFieldKey::InputShape, PluginContractFieldKey::FrameType,
      PluginContractFieldKey::SliceShape, PluginContractFieldKey::AlignC16,
      PluginContractFieldKey::Cblock,
  };
  validate_present_fields("tessellate", present, stage.name);
  require_non_empty_value(subset.input_shape, "tessellate", PluginContractFieldKey::InputShape,
                          stage.name);
  require_non_empty_value(subset.frame_type, "tessellate", PluginContractFieldKey::FrameType,
                          stage.name);
  require_non_empty_value(subset.slice_shape, "tessellate", PluginContractFieldKey::SliceShape,
                          stage.name);
  return subset;
}

QuantTessContractSubset extract_quanttess_contract_subset_from_mpk(const MpkContract& contract) {
  const auto* quant_stage = find_first_stage_for_family(contract, "quant");
  const auto* tess_stage = find_first_stage_for_family(contract, "tess");
  if (!quant_stage || !tess_stage) {
    throw std::runtime_error("quanttess contract subset extraction requires quant and tess stages");
  }
  if (quant_stage == tess_stage) {
    return extract_quanttess_contract_subset_from_stage(*quant_stage);
  }
  if (quant_stage->input_tensors.empty()) {
    throw std::runtime_error("quanttess contract subset extraction missing quant input tensor");
  }

  QuantTessContractSubset subset;
  subset.quant_params = quant_stage->quant.value_or(MpkQuantContract{});
  subset.input_shape =
      preferred_stage_input_tensor_shape(*quant_stage, quant_stage->input_tensors.front());
  subset.output_shape = preferred_geometry_shape_from_stage_tensor(
      *tess_stage,
      tess_stage->output_tensors.empty() ? nullptr : &tess_stage->output_tensors.front(), true);
  for (const auto& tensor : tess_stage->output_tensors) {
    subset.output_size_bytes += static_cast<std::uint64_t>(tensor.size_bytes);
  }
  subset.input_dtype = normalize_dtype_token(!quant_stage->canonical_input_dtype.empty()
                                                 ? quant_stage->canonical_input_dtype
                                                 : quant_stage->input_tensors.front().dtype);
  subset.output_dtype = normalize_dtype_token(!quant_stage->canonical_output_dtype.empty()
                                                  ? quant_stage->canonical_output_dtype
                                                  : quant_stage->output_tensors.front().dtype);
  subset.round_off = normalize_round_off_token(quant_stage->round_off, quant_stage->name);
  subset.slice_shape = tess_stage->slice_shape;
  subset.frame_type = normalize_dtype_token(tess_stage->frame_type);
  subset.align_c16 = tess_stage->has_align_c16 && tess_stage->align_c16;
  subset.cblock = tess_stage->has_cblock && tess_stage->cblock;
  subset.batch_size = explicit_pair_batch_size_or_one(*quant_stage, *tess_stage, "quanttess");

  const std::array present = {
      PluginContractFieldKey::QuantParams, PluginContractFieldKey::InputShape,
      PluginContractFieldKey::InputDtype,  PluginContractFieldKey::OutputDtype,
      PluginContractFieldKey::RoundOff,    PluginContractFieldKey::SliceShape,
      PluginContractFieldKey::FrameType,   PluginContractFieldKey::AlignC16,
      PluginContractFieldKey::Cblock,
  };
  validate_present_fields("quanttess", present, quant_stage->name);
  require_non_empty_value(subset.quant_params.scales, "quanttess",
                          PluginContractFieldKey::QuantParams, quant_stage->name);
  require_non_empty_value(subset.input_shape, "quanttess", PluginContractFieldKey::InputShape,
                          quant_stage->name);
  require_non_empty_value(subset.input_dtype, "quanttess", PluginContractFieldKey::InputDtype,
                          quant_stage->name);
  require_non_empty_value(subset.output_dtype, "quanttess", PluginContractFieldKey::OutputDtype,
                          quant_stage->name);
  require_non_empty_value(subset.slice_shape, "quanttess", PluginContractFieldKey::SliceShape,
                          tess_stage->name);
  require_non_empty_value(subset.frame_type, "quanttess", PluginContractFieldKey::FrameType,
                          tess_stage->name);

  return subset;
}

QuantTessContractSubset
extract_quanttess_contract_subset_from_stage(const MpkPluginIoContract& stage) {
  if (canonical_stage_family(stage) != "quanttess") {
    throw std::runtime_error("quanttess contract subset extraction requires a quanttess stage");
  }
  if (stage.input_tensors.empty() || stage.output_tensors.empty()) {
    throw std::runtime_error("quanttess contract subset extraction missing tensor metadata");
  }

  QuantTessContractSubset subset;
  subset.quant_params = stage.quant.value_or(MpkQuantContract{});
  subset.input_shape = preferred_stage_input_tensor_shape(stage, stage.input_tensors.front());
  subset.output_shape = preferred_geometry_shape_from_stage_tensor(
      stage, stage.output_tensors.empty() ? nullptr : &stage.output_tensors.front(), true);
  for (const auto& tensor : stage.output_tensors) {
    subset.output_size_bytes += static_cast<std::uint64_t>(tensor.size_bytes);
  }
  subset.input_dtype = normalize_dtype_token(!stage.canonical_input_dtype.empty()
                                                 ? stage.canonical_input_dtype
                                                 : stage.input_tensors.front().dtype);
  subset.output_dtype = normalize_dtype_token(!stage.canonical_output_dtype.empty()
                                                  ? stage.canonical_output_dtype
                                                  : stage.output_tensors.front().dtype);
  subset.round_off = normalize_round_off_token(stage.round_off, stage.name);
  subset.slice_shape = stage.slice_shape;
  subset.frame_type = normalize_dtype_token(stage.frame_type);
  subset.align_c16 = stage.has_align_c16 && stage.align_c16;
  subset.cblock = stage.has_cblock && stage.cblock;
  subset.batch_size = explicit_stage_batch_size_or_one(stage);

  const std::array present = {
      PluginContractFieldKey::QuantParams, PluginContractFieldKey::InputShape,
      PluginContractFieldKey::InputDtype,  PluginContractFieldKey::OutputDtype,
      PluginContractFieldKey::RoundOff,    PluginContractFieldKey::SliceShape,
      PluginContractFieldKey::FrameType,   PluginContractFieldKey::AlignC16,
      PluginContractFieldKey::Cblock,
  };
  validate_present_fields("quanttess", present, stage.name);
  require_non_empty_value(subset.quant_params.scales, "quanttess",
                          PluginContractFieldKey::QuantParams, stage.name);
  require_non_empty_value(subset.input_shape, "quanttess", PluginContractFieldKey::InputShape,
                          stage.name);
  require_non_empty_value(subset.input_dtype, "quanttess", PluginContractFieldKey::InputDtype,
                          stage.name);
  require_non_empty_value(subset.output_dtype, "quanttess", PluginContractFieldKey::OutputDtype,
                          stage.name);
  require_non_empty_value(subset.slice_shape, "quanttess", PluginContractFieldKey::SliceShape,
                          stage.name);
  require_non_empty_value(subset.frame_type, "quanttess", PluginContractFieldKey::FrameType,
                          stage.name);

  return subset;
}

std::vector<DequantizeContractSubset>
extract_dequantize_contract_subsets_from_mpk(const MpkContract& contract) {
  const auto stages = collect_stages_for_family(contract, "dequant");
  if (stages.empty()) {
    throw std::runtime_error(
        "dequantize contract subset extraction requires at least one dequant stage");
  }

  std::vector<DequantizeContractSubset> subsets;
  subsets.reserve(stages.size());
  for (const auto* stage : stages) {
    if (!stage) {
      continue;
    }
    if (stage->input_tensors.empty() || stage->output_tensors.empty()) {
      throw std::runtime_error("dequantize contract subset extraction missing tensor metadata");
    }
    require_single_batch_stage(*stage, "dequantize");

    DequantizeContractSubset subset;
    subset.quant_params = stage->quant.value_or(MpkQuantContract{});
    subset.input_shape = preferred_shape_from_tensor(stage->input_tensors.front());
    subset.output_shape = preferred_shape_from_tensor(stage->output_tensors.front());
    subset.input_dtype = normalize_dtype_token(!stage->canonical_input_dtype.empty()
                                                   ? stage->canonical_input_dtype
                                                   : stage->input_tensors.front().dtype);
    subset.output_dtype = normalize_dtype_token(!stage->canonical_output_dtype.empty()
                                                    ? stage->canonical_output_dtype
                                                    : stage->output_tensors.front().dtype);

    const std::array present = {
        PluginContractFieldKey::QuantParams, PluginContractFieldKey::InputShape,
        PluginContractFieldKey::OutputShape, PluginContractFieldKey::InputDtype,
        PluginContractFieldKey::OutputDtype,
    };
    validate_present_fields("dequantize", present, stage->name);
    require_non_empty_value(subset.quant_params.scales, "dequantize",
                            PluginContractFieldKey::QuantParams, stage->name);
    require_non_empty_value(subset.quant_params.zero_points, "dequantize",
                            PluginContractFieldKey::QuantParams, stage->name);
    require_non_empty_value(subset.input_shape, "dequantize", PluginContractFieldKey::InputShape,
                            stage->name);
    require_non_empty_value(subset.output_shape, "dequantize", PluginContractFieldKey::OutputShape,
                            stage->name);
    require_non_empty_value(subset.input_dtype, "dequantize", PluginContractFieldKey::InputDtype,
                            stage->name);
    require_non_empty_value(subset.output_dtype, "dequantize", PluginContractFieldKey::OutputDtype,
                            stage->name);
    subsets.push_back(std::move(subset));
  }
  return subsets;
}

ProcessMlaContractSubset extract_processmla_contract_subset_from_static_contract(
    const MlaStaticContract& contract, const bool include_dispatcher_output_names) {
  ProcessMlaContractSubset subset;
  subset.model_path = contract.model_path;
  subset.batch_size = contract.batch_size;
  subset.batch_sz_model = contract.batch_sz_model;
  subset.dispatcher_output_sizes.reserve(contract.dispatcher_physical_outputs.size());
  if (include_dispatcher_output_names) {
    subset.dispatcher_output_names.reserve(contract.dispatcher_physical_outputs.size());
  }
  for (const auto& output : contract.dispatcher_physical_outputs) {
    subset.dispatcher_output_sizes.push_back(output.size_bytes);
    if (include_dispatcher_output_names) {
      subset.dispatcher_output_names.push_back(output.segment_name);
    }
  }

  const std::array present = {
      PluginContractFieldKey::ModelPath,
      PluginContractFieldKey::BatchSize,
      PluginContractFieldKey::BatchSzModel,
      PluginContractFieldKey::DispatcherOutputSizes,
  };
  validate_present_fields("processmla", present, contract.node_name);
  require_non_empty_value(subset.model_path, "processmla", PluginContractFieldKey::ModelPath,
                          contract.node_name);
  require_non_empty_value(subset.batch_size, "processmla", PluginContractFieldKey::BatchSize,
                          contract.node_name);
  require_non_empty_value(subset.batch_sz_model, "processmla", PluginContractFieldKey::BatchSzModel,
                          contract.node_name);
  require_non_empty_value(subset.dispatcher_output_sizes, "processmla",
                          PluginContractFieldKey::DispatcherOutputSizes, contract.node_name);
  return subset;
}

const MpkTensorContract*
match_published_output_for_transport(const std::vector<MpkTensorContract>& published_outputs,
                                     const std::string& transport_name,
                                     std::size_t fallback_index) {
  if (!transport_name.empty()) {
    for (const auto& candidate : published_outputs) {
      if (candidate.name == transport_name) {
        return &candidate;
      }
    }
    if (fallback_index < published_outputs.size()) {
      return &published_outputs[fallback_index];
    }
    std::ostringstream names;
    names << "[";
    for (std::size_t i = 0; i < published_outputs.size(); ++i) {
      if (i > 0) {
        names << ", ";
      }
      names << (published_outputs[i].name.empty() ? "<unnamed>" : published_outputs[i].name);
    }
    names << "]";
    throw std::runtime_error(
        "Model tensor names do not match the expected model connections. Neat could not find "
        "output tensor '" +
        transport_name + "'. Available model outputs: " + names.str() +
        ". Rebuild the model package with consistent tensor names, or update the graph to use "
        "one of the available output names.");
  }
  if (fallback_index < published_outputs.size()) {
    return &published_outputs[fallback_index];
  }
  return nullptr;
}

std::vector<DetessellateContractSubset>
extract_detessellate_contract_subsets_from_mpk(const MpkContract& contract) {
  const auto stages = collect_stages_for_family(contract, "detess");
  if (stages.empty()) {
    throw std::runtime_error(
        "detessellate contract subset extraction requires at least one detess stage");
  }
  const auto mla_published_outputs = get_mla_published_outputs_contract(contract);
  if (mla_published_outputs.size() < stages.size()) {
    throw std::runtime_error(
        "detessellate contract subset extraction requires MLA published boundary views");
  }

  std::vector<DetessellateContractSubset> subsets;
  subsets.reserve(stages.size());
  for (std::size_t i = 0; i < stages.size(); ++i) {
    const auto* stage = stages[i];
    if (!stage) {
      continue;
    }
    require_single_batch_stage(*stage, "detessellate");
    if (stage->input_tensors.empty()) {
      throw std::runtime_error(
          "detessellate contract subset extraction requires one packed input tensor");
    }

    DetessellateContractSubset subset;
    const auto* published_output_ptr = match_published_output_for_transport(
        mla_published_outputs, stage->input_tensors.front().name, i);
    if (published_output_ptr == nullptr) {
      throw std::runtime_error(
          "detessellate contract subset extraction could not match a published MLA boundary for '" +
          stage->name + "'");
    }
    const auto& published_output = *published_output_ptr;
    const auto boundary = detess_boundary_shape_view_from_stage(
        *stage, &stage->input_tensors.front(), published_output.size_bytes);
    subset.input_shape = boundary.logical_input_shape;
    subset.input_transport_shape = boundary.transport_shape;
    subset.input_transport_size_bytes = boundary.transport_size_bytes;
    subset.frame_shape = stage->frame_shape;
    subset.frame_type = normalize_dtype_token(stage->frame_type);
    subset.slice_shape = stage->slice_shape;
    subset.align_c16 = stage->has_align_c16 && stage->align_c16;
    subset.cblock = stage->has_cblock && stage->cblock;

    const std::array present = {
        PluginContractFieldKey::InputShape, PluginContractFieldKey::FrameShape,
        PluginContractFieldKey::FrameType,  PluginContractFieldKey::SliceShape,
        PluginContractFieldKey::AlignC16,   PluginContractFieldKey::Cblock,
    };
    validate_present_fields("detessellate", present, stage->name);
    require_non_empty_value(subset.input_shape, "detessellate", PluginContractFieldKey::InputShape,
                            stage->name);
    require_non_empty_value(subset.frame_shape, "detessellate", PluginContractFieldKey::FrameShape,
                            stage->name);
    require_non_empty_value(subset.frame_type, "detessellate", PluginContractFieldKey::FrameType,
                            stage->name);
    require_non_empty_value(subset.slice_shape, "detessellate", PluginContractFieldKey::SliceShape,
                            stage->name);
    if (subset.input_transport_shape.empty() || subset.input_transport_size_bytes == 0U) {
      throw std::runtime_error(
          "detessellate contract subset extraction requires packed input bytes for stage '" +
          stage->name + "'");
    }
    subsets.push_back(std::move(subset));
  }
  return subsets;
}

std::vector<DetessDequantStagePair>
resolve_detessdequant_stage_pairs_from_mpk(const MpkContract& contract) {
  std::vector<DetessDequantStagePair> pairs;
  const auto ordered = plugins_in_execution_order(contract);
  const auto outgoing_edges = build_outgoing_edges_local(contract);
  pairs.reserve(ordered.size());

  for (const auto index : ordered) {
    if (index >= contract.plugins.size()) {
      continue;
    }
    const auto& stage = contract.plugins[index];
    const std::string family = canonical_stage_family(stage);
    if (family == "detessdequant") {
      pairs.push_back(DetessDequantStagePair{&stage, &stage});
      continue;
    }
    if (family != "detess") {
      continue;
    }

    const auto* dequant = resolve_dequant_stage_for_detess_local(contract, stage, outgoing_edges);
    if (!dequant) {
      throw std::runtime_error("detessdequant contract subset extraction requires one downstream "
                               "dequant stage for detess '" +
                               stage.name + "'");
    }
    pairs.push_back(DetessDequantStagePair{&stage, dequant});
  }

  return pairs;
}

DetessDequantContractSubset
extract_detessdequant_contract_subset_from_mpk(const MpkContract& contract) {
  const auto stage_pairs = resolve_detessdequant_stage_pairs_from_mpk(contract);
  if (stage_pairs.empty()) {
    throw std::runtime_error(
        "detessdequant contract subset extraction requires detess/dequant heads");
  }
  const std::size_t count = stage_pairs.size();
  const auto mla_published_outputs = get_mla_published_outputs_contract(contract);
  if (mla_published_outputs.size() < count) {
    throw std::runtime_error(
        "detessdequant contract subset extraction requires MLA published boundary views");
  }
  DetessDequantContractSubset subset;
  subset.heads.reserve(count);
  for (std::size_t i = 0; i < count; ++i) {
    const auto& detess = *stage_pairs[i].detess;
    const auto& dequant = *stage_pairs[i].dequant;
    if (detess.input_tensors.empty()) {
      throw std::runtime_error(
          "detessdequant contract subset extraction missing detess input tensor");
    }
    // Match the published MLA boundary to THIS detess by its transport-tensor name, not by
    // position. A heterogeneous egress -- where one MLA output is published in normal layout and
    // bypasses detessellation -- makes the detess-head list a SUBSET of the published-output list,
    // so positional indexing (mla_published_outputs[i]) grabs the wrong boundary (e.g. output_0's
    // 627200-byte size attached to output_1's detess head). The detess input tensor name equals
    // the published unpack-output name, so match on that; fall back to positional for legacy
    // homogeneous MPKs whose names may not align.
    const std::string& transport_name = detess.input_tensors.front().name;
    const MpkTensorContract* published_input_ptr = nullptr;
    if (!transport_name.empty()) {
      for (const auto& candidate : mla_published_outputs) {
        if (candidate.name == transport_name) {
          published_input_ptr = &candidate;
          break;
        }
      }
    }
    if (published_input_ptr == nullptr && i < mla_published_outputs.size()) {
      published_input_ptr = &mla_published_outputs[i];
    }
    if (published_input_ptr == nullptr) {
      throw std::runtime_error("detessdequant contract subset extraction could not match a "
                               "published MLA boundary for '" +
                               detess.name + "'");
    }
    const auto& published_input = *published_input_ptr;
    DetessDequantHeadContractSubset head;
    const auto boundary = detess_boundary_shape_view_from_stage(
        detess, &detess.input_tensors.front(), published_input.size_bytes);
    head.per_head_input_shape = boundary.logical_input_shape;
    head.input_transport_shape = boundary.transport_shape;
    head.input_transport_size_bytes = boundary.transport_size_bytes;
    head.per_head_quant_params = dequant.quant.value_or(MpkQuantContract{});
    head.frame_shape = detess.frame_shape;
    head.frame_type = normalize_dtype_token(detess.frame_type);
    head.slice_shape = detess.slice_shape;
    head.align_c16 = detess.has_align_c16 && detess.align_c16;
    head.cblock = detess.has_cblock && detess.cblock;
    head.output_dtype = normalize_dtype_token(!dequant.canonical_output_dtype.empty()
                                                  ? dequant.canonical_output_dtype
                                                  : dequant.output_tensors.front().dtype);

    const std::array present = {
        PluginContractFieldKey::PerHeadInputShape,
        PluginContractFieldKey::PerHeadQuantParams,
        PluginContractFieldKey::FrameShape,
        PluginContractFieldKey::FrameType,
        PluginContractFieldKey::SliceShape,
        PluginContractFieldKey::AlignC16,
        PluginContractFieldKey::Cblock,
        PluginContractFieldKey::OutputDtype,
    };
    validate_present_fields("detessdequant", present, detess.name);
    require_non_empty_value(head.per_head_input_shape, "detessdequant",
                            PluginContractFieldKey::PerHeadInputShape, detess.name);
    require_non_empty_value(head.per_head_quant_params.scales, "detessdequant",
                            PluginContractFieldKey::PerHeadQuantParams, dequant.name);
    require_non_empty_value(head.frame_shape, "detessdequant", PluginContractFieldKey::FrameShape,
                            detess.name);
    require_non_empty_value(head.frame_type, "detessdequant", PluginContractFieldKey::FrameType,
                            detess.name);
    require_non_empty_value(head.slice_shape, "detessdequant", PluginContractFieldKey::SliceShape,
                            detess.name);
    require_non_empty_value(head.output_dtype, "detessdequant", PluginContractFieldKey::OutputDtype,
                            dequant.name);
    if (head.input_transport_shape.empty() || head.input_transport_size_bytes == 0U) {
      throw std::runtime_error("detessdequant contract subset extraction requires packed transport "
                               "geometry for '" +
                               detess.name + "'");
    }
    const std::string published_dtype = normalize_dtype_token(
        !published_input.dtype.empty() ? published_input.dtype : head.frame_type);
    const std::uint64_t published_size = static_cast<std::uint64_t>(published_input.size_bytes);
    if (published_dtype != head.frame_type || published_size == 0U ||
        published_size != head.input_transport_size_bytes) {
      throw std::runtime_error(
          "detessdequant contract subset extraction requires the published MLA "
          "transport view to match detess input '" +
          detess.name + "'");
    }
    validate_detessdequant_head_contract_local(detess, dequant, head);
    if (detess_layout_debug_enabled_local()) {
      std::fprintf(stderr,
                   "[detess-layout-debug] where=subset.extract index=%zu detess=%s dequant=%s "
                   "logical_input_shape=%s transport_shape=%s frame_shape=%s slice_shape=%s "
                   "frame_type=%s output_dtype=%s published_name=%s published_size=%" PRIu64 "\n",
                   i, detess.name.c_str(), dequant.name.c_str(),
                   join_i64_debug_local(head.per_head_input_shape).c_str(),
                   join_i64_debug_local(head.input_transport_shape).c_str(),
                   join_i64_debug_local(head.frame_shape).c_str(),
                   join_i64_debug_local(head.slice_shape).c_str(), head.frame_type.c_str(),
                   head.output_dtype.c_str(), published_input.name.c_str(), published_size);
    }
    subset.heads.push_back(std::move(head));
  }
  return subset;
}

BoxDecodeContractSubset
extract_boxdecode_contract_subset_from_static_contract(const BoxDecodeStaticContract& contract) {
  BoxDecodeContractSubset subset;
  subset.logical_inputs.reserve(contract.tensors.size());
  subset.input_bindings.reserve(contract.tensors.size());
  subset.slice_shapes.reserve(contract.tensors.size());
  subset.tensor_storage_kind.reserve(contract.tensors.size());
  subset.decode_type = contract.decode_type;
  subset.tess_needed = contract.tess_needed;
  subset.quant_needed = contract.quant_needed;
  if (contract.decode_type_option != BoxDecodeTypeOption::Auto) {
    subset.decode_type_option = contract.decode_type_option;
  }
  subset.score_activation = contract.score_activation;
  subset.num_classes = contract.num_classes;

  auto fill_shape_desc = [](const std::vector<int>& shape, sima_ev_shape_desc* out) -> bool {
    if (!out || shape.empty() || shape.size() > SIMA_EV_MAX_RANK) {
      return false;
    }
    std::memset(out, 0, sizeof(*out));
    out->rank = static_cast<uint32_t>(shape.size());
    for (std::size_t i = 0; i < shape.size(); ++i) {
      if (shape[i] <= 0) {
        return false;
      }
      out->sizes[i] = static_cast<int64_t>(shape[i]);
    }
    return true;
  };

  for (std::size_t i = 0; i < contract.tensors.size(); ++i) {
    auto logical = logical_input_from_boxdecode_static_contract(contract, i);
    subset.input_bindings.push_back(
        input_binding_from_boxdecode_static_contract(contract, logical, i));
    subset.logical_inputs.push_back(std::move(logical));
    sima_ev_shape_desc slice_shape_desc{};
    if (!fill_shape_desc(contract.tensors[i].slice_shape, &slice_shape_desc)) {
      throw std::invalid_argument("boxdecode static contract requires positive slice_shape dims");
    }
    subset.slice_shapes.push_back(slice_shape_desc);
    if (contract.tensors[i].source_storage_kind == BoxDecodeSourceStorageKind::Unknown) {
      throw std::invalid_argument(
          "boxdecode source storage kind is unspecified: provide a model pack, or set "
          "source_storage "
          "explicitly on SimaBoxDecode (it cannot be inferred from a hand-built upstream)");
    }
    subset.tensor_storage_kind.push_back(static_cast<int>(contract.tensors[i].source_storage_kind));
  }
  return subset;
}

std::optional<BoxDecodeContractSubset> extract_boxdecode_contract_subset_from_mpk(
    const MpkContract& contract, const ModelManagedRouteFlags& route_flags,
    const MpkPluginIoContract* terminal_stage, std::string* error_message) {
  auto extracted = build_boxdecode_static_contract_from_mpk(contract, route_flags, terminal_stage,
                                                            error_message);
  if (!extracted.has_value()) {
    return std::nullopt;
  }
  // Resolve the SSD defaults before lowering, so the subset carries a valid class count
  // and layout into neatobjectdecode instead of the raw MPK defaults (Unknown/Auto/0).
  stagesemantics::apply_ssd_model_managed_contract_defaults(&*extracted);
  return extract_boxdecode_contract_subset_from_static_contract(*extracted);
}

void validate_boxdecode_contract_subset(const BoxDecodeContractSubset& subset,
                                        const std::string& stage_name) {
  const std::array present = {
      PluginContractFieldKey::LogicalInputs,
      PluginContractFieldKey::InputBindings,
      PluginContractFieldKey::SliceGeometry,
      PluginContractFieldKey::RouteFlags,
  };
  validate_present_fields("boxdecode", present, stage_name);
  require_non_empty_value(subset.logical_inputs, "boxdecode", PluginContractFieldKey::LogicalInputs,
                          stage_name);
  require_non_empty_value(subset.input_bindings, "boxdecode", PluginContractFieldKey::InputBindings,
                          stage_name);
  require_non_empty_value(subset.slice_shapes, "boxdecode", PluginContractFieldKey::SliceGeometry,
                          stage_name);
  if (subset.logical_inputs.size() != subset.input_bindings.size() ||
      subset.logical_inputs.size() != subset.slice_shapes.size() ||
      subset.logical_inputs.size() != subset.tensor_storage_kind.size()) {
    throw std::invalid_argument("plugin contract subset 'boxdecode' requires aligned logical "
                                "inputs, bindings, slice geometry, and tensor storage kinds for "
                                "stage '" +
                                stage_name + "'");
  }
  for (std::size_t i = 0; i < subset.logical_inputs.size(); ++i) {
    const auto& logical = subset.logical_inputs[i];
    const auto& binding = subset.input_bindings[i];
    require_non_empty_value(logical.shape, "boxdecode", PluginContractFieldKey::LogicalInputs,
                            stage_name);
    require_non_empty_value(logical.dtype, "boxdecode", PluginContractFieldKey::LogicalInputs,
                            stage_name);
    require_non_empty_value(logical.logical_name, "boxdecode",
                            PluginContractFieldKey::LogicalInputs, stage_name);
    require_non_empty_value(logical.backend_name, "boxdecode",
                            PluginContractFieldKey::LogicalInputs, stage_name);
    require_non_empty_value(logical.segment_name, "boxdecode",
                            PluginContractFieldKey::LogicalInputs, stage_name);
    require_non_empty_value(binding.cm_input_name, "boxdecode",
                            PluginContractFieldKey::InputBindings, stage_name);
    require_non_empty_value(binding.source_segment_name, "boxdecode",
                            PluginContractFieldKey::InputBindings, stage_name);
    const auto& slice_shape = subset.slice_shapes[i];
    if (slice_shape.rank == 0U || slice_shape.rank > static_cast<uint32_t>(SIMA_EV_MAX_RANK) ||
        std::any_of(slice_shape.sizes, slice_shape.sizes + slice_shape.rank,
                    [](int64_t d) { return d <= 0; })) {
      throw std::invalid_argument(
          "plugin contract subset 'boxdecode' missing required field '" +
          std::string(plugin_contract_field_key_name(PluginContractFieldKey::SliceGeometry)) +
          "' for stage '" + stage_name + "'");
    }
    if (subset.quant_needed) {
      if (!logical.quant.has_value() || logical.quant->scales.empty() ||
          logical.quant->zero_points.empty()) {
        throw std::invalid_argument(
            "plugin contract subset 'boxdecode' quantized route requires per-input logical quant "
            "facts for stage '" +
            stage_name + "'");
      }
    }
  }
}

CompiledProcessCvuRuntimeConfig
build_quanttess_runtime_config_from_subset(const QuantTessContractSubset& subset,
                                           const std::string& published_output_name) {
  require_non_empty_value(subset.quant_params.scales, "quanttess",
                          PluginContractFieldKey::QuantParams, "quanttess");
  require_non_empty_value(subset.input_shape, "quanttess", PluginContractFieldKey::InputShape,
                          "quanttess");
  if (subset.output_shape.empty()) {
    throw std::invalid_argument("quanttess runtime config requires explicit fused output_shape");
  }
  require_non_empty_value(subset.input_dtype, "quanttess", PluginContractFieldKey::InputDtype,
                          "quanttess");
  require_non_empty_value(subset.output_dtype, "quanttess", PluginContractFieldKey::OutputDtype,
                          "quanttess");
  if (subset.round_off < 0) {
    throw std::invalid_argument("quanttess runtime config requires an explicit round_off");
  }
  require_non_empty_value(subset.slice_shape, "quanttess", PluginContractFieldKey::SliceShape,
                          "quanttess");
  require_non_empty_value(subset.frame_type, "quanttess", PluginContractFieldKey::FrameType,
                          "quanttess");

  // Phase 1 (Option A++): geometric invariants — debug-mode only, off unless
  // SIMA_NEAT_CONTRACT_INVARIANTS=1. Validates that input.rank == output.rank,
  // and (when slice_shape is non-empty) that slice.rank == input.rank, before
  // the descriptor builders consume the subset. Logs a diagnostic warning on
  // violation; does NOT throw or alter control flow. Intended to surface the
  // very class of bug that caci bs=10 hit (4D shape vs 3D slice).
  check_per_frame_geometric_invariants("quanttess", "quanttess", subset.input_shape,
                                       subset.output_shape, subset.slice_shape, subset.batch_size);

  CompiledProcessCvuRuntimeConfig runtime;
  runtime.graph_family = "quanttess";
  runtime.graph_name = "quantizetessellate";
  runtime.graph_id = 226;
  runtime.default_input_name = "input_tensor";
  runtime.runtime_input_names = {"input_tensor"};
  runtime.runtime_output_names = {"output_tensor"};
  runtime.physical_input_names = {"input_tensor"};
  runtime.physical_output_names = {"output_tensor"};
  runtime.published_output_names = {published_output_name.empty() ? std::string("output_tensor")
                                                                  : published_output_name};
  runtime.primary_output_name = runtime.published_output_names.front();
  // Keep batch as the explicit runtime field. Tensor descriptors below use the
  // authored MPK/runtime shapes exactly; they do not reconstruct batch from rank.
  runtime.batch_size = subset.batch_size > 0 ? subset.batch_size : 1;
  runtime.byte_align = 1;
  runtime.round_off = subset.round_off;
  runtime.tessellate = 1;

  {
    std::vector<int> input_shape_int(subset.input_shape.begin(), subset.input_shape.end());
    runtime.input_shapes = {input_shape_int};
    sima_ev_tensor_desc input_desc{};
    if (!build_tensor_dense_desc_local(input_shape_int, normalize_dtype_token(subset.input_dtype),
                                       &input_desc)) {
      throw std::invalid_argument(
          "quanttess runtime config could not synthesize typed input tensor");
    }
    runtime.input_tensors = {input_desc};
  }
  runtime.input_dtype = normalize_dtype_token(subset.input_dtype);
  runtime.output_dtype = "INT8";
  runtime.out_dtype = "INT8";
  {
    std::vector<int> output_shape_int(subset.output_shape.begin(), subset.output_shape.end());
    runtime.output_shapes = {output_shape_int};
    runtime.runtime_output_logical_shapes = {output_shape_int};
    sima_ev_tensor_desc output_desc{};
    std::vector<int> tile_shape_int(subset.slice_shape.begin(), subset.slice_shape.end());
    const std::uint32_t tile_align = resolve_tile_align_bytes_local(runtime.byte_align);
    if (!build_tensor_tiled_desc_local(output_shape_int, tile_shape_int, runtime.output_dtype,
                                       tile_align, subset.align_c16 || subset.cblock,
                                       &output_desc)) {
      throw std::invalid_argument(
          "quanttess runtime config could not synthesize typed output tensor");
    }
    int slice_d = 0;
    int slice_h = 0;
    int slice_w = 0;
    int slice_c = 0;
    if (!canonical_slice_dhwc_from_shape(subset.slice_shape, &slice_d, &slice_h, &slice_w,
                                         &slice_c)) {
      throw std::invalid_argument("quanttess runtime config missing canonical slice geometry");
    }
    const auto output_dims = dims_from_shape(subset.output_shape);
    const std::uint64_t synthesized_packed_output_bytes = tessellated_output_size_bytes_local(
        output_dims, slice_w, slice_h, slice_d > 0 ? slice_d : 1, slice_c > 0 ? slice_c : 1,
        runtime.output_dtype, subset.align_c16 || subset.cblock);
    const std::uint64_t output_storage_bytes =
        subset.output_size_bytes > 0U ? subset.output_size_bytes : synthesized_packed_output_bytes;
    if (output_storage_bytes == 0U) {
      throw std::invalid_argument("quanttess runtime config could not size typed output tensor");
    }
    output_desc.storage.nbytes = output_storage_bytes;
    runtime.output_tensors = {output_desc};
  }
  runtime.primary_output_transport_kind = ProcessCvuOutputTransportKind::Packed;
  runtime.primary_output_semantic_kind = ProcessCvuOutputSemanticKind::QuantTessTensor;
  runtime.runtime_output_logical_index_list = {0};
  runtime.runtime_output_output_slot_list = {0};
  runtime.runtime_output_physical_index_list = {0};
  runtime.runtime_output_dtype_list = {"INT8"};
  runtime.runtime_output_transport_kind_list = {ProcessCvuOutputTransportKind::Packed};
  runtime.runtime_output_semantic_kind_list = {ProcessCvuOutputSemanticKind::QuantTessTensor};
  runtime.runtime_output_logical_layout_list.clear();
  runtime.has_q_scale = true;
  runtime.q_scale = subset.quant_params.scales.front();
  runtime.has_q_zp = !subset.quant_params.zero_points.empty();
  runtime.q_zp = runtime.has_q_zp ? subset.quant_params.zero_points.front() : 0;
  runtime.q_scale_list = {runtime.q_scale};
  runtime.q_zp_list = {runtime.has_q_zp ? runtime.q_zp : 0};

  {
    std::vector<int> slice_shape_int(subset.slice_shape.begin(), subset.slice_shape.end());
    runtime.slice_shapes = {slice_shape_int};
  }
  return runtime;
}

CompiledProcessCvuRuntimeConfig build_tessellate_runtime_config_from_subsets(
    const CastContractSubset& cast_subset, const TessellateContractSubset& tess_subset,
    const std::string& physical_output_name, const std::string& published_output_name) {
  require_non_empty_value(cast_subset.input_shape, "cast", PluginContractFieldKey::InputShape,
                          "cast");
  require_non_empty_value(cast_subset.input_dtype, "cast", PluginContractFieldKey::InputDtype,
                          "cast");
  require_non_empty_value(cast_subset.output_dtype, "cast", PluginContractFieldKey::OutputDtype,
                          "cast");
  require_non_empty_value(tess_subset.input_shape, "tessellate", PluginContractFieldKey::InputShape,
                          "tessellate");
  require_non_empty_value(tess_subset.frame_type, "tessellate", PluginContractFieldKey::FrameType,
                          "tessellate");
  require_non_empty_value(tess_subset.slice_shape, "tessellate", PluginContractFieldKey::SliceShape,
                          "tessellate");

  const std::string cast_output_dtype = normalize_dtype_token(cast_subset.output_dtype);
  const std::string cast_input_dtype = normalize_dtype_token(cast_subset.input_dtype);
  const std::string frame_type = normalize_dtype_token(tess_subset.frame_type);
  if (cast_input_dtype.empty()) {
    throw std::invalid_argument("tessellate runtime config requires a normalized cast input dtype");
  }
  if (cast_output_dtype.empty()) {
    throw std::invalid_argument(
        "tessellate runtime config requires a normalized cast output dtype");
  }
  if (frame_type.empty()) {
    throw std::invalid_argument("tessellate runtime config requires a normalized frame_type");
  }
  if (cast_output_dtype != frame_type) {
    throw std::invalid_argument(
        "tessellate runtime config requires cast output dtype to match tess frame_type");
  }

  const auto dims = dims_from_shape(cast_subset.input_shape);
  int slice_d = 0;
  int slice_h = 0;
  int slice_w = 0;
  int slice_c = 0;
  if (!canonical_slice_dhwc_from_shape(tess_subset.slice_shape, &slice_d, &slice_h, &slice_w,
                                       &slice_c)) {
    throw std::invalid_argument("tessellate runtime config missing canonical slice geometry");
  }
  const std::uint64_t synthesized_packed_output_bytes = tessellated_output_size_bytes_local(
      dims, slice_w, slice_h, slice_d > 0 ? slice_d : 1, slice_c > 0 ? slice_c : 1, frame_type,
      tess_subset.align_c16 || tess_subset.cblock);
  const std::uint64_t packed_output_bytes = tess_subset.output_size_bytes > 0U
                                                ? tess_subset.output_size_bytes
                                                : synthesized_packed_output_bytes;
  if (packed_output_bytes == 0U) {
    throw std::invalid_argument("tessellate runtime config could not compute packed output size");
  }

  // Phase 1 invariants: cast.input.rank == tess.input.rank == tess.slice.rank.
  check_per_frame_geometric_invariants("tessellate", "tessellate", tess_subset.input_shape,
                                       tess_subset.input_shape, tess_subset.slice_shape,
                                       /*expected_batch_size=*/0);

  CompiledProcessCvuRuntimeConfig runtime;
  runtime.graph_family = "tessellate";
  runtime.graph_name = "tessellate";
  runtime.graph_id = 2;
  runtime.default_input_name = "input_tensor";
  runtime.runtime_input_names = {"input_tensor"};
  runtime.runtime_output_names = {"output_tensor"};
  runtime.physical_input_names = {"input_tensor"};
  runtime.physical_output_names = {!physical_output_name.empty() ? physical_output_name
                                                                 : std::string("output_tensor")};
  runtime.published_output_names = {published_output_name.empty() ? std::string("output_tensor")
                                                                  : published_output_name};
  if (tess_segment_debug_enabled_local()) {
    std::fprintf(stderr,
                 "[tess-segment-debug] where=build_tess_runtime published_output_name=%s "
                 "runtime_output_name=%s physical_output_name=%s input_shape=%s slice_shape=%s "
                 "output_size_bytes=%" PRIu64 "\n",
                 runtime.published_output_names.front().c_str(),
                 runtime.runtime_output_names.front().c_str(),
                 runtime.physical_output_names.front().c_str(),
                 join_i64_debug_local(cast_subset.input_shape).c_str(),
                 join_i64_debug_local(tess_subset.slice_shape).c_str(), packed_output_bytes);
  }
  runtime.primary_output_name = runtime.published_output_names.front();
  runtime.batch_size = tess_subset.batch_size > 0 ? tess_subset.batch_size : 1;
  if (runtime.batch_size <= 0) {
    throw std::invalid_argument("tessellate runtime config requires a positive batch_size");
  }
  runtime.byte_align = 1;
  runtime.tessellate = 1;
  {
    std::vector<int> input_shape_int(cast_subset.input_shape.begin(),
                                     cast_subset.input_shape.end());
    runtime.input_shapes = {input_shape_int};
    sima_ev_tensor_desc input_desc{};
    if (!build_tensor_dense_desc_local(input_shape_int, cast_input_dtype, &input_desc)) {
      throw std::invalid_argument(
          "tessellate runtime config could not synthesize typed input tensor");
    }
    runtime.input_tensors = {input_desc};
  }
  const auto semantic_output_shape = cast_subset.input_shape;
  {
    std::vector<int> output_shape_int(semantic_output_shape.begin(), semantic_output_shape.end());
    runtime.output_shapes = {output_shape_int};
    std::vector<int> tile_shape_int(tess_subset.slice_shape.begin(), tess_subset.slice_shape.end());
    sima_ev_tensor_desc output_desc{};
    const std::uint32_t tile_align = resolve_tile_align_bytes_local(runtime.byte_align);
    std::string desc_error;
    if (!build_tensor_tiled_desc_local(output_shape_int, tile_shape_int, frame_type, tile_align,
                                       tess_subset.align_c16 || tess_subset.cblock, &output_desc,
                                       &desc_error)) {
      std::ostringstream msg;
      msg << "tessellate runtime config could not synthesize typed output tensor"
          << " for published_output='" << runtime.published_output_names.front()
          << "' physical_output='" << runtime.physical_output_names.front() << "'";
      if (!desc_error.empty()) {
        msg << ": " << desc_error;
      }
      msg << "; semantic_output_shape=" << join_i64_debug_local(semantic_output_shape)
          << " (rank=" << semantic_output_shape.size() << ")"
          << ", slice_shape=" << join_i64_debug_local(tess_subset.slice_shape)
          << " (rank=" << tess_subset.slice_shape.size() << ")" << ", frame_type=" << frame_type
          << ", cast_input_dtype=" << cast_input_dtype
          << ", cast_output_dtype=" << cast_output_dtype
          << ", packed_output_bytes=" << packed_output_bytes
          << ", batch_size=" << runtime.batch_size
          << ". Tile/slice shape is normalized against semantic_output_shape by trailing "
             "alignment: shorter tile ranks are left-padded with 1, longer ranks may only trim "
             "leading 1s, and every normalized tile dim must be > 0 and <= the tensor dim.";
      throw std::invalid_argument(msg.str());
    }
    output_desc.storage.nbytes = packed_output_bytes;
    runtime.output_tensors = {output_desc};
  }
  runtime.primary_output_transport_kind = ProcessCvuOutputTransportKind::Packed;
  runtime.primary_output_semantic_kind = ProcessCvuOutputSemanticKind::TessellatedImage;
  runtime.input_dtype = cast_input_dtype;
  runtime.output_dtype = frame_type;
  runtime.out_dtype = frame_type;
  {
    std::vector<int> slice_shape_int(tess_subset.slice_shape.begin(),
                                     tess_subset.slice_shape.end());
    runtime.slice_shapes = {slice_shape_int};
  }
  runtime.runtime_output_logical_index_list = {0};
  runtime.runtime_output_output_slot_list = {0};
  runtime.runtime_output_physical_index_list = {0};
  runtime.runtime_output_dtype_list = {frame_type};
  runtime.runtime_output_transport_kind_list = {ProcessCvuOutputTransportKind::Packed};
  runtime.runtime_output_semantic_kind_list = {ProcessCvuOutputSemanticKind::TessellatedImage};
  {
    std::vector<int> output_shape_int(semantic_output_shape.begin(), semantic_output_shape.end());
    runtime.runtime_output_logical_shapes = {output_shape_int};
  }
  runtime.runtime_output_logical_layout_list.clear();
  return runtime;
}

CompiledProcessCvuRuntimeConfig
build_quantize_runtime_config_from_subset(const QuantizeContractSubset& subset,
                                          const std::string& physical_output_name,
                                          const std::string& published_output_name) {
  require_non_empty_value(subset.quant_params.scales, "quantize",
                          PluginContractFieldKey::QuantParams, "quantize");
  require_non_empty_value(subset.quant_params.zero_points, "quantize",
                          PluginContractFieldKey::QuantParams, "quantize");
  require_non_empty_value(subset.input_shape, "quantize", PluginContractFieldKey::InputShape,
                          "quantize");
  require_non_empty_value(subset.input_dtype, "quantize", PluginContractFieldKey::InputDtype,
                          "quantize");
  require_non_empty_value(subset.output_dtype, "quantize", PluginContractFieldKey::OutputDtype,
                          "quantize");
  if (subset.round_off < 0) {
    throw std::invalid_argument("quantize runtime config requires an explicit round_off");
  }

  const std::string input_dtype = normalize_dtype_token(subset.input_dtype);
  const std::string output_dtype = normalize_dtype_token(subset.output_dtype);
  if (input_dtype != "FP32") {
    throw std::invalid_argument("quantize runtime config requires FP32 input dtype");
  }
  if (output_dtype != "INT8" && output_dtype != "INT16" && output_dtype != "INT32") {
    throw std::invalid_argument("quantize runtime config requires INT8/INT16/INT32 output dtype");
  }

  // Phase 1 invariants: dense-only stage. Same input/output rank; no slice.
  check_per_frame_geometric_invariants("quantize", "quantize", subset.input_shape,
                                       subset.input_shape,
                                       /*slice_shape=*/std::vector<std::int64_t>{},
                                       /*expected_batch_size=*/0);

  CompiledProcessCvuRuntimeConfig runtime;
  runtime.graph_family = "quantize";
  runtime.graph_name = "quantize";
  runtime.graph_id =
      222; /* SIMA_GRAPH_QUANTIZE — graph 220 is header-only with no kernel impl in cvu-sw */
  runtime.default_input_name = "input_tensor";
  runtime.runtime_input_names = {"input_tensor"};
  runtime.runtime_output_names = {"output_tensor"};
  runtime.physical_input_names = {"input_tensor"};
  runtime.physical_output_names = {!physical_output_name.empty() ? physical_output_name
                                                                 : std::string("output_tensor")};
  runtime.published_output_names = {published_output_name.empty() ? std::string("output_tensor")
                                                                  : published_output_name};
  runtime.primary_output_name = runtime.published_output_names.front();
  runtime.batch_size = 1;
  runtime.byte_align = 1;
  runtime.round_off = subset.round_off;
  {
    std::vector<int> shape_int(subset.input_shape.begin(), subset.input_shape.end());
    runtime.input_shapes = {shape_int};
    runtime.output_shapes = {shape_int};
    sima_ev_tensor_desc input_desc{};
    sima_ev_tensor_desc output_desc{};
    if (!build_tensor_dense_desc_local(shape_int, input_dtype, &input_desc) ||
        !build_tensor_dense_desc_local(shape_int, output_dtype, &output_desc)) {
      throw std::invalid_argument("quantize runtime config could not synthesize typed tensors");
    }
    runtime.input_tensors = {input_desc};
    runtime.output_tensors = {output_desc};
  }
  runtime.input_dtype = input_dtype;
  runtime.output_dtype = output_dtype;
  runtime.out_dtype = output_dtype;
  runtime.runtime_output_logical_index_list = {0};
  runtime.runtime_output_output_slot_list = {0};
  runtime.runtime_output_physical_index_list = {0};
  runtime.runtime_output_dtype_list = {runtime.output_dtype};
  runtime.has_q_scale = true;
  runtime.q_scale = subset.quant_params.scales.front();
  runtime.has_q_zp = true;
  runtime.q_zp = subset.quant_params.zero_points.front();
  runtime.q_scale_list = {runtime.q_scale};
  runtime.q_zp_list = {runtime.q_zp};
  return runtime;
}

ProcessMlaStagePayload
build_processmla_payload_from_subset(const ProcessMlaContractSubset& subset) {
  require_non_empty_value(subset.model_path, "processmla", PluginContractFieldKey::ModelPath,
                          "processmla");
  require_non_empty_value(subset.batch_size, "processmla", PluginContractFieldKey::BatchSize,
                          "processmla");
  require_non_empty_value(subset.batch_sz_model, "processmla", PluginContractFieldKey::BatchSzModel,
                          "processmla");
  require_non_empty_value(subset.dispatcher_output_sizes, "processmla",
                          PluginContractFieldKey::DispatcherOutputSizes, "processmla");

  ProcessMlaStagePayload payload;
  payload.model_path = subset.model_path;
  payload.batch_size = subset.batch_size;
  payload.batch_sz_model = subset.batch_sz_model;
  payload.dispatcher_output_sizes = subset.dispatcher_output_sizes;
  payload.dispatcher_output_names = subset.dispatcher_output_names;
  return payload;
}

CompiledProcessCvuRuntimeConfig build_detessellate_runtime_config_from_subsets(
    const std::vector<DetessellateContractSubset>& subsets,
    const std::vector<std::string>& runtime_output_names,
    const std::vector<std::string>& published_output_names) {
  if (subsets.empty()) {
    throw std::invalid_argument("detessellate runtime config requires at least one subset");
  }

  CompiledProcessCvuRuntimeConfig runtime;
  runtime.graph_family = "detessellate";
  runtime.graph_name = "detessellate";
  runtime.graph_id = 7;
  runtime.batch_size = 0;
  runtime.byte_align = 1;
  runtime.tessellate = 1;
  runtime.runtime_input_names.reserve(subsets.size());
  runtime.runtime_output_names.reserve(subsets.size());
  runtime.published_output_names.reserve(subsets.size());
  runtime.input_shapes.reserve(subsets.size());
  runtime.slice_shapes.reserve(subsets.size());
  runtime.output_shapes.reserve(subsets.size());
  runtime.runtime_output_logical_index_list.reserve(subsets.size());
  runtime.runtime_output_output_slot_list.reserve(subsets.size());
  runtime.runtime_output_physical_index_list.reserve(subsets.size());
  runtime.runtime_output_dtype_list.reserve(subsets.size());
  runtime.runtime_output_logical_layout_list.reserve(subsets.size());

  for (std::size_t i = 0; i < subsets.size(); ++i) {
    const auto& subset = subsets[i];
    // Phase 1 invariants for each per-output detess subset. detess uses
    // frame_shape (not input_shape) as the per-frame logical view; output
    // shape is the dense view it produces. Slice rank should match frame.
    check_per_frame_geometric_invariants("detessellate", "detessellate", subset.frame_shape,
                                         subset.frame_shape, subset.slice_shape,
                                         /*expected_batch_size=*/0);
    require_non_empty_value(subset.frame_shape, "detessellate", PluginContractFieldKey::FrameShape,
                            "detessellate");
    if (subset.input_transport_shape.empty() || subset.input_transport_size_bytes == 0U) {
      throw std::invalid_argument("detessellate runtime config requires packed transport view");
    }
    require_non_empty_value(subset.frame_shape, "detessellate", PluginContractFieldKey::FrameShape,
                            "detessellate");
    require_non_empty_value(subset.frame_type, "detessellate", PluginContractFieldKey::FrameType,
                            "detessellate");
    require_non_empty_value(subset.slice_shape, "detessellate", PluginContractFieldKey::SliceShape,
                            "detessellate");
    const int subset_batch_size = inferred_batch_size_from_shape(subset.frame_shape);
    if (subset_batch_size <= 0) {
      throw std::invalid_argument("detessellate runtime config requires a positive batch_size");
    }
    if (runtime.batch_size == 0) {
      runtime.batch_size = subset_batch_size;
    } else {
      require_matching_batch_size(runtime.batch_size, subset_batch_size, "detessellate");
    }

    const auto transport_input_dims = dims_from_detess_transport_shape(
        subset.input_transport_shape, "detessellate transport input");
    if (transport_input_dims.width <= 0 || transport_input_dims.height <= 0 ||
        transport_input_dims.depth <= 0 || transport_input_dims.channels <= 0) {
      throw std::invalid_argument("detessellate runtime config requires packed transport geometry");
    }
    const auto output_dims = dims_from_detess_shape(subset.frame_shape, "detessellate output");
    int slice_d = 0;
    int slice_h = 0;
    int slice_w = 0;
    int slice_c = 0;
    if (!canonical_slice_dhwc_from_shape(subset.slice_shape, &slice_d, &slice_h, &slice_w,
                                         &slice_c)) {
      throw std::invalid_argument("detessellate runtime config missing canonical slice geometry");
    }
    const std::string frame_type = normalize_dtype_token(subset.frame_type);
    if (frame_type.empty()) {
      throw std::invalid_argument("detessellate runtime config requires a normalized frame_type");
    }
    const std::string input_name = "input_tensor_" + std::to_string(i);
    const std::string output_name =
        i < runtime_output_names.size() && !runtime_output_names[i].empty()
            ? runtime_output_names[i]
            : ("output_tensor_" + std::to_string(i));
    const std::string published_name =
        i < published_output_names.size() && !published_output_names[i].empty()
            ? published_output_names[i]
            : output_name;
    const std::string semantic_layout;
    if (runtime.default_input_name.empty()) {
      runtime.default_input_name = input_name;
      runtime.primary_output_name = published_name;
      runtime.input_dtype = frame_type;
      runtime.output_dtype = frame_type;
      runtime.out_dtype = frame_type;
    }

    runtime.runtime_input_names.push_back(input_name);
    runtime.runtime_output_names.push_back(output_name);
    runtime.published_output_names.push_back(published_name);
    {
      std::vector<int> input_shape_int(subset.frame_shape.begin(), subset.frame_shape.end());
      runtime.input_shapes.push_back(input_shape_int);
      std::vector<int> tile_shape_int(subset.slice_shape.begin(), subset.slice_shape.end());
      tile_shape_int = tensor_desc_tile_shape_from_slice_shape(input_shape_int, tile_shape_int);
      sima_ev_tensor_desc input_desc{};
      if (!build_tensor_tiled_desc_local(input_shape_int, tile_shape_int, frame_type, 0U,
                                         subset.align_c16 || subset.cblock, &input_desc)) {
        throw std::invalid_argument(
            "detessellate runtime config could not synthesize typed input tensor");
      }
      input_desc.storage.nbytes = subset.input_transport_size_bytes;
      runtime.input_tensors.push_back(input_desc);
    }
    {
      std::vector<int> slice_shape_int(subset.slice_shape.begin(), subset.slice_shape.end());
      runtime.slice_shapes.push_back(slice_shape_int);
    }
    {
      std::vector<int> output_shape_int(subset.frame_shape.begin(), subset.frame_shape.end());
      runtime.output_shapes.push_back(output_shape_int);
      sima_ev_tensor_desc output_desc{};
      if (!build_tensor_dense_desc_local(output_shape_int, frame_type, &output_desc)) {
        throw std::invalid_argument(
            "detessellate runtime config could not synthesize typed output tensor");
      }
      runtime.output_tensors.push_back(output_desc);
    }
    runtime.runtime_output_logical_index_list.push_back(static_cast<int>(i));
    runtime.runtime_output_output_slot_list.push_back(static_cast<int>(i));
    runtime.runtime_output_physical_index_list.push_back(static_cast<int>(i));
    runtime.runtime_output_dtype_list.push_back(frame_type);
    runtime.runtime_output_logical_layout_list.push_back(semantic_layout);
  }
  return runtime;
}

CompiledProcessCvuRuntimeConfig build_dequantize_runtime_config_from_subsets(
    const std::vector<DequantizeContractSubset>& subsets,
    const std::vector<std::string>& published_output_names) {
  if (subsets.empty()) {
    throw std::invalid_argument("dequantize runtime config requires at least one subset");
  }

  CompiledProcessCvuRuntimeConfig runtime;
  runtime.graph_family = "dequantize";
  runtime.graph_name = "dequantize";
  runtime.graph_id = 6;
  runtime.batch_size = 0;
  runtime.byte_align = 1;
  runtime.default_input_name = "input_tensor";
  runtime.runtime_input_names = {"input_tensor"};
  runtime.runtime_output_names = {"output_tensor"};
  runtime.physical_input_names = {"input_tensor"};
  runtime.physical_output_names = {"output_tensor"};
  runtime.published_output_names.reserve(subsets.size());
  runtime.input_shapes.reserve(subsets.size());
  runtime.output_shapes.reserve(subsets.size());
  runtime.runtime_output_logical_shapes.reserve(subsets.size());
  runtime.runtime_output_logical_index_list.reserve(subsets.size());
  runtime.runtime_output_output_slot_list.reserve(subsets.size());
  runtime.runtime_output_physical_index_list.reserve(subsets.size());
  runtime.runtime_output_dtype_list.reserve(subsets.size());
  runtime.runtime_output_transport_kind_list.reserve(subsets.size());
  runtime.runtime_output_semantic_kind_list.reserve(subsets.size());
  runtime.runtime_output_logical_layout_list.reserve(subsets.size());
  runtime.q_scale_list.reserve(subsets.size());
  runtime.q_zp_list.reserve(subsets.size());

  for (std::size_t i = 0; i < subsets.size(); ++i) {
    const auto& subset = subsets[i];
    // Phase 1 invariants: dequantize is dense-only — no slice; in/out ranks match.
    check_per_frame_geometric_invariants("dequantize", "dequantize", subset.input_shape,
                                         subset.output_shape,
                                         /*slice_shape=*/std::vector<std::int64_t>{},
                                         /*expected_batch_size=*/0);
    require_non_empty_value(subset.quant_params.scales, "dequantize",
                            PluginContractFieldKey::QuantParams, "dequantize");
    require_non_empty_value(subset.quant_params.zero_points, "dequantize",
                            PluginContractFieldKey::QuantParams, "dequantize");
    require_non_empty_value(subset.input_shape, "dequantize", PluginContractFieldKey::InputShape,
                            "dequantize");
    require_non_empty_value(subset.output_shape, "dequantize", PluginContractFieldKey::OutputShape,
                            "dequantize");
    require_non_empty_value(subset.input_dtype, "dequantize", PluginContractFieldKey::InputDtype,
                            "dequantize");
    require_non_empty_value(subset.output_dtype, "dequantize", PluginContractFieldKey::OutputDtype,
                            "dequantize");
    const int subset_batch_size = inferred_batch_size_from_shape(subset.input_shape);
    if (subset_batch_size <= 0) {
      throw std::invalid_argument("dequantize runtime config requires a positive batch_size");
    }
    if (runtime.batch_size == 0) {
      runtime.batch_size = subset_batch_size;
    } else {
      require_matching_batch_size(runtime.batch_size, subset_batch_size, "dequantize");
    }

    const std::string input_dtype = normalize_dtype_token(subset.input_dtype);
    const std::string output_dtype = normalize_dtype_token(subset.output_dtype);
    if (input_dtype != "INT8" && input_dtype != "INT16" && input_dtype != "INT32") {
      throw std::invalid_argument(
          "dequantize runtime config requires INT8/INT16/INT32 input dtype");
    }
    if (output_dtype != "FP16" && output_dtype != "FP32") {
      throw std::invalid_argument("dequantize runtime config requires FP16/FP32 output dtype");
    }

    runtime.published_output_names.push_back(i < published_output_names.size() &&
                                                     !published_output_names[i].empty()
                                                 ? published_output_names[i]
                                                 : ("output_tensor_" + std::to_string(i)));
    {
      std::vector<int> input_shape_int(subset.input_shape.begin(), subset.input_shape.end());
      runtime.input_shapes.push_back(input_shape_int);
      sima_ev_tensor_desc input_desc{};
      if (!build_tensor_dense_desc_local(input_shape_int, input_dtype, &input_desc)) {
        throw std::invalid_argument(
            "dequantize runtime config could not synthesize typed input tensor");
      }
      runtime.input_tensors.push_back(input_desc);
    }
    const auto canonical_output_shape = canonical_value_transform_shape_public(
        "dequantize", subset.input_shape, subset.output_shape, subset.input_shape);
    std::vector<int> output_shape_int(canonical_output_shape.begin(), canonical_output_shape.end());
    runtime.output_shapes.push_back(output_shape_int);
    runtime.runtime_output_logical_shapes.push_back(output_shape_int);
    sima_ev_tensor_desc output_desc{};
    if (!build_tensor_dense_desc_local(output_shape_int, output_dtype, &output_desc)) {
      throw std::invalid_argument(
          "dequantize runtime config could not synthesize typed output tensor");
    }
    runtime.output_tensors.push_back(output_desc);
    runtime.runtime_output_logical_index_list.push_back(static_cast<int>(i));
    runtime.runtime_output_output_slot_list.push_back(static_cast<int>(i));
    runtime.runtime_output_physical_index_list.push_back(static_cast<int>(i));
    runtime.runtime_output_dtype_list.push_back(output_dtype);
    runtime.runtime_output_transport_kind_list.push_back(ProcessCvuOutputTransportKind::Dense);
    runtime.runtime_output_semantic_kind_list.push_back(ProcessCvuOutputSemanticKind::Tensor);
    runtime.q_scale_list.push_back(subset.quant_params.scales.front());
    runtime.q_zp_list.push_back(static_cast<std::int64_t>(subset.quant_params.zero_points.front()));
  }

  runtime.primary_output_name = runtime.published_output_names.front();
  runtime.input_dtype = normalize_dtype_token(subsets.front().input_dtype);
  runtime.output_dtype = normalize_dtype_token(subsets.front().output_dtype);
  runtime.out_dtype = runtime.output_dtype;
  runtime.has_q_scale = true;
  runtime.q_scale = subsets.front().quant_params.scales.front();
  runtime.has_q_zp = true;
  runtime.q_zp = subsets.front().quant_params.zero_points.front();
  return runtime;
}

CompiledProcessCvuRuntimeConfig build_detessdequant_runtime_config_from_subset(
    const DetessDequantContractSubset& subset, const std::vector<std::string>& runtime_output_names,
    const std::vector<std::string>& published_output_names) {
  if (subset.heads.empty()) {
    throw std::invalid_argument("detessdequant runtime config requires at least one head");
  }

  CompiledProcessCvuRuntimeConfig runtime;
  runtime.graph_family = "detessdequant";
  runtime.graph_name = "detessdequant";
  runtime.graph_id = 8;
  runtime.batch_size = 0;
  runtime.byte_align = 1;
  runtime.tessellate = 1;
  runtime.default_input_name = "input_tensor";
  runtime.runtime_input_names = {"input_tensor"};
  runtime.physical_input_names = {"input_tensor"};
  runtime.physical_output_names = {"output_tensor"};
  runtime.runtime_output_names.reserve(subset.heads.size());
  runtime.published_output_names.reserve(subset.heads.size());
  for (std::size_t i = 0; i < subset.heads.size(); ++i) {
    runtime.runtime_output_names.push_back(i < runtime_output_names.size() &&
                                                   !runtime_output_names[i].empty()
                                               ? runtime_output_names[i]
                                               : ("output_tensor_" + std::to_string(i)));
    runtime.published_output_names.push_back(i < published_output_names.size() &&
                                                     !published_output_names[i].empty()
                                                 ? published_output_names[i]
                                                 : runtime.runtime_output_names.back());
  }
  runtime.primary_output_name = runtime.published_output_names.front();

  runtime.input_shapes.reserve(subset.heads.size());
  runtime.slice_shapes.reserve(subset.heads.size());
  runtime.output_shapes.reserve(subset.heads.size());
  runtime.input_tensors.reserve(subset.heads.size());
  runtime.output_tensors.reserve(subset.heads.size());
  runtime.runtime_output_logical_shapes.reserve(subset.heads.size());
  runtime.runtime_output_logical_index_list.reserve(subset.heads.size());
  runtime.runtime_output_output_slot_list.reserve(subset.heads.size());
  runtime.runtime_output_physical_index_list.reserve(subset.heads.size());
  runtime.runtime_output_dtype_list.reserve(subset.heads.size());
  runtime.runtime_output_transport_kind_list.reserve(subset.heads.size());
  runtime.runtime_output_semantic_kind_list.reserve(subset.heads.size());
  runtime.runtime_output_logical_layout_list.reserve(subset.heads.size());
  runtime.dq_scale_list.reserve(subset.heads.size());
  runtime.dq_zp_list.reserve(subset.heads.size());

  for (std::size_t i = 0; i < subset.heads.size(); ++i) {
    const auto& head = subset.heads[i];
    // Phase 1 invariants for each detessdequant head: frame_shape vs slice_shape.
    check_per_frame_geometric_invariants("detessdequant", "detessdequant", head.frame_shape,
                                         head.frame_shape, head.slice_shape,
                                         /*expected_batch_size=*/0);
    require_non_empty_value(head.per_head_input_shape, "detessdequant",
                            PluginContractFieldKey::PerHeadInputShape, "detessdequant");
    if (head.input_transport_shape.empty() || head.input_transport_size_bytes == 0U) {
      throw std::invalid_argument("detessdequant runtime config requires packed transport view");
    }
    require_non_empty_value(head.per_head_quant_params.scales, "detessdequant",
                            PluginContractFieldKey::PerHeadQuantParams, "detessdequant");
    require_non_empty_value(head.frame_shape, "detessdequant", PluginContractFieldKey::FrameShape,
                            "detessdequant");
    require_non_empty_value(head.frame_type, "detessdequant", PluginContractFieldKey::FrameType,
                            "detessdequant");
    require_non_empty_value(head.slice_shape, "detessdequant", PluginContractFieldKey::SliceShape,
                            "detessdequant");
    require_non_empty_value(head.output_dtype, "detessdequant", PluginContractFieldKey::OutputDtype,
                            "detessdequant");
    const int head_batch_size = inferred_batch_size_from_shape(head.frame_shape);
    if (head_batch_size <= 0) {
      throw std::invalid_argument("detessdequant runtime config requires a positive batch_size");
    }
    if (runtime.batch_size == 0) {
      runtime.batch_size = head_batch_size;
    } else {
      require_matching_batch_size(runtime.batch_size, head_batch_size, "detessdequant");
    }

    const auto transport_input_dims = dims_from_detess_transport_shape(
        head.input_transport_shape, "detessdequant transport input");
    if (transport_input_dims.width <= 0 || transport_input_dims.height <= 0 ||
        transport_input_dims.depth <= 0 || transport_input_dims.channels <= 0) {
      throw std::invalid_argument(
          "detessdequant runtime config requires packed transport geometry");
    }
    const std::uint64_t expected_transport_size =
        specbuilders::tensor_size_bytes_from_shape_dtype(head.input_transport_shape, "INT8");
    if (expected_transport_size == 0U ||
        expected_transport_size != head.input_transport_size_bytes) {
      throw std::invalid_argument(
          "detessdequant runtime config requires transport size matching packed transport shape");
    }

    const auto semantic_input_dims =
        dims_from_detess_shape(head.frame_shape, "detessdequant input");
    const std::string semantic_input_layout;
    {
      // Phase 3a (Option A++): caps fields keep the MPK-batched shape; the
      // kernel descriptor takes the per-frame shape so its rank matches the
      // (rank-equal) tile geometry, with batch_size carried separately on
      // runtime.batch_size. Pre-Phase-3a this code path padded slice_shape
      // up to the batched-rank with full-axis leading dims, which produced a
      // single super-tile spanning the whole batch — wrong for batch>1.
      std::vector<int> input_shape_int(head.frame_shape.begin(), head.frame_shape.end());
      runtime.input_shapes.push_back(input_shape_int);
      const int per_frame_rank =
          derive_per_frame_rank(head.slice_shape, /*peer_per_frame_shape=*/{});
      const auto frame_shape_per_frame =
          semantic_shape_without_batch(head.frame_shape, per_frame_rank);
      std::vector<int> input_shape_per_frame_int(frame_shape_per_frame.begin(),
                                                 frame_shape_per_frame.end());
      std::vector<int> tile_shape_int(head.slice_shape.begin(), head.slice_shape.end());
      tile_shape_int =
          tensor_desc_tile_shape_from_slice_shape(input_shape_per_frame_int, tile_shape_int);
      sima_ev_tensor_desc input_desc{};
      if (!build_tensor_tiled_desc_local(input_shape_per_frame_int, tile_shape_int,
                                         normalize_dtype_token(head.frame_type), 0U,
                                         head.align_c16 || head.cblock, &input_desc)) {
        throw std::invalid_argument(
            "detessdequant runtime config could not synthesize typed input tensor");
      }
      // Phase 3a (Option A++): the kernel scales storage.nbytes by
      // runtime.batch_size when sizing the segment input. head.input_transport_
      // size_bytes is the BATCHED transport bytes (the MPK never authors a
      // per-frame transport size when batch>1), so divide it by batch to keep
      // required == actual. For batch==1 this is a no-op.
      const std::uint64_t per_frame_transport_size =
          head_batch_size > 0
              ? head.input_transport_size_bytes / static_cast<std::uint64_t>(head_batch_size)
              : head.input_transport_size_bytes;
      input_desc.storage.nbytes = per_frame_transport_size;
      runtime.input_tensors.push_back(input_desc);
    }
    {
      std::vector<int> slice_shape_int(head.slice_shape.begin(), head.slice_shape.end());
      runtime.slice_shapes.push_back(slice_shape_int);
    }
    runtime.dq_scale_list.push_back(head.per_head_quant_params.scales.front());
    runtime.dq_zp_list.push_back(
        head.per_head_quant_params.zero_points.empty()
            ? 0
            : static_cast<int>(head.per_head_quant_params.zero_points.front()));
    const std::string output_dtype = normalize_dtype_token(head.output_dtype);
    std::vector<int> output_shape_int(head.frame_shape.begin(), head.frame_shape.end());
    runtime.output_shapes.push_back(output_shape_int);
    runtime.runtime_output_logical_shapes.push_back(output_shape_int);
    sima_ev_tensor_desc output_desc{};
    if (!build_tensor_dense_desc_local(output_shape_int, output_dtype, &output_desc)) {
      throw std::invalid_argument(
          "detessdequant runtime config could not synthesize typed output tensor");
    }
    runtime.output_tensors.push_back(output_desc);
    runtime.runtime_output_logical_index_list.push_back(static_cast<int>(i));
    runtime.runtime_output_output_slot_list.push_back(static_cast<int>(i));
    runtime.runtime_output_physical_index_list.push_back(static_cast<int>(i));
    runtime.runtime_output_dtype_list.push_back(output_dtype);
    runtime.runtime_output_transport_kind_list.push_back(ProcessCvuOutputTransportKind::Dense);
    runtime.runtime_output_semantic_kind_list.push_back(ProcessCvuOutputSemanticKind::Tensor);
    runtime.runtime_output_logical_layout_list.push_back(semantic_input_layout);
    if (detess_layout_debug_enabled_local()) {
      std::fprintf(stderr,
                   "[detess-layout-debug] where=subset.runtime.head index=%zu input_shape=%s "
                   "transport_shape=%s frame_shape=%s input_dims={layout=%s,w=%d,h=%d,d=%d,c=%d} "
                   "transport_dims={layout=%s,w=%d,h=%d,d=%d,c=%d}\n",
                   i, join_i64_debug_local(head.input_transport_shape).c_str(),
                   join_i64_debug_local(head.input_transport_shape).c_str(),
                   join_i64_debug_local(head.frame_shape).c_str(),
                   transport_input_dims.layout.c_str(), transport_input_dims.width,
                   transport_input_dims.height, transport_input_dims.depth,
                   transport_input_dims.channels, transport_input_dims.layout.c_str(),
                   transport_input_dims.width, transport_input_dims.height,
                   transport_input_dims.depth, transport_input_dims.channels);
    }
  }

  const auto first_input_dims =
      dims_from_detess_shape(subset.heads.front().frame_shape, "detessdequant input");
  const auto first_output_dims =
      dims_from_detess_shape(subset.heads.front().frame_shape, "detessdequant output");
  runtime.input_dtype = normalize_dtype_token(subset.heads.front().frame_type);
  runtime.output_dtype = normalize_dtype_token(subset.heads.front().output_dtype);
  runtime.out_dtype = runtime.output_dtype;
  if (detess_layout_debug_enabled_local()) {
    std::fprintf(
        stderr,
        "[detess-layout-debug] where=subset.runtime.final heads=%zu input_layout=%s "
        "output_layout=%s input_shape0=%s output_shape0=%s "
        "first_output_dims={w=%d,h=%d,d=%d,c=%d,layout=%s}\n",
        subset.heads.size(), first_input_dims.layout.c_str(), first_output_dims.layout.c_str(),
        subset.heads.empty()
            ? "<none>"
            : join_i64_debug_local(subset.heads.front().input_transport_shape).c_str(),
        subset.heads.empty() ? "<none>"
                             : join_i64_debug_local(subset.heads.front().frame_shape).c_str(),
        first_output_dims.width, first_output_dims.height, first_output_dims.depth,
        first_output_dims.channels, first_output_dims.layout.c_str());
  }
  return runtime;
}

} // namespace simaai::neat::pipeline_internal::sima::plugin_contracts
