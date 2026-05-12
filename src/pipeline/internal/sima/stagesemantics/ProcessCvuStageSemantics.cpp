#include "pipeline/internal/sima/stagesemantics/ProcessCvuStageSemantics.h"
#include "pipeline/internal/sima/stagesemantics/ProcessCvuRuntimeConfigAdapterInternal.h"

#include "model/internal/ModelPack.h"

#include "nodes/sima/Preproc.h"
#include "pipeline/internal/packedio/PackedIoAdapter.h"
#include "pipeline/internal/sima/CompiledProcessCvuContractQuery.h"
#include "pipeline/internal/sima/PluginContractSubsets.h"
#include "pipeline/internal/sima/ProcessCvuFamily.h"
#include "pipeline/internal/sima/StaticSpecBuilders.h"
#include "pipeline/internal/sima/TensorSemanticsUtil.h"
#include "pipeline/internal/TensorMath.h"
#include "pipeline/internal/InputPolicy.h"
#include "pipeline/internal/EnvUtil.h"

#include <algorithm>
#include <cinttypes>
#include <cctype>
#include <cmath>
#include <cstring>
#include <cstdio>
#include <cstdlib>
#include <limits>
#include <queue>
#include <sstream>
#include <stdexcept>

namespace simaai::neat::pipeline_internal::sima::stagesemantics {

std::vector<TensorStaticSpec>
synthesize_preproc_runtime_outputs(const ProcessCvuStagePayload& payload);
std::uint64_t synthesize_preproc_packed_output_size_bytes(const ProcessCvuStagePayload& payload,
                                                          const std::string& dtype);
void canonicalize_preproc_single_handoff_payload(ProcessCvuStagePayload* payload);

namespace {

// Helper to extract named dims from an opaque shape vector (HWC or DHWC ordering).
struct ShapeDims {
  int width = 0, height = 0, depth = 0, channels = 0;
};

constexpr std::uint32_t kDetesscastOptRowStripeEnable = 1U << 1U;
constexpr std::uint32_t kDetesscastOptRowStripeBf16TensorC1 = 1U << 2U;
constexpr std::uint32_t kDetesscastOptRowStripeBf16TensorC2 = 1U << 3U;
constexpr std::uint32_t kDetesscastOptRowStripeBf16ImageC3 = 1U << 4U;
constexpr std::uint32_t kDetesscastOptRowStripeBf16TensorC4 = 1U << 5U;
constexpr std::uint32_t kDetesscastOptBf16NoncompactC16LaneSplit = 1U << 6U;
constexpr std::uint32_t kDetesscastDefaultOptimizedFlags =
    kDetesscastOptRowStripeEnable | kDetesscastOptRowStripeBf16TensorC1 |
    kDetesscastOptRowStripeBf16TensorC2 | kDetesscastOptRowStripeBf16ImageC3 |
    kDetesscastOptRowStripeBf16TensorC4;

ShapeDims dims_from_shape_vec(const std::vector<int>& s) {
  ShapeDims r;
  if (s.size() == 4) {
    r.depth = s[0];
    r.height = s[1];
    r.width = s[2];
    r.channels = s[3];
  } else if (s.size() == 3) {
    r.height = s[0];
    r.width = s[1];
    r.channels = s[2];
    r.depth = r.channels;
  } else if (s.size() == 2) {
    r.height = s[0];
    r.width = s[1];
    r.depth = 1;
    r.channels = 1;
  }
  return r;
}

bool canonical_slice_dhwc_from_shape_local(const std::vector<std::int64_t>& shape, int* out_d,
                                           int* out_h, int* out_w, int* out_c) {
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

std::vector<std::int64_t> semantic_shape_without_batch_local(std::vector<std::int64_t> shape) {
  const bool looks_like_explicit_batch =
      !shape.empty() && shape.front() == 1 && (shape.size() == 2U || shape.size() >= 4U);
  if (looks_like_explicit_batch) {
    shape.erase(shape.begin());
  }
  return shape;
}

std::vector<std::int64_t> graph_semantic_shape_local(std::vector<std::int64_t> shape) {
  if (shape.size() > 3U) {
    shape.erase(shape.begin());
  }
  return shape;
}

std::string layout_from_graph_shape_local(std::vector<std::int64_t> shape) {
  shape = graph_semantic_shape_local(std::move(shape));
  if (shape.size() >= 3U) {
    return "HWC";
  }
  if (!shape.empty()) {
    return "HW";
  }
  return {};
}

ShapeDims detess_dims_from_shape_local(const std::vector<std::int64_t>& shape,
                                       const std::string& context) {
  const auto semantic_shape = semantic_shape_without_batch_local(shape);
  int depth = 0;
  int height = 0;
  int width = 0;
  int channels = 0;
  if (!canonical_slice_dhwc_from_shape_local(semantic_shape, &depth, &height, &width, &channels)) {
    throw std::invalid_argument("processcvu detess geometry requires canonical tensor shape for '" +
                                context + "'");
  }
  ShapeDims dims;
  dims.width = width;
  dims.height = height;
  dims.depth = std::max(depth, 1);
  dims.channels = std::max(channels, 1);
  return dims;
}

// Helper to extract named dims from MpkPluginIoContract slice_shape (int64).
struct SliceDims {
  int d = 1, h = 0, w = 0, c = 0;
};
SliceDims dims_from_slice_shape(const std::vector<std::int64_t>& s) {
  SliceDims r;
  if (s.size() == 4) {
    r.d = static_cast<int>(s[0]);
    r.h = static_cast<int>(s[1]);
    r.w = static_cast<int>(s[2]);
    r.c = static_cast<int>(s[3]);
  } else if (s.size() == 3) {
    r.d = 1;
    r.h = static_cast<int>(s[0]);
    r.w = static_cast<int>(s[1]);
    r.c = static_cast<int>(s[2]);
  } else if (s.size() == 2) {
    r.d = 1;
    r.h = static_cast<int>(s[0]);
    r.w = static_cast<int>(s[1]);
    r.c = 1;
  }
  return r;
}

// Extract a single int from shape by index from end (0 = last).
int shape_dim_from_end(const std::vector<int>& shape, std::size_t from_end, int fallback = 0) {
  if (shape.size() > from_end) {
    return shape[shape.size() - 1 - from_end];
  }
  return fallback;
}

// Helper to get width/height/depth from an indexed or first input_shapes entry on the payload.
struct PayloadInputDims {
  int width = 0, height = 0, depth = 0, channels = 0;
};

std::vector<int> shape_vec_from_tensor_desc_local(const sima_ev_tensor_desc& desc) {
  std::vector<int> shape;
  const auto rank =
      std::min<std::uint32_t>(desc.shape.rank, static_cast<std::uint32_t>(SIMA_EV_MAX_RANK));
  shape.reserve(rank);
  for (std::uint32_t i = 0; i < rank; ++i) {
    shape.push_back(static_cast<int>(desc.shape.sizes[i]));
  }
  return shape;
}

std::vector<int> tile_shape_vec_from_tensor_desc_local(const sima_ev_tensor_desc& desc) {
  if (desc.layout_kind != SIMA_EV_LAYOUT_TILED) {
    return {};
  }
  std::vector<int> shape;
  const auto rank =
      std::min<std::uint32_t>(desc.shape.rank, static_cast<std::uint32_t>(SIMA_EV_MAX_RANK));
  shape.reserve(rank);
  for (std::uint32_t i = 0; i < rank; ++i) {
    shape.push_back(static_cast<int>(desc.layout.tiled.tile_sizes[i]));
  }
  return shape;
}

PayloadInputDims dims_from_shape_maybe_local(const std::vector<int>& shape) {
  const auto dims = dims_from_shape_vec(shape);
  return {
      .width = dims.width, .height = dims.height, .depth = dims.depth, .channels = dims.channels};
}

PayloadInputDims payload_input_dims_at(const ProcessCvuStagePayload& payload, std::size_t index) {
  if (index < payload.input_tensors.size()) {
    return dims_from_shape_maybe_local(
        shape_vec_from_tensor_desc_local(payload.input_tensors[index]));
  }
  if (!payload.input_tensors.empty()) {
    return dims_from_shape_maybe_local(
        shape_vec_from_tensor_desc_local(payload.input_tensors.front()));
  }
  if (index < payload.input_shapes.size()) {
    return dims_from_shape_maybe_local(payload.input_shapes[index]);
  }
  if (!payload.input_shapes.empty()) {
    return dims_from_shape_maybe_local(payload.input_shapes.front());
  }
  return {};
}

PayloadInputDims payload_output_dims_at(const ProcessCvuStagePayload& payload, std::size_t index) {
  if (index < payload.output_tensors.size()) {
    return dims_from_shape_maybe_local(
        shape_vec_from_tensor_desc_local(payload.output_tensors[index]));
  }
  if (!payload.output_tensors.empty()) {
    return dims_from_shape_maybe_local(
        shape_vec_from_tensor_desc_local(payload.output_tensors.front()));
  }
  if (index < payload.output_shapes.size()) {
    return dims_from_shape_maybe_local(payload.output_shapes[index]);
  }
  if (!payload.output_shapes.empty()) {
    return dims_from_shape_maybe_local(payload.output_shapes.front());
  }
  return {};
}

std::string payload_input_layout_token_local(const ProcessCvuStagePayload& payload,
                                             std::size_t index = 0U) {
  return payload.typed_input_layout_token(index);
}

std::string payload_output_layout_token_local(const ProcessCvuStagePayload& payload,
                                              std::size_t index = 0U) {
  return payload.typed_output_layout_token(index);
}

std::string payload_runtime_output_layout_token_local(const ProcessCvuStagePayload& payload,
                                                      std::size_t index = 0U) {
  return payload.logical_output_layout_token(index);
}

PayloadInputDims payload_slice_dims_at(const ProcessCvuStagePayload& payload, std::size_t index) {
  if (index < payload.slice_shapes.size()) {
    return dims_from_shape_maybe_local(payload.slice_shapes[index]);
  }
  if (!payload.slice_shapes.empty()) {
    return dims_from_shape_maybe_local(payload.slice_shapes.front());
  }
  if (index < payload.output_tensors.size()) {
    return dims_from_shape_maybe_local(
        tile_shape_vec_from_tensor_desc_local(payload.output_tensors[index]));
  }
  if (!payload.output_tensors.empty()) {
    return dims_from_shape_maybe_local(
        tile_shape_vec_from_tensor_desc_local(payload.output_tensors.front()));
  }
  if (index < payload.input_tensors.size()) {
    return dims_from_shape_maybe_local(
        tile_shape_vec_from_tensor_desc_local(payload.input_tensors[index]));
  }
  if (!payload.input_tensors.empty()) {
    return dims_from_shape_maybe_local(
        tile_shape_vec_from_tensor_desc_local(payload.input_tensors.front()));
  }
  return {};
}

// Helper to extract CompiledProcessCvuRuntimeConfig shape dims.
struct RuntimeInputDims {
  int width = 0, height = 0, depth = 0, channels = 0;
};

RuntimeInputDims runtime_input_dims_at(const CompiledProcessCvuRuntimeConfig& cfg,
                                       std::size_t index) {
  if (index < cfg.input_tensors.size()) {
    const auto dims =
        dims_from_shape_maybe_local(shape_vec_from_tensor_desc_local(cfg.input_tensors[index]));
    return {
        .width = dims.width, .height = dims.height, .depth = dims.depth, .channels = dims.channels};
  }
  if (!cfg.input_tensors.empty()) {
    const auto dims =
        dims_from_shape_maybe_local(shape_vec_from_tensor_desc_local(cfg.input_tensors.front()));
    return {
        .width = dims.width, .height = dims.height, .depth = dims.depth, .channels = dims.channels};
  }
  if (index < cfg.input_shapes.size()) {
    const auto dims = dims_from_shape_maybe_local(cfg.input_shapes[index]);
    return {
        .width = dims.width, .height = dims.height, .depth = dims.depth, .channels = dims.channels};
  }
  if (!cfg.input_shapes.empty()) {
    const auto dims = dims_from_shape_maybe_local(cfg.input_shapes.front());
    return {
        .width = dims.width, .height = dims.height, .depth = dims.depth, .channels = dims.channels};
  }
  return {};
}

RuntimeInputDims runtime_output_dims_at(const CompiledProcessCvuRuntimeConfig& cfg,
                                        std::size_t index) {
  if (index < cfg.output_tensors.size()) {
    const auto dims =
        dims_from_shape_maybe_local(shape_vec_from_tensor_desc_local(cfg.output_tensors[index]));
    return {
        .width = dims.width, .height = dims.height, .depth = dims.depth, .channels = dims.channels};
  }
  if (!cfg.output_tensors.empty()) {
    const auto dims =
        dims_from_shape_maybe_local(shape_vec_from_tensor_desc_local(cfg.output_tensors.front()));
    return {
        .width = dims.width, .height = dims.height, .depth = dims.depth, .channels = dims.channels};
  }
  if (index < cfg.output_shapes.size()) {
    const auto dims = dims_from_shape_maybe_local(cfg.output_shapes[index]);
    return {
        .width = dims.width, .height = dims.height, .depth = dims.depth, .channels = dims.channels};
  }
  if (!cfg.output_shapes.empty()) {
    const auto dims = dims_from_shape_maybe_local(cfg.output_shapes.front());
    return {
        .width = dims.width, .height = dims.height, .depth = dims.depth, .channels = dims.channels};
  }
  return {};
}

RuntimeInputDims runtime_slice_dims_at(const CompiledProcessCvuRuntimeConfig& cfg,
                                       std::size_t index) {
  if (index < cfg.slice_shapes.size()) {
    const auto dims = dims_from_shape_maybe_local(cfg.slice_shapes[index]);
    return {
        .width = dims.width, .height = dims.height, .depth = dims.depth, .channels = dims.channels};
  }
  if (!cfg.slice_shapes.empty()) {
    const auto dims = dims_from_shape_maybe_local(cfg.slice_shapes.front());
    return {
        .width = dims.width, .height = dims.height, .depth = dims.depth, .channels = dims.channels};
  }
  if (index < cfg.output_tensors.size()) {
    const auto dims = dims_from_shape_maybe_local(
        tile_shape_vec_from_tensor_desc_local(cfg.output_tensors[index]));
    return {
        .width = dims.width, .height = dims.height, .depth = dims.depth, .channels = dims.channels};
  }
  if (!cfg.output_tensors.empty()) {
    const auto dims = dims_from_shape_maybe_local(
        tile_shape_vec_from_tensor_desc_local(cfg.output_tensors.front()));
    return {
        .width = dims.width, .height = dims.height, .depth = dims.depth, .channels = dims.channels};
  }
  if (index < cfg.input_tensors.size()) {
    const auto dims = dims_from_shape_maybe_local(
        tile_shape_vec_from_tensor_desc_local(cfg.input_tensors[index]));
    return {
        .width = dims.width, .height = dims.height, .depth = dims.depth, .channels = dims.channels};
  }
  if (!cfg.input_tensors.empty()) {
    const auto dims = dims_from_shape_maybe_local(
        tile_shape_vec_from_tensor_desc_local(cfg.input_tensors.front()));
    return {
        .width = dims.width, .height = dims.height, .depth = dims.depth, .channels = dims.channels};
  }
  return {};
}

std::uint32_t resolve_tile_align_bytes_processcvu_local(int byte_align) {
  if (byte_align <= 0) {
    return 0U;
  }
  return byte_align == 1 ? 16U : static_cast<std::uint32_t>(byte_align);
}

bool build_dense_desc_processcvu_local(const std::vector<int>& shape, const std::string& dtype,
                                       const std::string& layout, sima_ev_tensor_desc* out) {
  std::string error_detail;
  const std::string normalized_layout = tensorsemantics::normalize_layout_token(layout);
  if (!layout.empty() && normalized_layout.empty()) {
    return false;
  }
  if (normalized_layout.empty()) {
    return tensorsemantics::build_generic_dense_tensor_desc(
        shape, dtype, out, &error_detail, "stage_dense_tensor_desc_output_missing",
        "stage_shape_rank_invalid", "stage_shape_dim_invalid", "stage_dtype_invalid",
        "stage_dense_stride_output_missing");
  }
  return tensorsemantics::build_dense_tensor_desc(
      shape, dtype, normalized_layout, out, &error_detail, "stage_dense_tensor_desc_output_missing",
      "stage_shape_rank_invalid", "stage_shape_dim_invalid", "stage_dtype_invalid",
      "stage_dense_stride_output_missing");
}

bool build_tiled_desc_processcvu_local(const std::vector<int>& shape,
                                       const std::vector<int>& tile_shape, const std::string& dtype,
                                       const std::string& layout, std::uint32_t tile_align_bytes,
                                       sima_ev_tensor_desc* out) {
  std::string error_detail;
  return tensorsemantics::build_tiled_tensor_desc(
      shape, tile_shape, dtype, layout, tile_align_bytes, out, &error_detail,
      "stage_tiled_tensor_desc_output_missing", "stage_shape_rank_invalid",
      "stage_shape_dim_invalid", "stage_dtype_invalid",
      "stage_tiled_tensor_desc_shape_rank_mismatch", "stage_tiled_tensor_desc_tile_dim_invalid");
}

void apply_tensor_axes_processcvu_local(sima_ev_tensor_desc* desc) {
  if (!desc) {
    return;
  }
  const std::uint32_t rank =
      std::min<std::uint32_t>(desc->shape.rank, static_cast<std::uint32_t>(SIMA_EV_MAX_RANK));
  for (std::uint32_t i = 0; i < SIMA_EV_MAX_RANK; ++i) {
    desc->shape.axis_semantics[i] = SIMA_EV_AXIS_UNKNOWN;
  }
  if (rank == 0U) {
    return;
  }
  if (rank == 1U) {
    desc->shape.axis_semantics[0] = SIMA_EV_AXIS_W;
  } else if (rank == 2U) {
    desc->shape.axis_semantics[0] = SIMA_EV_AXIS_H;
    desc->shape.axis_semantics[1] = SIMA_EV_AXIS_W;
  } else {
    desc->shape.axis_semantics[rank - 1U] = SIMA_EV_AXIS_C;
    desc->shape.axis_semantics[rank - 2U] = SIMA_EV_AXIS_W;
    desc->shape.axis_semantics[rank - 3U] = SIMA_EV_AXIS_H;
    if (rank >= 5U) {
      desc->shape.axis_semantics[rank - 4U] = SIMA_EV_AXIS_D;
      desc->shape.axis_semantics[0] = SIMA_EV_AXIS_N;
    }
  }
  if (desc->layout_kind == SIMA_EV_LAYOUT_TILED &&
      tensorsemantics::find_shape_axis(desc->shape, SIMA_EV_AXIS_C) >= 0) {
    desc->layout.tiled.flags |= SIMA_EV_TILED_FLAG_COMPACT_CHANNELS;
  }
}

void apply_tiled_channel_storage_policy_processcvu_local(sima_ev_tensor_desc* desc,
                                                         bool c16_packed) {
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

bool build_tensor_tiled_desc_processcvu_local(const std::vector<int>& shape,
                                              const std::vector<int>& tile_shape,
                                              const std::string& dtype,
                                              std::uint32_t tile_align_bytes, bool c16_packed,
                                              sima_ev_tensor_desc* out) {
  if (!build_tiled_desc_processcvu_local(shape, tile_shape, dtype, "", tile_align_bytes, out)) {
    return false;
  }
  apply_tensor_axes_processcvu_local(out);
  apply_tiled_channel_storage_policy_processcvu_local(out, c16_packed);
  return true;
}

bool build_tensor_tiled_desc_processcvu_local(const std::vector<int>& shape,
                                              const std::vector<int>& tile_shape,
                                              const std::string& dtype,
                                              std::uint32_t tile_align_bytes,
                                              sima_ev_tensor_desc* out) {
  return build_tensor_tiled_desc_processcvu_local(shape, tile_shape, dtype, tile_align_bytes, false,
                                                  out);
}

bool build_tensor_dense_desc_processcvu_local(const std::vector<int>& shape,
                                              const std::string& dtype, sima_ev_tensor_desc* out) {
  if (!build_dense_desc_processcvu_local(shape, dtype, "", out)) {
    return false;
  }
  apply_tensor_axes_processcvu_local(out);
  return true;
}

std::vector<int>
tensor_desc_tile_shape_from_slice_shape_processcvu_local(const std::vector<int>& tensor_shape,
                                                         const std::vector<int>& slice_shape) {
  if (tensor_shape.empty() || slice_shape.empty() || slice_shape.size() == tensor_shape.size()) {
    return slice_shape;
  }
  if (slice_shape.size() > tensor_shape.size()) {
    return slice_shape;
  }
  std::vector<int> out;
  out.reserve(tensor_shape.size());
  const std::size_t missing = tensor_shape.size() - slice_shape.size();
  std::size_t slice_index = 0U;
  for (std::size_t axis = 0U; axis < tensor_shape.size(); ++axis) {
    if (axis < missing) {
      out.push_back(tensor_shape[axis]);
    } else {
      out.push_back(slice_shape[slice_index++]);
    }
  }
  return out;
}

std::string runtime_input_layout_token_local(const CompiledProcessCvuRuntimeConfig& cfg,
                                             std::size_t index = 0U) {
  if (index < cfg.input_tensors.size()) {
    return tensorsemantics::layout_token_from_ev_tensor_desc(cfg.input_tensors[index]);
  }
  if (!cfg.input_tensors.empty()) {
    return tensorsemantics::layout_token_from_ev_tensor_desc(cfg.input_tensors.front());
  }
  return {};
}

std::string runtime_output_layout_token_local(const CompiledProcessCvuRuntimeConfig& cfg,
                                              std::size_t index = 0U) {
  if (index < cfg.output_tensors.size()) {
    return tensorsemantics::layout_token_from_ev_tensor_desc(cfg.output_tensors[index]);
  }
  if (!cfg.output_tensors.empty()) {
    return tensorsemantics::layout_token_from_ev_tensor_desc(cfg.output_tensors.front());
  }
  if (index < cfg.runtime_output_logical_layout_list.size()) {
    return tensorsemantics::normalize_layout_token(cfg.runtime_output_logical_layout_list[index]);
  }
  if (!cfg.runtime_output_logical_layout_list.empty()) {
    return tensorsemantics::normalize_layout_token(cfg.runtime_output_logical_layout_list.front());
  }
  return {};
}

struct ProcessCvuSingleOutputFactsSpec {
  std::string physical_input_name;
  std::string source_segment_name;
  std::string physical_output_name;
  std::string logical_output_name;
  std::vector<std::string> published_output_names;
  std::string primary_output_name;
  std::vector<std::int64_t> input_shape;
  std::string input_dtype;
  std::string input_layout;
  ProcessCvuOutputRepresentation output_representation =
      ProcessCvuOutputRepresentation::DenseTensor;
  std::vector<std::int64_t> output_shape;
  std::string output_dtype;
  std::string output_layout;
  const MpkTensorContract* packed_output_tensor = nullptr;
  std::uint64_t packed_output_size_bytes = 0;
};

struct ProcessCvuSingleOutputIdentity {
  std::string runtime_output_name;
  std::string physical_output_name;
  std::string published_output_name;
};

struct ProcessCvuPackedRouteEntry {
  int logical_index = -1;
  int output_slot = -1;
  int physical_index = -1;
  const MpkTensorContract* input_tensor = nullptr;
  std::string input_dtype;
  std::string input_layout;
  std::int64_t input_byte_offset = 0;
  const MpkTensorContract* output_tensor = nullptr;
  std::string output_physical_name;
  std::string output_logical_name;
  std::string output_dtype;
  std::string output_layout;
  std::int64_t output_byte_offset = 0;
  int src_output_slot = -1;
  int src_physical_output_index = -1;
};

static ProcessCvuCanonicalCompileInputs build_processcvu_mpk_preproc_compile_inputs_local(
    const MpkContract& contract, const std::string& input_format, int input_depth,
    int max_input_width, int max_input_height, bool normalize, const std::vector<float>& mean,
    const std::vector<float>& stddev, bool single_output_handoff);
static ProcessCvuCanonicalCompileInputs build_processcvu_mpk_preadapter_compile_inputs_local(
    const MpkContract& contract, const std::string& graph_family,
    const std::optional<std::string>& exact_stage_name_or_id,
    const std::optional<std::string>& canonical_handoff_segment_name);
static ProcessCvuCanonicalCompileInputs
build_processcvu_mpk_detess_compile_inputs_local(const MpkContract& contract);
static ProcessCvuCanonicalCompileInputs
build_processcvu_mpk_detesscast_compile_inputs_local(const MpkContract& contract);
static ProcessCvuCanonicalCompileInputs
build_processcvu_mpk_cast_compile_inputs_local(const MpkContract& contract);
static ProcessCvuCanonicalCompileInputs
build_processcvu_mpk_dequant_compile_inputs_local(const MpkContract& contract);
static ProcessCvuCanonicalCompileInputs
build_processcvu_mpk_detessdequant_compile_inputs_local(const MpkContract& contract);
std::vector<const MpkPluginIoContract*>
collect_post_stages_for_kind_names_local(const MpkContract& contract,
                                         std::initializer_list<const char*> preferred);

namespace specbuilders = pipeline_internal::sima::specbuilders;

std::string lower_copy(std::string value) {
  std::transform(value.begin(), value.end(), value.begin(),
                 [](unsigned char c) { return static_cast<char>(std::tolower(c)); });
  return value;
}

const MpkGraphNode*
find_graph_node_by_name_for_runtime_local(const std::vector<MpkGraphNode>& nodes,
                                          const std::string& name) {
  if (name.empty()) {
    return nullptr;
  }
  for (const auto& node : nodes) {
    if (node.name == name || node.node_id == name || node.plugin_id == name) {
      return &node;
    }
  }
  return nullptr;
}

std::vector<const MpkGraphNode*>
find_graph_raw_member_nodes_for_runtime_local(const MpkContract& contract,
                                              const MpkGraphNode& node) {
  std::vector<const MpkGraphNode*> members;
  members.reserve(node.member_node_ids.size());
  for (const auto& member_name : node.member_node_ids) {
    if (const auto* member =
            find_graph_node_by_name_for_runtime_local(contract.graph.raw_nodes, member_name)) {
      members.push_back(member);
    }
  }
  std::sort(members.begin(), members.end(), [](const MpkGraphNode* lhs, const MpkGraphNode* rhs) {
    if (lhs->sequence != rhs->sequence) {
      return lhs->sequence < rhs->sequence;
    }
    return lhs->name < rhs->name;
  });
  return members;
}

bool graph_member_runs_in_processcvu_plugin_for_runtime_local(const MpkGraphNode& node) {
  (void)node;
  return true;
}

bool processcvu_graph_member_matches_family_for_runtime_local(const MpkGraphNode& node,
                                                              const std::string& canonical_family) {
  const std::string op = lower_copy(node.canonical_op);
  if (canonical_family == "quantize") {
    return op == "quantize" || op == "quant";
  }
  if (canonical_family == "tessellate") {
    return op == "tessellate" || op == "tess";
  }
  if (canonical_family == "dequantize") {
    return op == "dequantize" || op == "dequant";
  }
  if (canonical_family == "detessellate") {
    return op == "detessellate" || op == "detess";
  }
  return op == canonical_family;
}

const MpkGraphNode* select_processcvu_graph_kernel_contract_node_for_runtime_local(
    const MpkContract& contract, const std::string& graph_family,
    const std::optional<std::string>& exact_stage_name_or_id) {
  if (exact_stage_name_or_id.has_value() && !exact_stage_name_or_id->empty()) {
    if (const auto* raw = find_graph_node_by_name_for_runtime_local(contract.graph.raw_nodes,
                                                                    *exact_stage_name_or_id)) {
      return raw;
    }
  }

  const std::string canonical_family = lower_copy(canonical_processcvu_graph_family(graph_family));

  if (canonical_family == "quantize" || canonical_family == "tessellate" ||
      canonical_family == "preproc") {
    for (const auto& node : contract.graph.nodes) {
      if (node.kind != MpkGraphNodeKind::FusedPreproc) {
        continue;
      }
      const auto members = find_graph_raw_member_nodes_for_runtime_local(contract, node);
      for (const auto* member : members) {
        if (!member || !graph_member_runs_in_processcvu_plugin_for_runtime_local(*member)) {
          continue;
        }
        if (processcvu_graph_member_matches_family_for_runtime_local(*member, canonical_family)) {
          return member;
        }
      }
    }
  }

  if (canonical_family == "quanttess") {
    for (const auto& node : contract.graph.nodes) {
      if (node.kind == MpkGraphNodeKind::FusedQuantTess) {
        return &node;
      }
    }
  }

  if (canonical_family == "detessdequant") {
    for (const auto& node : contract.graph.nodes) {
      if (node.kind == MpkGraphNodeKind::FusedDetessDequant) {
        return &node;
      }
    }
  }

  for (const auto& node : contract.graph.raw_nodes) {
    if (!graph_member_runs_in_processcvu_plugin_for_runtime_local(node)) {
      continue;
    }
    if (processcvu_graph_member_matches_family_for_runtime_local(node, canonical_family)) {
      return &node;
    }
  }
  return nullptr;
}

bool overlay_processcvu_runtime_geometry_from_graph_local(const MpkGraphNode& node,
                                                          CompiledProcessCvuRuntimeConfig* cfg,
                                                          std::string* error_message) {
  (void)node;
  if (!cfg) {
    if (error_message) {
      *error_message = "processcvu graph runtime geometry overlay requires config storage";
    }
    return false;
  }

  const auto populate_shapes_from_descs = [](const std::vector<sima_ev_tensor_desc>& descs,
                                             std::vector<std::vector<int>>* out_shapes) {
    if (!out_shapes || !out_shapes->empty()) {
      return;
    }
    out_shapes->reserve(descs.size());
    for (const auto& desc : descs) {
      const auto shape = shape_vec_from_tensor_desc_local(desc);
      if (!shape.empty()) {
        out_shapes->push_back(shape);
      }
    }
  };
  const auto populate_tile_shapes_from_descs = [](const std::vector<sima_ev_tensor_desc>& descs,
                                                  std::vector<std::vector<int>>* out_shapes) {
    if (!out_shapes || !out_shapes->empty()) {
      return;
    }
    for (const auto& desc : descs) {
      const auto shape = tile_shape_vec_from_tensor_desc_local(desc);
      if (!shape.empty()) {
        out_shapes->push_back(shape);
      }
    }
  };

  populate_shapes_from_descs(cfg->input_tensors, &cfg->input_shapes);
  populate_shapes_from_descs(cfg->output_tensors, &cfg->output_shapes);
  populate_tile_shapes_from_descs(cfg->output_tensors, &cfg->slice_shapes);
  populate_tile_shapes_from_descs(cfg->input_tensors, &cfg->slice_shapes);

  if (cfg->runtime_output_logical_shapes.empty() && !cfg->output_shapes.empty()) {
    cfg->runtime_output_logical_shapes = cfg->output_shapes;
  }

  if (cfg->input_shapes.empty()) {
    if (error_message) {
      *error_message = "processcvu runtime config missing semantic input shapes";
    }
    return false;
  }
  if (cfg->output_shapes.empty()) {
    if (error_message) {
      *error_message = "processcvu runtime config missing semantic output shapes";
    }
    return false;
  }
  return true;
}

void require_graph_processcvu_runtime_geometry_local(
    const MpkContract& contract, const std::string& graph_family,
    const std::optional<std::string>& exact_stage_name_or_id,
    CompiledProcessCvuRuntimeConfig* cfg) {
  std::string error_message;
  const auto* graph_node = select_processcvu_graph_kernel_contract_node_for_runtime_local(
      contract, graph_family, exact_stage_name_or_id);
  if (!graph_node) {
    throw std::runtime_error(
        "processcvu MPK route missing graph kernel contract node for family '" +
        canonical_processcvu_graph_family(graph_family) + "'");
  }
  if (!overlay_processcvu_runtime_geometry_from_graph_local(*graph_node, cfg, &error_message)) {
    throw std::runtime_error(error_message.empty()
                                 ? "processcvu MPK graph runtime geometry overlay failed"
                                 : error_message);
  }
}

bool processcvu_contract_compare_enabled() {
  static int cached = -1;
  if (cached >= 0) {
    return cached == 1;
  }
  const char* raw = std::getenv("SIMA_PROCESSCVU_CONTRACT_COMPARE");
  cached = (raw && *raw && std::strcmp(raw, "0") != 0) ? 1 : 0;
  return cached == 1;
}

bool processcvu_contract_compare_exit_enabled() {
  static int cached = -1;
  if (cached >= 0) {
    return cached == 1;
  }
  const char* raw = std::getenv("SIMA_PROCESSCVU_CONTRACT_COMPARE_EXIT_AFTER_DUMP");
  cached = (raw && *raw && std::strcmp(raw, "0") != 0) ? 1 : 0;
  return cached == 1;
}

bool processcvu_detess_layout_debug_enabled() {
  return pipeline_internal::env_bool("SIMA_DETESS_LAYOUT_DEBUG", false);
}

bool processcvu_tess_segment_debug_enabled() {
  return pipeline_internal::env_bool("SIMA_TESS_SEGMENT_DEBUG", false);
}

std::string processcvu_contract_compare_family_filter() {
  static bool initialized = false;
  static std::string cached;
  if (!initialized) {
    initialized = true;
    if (const char* raw = std::getenv("SIMA_PROCESSCVU_CONTRACT_COMPARE_GRAPH_FAMILY");
        raw && *raw) {
      cached = lower_copy(raw);
    }
  }
  return cached;
}

std::string join_i64_debug_processcvu_local(const std::vector<std::int64_t>& values) {
  std::ostringstream oss;
  oss << "[";
  for (std::size_t i = 0; i < values.size(); ++i) {
    if (i > 0U) {
      oss << "x";
    }
    oss << values[i];
  }
  oss << "]";
  return oss.str();
}

std::string canonical_family_name(std::string graph_family);
ProcessCvuGraphFamily family_enum_from_name(const std::string& graph_family);
TensorStaticSpec synthesize_single_io_output_tensor(const ProcessCvuStagePayload& payload);
void synthesize_runtime_output_arrays_from_payload(ProcessCvuStagePayload* payload);
ProcessCvuCanonicalFacts build_preproc_facts_from_payload(const ProcessCvuStagePayload& payload);
ProcessCvuCanonicalFacts
build_single_io_processcvu_facts_from_payload(const ProcessCvuStagePayload& payload);
ProcessCvuCanonicalFacts
build_multi_io_processcvu_facts_from_payload(const ProcessCvuStagePayload& payload,
                                             const std::vector<std::string>& runtime_input_names);
ProcessCvuCanonicalFacts
build_processcvu_facts_from_payload_local(const ProcessCvuStagePayload& payload);
std::string preferred_tensor_dtype_local(const MpkTensorContract& tensor,
                                         const std::string& fallback_dtype);
std::uint64_t preferred_mpk_tensor_size_bytes_local(const MpkTensorContract& tensor,
                                                    const std::string& fallback_dtype);
std::uint64_t logical_mpk_tensor_size_bytes_local(const MpkTensorContract& tensor,
                                                  const std::string& dtype);

std::string ints64_dbg_processcvu_local(const std::vector<std::int64_t>& values) {
  std::string out = "[";
  for (std::size_t i = 0; i < values.size(); ++i) {
    if (i > 0U) {
      out += ",";
    }
    out += std::to_string(values[i]);
  }
  out += "]";
  return out;
}

std::string ints_dbg_processcvu_local(const std::vector<int>& values) {
  std::string out = "[";
  for (std::size_t i = 0; i < values.size(); ++i) {
    if (i > 0U) {
      out += ",";
    }
    out += std::to_string(values[i]);
  }
  out += "]";
  return out;
}

std::string ints2d_dbg_processcvu_local(const std::vector<std::vector<int>>& values) {
  std::ostringstream out;
  out << "[";
  for (std::size_t i = 0; i < values.size(); ++i) {
    if (i > 0U) {
      out << ",";
    }
    out << ints_dbg_processcvu_local(values[i]);
  }
  out << "]";
  return out.str();
}

std::string doubles_dbg_processcvu_local(const std::vector<double>& values) {
  std::ostringstream out;
  out << "[";
  for (std::size_t i = 0; i < values.size(); ++i) {
    if (i > 0U) {
      out << ",";
    }
    out << values[i];
  }
  out << "]";
  return out.str();
}

std::string strings_dbg_processcvu_local(const std::vector<std::string>& values) {
  std::ostringstream out;
  out << "[";
  for (std::size_t i = 0; i < values.size(); ++i) {
    if (i > 0U) {
      out << ",";
    }
    out << "\"" << values[i] << "\"";
  }
  out << "]";
  return out.str();
}

std::string quant_dbg_processcvu_local(const std::optional<QuantStaticSpec>& quant) {
  if (!quant.has_value()) {
    return "<none>";
  }
  std::ostringstream out;
  out << "{axis=" << quant->axis << ",scales=" << doubles_dbg_processcvu_local(quant->scales)
      << ",zero_points=" << ints64_dbg_processcvu_local(quant->zero_points) << "}";
  return out.str();
}

const char* device_kind_dbg_processcvu_local(const DeviceKind kind) {
  switch (kind) {
  case DeviceKind::Unknown:
    return "Unknown";
  case DeviceKind::Cpu:
    return "Cpu";
  case DeviceKind::Mla:
    return "Mla";
  case DeviceKind::Evxx:
    return "Evxx";
  }
  return "Unknown";
}

std::string payload_dbg_processcvu_local(const ProcessCvuStagePayload& payload) {
  std::ostringstream out;
  out << "{graph_family=\"" << payload.graph_family << "\""
      << ",graph_name=\"" << payload.graph_name << "\""
      << ",graph_id=" << payload.graph_id
      << ",canonical_contract=" << (payload.canonical_contract ? 1 : 0)
      << ",slice_shape_raw=" << ints64_dbg_processcvu_local(payload.slice_shape_raw)
      << ",out_shape_raw=" << ints64_dbg_processcvu_local(payload.out_shape_raw)
      << ",has_align_c16=" << (payload.has_align_c16 ? 1 : 0)
      << ",align_c16=" << (payload.align_c16 ? 1 : 0)
      << ",has_cblock=" << (payload.has_cblock ? 1 : 0) << ",cblock=" << (payload.cblock ? 1 : 0)
      << ",default_input_name=\"" << payload.default_input_name << "\""
      << ",default_output_names=" << strings_dbg_processcvu_local(payload.default_output_names)
      << ",primary_output_name=\"" << payload.primary_output_name << "\""
      << ",primary_output_transport_kind="
      << static_cast<int>(payload.primary_output_transport_kind)
      << ",primary_output_semantic_kind=" << static_cast<int>(payload.primary_output_semantic_kind)
      << ",input_dtype=\"" << payload.input_dtype << "\""
      << ",output_dtype=\"" << payload.output_dtype << "\""
      << ",out_dtype=\"" << payload.out_dtype << "\""
      << ",input_layout=\"" << payload_input_layout_token_local(payload) << "\""
      << ",output_layout=\"" << payload_output_layout_token_local(payload) << "\""
      << ",scaled_width=" << payload.scaled_width << ",scaled_height=" << payload.scaled_height
      << ",input_stride=" << payload.input_stride << ",output_stride=" << payload.output_stride
      << ",input_offset=" << payload.input_offset << ",batch_size=" << payload.batch_size
      << ",round_off=" << payload.round_off << ",byte_align=" << payload.byte_align
      << ",num_in_tensor=" << payload.num_in_tensor
      << ",has_q_scale=" << (payload.has_q_scale ? 1 : 0) << ",q_scale=" << payload.q_scale
      << ",has_q_zp=" << (payload.has_q_zp ? 1 : 0) << ",q_zp=" << payload.q_zp
      << ",q_scale_list=" << doubles_dbg_processcvu_local(payload.q_scale_list)
      << ",q_zp_list=" << ints_dbg_processcvu_local(payload.q_zp_list)
      << ",dq_scale_list=" << doubles_dbg_processcvu_local(payload.dq_scale_list)
      << ",dq_zp_list=" << ints_dbg_processcvu_local(payload.dq_zp_list)
      << ",input_shapes=" << ints2d_dbg_processcvu_local(payload.input_shapes)
      << ",slice_shapes=" << ints2d_dbg_processcvu_local(payload.slice_shapes)
      << ",output_shapes=" << ints2d_dbg_processcvu_local(payload.output_shapes)
      << ",runtime_output_logical_index_list="
      << ints_dbg_processcvu_local(payload.runtime_output_logical_index_list)
      << ",runtime_output_output_slot_list="
      << ints_dbg_processcvu_local(payload.runtime_output_output_slot_list)
      << ",runtime_output_physical_index_list="
      << ints_dbg_processcvu_local(payload.runtime_output_physical_index_list)
      << ",runtime_output_dtype_list="
      << strings_dbg_processcvu_local(payload.runtime_output_dtype_list)
      << ",runtime_output_logical_layout_list="
      << strings_dbg_processcvu_local(payload.runtime_output_logical_layout_list)
      << ",runtime_output_logical_shapes="
      << ints2d_dbg_processcvu_local(payload.runtime_output_logical_shapes)
      << ",runtime_output_logical_layout_list="
      << strings_dbg_processcvu_local(payload.runtime_output_logical_layout_list) << "}";
  return out.str();
}

std::string input_fact_dbg_processcvu_local(const ProcessCvuCanonicalInputFact& fact) {
  std::ostringstream out;
  out << "{logical_index=" << fact.logical_index << ",physical_index=" << fact.physical_index
      << ",physical_name=\"" << fact.physical_name << "\""
      << ",logical_name=\"" << fact.logical_name << "\""
      << ",shape=" << ints64_dbg_processcvu_local(fact.shape) << ",size_bytes=" << fact.size_bytes
      << ",dtype=\"" << fact.dtype << "\""
      << ",layout=\"" << fact.layout << "\""
      << ",byte_offset=" << fact.byte_offset << ",quant=" << quant_dbg_processcvu_local(fact.quant)
      << "}";
  return out.str();
}

std::string binding_fact_dbg_processcvu_local(const ProcessCvuCanonicalBindingFact& fact) {
  std::ostringstream out;
  out << "{sink_pad_index=" << fact.sink_pad_index
      << ",local_logical_input_index=" << fact.local_logical_input_index
      << ",src_logical_output_index=" << fact.src_logical_output_index
      << ",src_output_slot=" << fact.src_output_slot
      << ",src_physical_output_index=" << fact.src_physical_output_index
      << ",src_physical_size_bytes=" << fact.src_physical_size_bytes
      << ",src_physical_byte_offset=" << fact.src_physical_byte_offset
      << ",required=" << (fact.required ? 1 : 0) << ",cm_input_name=\"" << fact.cm_input_name
      << "\""
      << ",source_segment_name=\"" << fact.source_segment_name << "\""
      << "}";
  return out.str();
}

std::string output_fact_dbg_processcvu_local(const ProcessCvuCanonicalOutputFact& fact) {
  std::ostringstream out;
  out << "{representation=" << static_cast<int>(fact.representation)
      << ",logical_index=" << fact.logical_index << ",physical_index=" << fact.physical_index
      << ",output_slot=" << fact.output_slot << ",tensor_index=" << fact.tensor_index
      << ",physical_name=\"" << fact.physical_name << "\""
      << ",logical_name=\"" << fact.logical_name << "\""
      << ",shape=" << ints64_dbg_processcvu_local(fact.shape) << ",dtype=\"" << fact.dtype << "\""
      << ",layout=\"" << fact.layout << "\""
      << ",byte_offset=" << fact.byte_offset << ",size_bytes=" << fact.size_bytes
      << ",quant=" << quant_dbg_processcvu_local(fact.quant) << "}";
  return out.str();
}

std::string route_fact_dbg_processcvu_local(const ProcessCvuCanonicalRouteFact& fact) {
  std::ostringstream out;
  out << "{output_slot=" << fact.output_slot
      << ",logical_output_index=" << fact.logical_output_index
      << ",tensor_index=" << fact.tensor_index << ",cm_output_name=\"" << fact.cm_output_name
      << "\""
      << ",segment_name=\"" << fact.segment_name << "\""
      << "}";
  return out.str();
}

std::string logical_input_dbg_processcvu_local(const LogicalInputStaticSpec& spec) {
  std::ostringstream out;
  out << "{logical_index=" << spec.logical_index
      << ",backend_input_index=" << spec.backend_input_index
      << ",physical_index=" << spec.physical_index
      << ",shape=" << ints64_dbg_processcvu_local(spec.shape)
      << ",stride_bytes=" << ints64_dbg_processcvu_local(spec.stride_bytes)
      << ",byte_offset=" << spec.byte_offset << ",size_bytes=" << spec.size_bytes << ",dtype=\""
      << spec.dtype << "\""
      << ",layout=\"" << spec.layout << "\""
      << ",logical_name=\"" << spec.logical_name << "\""
      << ",backend_name=\"" << spec.backend_name << "\""
      << ",segment_name=\"" << spec.segment_name << "\""
      << ",quant=" << quant_dbg_processcvu_local(spec.quant) << "}";
  return out.str();
}

std::string binding_dbg_processcvu_local(const InputBindingStaticSpec& spec) {
  std::ostringstream out;
  out << "{sink_pad_index=" << spec.sink_pad_index
      << ",local_logical_input_index=" << spec.local_logical_input_index
      << ",src_stage_index=" << spec.src_stage_index << ",src_stage_id=\"" << spec.src_stage_id
      << "\""
      << ",src_logical_output_index=" << spec.src_logical_output_index
      << ",src_output_slot=" << spec.src_output_slot
      << ",src_physical_output_index=" << spec.src_physical_output_index
      << ",src_physical_size_bytes=" << spec.src_physical_size_bytes
      << ",src_physical_byte_offset=" << spec.src_physical_byte_offset
      << ",required=" << (spec.required ? 1 : 0) << ",cm_input_name=\"" << spec.cm_input_name
      << "\""
      << ",source_segment_name=\"" << spec.source_segment_name << "\""
      << "}";
  return out.str();
}

std::string physical_dbg_processcvu_local(const PhysicalBufferStaticSpec& spec) {
  std::ostringstream out;
  out << "{physical_index=" << spec.physical_index << ",allocator_index=" << spec.allocator_index
      << ",source_physical_index=" << spec.source_physical_index
      << ",size_bytes=" << spec.size_bytes << ",source_byte_offset=" << spec.source_byte_offset
      << ",device_kind=" << device_kind_dbg_processcvu_local(spec.device_kind)
      << ",memory_flags=" << spec.memory_flags << ",segment_name=\"" << spec.segment_name << "\""
      << "}";
  return out.str();
}

std::string logical_output_dbg_processcvu_local(const LogicalTensorStaticSpec& spec) {
  std::ostringstream out;
  out << "{logical_index=" << spec.logical_index
      << ",backend_output_index=" << spec.backend_output_index
      << ",physical_index=" << spec.physical_index << ",output_slot=" << spec.output_slot
      << ",tensor_index=" << spec.tensor_index << ",byte_offset=" << spec.byte_offset
      << ",size_bytes=" << spec.size_bytes << ",shape=" << ints64_dbg_processcvu_local(spec.shape)
      << ",stride_bytes=" << ints64_dbg_processcvu_local(spec.stride_bytes) << ",dtype=\""
      << spec.dtype << "\""
      << ",layout=\"" << spec.layout << "\""
      << ",logical_name=\"" << spec.logical_name << "\""
      << ",backend_name=\"" << spec.backend_name << "\""
      << ",segment_name=\"" << spec.segment_name << "\""
      << ",quant=" << quant_dbg_processcvu_local(spec.quant) << "}";
  return out.str();
}

std::string route_dbg_processcvu_local(const StageOutputRoute& route) {
  std::ostringstream out;
  out << "{output_slot=" << route.output_slot
      << ",logical_output_index=" << route.logical_output_index
      << ",tensor_index=" << route.tensor_index << ",cm_output_name=\"" << route.cm_output_name
      << "\""
      << ",segment_name=\"" << route.segment_name << "\""
      << "}";
  return out.str();
}

bool processcvu_contract_compare_matches(const ProcessCvuStagePayload& payload) {
  if (!processcvu_contract_compare_enabled()) {
    return false;
  }
  const std::string filter = processcvu_contract_compare_family_filter();
  if (filter.empty()) {
    return true;
  }
  return canonical_family_name(payload.graph_family) == filter;
}

void dump_processcvu_contract_compare_local(const CompiledProcessCvuContract& compiled,
                                            const ProcessCvuCanonicalFacts& facts) {
  const std::string family = canonical_family_name(compiled.payload.graph_family);
  std::fprintf(stderr, "[processcvu-compare] family=%s payload=%s\n", family.c_str(),
               payload_dbg_processcvu_local(compiled.payload).c_str());
  std::fprintf(stderr,
               "[processcvu-compare] family=%s facts physical_input_names=%s "
               "physical_output_names=%s published_output_names=%s primary_output_name=\"%s\"\n",
               family.c_str(), strings_dbg_processcvu_local(facts.physical_input_names).c_str(),
               strings_dbg_processcvu_local(facts.physical_output_names).c_str(),
               strings_dbg_processcvu_local(facts.published_output_names).c_str(),
               facts.primary_output_name.c_str());
  for (std::size_t i = 0; i < facts.inputs.size(); ++i) {
    std::fprintf(stderr, "[processcvu-compare] family=%s facts.input index=%zu value=%s\n",
                 family.c_str(), i, input_fact_dbg_processcvu_local(facts.inputs[i]).c_str());
  }
  for (std::size_t i = 0; i < facts.input_bindings.size(); ++i) {
    std::fprintf(stderr, "[processcvu-compare] family=%s facts.binding index=%zu value=%s\n",
                 family.c_str(), i,
                 binding_fact_dbg_processcvu_local(facts.input_bindings[i]).c_str());
  }
  for (std::size_t i = 0; i < facts.outputs.size(); ++i) {
    std::fprintf(stderr, "[processcvu-compare] family=%s facts.output index=%zu value=%s\n",
                 family.c_str(), i, output_fact_dbg_processcvu_local(facts.outputs[i]).c_str());
  }
  for (std::size_t i = 0; i < facts.output_order.size(); ++i) {
    std::fprintf(stderr, "[processcvu-compare] family=%s facts.route index=%zu value=%s\n",
                 family.c_str(), i, route_fact_dbg_processcvu_local(facts.output_order[i]).c_str());
  }
  std::fprintf(
      stderr,
      "[processcvu-compare] family=%s runtime plugin_kind=\"%s\" "
      "required_preprocess_meta_fields=%s\n",
      family.c_str(), compiled.runtime_contract.plugin_kind.c_str(),
      strings_dbg_processcvu_local(compiled.runtime_contract.required_preprocess_meta_fields)
          .c_str());
  for (std::size_t i = 0; i < compiled.runtime_contract.logical_inputs.size(); ++i) {
    std::fprintf(
        stderr, "[processcvu-compare] family=%s runtime.logical_input index=%zu value=%s\n",
        family.c_str(), i,
        logical_input_dbg_processcvu_local(compiled.runtime_contract.logical_inputs[i]).c_str());
  }
  for (std::size_t i = 0; i < compiled.runtime_contract.input_bindings.size(); ++i) {
    std::fprintf(stderr,
                 "[processcvu-compare] family=%s runtime.input_binding index=%zu value=%s\n",
                 family.c_str(), i,
                 binding_dbg_processcvu_local(compiled.runtime_contract.input_bindings[i]).c_str());
  }
  for (std::size_t i = 0; i < compiled.runtime_contract.physical_inputs.size(); ++i) {
    std::fprintf(
        stderr, "[processcvu-compare] family=%s runtime.physical_input index=%zu value=%s\n",
        family.c_str(), i,
        physical_dbg_processcvu_local(compiled.runtime_contract.physical_inputs[i]).c_str());
  }
  for (std::size_t i = 0; i < compiled.runtime_contract.logical_outputs.size(); ++i) {
    std::fprintf(
        stderr, "[processcvu-compare] family=%s runtime.logical_output index=%zu value=%s\n",
        family.c_str(), i,
        logical_output_dbg_processcvu_local(compiled.runtime_contract.logical_outputs[i]).c_str());
  }
  for (std::size_t i = 0; i < compiled.runtime_contract.physical_outputs.size(); ++i) {
    std::fprintf(
        stderr, "[processcvu-compare] family=%s runtime.physical_output index=%zu value=%s\n",
        family.c_str(), i,
        physical_dbg_processcvu_local(compiled.runtime_contract.physical_outputs[i]).c_str());
  }
  for (std::size_t i = 0; i < compiled.runtime_contract.output_order.size(); ++i) {
    std::fprintf(stderr, "[processcvu-compare] family=%s runtime.route index=%zu value=%s\n",
                 family.c_str(), i,
                 route_dbg_processcvu_local(compiled.runtime_contract.output_order[i]).c_str());
  }
}

struct DetessIngressTransportView {
  std::vector<std::int64_t> transport_shape;
  std::uint64_t transport_size_bytes = 0U;
};

DetessIngressTransportView validate_detess_ingress_transport_local(
    const MpkTensorContract& published_output, const std::string& expected_dtype,
    const std::vector<std::int64_t>& transport_shape, std::uint64_t transport_size_bytes,
    const std::string& context) {
  if (transport_shape.empty() || transport_size_bytes == 0U) {
    throw std::runtime_error("processcvu MPK detess route requires packed transport bytes for '" +
                             context + "'");
  }

  const std::string published_dtype =
      preferred_tensor_dtype_local(published_output, expected_dtype);
  const bool packed_extent_transport =
      published_output.shape_semantics == MpkShapeSemantics::PackedExtent;
  if (!packed_extent_transport && !expected_dtype.empty() && !published_dtype.empty() &&
      expected_dtype != published_dtype) {
    throw std::runtime_error(
        "processcvu MPK detess route published MLA dtype disagrees with detess input for '" +
        context + "'");
  }

  const std::vector<std::int64_t> published_transport_shape = !published_output.mpk_shape.empty()
                                                                  ? published_output.mpk_shape
                                                                  : published_output.logical_shape;
  if (!published_transport_shape.empty() && published_transport_shape != transport_shape) {
    throw std::runtime_error("processcvu MPK detess route published MLA transport shape disagrees "
                             "with detess input for '" +
                             context + "'");
  }

  const std::uint64_t published_size_bytes =
      preferred_mpk_tensor_size_bytes_local(published_output, published_dtype);
  if (published_size_bytes == 0U || published_size_bytes != transport_size_bytes) {
    throw std::runtime_error("processcvu MPK detess route published MLA boundary view disagrees "
                             "with detess input bytes for '" +
                             context + "'");
  }

  return {transport_shape, transport_size_bytes};
}

std::string upper_copy_local(std::string value) {
  std::transform(value.begin(), value.end(), value.begin(),
                 [](unsigned char c) { return static_cast<char>(std::toupper(c)); });
  return value;
}

std::string normalize_layout(std::string layout) {
  return tensorsemantics::normalize_layout_token(std::move(layout));
}

bool graph_family_implies_tessellate(const std::string& family) {
  const std::string canonical = lower_copy(family);
  return canonical == "tessellate" || canonical == "quanttess" ||
         canonical == "quantizetessellate" || canonical == "quantize_tessellate";
}

std::string layout_from_image_format(const std::string& format, int channels) {
  const std::string up = upper_copy_local(format);
  const int resolved_channels = (up == "GRAY" || up == "GRAY8") ? 1
                                : (up == "RGB" || up == "BGR" || up == "NV12" || up == "I420")
                                    ? 3
                                    : channels;
  return resolved_channels <= 1 ? "HW" : "HWC";
}

void apply_quant_payload_fields(ProcessCvuStagePayload* payload,
                                const std::optional<double>& q_scale,
                                const std::optional<std::int64_t>& q_zp) {
  if (!payload) {
    return;
  }
  payload->q_scale = q_scale.value_or(0.0);
  payload->q_zp = static_cast<int>(q_zp.value_or(0));
  payload->has_q_scale = q_scale.has_value();
  payload->has_q_zp = q_zp.has_value();
}

int pick_first_positive_dim(std::initializer_list<int> values) {
  for (const int value : values) {
    if (value > 0) {
      return value;
    }
  }
  return 0;
}

int effective_dense_depth_for_layout(const std::string& layout, int depth, int channels) {
  const std::string normalized_layout = normalize_layout(layout);
  if (normalized_layout == "HWC") {
    if (depth <= 1 && channels > 0) {
      return channels;
    }
    if (channels <= 1 && depth > 0) {
      return depth;
    }
    return std::max(depth, channels);
  }
  return pick_first_positive_dim({depth, channels, 1});
}

void require_positive_runtime_fact(int value, const char* field_name) {
  if (value > 0) {
    return;
  }
  throw std::invalid_argument(std::string("processcvu runtime config missing required field '") +
                              field_name + "'");
}

int require_equal_positive_pair(int first, int second, const char* first_name,
                                const char* second_name) {
  require_positive_runtime_fact(first, first_name);
  require_positive_runtime_fact(second, second_name);
  if (first != second) {
    throw std::invalid_argument(std::string("processcvu runtime config fields '") + first_name +
                                "' and '" + second_name + "' must match");
  }
  return first;
}

void require_runtime_depth_channel_pair(int depth, int channels, const std::string& layout,
                                        const std::string& family, const char* depth_name,
                                        const char* channels_name) {
  require_positive_runtime_fact(depth, depth_name);
  require_positive_runtime_fact(channels, channels_name);
  const std::string normalized_layout = normalize_layout(layout);
  const std::string normalized_family = canonical_family_name(family);
  const bool image_like_hwc =
      normalized_layout == "HWC" && ((depth == 1 && channels > 0) || (channels == 1 && depth > 0));
  const bool rank_aware_detess_hwc =
      normalized_layout == "HWC" &&
      (normalized_family == "detessellate" || normalized_family == "detessdequant");
  if (!image_like_hwc && !rank_aware_detess_hwc && depth != channels) {
    throw std::invalid_argument(std::string("processcvu runtime config fields '") + depth_name +
                                "' and '" + channels_name + "' must match");
  }
}

void require_non_empty_runtime_token(const std::string& value, const char* field_name) {
  if (!value.empty()) {
    return;
  }
  throw std::invalid_argument(std::string("processcvu runtime config missing required field '") +
                              field_name + "'");
}

bool is_known_preproc_output_name(const std::string& name) {
  return name == "output_rgb_image" || name == "output_tessellated_image";
}

void require_supported_preproc_single_output_handoff(bool enabled) {
  if (enabled) {
    return;
  }
  throw std::invalid_argument(
      "Preproc dual-output contract is currently unsupported; use single_output_handoff=true.");
}

struct ProcessCvuLogicalDims {
  int width = 0;
  int height = 0;
  int depth = 0;
};

ProcessCvuOutputTransportKind
transport_kind_from_representation(ProcessCvuOutputRepresentation representation) {
  switch (representation) {
  case ProcessCvuOutputRepresentation::DenseTensor:
    return ProcessCvuOutputTransportKind::Dense;
  case ProcessCvuOutputRepresentation::PackedBlob:
  case ProcessCvuOutputRepresentation::PackedTensor:
    return ProcessCvuOutputTransportKind::Packed;
  }
  return ProcessCvuOutputTransportKind::Unknown;
}

bool transport_kind_is_packed(ProcessCvuOutputTransportKind kind) {
  return kind == ProcessCvuOutputTransportKind::Packed;
}

ProcessCvuOutputSemanticKind semantic_kind_from_family(const std::string& family) {
  const std::string canonical = canonical_family_name(family);
  if (canonical == "preproc") {
    return ProcessCvuOutputSemanticKind::Image;
  }
  if (canonical == "quantize") {
    return ProcessCvuOutputSemanticKind::QuantizedTensor;
  }
  if (canonical == "quanttess") {
    return ProcessCvuOutputSemanticKind::QuantTessTensor;
  }
  if (canonical == "tessellate" || canonical == "casttess") {
    return ProcessCvuOutputSemanticKind::TessellatedImage;
  }
  return ProcessCvuOutputSemanticKind::Tensor;
}

ProcessCvuOutputSemanticKind
preproc_semantic_kind_for_output_name(const std::string& output_name,
                                      ProcessCvuOutputTransportKind transport_kind) {
  (void)transport_kind;
  if (output_name == "output_rgb_image") {
    return ProcessCvuOutputSemanticKind::Image;
  }
  if (output_name == "output_tessellated_image") {
    return ProcessCvuOutputSemanticKind::TessellatedImage;
  }
  return ProcessCvuOutputSemanticKind::Image;
}

int packed_transport_width_for_preproc_output(const ProcessCvuStagePayload& payload,
                                              const std::string& dtype) {
  const auto out_d = payload_output_dims_at(payload, 0);
  const auto in_d = payload_input_dims_at(payload, 0);
  const auto sl_d = payload_slice_dims_at(payload, 0);
  const int width = pick_first_positive_dim({out_d.width, payload.scaled_width});
  const int height = pick_first_positive_dim({out_d.height, payload.scaled_height});
  const int depth = pick_first_positive_dim(
      {out_d.channels, out_d.depth, in_d.channels, in_d.depth, sl_d.channels, sl_d.depth, 3});
  const std::string layout = payload_output_layout_token_local(payload);
  if (layout.empty()) {
    return 0;
  }
  const auto shape = specbuilders::dense_shape_from_dims(width, height, depth, layout);
  const std::uint64_t packed_size_bytes =
      specbuilders::tensor_size_bytes_from_shape_dtype(shape, dtype);
  if (packed_size_bytes == 0U) {
    return 0;
  }
  const std::uint64_t elem_bytes = specbuilders::dtype_size_bytes_from_token(dtype);
  if (elem_bytes == 0U) {
    return 0;
  }
  return static_cast<int>(
      std::max<std::uint64_t>((packed_size_bytes + elem_bytes - 1U) / elem_bytes, 1U));
}

ProcessCvuLogicalDims preproc_logical_dims_from_payload(const ProcessCvuStagePayload& payload) {
  const auto out_d = payload_output_dims_at(payload, 0);
  const auto in_d = payload_input_dims_at(payload, 0);
  const auto sl_d = payload_slice_dims_at(payload, 0);
  ProcessCvuLogicalDims dims;
  dims.width = pick_first_positive_dim({out_d.width, payload.scaled_width, sl_d.width});
  dims.height = pick_first_positive_dim({out_d.height, payload.scaled_height, sl_d.height});
  dims.depth = pick_first_positive_dim({out_d.channels, out_d.depth, in_d.channels, in_d.depth, 1});
  return dims;
}

std::string preproc_logical_layout_from_payload(const ProcessCvuStagePayload& payload,
                                                const ProcessCvuLogicalDims& dims) {
  (void)dims;
  return payload_runtime_output_layout_token_local(payload);
}

std::string preproc_image_axis_layout_token_local(const std::vector<int>& shape) {
  if (shape.size() == 2U) {
    return "HW";
  }
  if (shape.size() >= 3U) {
    return "HWC";
  }
  return {};
}

std::vector<int>
first_non_empty_shape_local(std::initializer_list<const std::vector<int>*> shapes) {
  for (const auto* shape : shapes) {
    if (shape && !shape->empty()) {
      return *shape;
    }
  }
  return {};
}

std::vector<int> preproc_payload_input_shape_for_desc(const ProcessCvuStagePayload& payload) {
  if (!payload.input_shapes.empty() && !payload.input_shapes.front().empty()) {
    return payload.input_shapes.front();
  }
  if (!payload.input_tensors.empty()) {
    return shape_vec_from_tensor_desc_local(payload.input_tensors.front());
  }
  return {};
}

std::string preproc_output_dtype_for_desc(const ProcessCvuStagePayload& payload,
                                          std::size_t index) {
  if (index < payload.runtime_output_dtype_list.size() &&
      !payload.runtime_output_dtype_list[index].empty()) {
    return payload.runtime_output_dtype_list[index];
  }
  if (!payload.output_dtype.empty()) {
    return payload.output_dtype;
  }
  if (!payload.out_dtype.empty()) {
    return payload.out_dtype;
  }
  if (!payload.input_dtype.empty()) {
    return payload.input_dtype;
  }
  return "INT8";
}

std::vector<int> preproc_payload_output_shape_for_desc(const ProcessCvuStagePayload& payload,
                                                       std::size_t index) {
  const std::vector<int>* logical_shape = index < payload.runtime_output_logical_shapes.size()
                                              ? &payload.runtime_output_logical_shapes[index]
                                              : nullptr;
  const std::vector<int>* output_shape =
      index < payload.output_shapes.size() ? &payload.output_shapes[index] : nullptr;
  std::vector<int> tensor_shape;
  if (index < payload.output_tensors.size()) {
    tensor_shape = shape_vec_from_tensor_desc_local(payload.output_tensors[index]);
  }
  return first_non_empty_shape_local({logical_shape, output_shape, &tensor_shape});
}

std::vector<int> preproc_payload_slice_shape_for_desc(const ProcessCvuStagePayload& payload,
                                                      std::size_t index) {
  if (index < payload.slice_shapes.size() && !payload.slice_shapes[index].empty()) {
    return payload.slice_shapes[index];
  }
  if (!payload.slice_shapes.empty() && !payload.slice_shapes.front().empty()) {
    return payload.slice_shapes.front();
  }
  return {};
}

ProcessCvuOutputTransportKind
preproc_output_transport_kind_for_desc(const ProcessCvuStagePayload& payload, std::size_t index,
                                       const std::string& output_name) {
  if (index < payload.runtime_output_transport_kind_list.size()) {
    return payload.runtime_output_transport_kind_list[index];
  }
  return payload.tessellate == 1 && output_name == "output_tessellated_image"
             ? ProcessCvuOutputTransportKind::Packed
             : ProcessCvuOutputTransportKind::Dense;
}

void synthesize_preproc_payload_tensor_descs(ProcessCvuStagePayload* payload) {
  if (!payload || canonical_family_name(payload->graph_family) != "preproc") {
    return;
  }

  if (payload->input_tensors.empty()) {
    const std::vector<int> input_shape = preproc_payload_input_shape_for_desc(*payload);
    const std::string input_layout = preproc_image_axis_layout_token_local(input_shape);
    if (!input_shape.empty() && !input_layout.empty()) {
      sima_ev_tensor_desc input_desc{};
      const std::string input_dtype =
          !payload->input_dtype.empty() ? payload->input_dtype : std::string("UINT8");
      if (build_dense_desc_processcvu_local(input_shape, input_dtype, input_layout, &input_desc)) {
        payload->input_tensors = {input_desc};
        payload->num_in_tensor = 1;
      }
    }
  }

  if (!payload->output_tensors.empty()) {
    return;
  }

  payload->output_tensors.reserve(payload->default_output_names.size());
  for (std::size_t i = 0; i < payload->default_output_names.size(); ++i) {
    const std::string& output_name = payload->default_output_names[i];
    const std::vector<int> output_shape = preproc_payload_output_shape_for_desc(*payload, i);
    const std::string output_layout = preproc_image_axis_layout_token_local(output_shape);
    if (output_shape.empty() || output_layout.empty()) {
      payload->output_tensors.clear();
      return;
    }

    const std::string output_dtype = preproc_output_dtype_for_desc(*payload, i);
    const auto transport_kind = preproc_output_transport_kind_for_desc(*payload, i, output_name);
    sima_ev_tensor_desc output_desc{};
    if (transport_kind_is_packed(transport_kind)) {
      const std::vector<int> raw_tile_shape = preproc_payload_slice_shape_for_desc(*payload, i);
      std::vector<int> tile_shape;
      std::string tile_error;
      if (!tensorsemantics::normalize_tile_shape(
              output_shape, raw_tile_shape, &tile_shape, &tile_error,
              "processcvu preproc tessellated output requires explicit slice_shape",
              "processcvu preproc tessellated output slice_shape rank invalid",
              "processcvu preproc tessellated output slice_shape dim invalid") ||
          !build_tiled_desc_processcvu_local(
              output_shape, tile_shape, output_dtype, output_layout,
              resolve_tile_align_bytes_processcvu_local(payload->byte_align), &output_desc)) {
        payload->output_tensors.clear();
        return;
      }
    } else if (!build_dense_desc_processcvu_local(output_shape, output_dtype, output_layout,
                                                  &output_desc)) {
      payload->output_tensors.clear();
      return;
    }
    payload->output_tensors.push_back(output_desc);
  }
}

void populate_preproc_payload_semantics(ProcessCvuStagePayload* payload) {
  if (!payload) {
    return;
  }
  if (payload->default_output_names.empty()) {
    payload->default_output_names = {"output_rgb_image", "output_tessellated_image"};
  }

  const ProcessCvuLogicalDims logical_dims = preproc_logical_dims_from_payload(*payload);
  const std::string logical_layout = preproc_logical_layout_from_payload(*payload, logical_dims);
  const std::vector<int> logical_shape = [&]() {
    if (!payload->output_tensors.empty()) {
      const auto shape = shape_vec_from_tensor_desc_local(payload->output_tensors.front());
      if (!shape.empty()) {
        return shape;
      }
    }
    if (!payload->output_shapes.empty() && !payload->output_shapes.front().empty()) {
      return payload->output_shapes.front();
    }
    const auto shape64 = specbuilders::dense_shape_from_dims(
        logical_dims.width, logical_dims.height, logical_dims.depth, logical_layout);
    std::vector<int> shape(shape64.begin(), shape64.end());
    if (!shape.empty()) {
      return shape;
    }
    if (logical_dims.height > 0 && logical_dims.width > 0 && logical_dims.depth > 0) {
      return std::vector<int>{logical_dims.height, logical_dims.width, logical_dims.depth};
    }
    return std::vector<int>{};
  }();
  const std::string dtype = !payload->output_dtype.empty() ? payload->output_dtype
                            : !payload->out_dtype.empty()  ? payload->out_dtype
                                                           : payload->input_dtype;
  const std::string family = canonical_family_name(payload->graph_family);
  if (family != "preproc") {
    return;
  }

  payload->output_shapes.clear();
  payload->runtime_output_logical_index_list.clear();
  payload->runtime_output_output_slot_list.clear();
  payload->runtime_output_physical_index_list.clear();
  payload->runtime_output_dtype_list.clear();
  payload->runtime_output_transport_kind_list.clear();
  payload->runtime_output_semantic_kind_list.clear();
  payload->runtime_output_logical_shapes.clear();
  payload->runtime_output_logical_layout_list.clear();
  payload->output_tensors.clear();

  payload->output_shapes.reserve(payload->default_output_names.size());
  payload->runtime_output_logical_index_list.reserve(payload->default_output_names.size());
  payload->runtime_output_output_slot_list.reserve(payload->default_output_names.size());
  payload->runtime_output_physical_index_list.reserve(payload->default_output_names.size());
  payload->runtime_output_dtype_list.reserve(payload->default_output_names.size());
  payload->runtime_output_transport_kind_list.reserve(payload->default_output_names.size());
  payload->runtime_output_semantic_kind_list.reserve(payload->default_output_names.size());
  payload->runtime_output_logical_shapes.reserve(payload->default_output_names.size());
  payload->runtime_output_logical_layout_list.reserve(payload->default_output_names.size());

  for (std::size_t i = 0; i < payload->default_output_names.size(); ++i) {
    const std::string& output_name = payload->default_output_names[i];
    const bool packed_transport =
        payload->tessellate == 1 && output_name == "output_tessellated_image";
    const auto transport_kind = packed_transport ? ProcessCvuOutputTransportKind::Packed
                                                 : ProcessCvuOutputTransportKind::Dense;
    const auto semantic_kind = preproc_semantic_kind_for_output_name(output_name, transport_kind);
    payload->output_shapes.push_back(logical_shape);
    payload->runtime_output_logical_index_list.push_back(static_cast<int>(i));
    payload->runtime_output_output_slot_list.push_back(static_cast<int>(i));
    payload->runtime_output_physical_index_list.push_back(static_cast<int>(i));
    payload->runtime_output_dtype_list.push_back(dtype);
    payload->runtime_output_transport_kind_list.push_back(transport_kind);
    payload->runtime_output_semantic_kind_list.push_back(semantic_kind);
    payload->runtime_output_logical_shapes.push_back(logical_shape);
    payload->runtime_output_logical_layout_list.push_back(logical_layout);

    if (output_name == payload->primary_output_name) {
      payload->primary_output_transport_kind = transport_kind;
      payload->primary_output_semantic_kind = semantic_kind;
    }
  }
  synthesize_preproc_payload_tensor_descs(payload);
}

void apply_tensor_spatial_extents_from_shape(const std::vector<std::int64_t>& shape,
                                             const std::string& layout, TensorStaticSpec* tensor) {
  if (!tensor) {
    return;
  }
  tensor->max_w = 0;
  tensor->max_h = 0;
  const auto to_non_negative_int = [](std::int64_t value) {
    return static_cast<int>(std::max<std::int64_t>(0, value));
  };
  if (shape.empty()) {
    return;
  }
  if (layout == "CHW" && shape.size() >= 3U) {
    tensor->max_h = to_non_negative_int(shape[shape.size() - 2U]);
    tensor->max_w = to_non_negative_int(shape.back());
    return;
  }
  if (shape.size() >= 3U) {
    tensor->max_h = to_non_negative_int(shape[shape.size() - 3U]);
    tensor->max_w = to_non_negative_int(shape[shape.size() - 2U]);
    return;
  }
  if (shape.size() == 2U) {
    tensor->max_h = to_non_negative_int(shape[0]);
    tensor->max_w = to_non_negative_int(shape[1]);
    return;
  }
  tensor->max_h = 1;
  tensor->max_w = to_non_negative_int(shape.front());
}

TensorStaticSpec make_tensor_spec_from_shape(std::vector<std::int64_t> shape, std::string dtype,
                                             std::string layout, std::string semantic_tag,
                                             int tensor_index) {
  TensorStaticSpec tensor;
  tensor.tensor_index = tensor_index;
  tensor.shape = std::move(shape);
  tensor.layout = normalize_layout(std::move(layout));
  tensor.dtype = dtype.empty() ? "INT8" : std::move(dtype);
  tensor.max_stride = 0;
  tensor.semantic_tag = std::move(semantic_tag);
  apply_tensor_spatial_extents_from_shape(tensor.shape, tensor.layout, &tensor);
  return tensor;
}

std::vector<std::int64_t> preferred_mpk_tensor_shape_local(const MpkTensorContract& tensor) {
  if (!tensor.logical_shape.empty()) {
    return tensor.logical_shape;
  }
  return tensor.mpk_shape;
}

std::vector<std::int64_t>
preferred_physical_mpk_tensor_shape_local(const MpkTensorContract& tensor) {
  if (!tensor.mpk_shape.empty()) {
    return tensor.mpk_shape;
  }
  return tensor.logical_shape;
}

std::vector<std::int64_t> preferred_packed_mpk_tensor_shape_local(const MpkTensorContract& tensor) {
  if (!tensor.logical_shape.empty()) {
    return tensor.logical_shape;
  }
  return tensor.mpk_shape;
}

std::vector<std::int64_t>
preferred_stage_input_tensor_shape_local(const MpkPluginIoContract& stage,
                                         const MpkTensorContract& tensor) {
  if (!tensor.logical_shape.empty()) {
    return tensor.logical_shape;
  }
  if (!tensor.mpk_shape.empty()) {
    return tensor.mpk_shape;
  }
  return preferred_mpk_tensor_shape_local(tensor);
}

std::vector<std::int64_t> drop_leading_unit_batch_local(std::vector<std::int64_t> shape) {
  if (shape.size() > 1U && !shape.empty() && shape.front() == 1) {
    shape.erase(shape.begin());
  }
  return shape;
}

MpkTensorContract make_synthetic_tensor_contract_local(const std::vector<std::int64_t>& shape,
                                                       const std::string& dtype, int tensor_index,
                                                       const std::string& name) {
  MpkTensorContract tensor;
  tensor.tensor_index = tensor_index;
  tensor.name = name;
  tensor.kind = "buffer";
  tensor.dtype = dtype;
  tensor.mpk_shape = shape;
  tensor.logical_shape = drop_leading_unit_batch_local(shape);
  tensor.logical_dtype = dtype;
  tensor.shape_semantics = MpkShapeSemantics::Geometry;
  tensor.size_bytes = static_cast<std::size_t>(specbuilders::tensor_size_bytes_from_shape_dtype(
      !tensor.logical_shape.empty() ? tensor.logical_shape : tensor.mpk_shape, dtype));
  return tensor;
}

std::uint64_t preferred_mpk_tensor_size_bytes_local(const MpkTensorContract& tensor,
                                                    const std::string& dtype);

std::string processcvu_mpk_tensor_name(const MpkTensorContract& tensor,
                                       const std::string& fallback_name) {
  if (!tensor.name.empty()) {
    return tensor.name;
  }
  if (!tensor.segment_name.empty()) {
    return tensor.segment_name;
  }
  return fallback_name;
}

std::string processcvu_output_name_from_logical(const LogicalTensorStaticSpec& logical) {
  if (!logical.backend_name.empty()) {
    return logical.backend_name;
  }
  if (!logical.segment_name.empty()) {
    return logical.segment_name;
  }
  return logical.logical_name;
}

int tensor_depth_from_shape(const TensorStaticSpec& tensor) {
  auto positive_or_zero = [](std::int64_t value) {
    return value > 0 ? static_cast<int>(value) : 0;
  };
  const std::string layout = upper_copy_local(tensor.layout);
  if (layout == "CHW" || layout == "NCHW") {
    return tensor.shape.size() >= 3U ? positive_or_zero(tensor.shape[tensor.shape.size() - 3U]) : 1;
  }
  if (layout == "HW" || layout == "NHW") {
    return 1;
  }
  return tensor.shape.size() >= 3U ? positive_or_zero(tensor.shape.back()) : 1;
}

std::string logical_output_name_for_selection(const LogicalTensorStaticSpec& logical) {
  if (!logical.logical_name.empty()) {
    return logical.logical_name;
  }
  if (!logical.backend_name.empty()) {
    return logical.backend_name;
  }
  return logical.segment_name;
}

int pick_indexed_or_scalar(const std::vector<int>& values, std::size_t index, int fallback) {
  return index < values.size() ? values[index] : fallback;
}

std::string pick_indexed_or_scalar(const std::vector<std::string>& values, std::size_t index,
                                   const std::string& fallback) {
  return index < values.size() && !values[index].empty() ? values[index] : fallback;
}

std::vector<std::size_t>
select_processcvu_runtime_output_indices(const CompiledRuntimeContract& runtime,
                                         const std::vector<std::string>& selected_output_names) {
  std::vector<std::size_t> indices;
  auto unique_logical_index_for_segment =
      [&](const std::string& segment_name) -> std::optional<std::size_t> {
    if (segment_name.empty()) {
      return std::nullopt;
    }
    std::optional<std::size_t> matched;
    for (std::size_t i = 0; i < runtime.logical_outputs.size(); ++i) {
      if (runtime.logical_outputs[i].segment_name != segment_name) {
        continue;
      }
      if (matched.has_value()) {
        return std::nullopt;
      }
      matched = i;
    }
    return matched;
  };
  auto find_runtime_logical_index_for_route =
      [&](const StageOutputRoute& route) -> std::optional<std::size_t> {
    for (std::size_t i = 0; i < runtime.logical_outputs.size(); ++i) {
      const auto& logical = runtime.logical_outputs[i];
      if (route.logical_output_index >= 0 && logical.logical_index == route.logical_output_index) {
        return i;
      }
      if (route.output_slot >= 0 && logical.output_slot == route.output_slot) {
        return i;
      }
      if (route.tensor_index >= 0 && logical.tensor_index == route.tensor_index) {
        return i;
      }
      if (!route.cm_output_name.empty() &&
          processcvu_output_name_from_logical(logical) == route.cm_output_name) {
        return i;
      }
    }
    if (const auto unique_match = unique_logical_index_for_segment(route.segment_name);
        unique_match.has_value()) {
      return *unique_match;
    }
    return std::nullopt;
  };
  auto append_index = [&](std::size_t index) {
    if (index >= runtime.logical_outputs.size()) {
      return;
    }
    if (std::find(indices.begin(), indices.end(), index) == indices.end()) {
      indices.push_back(index);
    }
  };
  auto append_name = [&](const std::string& wanted_name) {
    if (wanted_name.empty()) {
      return;
    }
    for (std::size_t i = 0; i < runtime.output_order.size(); ++i) {
      const auto& route = runtime.output_order[i];
      if (output_name_from_route(route) == wanted_name ||
          (!route.cm_output_name.empty() && route.cm_output_name == wanted_name) ||
          (!route.segment_name.empty() && route.segment_name == wanted_name)) {
        if (const auto logical_index =
                find_runtime_logical_index_for_route(runtime.output_order[i]);
            logical_index.has_value()) {
          append_index(*logical_index);
        }
        return;
      }
    }
    for (std::size_t i = 0; i < runtime.logical_outputs.size(); ++i) {
      if (processcvu_output_name_from_logical(runtime.logical_outputs[i]) == wanted_name) {
        append_index(i);
        return;
      }
    }
  };

  for (const auto& name : selected_output_names) {
    append_name(name);
  }
  if (!selected_output_names.empty()) {
    return indices;
  }
  if (!indices.empty()) {
    return indices;
  }

  indices.reserve(runtime.logical_outputs.size());
  for (std::size_t i = 0; i < runtime.logical_outputs.size(); ++i) {
    indices.push_back(i);
  }
  return indices;
}

CompiledExposedView
build_processcvu_exposed_view_from_runtime(const CompiledRuntimeContract& runtime,
                                           const std::vector<std::string>& selected_output_names,
                                           std::string primary_output_name) {
  if (primary_output_name.empty()) {
    throw std::invalid_argument("processcvu exposed view requires explicit primary_output_name");
  }
  CompiledExposedView exposed;
  exposed.primary_output_name = std::move(primary_output_name);

  const auto selected_indices =
      select_processcvu_runtime_output_indices(runtime, selected_output_names);
  if (!selected_output_names.empty() && selected_indices.size() != selected_output_names.size()) {
    throw std::invalid_argument(
        "processcvu exposed view selection did not resolve every published output");
  }
  if (pipeline_internal::env_bool("SIMA_RENDER_STAGE_DEBUG", false) &&
      selected_output_names.size() > 1U) {
    std::fprintf(
        stderr, "[exposed-view-debug] selected_names=%zu selected_indices=%zu primary=%s\n",
        selected_output_names.size(), selected_indices.size(), primary_output_name.c_str());
    for (std::size_t i = 0; i < runtime.output_order.size(); ++i) {
      std::fprintf(stderr,
                   "  [exposed-view-debug] route[%zu]=cm:%s seg:%s logical:%d slot:%d tensor:%d\n",
                   i, runtime.output_order[i].cm_output_name.c_str(),
                   runtime.output_order[i].segment_name.c_str(),
                   runtime.output_order[i].logical_output_index,
                   runtime.output_order[i].output_slot, runtime.output_order[i].tensor_index);
    }
    for (std::size_t i = 0; i < runtime.logical_outputs.size(); ++i) {
      const auto& logical = runtime.logical_outputs[i];
      std::fprintf(stderr,
                   "  [exposed-view-debug] logical[%zu]=name:%s backend:%s seg:%s logical:%d "
                   "slot:%d tensor:%d\n",
                   i, logical.logical_name.c_str(), logical.backend_name.c_str(),
                   logical.segment_name.c_str(), logical.logical_index, logical.output_slot,
                   logical.tensor_index);
    }
    for (std::size_t i = 0; i < selected_output_names.size(); ++i) {
      std::fprintf(stderr, "  [exposed-view-debug] wanted[%zu]=%s\n", i,
                   selected_output_names[i].c_str());
    }
  }
  exposed.exposed_logical_outputs.reserve(selected_indices.size());
  exposed.exposed_output_order.reserve(selected_indices.size());

  auto find_runtime_route_for_logical =
      [&](const LogicalTensorStaticSpec& logical) -> std::optional<StageOutputRoute> {
    auto unique_route_index_for_segment =
        [&](const std::string& segment_name) -> std::optional<std::size_t> {
      if (segment_name.empty()) {
        return std::nullopt;
      }
      std::optional<std::size_t> matched;
      for (std::size_t i = 0; i < runtime.output_order.size(); ++i) {
        if (runtime.output_order[i].segment_name != segment_name) {
          continue;
        }
        if (matched.has_value()) {
          return std::nullopt;
        }
        matched = i;
      }
      return matched;
    };
    for (const auto& route : runtime.output_order) {
      if (route.logical_output_index >= 0 && route.logical_output_index == logical.logical_index) {
        return route;
      }
      if (route.output_slot >= 0 && route.output_slot == logical.output_slot) {
        return route;
      }
      if (route.tensor_index >= 0 && route.tensor_index == logical.tensor_index) {
        return route;
      }
      if (!route.cm_output_name.empty() &&
          route.cm_output_name == processcvu_output_name_from_logical(logical)) {
        return route;
      }
    }
    if (const auto unique_match = unique_route_index_for_segment(logical.segment_name);
        unique_match.has_value()) {
      return runtime.output_order[*unique_match];
    }
    return std::nullopt;
  };

  for (std::size_t exposed_slot = 0; exposed_slot < selected_indices.size(); ++exposed_slot) {
    const std::size_t runtime_index = selected_indices[exposed_slot];
    if (runtime_index >= runtime.logical_outputs.size()) {
      continue;
    }

    const auto& runtime_logical = runtime.logical_outputs[runtime_index];
    LogicalTensorStaticSpec logical = runtime_logical;
    logical.output_slot = static_cast<int>(exposed_slot);
    exposed.exposed_logical_outputs.push_back(logical);

    StageOutputRoute route;
    if (const auto matched_route = find_runtime_route_for_logical(runtime_logical);
        matched_route.has_value()) {
      route = *matched_route;
    }
    route.output_slot = static_cast<int>(exposed_slot);
    route.logical_output_index = logical.logical_index;
    route.tensor_index = logical.tensor_index;
    if (route.cm_output_name.empty()) {
      route.cm_output_name = processcvu_output_name_from_logical(logical);
    }
    if (route.segment_name.empty()) {
      route.segment_name = logical.segment_name;
    }
    exposed.exposed_output_order.push_back(std::move(route));
  }

  if (exposed.exposed_output_order.empty()) {
    throw std::invalid_argument("processcvu exposed view requires explicit selected outputs");
  }
  const bool primary_exposed = std::any_of(
      exposed.exposed_output_order.begin(), exposed.exposed_output_order.end(),
      [&](const StageOutputRoute& route) {
        return output_name_from_route(route) == exposed.primary_output_name ||
               (!route.segment_name.empty() && route.segment_name == exposed.primary_output_name);
      });
  if (!primary_exposed) {
    if (pipeline_internal::env_bool("SIMA_RENDER_STAGE_DEBUG", false)) {
      std::fprintf(
          stderr,
          "[exposed-view-debug] primary-miss primary=%s selected_count=%zu exposed_count=%zu\n",
          exposed.primary_output_name.c_str(), selected_output_names.size(),
          exposed.exposed_output_order.size());
      for (std::size_t i = 0; i < selected_output_names.size(); ++i) {
        std::fprintf(stderr, "  [exposed-view-debug] selected[%zu]=%s\n", i,
                     selected_output_names[i].c_str());
      }
      for (std::size_t i = 0; i < exposed.exposed_output_order.size(); ++i) {
        const auto& route = exposed.exposed_output_order[i];
        std::fprintf(stderr,
                     "  [exposed-view-debug] exposed_route[%zu]=name:%s cm:%s seg:%s logical:%d "
                     "slot:%d tensor:%d\n",
                     i, output_name_from_route(route).c_str(), route.cm_output_name.c_str(),
                     route.segment_name.c_str(), route.logical_output_index, route.output_slot,
                     route.tensor_index);
      }
      for (std::size_t i = 0; i < exposed.exposed_logical_outputs.size(); ++i) {
        const auto& logical = exposed.exposed_logical_outputs[i];
        std::fprintf(stderr,
                     "  [exposed-view-debug] exposed_logical[%zu]=logical:%s backend:%s segment:%s "
                     "idx:%d slot:%d tensor:%d\n",
                     i, logical.logical_name.c_str(), logical.backend_name.c_str(),
                     logical.segment_name.c_str(), logical.logical_index, logical.output_slot,
                     logical.tensor_index);
      }
    }
    throw std::invalid_argument(
        "processcvu exposed view primary_output_name must name an exposed output");
  }
  return exposed;
}

LogicalInputStaticSpec logical_input_from_fact(const ProcessCvuCanonicalInputFact& fact) {
  const std::string segment_name =
      !fact.physical_name.empty() ? fact.physical_name : fact.logical_name;
  const std::string logical_name = !fact.logical_name.empty() ? fact.logical_name : segment_name;
  return specbuilders::build_logical_input_static_spec(
      fact.logical_index, fact.logical_index, fact.physical_index, fact.shape, fact.dtype,
      fact.layout, logical_name, logical_name, segment_name, fact.byte_offset, fact.size_bytes,
      fact.materialization_kind, fact.quant);
}

InputBindingStaticSpec input_binding_from_fact(const ProcessCvuCanonicalBindingFact& fact) {
  return specbuilders::build_input_binding_static_spec(
      fact.sink_pad_index, fact.local_logical_input_index, fact.cm_input_name,
      fact.source_segment_name, fact.src_logical_output_index, fact.src_output_slot,
      fact.src_physical_output_index, fact.src_physical_size_bytes, fact.src_physical_byte_offset,
      fact.required);
}

LogicalTensorStaticSpec logical_output_from_fact(const ProcessCvuCanonicalOutputFact& fact) {
  const std::string segment_name =
      !fact.physical_name.empty() ? fact.physical_name : fact.logical_name;
  const std::string logical_name = !fact.logical_name.empty() ? fact.logical_name : segment_name;
  const std::uint64_t size_override =
      fact.representation == ProcessCvuOutputRepresentation::DenseTensor ? 0U : fact.size_bytes;
  auto logical = specbuilders::build_logical_output_static_spec(
      fact.logical_index, fact.logical_index, fact.physical_index, fact.output_slot,
      fact.tensor_index, fact.shape, fact.dtype, fact.layout, logical_name, logical_name,
      segment_name, fact.byte_offset, size_override, fact.quant);
  if (processcvu_detess_layout_debug_enabled()) {
    std::fprintf(stderr,
                 "[detess-layout-debug] where=logical_output_from_fact logical=%d slot=%d "
                 "physical=%d repr=%d "
                 "fact_layout=%s fact_shape=%s fact_dtype=%s fact_name=%s built_layout=%s "
                 "built_shape=%s built_segment=%s\n",
                 fact.logical_index, fact.output_slot, fact.physical_index,
                 static_cast<int>(fact.representation), fact.layout.c_str(),
                 join_i64_debug_processcvu_local(fact.shape).c_str(), fact.dtype.c_str(),
                 logical_name.c_str(), logical.layout.c_str(),
                 join_i64_debug_processcvu_local(logical.shape).c_str(),
                 logical.segment_name.c_str());
  }
  return logical;
}

StageOutputRoute output_route_from_fact(const ProcessCvuCanonicalRouteFact& fact) {
  return specbuilders::build_output_route_static_spec(fact.output_slot, fact.logical_output_index,
                                                      fact.tensor_index, fact.cm_output_name,
                                                      fact.segment_name);
}

std::vector<std::string> derive_processcvu_physical_names_from_facts(
    const std::vector<std::string>& configured_names,
    const std::vector<ProcessCvuCanonicalInputFact>& inputs,
    const std::vector<ProcessCvuCanonicalOutputFact>& outputs, bool output_names) {
  std::size_t count = configured_names.size();
  const auto update_count = [&](int physical_index) {
    if (physical_index >= 0) {
      count = std::max(count, static_cast<std::size_t>(physical_index + 1));
    }
  };
  if (output_names) {
    for (const auto& output : outputs) {
      update_count(output.physical_index);
    }
  } else {
    for (const auto& input : inputs) {
      update_count(input.physical_index);
    }
  }
  count = std::max<std::size_t>(count, 1U);

  std::vector<std::string> names = configured_names;
  names.resize(count);
  if (output_names) {
    for (const auto& output : outputs) {
      if (output.physical_index >= 0 &&
          static_cast<std::size_t>(output.physical_index) < names.size() &&
          names[static_cast<std::size_t>(output.physical_index)].empty()) {
        names[static_cast<std::size_t>(output.physical_index)] = output.physical_name;
      }
    }
  } else {
    for (const auto& input : inputs) {
      if (input.physical_index >= 0 &&
          static_cast<std::size_t>(input.physical_index) < names.size() &&
          names[static_cast<std::size_t>(input.physical_index)].empty()) {
        names[static_cast<std::size_t>(input.physical_index)] = input.physical_name;
      }
    }
  }
  for (std::size_t i = 0; i < names.size(); ++i) {
    if (names[i].empty()) {
      names[i] = output_names
                     ? (names.size() == 1U ? "output_tensor" : "output_tensor_" + std::to_string(i))
                     : (names.size() == 1U ? "input_tensor" : "input_tensor_" + std::to_string(i));
    }
  }
  return names;
}

bool facts_use_packed_contract(const ProcessCvuCanonicalFacts& facts) {
  auto has_duplicate_physical_indices = [](const auto& values) {
    std::vector<int> seen;
    for (const auto& value : values) {
      if (value.physical_index < 0) {
        continue;
      }
      if (std::find(seen.begin(), seen.end(), value.physical_index) != seen.end()) {
        return true;
      }
      seen.push_back(value.physical_index);
    }
    return false;
  };

  if (has_duplicate_physical_indices(facts.inputs) ||
      has_duplicate_physical_indices(facts.outputs)) {
    return true;
  }
  if (std::any_of(facts.inputs.begin(), facts.inputs.end(),
                  [](const auto& input) { return input.byte_offset != 0; })) {
    return true;
  }
  return std::any_of(facts.outputs.begin(), facts.outputs.end(), [](const auto& output) {
    return output.byte_offset != 0 ||
           output.representation != ProcessCvuOutputRepresentation::DenseTensor;
  });
}

std::string canonical_family_name(std::string graph_family) {
  graph_family = lower_copy(std::move(graph_family));
  if (graph_family == "preproc" || graph_family == "preprocess") {
    return "preproc";
  }
  if (graph_family == "quant" || graph_family == "quantize") {
    return "quantize";
  }
  if (graph_family == "quantizetensor" || graph_family == "quantize_tensor" ||
      graph_family == "quantizegeneric" || graph_family == "quantize_generic") {
    return "quantize";
  }
  if (graph_family == "quantizetessellate" || graph_family == "quantize_tessellate") {
    return "quanttess";
  }
  if (graph_family == "tess" || graph_family == "tessellate") {
    return "tessellate";
  }
  if (graph_family == "quanttess" || graph_family == "quant_tess") {
    return "quanttess";
  }
  if (graph_family == "casttess" || graph_family == "cast_tess" ||
      graph_family == "casttessellate") {
    return "casttess";
  }
  if (graph_family == "dequant" || graph_family == "dequantize") {
    return "dequantize";
  }
  if (graph_family == "detess" || graph_family == "detessellate") {
    return "detessellate";
  }
  if (graph_family == "detessdequant" || graph_family == "detess_dequant") {
    return "detessdequant";
  }
  if (graph_family == "detesscast" || graph_family == "detess_cast" ||
      graph_family == "detessellatecast") {
    return "detesscast";
  }
  return graph_family;
}

ProcessCvuGraphFamily family_enum_from_name(const std::string& graph_family) {
  const std::string family = canonical_family_name(graph_family);
  if (family == "preproc") {
    return ProcessCvuGraphFamily::Preproc;
  }
  if (family == "cast") {
    return ProcessCvuGraphFamily::Cast;
  }
  if (family == "quantize") {
    return ProcessCvuGraphFamily::Quant;
  }
  if (family == "tessellate") {
    return ProcessCvuGraphFamily::Tess;
  }
  if (family == "casttess") {
    return ProcessCvuGraphFamily::CastTess;
  }
  if (family == "quanttess") {
    return ProcessCvuGraphFamily::QuantTess;
  }
  if (family == "detessellate") {
    return ProcessCvuGraphFamily::Detess;
  }
  if (family == "dequantize") {
    return ProcessCvuGraphFamily::Dequant;
  }
  if (family == "detesscast") {
    return ProcessCvuGraphFamily::DetessCast;
  }
  if (family == "detessdequant") {
    return ProcessCvuGraphFamily::DetessDequant;
  }
  return ProcessCvuGraphFamily::Unknown;
}

template <typename T> std::vector<T> copy_numeric_vector(const std::vector<int>& values) {
  return std::vector<T>(values.begin(), values.end());
}

template <typename T>
std::vector<T> copy_numeric_vector64(const std::vector<std::int64_t>& values) {
  return std::vector<T>(values.begin(), values.end());
}

void synthesize_runtime_output_arrays_from_payload(ProcessCvuStagePayload* payload) {
  if (!payload) {
    return;
  }
  const std::string family = canonical_family_name(payload->graph_family);
  if (family == "preproc") {
    canonicalize_preproc_single_handoff_payload(payload);
    populate_preproc_payload_semantics(payload);
    return;
  }

  if (payload->output_shapes.empty()) {
    if (!payload->output_tensors.empty()) {
      const auto shape_int = shape_vec_from_tensor_desc_local(payload->output_tensors.front());
      if (!shape_int.empty()) {
        payload->output_shapes = {shape_int};
      }
    }
  }
  if (payload->output_shapes.empty()) {
    const TensorStaticSpec tensor = synthesize_single_io_output_tensor(*payload);
    std::vector<int> synthesized_shape;
    synthesized_shape.reserve(tensor.shape.size());
    for (const auto dim : tensor.shape) {
      synthesized_shape.push_back(static_cast<int>(dim));
    }
    if (!synthesized_shape.empty()) {
      payload->output_shapes = {std::move(synthesized_shape)};
    }
  }

  const std::size_t runtime_output_count = payload->output_shapes.size();
  if (runtime_output_count == 0U) {
    return;
  }

  auto first_output_tensor = [&]() -> TensorStaticSpec {
    return synthesize_single_io_output_tensor(*payload);
  };
  const TensorStaticSpec synthesized_output_tensor = first_output_tensor();

  if (payload->runtime_output_logical_index_list.empty()) {
    payload->runtime_output_logical_index_list.resize(runtime_output_count);
    for (std::size_t i = 0; i < runtime_output_count; ++i) {
      payload->runtime_output_logical_index_list[i] = static_cast<int>(i);
    }
  }
  if (payload->runtime_output_output_slot_list.empty()) {
    payload->runtime_output_output_slot_list.resize(runtime_output_count);
    for (std::size_t i = 0; i < runtime_output_count; ++i) {
      payload->runtime_output_output_slot_list[i] = static_cast<int>(i);
    }
  }
  if (payload->runtime_output_physical_index_list.empty()) {
    payload->runtime_output_physical_index_list.resize(runtime_output_count);
    for (std::size_t i = 0; i < runtime_output_count; ++i) {
      payload->runtime_output_physical_index_list[i] = static_cast<int>(i);
    }
  }
  if (payload->runtime_output_dtype_list.empty()) {
    payload->runtime_output_dtype_list.assign(runtime_output_count,
                                              synthesized_output_tensor.dtype);
  }
  if (payload->runtime_output_transport_kind_list.empty()) {
    payload->runtime_output_transport_kind_list.assign(runtime_output_count,
                                                       ProcessCvuOutputTransportKind::Dense);
  }
  if (payload->runtime_output_semantic_kind_list.empty()) {
    payload->runtime_output_semantic_kind_list.assign(runtime_output_count,
                                                      semantic_kind_from_family(family));
  }
  if (payload->runtime_output_logical_shapes.empty()) {
    payload->runtime_output_logical_shapes = payload->output_shapes;
  }
  if (payload->primary_output_transport_kind == ProcessCvuOutputTransportKind::Unknown &&
      !payload->runtime_output_transport_kind_list.empty()) {
    payload->primary_output_transport_kind = payload->runtime_output_transport_kind_list.front();
  }
  if (payload->primary_output_semantic_kind == ProcessCvuOutputSemanticKind::Unknown &&
      !payload->runtime_output_semantic_kind_list.empty()) {
    payload->primary_output_semantic_kind = payload->runtime_output_semantic_kind_list.front();
  }
}

TensorStaticSpec synthesize_single_io_input_tensor(const ProcessCvuStagePayload& payload) {
  const std::string dtype = payload.input_dtype.empty() ? std::string("INT8") : payload.input_dtype;
  std::string layout = payload_input_layout_token_local(payload);
  std::vector<std::int64_t> shape;
  if (!payload.input_tensors.empty()) {
    shape.assign(payload.input_tensors.front().shape.sizes,
                 payload.input_tensors.front().shape.sizes +
                     std::min<std::uint32_t>(payload.input_tensors.front().shape.rank,
                                             static_cast<std::uint32_t>(SIMA_EV_MAX_RANK)));
  } else if (!payload.input_shapes.empty()) {
    shape.assign(payload.input_shapes.front().begin(), payload.input_shapes.front().end());
  }
  if (shape.empty()) {
    throw std::runtime_error("processcvu single-io payload input shape missing");
  }
  const std::string& input_name = payload.default_input_name;
  return make_tensor_spec_from_shape(std::move(shape), dtype, layout, input_name, 0);
}

TensorStaticSpec synthesize_single_io_output_tensor(const ProcessCvuStagePayload& payload) {
  std::vector<std::int64_t> semantic_output_shape;
  if (!payload.runtime_output_logical_shapes.empty() &&
      !payload.runtime_output_logical_shapes.front().empty()) {
    semantic_output_shape.assign(payload.runtime_output_logical_shapes.front().begin(),
                                 payload.runtime_output_logical_shapes.front().end());
  } else if (!payload.output_tensors.empty()) {
    const auto shape = shape_vec_from_tensor_desc_local(payload.output_tensors.front());
    semantic_output_shape.assign(shape.begin(), shape.end());
  } else if (!payload.output_shapes.empty()) {
    semantic_output_shape.assign(payload.output_shapes.front().begin(),
                                 payload.output_shapes.front().end());
  }
  if (!semantic_output_shape.empty()) {
    std::string dtype =
        !payload.runtime_output_dtype_list.empty() &&
                !payload.runtime_output_dtype_list.front().empty()
            ? payload.runtime_output_dtype_list.front()
            : (payload.output_dtype.empty() ? payload.out_dtype : payload.output_dtype);
    if (dtype.empty()) {
      dtype = payload.input_dtype.empty() ? std::string("INT8") : payload.input_dtype;
    }
    std::string layout = payload_runtime_output_layout_token_local(payload);
    std::string output_name = payload.primary_output_name;
    if (output_name.empty() && !payload.default_output_names.empty()) {
      output_name = payload.default_output_names.front();
    }
    return make_tensor_spec_from_shape(std::move(semantic_output_shape), std::move(dtype),
                                       std::move(layout), std::move(output_name), 0);
  }
  throw std::runtime_error("processcvu single-io payload output shape missing");
}

} // namespace

static TensorStaticSpec synthesize_preproc_input_tensor(const ProcessCvuStagePayload& payload) {
  const auto authored_input_shape = [&]() -> std::vector<int> {
    if (!payload.input_shapes.empty() && !payload.input_shapes.front().empty()) {
      return payload.input_shapes.front();
    }
    if (!payload.input_tensors.empty()) {
      return shape_vec_from_tensor_desc_local(payload.input_tensors.front());
    }
    return {};
  }();
  const auto authored_input_dims = !authored_input_shape.empty()
                                       ? dims_from_shape_maybe_local(authored_input_shape)
                                       : PayloadInputDims{};
  const auto in_d = (authored_input_dims.width > 0 && authored_input_dims.height > 0)
                        ? authored_input_dims
                        : payload_input_dims_at(payload, 0);
  TensorStaticSpec input;
  input.tensor_index = 0;
  input.dtype = payload.input_dtype.empty() ? std::string("UINT8") : payload.input_dtype;
  input.layout = payload_input_layout_token_local(payload);
  input.max_w = in_d.width;
  input.max_h = in_d.height;
  input.semantic_tag =
      payload.default_input_name.empty() ? std::string("input_image") : payload.default_input_name;
  const std::string input_img_type = upper_copy_local(payload.input_img_type);
  const bool planar_yuv_input =
      input_img_type == "NV12" || input_img_type == "IYUV" || input_img_type == "I420";
  const bool gray_input = input_img_type == "GRAY" || input_img_type == "GRAY8";
  if (in_d.width > 0 && in_d.height > 0 && planar_yuv_input) {
    input.max_h = in_d.height + (in_d.height / 2);
    input.shape = {input.max_h, in_d.width};
    return input;
  }
  if (in_d.width > 0 && in_d.height > 0 && gray_input) {
    input.shape = {in_d.height, in_d.width};
    return input;
  }
  if (!authored_input_shape.empty()) {
    input.shape.assign(authored_input_shape.begin(), authored_input_shape.end());
    return input;
  }
  if (in_d.width > 0 && in_d.height > 0 && in_d.channels > 0) {
    const std::string layout = normalize_layout(input.layout);
    if (layout == "HW") {
      input.shape = {in_d.height, in_d.width};
    } else if (layout == "CHW") {
      input.shape = {in_d.channels, in_d.height, in_d.width};
    } else if (layout == "HWC") {
      input.shape = {in_d.height, in_d.width, in_d.channels};
    } else {
      throw std::runtime_error("processcvu preproc input tensor requires explicit layout");
    }
  }
  return input;
}

static std::vector<std::string> default_preproc_runtime_output_names() {
  return {"output_rgb_image", "output_tessellated_image"};
}

CompiledProcessCvuContract
build_processcvu_compiled_contract(const ProcessCvuCanonicalCompileInputs& inputs) {
  return build_processcvu_compiled_contract_from_facts(inputs.payload, inputs.facts);
}

CompiledProcessCvuContract build_processcvu_mpk_preadapter_compiled_contract_for_stage_kind(
    const MpkContract& contract, ::simaai::neat::internal::ExecutionStageKind stage_kind,
    const std::optional<std::string>& exact_stage_name_or_id,
    const std::optional<std::string>& canonical_handoff_segment_name) {
  using ::simaai::neat::internal::ExecutionStageKind;
  std::string graph_family;
  switch (stage_kind) {
  case ExecutionStageKind::Quant:
    graph_family = "quantize";
    break;
  case ExecutionStageKind::Tess:
    graph_family = "tessellate";
    break;
  case ExecutionStageKind::QuantTess:
    graph_family = "quanttess";
    break;
  case ExecutionStageKind::Cast:
    graph_family = "cast";
    break;
  case ExecutionStageKind::CastTess:
    graph_family = "casttess";
    break;
  default:
    throw std::runtime_error(
        "ProcessCvuStageSemantics: requested stage is not a pre-adapter processcvu family");
  }
  return build_processcvu_compiled_contract(build_processcvu_mpk_preadapter_compile_inputs_local(
      contract, graph_family, exact_stage_name_or_id, canonical_handoff_segment_name));
}

CompiledProcessCvuContract build_processcvu_mpk_compiled_contract_for_stage_kind(
    const MpkContract& contract, ::simaai::neat::internal::ExecutionStageKind stage_kind,
    const std::optional<std::string>& exact_stage_name_or_id,
    const std::optional<std::string>& canonical_handoff_segment_name,
    const std::optional<bool>& preproc_single_output_handoff, const std::string& input_format,
    int input_depth, int max_input_width, int max_input_height, bool normalize,
    const std::vector<float>& mean, const std::vector<float>& stddev) {
  using ::simaai::neat::internal::ExecutionStageKind;

  switch (stage_kind) {
  case ExecutionStageKind::Preproc:
    if (!preproc_single_output_handoff.has_value()) {
      throw std::runtime_error("ProcessCvuStageSemantics: preproc compiled contract requires "
                               "single-output handoff fact");
    }
    return build_processcvu_compiled_contract(build_processcvu_mpk_preproc_compile_inputs_local(
        contract, input_format, input_depth, max_input_width, max_input_height, normalize, mean,
        stddev, *preproc_single_output_handoff));
  case ExecutionStageKind::Detess:
    return build_processcvu_compiled_contract(
        build_processcvu_mpk_detess_compile_inputs_local(contract));
  case ExecutionStageKind::DetessCast:
    return build_processcvu_compiled_contract(
        build_processcvu_mpk_detesscast_compile_inputs_local(contract));
  case ExecutionStageKind::Dequant:
    return build_processcvu_compiled_contract(
        build_processcvu_mpk_dequant_compile_inputs_local(contract));
  case ExecutionStageKind::DetessDequant:
    return build_processcvu_compiled_contract(
        build_processcvu_mpk_detessdequant_compile_inputs_local(contract));
  case ExecutionStageKind::Cast: {
    const auto post_cast_stages = collect_post_stages_for_kind_names_local(contract, {"cast"});
    if (!post_cast_stages.empty()) {
      return build_processcvu_compiled_contract(
          build_processcvu_mpk_cast_compile_inputs_local(contract));
    }
    return build_processcvu_compiled_contract(build_processcvu_mpk_preadapter_compile_inputs_local(
        contract, "cast", exact_stage_name_or_id, canonical_handoff_segment_name));
  }
  case ExecutionStageKind::Quant:
  case ExecutionStageKind::Tess:
  case ExecutionStageKind::QuantTess:
  case ExecutionStageKind::CastTess: {
    const std::string graph_family =
        pipeline_internal::sima::processcvu_graph_family_for_stage_kind(stage_kind);
    if (graph_family.empty()) {
      break;
    }
    return build_processcvu_compiled_contract(build_processcvu_mpk_preadapter_compile_inputs_local(
        contract, graph_family, exact_stage_name_or_id, canonical_handoff_segment_name));
  }
  case ExecutionStageKind::Mla:
  case ExecutionStageKind::BoxDecode:
  case ExecutionStageKind::Unknown:
    break;
  }

  throw std::runtime_error(
      "ProcessCvuStageSemantics: unsupported execution stage kind for processcvu contract");
}

std::string
resolve_preproc_primary_output_name(const std::vector<std::string>& runtime_output_names,
                                    bool tessellate,
                                    const std::string& requested_primary_output_name) {
  const auto contains_name = [&](const char* wanted) {
    return std::any_of(runtime_output_names.begin(), runtime_output_names.end(),
                       [&](const std::string& candidate) { return candidate == wanted; });
  };
  if (is_known_preproc_output_name(requested_primary_output_name) &&
      contains_name(requested_primary_output_name.c_str())) {
    return requested_primary_output_name;
  }
  if (tessellate && contains_name("output_tessellated_image")) {
    return "output_tessellated_image";
  }
  if (!tessellate && contains_name("output_rgb_image")) {
    return "output_rgb_image";
  }
  for (const auto& name : runtime_output_names) {
    if (!name.empty()) {
      return name;
    }
  }
  return tessellate ? "output_tessellated_image" : "output_rgb_image";
}

struct MpkTensorDims {
  int width = 0;
  int height = 0;
  int depth = 0;
  std::string layout;
};

struct DetessPackedInputDims {
  int width = 0;
  int height = 0;
  int depth = 0;
  int channels = 0;
  std::string layout;
};

MpkTensorDims dims_from_tensor_shape_local(std::vector<std::int64_t> shape) {
  shape = semantic_shape_without_batch_local(std::move(shape));

  MpkTensorDims out;
  if (shape.size() >= 3U) {
    out.height = static_cast<int>(shape[shape.size() - 3U]);
    out.width = static_cast<int>(shape[shape.size() - 2U]);
    out.depth = static_cast<int>(shape[shape.size() - 1U]);
    return out;
  }
  if (shape.size() == 2U) {
    out.height = static_cast<int>(shape[0]);
    out.width = static_cast<int>(shape[1]);
    out.depth = 1;
    return out;
  }
  if (shape.size() == 1U) {
    out.width = static_cast<int>(shape[0]);
    out.height = 1;
    out.depth = 1;
  }
  return out;
}

MpkTensorDims dims_from_mpk_shape(std::vector<std::int64_t> shape) {
  const bool leading_batch_feature = shape.size() >= 4U && shape.front() == 1;
  shape = semantic_shape_without_batch_local(std::move(shape));

  MpkTensorDims out;
  if (leading_batch_feature && shape.size() >= 3U) {
    out.depth = static_cast<int>(shape[0]);
    out.height = static_cast<int>(shape[1]);
    out.width = static_cast<int>(shape[2]);
    return out;
  }
  if (shape.size() >= 3U) {
    out.height = static_cast<int>(shape[shape.size() - 3U]);
    out.width = static_cast<int>(shape[shape.size() - 2U]);
    out.depth = static_cast<int>(shape[shape.size() - 1U]);
    return out;
  }
  if (shape.size() == 2U) {
    out.height = static_cast<int>(shape[0]);
    out.width = static_cast<int>(shape[1]);
    out.depth = 1;
    return out;
  }
  if (shape.size() == 1U) {
    out.width = static_cast<int>(shape[0]);
    out.height = 1;
    out.depth = 1;
  }
  return out;
}

std::string processcvu_output_name_from_mpk_tensor(const MpkTensorContract& tensor,
                                                   std::size_t index) {
  (void)index;
  if (!tensor.name.empty()) {
    return tensor.name;
  }
  if (!tensor.segment_name.empty()) {
    return tensor.segment_name;
  }
  return {};
}

std::string processcvu_physical_output_name_from_mpk_tensor(const MpkTensorContract& tensor,
                                                            const std::string& fallback) {
  if (!tensor.segment_name.empty()) {
    return tensor.segment_name;
  }
  if (!tensor.name.empty()) {
    return tensor.name;
  }
  return fallback;
}

std::string processcvu_input_name_from_mpk_tensor(const MpkTensorContract& tensor,
                                                  const std::string& graph_family) {
  (void)graph_family;
  if (!tensor.name.empty()) {
    return tensor.name;
  }
  if (!tensor.segment_name.empty()) {
    return tensor.segment_name;
  }
  return {};
}

ProcessCvuSingleOutputIdentity
build_processcvu_single_output_identity_local(const std::string& runtime_output_name,
                                              const std::string& physical_output_name,
                                              const std::string& published_output_name) {
  ProcessCvuSingleOutputIdentity id;
  id.runtime_output_name = runtime_output_name;
  id.physical_output_name =
      !physical_output_name.empty() ? physical_output_name : runtime_output_name;
  id.published_output_name =
      !published_output_name.empty() ? published_output_name : runtime_output_name;
  return id;
}

void apply_processcvu_single_output_identity_local(CompiledProcessCvuRuntimeConfig* runtime,
                                                   const ProcessCvuSingleOutputIdentity& id) {
  if (!runtime) {
    return;
  }
  runtime->runtime_output_names = {id.runtime_output_name};
  runtime->physical_output_names = {id.physical_output_name};
  runtime->published_output_names = {id.published_output_name};
  runtime->primary_output_name = id.published_output_name;
}

void apply_processcvu_single_output_identity_local(ProcessCvuSingleOutputFactsSpec* facts_spec,
                                                   const ProcessCvuSingleOutputIdentity& id) {
  if (!facts_spec) {
    return;
  }
  facts_spec->physical_output_name = id.physical_output_name;
  facts_spec->logical_output_name = id.published_output_name;
  facts_spec->published_output_names = {id.published_output_name};
  facts_spec->primary_output_name = id.published_output_name;
}

static std::vector<std::string>
build_preproc_internal_output_names(const ProcessCvuStagePayload& payload) {
  std::vector<std::string> names = !payload.default_output_names.empty()
                                       ? payload.default_output_names
                                       : default_preproc_runtime_output_names();
  if (payload.tessellate != 1) {
    names.erase(std::remove(names.begin(), names.end(), "output_tessellated_image"), names.end());
  }
  return names;
}

static std::vector<std::string>
build_preproc_runtime_output_names(const ProcessCvuStagePayload& payload) {
  return build_preproc_internal_output_names(payload);
}

static std::optional<std::size_t>
find_preproc_internal_output_index(const ProcessCvuStagePayload& payload,
                                   const std::string& output_name) {
  const auto internal_output_names = build_preproc_internal_output_names(payload);
  const auto it =
      std::find(internal_output_names.begin(), internal_output_names.end(), output_name);
  if (it == internal_output_names.end()) {
    return std::nullopt;
  }
  return static_cast<std::size_t>(std::distance(internal_output_names.begin(), it));
}

std::vector<TensorStaticSpec>
synthesize_preproc_runtime_outputs(const ProcessCvuStagePayload& payload) {
  const std::vector<std::string> ordered_outputs = build_preproc_runtime_output_names(payload);
  std::vector<TensorStaticSpec> outputs;
  outputs.reserve(ordered_outputs.size());

  std::string dtype = payload.output_dtype.empty() ? payload.out_dtype : payload.output_dtype;
  if (dtype.empty()) {
    dtype = payload.input_dtype;
  }

  int tensor_index = 0;
  for (std::size_t i = 0; i < ordered_outputs.size(); ++i) {
    const auto& output_name = ordered_outputs[i];
    std::vector<std::int64_t> output_shape;
    if (i < payload.runtime_output_logical_shapes.size() &&
        !payload.runtime_output_logical_shapes[i].empty()) {
      output_shape.assign(payload.runtime_output_logical_shapes[i].begin(),
                          payload.runtime_output_logical_shapes[i].end());
    } else if (i < payload.output_tensors.size()) {
      const auto shape = shape_vec_from_tensor_desc_local(payload.output_tensors[i]);
      output_shape.assign(shape.begin(), shape.end());
    } else if (i < payload.output_shapes.size()) {
      output_shape.assign(payload.output_shapes[i].begin(), payload.output_shapes[i].end());
    } else if (!payload.output_shapes.empty()) {
      output_shape.assign(payload.output_shapes.front().begin(),
                          payload.output_shapes.front().end());
    } else if (!payload.runtime_output_logical_shapes.empty() &&
               !payload.runtime_output_logical_shapes.front().empty()) {
      output_shape.assign(payload.runtime_output_logical_shapes.front().begin(),
                          payload.runtime_output_logical_shapes.front().end());
    }
    if (output_shape.empty()) {
      throw std::runtime_error("processcvu preproc runtime output shape missing for '" +
                               output_name + "'");
    }

    std::string output_layout = i < payload.runtime_output_logical_layout_list.size() &&
                                        !payload.runtime_output_logical_layout_list[i].empty()
                                    ? payload.runtime_output_logical_layout_list[i]
                                    : payload_runtime_output_layout_token_local(payload, i);
    outputs.push_back(make_tensor_spec_from_shape(std::move(output_shape), dtype, output_layout,
                                                  output_name, tensor_index++));
  }

  return outputs;
}

void canonicalize_preproc_single_handoff_payload(ProcessCvuStagePayload* payload) {
  if (!payload || canonical_family_name(payload->graph_family) != "preproc") {
    return;
  }
  require_supported_preproc_single_output_handoff(payload->preproc_single_output_handoff);
  if (payload->default_output_names.empty()) {
    payload->default_output_names = default_preproc_runtime_output_names();
  }
  if (payload->primary_output_name.empty()) {
    throw std::invalid_argument(
        "processcvu preproc single-output handoff payload missing primary_output_name");
  }

  const auto output_it =
      std::find(payload->default_output_names.begin(), payload->default_output_names.end(),
                payload->primary_output_name);
  if (output_it == payload->default_output_names.end()) {
    throw std::invalid_argument("processcvu preproc single-output handoff payload "
                                "primary_output_name must name a runtime output");
  }

  const std::size_t selected_index =
      static_cast<std::size_t>(std::distance(payload->default_output_names.begin(), output_it));
  if (selected_index < payload->runtime_output_transport_kind_list.size()) {
    payload->primary_output_transport_kind =
        payload->runtime_output_transport_kind_list[selected_index];
  }
  if (selected_index < payload->runtime_output_semantic_kind_list.size()) {
    payload->primary_output_semantic_kind =
        payload->runtime_output_semantic_kind_list[selected_index];
  }
}

std::uint64_t synthesize_preproc_packed_output_size_bytes(const ProcessCvuStagePayload& payload,
                                                          const std::string& dtype) {
  std::vector<std::int64_t> shape;
  if (const auto index = find_preproc_internal_output_index(payload, "output_tessellated_image");
      index.has_value()) {
    const std::size_t i = *index;
    if (i < payload.runtime_output_logical_shapes.size() &&
        !payload.runtime_output_logical_shapes[i].empty()) {
      shape.assign(payload.runtime_output_logical_shapes[i].begin(),
                   payload.runtime_output_logical_shapes[i].end());
    } else if (i < payload.output_tensors.size()) {
      const auto shape_int = shape_vec_from_tensor_desc_local(payload.output_tensors[i]);
      shape.assign(shape_int.begin(), shape_int.end());
    } else if (i < payload.output_shapes.size()) {
      shape.assign(payload.output_shapes[i].begin(), payload.output_shapes[i].end());
    }
  }
  if (shape.empty() && !payload.runtime_output_logical_shapes.empty() &&
      !payload.runtime_output_logical_shapes.front().empty()) {
    shape.assign(payload.runtime_output_logical_shapes.front().begin(),
                 payload.runtime_output_logical_shapes.front().end());
  }
  if (shape.empty() && !payload.output_shapes.empty()) {
    shape.assign(payload.output_shapes.front().begin(), payload.output_shapes.front().end());
  }
  if (shape.empty()) {
    throw std::runtime_error("processcvu preproc packed output shape missing");
  }
  return specbuilders::tensor_size_bytes_from_shape_dtype(shape, dtype);
}

std::uint64_t processcvu_dtype_size_bytes_from_token(const std::string& raw_dtype) {
  return specbuilders::dtype_size_bytes_from_token(raw_dtype);
}

std::vector<std::int64_t>
contiguous_stride_bytes_for_shape_local(const std::vector<std::int64_t>& shape,
                                        std::uint64_t elem_bytes) {
  if (shape.empty() || elem_bytes == 0U) {
    return {};
  }
  std::vector<std::int64_t> stride_bytes(shape.size(), 0);
  std::uint64_t running = elem_bytes;
  for (std::size_t idx = shape.size(); idx-- > 0;) {
    if (running > static_cast<std::uint64_t>(std::numeric_limits<std::int64_t>::max())) {
      throw std::runtime_error("processcvu contiguous stride exceeds int64 range");
    }
    stride_bytes[idx] = static_cast<std::int64_t>(running);
    const std::int64_t dim = shape[idx];
    if (dim <= 0) {
      return {};
    }
    if (running > std::numeric_limits<std::uint64_t>::max() / static_cast<std::uint64_t>(dim)) {
      throw std::runtime_error("processcvu contiguous stride overflow");
    }
    running *= static_cast<std::uint64_t>(dim);
  }
  return stride_bytes;
}

std::uint64_t multiply_u64_checked_local(std::uint64_t lhs, std::uint64_t rhs,
                                         const char* context) {
  if (lhs == 0U || rhs == 0U) {
    return 0U;
  }
  if (lhs > std::numeric_limits<std::uint64_t>::max() / rhs) {
    throw std::runtime_error(std::string("processcvu overflow while computing ") + context);
  }
  return lhs * rhs;
}

std::uint64_t detessdequant_padded_output_size_bytes_local(int input_width, int input_height,
                                                           int input_depth, int input_channels,
                                                           const std::string& output_dtype) {
  if (input_width <= 0 || input_height <= 0 || input_depth <= 0 || input_channels <= 0 ||
      output_dtype.empty()) {
    return 0U;
  }
  const std::uint64_t elem_bytes = processcvu_dtype_size_bytes_from_token(output_dtype);
  if (elem_bytes == 0U) {
    return 0U;
  }
  std::uint64_t total = multiply_u64_checked_local(static_cast<std::uint64_t>(input_width),
                                                   static_cast<std::uint64_t>(input_height),
                                                   "detessdequant padded output bytes");
  total = multiply_u64_checked_local(total, static_cast<std::uint64_t>(input_depth),
                                     "detessdequant padded output bytes");
  total = multiply_u64_checked_local(total, static_cast<std::uint64_t>(input_channels),
                                     "detessdequant padded output bytes");
  return multiply_u64_checked_local(total, elem_bytes, "detessdequant padded output bytes");
}

std::vector<std::int64_t>
detessdequant_padded_output_stride_bytes_local(const LogicalTensorStaticSpec& logical,
                                               int padded_channels) {
  const std::uint64_t elem_bytes = processcvu_dtype_size_bytes_from_token(logical.dtype);
  if (logical.shape.empty() || elem_bytes == 0U) {
    return {};
  }
  const std::string layout = upper_copy_local(logical.layout);
  const bool collapsed_single_channel_hw = logical.shape.size() >= 3U && layout == "HW" &&
                                           logical.shape.back() == 1 &&
                                           padded_channels > logical.shape.back();
  if (padded_channels <= 0 || logical.shape.size() < 3U ||
      ((layout != "HWC" && layout != "NHWC") && !collapsed_single_channel_hw)) {
    return contiguous_stride_bytes_for_shape_local(logical.shape, elem_bytes);
  }

  std::vector<std::int64_t> stride_bytes(logical.shape.size(), 0);
  std::uint64_t running =
      multiply_u64_checked_local(static_cast<std::uint64_t>(padded_channels), elem_bytes,
                                 "detessdequant padded output stride");
  if (running > static_cast<std::uint64_t>(std::numeric_limits<std::int64_t>::max())) {
    throw std::runtime_error("processcvu detessdequant padded stride exceeds int64 range");
  }
  stride_bytes[logical.shape.size() - 1U] = static_cast<std::int64_t>(elem_bytes);
  stride_bytes[logical.shape.size() - 2U] = static_cast<std::int64_t>(running);
  for (std::size_t idx = logical.shape.size() - 2U; idx-- > 0;) {
    const std::int64_t dim = logical.shape[idx + 1U];
    if (dim <= 0) {
      return {};
    }
    running = multiply_u64_checked_local(running, static_cast<std::uint64_t>(dim),
                                         "detessdequant padded output stride");
    if (running > static_cast<std::uint64_t>(std::numeric_limits<std::int64_t>::max())) {
      throw std::runtime_error("processcvu detessdequant padded stride exceeds int64 range");
    }
    stride_bytes[idx] = static_cast<std::int64_t>(running);
  }
  return stride_bytes;
}

std::optional<int>
detessdequant_infer_padded_channels_from_input_local(const LogicalInputStaticSpec& input,
                                                     const ShapeDims& logical_dims) {
  if (input.size_bytes == 0U || input.dtype.empty() || logical_dims.width <= 0 ||
      logical_dims.height <= 0 || logical_dims.depth <= 0 || logical_dims.channels <= 0) {
    return std::nullopt;
  }

  const std::uint64_t elem_bytes = processcvu_dtype_size_bytes_from_token(input.dtype);
  if (elem_bytes == 0U || (input.size_bytes % elem_bytes) != 0U) {
    return std::nullopt;
  }

  std::uint64_t spatial_elems = multiply_u64_checked_local(
      static_cast<std::uint64_t>(logical_dims.width),
      static_cast<std::uint64_t>(logical_dims.height), "detessdequant padded channel inference");
  spatial_elems =
      multiply_u64_checked_local(spatial_elems, static_cast<std::uint64_t>(logical_dims.depth),
                                 "detessdequant padded channel inference");
  if (spatial_elems == 0U) {
    return std::nullopt;
  }

  const std::uint64_t dense_elems =
      multiply_u64_checked_local(spatial_elems, static_cast<std::uint64_t>(logical_dims.channels),
                                 "detessdequant padded channel inference");
  const std::uint64_t dense_size_bytes =
      multiply_u64_checked_local(dense_elems, elem_bytes, "detessdequant padded channel inference");
  if (input.size_bytes <= dense_size_bytes) {
    return std::nullopt;
  }

  const std::uint64_t transport_elems = input.size_bytes / elem_bytes;
  if ((transport_elems % spatial_elems) != 0U) {
    return std::nullopt;
  }

  const std::uint64_t inferred_channels = transport_elems / spatial_elems;
  if (inferred_channels <= static_cast<std::uint64_t>(logical_dims.channels) ||
      inferred_channels > static_cast<std::uint64_t>(std::numeric_limits<int>::max())) {
    return std::nullopt;
  }

  return static_cast<int>(inferred_channels);
}

std::uint64_t processcvu_tensor_size_bytes_from_spec(const TensorStaticSpec& tensor) {
  if (!tensor.shape.empty()) {
    return specbuilders::tensor_size_bytes_from_shape_dtype(tensor.shape, tensor.dtype);
  }
  if (tensor.max_w > 0 && tensor.max_h > 0) {
    const std::uint64_t depth = (upper_copy_local(tensor.layout) == "HW") ? 1U : 3U;
    return static_cast<std::uint64_t>(tensor.max_w) * static_cast<std::uint64_t>(tensor.max_h) *
           depth * processcvu_dtype_size_bytes_from_token(tensor.dtype);
  }
  return 0U;
}

static ProcessCvuCanonicalInputFact build_dense_processcvu_input_fact(
    int logical_index, int physical_index, const std::string& physical_name,
    const std::vector<std::int64_t>& shape, const std::string& dtype, const std::string& layout) {
  ProcessCvuCanonicalInputFact fact;
  fact.logical_index = logical_index;
  fact.physical_index = physical_index;
  fact.physical_name = physical_name;
  fact.logical_name = physical_name;
  fact.shape = shape;
  fact.size_bytes = specbuilders::tensor_size_bytes_from_shape_dtype(fact.shape, dtype);
  fact.dtype = dtype;
  fact.layout = layout;
  return fact;
}

static std::optional<QuantStaticSpec>
build_runtime_input_quant_spec_from_payload(const ProcessCvuStagePayload& payload,
                                            std::size_t input_index) {
  std::vector<double> scales;
  std::vector<std::int64_t> zero_points;

  if (!payload.q_scale_list.empty()) {
    if (payload.q_scale_list.size() == 1U) {
      scales = {payload.q_scale_list.front()};
    } else if (input_index < payload.q_scale_list.size()) {
      scales = {payload.q_scale_list[input_index]};
    }
  } else if (payload.has_q_scale) {
    scales = {payload.q_scale};
  }

  if (!payload.q_zp_list.empty()) {
    if (payload.q_zp_list.size() == 1U) {
      zero_points = {payload.q_zp_list.front()};
    } else if (input_index < payload.q_zp_list.size()) {
      zero_points = {payload.q_zp_list[input_index]};
    }
  } else if (payload.has_q_zp) {
    zero_points = {payload.q_zp};
  }

  if (scales.empty() || zero_points.empty()) {
    return std::nullopt;
  }

  QuantStaticSpec quant;
  quant.granularity = QuantGranularity::PerTensor;
  quant.axis = -1;
  quant.scales = std::move(scales);
  quant.zero_points = std::move(zero_points);
  return quant;
}

static ProcessCvuCanonicalInputFact
build_packed_processcvu_input_fact(int logical_index, int physical_index,
                                   const std::string& physical_name,
                                   const MpkTensorContract& tensor, const std::string& dtype,
                                   const std::string& layout, std::int64_t byte_offset) {
  const auto materialization_kind_from_tensor = [](const MpkTensorContract& source) {
    switch (source.materialization_kind) {
    case MpkTensorMaterializationKind::OffsetView:
      return TensorMaterializationKind::OffsetView;
    case MpkTensorMaterializationKind::Bf16LaneSplitRepack:
      return TensorMaterializationKind::Bf16LaneSplitRepack;
    case MpkTensorMaterializationKind::Unknown:
    case MpkTensorMaterializationKind::Direct:
    default:
      return TensorMaterializationKind::Direct;
    }
  };
  ProcessCvuCanonicalInputFact fact;
  fact.logical_index = logical_index;
  fact.physical_index = physical_index;
  fact.physical_name = physical_name;
  fact.logical_name = processcvu_mpk_tensor_name(tensor, physical_name);
  fact.shape = preferred_packed_mpk_tensor_shape_local(tensor);
  fact.size_bytes = logical_mpk_tensor_size_bytes_local(tensor, dtype);
  fact.dtype = dtype;
  fact.layout = layout;
  fact.byte_offset = byte_offset;
  fact.materialization_kind = materialization_kind_from_tensor(tensor);
  return fact;
}

static ProcessCvuCanonicalBindingFact build_packed_processcvu_input_binding_fact(
    int local_logical_input_index, const std::string& cm_input_name,
    const MpkTensorContract& upstream_tensor, const std::string& dtype, int src_output_slot,
    int src_physical_output_index) {
  ProcessCvuCanonicalBindingFact fact;
  fact.local_logical_input_index = local_logical_input_index;
  fact.src_logical_output_index = local_logical_input_index;
  fact.src_output_slot = src_output_slot;
  fact.src_physical_output_index = src_physical_output_index;
  fact.src_physical_size_bytes = preferred_mpk_tensor_size_bytes_local(upstream_tensor, dtype);
  fact.required = true;
  fact.cm_input_name = cm_input_name;
  fact.source_segment_name = processcvu_mpk_tensor_name(
      upstream_tensor, cm_input_name + "_" + std::to_string(local_logical_input_index));
  return fact;
}

static ProcessCvuCanonicalOutputFact build_packed_processcvu_output_fact(
    int logical_index, int output_slot, int physical_index, const std::string& physical_name,
    const MpkTensorContract& tensor, const std::string& logical_name, const std::string& dtype,
    const std::string& layout, std::int64_t byte_offset) {
  ProcessCvuCanonicalOutputFact fact;
  fact.representation = ProcessCvuOutputRepresentation::PackedTensor;
  fact.logical_index = logical_index;
  fact.output_slot = output_slot;
  fact.physical_index = physical_index;
  fact.tensor_index = logical_index;
  fact.physical_name = physical_name;
  fact.logical_name = logical_name;
  fact.shape = preferred_packed_mpk_tensor_shape_local(tensor);
  fact.dtype = dtype;
  fact.layout = layout;
  fact.byte_offset = byte_offset;
  fact.size_bytes = logical_mpk_tensor_size_bytes_local(tensor, dtype);
  if (processcvu_detess_layout_debug_enabled()) {
    std::fprintf(
        stderr,
        "[detess-layout-debug] where=build_packed_output_fact logical=%d slot=%d physical=%d "
        "layout_arg=%s dtype_arg=%s logical_name=%s physical_name=%s tensor_name=%s "
        "tensor_logical_shape=%s tensor_mpk_shape=%s packed_shape=%s byte_offset=%" PRId64 "\n",
        logical_index, output_slot, physical_index, layout.c_str(), dtype.c_str(),
        logical_name.c_str(), physical_name.c_str(), tensor.name.c_str(),
        join_i64_debug_processcvu_local(tensor.logical_shape).c_str(),
        join_i64_debug_processcvu_local(tensor.mpk_shape).c_str(),
        join_i64_debug_processcvu_local(fact.shape).c_str(), byte_offset);
  }
  return fact;
}

static ProcessCvuCanonicalOutputFact build_packed_blob_processcvu_output_fact(
    int logical_index, int output_slot, int physical_index, const std::string& physical_name,
    const std::string& logical_name, const std::vector<std::int64_t>& logical_shape,
    const std::string& dtype, const std::string& layout, std::uint64_t size_bytes,
    std::int64_t byte_offset) {
  ProcessCvuCanonicalOutputFact fact;
  fact.representation = ProcessCvuOutputRepresentation::PackedBlob;
  fact.logical_index = logical_index;
  fact.output_slot = output_slot;
  fact.physical_index = physical_index;
  fact.tensor_index = logical_index;
  fact.physical_name = physical_name;
  fact.logical_name = logical_name;
  fact.shape = logical_shape;
  fact.dtype = dtype;
  fact.layout = layout;
  fact.byte_offset = byte_offset;
  fact.size_bytes = size_bytes;
  return fact;
}

static ProcessCvuCanonicalRouteFact
build_processcvu_published_output_route_fact(const ProcessCvuCanonicalOutputFact& output) {
  ProcessCvuCanonicalRouteFact route;
  route.output_slot = output.output_slot;
  route.logical_output_index = output.logical_index;
  route.tensor_index = output.tensor_index;
  route.cm_output_name = output.logical_name;
  route.segment_name = output.physical_name;
  return route;
}

static ProcessCvuCanonicalFacts
build_processcvu_single_output_facts(const ProcessCvuSingleOutputFactsSpec& spec) {
  if (spec.physical_input_name.empty()) {
    throw std::invalid_argument("processcvu single-output facts require physical_input_name");
  }
  if (spec.physical_output_name.empty()) {
    throw std::invalid_argument("processcvu single-output facts require physical_output_name");
  }
  if (spec.logical_output_name.empty()) {
    throw std::invalid_argument("processcvu single-output facts require logical_output_name");
  }
  if (spec.input_shape.empty()) {
    throw std::invalid_argument("processcvu single-output facts require input_shape");
  }

  ProcessCvuCanonicalFacts facts;
  facts.physical_input_names = {spec.physical_input_name};
  facts.physical_output_names = {spec.physical_output_name};
  facts.primary_output_name =
      !spec.primary_output_name.empty() ? spec.primary_output_name : spec.logical_output_name;
  facts.published_output_names = !spec.published_output_names.empty()
                                     ? spec.published_output_names
                                     : std::vector<std::string>{facts.primary_output_name};
  if (processcvu_tess_segment_debug_enabled()) {
    std::fprintf(
        stderr,
        "[tess-segment-debug] where=single_output_facts physical_input=%s physical_output=%s "
        "logical_output=%s primary_output=%s published_outputs=%zu representation=%d "
        "output_dtype=%s output_layout=%s packed_bytes=%" PRIu64 "\n",
        spec.physical_input_name.c_str(), spec.physical_output_name.c_str(),
        spec.logical_output_name.c_str(), facts.primary_output_name.c_str(),
        facts.published_output_names.size(), static_cast<int>(spec.output_representation),
        spec.output_dtype.c_str(), spec.output_layout.c_str(), spec.packed_output_size_bytes);
  }
  facts.inputs.push_back(build_dense_processcvu_input_fact(
      0, 0, spec.physical_input_name, spec.input_shape, spec.input_dtype, spec.input_layout));

  ProcessCvuCanonicalBindingFact binding;
  binding.sink_pad_index = 0;
  binding.local_logical_input_index = 0;
  binding.required = true;
  binding.cm_input_name = spec.physical_input_name;
  binding.source_segment_name =
      !spec.source_segment_name.empty() ? spec.source_segment_name : spec.physical_input_name;
  facts.input_bindings.push_back(std::move(binding));

  switch (spec.output_representation) {
  case ProcessCvuOutputRepresentation::PackedBlob:
    facts.outputs.push_back(build_packed_blob_processcvu_output_fact(
        0, 0, 0, spec.physical_output_name, spec.logical_output_name, spec.output_shape,
        spec.output_dtype, spec.output_layout, spec.packed_output_size_bytes, 0));
    break;
  case ProcessCvuOutputRepresentation::PackedTensor:
    if (!spec.packed_output_tensor) {
      throw std::invalid_argument(
          "processcvu single-output packed-tensor facts require packed_output_tensor");
    }
    facts.outputs.push_back(build_packed_processcvu_output_fact(
        0, 0, 0, spec.physical_output_name, *spec.packed_output_tensor, spec.logical_output_name,
        spec.output_dtype, spec.output_layout, 0));
    break;
  case ProcessCvuOutputRepresentation::DenseTensor: {
    if (spec.output_shape.empty()) {
      throw std::invalid_argument("processcvu single-output dense facts require output_shape");
    }
    ProcessCvuCanonicalOutputFact output;
    output.representation = ProcessCvuOutputRepresentation::DenseTensor;
    output.logical_index = 0;
    output.physical_index = 0;
    output.output_slot = 0;
    output.tensor_index = 0;
    output.physical_name = spec.physical_output_name;
    output.logical_name = spec.logical_output_name;
    output.shape = spec.output_shape;
    output.dtype = spec.output_dtype;
    output.layout = spec.output_layout;
    facts.outputs.push_back(std::move(output));
    break;
  }
  }

  facts.output_order.push_back(build_processcvu_published_output_route_fact(facts.outputs.front()));
  return facts;
}

static ProcessCvuCanonicalFacts build_processcvu_packed_route_facts(
    const std::string& physical_input_name, const std::string& physical_output_name,
    const std::vector<ProcessCvuPackedRouteEntry>& entries, const std::string& primary_output_name,
    const std::vector<std::string>& published_output_names) {
  if (physical_input_name.empty() || physical_output_name.empty()) {
    throw std::invalid_argument("processcvu packed-route facts require physical buffer names");
  }
  ProcessCvuCanonicalFacts facts;
  facts.physical_input_names = {physical_input_name};
  facts.physical_output_names = {physical_output_name};
  facts.primary_output_name = primary_output_name;
  facts.published_output_names = published_output_names;
  facts.inputs.reserve(entries.size());
  facts.input_bindings.reserve(entries.size());
  facts.outputs.reserve(entries.size());
  facts.output_order.reserve(entries.size());

  for (std::size_t i = 0; i < entries.size(); ++i) {
    const auto& entry = entries[i];
    if (!entry.input_tensor || !entry.output_tensor) {
      throw std::invalid_argument(
          "processcvu packed-route facts require input_tensor and output_tensor");
    }
    const int logical_index = entry.logical_index >= 0 ? entry.logical_index : static_cast<int>(i);
    const int output_slot = entry.output_slot >= 0 ? entry.output_slot : logical_index;
    const int physical_index = entry.physical_index >= 0 ? entry.physical_index : logical_index;

    facts.inputs.push_back(build_packed_processcvu_input_fact(
        logical_index, 0, physical_input_name, *entry.input_tensor, entry.input_dtype,
        entry.input_layout, entry.input_byte_offset));
    facts.input_bindings.push_back(build_packed_processcvu_input_binding_fact(
        logical_index, physical_input_name, *entry.input_tensor, entry.input_dtype,
        entry.src_output_slot >= 0 ? entry.src_output_slot : output_slot,
        entry.src_physical_output_index >= 0 ? entry.src_physical_output_index : physical_index));
    auto output = build_packed_processcvu_output_fact(
        logical_index, output_slot, 0, physical_output_name, *entry.output_tensor,
        entry.output_logical_name, entry.output_dtype, entry.output_layout,
        entry.output_byte_offset);
    facts.output_order.push_back(build_processcvu_published_output_route_fact(output));
    facts.outputs.push_back(std::move(output));
  }
  return facts;
}

namespace {

std::string processcvu_routed_input_segment_name(const MpkTensorContract& published_input,
                                                 const ProcessCvuCanonicalInputFact& fallback) {
  if (!published_input.segment_name.empty()) {
    return published_input.segment_name;
  }
  if (!published_input.name.empty()) {
    return published_input.name;
  }
  if (!fallback.physical_name.empty()) {
    return fallback.physical_name;
  }
  return fallback.logical_name;
}

std::uint64_t processcvu_routed_input_size_bytes(const MpkTensorContract& published_input,
                                                 const ProcessCvuCanonicalInputFact& fallback) {
  const std::string dtype = !published_input.dtype.empty() ? published_input.dtype : fallback.dtype;
  const std::uint64_t published_size =
      preferred_mpk_tensor_size_bytes_local(published_input, dtype);
  if (published_size > 0U) {
    return published_size;
  }
  if (fallback.size_bytes > 0U) {
    return fallback.size_bytes;
  }
  return specbuilders::tensor_size_bytes_from_shape_dtype(fallback.shape, fallback.dtype);
}

TensorMaterializationKind
processcvu_routed_input_materialization_kind(const MpkTensorContract& published_input) {
  switch (published_input.materialization_kind) {
  case MpkTensorMaterializationKind::OffsetView:
    return TensorMaterializationKind::OffsetView;
  case MpkTensorMaterializationKind::Bf16LaneSplitRepack:
    return TensorMaterializationKind::Bf16LaneSplitRepack;
  case MpkTensorMaterializationKind::Unknown:
  case MpkTensorMaterializationKind::Direct:
  default:
    return TensorMaterializationKind::Direct;
  }
}

void apply_published_physical_view_to_payload_input_desc_local(
    ProcessCvuCanonicalCompileInputs* out, std::size_t index,
    const MpkTensorContract& published_input, const std::string& dtype);

void apply_published_routed_input_metadata(ProcessCvuCanonicalInputFact* input,
                                           ProcessCvuCanonicalBindingFact* binding,
                                           const MpkTensorContract& published_input,
                                           const std::string& routed_segment_name,
                                           const std::int64_t source_byte_offset,
                                           const int src_physical_output_index,
                                           const std::uint64_t source_size_bytes) {
  if (!input || !binding) {
    throw std::invalid_argument(
        "processcvu routed input metadata update requires input and binding storage");
  }
  input->physical_name = routed_segment_name;
  if (src_physical_output_index >= 0) {
    input->physical_index = src_physical_output_index;
  }
  if (!published_input.name.empty()) {
    input->logical_name = published_input.name;
  }
  input->byte_offset = source_byte_offset;
  input->materialization_kind = processcvu_routed_input_materialization_kind(published_input);

  binding->source_segment_name = routed_segment_name;
  binding->src_physical_output_index = src_physical_output_index;
  binding->src_physical_byte_offset = source_byte_offset;
  if (source_size_bytes > 0U) {
    binding->src_physical_size_bytes = source_size_bytes;
  }
}

void canonicalize_published_routed_inputs(ProcessCvuCanonicalCompileInputs* out,
                                          const std::vector<MpkTensorContract>& published_inputs,
                                          const std::string& graph_family) {
  if (!out) {
    throw std::invalid_argument("processcvu routed input canonicalization requires output storage");
  }
  out->facts.physical_input_names.clear();
  const std::size_t count = std::min(
      {published_inputs.size(), out->facts.inputs.size(), out->facts.input_bindings.size()});
  for (std::size_t i = 0; i < count; ++i) {
    const auto& published_input = published_inputs[i];
    auto& input = out->facts.inputs[i];
    auto& binding = out->facts.input_bindings[i];

    const std::string routed_segment_name =
        processcvu_routed_input_segment_name(published_input, input);
    const std::int64_t selector_byte_offset = published_input.byte_offset;
    const int src_physical_output_index =
        published_input.source_physical_index >= 0 ? published_input.source_physical_index
        : published_input.physical_index >= 0      ? published_input.physical_index
                                                   : binding.src_physical_output_index;
    const std::uint64_t source_size_bytes =
        processcvu_routed_input_size_bytes(published_input, input);
    apply_published_routed_input_metadata(
        &input, &binding, published_input, routed_segment_name, selector_byte_offset,
        src_physical_output_index, std::max(binding.src_physical_size_bytes, source_size_bytes));
    apply_published_physical_view_to_payload_input_desc_local(out, i, published_input, input.dtype);

    const bool child_segment_with_parent_offset =
        selector_byte_offset != 0 && !published_input.segment_name.empty() &&
        !published_input.name.empty() && published_input.segment_name == published_input.name;
    if (child_segment_with_parent_offset) {
      throw std::invalid_argument("processcvu routed packed input mixes child segment '" +
                                  routed_segment_name + "' with parent-relative byte offset " +
                                  std::to_string(selector_byte_offset) +
                                  " graph_family=" + graph_family);
    }
    if (input.physical_name != binding.source_segment_name ||
        input.byte_offset != binding.src_physical_byte_offset) {
      throw std::invalid_argument(
          "processcvu routed packed input addressing mismatch for segment '" + routed_segment_name +
          "' graph_family=" + graph_family);
    }
  }
}

void apply_published_routed_input_bindings(ProcessCvuCanonicalCompileInputs* out,
                                           const std::vector<MpkTensorContract>& published_inputs,
                                           const std::vector<std::uint64_t>* packed_input_sizes,
                                           const std::string& graph_family) {
  if (!out) {
    throw std::invalid_argument("processcvu routed input binding update requires output storage");
  }
  const std::size_t count = std::min(
      {published_inputs.size(), out->facts.inputs.size(), out->facts.input_bindings.size()});
  for (std::size_t i = 0; i < count; ++i) {
    const auto& published_input = published_inputs[i];
    auto& input = out->facts.inputs[i];
    auto& binding = out->facts.input_bindings[i];

    const std::string routed_segment_name =
        processcvu_routed_input_segment_name(published_input, input);
    const std::int64_t source_byte_offset = published_input.byte_offset;
    const int src_physical_output_index =
        published_input.source_physical_index >= 0 ? published_input.source_physical_index
        : published_input.physical_index >= 0      ? published_input.physical_index
                                                   : binding.src_physical_output_index;
    const std::uint64_t packed_input_size =
        packed_input_sizes && i < packed_input_sizes->size() ? (*packed_input_sizes)[i] : 0U;
    const std::uint64_t source_size_bytes =
        packed_input_size > 0U ? packed_input_size
                               : processcvu_routed_input_size_bytes(published_input, input);
    apply_published_routed_input_metadata(&input, &binding, published_input, routed_segment_name,
                                          source_byte_offset, src_physical_output_index,
                                          source_size_bytes);
    apply_published_physical_view_to_payload_input_desc_local(out, i, published_input, input.dtype);

    const bool child_segment_with_parent_offset =
        source_byte_offset != 0 && !published_input.segment_name.empty() &&
        !published_input.name.empty() && published_input.segment_name == published_input.name;
    if (child_segment_with_parent_offset) {
      throw std::invalid_argument("processcvu routed packed input mixes child segment '" +
                                  routed_segment_name + "' with parent-relative byte offset " +
                                  std::to_string(source_byte_offset) +
                                  " graph_family=" + graph_family);
    }
  }
}

void force_direct_materialization_for_inputs(ProcessCvuCanonicalCompileInputs* out) {
  if (!out) {
    throw std::invalid_argument(
        "processcvu direct-materialization normalization requires output storage");
  }
  for (auto& input : out->facts.inputs) {
    input.materialization_kind = TensorMaterializationKind::Direct;
  }
}

std::int64_t
processcvu_packed_parent_input_byte_offset(const ProcessCvuPackedRouteEntry& entry,
                                           const std::vector<MpkTensorContract>* published_inputs,
                                           const std::size_t index) {
  const std::int64_t fallback = std::max<std::int64_t>(entry.input_byte_offset, 0);
  if (!published_inputs || index >= published_inputs->size()) {
    return fallback;
  }

  const auto& published = (*published_inputs)[index];
  if (published.materialization_kind == MpkTensorMaterializationKind::OffsetView) {
    if (published.source_byte_offset > 0) {
      return published.source_byte_offset;
    }
    if (published.byte_offset > 0) {
      return published.byte_offset;
    }
    if (published.source_byte_offset >= 0) {
      return published.source_byte_offset;
    }
    if (published.byte_offset >= 0) {
      return published.byte_offset;
    }
    return fallback;
  }

  // Older contracts did not always tag offset views explicitly. A non-zero
  // source_byte_offset still means the published tensor is a semantic view
  // into a parent physical buffer, so bind the EVXX input descriptor to that
  // parent-relative address instead of to a compact logical accumulation.
  if (published.source_byte_offset > 0) {
    return published.source_byte_offset;
  }

  // Direct parent views may carry their physical offset in byte_offset.
  if (published.byte_offset > 0) {
    return published.byte_offset;
  }

  return fallback;
}

void enforce_packed_parent_input_views(
    ProcessCvuCanonicalCompileInputs* out, const std::string& parent_input_name,
    const std::vector<ProcessCvuPackedRouteEntry>& entries,
    const std::vector<std::uint64_t>& packed_input_sizes,
    const std::vector<MpkTensorContract>* published_inputs = nullptr) {
  if (!out) {
    throw std::invalid_argument(
        "processcvu packed-parent input normalization requires output storage");
  }
  if (parent_input_name.empty()) {
    throw std::invalid_argument(
        "processcvu packed-parent input normalization requires parent input name");
  }
  const std::size_t count =
      std::min({entries.size(), out->facts.inputs.size(), out->facts.input_bindings.size()});
  if (count == 0U) {
    return;
  }

  out->facts.physical_input_names = {parent_input_name};
  std::uint64_t packed_parent_size = 0U;
  for (std::size_t i = 0; i < count; ++i) {
    const auto& entry = entries[i];
    auto& input = out->facts.inputs[i];
    auto& binding = out->facts.input_bindings[i];
    const std::uint64_t size_bytes =
        (i < packed_input_sizes.size() && packed_input_sizes[i] > 0U)
            ? packed_input_sizes[i]
            : (binding.src_physical_size_bytes > 0U ? binding.src_physical_size_bytes
                                                    : input.size_bytes);
    const std::int64_t byte_offset =
        processcvu_packed_parent_input_byte_offset(entry, published_inputs, i);

    input.physical_index = 0;
    input.physical_name = parent_input_name;
    input.byte_offset = byte_offset;
    input.materialization_kind = TensorMaterializationKind::Direct;

    binding.sink_pad_index = 0;
    binding.local_logical_input_index =
        input.logical_index >= 0 ? input.logical_index : static_cast<int>(i);
    binding.required = true;
    binding.cm_input_name = parent_input_name;
    binding.source_segment_name = parent_input_name;
    binding.src_physical_output_index = 0;
    binding.src_physical_byte_offset = byte_offset;
    binding.src_physical_size_bytes = size_bytes;

    if (i < out->payload.input_tensors.size()) {
      out->payload.input_tensors[i].storage.addr = static_cast<std::uint64_t>(byte_offset);
      if (size_bytes > 0U) {
        out->payload.input_tensors[i].storage.nbytes = size_bytes;
      }
    }

    const std::uint64_t offset_u = static_cast<std::uint64_t>(byte_offset);
    if (size_bytes > (std::numeric_limits<std::uint64_t>::max() - offset_u)) {
      throw std::overflow_error("processcvu packed-parent input byte span overflow");
    }
    packed_parent_size = std::max(packed_parent_size, offset_u + size_bytes);
  }
  (void)packed_parent_size;
}

bool published_inputs_share_single_physical_parent(
    const std::vector<MpkTensorContract>& published_inputs) {
  if (published_inputs.size() <= 1U) {
    return true;
  }

  std::vector<int> physical_indices;
  physical_indices.reserve(published_inputs.size());
  bool all_have_physical_index = true;
  for (const auto& published : published_inputs) {
    const int physical_index = published.source_physical_index >= 0
                                   ? published.source_physical_index
                               : published.physical_index >= 0 ? published.physical_index
                                                               : -1;
    if (physical_index < 0) {
      all_have_physical_index = false;
      break;
    }
    physical_indices.push_back(physical_index);
  }
  if (all_have_physical_index && !physical_indices.empty()) {
    return std::all_of(physical_indices.begin() + 1, physical_indices.end(),
                       [&](int index) { return index == physical_indices.front(); });
  }

  std::vector<std::string> segment_names;
  segment_names.reserve(published_inputs.size());
  bool all_have_segment_name = true;
  for (const auto& published : published_inputs) {
    const std::string segment_name =
        !published.segment_name.empty() ? published.segment_name : published.name;
    if (segment_name.empty()) {
      all_have_segment_name = false;
      break;
    }
    segment_names.push_back(segment_name);
  }
  if (all_have_segment_name && !segment_names.empty()) {
    return std::all_of(segment_names.begin() + 1, segment_names.end(),
                       [&](const std::string& name) { return name == segment_names.front(); });
  }

  // When the MPK boundary does not carry enough parent identity to prove a
  // native-distinct multi-buffer handoff, preserve the historical packed-parent
  // behavior.  Native MLATess multi-output packs do carry either distinct
  // source_physical_index/physical_index values or distinct segment names.
  return true;
}

void preserve_routed_source_segment_input_views(ProcessCvuCanonicalCompileInputs* out,
                                                const std::string& graph_input_name) {
  if (!out) {
    throw std::invalid_argument("processcvu routed input preservation requires output storage");
  }
  if (graph_input_name.empty()) {
    throw std::invalid_argument("processcvu routed input preservation requires graph input name");
  }

  const std::size_t count = std::min(out->facts.inputs.size(), out->facts.input_bindings.size());
  if (count == 0U) {
    return;
  }

  out->facts.physical_input_names.clear();
  out->facts.physical_input_names.reserve(count);
  for (std::size_t i = 0; i < count; ++i) {
    auto& input = out->facts.inputs[i];
    auto& binding = out->facts.input_bindings[i];

    std::string source_segment_name =
        !binding.source_segment_name.empty() ? binding.source_segment_name
        : !input.physical_name.empty()       ? input.physical_name
        : !input.logical_name.empty()        ? input.logical_name
                                             : "input_tensor_" + std::to_string(i);

    auto existing = std::find(out->facts.physical_input_names.begin(),
                              out->facts.physical_input_names.end(), source_segment_name);
    if (existing == out->facts.physical_input_names.end()) {
      out->facts.physical_input_names.push_back(source_segment_name);
      existing = out->facts.physical_input_names.end() - 1;
    }
    const int local_physical_index =
        static_cast<int>(std::distance(out->facts.physical_input_names.begin(), existing));

    input.physical_index = local_physical_index;
    input.physical_name = source_segment_name;
    binding.sink_pad_index = 0;
    binding.local_logical_input_index =
        input.logical_index >= 0 ? input.logical_index : static_cast<int>(i);
    binding.required = true;
    // Keep the ConfigManager graph input as the authored packed key, but keep
    // the resolver-facing source segment as the real upstream MPK segment
    // (for example MLA_0_0..MLA_0_N).  This reuses the existing bundled
    // multi-IO direct-alias path without fabricating an unresolved
    // source_segment=input_tensor.
    binding.cm_input_name = graph_input_name;
    binding.source_segment_name = source_segment_name;
    if (binding.src_physical_output_index < 0) {
      binding.src_physical_output_index = local_physical_index;
    }
  }
}

void preserve_distinct_physical_output_views(
    ProcessCvuCanonicalCompileInputs* out,
    const std::vector<std::string>& preferred_physical_names) {
  if (!out) {
    throw std::invalid_argument("processcvu distinct-output normalization requires output storage");
  }
  const std::size_t count = out->facts.outputs.size();
  if (count == 0U) {
    return;
  }

  out->facts.physical_output_names.clear();
  out->facts.physical_output_names.reserve(count);
  for (std::size_t i = 0; i < count; ++i) {
    auto& output = out->facts.outputs[i];
    std::string physical_name =
        i < preferred_physical_names.size() ? preferred_physical_names[i] : std::string{};
    if (physical_name.empty()) {
      physical_name = !output.logical_name.empty()
                          ? output.logical_name
                          : (!output.physical_name.empty() ? output.physical_name
                                                           : "output_tensor_" + std::to_string(i));
    }

    output.representation = ProcessCvuOutputRepresentation::DenseTensor;
    output.physical_index = static_cast<int>(i);
    output.physical_name = physical_name;
    output.byte_offset = 0;
    if (output.logical_index < 0) {
      output.logical_index = static_cast<int>(i);
    }
    if (output.output_slot < 0) {
      output.output_slot = static_cast<int>(i);
    }
    if (output.tensor_index < 0) {
      output.tensor_index = output.logical_index;
    }
    out->facts.physical_output_names.push_back(physical_name);
  }

  out->facts.output_order.clear();
  out->facts.output_order.reserve(count);
  for (const auto& output : out->facts.outputs) {
    out->facts.output_order.push_back(build_processcvu_published_output_route_fact(output));
  }
  out->facts.preserve_physical_outputs = true;
}

bool processcvu_tensor_desc_has_leading_unit_batch_local(const sima_ev_tensor_desc& desc) {
  if (desc.shape.rank == 0U) {
    return false;
  }
  const auto first_axis = static_cast<sima_ev_axis_semantic>(desc.shape.axis_semantics[0]);
  return desc.shape.sizes[0] == 1 &&
         (first_axis == SIMA_EV_AXIS_N || first_axis == SIMA_EV_AXIS_UNKNOWN);
}

bool processcvu_published_shape_has_leading_unit_batch_local(
    const MpkTensorContract& published_input) {
  const auto shape = !published_input.logical_shape.empty() ? published_input.logical_shape
                                                            : published_input.mpk_shape;
  return !shape.empty() && shape.front() == 1;
}

bool processcvu_try_promote_desc_to_published_physical_shape_local(
    sima_ev_tensor_desc& desc, const MpkTensorContract& published_input) {
  if (desc.layout_kind != SIMA_EV_LAYOUT_STRIDED || published_input.mpk_shape.empty() ||
      published_input.logical_shape.empty() || published_input.stride_bytes.empty()) {
    return false;
  }

  const std::size_t desc_rank = static_cast<std::size_t>(desc.shape.rank);
  const std::size_t physical_rank = published_input.mpk_shape.size();
  const std::size_t logical_rank = published_input.logical_shape.size();
  if (desc_rank == 0U || physical_rank == desc_rank ||
      physical_rank > static_cast<std::size_t>(SIMA_EV_MAX_RANK) ||
      published_input.stride_bytes.size() != physical_rank || logical_rank != desc_rank ||
      physical_rank < logical_rank) {
    return false;
  }

  for (std::size_t axis = 0; axis < logical_rank; ++axis) {
    if (published_input.logical_shape[axis] <= 0 ||
        desc.shape.sizes[axis] != published_input.logical_shape[axis]) {
      return false;
    }
  }

  const std::size_t leading_physical_axes = physical_rank - logical_rank;
  for (std::size_t axis = 0; axis < leading_physical_axes; ++axis) {
    if (published_input.mpk_shape[axis] != 1) {
      return false;
    }
  }
  for (std::size_t axis = 0; axis < logical_rank; ++axis) {
    const auto physical_dim = published_input.mpk_shape[leading_physical_axes + axis];
    if (physical_dim <= 0 || physical_dim != published_input.logical_shape[axis]) {
      return false;
    }
  }

  desc.shape.rank = static_cast<std::uint32_t>(physical_rank);
  for (std::uint32_t axis = 0; axis < SIMA_EV_MAX_RANK; ++axis) {
    desc.shape.sizes[axis] = 0;
  }
  for (std::size_t axis = 0; axis < physical_rank; ++axis) {
    desc.shape.sizes[axis] = published_input.mpk_shape[axis];
  }
  apply_tensor_axes_processcvu_local(&desc);
  return true;
}

template <typename AssignFn>
void assign_published_desc_values_preserving_logical_shape_local(
    sima_ev_tensor_desc& desc, const MpkTensorContract& published_input,
    const std::vector<std::int64_t>& values, AssignFn assign_fn, const char* mismatch_message) {
  if (values.empty()) {
    return;
  }

  const auto desc_rank = static_cast<std::size_t>(desc.shape.rank);
  std::size_t desc_offset = 0U;
  std::size_t value_offset = 0U;
  if (values.size() == desc_rank) {
    // Same-rank published view: copy one-to-one.
  } else if (values.size() + 1U == desc_rank &&
             processcvu_tensor_desc_has_leading_unit_batch_local(desc)) {
    // Published view dropped a leading N=1, while the runtime descriptor kept it.
    desc_offset = 1U;
  } else if (values.size() == desc_rank + 1U &&
             processcvu_published_shape_has_leading_unit_batch_local(published_input)) {
    // Published view kept a leading N=1, while the consuming descriptor is per-frame.
    value_offset = 1U;
  } else {
    throw std::invalid_argument(mismatch_message);
  }

  const std::size_t count = values.size() - value_offset;
  if (desc_offset + count > desc_rank) {
    throw std::invalid_argument(mismatch_message);
  }
  for (std::size_t i = 0; i < count; ++i) {
    assign_fn(desc_offset + i, values[value_offset + i]);
  }
}

std::uint64_t processcvu_required_strided_storage_bytes_local(const sima_ev_tensor_desc& desc,
                                                              const std::string& dtype) {
  if (desc.layout_kind != SIMA_EV_LAYOUT_STRIDED || desc.shape.rank == 0U ||
      desc.shape.rank > SIMA_EV_MAX_RANK) {
    return 0U;
  }
  const std::uint64_t elem_bytes = processcvu_dtype_size_bytes_from_token(dtype);
  if (elem_bytes == 0U) {
    return 0U;
  }

  std::uint64_t max_offset = 0U;
  for (std::uint32_t axis = 0; axis < desc.shape.rank; ++axis) {
    const auto extent = desc.shape.sizes[axis];
    const auto stride = desc.layout.strided.strides_bytes[axis];
    if (extent <= 0 || stride < 0) {
      return 0U;
    }
    std::uint64_t dim_offset = multiply_u64_checked_local(static_cast<std::uint64_t>(extent - 1),
                                                          static_cast<std::uint64_t>(stride),
                                                          "published routed input strided storage");
    if (max_offset > std::numeric_limits<std::uint64_t>::max() - dim_offset) {
      throw std::overflow_error("processcvu published routed input strided storage overflow");
    }
    max_offset += dim_offset;
  }
  if (max_offset > std::numeric_limits<std::uint64_t>::max() - elem_bytes) {
    throw std::overflow_error("processcvu published routed input strided storage overflow");
  }
  return max_offset + elem_bytes;
}

void apply_published_physical_view_to_payload_input_desc_local(
    ProcessCvuCanonicalCompileInputs* out, std::size_t index,
    const MpkTensorContract& published_input, const std::string& dtype) {
  if (!out || index >= out->payload.input_tensors.size()) {
    return;
  }

  auto& desc = out->payload.input_tensors[index];
  if (desc.layout_kind != SIMA_EV_LAYOUT_STRIDED) {
    // Tiled kernels already carry their physical transport layout in the
    // authored descriptor.  The routed-view fix is for strided tensor ABI
    // inputs such as cast/dequant reading padded MLA views.
    return;
  }

  const std::string resolved_dtype =
      !dtype.empty() ? dtype
                     : (!published_input.dtype.empty() ? published_input.dtype
                                                       : published_input.logical_dtype);
  if (!published_input.stride_bytes.empty()) {
    processcvu_try_promote_desc_to_published_physical_shape_local(desc, published_input);
    assign_published_desc_values_preserving_logical_shape_local(
        desc, published_input, published_input.stride_bytes,
        [&](std::size_t axis, std::int64_t value) {
          desc.layout.strided.strides_bytes[axis] = value;
        },
        "processcvu published routed input stride rank mismatch");
  }

  const std::uint64_t published_storage =
      preferred_mpk_tensor_size_bytes_local(published_input, resolved_dtype);
  const std::uint64_t required_storage =
      processcvu_required_strided_storage_bytes_local(desc, resolved_dtype);
  const std::uint64_t storage_nbytes = std::max(published_storage, required_storage);
  if (storage_nbytes > 0U) {
    desc.storage.nbytes = storage_nbytes;
  }
}

void override_payload_input_desc_from_published_view(ProcessCvuCanonicalCompileInputs* out,
                                                     std::size_t index,
                                                     const MpkTensorContract& published_input,
                                                     const std::string& dtype) {
  if (!out) {
    throw std::invalid_argument("processcvu payload input override requires output storage");
  }
  if (index >= out->payload.input_tensors.size()) {
    throw std::out_of_range(
        "processcvu payload input override index exceeds authored tensor descs");
  }
  auto& desc = out->payload.input_tensors[index];
  if (desc.layout_kind != SIMA_EV_LAYOUT_STRIDED) {
    throw std::invalid_argument("processcvu payload input override requires strided descriptor");
  }

  const auto published_shape = preferred_packed_mpk_tensor_shape_local(published_input);
  assign_published_desc_values_preserving_logical_shape_local(
      desc, published_input, published_shape,
      [&](std::size_t axis, std::int64_t value) { desc.shape.sizes[axis] = value; },
      "processcvu payload input override shape rank mismatch");
  apply_published_physical_view_to_payload_input_desc_local(out, index, published_input, dtype);
  if (desc.storage.nbytes == 0U) {
    throw std::invalid_argument("processcvu payload input override requires concrete storage span");
  }
}

void sync_payload_tensor_desc_offsets_from_facts(ProcessCvuStagePayload* payload,
                                                 const ProcessCvuCanonicalFacts& facts) {
  if (!payload) {
    throw std::invalid_argument("processcvu payload tensor-desc sync requires payload storage");
  }
  if (!payload->input_tensors.empty()) {
    if (payload->input_tensors.size() != facts.inputs.size()) {
      throw std::invalid_argument(
          "processcvu payload input descriptor count does not match canonical inputs");
    }
    for (std::size_t i = 0; i < facts.inputs.size(); ++i) {
      const auto& fact = facts.inputs[i];
      if (fact.byte_offset < 0) {
        throw std::invalid_argument(
            "processcvu payload input descriptor requires non-negative byte_offset");
      }
      auto& desc = payload->input_tensors[i];
      desc.storage.addr = static_cast<std::uint64_t>(fact.byte_offset);
      // Phase 3a (Option A++): the runtime-config builder authors
      // `desc.storage.nbytes` as PER-FRAME storage (batch is carried as
      // payload->batch_size and the kernel scales by it). Taking max with the
      // upstream binding's `src_physical_size_bytes` (which is the full
      // batched upstream segment) overrides per-frame with batched and then
      // gets multiplied by batch_size again downstream — required = N*N*frame
      // instead of N*frame. Only fall back to the upstream size when
      // storage.nbytes is uninitialized (==0), which preserves the legacy
      // batch=1 behavior for stages that haven't been migrated.
      if (desc.storage.nbytes == 0U && i < facts.input_bindings.size() &&
          facts.input_bindings[i].src_physical_size_bytes > 0U) {
        desc.storage.nbytes = facts.input_bindings[i].src_physical_size_bytes;
      }
    }
  }
  if (!payload->output_tensors.empty()) {
    if (payload->output_tensors.size() != facts.outputs.size()) {
      throw std::invalid_argument(
          "processcvu payload output descriptor count does not match canonical outputs");
    }
    for (std::size_t i = 0; i < facts.outputs.size(); ++i) {
      const auto& fact = facts.outputs[i];
      if (fact.byte_offset < 0) {
        throw std::invalid_argument(
            "processcvu payload output descriptor requires non-negative byte_offset");
      }
      payload->output_tensors[i].storage.addr = static_cast<std::uint64_t>(fact.byte_offset);
    }
  }
}

} // namespace

namespace {

ProcessCvuCanonicalFacts
build_processcvu_facts_from_payload_local(const ProcessCvuStagePayload& payload) {
  const std::string family = canonical_family_name(payload.graph_family);
  if (family == "preproc") {
    return build_preproc_facts_from_payload(payload);
  }
  const std::size_t input_count =
      std::max({payload.default_output_names.size(),
                static_cast<std::size_t>(payload.num_in_tensor > 0 ? payload.num_in_tensor : 0),
                payload.input_shapes.size(), payload.slice_shapes.size(), std::size_t{1}});
  if (input_count > 1U || payload.default_output_names.size() > 1U) {
    return build_multi_io_processcvu_facts_from_payload(payload, {});
  }
  return build_single_io_processcvu_facts_from_payload(payload);
}

ProcessCvuCanonicalFacts build_preproc_facts_from_payload(const ProcessCvuStagePayload& payload) {
  ProcessCvuStagePayload canonical_payload = payload;
  canonicalize_preproc_single_handoff_payload(&canonical_payload);
  populate_preproc_payload_semantics(&canonical_payload);
  require_supported_preproc_single_output_handoff(canonical_payload.preproc_single_output_handoff);
  const std::size_t input_count =
      std::max({canonical_payload.input_shapes.size(), canonical_payload.input_tensors.size(),
                static_cast<std::size_t>(
                    canonical_payload.num_in_tensor > 0 ? canonical_payload.num_in_tensor : 0)});
  if (input_count != 1U) {
    throw std::invalid_argument("processcvu preproc payload requires exactly one input");
  }
  if (canonical_payload.default_input_name.empty()) {
    throw std::invalid_argument("processcvu preproc payload missing explicit default_input_name");
  }
  if (canonical_payload.primary_output_name.empty()) {
    throw std::invalid_argument("processcvu preproc payload missing explicit primary_output_name");
  }
  const auto internal_output_names = build_preproc_internal_output_names(canonical_payload);
  if (internal_output_names.empty()) {
    throw std::invalid_argument("processcvu preproc payload missing explicit runtime outputs");
  }
  if (std::find(internal_output_names.begin(), internal_output_names.end(),
                canonical_payload.primary_output_name) == internal_output_names.end()) {
    throw std::invalid_argument(
        "processcvu preproc payload primary_output_name must name a runtime output");
  }
  const auto runtime_output_names = build_preproc_runtime_output_names(canonical_payload);
  if (runtime_output_names.empty()) {
    throw std::invalid_argument("processcvu preproc payload missing explicit runtime outputs");
  }
  if (std::find(runtime_output_names.begin(), runtime_output_names.end(),
                canonical_payload.primary_output_name) == runtime_output_names.end()) {
    throw std::invalid_argument(
        "processcvu preproc payload primary_output_name must name a published runtime output");
  }
  ProcessCvuCanonicalFacts facts;
  facts.physical_input_names = {canonical_payload.default_input_name};
  facts.physical_output_names = internal_output_names;
  facts.primary_output_name = canonical_payload.primary_output_name;
  facts.published_output_names =
      canonical_payload.preproc_single_output_handoff
          ? std::vector<std::string>{canonical_payload.primary_output_name}
          : runtime_output_names;

  const auto input_tensor = synthesize_preproc_input_tensor(canonical_payload);
  facts.inputs.push_back(build_dense_processcvu_input_fact(
      0, 0, canonical_payload.default_input_name, input_tensor.shape, input_tensor.dtype,
      input_tensor.layout));
  ProcessCvuCanonicalBindingFact preproc_binding;
  preproc_binding.sink_pad_index = 0;
  preproc_binding.local_logical_input_index = 0;
  preproc_binding.required = true;
  preproc_binding.cm_input_name = canonical_payload.default_input_name;
  preproc_binding.source_segment_name = "parent";
  facts.input_bindings.push_back(std::move(preproc_binding));

  const auto runtime_outputs = synthesize_preproc_runtime_outputs(canonical_payload);
  if (runtime_outputs.empty()) {
    throw std::invalid_argument(
        "processcvu preproc payload did not synthesize any runtime outputs");
  }
  for (std::size_t i = 0; i < runtime_outputs.size(); ++i) {
    const auto& tensor = runtime_outputs[i];
    const auto internal_index =
        find_preproc_internal_output_index(canonical_payload, tensor.semantic_tag);
    if (!internal_index.has_value()) {
      throw std::invalid_argument(
          "processcvu preproc payload runtime output selection lost internal output identity");
    }
    const std::size_t metadata_index = *internal_index;
    const auto transport_kind =
        metadata_index < canonical_payload.runtime_output_transport_kind_list.size()
            ? canonical_payload.runtime_output_transport_kind_list[metadata_index]
            : (canonical_payload.tessellate && tensor.semantic_tag == "output_tessellated_image"
                   ? ProcessCvuOutputTransportKind::Packed
                   : ProcessCvuOutputTransportKind::Dense);
    const int logical_index =
        metadata_index < canonical_payload.runtime_output_logical_index_list.size()
            ? canonical_payload.runtime_output_logical_index_list[metadata_index]
            : static_cast<int>(metadata_index);
    const int output_slot =
        metadata_index < canonical_payload.runtime_output_output_slot_list.size()
            ? canonical_payload.runtime_output_output_slot_list[metadata_index]
            : static_cast<int>(metadata_index);
    const int physical_index =
        metadata_index < canonical_payload.runtime_output_physical_index_list.size()
            ? canonical_payload.runtime_output_physical_index_list[metadata_index]
            : static_cast<int>(metadata_index);
    const std::string physical_name =
        physical_index >= 0 &&
                static_cast<std::size_t>(physical_index) < internal_output_names.size()
            ? internal_output_names[static_cast<std::size_t>(physical_index)]
            : tensor.semantic_tag;
    const bool packed_tess_handoff = transport_kind == ProcessCvuOutputTransportKind::Packed;
    if (packed_tess_handoff) {
      const std::uint64_t packed_size_bytes =
          synthesize_preproc_packed_output_size_bytes(canonical_payload, tensor.dtype);
      if (packed_size_bytes == 0U) {
        throw std::invalid_argument(
            "processcvu preproc tessellated output requires a non-zero packed byte size");
      }
      facts.outputs.push_back(build_packed_blob_processcvu_output_fact(
          logical_index, output_slot, physical_index, physical_name, tensor.semantic_tag,
          tensor.shape, tensor.dtype, tensor.layout, packed_size_bytes, 0));
      continue;
    }

    ProcessCvuCanonicalOutputFact output;
    output.logical_index = logical_index;
    output.physical_index = physical_index;
    output.output_slot = output_slot;
    output.tensor_index = logical_index;
    output.physical_name = physical_name;
    output.logical_name = tensor.semantic_tag;
    output.shape = tensor.shape;
    output.dtype = tensor.dtype;
    output.layout = tensor.layout;
    facts.outputs.push_back(std::move(output));
  }
  const auto append_output_route = [&](const ProcessCvuCanonicalOutputFact& output) {
    facts.output_order.push_back(build_processcvu_published_output_route_fact(output));
  };
  for (const auto& output : facts.outputs) {
    if (output.logical_name == facts.primary_output_name) {
      append_output_route(output);
      break;
    }
  }
  if (!canonical_payload.preproc_single_output_handoff) {
    for (const auto& output : facts.outputs) {
      if (output.logical_name != facts.primary_output_name) {
        append_output_route(output);
      }
    }
  }
  return facts;
}

ProcessCvuCanonicalFacts
build_single_io_processcvu_facts_from_payload(const ProcessCvuStagePayload& payload) {
  if (payload.default_input_name.empty()) {
    throw std::invalid_argument("processcvu payload missing explicit default_input_name");
  }
  if (payload.primary_output_name.empty()) {
    throw std::invalid_argument("processcvu payload missing explicit primary_output_name");
  }
  if (payload.default_output_names.size() != 1U || payload.default_output_names.front().empty()) {
    throw std::invalid_argument(
        "single-io processcvu payload requires exactly one explicit runtime output");
  }
  if (payload.primary_output_name != payload.default_output_names.front()) {
    throw std::invalid_argument(
        "single-io processcvu payload primary_output_name must match the runtime output");
  }

  ProcessCvuCanonicalFacts facts;
  facts.physical_input_names = {payload.default_input_name};
  facts.physical_output_names = {payload.default_output_names.front()};
  facts.primary_output_name = payload.primary_output_name;
  facts.published_output_names = {payload.default_output_names.front()};

  const auto input_tensor = synthesize_single_io_input_tensor(payload);
  if (input_tensor.shape.empty()) {
    throw std::invalid_argument("single-io processcvu payload input shape missing");
  }
  auto input_fact =
      build_dense_processcvu_input_fact(0, 0, payload.default_input_name, input_tensor.shape,
                                        input_tensor.dtype, input_tensor.layout);
  input_fact.quant = build_runtime_input_quant_spec_from_payload(payload, 0);
  facts.inputs.push_back(std::move(input_fact));
  ProcessCvuCanonicalBindingFact single_binding;
  single_binding.sink_pad_index = 0;
  single_binding.local_logical_input_index = 0;
  single_binding.required = true;
  single_binding.cm_input_name = payload.default_input_name;
  single_binding.source_segment_name = payload.default_input_name;
  facts.input_bindings.push_back(std::move(single_binding));

  const auto output_tensor = synthesize_single_io_output_tensor(payload);
  if (output_tensor.shape.empty()) {
    throw std::invalid_argument("single-io processcvu payload output shape missing");
  }
  const auto transport_kind = !payload.runtime_output_transport_kind_list.empty()
                                  ? payload.runtime_output_transport_kind_list.front()
                                  : ProcessCvuOutputTransportKind::Dense;
  if (transport_kind_is_packed(transport_kind)) {
    ProcessCvuCanonicalOutputFact output;
    output.representation = ProcessCvuOutputRepresentation::PackedTensor;
    output.logical_index = 0;
    output.physical_index = 0;
    output.output_slot = 0;
    output.tensor_index = output_tensor.tensor_index;
    output.physical_name = payload.default_output_names.front();
    output.logical_name = payload.default_output_names.front();
    output.shape =
        !payload.runtime_output_logical_shapes.empty() &&
                !payload.runtime_output_logical_shapes.front().empty()
            ? std::vector<std::int64_t>(payload.runtime_output_logical_shapes.front().begin(),
                                        payload.runtime_output_logical_shapes.front().end())
            : output_tensor.shape;
    output.dtype = output_tensor.dtype;
    output.layout = !payload.runtime_output_logical_layout_list.empty() &&
                            !payload.runtime_output_logical_layout_list.front().empty()
                        ? payload.runtime_output_logical_layout_list.front()
                        : output_tensor.layout;
    facts.outputs.push_back(std::move(output));
  } else {
    ProcessCvuCanonicalOutputFact output;
    output.logical_index = 0;
    output.physical_index = 0;
    output.output_slot = 0;
    output.tensor_index = output_tensor.tensor_index;
    output.physical_name = payload.default_output_names.front();
    output.logical_name = payload.default_output_names.front();
    output.shape =
        !payload.runtime_output_logical_shapes.empty() &&
                !payload.runtime_output_logical_shapes.front().empty()
            ? std::vector<std::int64_t>(payload.runtime_output_logical_shapes.front().begin(),
                                        payload.runtime_output_logical_shapes.front().end())
            : output_tensor.shape;
    output.dtype = output_tensor.dtype;
    output.layout = !payload.runtime_output_logical_layout_list.empty() &&
                            !payload.runtime_output_logical_layout_list.front().empty()
                        ? payload.runtime_output_logical_layout_list.front()
                        : output_tensor.layout;
    facts.outputs.push_back(std::move(output));
  }
  facts.output_order.push_back(build_processcvu_published_output_route_fact(facts.outputs.front()));
  return facts;
}

ProcessCvuCanonicalFacts
build_multi_io_processcvu_facts_from_payload(const ProcessCvuStagePayload& payload,
                                             const std::vector<std::string>& runtime_input_names) {
  if (payload.default_output_names.empty()) {
    throw std::invalid_argument("multi-io processcvu payload requires explicit runtime outputs");
  }

  ProcessCvuCanonicalFacts facts;
  facts.primary_output_name = payload.primary_output_name;
  facts.physical_output_names = payload.default_output_names;
  facts.published_output_names = payload.default_output_names;

  const std::size_t input_count = std::max(
      {runtime_input_names.size(), payload.input_shapes.size(), payload.slice_shapes.size(),
       static_cast<std::size_t>(payload.num_in_tensor > 0 ? payload.num_in_tensor : 0),
       std::size_t{1}});

  for (std::size_t i = 0; i < input_count; ++i) {
    const std::string input_name = i < runtime_input_names.size() && !runtime_input_names[i].empty()
                                       ? runtime_input_names[i]
                                       : (input_count == 1 ? (!payload.default_input_name.empty()
                                                                  ? payload.default_input_name
                                                                  : std::string("input_tensor"))
                                                           : "input_tensor_" + std::to_string(i));
    facts.physical_input_names.push_back(input_name);
    std::vector<std::int64_t> input_shape;
    if (i < payload.input_tensors.size()) {
      const auto shape = shape_vec_from_tensor_desc_local(payload.input_tensors[i]);
      input_shape.assign(shape.begin(), shape.end());
    } else if (i < payload.input_shapes.size()) {
      input_shape.assign(payload.input_shapes[i].begin(), payload.input_shapes[i].end());
    }
    if (input_shape.empty()) {
      throw std::invalid_argument("multi-io processcvu payload input shape missing");
    }
    std::string input_layout = payload_input_layout_token_local(payload, i);
    const std::string input_dtype =
        !payload.input_dtype.empty() ? payload.input_dtype : std::string("INT8");
    facts.inputs.push_back(
        build_dense_processcvu_input_fact(static_cast<int>(i), static_cast<int>(i), input_name,
                                          input_shape, input_dtype, input_layout));
    ProcessCvuCanonicalBindingFact multi_binding;
    multi_binding.sink_pad_index = 0;
    multi_binding.local_logical_input_index = static_cast<int>(i);
    multi_binding.src_logical_output_index = static_cast<int>(i);
    multi_binding.src_output_slot = static_cast<int>(i);
    multi_binding.src_physical_output_index = static_cast<int>(i);
    multi_binding.required = true;
    multi_binding.cm_input_name = input_name;
    multi_binding.source_segment_name = input_name;
    facts.input_bindings.push_back(std::move(multi_binding));
  }

  for (std::size_t i = 0; i < payload.default_output_names.size(); ++i) {
    const std::string& output_name = payload.default_output_names[i];
    if (output_name.empty()) {
      throw std::invalid_argument(
          "multi-io processcvu payload contains an empty runtime output name");
    }
    const std::string output_dtype = pick_indexed_or_scalar(
        payload.runtime_output_dtype_list, i,
        !payload.output_dtype.empty() ? payload.output_dtype : payload.out_dtype);
    const std::string output_layout = payload_runtime_output_layout_token_local(payload, i);
    std::vector<std::int64_t> output_shape;
    if (i < payload.runtime_output_logical_shapes.size() &&
        !payload.runtime_output_logical_shapes[i].empty()) {
      output_shape.assign(payload.runtime_output_logical_shapes[i].begin(),
                          payload.runtime_output_logical_shapes[i].end());
    } else if (i < payload.output_tensors.size()) {
      const auto shape = shape_vec_from_tensor_desc_local(payload.output_tensors[i]);
      output_shape.assign(shape.begin(), shape.end());
    } else if (i < payload.output_shapes.size()) {
      output_shape.assign(payload.output_shapes[i].begin(), payload.output_shapes[i].end());
    }
    if (output_shape.empty()) {
      throw std::invalid_argument("multi-io processcvu payload output shape missing");
    }
    const auto transport_kind = i < payload.runtime_output_transport_kind_list.size()
                                    ? payload.runtime_output_transport_kind_list[i]
                                    : ProcessCvuOutputTransportKind::Dense;
    if (transport_kind_is_packed(transport_kind)) {
      ProcessCvuCanonicalOutputFact output;
      output.representation = ProcessCvuOutputRepresentation::PackedTensor;
      output.logical_index = static_cast<int>(i);
      output.physical_index = static_cast<int>(i);
      output.output_slot = static_cast<int>(i);
      output.tensor_index = static_cast<int>(i);
      output.physical_name = output_name;
      output.logical_name = output_name;
      output.shape = output_shape;
      output.dtype = output_dtype;
      output.layout = i < payload.runtime_output_logical_layout_list.size() &&
                              !payload.runtime_output_logical_layout_list[i].empty()
                          ? payload.runtime_output_logical_layout_list[i]
                          : output_layout;
      facts.outputs.push_back(std::move(output));
    } else {
      ProcessCvuCanonicalOutputFact output;
      output.logical_index = static_cast<int>(i);
      output.physical_index = static_cast<int>(i);
      output.output_slot = static_cast<int>(i);
      output.tensor_index = static_cast<int>(i);
      output.physical_name = output_name;
      output.logical_name = output_name;
      output.shape = output_shape;
      output.dtype = output_dtype;
      output.layout = output_layout;
      facts.outputs.push_back(std::move(output));
    }
    facts.output_order.push_back(
        build_processcvu_published_output_route_fact(facts.outputs.back()));
  }
  return facts;
}

bool mpk_quant_contract_complete_local(const std::optional<MpkQuantContract>& quant) {
  return quant.has_value() && !quant->scales.empty() && !quant->zero_points.empty();
}

std::optional<MpkQuantContract>
resolve_processcvu_mpk_quant_contract_local(const MpkContract& contract,
                                            const MpkPluginIoContract& stage) {
  if (mpk_quant_contract_complete_local(stage.quant)) {
    return stage.quant;
  }

  const auto ordered = plugins_in_execution_order(contract);
  auto find_position = [&](const MpkPluginIoContract* want) -> std::optional<std::size_t> {
    if (!want) {
      return std::nullopt;
    }
    for (std::size_t pos = 0; pos < ordered.size(); ++pos) {
      const std::size_t idx = ordered[pos];
      if (idx < contract.plugins.size() && &contract.plugins[idx] == want) {
        return pos;
      }
    }
    return std::nullopt;
  };

  const auto stage_pos = find_position(&stage);
  if (!stage_pos.has_value()) {
    return std::nullopt;
  }
  const auto* mla_stage = get_mla_stage_io_contract(contract);
  const auto mla_pos = find_position(mla_stage);
  const std::size_t lower_bound = mla_pos.value_or(0U);
  if (*stage_pos > lower_bound) {
    for (std::size_t pos = *stage_pos; pos > lower_bound; --pos) {
      const std::size_t idx = ordered[pos - 1U];
      if (idx >= contract.plugins.size()) {
        continue;
      }
      const auto& candidate = contract.plugins[idx];
      if (mpk_quant_contract_complete_local(candidate.quant)) {
        return candidate.quant;
      }
    }
  }
  if (lower_bound < ordered.size()) {
    const std::size_t idx = ordered[lower_bound];
    if (idx < contract.plugins.size() &&
        mpk_quant_contract_complete_local(contract.plugins[idx].quant)) {
      return contract.plugins[idx].quant;
    }
  }
  return std::nullopt;
}

std::pair<double, std::int64_t>
require_uniform_dequant_params_local(const MpkPluginIoContract& stage) {
  if (!stage.quant.has_value() || stage.quant->scales.empty() || stage.quant->zero_points.empty()) {
    throw std::runtime_error("processcvu MPK dequant stage '" + stage.name +
                             "' is missing quant facts");
  }

  const double scale = stage.quant->scales.front();
  const std::int64_t zp = stage.quant->zero_points.front();
  const auto scale_differs = [&](double candidate) { return std::abs(candidate - scale) > 1e-12; };
  const auto zp_differs = [&](std::int64_t candidate) { return candidate != zp; };
  if (std::any_of(stage.quant->scales.begin(), stage.quant->scales.end(), scale_differs) ||
      std::any_of(stage.quant->zero_points.begin(), stage.quant->zero_points.end(), zp_differs)) {
    throw std::runtime_error("processcvu MPK dequant stage '" + stage.name +
                             "' requires unsupported per-channel quant facts");
  }
  return {scale, zp};
}

void require_positive_mpk_fact_local(int value, const std::string& context, const char* fact_name) {
  if (value > 0) {
    return;
  }
  throw std::runtime_error(context + " requires explicit MPK " + fact_name + ".");
}

std::string require_string_mpk_fact_local(std::string value, const std::string& context,
                                          const char* fact_name) {
  if (!value.empty()) {
    return value;
  }
  throw std::runtime_error(context + " requires explicit MPK " + fact_name + ".");
}

std::string normalize_dtype_token_local(std::string raw) {
  raw = upper_copy_local(std::move(raw));
  if (raw.find("BFLOAT16") != std::string::npos || raw.find("BF16") != std::string::npos) {
    return "BF16";
  }
  if (raw.find("FLOAT32") != std::string::npos || raw.find("FP32") != std::string::npos) {
    return "FP32";
  }
  if (raw.find("FLOAT16") != std::string::npos || raw.find("FP16") != std::string::npos) {
    return "FP16";
  }
  if (raw.find("UINT8") != std::string::npos || raw == "U8") {
    return "UINT8";
  }
  if (raw.find("INT8") != std::string::npos) {
    return "INT8";
  }
  if (raw.find("UINT16") != std::string::npos) {
    return "UINT16";
  }
  if (raw.find("INT16") != std::string::npos) {
    return "INT16";
  }
  if (raw.find("UINT32") != std::string::npos) {
    return "UINT32";
  }
  if (raw.find("INT32") != std::string::npos) {
    return "INT32";
  }
  return raw;
}

std::string preferred_tensor_dtype_local(const MpkTensorContract& tensor,
                                         const std::string& fallback = {}) {
  if (!tensor.logical_dtype.empty()) {
    return normalize_dtype_token_local(tensor.logical_dtype);
  }
  if (!tensor.dtype.empty()) {
    return normalize_dtype_token_local(tensor.dtype);
  }
  return normalize_dtype_token_local(fallback);
}

std::uint64_t packed_tensor_size_bytes_local(const std::vector<std::int64_t>& shape,
                                             const std::string& dtype) {
  if (shape.empty()) {
    return 0U;
  }
  std::uint64_t elems = 1U;
  for (const auto dim : shape) {
    if (dim <= 0) {
      return 0U;
    }
    elems *= static_cast<std::uint64_t>(dim);
  }
  return elems * processcvu_dtype_size_bytes_from_token(dtype);
}

std::uint64_t preferred_mpk_tensor_size_bytes_local(const MpkTensorContract& tensor,
                                                    const std::string& dtype) {
  if (tensor.size_bytes > 0U) {
    return static_cast<std::uint64_t>(tensor.size_bytes);
  }
  if (!tensor.logical_shape.empty()) {
    return packed_tensor_size_bytes_local(tensor.logical_shape, dtype);
  }
  if (tensor.shape_semantics == MpkShapeSemantics::Geometry) {
    return packed_tensor_size_bytes_local(tensor.mpk_shape, dtype);
  }
  return 0U;
}

std::uint64_t round_up_to_multiple_local(std::uint64_t value, std::uint64_t multiple) {
  if (value == 0U || multiple == 0U) {
    return value;
  }
  const std::uint64_t rem = value % multiple;
  return rem == 0U ? value : (value + multiple - rem);
}

std::uint64_t leading_extent_product_local(const std::vector<std::int64_t>& shape,
                                           std::size_t suffix_rank) {
  if (shape.size() <= suffix_rank) {
    return 1U;
  }
  std::uint64_t total = 1U;
  for (std::size_t i = 0; i + suffix_rank < shape.size(); ++i) {
    if (shape[i] <= 0) {
      return 0U;
    }
    total *= static_cast<std::uint64_t>(shape[i]);
  }
  return total;
}

std::optional<DetessPackedInputDims>
project_detess_packed_input_dims_local(std::vector<std::int64_t> shape) {
  if (shape.size() >= 4U && shape.front() == 1) {
    shape.erase(shape.begin());
  }
  if (shape.empty()) {
    return std::nullopt;
  }
  const auto fits_positive_int = [](std::int64_t value) {
    return value > 0 && value <= static_cast<std::int64_t>(std::numeric_limits<int>::max());
  };
  for (const auto dim : shape) {
    if (!fits_positive_int(dim)) {
      return std::nullopt;
    }
  }

  DetessPackedInputDims out;
  if (shape.size() >= 3U) {
    out.height = static_cast<int>(shape[shape.size() - 3U]);
    out.width = static_cast<int>(shape[shape.size() - 2U]);
    out.channels = static_cast<int>(shape.back());
    const std::uint64_t leading_depth = leading_extent_product_local(shape, 3U);
    if (leading_depth == 0U ||
        leading_depth > static_cast<std::uint64_t>(std::numeric_limits<int>::max())) {
      return std::nullopt;
    }
    out.depth = static_cast<int>(leading_depth);
    return out;
  }
  if (shape.size() == 2U) {
    out.height = static_cast<int>(shape[0]);
    out.width = static_cast<int>(shape[1]);
    out.depth = 1;
    out.channels = 1;
    return out;
  }
  out.height = 1;
  out.width = static_cast<int>(shape[0]);
  out.depth = 1;
  out.channels = 1;
  return out;
}

std::uint64_t
detess_packed_input_size_bytes_from_projected_dims_local(const DetessPackedInputDims& dims,
                                                         const bool align_c16, const bool cblock,
                                                         const std::string& dtype) {
  if (dims.width <= 0 || dims.height <= 0 || dims.depth <= 0 || dims.channels <= 0 ||
      dtype.empty()) {
    return 0U;
  }
  std::uint64_t channels = static_cast<std::uint64_t>(dims.channels);
  const bool align_channels = align_c16 || cblock;
  if (align_channels) {
    channels = round_up_to_multiple_local(channels, 16U);
  }
  const std::uint64_t elem_bytes = processcvu_dtype_size_bytes_from_token(dtype);
  if (elem_bytes == 0U) {
    return 0U;
  }
  const std::uint64_t factors[] = {static_cast<std::uint64_t>(dims.height),
                                   static_cast<std::uint64_t>(dims.width),
                                   static_cast<std::uint64_t>(dims.depth), channels, elem_bytes};
  std::uint64_t total = 1U;
  for (const std::uint64_t factor : factors) {
    if (factor == 0U) {
      return 0U;
    }
    if (total > std::numeric_limits<std::uint64_t>::max() / factor) {
      return 0U;
    }
    total *= factor;
  }
  return total;
}

std::uint64_t detess_packed_input_size_bytes_from_projected_dims_local(
    const DetessPackedInputDims& dims, const MpkPluginIoContract& stage, const std::string& dtype) {
  return detess_packed_input_size_bytes_from_projected_dims_local(
      dims, stage.has_align_c16 && stage.align_c16, stage.has_cblock && stage.cblock, dtype);
}

std::uint64_t logical_mpk_tensor_size_bytes_local(const MpkTensorContract& tensor,
                                                  const std::string& dtype) {
  if (!tensor.logical_shape.empty()) {
    return packed_tensor_size_bytes_local(tensor.logical_shape, dtype);
  }
  if (tensor.shape_semantics == MpkShapeSemantics::Geometry) {
    return packed_tensor_size_bytes_local(tensor.mpk_shape, dtype);
  }
  if (tensor.size_bytes > 0U) {
    return static_cast<std::uint64_t>(tensor.size_bytes);
  }
  return 0U;
}

std::uint64_t expected_detess_packed_input_size_bytes_local(const MpkPluginIoContract& stage,
                                                            const std::string& dtype) {
  if (stage.frame_shape.empty() || dtype.empty()) {
    return 0U;
  }
  const MpkTensorDims dims = dims_from_mpk_shape(stage.frame_shape);
  if (dims.width <= 0 || dims.height <= 0 || dims.depth <= 0) {
    return 0U;
  }
  const std::uint64_t batch = leading_extent_product_local(stage.frame_shape, 3U);
  if (batch == 0U) {
    return 0U;
  }
  DetessPackedInputDims packed;
  packed.width = dims.width;
  packed.height = dims.height;
  packed.depth = static_cast<int>(batch);
  packed.channels = dims.depth;
  return detess_packed_input_size_bytes_from_projected_dims_local(
      packed, stage.has_align_c16 && stage.align_c16, stage.has_cblock && stage.cblock, dtype);
}

std::uint64_t
expected_detess_packed_input_size_bytes_local(const std::vector<std::int64_t>& frame_shape,
                                              const bool align_c16, const bool cblock,
                                              const std::string& dtype) {
  if (frame_shape.empty() || dtype.empty()) {
    return 0U;
  }
  const std::uint64_t elem_bytes = processcvu_dtype_size_bytes_from_token(dtype);
  if (elem_bytes == 0U) {
    return 0U;
  }
  const MpkTensorDims dims = dims_from_mpk_shape(frame_shape);
  if (dims.width <= 0 || dims.height <= 0 || dims.depth <= 0) {
    return 0U;
  }
  const std::uint64_t batch = leading_extent_product_local(frame_shape, 3U);
  if (batch == 0U) {
    return 0U;
  }
  std::uint64_t channels = static_cast<std::uint64_t>(dims.depth);
  if (align_c16 || cblock) {
    channels = round_up_to_multiple_local(channels, 16U);
  }
  return batch * static_cast<std::uint64_t>(dims.height) * static_cast<std::uint64_t>(dims.width) *
         channels * elem_bytes;
}

std::uint64_t detess_tiled_input_size_bytes_local(const ShapeDims& input_dims,
                                                  const SliceDims& slice_dims,
                                                  const std::string& dtype) {
  if (input_dims.width <= 0 || input_dims.height <= 0 || input_dims.depth <= 0 ||
      input_dims.channels <= 0 || slice_dims.w <= 0 || slice_dims.h <= 0 || slice_dims.d <= 0 ||
      slice_dims.c <= 0 || dtype.empty()) {
    return 0U;
  }
  const std::uint64_t elem_bytes = processcvu_dtype_size_bytes_from_token(dtype);
  if (elem_bytes == 0U) {
    return 0U;
  }
  std::uint64_t total = 0U;
  for (int c0 = 0; c0 < input_dims.channels; c0 += slice_dims.c) {
    const std::uint64_t tile_channels =
        static_cast<std::uint64_t>(std::min(slice_dims.c, input_dims.channels - c0));
    const std::uint64_t aligned_channels = round_up_to_multiple_local(tile_channels, 16U);
    for (int d0 = 0; d0 < input_dims.depth; d0 += slice_dims.d) {
      const std::uint64_t tile_depth =
          static_cast<std::uint64_t>(std::min(slice_dims.d, input_dims.depth - d0));
      for (int h0 = 0; h0 < input_dims.height; h0 += slice_dims.h) {
        const std::uint64_t tile_height =
            static_cast<std::uint64_t>(std::min(slice_dims.h, input_dims.height - h0));
        for (int w0 = 0; w0 < input_dims.width; w0 += slice_dims.w) {
          const std::uint64_t tile_width =
              static_cast<std::uint64_t>(std::min(slice_dims.w, input_dims.width - w0));
          std::uint64_t tile_bytes =
              multiply_u64_checked_local(tile_width, tile_height, "detess tiled input bytes");
          tile_bytes =
              multiply_u64_checked_local(tile_bytes, tile_depth, "detess tiled input bytes");
          tile_bytes =
              multiply_u64_checked_local(tile_bytes, aligned_channels, "detess tiled input bytes");
          tile_bytes =
              multiply_u64_checked_local(tile_bytes, elem_bytes, "detess tiled input bytes");
          if (tile_bytes > std::numeric_limits<std::uint64_t>::max() - total) {
            throw std::runtime_error(
                "processcvu overflow while computing detess tiled input bytes");
          }
          total += tile_bytes;
        }
      }
    }
  }
  return total;
}

int positive_tile_channels_local(const MpkPluginIoContract& stage) {
  const auto sd = dims_from_slice_shape(stage.slice_shape);
  int input_channels = 0;
  if (!stage.input_tensors.empty()) {
    const auto input_shape =
        preferred_stage_input_tensor_shape_local(stage, stage.input_tensors.front());
    if (!input_shape.empty()) {
      input_channels = static_cast<int>(input_shape.back());
    }
  }
  return pick_first_positive_dim({sd.c, sd.d, input_channels, 1});
}

int logical_channels_from_dims_local(const MpkTensorDims& dims) {
  return upper_copy_local(dims.layout) == "HW" ? 1 : std::max(1, dims.depth);
}

int logical_depth_from_dims_local(const MpkTensorDims& dims) {
  const std::string format = upper_copy_local(dims.layout);
  if (format == "HW") {
    return 1;
  }
  return std::max(1, dims.depth);
}

std::string canonical_mpk_stage_kind_name(const MpkPluginIoContract& stage) {
  const std::string raw = lower_copy(!stage.kernel.empty() ? stage.kernel : stage.name);
  if (raw.find("detessdequant") != std::string::npos) {
    return "detessdequant";
  }
  if (raw.find("quanttess") != std::string::npos ||
      (raw.find("quant") != std::string::npos && raw.find("tess") != std::string::npos)) {
    return "quanttess";
  }
  if (raw.find("detess") != std::string::npos) {
    return "detess";
  }
  if (raw.find("dequant") != std::string::npos) {
    return "dequant";
  }
  if (raw.find("boxdecode") != std::string::npos || raw.find("objectdecode") != std::string::npos) {
    return "boxdecode";
  }
  if (raw.find("infer") != std::string::npos || raw.find("mla") != std::string::npos) {
    return "mla";
  }
  if (raw.find("tess") != std::string::npos) {
    return "tess";
  }
  if (raw.find("quant") != std::string::npos) {
    return "quant";
  }
  if (raw.find("cast") != std::string::npos) {
    return "cast";
  }
  if (raw.find("preproc") != std::string::npos) {
    return "preproc";
  }
  return raw;
}

std::vector<std::size_t> ordered_plugin_indices_local(const MpkContract& contract) {
  auto ordered = plugins_in_execution_order(contract);
  std::vector<bool> seen(contract.plugins.size(), false);
  for (const std::size_t idx : ordered) {
    if (idx < seen.size()) {
      seen[idx] = true;
    }
  }
  for (std::size_t idx = 0; idx < contract.plugins.size(); ++idx) {
    if (!seen[idx]) {
      ordered.push_back(idx);
    }
  }
  return ordered;
}

std::optional<std::size_t> mla_rank_in_order_local(const MpkContract& contract,
                                                   const std::vector<std::size_t>& ordered) {
  const auto* mla_stage = get_mla_stage_io_contract(contract);
  if (!mla_stage) {
    return std::nullopt;
  }
  for (std::size_t rank = 0; rank < ordered.size(); ++rank) {
    const std::size_t idx = ordered[rank];
    if (idx < contract.plugins.size() && &contract.plugins[idx] == mla_stage) {
      return rank;
    }
  }
  return std::nullopt;
}

const MpkPluginIoContract*
find_pre_stage_for_kind_names_local(const MpkContract& contract,
                                    std::initializer_list<const char*> preferred) {
  const auto ordered = ordered_plugin_indices_local(contract);
  const auto mla_rank = mla_rank_in_order_local(contract, ordered);
  for (const char* wanted : preferred) {
    if (!wanted || !*wanted) {
      continue;
    }
    for (std::size_t rank = 0; rank < ordered.size(); ++rank) {
      if (mla_rank.has_value() && !(rank < *mla_rank)) {
        break;
      }
      const std::size_t idx = ordered[rank];
      if (idx >= contract.plugins.size()) {
        continue;
      }
      if (canonical_mpk_stage_kind_name(contract.plugins[idx]) == wanted) {
        return &contract.plugins[idx];
      }
    }
  }
  return nullptr;
}

std::optional<std::size_t> plugin_index_from_stage_ptr_local(const MpkContract& contract,
                                                             const MpkPluginIoContract* stage) {
  if (!stage) {
    return std::nullopt;
  }
  for (std::size_t idx = 0; idx < contract.plugins.size(); ++idx) {
    if (&contract.plugins[idx] == stage) {
      return idx;
    }
  }
  return std::nullopt;
}

const MpkPluginIoContract*
resolve_connected_pre_stage_for_kind_local(const MpkContract& contract,
                                           const MpkPluginIoContract& anchor_stage,
                                           const char* wanted_kind, bool walk_upstream) {
  if (!wanted_kind || !*wanted_kind) {
    return nullptr;
  }
  const auto anchor_index = plugin_index_from_stage_ptr_local(contract, &anchor_stage);
  if (!anchor_index.has_value()) {
    return nullptr;
  }

  std::queue<std::size_t> pending;
  std::unordered_set<std::size_t> visited;
  pending.push(*anchor_index);
  visited.insert(*anchor_index);

  const MpkPluginIoContract* matched = nullptr;
  while (!pending.empty()) {
    const std::size_t current = pending.front();
    pending.pop();

    for (const auto& edge : contract.edges) {
      const bool edge_matches =
          walk_upstream ? (edge.dst_plugin_index == current) : (edge.src_plugin_index == current);
      if (!edge_matches) {
        continue;
      }

      const std::size_t next = walk_upstream ? edge.src_plugin_index : edge.dst_plugin_index;
      if (next >= contract.plugins.size() || !visited.insert(next).second) {
        continue;
      }

      const auto& candidate = contract.plugins[next];
      const std::string kind = canonical_mpk_stage_kind_name(candidate);
      if (kind == wanted_kind) {
        if (matched != nullptr && matched != &candidate) {
          throw std::runtime_error(
              std::string("processcvu MPK pre-adapter stage found multiple '") + wanted_kind +
              "' matches while resolving exact stage '" + anchor_stage.name + "'");
        }
        matched = &candidate;
        continue;
      }
      if (kind == "mla") {
        continue;
      }
      pending.push(next);
    }
  }

  return matched;
}

// One sibling branch upstream of the MLA: the "primary" pointer is the stage
// whose output is consumed by MLA (directly or via a packer), and the partner
// pointers are the connected upstream stages within the same branch. Mirrors
// `plugin_contracts::DetessDequantStagePair`, but extended for the pre-MLA
// case which has up to three roles (cast → quant → tess).
struct PreMlaSiblingStage {
  const MpkPluginIoContract* tess = nullptr;
  const MpkPluginIoContract* quant = nullptr;
  const MpkPluginIoContract* cast = nullptr;
  const MpkPluginIoContract* preproc = nullptr;
};

// Symmetric pre-MLA analog of resolve_detessdequant_stage_pairs_from_mpk: walk
// every pre-MLA plugin and return one PreMlaSiblingStage per branch the
// requested family contributes to MLA. For single-branch (monolithic) models
// this returns size 1; for native multi-IFM models it returns size N. Order is
// stabilized by the trailing index suffix on the primary stage's input tensor
// name (e.g. ..._0, ..._1, ...) — mirroring the detessdequant resolver's
// reordering logic at PluginContractSubsets.cpp:1746.
static std::vector<PreMlaSiblingStage>
resolve_pre_mla_sibling_stages_from_mpk_local(const MpkContract& contract,
                                              const std::string& family) {
  std::vector<PreMlaSiblingStage> stages;

  std::vector<const char*> primary_kinds;
  if (family == "quanttess") {
    primary_kinds = {"tess", "quanttess"};
  } else if (family == "tessellate") {
    primary_kinds = {"tess", "quanttess"};
  } else if (family == "quantize") {
    primary_kinds = {"quant", "quanttess"};
  } else if (family == "casttess") {
    primary_kinds = {"tess"};
  } else if (family == "cast") {
    primary_kinds = {"cast"};
  } else if (family == "preproc") {
    primary_kinds = {"preproc"};
  } else {
    return stages;
  }

  const auto ordered = ordered_plugin_indices_local(contract);
  const auto mla_rank = mla_rank_in_order_local(contract, ordered);

  for (std::size_t rank = 0; rank < ordered.size(); ++rank) {
    if (mla_rank.has_value() && !(rank < *mla_rank)) {
      break;
    }
    const std::size_t idx = ordered[rank];
    if (idx >= contract.plugins.size()) {
      continue;
    }
    const auto& plugin = contract.plugins[idx];
    const std::string kind = canonical_mpk_stage_kind_name(plugin);
    bool is_primary = false;
    for (const char* k : primary_kinds) {
      if (kind == k) {
        is_primary = true;
        break;
      }
    }
    if (!is_primary) {
      continue;
    }

    PreMlaSiblingStage set;
    if (kind == "preproc") {
      set.preproc = &plugin;
    } else if (kind == "quanttess") {
      set.tess = &plugin;
      set.quant = &plugin;
    } else if (kind == "tess") {
      set.tess = &plugin;
      if (family == "quanttess") {
        set.quant = resolve_connected_pre_stage_for_kind_local(contract, plugin, "quant", true);
      } else if (family == "casttess") {
        set.cast = resolve_connected_pre_stage_for_kind_local(contract, plugin, "cast", true);
      }
    } else if (kind == "quant") {
      set.quant = &plugin;
    } else if (kind == "cast") {
      set.cast = &plugin;
    }
    stages.push_back(set);
  }

  if (stages.size() <= 1U) {
    return stages;
  }

  std::vector<std::pair<int, std::size_t>> ranked;
  ranked.reserve(stages.size());
  bool can_sort = true;
  for (std::size_t i = 0; i < stages.size(); ++i) {
    const auto* anchor = stages[i].tess;
    if (!anchor) {
      anchor = stages[i].quant  ? stages[i].quant
               : stages[i].cast ? stages[i].cast
                                : stages[i].preproc;
    }
    if (!anchor || anchor->input_tensors.empty() || anchor->input_tensors.front().name.empty()) {
      can_sort = false;
      break;
    }
    const std::string& name = anchor->input_tensors.front().name;
    std::size_t pos = name.size();
    while (pos > 0U && std::isdigit(static_cast<unsigned char>(name[pos - 1U]))) {
      --pos;
    }
    if (pos == name.size() || pos == 0U || name[pos - 1U] != '_') {
      can_sort = false;
      break;
    }
    try {
      ranked.emplace_back(std::stoi(name.substr(pos)), i);
    } catch (const std::exception&) {
      can_sort = false;
      break;
    }
  }
  if (!can_sort) {
    return stages;
  }
  std::sort(ranked.begin(), ranked.end(),
            [](const auto& lhs, const auto& rhs) { return lhs.first < rhs.first; });
  for (std::size_t i = 1; i < ranked.size(); ++i) {
    if (ranked[i - 1U].first == ranked[i].first) {
      return stages;
    }
  }
  std::vector<PreMlaSiblingStage> reordered;
  reordered.reserve(stages.size());
  for (const auto& [_, original_index] : ranked) {
    reordered.push_back(stages[original_index]);
  }
  return reordered;
}

// Compute a permutation `out` such that branch_keys[out[i]] == boundary_keys[i]
// for all i, or std::nullopt if alignment is impossible (size mismatch, empty
// key, duplicate branch key, or unmatched boundary key). Used by the post-MLA
// fan-out renderers (detess/detesscast/detessdequant) to align sibling branches
// to MLA published outputs, and by the pre-MLA fan-in renderer to align
// sibling tess outputs to the IFM pack stage's input list (or the MLA's
// physical input list for native multi-IFM models).
std::optional<std::vector<std::size_t>>
reorder_indices_by_mla_boundary_local(const std::vector<std::string>& branch_keys,
                                      const std::vector<std::string>& boundary_keys) {
  if (branch_keys.empty() || branch_keys.size() != boundary_keys.size()) {
    return std::nullopt;
  }
  std::unordered_map<std::string, std::size_t> idx_by_key;
  idx_by_key.reserve(branch_keys.size());
  for (std::size_t i = 0; i < branch_keys.size(); ++i) {
    if (branch_keys[i].empty()) {
      return std::nullopt;
    }
    if (!idx_by_key.emplace(branch_keys[i], i).second) {
      return std::nullopt;
    }
  }
  std::vector<std::size_t> permutation;
  permutation.reserve(boundary_keys.size());
  for (const auto& key : boundary_keys) {
    if (key.empty()) {
      return std::nullopt;
    }
    const auto found = idx_by_key.find(key);
    if (found == idx_by_key.end()) {
      return std::nullopt;
    }
    permutation.push_back(found->second);
  }
  return permutation;
}

// Returns the stage whose `input_tensors` list carries the canonical sibling
// branch order at the MLA boundary. For native multi-IFM models (MLA consumes
// N>1 distinct physical inputs without an upstream pack stage) this is the
// MLA stage itself. For packer-style models (e.g. rpn_head_640_640_concat_4d)
// this is the IFM pack stage that consumes the N tess siblings before MLA.
// Returns nullptr if neither is unambiguously identifiable.
const MpkPluginIoContract* find_pre_mla_boundary_stage_local(const MpkContract& contract) {
  if (mla_consumer_keeps_distinct_physical_inputs(contract)) {
    return get_mla_stage_io_contract(contract);
  }
  const auto ordered = ordered_plugin_indices_local(contract);
  const auto mla_rank = mla_rank_in_order_local(contract, ordered);
  if (!mla_rank.has_value()) {
    return nullptr;
  }
  for (std::size_t rank = 0; rank < *mla_rank; ++rank) {
    const auto idx = ordered[rank];
    if (idx >= contract.plugins.size()) {
      continue;
    }
    const auto& plugin = contract.plugins[idx];
    const std::string raw = lower_copy(plugin.kernel);
    if (raw.find("pack") != std::string::npos && raw.find("unpack") == std::string::npos) {
      return &plugin;
    }
  }
  return nullptr;
}

// Reorders pre-MLA siblings to match the MLA boundary stage's input order
// (IFM pack for packer-style models, MLA itself for native multi-IFM models).
// Falls back to the resolver's natural (suffix-based) ordering if any branch
// key is missing or none of the branch keys aligns with the boundary keys.
void reorder_pre_mla_siblings_by_boundary_local(const MpkContract& contract,
                                                std::vector<PreMlaSiblingStage>* siblings) {
  if (siblings == nullptr || siblings->size() <= 1U) {
    return;
  }
  const auto* boundary = find_pre_mla_boundary_stage_local(contract);
  if (boundary == nullptr) {
    return;
  }
  std::vector<std::string> branch_keys;
  branch_keys.reserve(siblings->size());
  for (const auto& sib : *siblings) {
    const auto* anchor = sib.tess ? sib.tess : (sib.quant ? sib.quant : sib.cast);
    if (!anchor || anchor->output_tensors.empty() || anchor->output_tensors.front().name.empty()) {
      return;
    }
    branch_keys.push_back(anchor->output_tensors.front().name);
  }
  std::vector<std::string> boundary_keys;
  boundary_keys.reserve(boundary->input_tensors.size());
  for (const auto& it : boundary->input_tensors) {
    boundary_keys.push_back(it.name);
  }
  if (auto perm = reorder_indices_by_mla_boundary_local(branch_keys, boundary_keys)) {
    std::vector<PreMlaSiblingStage> reordered;
    reordered.reserve(perm->size());
    for (auto idx : *perm) {
      reordered.push_back((*siblings)[idx]);
    }
    *siblings = std::move(reordered);
  }
}

// Forward declarations of helpers that live further down in this file but are
// called by the symmetric pre-MLA multi-IO renderer below.
plugin_contracts::QuantTessContractSubset
build_quanttess_contract_subset_for_exact_stage_local(const MpkContract& contract,
                                                      const MpkPluginIoContract* exact_stage);

plugin_contracts::CastContractSubset
build_preadapter_cast_subset_for_tess_stage_local(const MpkContract& contract,
                                                  const MpkPluginIoContract& tess_stage);

CompiledProcessCvuRuntimeConfig build_preadapter_cast_runtime_config_local(
    const MpkContract& contract, const MpkPluginIoContract* input_stage,
    const MpkPluginIoContract* output_stage, const ProcessCvuSingleOutputIdentity& output_identity,
    const std::optional<std::string>& exact_stage_name_or_id) {
  CompiledProcessCvuRuntimeConfig runtime;
  runtime.graph_family = "cast";
  runtime.graph_name = "cast";
  runtime.graph_id = 221;
  runtime.default_input_name = "input_tensor";
  runtime.runtime_input_names = {"input_tensor"};
  runtime.physical_input_names = {"input_tensor"};
  apply_processcvu_single_output_identity_local(&runtime, output_identity);
  runtime.batch_size = 1;
  runtime.byte_align = 1;

  const auto preferred_input_shape = (input_stage && !input_stage->input_tensors.empty())
                                         ? preferred_stage_input_tensor_shape_local(
                                               *input_stage, input_stage->input_tensors.front())
                                         : std::vector<std::int64_t>{};
  const auto preferred_output_shape =
      (output_stage && !output_stage->output_tensors.empty())
          ? preferred_mpk_tensor_shape_local(output_stage->output_tensors.front())
          : std::vector<std::int64_t>{};
  if (preferred_input_shape.empty() || preferred_output_shape.empty()) {
    throw std::runtime_error(
        "processcvu MPK cast pre-adapter stage requires semantic input/output tensor shapes");
  }
  if (preferred_input_shape.size() != preferred_output_shape.size()) {
    throw std::runtime_error("processcvu MPK cast pre-adapter stage requires matching ranks");
  }
  for (std::size_t axis = 0; axis < preferred_input_shape.size(); ++axis) {
    if (preferred_input_shape[axis] != preferred_output_shape[axis]) {
      throw std::runtime_error(
          "processcvu MPK cast pre-adapter stage requires matching logical shapes");
    }
  }

  runtime.input_shapes = {
      std::vector<int>(preferred_input_shape.begin(), preferred_input_shape.end())};
  runtime.output_shapes = {
      std::vector<int>(preferred_output_shape.begin(), preferred_output_shape.end())};
  runtime.input_dtype = normalize_dtype_token_local(preferred_tensor_dtype_local(
      (input_stage && !input_stage->input_tensors.empty()) ? input_stage->input_tensors.front()
                                                           : MpkTensorContract{},
      input_stage ? input_stage->canonical_input_dtype : std::string{}));
  runtime.output_dtype = normalize_dtype_token_local(preferred_tensor_dtype_local(
      (output_stage && !output_stage->output_tensors.empty()) ? output_stage->output_tensors.front()
                                                              : MpkTensorContract{},
      output_stage ? output_stage->canonical_output_dtype : std::string{}));
  runtime.input_dtype = require_string_mpk_fact_local(
      std::move(runtime.input_dtype), "processcvu MPK cast pre-adapter stage", "input dtype");
  runtime.output_dtype = require_string_mpk_fact_local(
      std::move(runtime.output_dtype), "processcvu MPK cast pre-adapter stage", "output dtype");
  if (!((runtime.input_dtype == "FP32" && runtime.output_dtype == "BF16") ||
        (runtime.input_dtype == "BF16" && runtime.output_dtype == "FP32"))) {
    throw std::runtime_error(
        "processcvu MPK cast pre-adapter stage requires FP32<->BF16 dtype pair");
  }
  runtime.out_dtype = runtime.output_dtype;

  sima_ev_tensor_desc input_desc{};
  sima_ev_tensor_desc output_desc{};
  if (!build_tensor_dense_desc_processcvu_local(runtime.input_shapes.front(), runtime.input_dtype,
                                                &input_desc) ||
      !build_tensor_dense_desc_processcvu_local(runtime.output_shapes.front(), runtime.output_dtype,
                                                &output_desc)) {
    throw std::runtime_error(
        "processcvu MPK cast pre-adapter stage could not synthesize explicit typed tensors");
  }
  runtime.input_tensors = {input_desc};
  runtime.output_tensors = {output_desc};
  runtime.runtime_output_logical_index_list = {0};
  runtime.runtime_output_output_slot_list = {0};
  runtime.runtime_output_physical_index_list = {0};
  runtime.runtime_output_dtype_list = {runtime.output_dtype};
  runtime.runtime_output_transport_kind_list = {ProcessCvuOutputTransportKind::Dense};
  runtime.runtime_output_semantic_kind_list = {ProcessCvuOutputSemanticKind::Tensor};
  runtime.runtime_output_logical_shapes = {runtime.output_shapes.front()};
  runtime.runtime_output_logical_layout_list = {runtime_output_layout_token_local(runtime)};
  runtime.primary_output_transport_kind = ProcessCvuOutputTransportKind::Dense;
  runtime.primary_output_semantic_kind = ProcessCvuOutputSemanticKind::Tensor;
  require_graph_processcvu_runtime_geometry_local(contract, "cast", exact_stage_name_or_id,
                                                  &runtime);
  return runtime;
}

// Per-family per-branch runtime descriptors. For each pre-MLA sibling, this
// returns a single-IO `CompiledProcessCvuRuntimeConfig` produced by the
// family's existing runtime-config builder. The N per-branch runtimes are then
// merged by the caller into a single packed-multi-IO runtime config (mirroring
// build_dequantize_runtime_config_from_subsets / build_detessdequant_runtime_
// config_from_subset on the post side).
CompiledProcessCvuRuntimeConfig
build_pre_mla_branch_runtime_config_local(const MpkContract& contract, const std::string& family,
                                          const PreMlaSiblingStage& sib,
                                          const std::string& published_output_name) {
  if (family == "quanttess") {
    const auto* anchor = sib.tess ? sib.tess : sib.quant;
    if (!anchor) {
      throw std::runtime_error("processcvu MPK quanttess fan-in sibling missing primary stage");
    }
    const auto subset = build_quanttess_contract_subset_for_exact_stage_local(contract, anchor);
    return plugin_contracts::build_quanttess_runtime_config_from_subset(subset,
                                                                        published_output_name);
  }
  if (family == "quantize") {
    if (!sib.quant) {
      throw std::runtime_error("processcvu MPK quantize fan-in sibling missing quant stage");
    }
    const auto subset = plugin_contracts::extract_quantize_contract_subset_from_stage(*sib.quant);
    return plugin_contracts::build_quantize_runtime_config_from_subset(
        subset, published_output_name, published_output_name);
  }
  if (family == "cast") {
    if (!sib.cast) {
      throw std::runtime_error("processcvu MPK cast fan-in sibling missing cast stage");
    }
    const auto output_identity = build_processcvu_single_output_identity_local(
        published_output_name, published_output_name, published_output_name);
    return build_preadapter_cast_runtime_config_local(contract, sib.cast, sib.cast, output_identity,
                                                      std::nullopt);
  }
  if (family == "tessellate" || family == "casttess") {
    if (!sib.tess) {
      throw std::runtime_error("processcvu MPK tessellate fan-in sibling missing tess stage");
    }
    const auto cast_subset =
        sib.cast ? plugin_contracts::extract_cast_contract_subset_from_stage(*sib.cast)
                 : build_preadapter_cast_subset_for_tess_stage_local(contract, *sib.tess);
    const auto tess_subset =
        plugin_contracts::extract_tessellate_contract_subset_from_stage(*sib.tess);
    auto runtime = plugin_contracts::build_tessellate_runtime_config_from_subsets(
        cast_subset, tess_subset, published_output_name, published_output_name);
    if (family == "casttess") {
      runtime.graph_family = "casttess";
      runtime.graph_name = "casttess";
      runtime.graph_id = 224;
    }
    return runtime;
  }
  throw std::runtime_error(std::string("processcvu MPK pre-MLA fan-in does not support family '") +
                           family + "'");
}

// Picks the per-branch published output tensor name from the sibling's MPK
// metadata. This is the segment identity downstream consumers (the kernel CM,
// the GStreamer element's logical output table, the dispatcher's published
// segment table) all key on. Falls back to a synthetic numbered name only if
// the MPK genuinely has no name on the sibling's output.
std::string pre_mla_branch_published_output_name_local(const PreMlaSiblingStage& sib,
                                                       std::size_t branch_index) {
  const auto* anchor =
      sib.tess ? sib.tess : (sib.quant ? sib.quant : (sib.cast ? sib.cast : sib.preproc));
  if (anchor != nullptr && !anchor->output_tensors.empty() &&
      !anchor->output_tensors.front().name.empty()) {
    return anchor->output_tensors.front().name;
  }
  return "output_tensor_" + std::to_string(branch_index);
}

// Returns the per-branch upstream tensor identity that feeds the i-th sibling
// at the MLA-boundary. For native multi-IFM models the boundary stage IS the
// MLA itself and `boundary->input_tensors[i]` directly names the i-th MLA
// input tensor. For packer-style models the boundary stage is the IFM packer
// and `boundary->input_tensors[i]` names the i-th tessellated input that the
// packer consumes (which is exactly the i-th sibling's published output —
// i.e. the same identity rendered from the consumer side). Either way the
// result is an `MpkTensorContract` carrying the per-branch source segment
// name, dtype, shape and (when available) byte_offset / source_physical_index
// — the same shape that `mla_published_outputs` carries on the post side, so
// `apply_published_routed_input_bindings` consumes both uniformly.
const MpkTensorContract* pre_mla_branch_upstream_input_tensor_local(
    const MpkPluginIoContract& boundary, const PreMlaSiblingStage& sib, std::size_t branch_index) {
  // The "upstream input" of a pre-MLA fan-in branch is the tensor feeding the
  // first sibling stage (quant/cast/tess/preproc) — the FP32 ingress. The
  // boundary's input_tensors[i] is the MLA's input (this branch's OUTPUT,
  // typically INT8/BF16 tiled), so using it here would propagate the
  // post-quanttess dtype/size into the input binding's storage.nbytes and
  // break the kernel's strided_required vs storage.nbytes check.
  // Symmetric to detessdequant on the post side using MLA's published
  // *outputs* — pre side uses the sibling's *inputs*.
  const auto* first_stage =
      sib.quant ? sib.quant : (sib.cast ? sib.cast : (sib.tess ? sib.tess : sib.preproc));
  if (first_stage != nullptr && !first_stage->input_tensors.empty()) {
    return &first_stage->input_tensors.front();
  }
  // Last-resort fallback if no sibling stages were resolved: use the
  // MLA-boundary input. This is structurally wrong (wrong dtype/size) but
  // preserves prior behavior for any path that previously relied on it.
  if (branch_index < boundary.input_tensors.size()) {
    return &boundary.input_tensors[branch_index];
  }
  return nullptr;
}

// Symmetric counterpart of build_processcvu_mpk_dequant_compile_inputs_local /
// build_processcvu_mpk_detessdequant_compile_inputs_local for the pre-MLA
// fan-in case. The appsrc/pre-adapter ingress boundary is always one packed
// parent input, but the output boundary follows the next consumer:
//   - packer-style MLA paths publish one packed parent output_tensor
//   - native multi-IFM MLA paths publish N distinct physical outputs
//     (quantize_0/quantize_1, cast_0/cast_1, ...).
// The logical arrays remain size N in both cases.
//   - physical_input_names  = {"input_tensor"}  (single CM-keyed input buffer)
//   - input_bindings of size N, each `sink_pad=0` / `physical_index=0` with
//     `cm_input=input_tensor` and `source_segment=<MPK ingress tensor name>`.
// This is the form the GStreamer element's CM init demands for multi-IO over
// packed transport (see processcvu_facts_from_runtime_config_impl in
// ProcessCvuRuntimeConfigAdapter.cpp); the output representation is normalized
// per boundary instead of forcing a global multi-input policy.
ProcessCvuCanonicalCompileInputs build_processcvu_mpk_pre_mla_multi_io_compile_inputs_local(
    const MpkContract& contract, const std::string& family,
    std::vector<PreMlaSiblingStage> siblings) {
  if (siblings.size() <= 1U) {
    throw std::runtime_error("processcvu MPK pre-MLA multi-io route requires N>1 siblings");
  }
  // Reorder siblings to match the MLA-boundary input order. Mirrors the
  // detessdequant path's reorder against mla_published_outputs.
  reorder_pre_mla_siblings_by_boundary_local(contract, &siblings);
  const auto* boundary = find_pre_mla_boundary_stage_local(contract);
  if (boundary == nullptr) {
    throw std::runtime_error(
        "processcvu MPK pre-MLA multi-io route requires a resolvable boundary stage");
  }

  const std::size_t count = siblings.size();
  if (boundary->input_tensors.size() < count) {
    throw std::runtime_error(
        "processcvu MPK pre-MLA boundary stage has fewer inputs than resolved siblings");
  }

  // Build the per-branch single-IO runtimes and use branch 0 as the canonical
  // template (graph_family/name/id, dtypes, byte_align, batch_size). Then
  // overwrite the per-output / per-input parallel arrays with the merged N
  // entries below.
  const bool native_distinct_mla_boundary = mla_consumer_keeps_distinct_physical_inputs(contract);
  std::vector<CompiledProcessCvuRuntimeConfig> branch_runtimes;
  branch_runtimes.reserve(count);
  std::vector<std::string> published_output_names;
  published_output_names.reserve(count);
  for (std::size_t i = 0; i < count; ++i) {
    const std::string published = pre_mla_branch_published_output_name_local(siblings[i], i);
    published_output_names.push_back(published);
    branch_runtimes.push_back(
        build_pre_mla_branch_runtime_config_local(contract, family, siblings[i], published));
  }

  CompiledProcessCvuRuntimeConfig runtime = branch_runtimes.front();
  // Single physical buffers; multi-IO is in the descriptor array + byte
  // offsets, matching the post-side packed-route layout.
  runtime.physical_input_names = {"input_tensor"};
  runtime.physical_output_names = native_distinct_mla_boundary
                                      ? published_output_names
                                      : std::vector<std::string>{"output_tensor"};
  runtime.runtime_input_names = {"input_tensor"};
  runtime.default_input_name = "input_tensor";
  // Reset per-input / per-output parallel arrays; we'll repopulate them with
  // exactly N entries below.
  runtime.input_tensors.clear();
  runtime.output_tensors.clear();
  runtime.input_shapes.clear();
  runtime.output_shapes.clear();
  runtime.slice_shapes.clear();
  runtime.q_scale_list.clear();
  runtime.q_zp_list.clear();
  runtime.dq_scale_list.clear();
  runtime.dq_zp_list.clear();
  runtime.runtime_output_names.clear();
  runtime.published_output_names.clear();
  runtime.runtime_output_logical_shapes.clear();
  runtime.runtime_output_logical_index_list.clear();
  runtime.runtime_output_output_slot_list.clear();
  runtime.runtime_output_physical_index_list.clear();
  runtime.runtime_output_dtype_list.clear();
  runtime.runtime_output_transport_kind_list.clear();
  runtime.runtime_output_semantic_kind_list.clear();
  runtime.runtime_output_logical_layout_list.clear();
  runtime.input_tensors.reserve(count);
  runtime.output_tensors.reserve(count);
  runtime.input_shapes.reserve(count);
  runtime.output_shapes.reserve(count);
  runtime.slice_shapes.reserve(count);
  runtime.runtime_output_names.reserve(count);
  runtime.published_output_names.reserve(count);
  runtime.runtime_output_logical_shapes.reserve(count);
  runtime.runtime_output_logical_index_list.reserve(count);
  runtime.runtime_output_output_slot_list.reserve(count);
  runtime.runtime_output_physical_index_list.reserve(count);
  runtime.runtime_output_dtype_list.reserve(count);
  runtime.runtime_output_transport_kind_list.reserve(count);
  runtime.runtime_output_semantic_kind_list.reserve(count);
  runtime.runtime_output_logical_layout_list.reserve(count);

  // Synthetic per-branch upstream input contracts (mirrors detessdequant's
  // make_synthetic_tensor_contract_local on packed transport). Each entry is a
  // logical view into the single packed input buffer — its size_bytes is the
  // per-branch input size, used to size the kernel descriptor's storage and to
  // accumulate the byte_offset for the next branch.
  std::vector<MpkTensorContract> synthetic_inputs;
  synthetic_inputs.reserve(count);
  std::vector<std::uint64_t> packed_input_sizes;
  packed_input_sizes.reserve(count);

  std::vector<ProcessCvuPackedRouteEntry> entries;
  entries.reserve(count);
  std::uint64_t packed_input_offset = 0U;
  std::uint64_t packed_output_offset = 0U;

  for (std::size_t i = 0; i < count; ++i) {
    const auto& branch = branch_runtimes[i];
    if (branch.input_tensors.empty() || branch.output_tensors.empty()) {
      throw std::runtime_error("processcvu MPK pre-MLA multi-io branch missing tensor descriptors");
    }
    runtime.input_tensors.push_back(branch.input_tensors.front());
    runtime.output_tensors.push_back(branch.output_tensors.front());
    if (family == "cast") {
      runtime.output_tensors.back().storage.addr = static_cast<std::uint64_t>(packed_output_offset);
    }
    if (!branch.input_shapes.empty()) {
      runtime.input_shapes.push_back(branch.input_shapes.front());
    }
    if (!branch.output_shapes.empty()) {
      runtime.output_shapes.push_back(branch.output_shapes.front());
    }
    if (!branch.slice_shapes.empty()) {
      runtime.slice_shapes.push_back(branch.slice_shapes.front());
    }
    if (!branch.runtime_output_logical_shapes.empty()) {
      runtime.runtime_output_logical_shapes.push_back(branch.runtime_output_logical_shapes.front());
    }
    runtime.runtime_output_logical_index_list.push_back(static_cast<int>(i));
    runtime.runtime_output_output_slot_list.push_back(static_cast<int>(i));
    // Packer-style MLA paths share physical output 0 and use byte offsets.
    // Native multi-IFM MLA paths keep one physical output per branch.
    runtime.runtime_output_physical_index_list.push_back(
        native_distinct_mla_boundary ? static_cast<int>(i) : 0);
    if (!branch.runtime_output_dtype_list.empty()) {
      runtime.runtime_output_dtype_list.push_back(branch.runtime_output_dtype_list.front());
    }
    if (!branch.runtime_output_transport_kind_list.empty()) {
      runtime.runtime_output_transport_kind_list.push_back(
          branch.runtime_output_transport_kind_list.front());
    }
    if (!branch.runtime_output_semantic_kind_list.empty()) {
      runtime.runtime_output_semantic_kind_list.push_back(
          branch.runtime_output_semantic_kind_list.front());
    }
    if (!branch.runtime_output_logical_layout_list.empty()) {
      runtime.runtime_output_logical_layout_list.push_back(
          branch.runtime_output_logical_layout_list.front());
    }
    // Quant param lists (q_scale_list / q_zp_list) are per semantic input for
    // multi-tensor quantize.  Older single-branch builders sometimes expose
    // the same value only through the scalar has_q_scale/has_q_zp fields, so
    // preserve that value rather than silently falling back to tensor 0's
    // default for every descriptor.  Non-quantize families leave both list and
    // scalar flags empty, so they still produce empty merged lists.
    if (!branch.q_scale_list.empty()) {
      runtime.q_scale_list.push_back(branch.q_scale_list.front());
    } else if (branch.has_q_scale) {
      runtime.q_scale_list.push_back(branch.q_scale);
    }
    if (!branch.q_zp_list.empty()) {
      runtime.q_zp_list.push_back(branch.q_zp_list.front());
    } else if (branch.has_q_zp) {
      runtime.q_zp_list.push_back(branch.q_zp);
    }

    runtime.runtime_output_names.push_back(published_output_names[i]);
    runtime.published_output_names.push_back(published_output_names[i]);

    // Synthesize the per-branch input MpkTensorContract from the upstream
    // boundary input identity. This is the pre-side counterpart of the
    // detessdequant path's `synthetic_inputs.push_back(make_synthetic_
    // tensor_contract_local(...))` step.
    const auto* upstream_input =
        pre_mla_branch_upstream_input_tensor_local(*boundary, siblings[i], i);
    if (upstream_input == nullptr) {
      throw std::runtime_error("processcvu MPK pre-MLA branch missing upstream input identity");
    }
    const std::vector<std::int64_t> input_shape = !upstream_input->logical_shape.empty()
                                                      ? upstream_input->logical_shape
                                                      : upstream_input->mpk_shape;
    const std::string input_dtype = !upstream_input->logical_dtype.empty()
                                        ? upstream_input->logical_dtype
                                        : upstream_input->dtype;
    synthetic_inputs.push_back(make_synthetic_tensor_contract_local(
        input_shape, normalize_dtype_token_local(input_dtype), static_cast<int>(i),
        "input_tensor_" + std::to_string(i)));
    if (upstream_input->size_bytes > 0U) {
      synthetic_inputs.back().size_bytes = upstream_input->size_bytes;
      packed_input_sizes.push_back(static_cast<std::uint64_t>(upstream_input->size_bytes));
    } else {
      packed_input_sizes.push_back(static_cast<std::uint64_t>(synthetic_inputs.back().size_bytes));
    }
    const auto& input_tensor_contract = synthetic_inputs.back();

    // The output identity for this branch is the sibling's MPK output tensor
    // (e.g. the tess stage's output_tensors.front()). Pointer is stable for
    // the duration of this function — `siblings` is captured by value above.
    const auto* anchor = siblings[i].tess
                             ? siblings[i].tess
                             : (siblings[i].quant ? siblings[i].quant : siblings[i].cast);
    if (anchor == nullptr || anchor->output_tensors.empty()) {
      throw std::runtime_error("processcvu MPK pre-MLA branch missing published output tensor");
    }
    const auto& output_tensor_contract = anchor->output_tensors.front();
    std::uint64_t output_size_bytes =
        output_tensor_contract.size_bytes > 0U
            ? static_cast<std::uint64_t>(output_tensor_contract.size_bytes)
            : preferred_mpk_tensor_size_bytes_local(output_tensor_contract, runtime.output_dtype);
    if (family == "cast" && i < branch_runtimes.size() &&
        !branch_runtimes[i].output_tensors.empty() &&
        branch_runtimes[i].output_tensors.front().storage.nbytes > 0U) {
      output_size_bytes = branch_runtimes[i].output_tensors.front().storage.nbytes;
    }

    ProcessCvuPackedRouteEntry entry;
    entry.logical_index = static_cast<int>(i);
    entry.output_slot = static_cast<int>(i);
    // physical_index labels the i-th logical slot. For packed-output paths
    // build_processcvu_packed_route_facts maps all entries to physical 0 and
    // preserves the per-branch byte_offset. For native multi-IFM paths we
    // rewrite facts below so each logical output owns physical i at offset 0.
    entry.physical_index = static_cast<int>(i);
    entry.input_tensor = &input_tensor_contract;
    entry.input_dtype = normalize_dtype_token_local(input_dtype);
    entry.input_layout.clear();
    entry.input_byte_offset = static_cast<std::int64_t>(packed_input_offset);
    packed_input_offset += packed_input_sizes.back();
    entry.output_tensor = &output_tensor_contract;
    entry.output_physical_name =
        native_distinct_mla_boundary ? published_output_names[i] : "output_tensor";
    entry.output_logical_name = published_output_names[i];
    entry.output_dtype = runtime.output_dtype;
    entry.output_layout.clear();
    entry.output_byte_offset =
        native_distinct_mla_boundary ? 0 : static_cast<std::int64_t>(packed_output_offset);
    if (!native_distinct_mla_boundary) {
      packed_output_offset += output_size_bytes;
    }
    // All branches share a single upstream physical buffer — exactly the
    // post-side packed-route arrangement. The framework bundles the user's
    // TensorList into one `TensorBuffer` (with `sima_segments` describing
    // each per-branch region) before pushing through a single appsrc, so the
    // multi-IO pre element receives one GstBuffer with N GstMemory regions
    // and disambiguates branches via the per-binding byte_offset accumulated
    // below. This is the byte-for-byte mirror of how detessdequant on the
    // post side treats its single MLA-output upstream buffer.
    entry.src_output_slot = static_cast<int>(i);
    entry.src_physical_output_index = 0;
    entries.push_back(std::move(entry));
  }
  runtime.primary_output_name = runtime.published_output_names.front();
  if (!runtime.q_scale_list.empty()) {
    runtime.has_q_scale = true;
    runtime.q_scale = runtime.q_scale_list.front();
  }
  if (!runtime.q_zp_list.empty()) {
    runtime.has_q_zp = true;
    runtime.q_zp = static_cast<int>(runtime.q_zp_list.front());
  }

  ProcessCvuCanonicalCompileInputs out;
  out.payload = build_processcvu_payload_from_runtime_config_unchecked_internal(runtime);
  out.facts = build_processcvu_packed_route_facts("input_tensor", "output_tensor", entries,
                                                  runtime.primary_output_name,
                                                  runtime.published_output_names);
  if (native_distinct_mla_boundary) {
    preserve_distinct_physical_output_views(&out, published_output_names);
  }
  // The packed-route facts builder produces sink_pad_index=0 / physical_index=0
  // for every input binding by default — i.e. all N logical inputs share a
  // single GStreamer sink pad and a single upstream physical buffer. That is
  // the right layout here too: the pre-side fan-in framework path bundles
  // the user's TensorList into ONE TensorBuffer (with sima_segments
  // describing each per-branch region) and pushes ONE GstBuffer carrying N
  // GstMemory regions through ONE appsrc, mirroring the way the post side's
  // single MLA-output buffer carries N tensor regions into detessdequant.
  // Per-branch identity is encoded by the byte_offset on each binding.
  //
  // Use the per-branch UPSTREAM input contracts (synthesized from each
  // sibling's incoming tensor — FP32 ingress for quanttess) as the published
  // inputs. The post-side analogue uses MLA's published outputs because those
  // ARE detessdequant's inputs; the symmetric pre-side analogue uses the
  // upstream ingress tensors because those are quanttess/casttess inputs —
  // NOT boundary->input_tensors[i], which are MLA's inputs (i.e. quanttess's
  // OUTPUTS — INT8 tiled). Picking the wrong side propagates the output dtype
  // back into the input binding's storage.nbytes and the kernel's
  // strided-required check fails (see processcvu_ev_abi_a65_runner.cpp's
  // strided_required_bytes vs storage.nbytes).
  apply_published_routed_input_bindings(&out, synthetic_inputs, &packed_input_sizes,
                                        runtime.graph_family);
  enforce_packed_parent_input_views(&out, "input_tensor", entries, packed_input_sizes);
  force_direct_materialization_for_inputs(&out);
  return out;
}

plugin_contracts::QuantTessContractSubset
build_quanttess_contract_subset_for_exact_stage_local(const MpkContract& contract,
                                                      const MpkPluginIoContract* exact_stage) {
  if (!exact_stage) {
    return plugin_contracts::extract_quanttess_contract_subset_from_mpk(contract);
  }

  const std::string exact_kind = canonical_mpk_stage_kind_name(*exact_stage);
  if (exact_kind == "quanttess") {
    return plugin_contracts::extract_quanttess_contract_subset_from_stage(*exact_stage);
  }

  const MpkPluginIoContract* quant_stage = nullptr;
  const MpkPluginIoContract* tess_stage = nullptr;
  if (exact_kind == "quant") {
    quant_stage = exact_stage;
    tess_stage = resolve_connected_pre_stage_for_kind_local(contract, *exact_stage, "tess", false);
  } else if (exact_kind == "tess") {
    tess_stage = exact_stage;
    quant_stage = resolve_connected_pre_stage_for_kind_local(contract, *exact_stage, "quant", true);
  } else {
    return plugin_contracts::extract_quanttess_contract_subset_from_mpk(contract);
  }

  if (!quant_stage || !tess_stage) {
    throw std::runtime_error(
        std::string("processcvu MPK quanttess stage could not resolve paired quant/tess stages "
                    "from exact stage '") +
        (!exact_stage->name.empty() ? exact_stage->name : exact_stage->plugin_id) + "'");
  }
  if (quant_stage->input_tensors.empty()) {
    throw std::runtime_error(
        "processcvu MPK quanttess exact quant stage is missing input tensor metadata");
  }

  plugin_contracts::QuantTessContractSubset subset;
  subset.quant_params = quant_stage->quant.value_or(MpkQuantContract{});
  subset.input_shape =
      preferred_stage_input_tensor_shape_local(*quant_stage, quant_stage->input_tensors.front());
  if (!tess_stage->output_tensors.empty()) {
    subset.output_shape = !tess_stage->output_tensors.front().logical_shape.empty()
                              ? tess_stage->output_tensors.front().logical_shape
                              : subset.input_shape;
    if (tess_stage->output_tensors.front().size_bytes > 0U) {
      subset.output_size_bytes =
          static_cast<std::uint64_t>(tess_stage->output_tensors.front().size_bytes);
    }
  } else {
    subset.output_shape =
        !subset.input_shape.empty() ? subset.input_shape : tess_stage->out_shape_raw;
  }
  subset.input_dtype = normalize_dtype_token_local(!quant_stage->canonical_input_dtype.empty()
                                                       ? quant_stage->canonical_input_dtype
                                                       : quant_stage->input_tensors.front().dtype);
  subset.output_dtype = normalize_dtype_token_local(
      !quant_stage->canonical_output_dtype.empty()
          ? quant_stage->canonical_output_dtype
          : (quant_stage->output_tensors.empty() ? std::string{}
                                                 : quant_stage->output_tensors.front().dtype));
  subset.round_off =
      plugin_contracts::extract_quantize_contract_subset_from_stage(*quant_stage).round_off;
  subset.slice_shape = tess_stage->slice_shape;
  subset.frame_type = normalize_dtype_token_local(tess_stage->frame_type);
  subset.align_c16 = tess_stage->has_align_c16 && tess_stage->align_c16;
  subset.cblock = tess_stage->has_cblock && tess_stage->cblock;

  if (subset.quant_params.scales.empty() || subset.quant_params.zero_points.empty() ||
      subset.input_shape.empty() || subset.output_shape.empty() || subset.input_dtype.empty() ||
      subset.output_dtype.empty() || subset.round_off < 0 || subset.slice_shape.empty() ||
      subset.frame_type.empty()) {
    throw std::runtime_error(
        std::string("processcvu MPK quanttess exact-stage subset is missing required quant/tess "
                    "facts for '") +
        (!exact_stage->name.empty() ? exact_stage->name : exact_stage->plugin_id) + "'");
  }

  // Phase 3a (Option A++): mirror the batch_size hoisting done by
  // plugin_contracts::extract_quanttess_contract_subset_from_{mpk,stage}.
  // input_shape / output_shape stay BATCHED on the subset (caps negotiation
  // downstream depends on the batch dim being explicit); the runtime-config
  // builder strips on the fly for kernel descriptor synthesis.
  {
    const int per_frame_rank = plugin_contracts::derive_per_frame_rank_public(
        subset.slice_shape, /*peer_per_frame_shape=*/{});
    if (per_frame_rank > 0) {
      subset.batch_size = plugin_contracts::inferred_batch_size_from_shape_public(
          subset.input_shape, per_frame_rank);
    }
  }
  return subset;
}

plugin_contracts::CastContractSubset
build_preadapter_cast_subset_for_tess_stage_local(const MpkContract& contract,
                                                  const MpkPluginIoContract& tess_stage) {
  const auto* cast_stage =
      resolve_connected_pre_stage_for_kind_local(contract, tess_stage, "cast", true);
  if (!cast_stage) {
    throw std::runtime_error("processcvu MPK tess route requires an upstream cast stage for '" +
                             tess_stage.name + "'");
  }
  if (cast_stage->input_tensors.empty() || cast_stage->output_tensors.empty()) {
    throw std::runtime_error("processcvu MPK tess route missing cast tensor metadata for '" +
                             cast_stage->name + "'");
  }

  const auto preferred_input_shape = [&]() -> std::vector<std::int64_t> {
    const auto& tensor = cast_stage->input_tensors.front();
    return preferred_stage_input_tensor_shape_local(*cast_stage, tensor);
  }();

  plugin_contracts::CastContractSubset subset;
  subset.input_shape = preferred_input_shape;
  subset.input_dtype = normalize_dtype_token_local(!cast_stage->canonical_input_dtype.empty()
                                                       ? cast_stage->canonical_input_dtype
                                                       : cast_stage->input_tensors.front().dtype);
  subset.output_dtype = normalize_dtype_token_local(!cast_stage->canonical_output_dtype.empty()
                                                        ? cast_stage->canonical_output_dtype
                                                        : cast_stage->output_tensors.front().dtype);
  if (subset.input_shape.empty() || subset.input_dtype.empty() || subset.output_dtype.empty()) {
    throw std::runtime_error(
        "processcvu MPK tess route requires canonical cast tensor facts for '" + cast_stage->name +
        "'");
  }
  return subset;
}

std::vector<const MpkPluginIoContract*>
collect_post_stages_for_kind_names_local(const MpkContract& contract,
                                         std::initializer_list<const char*> preferred) {
  const auto ordered = ordered_plugin_indices_local(contract);
  const auto mla_rank = mla_rank_in_order_local(contract, ordered);
  std::vector<const MpkPluginIoContract*> matches;
  for (std::size_t rank = 0; rank < ordered.size(); ++rank) {
    if (mla_rank.has_value() && !(rank > *mla_rank)) {
      continue;
    }
    const std::size_t idx = ordered[rank];
    if (idx >= contract.plugins.size()) {
      continue;
    }
    const auto& stage = contract.plugins[idx];
    const std::string kind = canonical_mpk_stage_kind_name(stage);
    if (std::find_if(preferred.begin(), preferred.end(), [&](const char* wanted) {
          return wanted && kind == wanted;
        }) != preferred.end()) {
      matches.push_back(&stage);
    }
  }
  return matches;
}

const MpkPluginIoContract*
find_terminal_stage_after_outputs_local(const MpkContract& contract,
                                        const std::vector<const MpkPluginIoContract*>& producers) {
  if (producers.empty()) {
    return nullptr;
  }
  const auto ordered = plugins_in_execution_order(contract);
  if (ordered.empty()) {
    return nullptr;
  }

  std::optional<std::size_t> anchor_pos;
  for (std::size_t pos = 0; pos < ordered.size(); ++pos) {
    const std::size_t idx = ordered[pos];
    if (idx >= contract.plugins.size()) {
      continue;
    }
    for (const auto* producer : producers) {
      if (producer == &contract.plugins[idx]) {
        anchor_pos = std::max(anchor_pos.value_or(0U), pos);
      }
    }
  }
  if (!anchor_pos.has_value() || *anchor_pos + 1U >= ordered.size()) {
    return nullptr;
  }

  const auto& terminal = contract.plugins[ordered.back()];
  return terminal.output_tensors.empty() ? nullptr : &terminal;
}

bool stage_is_kernel_pass_through_local(const MpkPluginIoContract* stage) {
  if (!stage) {
    return false;
  }
  auto lower = [](std::string value) {
    for (char& ch : value) {
      ch = static_cast<char>(std::tolower(static_cast<unsigned char>(ch)));
    }
    return value;
  };
  const std::string kernel = lower(stage->kernel);
  return kernel == "pass_through" || kernel == "passthrough";
}

const MpkTensorContract*
terminal_output_tensor_for_index_local(const MpkPluginIoContract* terminal_stage, std::size_t index,
                                       std::size_t expected_count) {
  if (!terminal_stage || terminal_stage->output_tensors.empty()) {
    return nullptr;
  }
  if (terminal_stage->output_tensors.size() != expected_count || index >= expected_count) {
    return nullptr;
  }
  return &terminal_stage->output_tensors[index];
}

std::string published_output_name_from_terminal_or_producer_local(
    const MpkPluginIoContract* terminal_stage, const MpkTensorContract* terminal_output_tensor,
    const MpkTensorContract& producer_output_tensor, std::size_t index) {
  // Terminal pass-through stages are transport-only wrappers. Preserve the
  // producer tensor name so downstream standalone consumers retain semantic
  // hints like bbox/class_prob instead of synthetic pass_through_out_N names.
  if (!stage_is_kernel_pass_through_local(terminal_stage) && terminal_output_tensor &&
      !terminal_output_tensor->name.empty()) {
    return terminal_output_tensor->name;
  }
  if (!producer_output_tensor.name.empty()) {
    return producer_output_tensor.name;
  }
  if (terminal_output_tensor && !terminal_output_tensor->name.empty()) {
    return terminal_output_tensor->name;
  }
  return "output_" + std::to_string(index);
}

bool output_name_looks_generic_local(const std::string& raw_name) {
  if (raw_name.empty()) {
    return true;
  }
  const std::string lower = lower_copy(raw_name);
  return lower.rfind("pass_through_out_", 0) == 0 || lower.rfind("output_tensor_", 0) == 0 ||
         lower == "output_tensor";
}

std::string preferred_stage_name_for_output_local(const MpkPluginIoContract& stage,
                                                  std::size_t index) {
  if (!stage.name.empty()) {
    return stage.name;
  }
  if (!stage.plugin_id.empty()) {
    return stage.plugin_id;
  }
  return "output_" + std::to_string(index);
}

ProcessCvuStagePayload
build_preproc_payload_from_options_local(const ::simaai::neat::PreprocOptions& opt) {
  require_supported_preproc_single_output_handoff(opt.single_output_handoff);
  ProcessCvuStagePayload payload;
  payload.canonical_contract = true;
  payload.graph_family = "preproc";
  payload.graph_family_enum = ProcessCvuGraphFamily::Preproc;
  payload.graph_name = opt.graph_name.empty() ? std::string("preproc") : opt.graph_name;
  payload.graph_id = 200;
  payload.default_input_name =
      opt.graph_input_name.empty() ? std::string("input_image") : opt.graph_input_name;
  payload.default_output_names = default_preproc_runtime_output_names();
  payload.primary_output_name =
      resolve_preproc_primary_output_name(payload.default_output_names, opt.tessellate);
  payload.preproc_single_output_handoff = opt.single_output_handoff;
  if (!opt.input_shape.empty()) {
    payload.input_shapes = {opt.input_shape};
  }
  if (!opt.output_shape.empty()) {
    payload.output_shapes = {opt.output_shape};
  }
  if (!opt.slice_shape.empty()) {
    payload.slice_shapes = {opt.slice_shape};
  }
  payload.scaled_width = opt.scaled_width;
  payload.scaled_height = opt.scaled_height;
  payload.input_stride = opt.input_stride;
  payload.output_stride = opt.output_stride;
  payload.input_offset = opt.input_offset;
  payload.batch_size = opt.batch_size;
  payload.byte_align = 1;
  payload.aspect_ratio = opt.aspect_ratio ? 1 : 0;
  payload.normalize = opt.normalize ? 1 : 0;
  payload.tessellate = opt.tessellate ? 1 : 0;
  payload.channel_mean.assign(opt.channel_mean.begin(), opt.channel_mean.end());
  payload.channel_stddev.assign(opt.channel_stddev.begin(), opt.channel_stddev.end());
  payload.input_img_type = opt.input_img_type;
  payload.output_img_type = opt.output_img_type;
  payload.input_dtype = "UINT8";
  payload.output_dtype = opt.output_dtype;
  payload.out_dtype = opt.output_dtype;
  payload.scaling_type = opt.scaling_type;
  payload.padding_type = opt.padding_type;
  payload.has_q_scale = opt.q_scale.has_value();
  payload.q_scale = opt.q_scale.value_or(0.0);
  if (opt.q_scale.has_value()) {
    payload.q_scale_list = {*opt.q_scale};
  }
  payload.has_q_zp = opt.q_zp.has_value();
  payload.q_zp = static_cast<int>(opt.q_zp.value_or(0));
  if (opt.q_zp.has_value()) {
    payload.q_zp_list = {static_cast<int>(*opt.q_zp)};
  }
  synthesize_runtime_output_arrays_from_payload(&payload);
  canonicalize_preproc_single_handoff_payload(&payload);
  return payload;
}

static ProcessCvuCanonicalCompileInputs build_processcvu_mpk_preproc_compile_inputs_local(
    const MpkContract& contract, const std::string& input_format, int input_depth,
    int max_input_width, int max_input_height, bool normalize, const std::vector<float>& mean,
    const std::vector<float>& stddev, bool single_output_handoff) {
  const auto* mla_stage = get_mla_stage_io_contract(contract);
  if (!mla_stage || mla_stage->input_tensors.empty()) {
    throw std::runtime_error("processcvu MPK preproc route missing MLA ingress contract");
  }
  const auto* tess_stage = find_pre_stage_for_kind_names_local(contract, {"quanttess", "tess"});
  const auto* quant_stage = find_pre_stage_for_kind_names_local(contract, {"quanttess", "quant"});
  const auto* cast_stage = find_pre_stage_for_kind_names_local(contract, {"cast"});
  auto dims_from_tensor = [](const MpkTensorContract* tensor) -> MpkTensorDims {
    if (!tensor) {
      return {};
    }
    auto dims = dims_from_mpk_shape(tensor->logical_shape);
    if (dims.width > 0 && dims.height > 0 && dims.depth > 0) {
      return dims;
    }
    return dims_from_mpk_shape(tensor->mpk_shape);
  };
  const auto* pre_input_tensor =
      quant_stage && !quant_stage->input_tensors.empty() ? &quant_stage->input_tensors.front()
      : tess_stage && !tess_stage->input_tensors.empty() ? &tess_stage->input_tensors.front()
      : cast_stage && !cast_stage->input_tensors.empty() ? &cast_stage->input_tensors.front()
                                                         : nullptr;
  auto ingress_dims = dims_from_tensor(&mla_stage->input_tensors.front());
  if (ingress_dims.width <= 0 || ingress_dims.height <= 0 || ingress_dims.depth <= 0) {
    ingress_dims = dims_from_tensor(pre_input_tensor);
  }
  if (ingress_dims.width <= 0 || ingress_dims.height <= 0 || ingress_dims.depth <= 0) {
    throw std::runtime_error("processcvu MPK preproc route missing MLA ingress geometry");
  }

  ::simaai::neat::PreprocOptions opt;
  opt.input_img_type = input_format.empty() ? std::string("RGB") : input_format;
  {
    std::vector<int> input_shape = {
        max_input_height > 0 ? max_input_height : std::max(ingress_dims.height, 1),
        max_input_width > 0 ? max_input_width : std::max(ingress_dims.width, 1)};
    const int resolved_input_depth =
        input_depth > 0 ? input_depth
                        : pipeline_internal::default_depth_for_image_format(opt.input_img_type, 3);
    if (resolved_input_depth > 0) {
      input_shape.push_back(resolved_input_depth);
    }
    opt.set_input_shape(std::move(input_shape));
  }
  {
    std::vector<int> output_shape = {std::max(ingress_dims.height, 1),
                                     std::max(ingress_dims.width, 1)};
    if (ingress_dims.depth > 0) {
      output_shape.push_back(ingress_dims.depth);
    }
    opt.set_output_shape(std::move(output_shape));
  }
  opt.scaled_width = opt.output_width();
  opt.scaled_height = opt.output_height();
  opt.output_img_type = "RGB";
  opt.normalize = normalize;
  opt.channel_mean = mean.empty() ? std::vector<float>{0.0f, 0.0f, 0.0f} : mean;
  opt.channel_stddev = stddev.empty() ? std::vector<float>{1.0f, 1.0f, 1.0f} : stddev;
  opt.single_output_handoff = single_output_handoff;
  opt.tessellate = tess_stage != nullptr;
  if (tess_stage) {
    const auto tess_sd = dims_from_slice_shape(tess_stage->slice_shape);
    std::vector<int> slice_shape = {tess_sd.h, tess_sd.w};
    const int tile_channels = positive_tile_channels_local(*tess_stage);
    if (tile_channels > 0) {
      slice_shape.push_back(tile_channels);
    }
    opt.set_slice_shape(std::move(slice_shape));
    if (!opt.has_slice_shape()) {
      throw std::runtime_error("processcvu MPK preproc route missing tess tile geometry");
    }
  }

  const std::string mla_input_dtype = preferred_tensor_dtype_local(
      mla_stage->input_tensors.front(), mla_stage->canonical_input_dtype);
  const std::string cast_output_dtype =
      (cast_stage && !cast_stage->output_tensors.empty())
          ? preferred_tensor_dtype_local(cast_stage->output_tensors.front(),
                                         cast_stage->canonical_output_dtype)
          : std::string{};
  if (quant_stage) {
    opt.output_dtype = "INT8";
    const auto quant = resolve_processcvu_mpk_quant_contract_local(contract, *quant_stage);
    if (!quant.has_value() || quant->scales.empty() || quant->zero_points.empty()) {
      throw std::runtime_error("processcvu MPK preproc route missing quant facts");
    }
    opt.q_scale = quant->scales.front();
    opt.q_zp = quant->zero_points.front();
  } else if (cast_output_dtype == "BF16") {
    opt.output_dtype = "EVXX_BFLOAT16";
  } else if (cast_output_dtype == "FP32") {
    opt.output_dtype = "EVXX_FLOAT32";
  } else if (mla_input_dtype == "BF16") {
    opt.output_dtype = "EVXX_BFLOAT16";
  } else if (mla_input_dtype == "FP32") {
    opt.output_dtype = "EVXX_FLOAT32";
  } else {
    throw std::runtime_error("processcvu MPK preproc route requires an explicit output dtype");
  }

  const auto payload = build_preproc_payload_from_options_local(opt);
  ProcessCvuCanonicalCompileInputs out;
  out.payload = payload;
  out.facts = build_processcvu_facts_from_payload_local(payload);
  return out;
}

static ProcessCvuCanonicalCompileInputs build_processcvu_mpk_preadapter_compile_inputs_local(
    const MpkContract& contract, const std::string& graph_family,
    const std::optional<std::string>& exact_stage_name_or_id,
    const std::optional<std::string>& canonical_handoff_segment_name) {
  const std::string family = lower_copy(canonical_processcvu_graph_family(graph_family));
  if (family != "quantize" && family != "tessellate" && family != "quanttess" &&
      family != "casttess" && family != "cast") {
    throw std::runtime_error("processcvu MPK pre-adapter route requires quantize, tessellate, "
                             "quanttess, casttess, or cast graph_family");
  }

  // Multi-IO pre-MLA fan-in: when the renderer is asked for a pre-MLA family
  // without a specific exact stage AND the MPK contains N>1 sibling branches
  // feeding the MLA boundary (directly for native multi-IFM models, or via an
  // IFM pack stage for packer-style models like rpn_head_640_640_concat_4d),
  // build a single multi-input runtime config by merging N per-branch
  // single-IO runtimes and let auto-routing emit a multi-IO payload
  // (num_in_tensor=N). Applies to quanttess, casttess, tessellate, quantize.
  // Multi-IO pre-MLA fan-in: when the renderer is asked for a pre-MLA family
  // without a specific exact stage AND the MPK contains N>1 sibling branches
  // feeding the MLA boundary, dispatch to the dedicated packed-multi-IO
  // renderer. That renderer is a structural mirror of the post-side
  // build_processcvu_mpk_dequant_compile_inputs_local /
  // build_processcvu_mpk_detessdequant_compile_inputs_local and uses the same
  // post-side helpers (build_processcvu_packed_route_facts,
  // apply_published_routed_input_bindings, force_direct_materialization_for_inputs,
  // make_synthetic_tensor_contract_local) so the resulting compiled contract
  // has physical_inputs=1 / physical_outputs=1 with N logical regions
  // addressed by per-entry byte_offsets — the form the GStreamer element's
  // ConfigManager init expects for multi-IO over packed transport.
  const bool exact_stage_requested =
      exact_stage_name_or_id.has_value() && !exact_stage_name_or_id->empty();
  if (!exact_stage_requested) {
    auto siblings = resolve_pre_mla_sibling_stages_from_mpk_local(contract, family);
    if (siblings.size() > 1U) {
      return build_processcvu_mpk_pre_mla_multi_io_compile_inputs_local(contract, family,
                                                                        std::move(siblings));
    }
  }

  const auto* exact_stage = (exact_stage_name_or_id.has_value() && !exact_stage_name_or_id->empty())
                                ? get_stage_io_contract(contract, *exact_stage_name_or_id)
                                : nullptr;
  const std::string resolved_exact_stage_name_or_id =
      exact_stage ? (!exact_stage->name.empty() ? exact_stage->name : exact_stage->plugin_id)
                  : (exact_stage_name_or_id.has_value() ? *exact_stage_name_or_id : std::string{});
  const std::string exact_stage_kind =
      exact_stage ? canonical_mpk_stage_kind_name(*exact_stage) : std::string{};
  const auto* stage =
      exact_stage ? exact_stage
      : family == "quantize"
          ? find_pre_stage_for_kind_names_local(contract, {"quant", "quanttess", "preproc"})
      : family == "tessellate" ? find_pre_stage_for_kind_names_local(
                                     contract, {"tess", "quanttess", "casttess", "preproc"})
      : family == "casttess"
          ? find_pre_stage_for_kind_names_local(contract, {"casttess", "tess", "preproc"})
      : family == "cast" ? find_pre_stage_for_kind_names_local(contract, {"cast", "preproc"})
                         : find_pre_stage_for_kind_names_local(contract, {"quanttess", "preproc"});
  const auto* quant_stage =
      family == "quanttess"
          ? (exact_stage
                 ? (exact_stage_kind == "quant"
                        ? exact_stage
                        : (exact_stage_kind == "tess" ? resolve_connected_pre_stage_for_kind_local(
                                                            contract, *exact_stage, "quant", true)
                                                      : stage))
                 : find_pre_stage_for_kind_names_local(contract, {"quant", "preproc"}))
          : nullptr;
  const auto* tess_stage =
      family == "quanttess"
          ? (exact_stage
                 ? (exact_stage_kind == "tess"
                        ? exact_stage
                        : (exact_stage_kind == "quant" ? resolve_connected_pre_stage_for_kind_local(
                                                             contract, *exact_stage, "tess", false)
                                                       : stage))
                 : find_pre_stage_for_kind_names_local(contract, {"tess", "preproc"}))
      : family == "casttess"
          ? (exact_stage
                 ? (exact_stage_kind == "tess"
                        ? exact_stage
                        : (exact_stage_kind == "cast" ? resolve_connected_pre_stage_for_kind_local(
                                                            contract, *exact_stage, "tess", false)
                                                      : stage))
                 : find_pre_stage_for_kind_names_local(contract, {"tess", "preproc"}))
          : nullptr;
  const auto* cast_stage =
      family == "casttess"
          ? (exact_stage
                 ? (exact_stage_kind == "cast"
                        ? exact_stage
                        : (exact_stage_kind == "tess" ? resolve_connected_pre_stage_for_kind_local(
                                                            contract, *exact_stage, "cast", true)
                                                      : stage))
                 : find_pre_stage_for_kind_names_local(contract, {"cast", "preproc"}))
          : nullptr;
  const auto* geometry_stage =
      stage ? stage : (tess_stage ? tess_stage : (cast_stage ? cast_stage : quant_stage));
  const auto* input_stage = family == "quanttess"  ? (quant_stage ? quant_stage : geometry_stage)
                            : family == "casttess" ? (cast_stage ? cast_stage : geometry_stage)
                                                   : geometry_stage;
  const auto* output_stage = family == "quanttess"  ? (tess_stage ? tess_stage : geometry_stage)
                             : family == "casttess" ? (tess_stage ? tess_stage : geometry_stage)
                                                    : geometry_stage;
  const auto* tile_stage = family == "quanttess"  ? (tess_stage ? tess_stage : geometry_stage)
                           : family == "casttess" ? (tess_stage ? tess_stage : geometry_stage)
                                                  : geometry_stage;
  if (!geometry_stage) {
    throw std::runtime_error("processcvu MPK pre-adapter stage missing");
  }

  const std::string runtime_output_name = "output_tensor";
  const std::string physical_output_name =
      (output_stage && !output_stage->output_tensors.empty())
          ? processcvu_physical_output_name_from_mpk_tensor(output_stage->output_tensors.front(),
                                                            runtime_output_name)
          : runtime_output_name;
  const std::string published_output_name =
      (canonical_handoff_segment_name.has_value() && !canonical_handoff_segment_name->empty())
          ? *canonical_handoff_segment_name
          : runtime_output_name;
  const ProcessCvuSingleOutputIdentity output_identity =
      build_processcvu_single_output_identity_local(runtime_output_name, physical_output_name,
                                                    published_output_name);
  if (processcvu_tess_segment_debug_enabled()) {
    const auto* output_tensor = (output_stage && !output_stage->output_tensors.empty())
                                    ? &output_stage->output_tensors.front()
                                    : nullptr;
    std::fprintf(
        stderr,
        "[tess-segment-debug] where=preadapter_route family=%s stage=%s runtime_output=%s "
        "physical_output=%s published_output=%s canonical_handoff=%s output_tensor_name=%s "
        "output_tensor_segment=%s "
        "output_tensor_shape=%s output_tensor_bytes=%zu\n",
        family.c_str(), output_stage ? output_stage->name.c_str() : "<none>",
        runtime_output_name.c_str(), physical_output_name.c_str(), published_output_name.c_str(),
        canonical_handoff_segment_name.has_value() ? canonical_handoff_segment_name->c_str()
                                                   : "<none>",
        output_tensor ? output_tensor->name.c_str() : "<none>",
        output_tensor ? output_tensor->segment_name.c_str() : "<none>",
        output_tensor ? join_i64_debug_processcvu_local(output_tensor->mpk_shape).c_str() : "[]",
        output_tensor ? output_tensor->size_bytes : 0U);
  }
  if (family == "quantize") {
    auto runtime = plugin_contracts::build_quantize_runtime_config_from_subset(
        exact_stage ? plugin_contracts::extract_quantize_contract_subset_from_stage(*stage)
                    : plugin_contracts::extract_quantize_contract_subset_from_mpk(contract),
        output_identity.physical_output_name, published_output_name);
    apply_processcvu_single_output_identity_local(&runtime, output_identity);
    require_graph_processcvu_runtime_geometry_local(contract, family, exact_stage_name_or_id,
                                                    &runtime);
    ProcessCvuSingleOutputFactsSpec facts_spec;
    facts_spec.physical_input_name = "input_tensor";
    facts_spec.source_segment_name = "input_tensor";
    apply_processcvu_single_output_identity_local(&facts_spec, output_identity);
    if (!runtime.input_shapes.empty()) {
      facts_spec.input_shape.assign(runtime.input_shapes.front().begin(),
                                    runtime.input_shapes.front().end());
    }
    facts_spec.input_dtype = runtime.input_dtype;
    facts_spec.input_layout = runtime_input_layout_token_local(runtime);
    facts_spec.output_dtype = runtime.output_dtype;
    // Quantize may hand off through a named runtime segment, but its public
    // contract is still a dense tensor. Publishing it as a packed blob flattens
    // the descriptor and breaks the next session boundary.
    facts_spec.output_representation = ProcessCvuOutputRepresentation::DenseTensor;
    if (!runtime.output_shapes.empty()) {
      facts_spec.output_shape.assign(runtime.output_shapes.front().begin(),
                                     runtime.output_shapes.front().end());
    }
    facts_spec.output_layout = runtime_output_layout_token_local(runtime);

    ProcessCvuCanonicalCompileInputs out;
    out.payload = build_processcvu_payload_from_runtime_config_internal(runtime);
    out.payload.exact_stage_name_or_id = resolved_exact_stage_name_or_id;
    out.facts = build_processcvu_single_output_facts(facts_spec);
    return out;
  }
  if (family == "cast") {
    auto runtime = build_preadapter_cast_runtime_config_local(
        contract, input_stage, output_stage, output_identity, exact_stage_name_or_id);

    ProcessCvuSingleOutputFactsSpec facts_spec;
    facts_spec.physical_input_name = "input_tensor";
    facts_spec.source_segment_name = "input_tensor";
    apply_processcvu_single_output_identity_local(&facts_spec, output_identity);
    facts_spec.input_shape.assign(runtime.input_shapes.front().begin(),
                                  runtime.input_shapes.front().end());
    facts_spec.input_dtype = runtime.input_dtype;
    facts_spec.input_layout = runtime_input_layout_token_local(runtime);
    facts_spec.output_shape.assign(runtime.output_shapes.front().begin(),
                                   runtime.output_shapes.front().end());
    facts_spec.output_dtype = runtime.output_dtype;
    facts_spec.output_layout = runtime_output_layout_token_local(runtime);
    facts_spec.output_representation = ProcessCvuOutputRepresentation::DenseTensor;

    ProcessCvuCanonicalCompileInputs out;
    out.payload = build_processcvu_payload_from_runtime_config_internal(runtime);
    out.payload.exact_stage_name_or_id = resolved_exact_stage_name_or_id;
    out.facts = build_processcvu_single_output_facts(facts_spec);
    return out;
  }

  if (family == "tessellate" || family == "casttess") {
    const auto* tess_runtime_stage =
        family == "casttess" ? (tess_stage ? tess_stage : stage) : stage;
    const auto cast_subset = [&]() {
      if (family == "casttess") {
        if (cast_stage != nullptr) {
          return plugin_contracts::extract_cast_contract_subset_from_stage(*cast_stage);
        }
        if (exact_stage && exact_stage_kind == "cast") {
          return plugin_contracts::extract_cast_contract_subset_from_stage(*exact_stage);
        }
      }
      if (exact_stage) {
        const auto tess_subset =
            plugin_contracts::extract_tessellate_contract_subset_from_stage(*tess_runtime_stage);
        plugin_contracts::CastContractSubset subset;
        subset.input_shape = tess_subset.input_shape;
        subset.input_layout = tess_subset.input_layout;
        subset.input_dtype = normalize_dtype_token_local(
            !tess_runtime_stage->canonical_input_dtype.empty()
                ? tess_runtime_stage->canonical_input_dtype
                : (tess_runtime_stage->input_tensors.empty()
                       ? std::string{}
                       : tess_runtime_stage->input_tensors.front().dtype));
        subset.output_dtype = normalize_dtype_token_local(!tess_runtime_stage->frame_type.empty()
                                                              ? tess_runtime_stage->frame_type
                                                              : subset.input_dtype);
        return subset;
      }
      return build_preadapter_cast_subset_for_tess_stage_local(contract, *tess_runtime_stage);
    }();
    auto runtime = plugin_contracts::build_tessellate_runtime_config_from_subsets(
        cast_subset,
        (exact_stage || family == "casttess")
            ? plugin_contracts::extract_tessellate_contract_subset_from_stage(*tess_runtime_stage)
            : plugin_contracts::extract_tessellate_contract_subset_from_mpk(contract),
        output_identity.physical_output_name, output_identity.published_output_name);
    if (family == "casttess") {
      runtime.graph_family = "casttess";
      runtime.graph_name = "casttess";
      runtime.graph_id = 224;
    }
    apply_processcvu_single_output_identity_local(&runtime, output_identity);
    require_graph_processcvu_runtime_geometry_local(contract, family, exact_stage_name_or_id,
                                                    &runtime);
    const std::uint64_t packed_output_size_bytes =
        output_stage && !output_stage->output_tensors.empty()
            ? preferred_mpk_tensor_size_bytes_local(output_stage->output_tensors.front(),
                                                    runtime.output_dtype)
            : 0U;
    if (packed_output_size_bytes == 0U) {
      throw std::runtime_error("processcvu MPK tess stage requires a non-zero packed output size");
    }

    ProcessCvuSingleOutputFactsSpec facts_spec;
    facts_spec.physical_input_name = "input_tensor";
    facts_spec.source_segment_name = "input_tensor";
    apply_processcvu_single_output_identity_local(&facts_spec, output_identity);
    if (!runtime.input_shapes.empty()) {
      facts_spec.input_shape.assign(runtime.input_shapes.front().begin(),
                                    runtime.input_shapes.front().end());
    }
    facts_spec.input_dtype = runtime.input_dtype;
    facts_spec.input_layout = runtime_input_layout_token_local(runtime);
    MpkTensorContract semantic_packed_output_tensor;
    if (output_stage && !output_stage->output_tensors.empty()) {
      semantic_packed_output_tensor = output_stage->output_tensors.front();
      if (semantic_packed_output_tensor.logical_shape.empty()) {
        semantic_packed_output_tensor.logical_shape = cast_subset.input_shape;
      }
    }
    facts_spec.output_representation = ProcessCvuOutputRepresentation::PackedTensor;
    facts_spec.output_dtype = runtime.output_dtype;
    facts_spec.output_layout = runtime_output_layout_token_local(runtime);
    facts_spec.packed_output_tensor = (output_stage && !output_stage->output_tensors.empty())
                                          ? &semantic_packed_output_tensor
                                          : nullptr;

    ProcessCvuCanonicalCompileInputs out;
    out.payload = build_processcvu_payload_from_runtime_config_internal(runtime);
    out.payload.exact_stage_name_or_id = resolved_exact_stage_name_or_id;
    out.facts = build_processcvu_single_output_facts(facts_spec);
    return out;
  }
  CompiledProcessCvuRuntimeConfig runtime =
      family == "quanttess"
          ? plugin_contracts::build_quanttess_runtime_config_from_subset(
                build_quanttess_contract_subset_for_exact_stage_local(contract, exact_stage),
                published_output_name)
          : CompiledProcessCvuRuntimeConfig{};
  if (family != "quanttess") {
    runtime.default_input_name = "input_tensor";
    runtime.runtime_input_names = {"input_tensor"};
    runtime.physical_input_names = {"input_tensor"};
    apply_processcvu_single_output_identity_local(&runtime, output_identity);
    runtime.batch_size = 1;
    runtime.byte_align = 1;
    const auto preferred_input_shape = (input_stage && !input_stage->input_tensors.empty())
                                           ? preferred_stage_input_tensor_shape_local(
                                                 *input_stage, input_stage->input_tensors.front())
                                           : std::vector<std::int64_t>{};
    const auto preferred_output_shape =
        (output_stage && !output_stage->output_tensors.empty())
            ? preferred_mpk_tensor_shape_local(output_stage->output_tensors.front())
            : std::vector<std::int64_t>{};
    if (preferred_input_shape.empty() || preferred_output_shape.empty()) {
      throw std::runtime_error(
          "processcvu MPK pre-adapter stage requires semantic input/output tensor shapes");
    }
    runtime.input_shapes = {
        std::vector<int>(preferred_input_shape.begin(), preferred_input_shape.end())};
    runtime.output_shapes = {
        std::vector<int>(preferred_output_shape.begin(), preferred_output_shape.end())};
    runtime.input_dtype = preferred_tensor_dtype_local(
        (input_stage && !input_stage->input_tensors.empty()) ? input_stage->input_tensors.front()
                                                             : MpkTensorContract{},
        input_stage ? input_stage->canonical_input_dtype : std::string{});
    runtime.output_dtype = preferred_tensor_dtype_local(
        (output_stage && !output_stage->output_tensors.empty())
            ? output_stage->output_tensors.front()
            : MpkTensorContract{},
        output_stage ? output_stage->canonical_output_dtype : std::string{});
    runtime.input_dtype = require_string_mpk_fact_local(
        std::move(runtime.input_dtype), "processcvu MPK pre-adapter stage", "input dtype");
    runtime.output_dtype = require_string_mpk_fact_local(
        std::move(runtime.output_dtype), "processcvu MPK pre-adapter stage", "output dtype");
    sima_ev_tensor_desc input_desc{};
    sima_ev_tensor_desc output_desc{};
    if (!build_dense_desc_processcvu_local(runtime.input_shapes.front(), runtime.input_dtype, {},
                                           &input_desc) ||
        !build_dense_desc_processcvu_local(runtime.output_shapes.front(), runtime.output_dtype, {},
                                           &output_desc)) {
      throw std::runtime_error(
          "processcvu MPK pre-adapter stage could not synthesize explicit typed tensors");
    }
    runtime.input_tensors = {input_desc};
    runtime.output_tensors = {output_desc};
    runtime.runtime_output_logical_layout_list.clear();
  }
  require_graph_processcvu_runtime_geometry_local(contract, family, exact_stage_name_or_id,
                                                  &runtime);
  const bool packed_tess_output = (family == "tessellate" || family == "quanttess") &&
                                  output_stage && !output_stage->output_tensors.empty();
  {
    if (packed_tess_output) {
      const auto packed_output_shape =
          !output_stage->output_tensors.front().logical_shape.empty()
              ? output_stage->output_tensors.front().logical_shape
          : runtime.input_shapes.empty()
              ? preferred_mpk_tensor_shape_local(output_stage->output_tensors.front())
              : std::vector<std::int64_t>(runtime.input_shapes.front().begin(),
                                          runtime.input_shapes.front().end());
      std::vector<int> packed_output_shape_int(packed_output_shape.begin(),
                                               packed_output_shape.end());
      runtime.output_shapes = {packed_output_shape_int};
      runtime.runtime_output_transport_kind_list = {ProcessCvuOutputTransportKind::Packed};
      runtime.runtime_output_semantic_kind_list = {
          family == "quanttess" ? ProcessCvuOutputSemanticKind::QuantTessTensor
                                : ProcessCvuOutputSemanticKind::TessellatedImage};
      runtime.runtime_output_logical_shapes = {packed_output_shape_int};
      runtime.runtime_output_logical_layout_list.clear();
      if (!runtime.output_tensors.empty()) {
        std::vector<int> tile_shape_int =
            !runtime.slice_shapes.empty() ? runtime.slice_shapes.front() : std::vector<int>{};
        sima_ev_tensor_desc output_desc{};
        const bool c16_packed =
            tile_stage && ((tile_stage->has_align_c16 && tile_stage->align_c16) ||
                           (tile_stage->has_cblock && tile_stage->cblock));
        if (!tile_shape_int.empty() &&
            build_tensor_tiled_desc_processcvu_local(
                packed_output_shape_int, tile_shape_int, runtime.output_dtype,
                resolve_tile_align_bytes_processcvu_local(runtime.byte_align), c16_packed,
                &output_desc)) {
          const std::uint64_t packed_output_size_bytes = preferred_mpk_tensor_size_bytes_local(
              output_stage->output_tensors.front(), runtime.output_dtype);
          if (packed_output_size_bytes != 0U) {
            output_desc.storage.nbytes = packed_output_size_bytes;
          }
          runtime.output_tensors = {output_desc};
        }
      }
      runtime.primary_output_transport_kind = ProcessCvuOutputTransportKind::Packed;
      runtime.primary_output_semantic_kind = family == "quanttess"
                                                 ? ProcessCvuOutputSemanticKind::QuantTessTensor
                                                 : ProcessCvuOutputSemanticKind::TessellatedImage;
    } else {
      const auto preferred_output_shape =
          (output_stage && !output_stage->output_tensors.empty())
              ? preferred_mpk_tensor_shape_local(output_stage->output_tensors.front())
              : std::vector<std::int64_t>{};
      if (!preferred_output_shape.empty()) {
        runtime.output_shapes = {
            std::vector<int>(preferred_output_shape.begin(), preferred_output_shape.end())};
      }
    }
  }
  runtime.runtime_output_logical_index_list = {0};
  runtime.runtime_output_output_slot_list = {0};
  runtime.runtime_output_physical_index_list = {0};

  std::optional<ProcessCvuCanonicalFacts> canonical_facts;
  if (output_stage && !output_stage->output_tensors.empty()) {
    ProcessCvuSingleOutputFactsSpec facts_spec;
    facts_spec.physical_input_name = "input_tensor";
    facts_spec.source_segment_name = "input_tensor";
    apply_processcvu_single_output_identity_local(&facts_spec, output_identity);
    if (!runtime.input_shapes.empty()) {
      facts_spec.input_shape.assign(runtime.input_shapes.front().begin(),
                                    runtime.input_shapes.front().end());
    }
    facts_spec.input_dtype = runtime.input_dtype;
    facts_spec.input_layout = runtime_input_layout_token_local(runtime);
    facts_spec.output_dtype = runtime.output_dtype;
    facts_spec.output_layout = runtime_output_layout_token_local(runtime);
    if (!runtime.output_shapes.empty()) {
      facts_spec.output_shape.assign(runtime.output_shapes.front().begin(),
                                     runtime.output_shapes.front().end());
    }
    MpkTensorContract semantic_output_tensor;
    if (output_stage && !output_stage->output_tensors.empty()) {
      semantic_output_tensor = output_stage->output_tensors.front();
      if (semantic_output_tensor.logical_shape.empty() && !runtime.output_shapes.empty()) {
        semantic_output_tensor.logical_shape.assign(runtime.output_shapes.front().begin(),
                                                    runtime.output_shapes.front().end());
      }
      if (semantic_output_tensor.logical_dtype.empty()) {
        semantic_output_tensor.logical_dtype = runtime.output_dtype;
      }
    }
    if (packed_tess_output) {
      facts_spec.output_representation = ProcessCvuOutputRepresentation::PackedTensor;
      facts_spec.packed_output_tensor = (output_stage && !output_stage->output_tensors.empty())
                                            ? &semantic_output_tensor
                                            : nullptr;
    } else {
      facts_spec.output_representation = ProcessCvuOutputRepresentation::DenseTensor;
    }
    canonical_facts = build_processcvu_single_output_facts(facts_spec);
  }

  if (family == "quanttess") {
    runtime.runtime_output_dtype_list = {"INT8"};
  } else {
    {
      const auto rt_slice = runtime_slice_dims_at(runtime, 0);
      require_positive_mpk_fact_local(rt_slice.width, "processcvu MPK tess stage", "tile width");
      require_positive_mpk_fact_local(rt_slice.height, "processcvu MPK tess stage", "tile height");
      require_positive_mpk_fact_local(rt_slice.depth, "processcvu MPK tess stage", "tile depth");
      require_positive_mpk_fact_local(rt_slice.channels, "processcvu MPK tess stage",
                                      "tile channels");
      runtime.slice_shapes = {
          {rt_slice.height, rt_slice.width,
           effective_dense_depth_for_layout(runtime_output_layout_token_local(runtime),
                                            rt_slice.depth, rt_slice.channels)}};
    }
    runtime.tessellate = 1;
    runtime.runtime_output_dtype_list = {runtime.output_dtype};
    if (family == "tessellate") {
      runtime.graph_family = "tessellate";
      runtime.graph_name = "tessellate";
      runtime.graph_id = 2;
      runtime.out_dtype = runtime.output_dtype;
    } else {
      runtime.graph_family = "quanttess";
      runtime.graph_name = "quanttess";
      runtime.graph_id = 202;
      const auto quant = resolve_processcvu_mpk_quant_contract_local(
          contract, *(quant_stage ? quant_stage : geometry_stage));
      if (!quant.has_value() || quant->scales.empty() || quant->zero_points.empty()) {
        throw std::runtime_error("processcvu MPK quanttess stage requires explicit quant facts");
      }
      const std::string mpk_output_dtype = runtime.output_dtype;
      if (upper_copy_local(mpk_output_dtype).find("INT8") == std::string::npos) {
        throw std::runtime_error(
            "processcvu MPK quanttess stage requires an explicit INT8 output dtype");
      }
      runtime.output_dtype = "INT8";
      runtime.out_dtype = "INT8";
      runtime.runtime_output_dtype_list = {"INT8"};
      runtime.has_q_scale = true;
      runtime.q_scale = quant->scales.front();
      runtime.q_scale_list = {runtime.q_scale};
      runtime.has_q_zp = true;
      runtime.q_zp = quant->zero_points.front();
      runtime.q_zp_list = {static_cast<int>(runtime.q_zp)};
    }
  }

  ProcessCvuCanonicalCompileInputs out;
  out.payload = build_processcvu_payload_from_runtime_config_internal(runtime);
  out.payload.exact_stage_name_or_id = resolved_exact_stage_name_or_id;
  out.facts = canonical_facts.has_value()
                  ? *canonical_facts
                  : build_processcvu_facts_from_runtime_config_internal(runtime);
  return out;
}

static ProcessCvuCanonicalCompileInputs
build_processcvu_mpk_detess_compile_inputs_local(const MpkContract& contract) {
  std::vector<const MpkPluginIoContract*> detess_stages;
  {
    const auto ordered = plugins_in_execution_order(contract);
    const auto mla_rank = mla_rank_in_order_local(contract, ordered);
    for (std::size_t rank = 0; rank < ordered.size(); ++rank) {
      if (mla_rank.has_value() && !(rank > *mla_rank)) {
        continue;
      }
      const std::size_t idx = ordered[rank];
      if (idx >= contract.plugins.size()) {
        continue;
      }
      const auto& stage = contract.plugins[idx];
      if (canonical_mpk_stage_kind_name(stage) == "detess") {
        detess_stages.push_back(&stage);
      }
    }
  }
  if (detess_stages.empty()) {
    throw std::runtime_error("processcvu MPK detess route missing detess post stages");
  }
  const auto mla_published_outputs = get_mla_published_outputs_contract(contract);
  const auto subsets = plugin_contracts::extract_detessellate_contract_subsets_from_mpk(contract);
  if (subsets.size() != detess_stages.size()) {
    throw std::runtime_error(
        "processcvu MPK detess route subset count does not match routed post stage count");
  }

  std::vector<const MpkPluginIoContract*> ordered_detess_stages(detess_stages.begin(),
                                                                detess_stages.end());
  std::vector<plugin_contracts::DetessellateContractSubset> ordered_subsets = subsets;
  {
    std::vector<std::pair<int, std::size_t>> ranked;
    ranked.reserve(detess_stages.size());
    bool can_sort = true;
    for (std::size_t i = 0; i < detess_stages.size(); ++i) {
      const auto* stage = detess_stages[i];
      if (!stage || stage->input_tensors.empty() || stage->input_tensors.front().name.empty()) {
        can_sort = false;
        break;
      }
      const std::string& name = stage->input_tensors.front().name;
      std::size_t pos = name.size();
      while (pos > 0U && std::isdigit(static_cast<unsigned char>(name[pos - 1U]))) {
        --pos;
      }
      if (pos == name.size() || pos == 0U || name[pos - 1U] != '_') {
        can_sort = false;
        break;
      }
      ranked.emplace_back(std::stoi(name.substr(pos)), i);
    }
    if (can_sort && ranked.size() == detess_stages.size()) {
      std::sort(ranked.begin(), ranked.end(),
                [](const auto& lhs, const auto& rhs) { return lhs.first < rhs.first; });
      bool unique = true;
      for (std::size_t i = 1; i < ranked.size(); ++i) {
        if (ranked[i - 1U].first == ranked[i].first) {
          unique = false;
          break;
        }
      }
      if (unique) {
        std::vector<const MpkPluginIoContract*> reordered_stages;
        std::vector<plugin_contracts::DetessellateContractSubset> reordered_subsets;
        reordered_stages.reserve(detess_stages.size());
        reordered_subsets.reserve(subsets.size());
        for (const auto& [_, original_index] : ranked) {
          reordered_stages.push_back(detess_stages[original_index]);
          reordered_subsets.push_back(subsets[original_index]);
        }
        ordered_detess_stages = std::move(reordered_stages);
        ordered_subsets = std::move(reordered_subsets);
      }
    }
  }
  if (mla_published_outputs.size() == detess_stages.size() && !mla_published_outputs.empty()) {
    std::vector<std::string> branch_keys;
    branch_keys.reserve(detess_stages.size());
    bool keys_ok = true;
    for (const auto* stage : detess_stages) {
      if (!stage || stage->input_tensors.empty() || stage->input_tensors.front().name.empty()) {
        keys_ok = false;
        break;
      }
      branch_keys.push_back(stage->input_tensors.front().name);
    }
    if (keys_ok) {
      std::vector<std::string> boundary_keys;
      boundary_keys.reserve(mla_published_outputs.size());
      for (const auto& published_output : mla_published_outputs) {
        boundary_keys.push_back(published_output.name);
      }
      if (auto perm = reorder_indices_by_mla_boundary_local(branch_keys, boundary_keys)) {
        std::vector<const MpkPluginIoContract*> reordered_stages;
        std::vector<plugin_contracts::DetessellateContractSubset> reordered_subsets;
        reordered_stages.reserve(perm->size());
        reordered_subsets.reserve(perm->size());
        for (auto idx : *perm) {
          reordered_stages.push_back(detess_stages[idx]);
          reordered_subsets.push_back(subsets[idx]);
        }
        ordered_detess_stages = std::move(reordered_stages);
        ordered_subsets = std::move(reordered_subsets);
      }
    }
  }

  std::vector<std::string> runtime_output_names;
  runtime_output_names.reserve(ordered_detess_stages.size());
  for (std::size_t i = 0; i < ordered_detess_stages.size(); ++i) {
    const auto& stage = *ordered_detess_stages[i];
    runtime_output_names.push_back(!stage.output_tensors.empty() &&
                                           !stage.output_tensors.front().name.empty()
                                       ? stage.output_tensors.front().name
                                       : ("output_tensor_" + std::to_string(i)));
  }
  auto runtime = plugin_contracts::build_detessellate_runtime_config_from_subsets(
      ordered_subsets, runtime_output_names, runtime_output_names);
  std::vector<std::uint64_t> packed_input_sizes;
  packed_input_sizes.reserve(ordered_subsets.size());
  for (std::size_t i = 0; i < ordered_subsets.size(); ++i) {
    const auto& stage = *ordered_detess_stages[i];
    const auto& subset = ordered_subsets[i];
    const std::string frame_type = normalize_dtype_token_local(subset.frame_type);
    if (i >= mla_published_outputs.size()) {
      throw std::runtime_error("processcvu MPK detess route requires MLA published boundary views");
    }
    const auto transport_view = validate_detess_ingress_transport_local(
        mla_published_outputs[i], frame_type, subset.input_transport_shape,
        subset.input_transport_size_bytes, stage.name);
    packed_input_sizes.push_back(transport_view.transport_size_bytes);
  }

  auto out = build_processcvu_compile_inputs_from_runtime_config(runtime);
  apply_published_routed_input_bindings(&out, mla_published_outputs, &packed_input_sizes,
                                        runtime.graph_family);
  force_direct_materialization_for_inputs(&out);

  out.facts.primary_output_name = runtime.primary_output_name;
  out.facts.published_output_names = runtime.published_output_names;
  for (std::size_t i = 0; i < out.facts.outputs.size(); ++i) {
    const std::string runtime_output_name =
        i < runtime.runtime_output_names.size() ? runtime.runtime_output_names[i] : std::string();
    const std::string published_output_name = i < runtime.published_output_names.size()
                                                  ? runtime.published_output_names[i]
                                                  : runtime_output_name;
    if (!runtime_output_name.empty()) {
      out.facts.outputs[i].physical_name = runtime_output_name;
    }
    out.facts.outputs[i].logical_name = published_output_name;
  }
  out.facts.output_order.clear();
  out.facts.output_order.reserve(out.facts.outputs.size());
  for (std::size_t i = 0; i < out.facts.outputs.size(); ++i) {
    ProcessCvuCanonicalRouteFact route;
    route.output_slot = out.facts.outputs[i].output_slot;
    route.logical_output_index = out.facts.outputs[i].logical_index;
    route.tensor_index = out.facts.outputs[i].tensor_index;
    route.cm_output_name = i < runtime.runtime_output_names.size()
                               ? runtime.runtime_output_names[i]
                               : out.facts.outputs[i].physical_name;
    route.segment_name = i < runtime.published_output_names.size()
                             ? runtime.published_output_names[i]
                             : out.facts.outputs[i].logical_name;
    out.facts.output_order.push_back(std::move(route));
  }
  return out;
}

static ProcessCvuCanonicalCompileInputs
build_processcvu_mpk_detesscast_compile_inputs_local(const MpkContract& contract) {
  std::vector<const MpkPluginIoContract*> detess_stages;
  {
    const auto ordered = plugins_in_execution_order(contract);
    const auto mla_rank = mla_rank_in_order_local(contract, ordered);
    for (std::size_t rank = 0; rank < ordered.size(); ++rank) {
      if (mla_rank.has_value() && !(rank > *mla_rank)) {
        continue;
      }
      const std::size_t idx = ordered[rank];
      if (idx >= contract.plugins.size()) {
        continue;
      }
      const auto& stage = contract.plugins[idx];
      if (canonical_mpk_stage_kind_name(stage) == "detess") {
        detess_stages.push_back(&stage);
      }
    }
  }
  if (detess_stages.empty()) {
    throw std::runtime_error("processcvu MPK detesscast route missing detess post stages");
  }
  const auto cast_stages = collect_post_stages_for_kind_names_local(contract, {"cast"});
  if (cast_stages.size() != detess_stages.size()) {
    throw std::runtime_error(
        "processcvu MPK detesscast route requires cast post stages matching detess stage count");
  }
  const auto mla_published_outputs = get_mla_published_outputs_contract(contract);
  const auto subsets = plugin_contracts::extract_detessellate_contract_subsets_from_mpk(contract);
  if (subsets.size() != detess_stages.size()) {
    throw std::runtime_error(
        "processcvu MPK detesscast route subset count does not match routed post stage count");
  }

  std::vector<const MpkPluginIoContract*> ordered_detess_stages(detess_stages.begin(),
                                                                detess_stages.end());
  std::vector<plugin_contracts::DetessellateContractSubset> ordered_subsets = subsets;
  {
    std::vector<std::pair<int, std::size_t>> ranked;
    ranked.reserve(detess_stages.size());
    bool can_sort = true;
    for (std::size_t i = 0; i < detess_stages.size(); ++i) {
      const auto* stage = detess_stages[i];
      if (!stage || stage->input_tensors.empty() || stage->input_tensors.front().name.empty()) {
        can_sort = false;
        break;
      }
      const std::string& name = stage->input_tensors.front().name;
      std::size_t pos = name.size();
      while (pos > 0U && std::isdigit(static_cast<unsigned char>(name[pos - 1U]))) {
        --pos;
      }
      if (pos == name.size() || pos == 0U || name[pos - 1U] != '_') {
        can_sort = false;
        break;
      }
      ranked.emplace_back(std::stoi(name.substr(pos)), i);
    }
    if (can_sort && ranked.size() == detess_stages.size()) {
      std::sort(ranked.begin(), ranked.end(),
                [](const auto& lhs, const auto& rhs) { return lhs.first < rhs.first; });
      bool unique = true;
      for (std::size_t i = 1; i < ranked.size(); ++i) {
        if (ranked[i - 1U].first == ranked[i].first) {
          unique = false;
          break;
        }
      }
      if (unique) {
        std::vector<const MpkPluginIoContract*> reordered_stages;
        std::vector<plugin_contracts::DetessellateContractSubset> reordered_subsets;
        reordered_stages.reserve(detess_stages.size());
        reordered_subsets.reserve(subsets.size());
        for (const auto& [_, original_index] : ranked) {
          reordered_stages.push_back(detess_stages[original_index]);
          reordered_subsets.push_back(subsets[original_index]);
        }
        ordered_detess_stages = std::move(reordered_stages);
        ordered_subsets = std::move(reordered_subsets);
      }
    }
  }
  if (mla_published_outputs.size() == detess_stages.size() && !mla_published_outputs.empty()) {
    std::vector<std::string> branch_keys;
    branch_keys.reserve(detess_stages.size());
    bool keys_ok = true;
    for (const auto* stage : detess_stages) {
      if (!stage || stage->input_tensors.empty() || stage->input_tensors.front().name.empty()) {
        keys_ok = false;
        break;
      }
      branch_keys.push_back(stage->input_tensors.front().name);
    }
    if (keys_ok) {
      std::vector<std::string> boundary_keys;
      boundary_keys.reserve(mla_published_outputs.size());
      for (const auto& published_output : mla_published_outputs) {
        boundary_keys.push_back(published_output.name);
      }
      if (auto perm = reorder_indices_by_mla_boundary_local(branch_keys, boundary_keys)) {
        std::vector<const MpkPluginIoContract*> reordered_stages;
        std::vector<plugin_contracts::DetessellateContractSubset> reordered_subsets;
        reordered_stages.reserve(perm->size());
        reordered_subsets.reserve(perm->size());
        for (auto idx : *perm) {
          reordered_stages.push_back(detess_stages[idx]);
          reordered_subsets.push_back(subsets[idx]);
        }
        ordered_detess_stages = std::move(reordered_stages);
        ordered_subsets = std::move(reordered_subsets);
      }
    }
  }

  std::vector<const MpkPluginIoContract*> ordered_cast_stages;
  ordered_cast_stages.reserve(ordered_detess_stages.size());
  {
    std::unordered_map<std::string, const MpkPluginIoContract*> cast_stage_by_input_name;
    bool can_match = true;
    for (const auto* cast_stage : cast_stages) {
      if (!cast_stage || cast_stage->input_tensors.empty() ||
          cast_stage->input_tensors.front().name.empty()) {
        can_match = false;
        break;
      }
      cast_stage_by_input_name.emplace(cast_stage->input_tensors.front().name, cast_stage);
    }
    if (can_match) {
      for (const auto* detess_stage : ordered_detess_stages) {
        if (!detess_stage || detess_stage->output_tensors.empty() ||
            detess_stage->output_tensors.front().name.empty()) {
          can_match = false;
          break;
        }
        const auto found = cast_stage_by_input_name.find(detess_stage->output_tensors.front().name);
        if (found == cast_stage_by_input_name.end()) {
          can_match = false;
          break;
        }
        ordered_cast_stages.push_back(found->second);
      }
    }
    if (!can_match || ordered_cast_stages.size() != ordered_detess_stages.size()) {
      ordered_cast_stages.assign(cast_stages.begin(), cast_stages.end());
    }
  }

  std::vector<std::string> runtime_output_names;
  std::vector<std::string> published_output_names;
  runtime_output_names.reserve(ordered_cast_stages.size());
  published_output_names.reserve(ordered_cast_stages.size());
  const auto* terminal_stage =
      find_terminal_stage_after_outputs_local(contract, ordered_cast_stages);
  for (std::size_t i = 0; i < ordered_cast_stages.size(); ++i) {
    const auto& cast_stage = *ordered_cast_stages[i];
    if (cast_stage.output_tensors.empty()) {
      throw std::runtime_error("processcvu MPK detesscast route requires cast outputs");
    }
    const auto& output_tensor = cast_stage.output_tensors.front();
    runtime_output_names.push_back("output_tensor_" + std::to_string(i));
    const auto* terminal_output_tensor =
        terminal_output_tensor_for_index_local(terminal_stage, i, ordered_cast_stages.size());
    published_output_names.push_back(published_output_name_from_terminal_or_producer_local(
        terminal_stage, terminal_output_tensor, output_tensor, i));
  }

  auto runtime = plugin_contracts::build_detessellate_runtime_config_from_subsets(
      ordered_subsets, runtime_output_names, published_output_names);
  const std::string packed_parent_input_name = "input_tensor";
  runtime.graph_family = "detesscast";
  runtime.graph_name = "detesscast";
  runtime.graph_id = 225;
  // detesscast consumes N logical MLA output views from one physical packed
  // parent buffer. The detessellate runtime helper names per-view inputs
  // input_tensor_0..N, but the fused detesscast graph/CM contract has one
  // parent graph input. Keep the payload, physical input list, and routing
  // facts on that same parent name; source_segment_name remains the upstream
  // producer segment (for example MLA_0) and is handled separately below.
  runtime.default_input_name = packed_parent_input_name;
  runtime.runtime_input_names = {packed_parent_input_name};
  runtime.physical_input_names = {packed_parent_input_name};
  runtime.physical_output_names = {"output_tensor"};
  runtime.published_output_names.clear();
  runtime.input_shapes.clear();
  runtime.input_tensors.clear();
  runtime.output_tensors.clear();
  runtime.runtime_output_dtype_list.clear();
  runtime.runtime_output_transport_kind_list.clear();
  runtime.runtime_output_semantic_kind_list.clear();
  runtime.runtime_output_logical_shapes.clear();
  runtime.runtime_output_logical_layout_list.clear();
  runtime.runtime_output_logical_index_list.clear();
  runtime.runtime_output_output_slot_list.clear();
  runtime.runtime_output_physical_index_list.clear();
  runtime.output_shapes.clear();
  runtime.primary_output_transport_kind = ProcessCvuOutputTransportKind::Dense;
  runtime.primary_output_semantic_kind = ProcessCvuOutputSemanticKind::Tensor;

  std::vector<ProcessCvuPackedRouteEntry> entries;
  entries.reserve(ordered_cast_stages.size());
  std::vector<MpkTensorContract> transport_inputs;
  transport_inputs.reserve(ordered_cast_stages.size());
  std::vector<std::uint64_t> packed_input_sizes;
  packed_input_sizes.reserve(ordered_subsets.size());
  std::uint64_t packed_output_offset = 0U;
  for (std::size_t i = 0; i < ordered_subsets.size(); ++i) {
    const auto& detess_stage = *ordered_detess_stages[i];
    const auto& cast_stage = *ordered_cast_stages[i];
    const auto& subset = ordered_subsets[i];
    const std::string frame_type = normalize_dtype_token_local(subset.frame_type);
    if (i >= mla_published_outputs.size()) {
      throw std::runtime_error(
          "processcvu MPK detesscast route requires MLA published boundary views");
    }
    const auto& published_input = mla_published_outputs[i];
    if (detess_stage.input_tensors.empty()) {
      throw std::runtime_error(
          "processcvu MPK detesscast route requires original detess transport tensors");
    }
    const auto& transport_input = detess_stage.input_tensors.front();
    const auto transport_view = validate_detess_ingress_transport_local(
        transport_input, frame_type, subset.input_transport_shape,
        subset.input_transport_size_bytes, detess_stage.name);
    packed_input_sizes.push_back(transport_view.transport_size_bytes);
    transport_inputs.push_back(transport_input);
    const auto& routed_input = transport_inputs.back();

    if (cast_stage.output_tensors.empty()) {
      throw std::runtime_error("processcvu MPK detesscast route requires cast outputs");
    }
    const auto& output_tensor = cast_stage.output_tensors.front();
    const std::string output_dtype = normalize_dtype_token_local(
        preferred_tensor_dtype_local(output_tensor, cast_stage.canonical_output_dtype));
    if (output_dtype != "FP32") {
      throw std::runtime_error("processcvu MPK detesscast route requires FP32 cast outputs");
    }
    const auto output_shape = preferred_mpk_tensor_shape_local(output_tensor);
    const auto output_dims = dims_from_tensor_shape_local(output_shape);
    if (output_dims.width <= 0 || output_dims.height <= 0 ||
        logical_channels_from_dims_local(output_dims) <= 0) {
      throw std::runtime_error("processcvu MPK detesscast route missing cast output geometry");
    }
    std::vector<int> output_shape_int(output_shape.begin(), output_shape.end());
    if (output_shape_int.empty()) {
      output_shape_int = {output_dims.height, output_dims.width,
                          logical_channels_from_dims_local(output_dims)};
    }
    const std::string output_layout;

    const std::string input_layout;
    const ShapeDims graph_input_dims =
        detess_dims_from_shape_local(subset.frame_shape, detess_stage.name + " input");

    if (graph_input_dims.width <= 0 || graph_input_dims.height <= 0 ||
        graph_input_dims.channels <= 0) {
      throw std::runtime_error("processcvu MPK detesscast route requires semantic input geometry");
    }

    if (i == 0U) {
      runtime.primary_output_name = published_output_names[i];
      runtime.input_dtype = frame_type;
      runtime.output_dtype = output_dtype;
      runtime.out_dtype = output_dtype;
    }
    std::vector<int> input_shape_int(subset.frame_shape.begin(), subset.frame_shape.end());
    runtime.input_shapes.push_back(input_shape_int);
    std::vector<int> tile_shape_int(subset.slice_shape.begin(), subset.slice_shape.end());
    tile_shape_int =
        tensor_desc_tile_shape_from_slice_shape_processcvu_local(input_shape_int, tile_shape_int);
    sima_ev_tensor_desc input_desc{};
    sima_ev_tensor_desc output_desc{};
    const bool c16_packed = subset.align_c16 || subset.cblock;
    if (!build_tensor_tiled_desc_processcvu_local(input_shape_int, tile_shape_int, frame_type, 0U,
                                                  c16_packed, &input_desc) ||
        !build_tensor_dense_desc_processcvu_local(output_shape_int, output_dtype, &output_desc)) {
      throw std::runtime_error(
          "processcvu MPK detesscast route could not synthesize explicit typed tensors");
    }
    runtime.input_tensors.push_back(input_desc);
    runtime.output_tensors.push_back(output_desc);
    runtime.output_shapes.push_back(output_shape_int);
    runtime.runtime_output_logical_index_list.push_back(static_cast<int>(i));
    runtime.runtime_output_output_slot_list.push_back(static_cast<int>(i));
    runtime.runtime_output_physical_index_list.push_back(static_cast<int>(i));
    runtime.runtime_output_dtype_list.push_back(output_dtype);
    runtime.runtime_output_transport_kind_list.push_back(ProcessCvuOutputTransportKind::Dense);
    runtime.runtime_output_semantic_kind_list.push_back(ProcessCvuOutputSemanticKind::Tensor);
    runtime.runtime_output_logical_shapes.push_back(output_shape_int);
    runtime.runtime_output_logical_layout_list.push_back(output_layout);
    runtime.published_output_names.push_back(published_output_names[i]);

    int src_physical_output_index = static_cast<int>(i);
    src_physical_output_index =
        published_input.source_physical_index >= 0 ? published_input.source_physical_index
        : published_input.physical_index >= 0      ? published_input.physical_index
                                                   : src_physical_output_index;

    ProcessCvuPackedRouteEntry entry;
    entry.logical_index = static_cast<int>(i);
    entry.output_slot = static_cast<int>(i);
    entry.physical_index = static_cast<int>(i);
    entry.input_tensor = &routed_input;
    entry.input_dtype = frame_type;
    entry.input_layout = input_layout;
    entry.input_byte_offset = routed_input.byte_offset;
    entry.output_tensor = &output_tensor;
    entry.output_physical_name = "output_tensor";
    entry.output_logical_name = published_output_names[i];
    entry.output_dtype = output_dtype;
    entry.output_layout = output_layout;
    entry.output_byte_offset = static_cast<std::int64_t>(packed_output_offset);
    entry.src_output_slot = static_cast<int>(i);
    entry.src_physical_output_index = src_physical_output_index;
    const std::uint64_t logical_output_size_bytes =
        logical_mpk_tensor_size_bytes_local(output_tensor, output_dtype);
    if (logical_output_size_bytes == 0U) {
      throw std::runtime_error(
          "processcvu MPK detesscast route requires a concrete output size for stage '" +
          detess_stage.name + "'");
    }
    packed_output_offset += logical_output_size_bytes;
    entries.push_back(std::move(entry));
  }

  ProcessCvuCanonicalCompileInputs out;
  out.payload = build_processcvu_payload_from_runtime_config_internal(runtime);
  const bool needs_bf16_noncompact_c16_lanesplit =
      std::any_of(ordered_subsets.begin(), ordered_subsets.end(), [](const auto& subset) {
        return normalize_dtype_token_local(subset.frame_type) == "BF16" && subset.align_c16 &&
               subset.cblock;
      });
  if (needs_bf16_noncompact_c16_lanesplit) {
    // Non-zero detesscast flags are interpreted as an exact option set by the
    // kernel. Preserve the existing optimized row-stripe options and add only
    // the explicit BF16 noncompact C16 lane-split reader required by the MPK
    // transport contract.
    out.payload.opt_flags |=
        kDetesscastDefaultOptimizedFlags | kDetesscastOptBf16NoncompactC16LaneSplit;
  }
  if (out.payload.input_tensors.size() == packed_input_sizes.size()) {
    for (std::size_t i = 0; i < packed_input_sizes.size(); ++i) {
      out.payload.input_tensors[i].storage.nbytes = packed_input_sizes[i];
      if (i < ordered_subsets.size() &&
          out.payload.input_tensors[i].layout_kind == SIMA_EV_LAYOUT_TILED) {
        if (ordered_subsets[i].align_c16 || ordered_subsets[i].cblock) {
          const auto& desc = out.payload.input_tensors[i];
          const int c_axis = tensorsemantics::find_shape_axis(desc.shape, SIMA_EV_AXIS_C);
          const bool c16_exact_tiles =
              c_axis >= 0 && c_axis < static_cast<int>(SIMA_EV_MAX_RANK) &&
              desc.shape.sizes[static_cast<std::uint32_t>(c_axis)] > 0 &&
              desc.layout.tiled.tile_sizes[static_cast<std::uint32_t>(c_axis)] > 0 &&
              (desc.shape.sizes[static_cast<std::uint32_t>(c_axis)] % 16) == 0 &&
              (desc.layout.tiled.tile_sizes[static_cast<std::uint32_t>(c_axis)] % 16) == 0;
          if (!c16_exact_tiles) {
            out.payload.input_tensors[i].layout.tiled.flags &=
                ~static_cast<std::uint32_t>(SIMA_EV_TILED_FLAG_COMPACT_CHANNELS);
          }
          out.payload.input_tensors[i].layout.tiled.tile_align_bytes = 16U;
        }
      }
    }
  }
  out.facts = build_processcvu_packed_route_facts(packed_parent_input_name, "output_tensor",
                                                  entries, runtime.primary_output_name,
                                                  runtime.published_output_names);
  apply_published_routed_input_bindings(&out, mla_published_outputs, &packed_input_sizes,
                                        runtime.graph_family);
  force_direct_materialization_for_inputs(&out);
  for (std::size_t i = 0; i < out.facts.outputs.size() && i < entries.size(); ++i) {
    if (!entries[i].output_tensor) {
      continue;
    }
    out.facts.outputs[i].shape = preferred_mpk_tensor_shape_local(*entries[i].output_tensor);
  }
  return out;
}

struct ProcessCvuDenseUnaryPostSpec {
  std::vector<std::int64_t> input_shape;
  std::vector<std::int64_t> output_shape;
  std::string input_dtype;
  std::string output_dtype;
};

static ProcessCvuCanonicalCompileInputs build_processcvu_mpk_dense_unary_post_route_local(
    const MpkContract& contract, CompiledProcessCvuRuntimeConfig runtime,
    const std::vector<const MpkPluginIoContract*>& stages,
    const std::vector<ProcessCvuDenseUnaryPostSpec>& specs, const std::string& route_name) {
  if (stages.empty()) {
    throw std::runtime_error(route_name + " route missing post stages");
  }
  if (stages.size() != specs.size()) {
    throw std::runtime_error(route_name +
                             " route spec count does not match routed post stage count");
  }

  const auto mla_published_outputs = get_mla_published_outputs_contract(contract);
  if (mla_published_outputs.size() < stages.size()) {
    throw std::runtime_error(route_name + " route requires MLA published boundary views");
  }
  const std::vector<MpkTensorContract> routed_mla_published_outputs(
      mla_published_outputs.begin(),
      mla_published_outputs.begin() + static_cast<std::ptrdiff_t>(stages.size()));

  const auto* terminal_stage = find_terminal_stage_after_outputs_local(contract, stages);
  std::vector<std::string> published_output_names;
  published_output_names.reserve(stages.size());
  for (std::size_t i = 0; i < stages.size(); ++i) {
    const auto* stage = stages[i];
    if (!stage || stage->output_tensors.empty()) {
      throw std::runtime_error(route_name + " route requires post stage outputs");
    }
    const auto* terminal_output_tensor =
        terminal_output_tensor_for_index_local(terminal_stage, i, stages.size());
    published_output_names.push_back(published_output_name_from_terminal_or_producer_local(
        terminal_stage, terminal_output_tensor, stage->output_tensors.front(), i));
  }
  if (runtime.published_output_names.size() != published_output_names.size()) {
    runtime.published_output_names = published_output_names;
  }

  runtime.default_input_name = "input_tensor";
  runtime.runtime_input_names = {"input_tensor"};
  runtime.physical_input_names = {"input_tensor"};
  runtime.runtime_output_names = {"output_tensor"};
  runtime.physical_output_names = {"output_tensor"};
  if (runtime.primary_output_name.empty()) {
    runtime.primary_output_name = runtime.published_output_names.front();
  }
  runtime.primary_output_transport_kind = ProcessCvuOutputTransportKind::Packed;
  runtime.primary_output_semantic_kind = ProcessCvuOutputSemanticKind::Tensor;

  if (runtime.input_shapes.size() != specs.size()) {
    runtime.input_shapes.clear();
    runtime.input_shapes.reserve(specs.size());
    for (const auto& spec : specs) {
      runtime.input_shapes.emplace_back(spec.input_shape.begin(), spec.input_shape.end());
    }
  }
  if (runtime.output_shapes.size() != specs.size()) {
    runtime.output_shapes.clear();
    runtime.output_shapes.reserve(specs.size());
    for (const auto& spec : specs) {
      runtime.output_shapes.emplace_back(spec.output_shape.begin(), spec.output_shape.end());
    }
  }
  if (runtime.input_tensors.size() != specs.size()) {
    runtime.input_tensors.clear();
    runtime.input_tensors.reserve(specs.size());
    for (std::size_t i = 0; i < specs.size(); ++i) {
      const auto* stage = i < stages.size() ? stages[i] : nullptr;
      const auto desc_shape64 =
          stage && !stage->input_tensors.empty()
              ? preferred_physical_mpk_tensor_shape_local(stage->input_tensors.front())
              : specs[i].input_shape;
      std::vector<int> shape(desc_shape64.begin(), desc_shape64.end());
      sima_ev_tensor_desc desc{};
      if (!build_tensor_dense_desc_processcvu_local(shape, specs[i].input_dtype, &desc)) {
        throw std::runtime_error(route_name + " route could not synthesize input typed tensor");
      }
      runtime.input_tensors.push_back(desc);
    }
  }
  if (runtime.output_tensors.size() != specs.size()) {
    runtime.output_tensors.clear();
    runtime.output_tensors.reserve(specs.size());
    for (std::size_t i = 0; i < specs.size(); ++i) {
      const auto* stage = i < stages.size() ? stages[i] : nullptr;
      const auto desc_shape64 =
          stage && !stage->output_tensors.empty()
              ? preferred_physical_mpk_tensor_shape_local(stage->output_tensors.front())
              : specs[i].output_shape;
      std::vector<int> shape(desc_shape64.begin(), desc_shape64.end());
      sima_ev_tensor_desc desc{};
      if (!build_tensor_dense_desc_processcvu_local(shape, specs[i].output_dtype, &desc)) {
        throw std::runtime_error(route_name + " route could not synthesize output typed tensor");
      }
      runtime.output_tensors.push_back(desc);
    }
  }
  if (runtime.input_dtype.empty()) {
    runtime.input_dtype = specs.front().input_dtype;
  }
  if (runtime.output_dtype.empty()) {
    runtime.output_dtype = specs.front().output_dtype;
  }
  if (runtime.out_dtype.empty()) {
    runtime.out_dtype = runtime.output_dtype;
  }

  std::uint64_t packed_input_offset = 0U;
  std::uint64_t packed_output_offset = 0U;
  std::vector<MpkTensorContract> synthetic_inputs;
  std::vector<MpkTensorContract> synthetic_outputs;
  synthetic_inputs.reserve(stages.size());
  synthetic_outputs.reserve(stages.size());
  std::vector<ProcessCvuPackedRouteEntry> entries;
  entries.reserve(stages.size());

  for (std::size_t i = 0; i < stages.size(); ++i) {
    const auto& spec = specs[i];
    if (spec.input_shape.empty() || spec.output_shape.empty() || spec.input_dtype.empty() ||
        spec.output_dtype.empty()) {
      throw std::runtime_error(route_name + " route requires concrete shapes and dtypes");
    }
    synthetic_inputs.push_back(make_synthetic_tensor_contract_local(
        spec.input_shape, spec.input_dtype, static_cast<int>(i),
        "input_tensor_" + std::to_string(i)));
    synthetic_outputs.push_back(make_synthetic_tensor_contract_local(
        spec.output_shape, spec.output_dtype, static_cast<int>(i),
        runtime.published_output_names[i]));
    synthetic_inputs.back().logical_shape = synthetic_inputs.back().mpk_shape;
    synthetic_outputs.back().logical_shape = synthetic_outputs.back().mpk_shape;

    const auto& input_tensor = synthetic_inputs.back();
    const auto& output_tensor = synthetic_outputs.back();
    const MpkTensorDims input_dims = dims_from_tensor_shape_local(input_tensor.logical_shape);
    const MpkTensorDims output_dims = dims_from_tensor_shape_local(output_tensor.logical_shape);
    if (input_dims.width <= 0 || input_dims.height <= 0 ||
        logical_channels_from_dims_local(input_dims) <= 0 || output_dims.width <= 0 ||
        output_dims.height <= 0 || logical_channels_from_dims_local(output_dims) <= 0) {
      const std::string stage_name = stages[i] ? stages[i]->name : std::string("<unknown>");
      throw std::runtime_error(route_name + " route stage '" + stage_name +
                               "' is missing geometry");
    }

    ProcessCvuPackedRouteEntry entry;
    entry.logical_index = static_cast<int>(i);
    entry.output_slot = static_cast<int>(i);
    entry.physical_index = static_cast<int>(i);
    entry.input_tensor = &input_tensor;
    entry.input_dtype = spec.input_dtype;
    entry.input_byte_offset = static_cast<std::int64_t>(packed_input_offset);
    const std::uint64_t input_bytes =
        preferred_mpk_tensor_size_bytes_local(input_tensor, spec.input_dtype);
    if (input_bytes == 0U) {
      throw std::runtime_error(route_name + " route requires concrete input size");
    }
    packed_input_offset += input_bytes;
    entry.output_tensor = &output_tensor;
    entry.output_physical_name = "output_tensor";
    entry.output_logical_name = runtime.published_output_names[i];
    entry.output_dtype = spec.output_dtype;
    entry.output_byte_offset = static_cast<std::int64_t>(packed_output_offset);
    entry.src_output_slot = static_cast<int>(i);
    entry.src_physical_output_index = static_cast<int>(i);
    const std::uint64_t output_bytes =
        preferred_mpk_tensor_size_bytes_local(output_tensor, spec.output_dtype);
    if (output_bytes == 0U) {
      throw std::runtime_error(route_name + " route requires concrete output size");
    }
    packed_output_offset += output_bytes;
    entries.push_back(std::move(entry));
  }

  // The dense-unary post handoff has one physical parent output with N published
  // logical views. Keep typed descriptors per logical tensor for ConfigManager;
  // runtime-output metadata describes only the physical parent.
  runtime.output_shapes = runtime.output_shapes.empty()
                              ? std::vector<std::vector<int>>{}
                              : std::vector<std::vector<int>>{runtime.output_shapes.front()};
  runtime.runtime_output_logical_index_list = {0};
  runtime.runtime_output_output_slot_list = {0};
  runtime.runtime_output_physical_index_list = {0};
  runtime.runtime_output_dtype_list = {runtime.output_dtype.empty() ? specs.front().output_dtype
                                                                    : runtime.output_dtype};
  runtime.runtime_output_transport_kind_list = {ProcessCvuOutputTransportKind::Packed};
  runtime.runtime_output_semantic_kind_list = {ProcessCvuOutputSemanticKind::Tensor};
  runtime.runtime_output_logical_shapes.clear();
  if (!runtime.output_shapes.empty()) {
    runtime.runtime_output_logical_shapes.push_back(runtime.output_shapes.front());
  }
  runtime.runtime_output_logical_layout_list = {runtime_output_layout_token_local(runtime)};

  ProcessCvuCanonicalCompileInputs out;
  out.payload = build_processcvu_payload_from_runtime_config_internal(runtime);
  out.facts = build_processcvu_packed_route_facts("input_tensor", "output_tensor", entries,
                                                  runtime.primary_output_name,
                                                  runtime.published_output_names);
  apply_published_routed_input_bindings(&out, routed_mla_published_outputs, nullptr,
                                        runtime.graph_family);
  if (published_inputs_share_single_physical_parent(routed_mla_published_outputs)) {
    enforce_packed_parent_input_views(&out, "input_tensor", entries, {},
                                      &routed_mla_published_outputs);
  } else {
    preserve_routed_source_segment_input_views(&out, "input_tensor");
  }
  for (std::size_t i = 0;
       i < routed_mla_published_outputs.size() && i < out.payload.input_tensors.size(); ++i) {
    const std::string input_dtype =
        i < out.facts.inputs.size() && !out.facts.inputs[i].dtype.empty()
            ? out.facts.inputs[i].dtype
            : runtime.input_dtype;
    override_payload_input_desc_from_published_view(&out, i, routed_mla_published_outputs[i],
                                                    input_dtype);
  }
  force_direct_materialization_for_inputs(&out);
  return out;
}

static ProcessCvuCanonicalCompileInputs
build_processcvu_mpk_cast_compile_inputs_local(const MpkContract& contract) {
  const auto cast_stages = collect_post_stages_for_kind_names_local(contract, {"cast"});
  if (cast_stages.empty()) {
    throw std::runtime_error("processcvu MPK cast route missing cast post stages");
  }

  std::vector<ProcessCvuDenseUnaryPostSpec> specs;
  specs.reserve(cast_stages.size());
  for (const auto* stage : cast_stages) {
    if (!stage || stage->input_tensors.empty() || stage->output_tensors.empty()) {
      throw std::runtime_error("processcvu MPK cast route requires input/output tensors");
    }
    ProcessCvuDenseUnaryPostSpec spec;
    spec.input_shape =
        preferred_stage_input_tensor_shape_local(*stage, stage->input_tensors.front());
    spec.output_shape = preferred_mpk_tensor_shape_local(stage->output_tensors.front());
    spec.input_dtype = normalize_dtype_token_local(
        preferred_tensor_dtype_local(stage->input_tensors.front(), stage->canonical_input_dtype));
    spec.output_dtype = normalize_dtype_token_local(
        preferred_tensor_dtype_local(stage->output_tensors.front(), stage->canonical_output_dtype));
    if (spec.input_shape.empty() || spec.output_shape.empty()) {
      throw std::runtime_error("processcvu MPK cast route requires concrete input/output shapes");
    }
    if (spec.input_shape.size() != spec.output_shape.size()) {
      throw std::runtime_error("processcvu MPK cast route requires matching input/output ranks");
    }
    for (std::size_t axis = 0; axis < spec.input_shape.size(); ++axis) {
      if (spec.input_shape[axis] != spec.output_shape[axis]) {
        throw std::runtime_error("processcvu MPK cast route requires matching logical shapes");
      }
    }
    if (!((spec.input_dtype == "FP32" && spec.output_dtype == "BF16") ||
          (spec.input_dtype == "BF16" && spec.output_dtype == "FP32"))) {
      throw std::runtime_error("processcvu MPK cast route requires FP32<->BF16 dtype pair");
    }
    specs.push_back(std::move(spec));
  }

  CompiledProcessCvuRuntimeConfig runtime;
  runtime.graph_family = "cast";
  runtime.graph_name = "cast";
  runtime.graph_id = 221;
  runtime.batch_size = 1;
  runtime.byte_align = 1;
  runtime.input_dtype = specs.front().input_dtype;
  runtime.output_dtype = specs.front().output_dtype;
  runtime.out_dtype = runtime.output_dtype;
  return build_processcvu_mpk_dense_unary_post_route_local(
      contract, std::move(runtime), cast_stages, specs, "processcvu MPK cast");
}

static ProcessCvuCanonicalCompileInputs
build_processcvu_mpk_dequant_compile_inputs_local(const MpkContract& contract) {
  const auto dequant_stages = collect_post_stages_for_kind_names_local(contract, {"dequant"});
  if (dequant_stages.empty()) {
    throw std::runtime_error("processcvu MPK dequant route missing dequant post stages");
  }
  const auto subsets = plugin_contracts::extract_dequantize_contract_subsets_from_mpk(contract);
  if (subsets.size() != dequant_stages.size()) {
    throw std::runtime_error(
        "processcvu MPK dequant route subset count does not match routed post stage count");
  }

  const auto* terminal_stage = find_terminal_stage_after_outputs_local(contract, dequant_stages);
  std::vector<std::string> published_output_names;
  published_output_names.reserve(dequant_stages.size());
  for (std::size_t i = 0; i < dequant_stages.size(); ++i) {
    const auto& stage = *dequant_stages[i];
    if (stage.output_tensors.empty()) {
      throw std::runtime_error("processcvu MPK dequant route requires dequant outputs");
    }
    const auto* terminal_output_tensor =
        terminal_output_tensor_for_index_local(terminal_stage, i, dequant_stages.size());
    published_output_names.push_back(published_output_name_from_terminal_or_producer_local(
        terminal_stage, terminal_output_tensor, stage.output_tensors.front(), i));
  }

  CompiledProcessCvuRuntimeConfig runtime =
      plugin_contracts::build_dequantize_runtime_config_from_subsets(subsets,
                                                                     published_output_names);

  std::vector<ProcessCvuDenseUnaryPostSpec> specs;
  specs.reserve(subsets.size());
  for (const auto& subset : subsets) {
    ProcessCvuDenseUnaryPostSpec spec;
    spec.input_shape = subset.input_shape;
    spec.output_shape = subset.output_shape;
    spec.input_dtype = normalize_dtype_token_local(subset.input_dtype);
    spec.output_dtype = normalize_dtype_token_local(subset.output_dtype);
    specs.push_back(std::move(spec));
  }

  return build_processcvu_mpk_dense_unary_post_route_local(
      contract, std::move(runtime), dequant_stages, specs, "processcvu MPK dequant");
}

static ProcessCvuCanonicalCompileInputs
build_processcvu_mpk_detessdequant_compile_inputs_local(const MpkContract& contract) {
  const auto stage_pairs = plugin_contracts::resolve_detessdequant_stage_pairs_from_mpk(contract);
  if (stage_pairs.empty()) {
    throw std::runtime_error("processcvu MPK detessdequant route missing post stages");
  }

  std::vector<const MpkPluginIoContract*> dequant_stages;
  dequant_stages.reserve(stage_pairs.size());
  for (const auto& pair : stage_pairs) {
    if (!pair.detess || !pair.dequant) {
      throw std::runtime_error(
          "processcvu MPK detessdequant route resolved an incomplete stage pair");
    }
    dequant_stages.push_back(pair.dequant);
  }

  const std::size_t count = stage_pairs.size();
  const auto mla_published_outputs = get_mla_published_outputs_contract(contract);
  const auto subset = plugin_contracts::extract_detessdequant_contract_subset_from_mpk(contract);
  if (subset.heads.size() < count) {
    throw std::runtime_error(
        "processcvu MPK detessdequant contract subset is smaller than the routed post stage count");
  }
  std::vector<plugin_contracts::DetessDequantStagePair> ordered_stage_pairs(stage_pairs.begin(),
                                                                            stage_pairs.end());
  plugin_contracts::DetessDequantContractSubset ordered_subset = subset;
  {
    std::vector<std::pair<int, std::size_t>> ranked;
    ranked.reserve(stage_pairs.size());
    bool can_sort = true;
    for (std::size_t i = 0; i < stage_pairs.size(); ++i) {
      const auto* detess = stage_pairs[i].detess;
      if (!detess || detess->input_tensors.empty() || detess->input_tensors.front().name.empty()) {
        can_sort = false;
        break;
      }
      const std::string& name = detess->input_tensors.front().name;
      std::size_t pos = name.size();
      while (pos > 0U && std::isdigit(static_cast<unsigned char>(name[pos - 1U]))) {
        --pos;
      }
      if (pos == name.size() || pos == 0U || name[pos - 1U] != '_') {
        can_sort = false;
        break;
      }
      ranked.emplace_back(std::stoi(name.substr(pos)), i);
    }
    if (can_sort && ranked.size() == stage_pairs.size()) {
      std::sort(ranked.begin(), ranked.end(),
                [](const auto& lhs, const auto& rhs) { return lhs.first < rhs.first; });
      bool unique = true;
      for (std::size_t i = 1; i < ranked.size(); ++i) {
        if (ranked[i - 1U].first == ranked[i].first) {
          unique = false;
          break;
        }
      }
      if (unique) {
        std::vector<plugin_contracts::DetessDequantStagePair> reordered_pairs;
        std::vector<plugin_contracts::DetessDequantHeadContractSubset> reordered_heads;
        reordered_pairs.reserve(stage_pairs.size());
        reordered_heads.reserve(ordered_subset.heads.size());
        for (const auto& [_, original_index] : ranked) {
          reordered_pairs.push_back(stage_pairs[original_index]);
          reordered_heads.push_back(ordered_subset.heads[original_index]);
        }
        ordered_stage_pairs = std::move(reordered_pairs);
        ordered_subset.heads = std::move(reordered_heads);
      }
    }
  }
  bool direct_mla_boundary_match = false;
  if (mla_published_outputs.size() == count && !mla_published_outputs.empty()) {
    std::vector<std::string> branch_keys;
    branch_keys.reserve(ordered_stage_pairs.size());
    bool keys_ok = true;
    for (const auto& pair : ordered_stage_pairs) {
      const auto* detess = pair.detess;
      if (!detess || detess->input_tensors.empty() || detess->input_tensors.front().name.empty()) {
        keys_ok = false;
        break;
      }
      branch_keys.push_back(detess->input_tensors.front().name);
    }
    if (keys_ok) {
      std::vector<std::string> boundary_keys;
      boundary_keys.reserve(mla_published_outputs.size());
      for (const auto& published_output : mla_published_outputs) {
        boundary_keys.push_back(published_output.name);
      }
      if (auto perm = reorder_indices_by_mla_boundary_local(branch_keys, boundary_keys)) {
        std::vector<plugin_contracts::DetessDequantStagePair> reordered_pairs;
        std::vector<plugin_contracts::DetessDequantHeadContractSubset> reordered_heads;
        reordered_pairs.reserve(perm->size());
        reordered_heads.reserve(perm->size());
        for (auto idx : *perm) {
          reordered_pairs.push_back(ordered_stage_pairs[idx]);
          reordered_heads.push_back(ordered_subset.heads[idx]);
        }
        ordered_stage_pairs = std::move(reordered_pairs);
        ordered_subset.heads = std::move(reordered_heads);
        direct_mla_boundary_match = true;
      }
    }
  }

  dequant_stages.clear();
  dequant_stages.reserve(ordered_stage_pairs.size());
  for (const auto& pair : ordered_stage_pairs) {
    dequant_stages.push_back(pair.dequant);
  }

  // Graph 227 has six (or generally N) logical YOLO outputs, but one physical
  // dispatcher/ConfigManager output segment.  Keep runtime outputs physical so
  // ConfigManager allocates a single `output_tensor`; the logical outputs are
  // represented below by packed-route facts with byte offsets into that parent.
  std::vector<std::string> runtime_output_names = {"output_tensor"};

  CompiledProcessCvuRuntimeConfig runtime =
      plugin_contracts::build_detessdequant_runtime_config_from_subset(ordered_subset,
                                                                       runtime_output_names);
  if (processcvu_detess_layout_debug_enabled()) {
    const auto first_frame_shape = ordered_subset.heads.empty()
                                       ? std::vector<std::int64_t>{}
                                       : ordered_subset.heads.front().frame_shape;
    std::fprintf(stderr,
                 "[detess-layout-debug] where=stage.runtime_from_subset heads=%zu input_layout=%s "
                 "output_layout=%s output_shapes=%s frame_shape0=%s published_outputs=%zu\n",
                 ordered_subset.heads.size(), runtime_input_layout_token_local(runtime).c_str(),
                 runtime_output_layout_token_local(runtime).c_str(),
                 ints2d_dbg_processcvu_local(runtime.output_shapes).c_str(),
                 join_i64_debug_processcvu_local(first_frame_shape).c_str(),
                 runtime.published_output_names.size());
  }
  std::vector<ProcessCvuPackedRouteEntry> entries;
  entries.reserve(count);
  std::vector<MpkTensorContract> synthetic_inputs;
  synthetic_inputs.reserve(count);
  std::vector<std::uint64_t> packed_input_sizes;
  packed_input_sizes.reserve(count);
  std::vector<std::vector<std::int64_t>> canonical_output_shapes;
  canonical_output_shapes.reserve(count);
  const auto* terminal_stage = find_terminal_stage_after_outputs_local(contract, dequant_stages);
  runtime.published_output_names.clear();
  runtime.published_output_names.reserve(count);
  runtime.input_shapes.clear();
  runtime.input_shapes.reserve(count);
  runtime.input_tensors.clear();
  runtime.input_tensors.reserve(count);
  runtime.output_shapes.clear();
  runtime.output_tensors.clear();
  runtime.output_tensors.reserve(count);
  runtime.runtime_output_logical_index_list.clear();
  runtime.runtime_output_output_slot_list.clear();
  runtime.runtime_output_physical_index_list.clear();
  runtime.runtime_output_dtype_list.clear();
  runtime.runtime_output_transport_kind_list.clear();
  runtime.runtime_output_semantic_kind_list.clear();
  runtime.runtime_output_logical_shapes.clear();
  runtime.runtime_output_logical_layout_list.clear();
  runtime.output_shapes.reserve(count);
  runtime.runtime_output_logical_index_list.reserve(count);
  runtime.runtime_output_output_slot_list.reserve(count);
  runtime.runtime_output_physical_index_list.reserve(count);
  runtime.runtime_output_dtype_list.reserve(count);
  for (const auto& head : ordered_subset.heads) {
    std::vector<int> input_shape_int(head.frame_shape.begin(), head.frame_shape.end());
    runtime.input_shapes.push_back(std::move(input_shape_int));
  }
  std::uint64_t packed_input_offset = 0U;
  std::uint64_t packed_output_offset = 0U;
  if (mla_published_outputs.size() < count) {
    throw std::runtime_error(
        "processcvu MPK detessdequant route requires MLA published boundary views");
  }

  for (std::size_t i = 0; i < count; ++i) {
    const auto& detess = *ordered_stage_pairs[i].detess;
    const auto& dequant = *ordered_stage_pairs[i].dequant;
    const auto& head = ordered_subset.heads[i];
    const auto& published_input = mla_published_outputs[i];
    if (detess.input_tensors.empty() || detess.output_tensors.empty() ||
        dequant.output_tensors.empty()) {
      throw std::runtime_error("processcvu MPK detessdequant route missing tensor metadata");
    }
    const auto& dequant_output_tensor = dequant.output_tensors.front();
    const std::string resolved_input_dtype = normalize_dtype_token_local(head.frame_type);
    const auto transport_view = validate_detess_ingress_transport_local(
        published_input, resolved_input_dtype, head.input_transport_shape,
        head.input_transport_size_bytes, detess.name);
    synthetic_inputs.push_back(make_synthetic_tensor_contract_local(
        head.frame_shape, resolved_input_dtype, static_cast<int>(i),
        "input_tensor_" + std::to_string(i)));
    synthetic_inputs.back().size_bytes =
        static_cast<std::size_t>(transport_view.transport_size_bytes);
    packed_input_sizes.push_back(transport_view.transport_size_bytes);
    const auto& canonical_input_tensor = synthetic_inputs.back();
    const auto* terminal_output_tensor =
        terminal_output_tensor_for_index_local(terminal_stage, i, count);
    const MpkTensorDims detess_output_dims =
        dims_from_tensor_shape_local(detess.output_tensors.front().logical_shape);
    const MpkTensorDims terminal_dims =
        terminal_output_tensor ? dims_from_tensor_shape_local(terminal_output_tensor->logical_shape)
                               : MpkTensorDims{};
    const MpkTensorDims dequant_dims =
        dims_from_tensor_shape_local(dequant_output_tensor.logical_shape);
    const ShapeDims graph_output_dims =
        detess_dims_from_shape_local(head.frame_shape, detess.name + " output");
    const std::string graph_output_layout;
    const bool use_terminal_dims = terminal_output_tensor && terminal_dims.width > 0 &&
                                   terminal_dims.height > 0 && terminal_dims.depth > 0;
    std::string logical_input_layout;
    const ShapeDims graph_input_dims =
        detess_dims_from_shape_local(head.frame_shape, detess.name + " input");
    const int input_width = graph_input_dims.width;
    const int input_height = graph_input_dims.height;
    const int input_channels = graph_input_dims.channels;
    const int output_width = graph_output_dims.width;
    const int output_height = graph_output_dims.height;
    const int output_channels = graph_output_dims.channels;
    const int output_depth = graph_output_dims.depth;

    const std::string published_output_name = published_output_name_from_terminal_or_producer_local(
        terminal_stage, terminal_output_tensor, dequant_output_tensor, i);
    if (input_width <= 0 || input_height <= 0 || input_channels <= 0 || output_width <= 0 ||
        output_height <= 0 || output_channels <= 0 ||
        runtime_slice_dims_at(runtime, i).width <= 0 ||
        runtime_slice_dims_at(runtime, i).height <= 0 ||
        runtime_slice_dims_at(runtime, i).channels <= 0 || resolved_input_dtype.empty() ||
        head.output_dtype.empty()) {
      throw std::runtime_error("processcvu MPK detessdequant route requires complete geometry, "
                               "tile, dtype, and quant facts");
    }

    if (i == 0U) {
      runtime.primary_output_name = published_output_name;
      // slice_shapes already populated by build_detessdequant_runtime_config_from_subset.
      runtime.input_dtype = resolved_input_dtype;
      runtime.output_dtype = head.output_dtype;
      runtime.out_dtype = runtime.output_dtype;
    }
    // Phase 3a (Option A++): build the kernel input descriptor from a
    // per-frame shape so its rank matches `slice_shape` rank, and divide the
    // transport size by batch so storage.nbytes is per-frame. The dispatcher
    // sizes the input segment as storage.nbytes * runtime.batch_size, so
    // passing a batched storage size double-counts the batch dim.
    const int per_frame_rank_local = plugin_contracts::derive_per_frame_rank_public(
        head.slice_shape, /*peer_per_frame_shape=*/{});
    const auto frame_shape_per_frame = plugin_contracts::semantic_shape_without_batch_public(
        head.frame_shape, per_frame_rank_local);
    const std::vector<int> input_shape_int(frame_shape_per_frame.begin(),
                                           frame_shape_per_frame.end());
    std::vector<int> tile_shape_int =
        i < runtime.slice_shapes.size() ? runtime.slice_shapes[i] : std::vector<int>{};
    tile_shape_int =
        tensor_desc_tile_shape_from_slice_shape_processcvu_local(input_shape_int, tile_shape_int);
    sima_ev_tensor_desc input_desc{};
    if (tile_shape_int.empty() || !build_tensor_tiled_desc_processcvu_local(
                                      input_shape_int, tile_shape_int, resolved_input_dtype, 0U,
                                      head.align_c16 || head.cblock, &input_desc)) {
      throw std::runtime_error(
          "processcvu MPK detessdequant route could not synthesize explicit input tensor");
    }
    const int local_head_batch_size = plugin_contracts::inferred_batch_size_from_shape_public(
        head.frame_shape, per_frame_rank_local);
    const std::uint64_t per_frame_transport_size =
        local_head_batch_size > 0
            ? head.input_transport_size_bytes / static_cast<std::uint64_t>(local_head_batch_size)
            : head.input_transport_size_bytes;
    input_desc.storage.nbytes = per_frame_transport_size;
    runtime.input_tensors.push_back(input_desc);

    runtime.published_output_names.push_back(published_output_name);

    int src_physical_output_index = static_cast<int>(i);
    src_physical_output_index =
        published_input.source_physical_index >= 0 ? published_input.source_physical_index
        : published_input.physical_index >= 0      ? published_input.physical_index
                                                   : src_physical_output_index;
    ProcessCvuPackedRouteEntry entry;
    entry.logical_index = static_cast<int>(i);
    entry.output_slot = static_cast<int>(i);
    entry.physical_index = static_cast<int>(i);
    entry.input_tensor = &canonical_input_tensor;
    entry.input_dtype = resolved_input_dtype;
    entry.input_layout = logical_input_layout;
    entry.input_byte_offset = static_cast<std::int64_t>(packed_input_offset);
    const std::uint64_t logical_input_size_bytes =
        preferred_mpk_tensor_size_bytes_local(canonical_input_tensor, resolved_input_dtype);
    if (logical_input_size_bytes == 0U) {
      throw std::runtime_error(
          "processcvu MPK detessdequant route requires a concrete input size for stage '" +
          detess.name + "'");
    }
    packed_input_offset += logical_input_size_bytes;

    const std::string output_dtype = head.output_dtype;
    const auto& logical_output_tensor = (use_terminal_dims && terminal_output_tensor)
                                            ? *terminal_output_tensor
                                            : dequant_output_tensor;
    const auto logical_output_shape = preferred_mpk_tensor_shape_local(logical_output_tensor);
    const auto logical_output_dims = dims_from_tensor_shape_local(logical_output_shape);
    const std::string logical_output_layout;
    const std::string output_layout;
    entry.output_tensor = &logical_output_tensor;
    entry.output_physical_name = "output_tensor";
    entry.output_logical_name = published_output_name;
    entry.output_dtype = output_dtype;
    entry.output_layout = output_layout;
    entry.output_byte_offset = static_cast<std::int64_t>(packed_output_offset);
    entry.src_output_slot = static_cast<int>(i);
    entry.src_physical_output_index = src_physical_output_index;
    const std::uint64_t logical_output_size_bytes =
        logical_mpk_tensor_size_bytes_local(logical_output_tensor, output_dtype);
    if (logical_output_size_bytes == 0U) {
      throw std::runtime_error(
          "processcvu MPK detessdequant route requires a concrete output size for stage '" +
          detess.name + "'");
    }
    packed_output_offset += logical_output_size_bytes;
    if (i == 0U) {
      runtime.primary_output_transport_kind = ProcessCvuOutputTransportKind::Dense;
      runtime.primary_output_semantic_kind = ProcessCvuOutputSemanticKind::Tensor;
    }
    const auto canonical_runtime_output_shape =
        plugin_contracts::canonical_value_transform_shape_public(
            "detessdequant", head.frame_shape, logical_output_shape, head.frame_shape);
    canonical_output_shapes.push_back(canonical_runtime_output_shape);
    std::vector<int> output_shape_int(canonical_runtime_output_shape.begin(),
                                      canonical_runtime_output_shape.end());
    const auto output_shape_per_frame = plugin_contracts::semantic_shape_without_batch_public(
        canonical_runtime_output_shape, per_frame_rank_local);
    std::vector<int> output_desc_shape_int(output_shape_per_frame.begin(),
                                           output_shape_per_frame.end());
    sima_ev_tensor_desc output_desc{};
    if (!build_tensor_dense_desc_processcvu_local(output_desc_shape_int, output_dtype,
                                                  &output_desc)) {
      throw std::runtime_error(
          "processcvu MPK detessdequant route could not synthesize explicit output tensor");
    }
    runtime.output_tensors.push_back(output_desc);
    runtime.output_shapes.push_back(output_shape_int);
    runtime.runtime_output_logical_index_list.push_back(static_cast<int>(i));
    runtime.runtime_output_output_slot_list.push_back(static_cast<int>(i));
    runtime.runtime_output_physical_index_list.push_back(static_cast<int>(i));
    runtime.runtime_output_dtype_list.push_back(output_dtype);
    runtime.runtime_output_transport_kind_list.push_back(ProcessCvuOutputTransportKind::Dense);
    runtime.runtime_output_semantic_kind_list.push_back(ProcessCvuOutputSemanticKind::Tensor);
    runtime.runtime_output_logical_shapes.push_back(output_shape_int);
    runtime.runtime_output_logical_layout_list.push_back(output_layout);
    if (processcvu_detess_layout_debug_enabled()) {
      std::fprintf(
          stderr,
          "[detess-layout-debug] where=builder.entry index=%zu name=%s "
          "detess_dims={layout=%s,h=%d,w=%d,d=%d} "
          "dequant_dims={layout=%s,h=%d,w=%d,d=%d} terminal_dims={layout=%s,h=%d,w=%d,d=%d} "
          "graph_output={layout=%s,h=%d,w=%d,d=%d,c=%d} output_channels=%d output_depth=%d "
          "entry_layout=%s logical_shape=%s "
          "dequant_shape=%s terminal_shape=%s use_terminal=%d\n",
          i, published_output_name.c_str(), detess_output_dims.layout.c_str(),
          detess_output_dims.height, detess_output_dims.width, detess_output_dims.depth,
          dequant_dims.layout.c_str(), dequant_dims.height, dequant_dims.width, dequant_dims.depth,
          terminal_dims.layout.c_str(), terminal_dims.height, terminal_dims.width,
          terminal_dims.depth, graph_output_layout.c_str(), graph_output_dims.height,
          graph_output_dims.width, graph_output_dims.depth, graph_output_dims.channels,
          output_channels, output_depth, entry.output_layout.c_str(),
          join_i64_debug_processcvu_local(logical_output_shape).c_str(),
          join_i64_debug_processcvu_local(preferred_mpk_tensor_shape_local(dequant_output_tensor))
              .c_str(),
          terminal_output_tensor ? join_i64_debug_processcvu_local(
                                       preferred_mpk_tensor_shape_local(*terminal_output_tensor))
                                       .c_str()
                                 : "<none>",
          use_terminal_dims ? 1 : 0);
    }
    entries.push_back(std::move(entry));
  }

  // Reconcile the runtime physical-output contract after the per-head loop.
  // `output_tensors`/`output_shapes` intentionally remain per-logical-head so
  // graph 227 still receives one descriptor per tensor.  The runtime-output
  // metadata arrays, however, describe the physical buffer(s) exposed through
  // processcvu/ConfigManager.  For the fused detessdequant route that is a
  // single packed parent buffer named `output_tensor`; the six published names
  // are logical views in `out.facts` below.
  runtime.runtime_output_names = {"output_tensor"};
  runtime.physical_output_names = {"output_tensor"};
  runtime.primary_output_transport_kind = ProcessCvuOutputTransportKind::Packed;
  runtime.primary_output_semantic_kind = ProcessCvuOutputSemanticKind::Tensor;
  runtime.runtime_output_logical_index_list = {0};
  runtime.runtime_output_output_slot_list = {0};
  runtime.runtime_output_physical_index_list = {0};
  runtime.runtime_output_dtype_list = {!runtime.output_dtype.empty() ? runtime.output_dtype
                                                                     : std::string("FP16")};
  runtime.runtime_output_transport_kind_list = {ProcessCvuOutputTransportKind::Packed};
  runtime.runtime_output_semantic_kind_list = {ProcessCvuOutputSemanticKind::Tensor};
  runtime.runtime_output_logical_shapes.clear();
  if (!runtime.output_shapes.empty()) {
    runtime.runtime_output_logical_shapes.push_back(runtime.output_shapes.front());
  }
  runtime.runtime_output_logical_layout_list = {runtime_output_layout_token_local(runtime)};

  ProcessCvuCanonicalCompileInputs out;
  out.payload = build_processcvu_payload_from_runtime_config_internal(runtime);
  out.facts = build_processcvu_packed_route_facts("input_tensor", "output_tensor", entries,
                                                  runtime.primary_output_name,
                                                  runtime.published_output_names);
  apply_published_routed_input_bindings(&out, mla_published_outputs, &packed_input_sizes,
                                        runtime.graph_family);
  force_direct_materialization_for_inputs(&out);
  for (std::size_t i = 0; i < out.facts.outputs.size() && i < entries.size(); ++i) {
    if (i >= canonical_output_shapes.size() || canonical_output_shapes[i].empty()) {
      continue;
    }
    out.facts.outputs[i].shape = canonical_output_shapes[i];
  }
  if (processcvu_detess_layout_debug_enabled()) {
    for (std::size_t i = 0; i < out.facts.outputs.size(); ++i) {
      const auto& output = out.facts.outputs[i];
      std::fprintf(stderr,
                   "[detess-layout-debug] where=builder.fact index=%zu logical=%d slot=%d "
                   "layout=%s shape=%s repr=%d name=%s\n",
                   i, output.logical_index, output.output_slot, output.layout.c_str(),
                   join_i64_debug_processcvu_local(output.shape).c_str(),
                   static_cast<int>(output.representation), output.logical_name.c_str());
    }
  }
  if (direct_mla_boundary_match) {
    canonicalize_published_routed_inputs(&out, mla_published_outputs, runtime.graph_family);
  }
  force_direct_materialization_for_inputs(&out);
  return out;
}

} // namespace

ProcessCvuCanonicalCompileInputs
build_processcvu_compile_inputs_from_options(const ::simaai::neat::PreprocOptions& opt) {
  ProcessCvuCanonicalCompileInputs out;
  out.payload = build_preproc_payload_from_options_local(opt);
  out.facts = build_preproc_facts_from_payload(out.payload);
  return out;
}

CompiledProcessCvuContract
build_processcvu_compiled_contract_from_options(const ::simaai::neat::PreprocOptions& opt) {
  return build_processcvu_compiled_contract(build_processcvu_compile_inputs_from_options(opt));
}

std::string canonical_family_name_internal(std::string graph_family) {
  return canonical_family_name(std::move(graph_family));
}

std::string fused_processcvu_stage_identity_local(const std::string& stage_name,
                                                  const std::string& canonical_family) {
  if (canonical_family != "detessdequant" || stage_name.empty()) {
    return stage_name;
  }
  if (stage_name.rfind(canonical_family, 0) == 0) {
    return stage_name;
  }

  const auto rewrite_prefix = [&](const char* prefix) -> std::string {
    const std::string prefix_string = prefix ? std::string(prefix) : std::string();
    if (!prefix_string.empty() && stage_name.rfind(prefix_string, 0) == 0) {
      return canonical_family + stage_name.substr(prefix_string.size());
    }
    return {};
  };

  if (const std::string rewritten = rewrite_prefix("dequantize"); !rewritten.empty()) {
    return rewritten;
  }
  if (const std::string rewritten = rewrite_prefix("detessellate"); !rewritten.empty()) {
    return rewritten;
  }
  if (const std::string rewritten = rewrite_prefix("detess"); !rewritten.empty()) {
    return rewritten;
  }
  return canonical_family + "_" + stage_name;
}

ProcessCvuGraphFamily family_enum_from_name_internal(const std::string& graph_family) {
  return family_enum_from_name(graph_family);
}

void synthesize_runtime_output_arrays_from_payload_internal(ProcessCvuStagePayload* payload) {
  synthesize_runtime_output_arrays_from_payload(payload);
}

void canonicalize_preproc_single_handoff_payload_internal(ProcessCvuStagePayload* payload) {
  canonicalize_preproc_single_handoff_payload(payload);
}

ProcessCvuCanonicalFacts
build_preproc_facts_from_payload_internal(const ProcessCvuStagePayload& payload) {
  return build_preproc_facts_from_payload(payload);
}

ProcessCvuCanonicalFacts
build_single_io_processcvu_facts_from_payload_internal(const ProcessCvuStagePayload& payload) {
  return build_single_io_processcvu_facts_from_payload(payload);
}

ProcessCvuCanonicalFacts build_multi_io_processcvu_facts_from_payload_internal(
    const ProcessCvuStagePayload& payload, const std::vector<std::string>& runtime_input_names) {
  return build_multi_io_processcvu_facts_from_payload(payload, runtime_input_names);
}

CompiledProcessCvuContract
build_processcvu_compiled_contract_from_facts(const ProcessCvuStagePayload& payload,
                                              const ProcessCvuCanonicalFacts& facts) {
  CompiledProcessCvuContract compiled;
  ProcessCvuStagePayload compiled_payload = payload;
  const std::string canonical_family = canonical_family_name(compiled_payload.graph_family);
  if (compiled_payload.graph_family_enum == ProcessCvuGraphFamily::Unknown) {
    compiled_payload.graph_family_enum = family_enum_from_name(canonical_family);
  }
  if (canonical_family == "preproc") {
    canonicalize_preproc_single_handoff_payload(&compiled_payload);
    populate_preproc_payload_semantics(&compiled_payload);
  } else if (!compiled_payload.input_tensors.empty() || !compiled_payload.output_tensors.empty()) {
    sync_payload_tensor_desc_offsets_from_facts(&compiled_payload, facts);
  }
  compiled.payload = compiled_payload;
  compiled.preproc_single_output_handoff = compiled_payload.preproc_single_output_handoff;
  compiled.runtime_contract.plugin_kind = "processcvu";
  compiled.runtime_contract.consumer_keeps_distinct_physical_inputs =
      facts.preserve_physical_outputs;
  compiled.runtime_contract.required_preprocess_meta_fields.clear();

  const auto physical_input_names = derive_processcvu_physical_names_from_facts(
      facts.physical_input_names, facts.inputs, facts.outputs, false);
  const auto physical_output_names = derive_processcvu_physical_names_from_facts(
      facts.physical_output_names, facts.inputs, facts.outputs, true);

  compiled.runtime_contract.logical_inputs.reserve(facts.inputs.size());
  for (const auto& fact : facts.inputs) {
    compiled.runtime_contract.logical_inputs.push_back(logical_input_from_fact(fact));
  }
  for (std::size_t i = 0; i < compiled.runtime_contract.logical_inputs.size(); ++i) {
    specbuilders::finalize_logical_input_spec(&compiled.runtime_contract.logical_inputs[i], i,
                                              physical_input_names);
  }

  if (!facts.input_bindings.empty()) {
    compiled.runtime_contract.input_bindings.reserve(facts.input_bindings.size());
    for (const auto& fact : facts.input_bindings) {
      compiled.runtime_contract.input_bindings.push_back(input_binding_from_fact(fact));
    }
  } else {
    compiled.runtime_contract.input_bindings.reserve(
        compiled.runtime_contract.logical_inputs.size());
    for (const auto& logical : compiled.runtime_contract.logical_inputs) {
      const std::string cm_input_name =
          logical.physical_index >= 0 &&
                  static_cast<std::size_t>(logical.physical_index) < physical_input_names.size()
              ? physical_input_names[static_cast<std::size_t>(logical.physical_index)]
              : logical.backend_name;
      compiled.runtime_contract.input_bindings.push_back(
          specbuilders::build_input_binding_static_spec(
              0, logical.logical_index, cm_input_name,
              !logical.segment_name.empty() ? logical.segment_name : logical.backend_name,
              logical.logical_index, logical.logical_index, logical.physical_index,
              logical.size_bytes, logical.byte_offset, true));
    }
  }

  compiled.runtime_contract.logical_outputs.reserve(facts.outputs.size());
  for (const auto& fact : facts.outputs) {
    compiled.runtime_contract.logical_outputs.push_back(logical_output_from_fact(fact));
  }
  for (std::size_t i = 0; i < compiled.runtime_contract.logical_outputs.size(); ++i) {
    specbuilders::finalize_logical_output_spec(&compiled.runtime_contract.logical_outputs[i], i,
                                               physical_output_names);
  }
  if (processcvu_detess_layout_debug_enabled() &&
      canonical_family_name(payload.graph_family) == "detessdequant") {
    for (std::size_t i = 0; i < compiled.runtime_contract.logical_outputs.size(); ++i) {
      const auto& logical = compiled.runtime_contract.logical_outputs[i];
      std::fprintf(stderr,
                   "[detess-layout-debug] where=compiled.runtime_contract index=%zu logical=%d "
                   "slot=%d layout=%s shape=%s segment=%s name=%s\n",
                   i, logical.logical_index, logical.output_slot, logical.layout.c_str(),
                   join_i64_debug_processcvu_local(logical.shape).c_str(),
                   logical.segment_name.c_str(), logical.logical_name.c_str());
    }
  }

  compiled.runtime_contract.physical_inputs.reserve(physical_input_names.size());
  for (std::size_t i = 0; i < physical_input_names.size(); ++i) {
    std::uint64_t size_bytes = 0U;
    for (const auto& binding : compiled.runtime_contract.input_bindings) {
      if (binding.src_physical_output_index == static_cast<int>(i) &&
          binding.src_physical_size_bytes > 0U) {
        const std::uint64_t physical_offset =
            binding.src_physical_byte_offset > 0
                ? static_cast<std::uint64_t>(binding.src_physical_byte_offset)
                : 0U;
        size_bytes = std::max(size_bytes, physical_offset + binding.src_physical_size_bytes);
      }
    }
    if (size_bytes == 0U) {
      for (const auto& logical : compiled.runtime_contract.logical_inputs) {
        if (logical.physical_index == static_cast<int>(i)) {
          size_bytes = std::max(size_bytes, static_cast<std::uint64_t>(logical.byte_offset) +
                                                logical.size_bytes);
        }
      }
    }
    compiled.runtime_contract.physical_inputs.push_back(
        specbuilders::build_physical_buffer_static_spec(static_cast<int>(i), static_cast<int>(i),
                                                        size_bytes, DeviceKind::Evxx,
                                                        physical_input_names[i]));
  }

  compiled.runtime_contract.physical_outputs.reserve(physical_output_names.size());
  for (std::size_t i = 0; i < physical_output_names.size(); ++i) {
    std::uint64_t size_bytes = 0U;
    for (const auto& logical : compiled.runtime_contract.logical_outputs) {
      if (logical.physical_index == static_cast<int>(i)) {
        size_bytes = std::max(size_bytes,
                              static_cast<std::uint64_t>(logical.byte_offset) + logical.size_bytes);
      }
    }
    compiled.runtime_contract.physical_outputs.push_back(
        specbuilders::build_physical_buffer_static_spec(static_cast<int>(i), static_cast<int>(i),
                                                        size_bytes, DeviceKind::Evxx,
                                                        physical_output_names[i]));
  }

  {
    std::string normalize_err;
    if (!pipeline_internal::packedio::normalize_shared_parent_input_views(
            &compiled.runtime_contract, &normalize_err)) {
      throw std::invalid_argument(
          normalize_err.empty()
              ? "processcvu shared parent input view normalization failed"
              : "processcvu shared parent input view normalization failed: " + normalize_err);
    }
  }

  if (!facts.output_order.empty()) {
    compiled.runtime_contract.output_order.reserve(facts.output_order.size());
    for (const auto& fact : facts.output_order) {
      compiled.runtime_contract.output_order.push_back(output_route_from_fact(fact));
    }
  } else {
    const auto& wanted_names = !facts.published_output_names.empty() ? facts.published_output_names
                                                                     : physical_output_names;
    compiled.runtime_contract.output_order.reserve(wanted_names.size());
    for (std::size_t exposed_slot = 0; exposed_slot < wanted_names.size(); ++exposed_slot) {
      const auto it = std::find_if(compiled.runtime_contract.logical_outputs.begin(),
                                   compiled.runtime_contract.logical_outputs.end(),
                                   [&](const LogicalTensorStaticSpec& logical) {
                                     return logical_output_name_for_selection(logical) ==
                                            wanted_names[exposed_slot];
                                   });
      if (it == compiled.runtime_contract.logical_outputs.end()) {
        throw std::invalid_argument(
            "processcvu facts published output name does not resolve to a logical output");
      }
      compiled.runtime_contract.output_order.push_back(specbuilders::build_output_route_static_spec(
          static_cast<int>(exposed_slot), it->logical_index, it->tensor_index,
          logical_output_name_for_selection(*it), it->segment_name));
    }
  }

  std::vector<std::string> exposed_output_names = facts.published_output_names;
  if (exposed_output_names.empty()) {
    exposed_output_names.reserve(compiled.runtime_contract.output_order.size());
    for (const auto& route : compiled.runtime_contract.output_order) {
      const std::string name = output_name_from_route(route);
      if (!name.empty()) {
        exposed_output_names.push_back(name);
      }
    }
  }
  const std::string primary_output_name =
      !facts.primary_output_name.empty() ? facts.primary_output_name : payload.primary_output_name;
  compiled.exposed_view = build_processcvu_exposed_view_from_runtime(
      compiled.runtime_contract, exposed_output_names, primary_output_name);

  if (facts_use_packed_contract(facts)) {
    std::string packed_err;
    if (!pipeline_internal::packedio::validate_packed_contract(compiled.runtime_contract,
                                                               &packed_err)) {
      throw std::invalid_argument(
          packed_err.empty() ? "processcvu packed contract validation failed" : packed_err);
    }
  }
  if (processcvu_contract_compare_matches(payload)) {
    dump_processcvu_contract_compare_local(compiled, facts);
    if (processcvu_contract_compare_exit_enabled()) {
      std::fflush(stderr);
      std::fflush(stdout);
      std::_Exit(0);
    }
  }
  return compiled;
}

std::string canonical_processcvu_graph_family(const std::string& graph_family) {
  return canonical_family_name(graph_family);
}

namespace {} // namespace

std::uint64_t processcvu_size_bytes_from_shape_dtype(const std::vector<std::int64_t>& shape,
                                                     const std::string& dtype) {
  if (shape.empty()) {
    return 0U;
  }
  std::uint64_t elems = 1U;
  for (const auto dim : shape) {
    if (dim <= 0) {
      return 0U;
    }
    elems *= static_cast<std::uint64_t>(dim);
  }
  return elems * processcvu_dtype_size_bytes_from_token(dtype);
}

namespace {

void populate_processcvu_node_contract_common(const std::string& node_kind,
                                              const std::string& element_name,
                                              const std::string& logical_stage_id,
                                              const NodeContractDefinition& definition,
                                              CompiledProcessCvuContract compiled,
                                              CompiledNodeContract* out);

} // namespace

bool build_processcvu_node_contract(const std::string& node_kind, const std::string& element_name,
                                    const std::string& logical_stage_id,
                                    const NodeContractDefinition& definition,
                                    const CompiledProcessCvuContract& compiled,
                                    CompiledNodeContract* out, std::string* err) {
  if (!out) {
    if (err) {
      *err = node_kind + " contract compile: output is null";
    }
    return false;
  }
  populate_processcvu_node_contract_common(node_kind, element_name, logical_stage_id, definition,
                                           compiled, out);
  if (err) {
    err->clear();
  }
  return true;
}

namespace {

void populate_processcvu_node_contract_common(const std::string& node_kind,
                                              const std::string& element_name,
                                              const std::string& logical_stage_id,
                                              const NodeContractDefinition& definition,
                                              CompiledProcessCvuContract compiled,
                                              CompiledNodeContract* out) {
  const std::string canonical_family =
      canonical_family_name(!compiled.payload.graph_family.empty() ? compiled.payload.graph_family
                                                                   : compiled.payload.graph_name);
  const std::string primary_element_name =
      fused_processcvu_stage_identity_local(element_name, canonical_family);
  const std::string primary_logical_stage_id = fused_processcvu_stage_identity_local(
      logical_stage_id.empty() ? element_name : logical_stage_id, canonical_family);
  out->node_kind = node_kind;
  out->plugin_kind = compiled.runtime_contract.plugin_kind.empty()
                         ? "processcvu"
                         : compiled.runtime_contract.plugin_kind;
  out->element_name = primary_element_name;
  out->logical_stage_id = primary_logical_stage_id;
  out->definition = definition;
  out->processcvu = std::move(compiled);
  out->renderable = true;
}

} // namespace

} // namespace simaai::neat::pipeline_internal::sima::stagesemantics
