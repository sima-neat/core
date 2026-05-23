#include "pipeline/internal/sima/stagesemantics/ProcessCvuRuntimeConfigAdapterInternal.h"

#include "pipeline/internal/sima/TensorSemanticsUtil.h"
#include "pipeline/internal/sima/stagesemantics/ProcessCvuStageSemanticsInternal.h"

#include <algorithm>
#include <cctype>
#include <cstdint>
#include <cstring>
#include <stdexcept>

namespace simaai::neat::pipeline_internal::sima::stagesemantics {
namespace {

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

bool graph_family_implies_tessellate_local(const std::string& family) {
  const std::string canonical = lower_copy(family);
  return canonical == "tessellate" || canonical == "quanttess" ||
         canonical == "quantizetessellate" || canonical == "quantize_tessellate";
}

void require_positive_runtime_fact(int value, const char* field_name) {
  if (value > 0) {
    return;
  }
  throw std::invalid_argument(std::string("processcvu runtime config missing required field '") +
                              field_name + "'");
}

void require_non_empty_runtime_token(const std::string& value, const char* field_name) {
  if (!value.empty()) {
    return;
  }
  throw std::invalid_argument(std::string("processcvu runtime config missing required field '") +
                              field_name + "'");
}

bool processcvu_semantic_len_compatible(std::size_t len, std::size_t semantic_count) {
  return semantic_count == 0U || len == 0U || len == 1U || len == semantic_count;
}

bool processcvu_dtype_token_to_ev_runtime(const std::string& raw_dtype, uint32_t* out_dtype) {
  if (!out_dtype) {
    return false;
  }
  std::string token = upper_copy(raw_dtype);
  if (token == "FP32" || token == "FLOAT32" || token == "EVXX_FLOAT32") {
    *out_dtype = SIMA_EV_DTYPE_FP32;
    return true;
  }
  if (token == "BF16" || token == "BFLOAT16" || token == "EVXX_BFLOAT16" || token == "INT16" ||
      token == "EVXX_INT16") {
    *out_dtype = (token.find("FLOAT") != std::string::npos || token == "BF16")
                     ? SIMA_EV_DTYPE_BF16
                     : SIMA_EV_DTYPE_INT16;
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
  if (token == "INT8" || token == "EVXX_INT8" || token == "UINT8" || token == "EVXX_UINT8") {
    *out_dtype = SIMA_EV_DTYPE_INT8;
    return true;
  }
  return false;
}

bool processcvu_fill_dense_strides_runtime(const sima_ev_shape_desc& shape,
                                           const std::string& raw_layout, const uint32_t dtype,
                                           sima_ev_strided_layout_desc* out,
                                           std::string* error_detail) {
  return tensorsemantics::fill_dense_strides(shape, raw_layout, dtype, out, error_detail,
                                             "runtime_dense_stride_output_storage_missing",
                                             "runtime_dense_stride_dtype_invalid");
}

bool processcvu_build_dense_tensor_desc_runtime(const std::vector<int>& shape,
                                                const std::string& dtype_token,
                                                const std::string& layout_token,
                                                sima_ev_tensor_desc* out,
                                                std::string* error_detail) {
  if (!out) {
    if (error_detail)
      *error_detail = "runtime_dense_tensor_desc_output_storage_missing";
    return false;
  }
  const std::string normalized_layout = tensorsemantics::normalize_layout_token(layout_token);
  if (!layout_token.empty() && normalized_layout.empty()) {
    if (error_detail)
      *error_detail = "runtime_dense_tensor_desc_layout_invalid";
    return false;
  }
  if (normalized_layout.empty()) {
    return tensorsemantics::build_generic_dense_tensor_desc(
        shape, dtype_token, out, error_detail, "runtime_dense_tensor_desc_output_storage_missing",
        "runtime_shape_desc_rank_invalid", "runtime_shape_desc_dim_invalid",
        "runtime_dense_tensor_desc_dtype_invalid", "runtime_dense_stride_output_storage_missing");
  }
  std::memset(out, 0, sizeof(*out));
  if (!tensorsemantics::fill_shape_desc(shape, normalized_layout, &out->shape, error_detail,
                                        "runtime_shape_desc_output_storage_missing",
                                        "runtime_shape_desc_rank_invalid",
                                        "runtime_shape_desc_dim_invalid")) {
    return false;
  }
  std::uint32_t dtype = 0;
  if (!processcvu_dtype_token_to_ev_runtime(dtype_token, &dtype)) {
    if (error_detail)
      *error_detail = "runtime_dense_tensor_desc_dtype_invalid";
    return false;
  }
  out->dtype = dtype;
  out->layout_kind = SIMA_EV_LAYOUT_STRIDED;
  return processcvu_fill_dense_strides_runtime(out->shape, normalized_layout, out->dtype,
                                               &out->layout.strided, error_detail);
}

bool processcvu_build_tiled_tensor_desc_runtime(
    const std::vector<int>& shape, const std::vector<int>& tile_shape,
    const std::string& dtype_token, const std::string& layout_token, const guint32 tile_align_bytes,
    sima_ev_tensor_desc* out, std::string* error_detail) {
  if (!out) {
    if (error_detail)
      *error_detail = "runtime_tiled_tensor_desc_output_storage_missing";
    return false;
  }
  std::memset(out, 0, sizeof(*out));
  std::vector<int> normalized_tile_shape;
  if (!tensorsemantics::normalize_tile_shape(
          shape, tile_shape, &normalized_tile_shape, error_detail, "runtime_tile_shape_missing",
          "runtime_tile_shape_rank_prefix_invalid", "runtime_tile_shape_dim_invalid")) {
    return false;
  }
  const std::string normalized_layout = tensorsemantics::normalize_layout_token(layout_token);
  if (!layout_token.empty() && normalized_layout.empty()) {
    if (error_detail)
      *error_detail = "runtime_tiled_tensor_desc_layout_invalid";
    return false;
  }
  if (normalized_layout.empty()) {
    return tensorsemantics::build_generic_tiled_tensor_desc(
        shape, normalized_tile_shape, dtype_token, tile_align_bytes, out, error_detail,
        "runtime_tiled_tensor_desc_output_storage_missing", "runtime_shape_desc_rank_invalid",
        "runtime_shape_desc_dim_invalid", "runtime_tiled_tensor_desc_dtype_invalid",
        "runtime_tiled_tensor_desc_shape_rank_mismatch",
        "runtime_tiled_tensor_desc_tile_shape_invalid");
  }
  if (!tensorsemantics::fill_shape_desc(shape, normalized_layout, &out->shape, error_detail,
                                        "runtime_shape_desc_output_storage_missing",
                                        "runtime_shape_desc_rank_invalid",
                                        "runtime_shape_desc_dim_invalid")) {
    return false;
  }
  std::uint32_t dtype = 0;
  if (!processcvu_dtype_token_to_ev_runtime(dtype_token, &dtype)) {
    if (error_detail)
      *error_detail = "runtime_tiled_tensor_desc_dtype_invalid";
    return false;
  }
  out->dtype = dtype;
  out->layout_kind = SIMA_EV_LAYOUT_TILED;
  for (std::size_t i = 0; i < normalized_tile_shape.size(); ++i) {
    if (normalized_tile_shape[i] <= 0 || normalized_tile_shape[i] > shape[i]) {
      if (error_detail)
        *error_detail = "runtime_tiled_tensor_desc_tile_shape_invalid";
      return false;
    }
    out->layout.tiled.tile_sizes[i] = static_cast<int64_t>(normalized_tile_shape[i]);
  }
  out->layout.tiled.tile_align_bytes = tile_align_bytes;
  out->layout.tiled.flags = tensorsemantics::find_shape_axis(out->shape, SIMA_EV_AXIS_C) >= 0
                                ? SIMA_EV_TILED_FLAG_COMPACT_CHANNELS
                                : SIMA_EV_TILED_FLAG_NONE;
  return true;
}

bool processcvu_normalize_tile_shape_runtime(const std::vector<int>& shape,
                                             const std::vector<int>& raw_tile_shape,
                                             std::vector<int>* out, std::string* error_detail) {
  return tensorsemantics::normalize_tile_shape(
      shape, raw_tile_shape, out, error_detail, "runtime_tile_shape_missing",
      "runtime_tile_shape_rank_prefix_invalid", "runtime_tile_shape_dim_invalid");
}

std::vector<int> processcvu_shape_from_desc_runtime(const sima_ev_tensor_desc& desc) {
  std::vector<int> shape;
  const auto rank =
      std::min<std::uint32_t>(desc.shape.rank, static_cast<std::uint32_t>(SIMA_EV_MAX_RANK));
  shape.reserve(rank);
  for (std::uint32_t i = 0; i < rank; ++i) {
    shape.push_back(static_cast<int>(desc.shape.sizes[i]));
  }
  return shape;
}

std::vector<int> processcvu_tile_shape_from_desc_runtime(const sima_ev_tensor_desc& desc) {
  if (desc.layout_kind != SIMA_EV_LAYOUT_TILED) {
    return {};
  }
  std::vector<int> tile_shape;
  const auto rank =
      std::min<std::uint32_t>(desc.shape.rank, static_cast<std::uint32_t>(SIMA_EV_MAX_RANK));
  tile_shape.reserve(rank);
  for (std::uint32_t i = 0; i < rank; ++i) {
    tile_shape.push_back(static_cast<int>(desc.layout.tiled.tile_sizes[i]));
  }
  return tile_shape;
}

std::vector<std::vector<int>>
processcvu_shapes_from_descs_runtime(const std::vector<sima_ev_tensor_desc>& descs) {
  std::vector<std::vector<int>> shapes;
  shapes.reserve(descs.size());
  for (const auto& desc : descs) {
    shapes.push_back(processcvu_shape_from_desc_runtime(desc));
  }
  return shapes;
}

std::vector<std::vector<int>>
processcvu_tile_shapes_from_descs_runtime(const std::vector<sima_ev_tensor_desc>& descs) {
  std::vector<std::vector<int>> shapes;
  shapes.reserve(descs.size());
  for (const auto& desc : descs) {
    const auto tile_shape = processcvu_tile_shape_from_desc_runtime(desc);
    if (!tile_shape.empty()) {
      shapes.push_back(tile_shape);
    }
  }
  return shapes;
}

std::vector<std::vector<int>>
processcvu_resolved_input_shapes_runtime(const CompiledProcessCvuRuntimeConfig& config) {
  if (!config.input_shapes.empty()) {
    return config.input_shapes;
  }
  return processcvu_shapes_from_descs_runtime(config.input_tensors);
}

std::vector<std::vector<int>>
processcvu_resolved_output_shapes_runtime(const CompiledProcessCvuRuntimeConfig& config) {
  if (!config.output_shapes.empty()) {
    return config.output_shapes;
  }
  return processcvu_shapes_from_descs_runtime(config.output_tensors);
}

std::vector<std::vector<int>>
processcvu_resolved_slice_shapes_runtime(const CompiledProcessCvuRuntimeConfig& config) {
  if (!config.slice_shapes.empty()) {
    return config.slice_shapes;
  }
  auto slice_shapes = processcvu_tile_shapes_from_descs_runtime(config.output_tensors);
  if (!slice_shapes.empty()) {
    return slice_shapes;
  }
  return processcvu_tile_shapes_from_descs_runtime(config.input_tensors);
}

std::vector<std::vector<int>> processcvu_resolved_runtime_output_logical_shapes_runtime(
    const CompiledProcessCvuRuntimeConfig& config) {
  if (!config.runtime_output_logical_shapes.empty()) {
    return config.runtime_output_logical_shapes;
  }
  const auto output_shapes = processcvu_resolved_output_shapes_runtime(config);
  if (!output_shapes.empty()) {
    return output_shapes;
  }
  return processcvu_shapes_from_descs_runtime(config.output_tensors);
}

guint32 processcvu_resolve_tile_align_bytes_runtime(const int byte_align_flag) {
  if (byte_align_flag <= 0) {
    return 0U;
  }
  if (byte_align_flag == 1) {
    return 16U;
  }
  return static_cast<guint32>(byte_align_flag);
}

bool processcvu_build_runtime_tensor_descs(const CompiledProcessCvuRuntimeConfig& config,
                                           std::vector<sima_ev_tensor_desc>* input_tensors,
                                           std::vector<sima_ev_tensor_desc>* output_tensors,
                                           std::string* error_detail) {
  if (!input_tensors || !output_tensors) {
    if (error_detail)
      *error_detail = "runtime_tensor_desc_storage_missing";
    return false;
  }
  input_tensors->clear();
  output_tensors->clear();
  if (config.input_tensors.empty() || config.output_tensors.empty()) {
    if (error_detail)
      *error_detail = "runtime_tensor_desc_missing_explicit_descriptor_set";
    return false;
  }
  if (config.input_tensors.size() != config.output_tensors.size()) {
    if (error_detail)
      *error_detail = "runtime_tensor_desc_explicit_tensor_count_mismatch";
    return false;
  }
  *input_tensors = config.input_tensors;
  *output_tensors = config.output_tensors;
  return true;
}

std::size_t processcvu_semantic_input_count(const CompiledProcessCvuRuntimeConfig& config) {
  const auto resolved_input_shapes = processcvu_resolved_input_shapes_runtime(config);
  const auto resolved_slice_shapes = processcvu_resolved_slice_shapes_runtime(config);
  const std::size_t derived =
      std::max({resolved_input_shapes.size(), resolved_slice_shapes.size(),
                config.input_tensors.size(), config.q_scale_list.size(), config.q_zp_list.size(),
                config.dq_scale_list.size(), config.dq_zp_list.size()});
  if (derived > 0U) {
    return derived;
  }
  if (!config.runtime_input_names.empty()) {
    return config.runtime_input_names.size();
  }
  return 1U;
}

std::vector<std::string>
processcvu_physical_input_names(const CompiledProcessCvuRuntimeConfig& config) {
  if (!config.physical_input_names.empty()) {
    return config.physical_input_names;
  }
  if (!config.runtime_input_names.empty()) {
    return config.runtime_input_names;
  }
  if (!config.default_input_name.empty()) {
    return {config.default_input_name};
  }
  return {};
}

std::vector<std::string>
processcvu_physical_output_names(const CompiledProcessCvuRuntimeConfig& config) {
  if (!config.physical_output_names.empty()) {
    return config.physical_output_names;
  }
  return config.runtime_output_names;
}

std::size_t processcvu_semantic_output_count(const CompiledProcessCvuRuntimeConfig& config) {
  if (!config.published_output_names.empty()) {
    return config.published_output_names.size();
  }
  if (!config.runtime_output_names.empty()) {
    return config.runtime_output_names.size();
  }
  const auto resolved_output_shapes = processcvu_resolved_output_shapes_runtime(config);
  if (!resolved_output_shapes.empty()) {
    return resolved_output_shapes.size();
  }
  if (!config.output_tensors.empty()) {
    return config.output_tensors.size();
  }
  return processcvu_physical_output_names(config).size();
}

void validate_runtime_output_config_strict(const CompiledProcessCvuRuntimeConfig& config) {
  auto require_non_empty_unique_names = [](const std::vector<std::string>& names,
                                           const char* field_name) {
    if (names.empty()) {
      throw std::invalid_argument(std::string("processcvu runtime config missing explicit ") +
                                  field_name);
    }
    for (std::size_t i = 0; i < names.size(); ++i) {
      if (names[i].empty()) {
        throw std::invalid_argument(std::string("processcvu runtime config field '") + field_name +
                                    "' contains an empty name");
      }
      if (std::find(names.begin(), names.begin() + static_cast<std::ptrdiff_t>(i), names[i]) !=
          names.begin() + static_cast<std::ptrdiff_t>(i)) {
        throw std::invalid_argument(std::string("processcvu runtime config field '") + field_name +
                                    "' contains duplicate names");
      }
    }
  };

  const auto physical_input_names = processcvu_physical_input_names(config);
  const auto physical_output_names = processcvu_physical_output_names(config);
  if (config.default_input_name.empty()) {
    throw std::invalid_argument("processcvu runtime config missing explicit default_input_name");
  }
  if (config.primary_output_name.empty()) {
    throw std::invalid_argument("processcvu runtime config missing explicit primary_output_name");
  }
  if (config.runtime_output_names.empty()) {
    throw std::invalid_argument("processcvu runtime config missing explicit runtime_output_names");
  }
  if (config.published_output_names.empty()) {
    throw std::invalid_argument(
        "processcvu runtime config missing explicit published_output_names");
  }

  require_non_empty_unique_names(physical_input_names, "physical_input_names");
  require_non_empty_unique_names(physical_output_names, "physical_output_names");
  require_non_empty_unique_names(config.published_output_names, "published_output_names");

  const auto resolved_input_shapes = processcvu_resolved_input_shapes_runtime(config);
  const auto resolved_slice_shapes = processcvu_resolved_slice_shapes_runtime(config);
  const auto resolved_output_shapes = processcvu_resolved_output_shapes_runtime(config);
  const auto resolved_runtime_output_logical_shapes =
      processcvu_resolved_runtime_output_logical_shapes_runtime(config);
  const std::size_t semantic_input_count = processcvu_semantic_input_count(config);
  const auto require_semantic_input_count = [&](const char* field, std::size_t count) {
    if (!processcvu_semantic_len_compatible(count, semantic_input_count)) {
      throw std::invalid_argument(std::string("processcvu runtime config field '") + field +
                                  "' must be scalar or match semantic input count");
    }
  };
  require_semantic_input_count("input_shapes", resolved_input_shapes.size());
  require_semantic_input_count("slice_shapes", resolved_slice_shapes.size());
  require_semantic_input_count("q_scale_list", config.q_scale_list.size());
  require_semantic_input_count("q_zp_list", config.q_zp_list.size());
  require_semantic_input_count("dq_scale_list", config.dq_scale_list.size());
  require_semantic_input_count("dq_zp_list", config.dq_zp_list.size());

  // Runtime outputs are the physical buffers exposed to processcvu/ConfigManager
  // (`default_output_names` in the ABI).  Published outputs are the logical
  // views that NEAT exposes downstream.  Packed fused graphs such as YOLOv8
  // detessdequant intentionally have one runtime/physical output
  // (`output_tensor`) and several published logical outputs that are byte-range
  // views inside that physical buffer.  Keep the runtime metadata arrays keyed
  // to runtime_output_names, not published_output_names.
  const std::size_t runtime_output_count = config.runtime_output_names.size();
  const std::size_t semantic_output_count = processcvu_semantic_output_count(config);
  auto require_runtime_count = [&](const char* field, std::size_t count) {
    if (count != runtime_output_count) {
      throw std::invalid_argument(std::string("processcvu runtime config field '") + field +
                                  "' must have one entry per runtime output");
    }
  };
  auto require_semantic_output_count = [&](const char* field, std::size_t count) {
    if (!processcvu_semantic_len_compatible(count, semantic_output_count)) {
      throw std::invalid_argument(std::string("processcvu runtime config field '") + field +
                                  "' must be scalar or match semantic output count");
    }
  };

  const bool has_any_runtime_output_arrays = !config.runtime_output_logical_index_list.empty() ||
                                             !config.runtime_output_output_slot_list.empty() ||
                                             !config.runtime_output_physical_index_list.empty() ||
                                             !config.runtime_output_dtype_list.empty() ||
                                             !config.runtime_output_transport_kind_list.empty() ||
                                             !config.runtime_output_semantic_kind_list.empty() ||
                                             !resolved_runtime_output_logical_shapes.empty();
  if (!resolved_output_shapes.empty()) {
    require_semantic_output_count("output_shapes", resolved_output_shapes.size());
  }
  if (has_any_runtime_output_arrays) {
    if (!config.runtime_output_logical_index_list.empty()) {
      require_runtime_count("runtime_output_logical_index_list",
                            config.runtime_output_logical_index_list.size());
    }
    if (!config.runtime_output_output_slot_list.empty()) {
      require_runtime_count("runtime_output_output_slot_list",
                            config.runtime_output_output_slot_list.size());
    }
    if (!config.runtime_output_physical_index_list.empty()) {
      require_runtime_count("runtime_output_physical_index_list",
                            config.runtime_output_physical_index_list.size());
    }
    if (!config.runtime_output_dtype_list.empty()) {
      require_runtime_count("runtime_output_dtype_list", config.runtime_output_dtype_list.size());
    }
    if (!config.runtime_output_transport_kind_list.empty()) {
      require_runtime_count("runtime_output_transport_kind_list",
                            config.runtime_output_transport_kind_list.size());
    }
    if (!config.runtime_output_semantic_kind_list.empty()) {
      require_runtime_count("runtime_output_semantic_kind_list",
                            config.runtime_output_semantic_kind_list.size());
    }
    if (!resolved_runtime_output_logical_shapes.empty()) {
      require_runtime_count("runtime_output_logical_shapes",
                            resolved_runtime_output_logical_shapes.size());
    }
  }

  const bool primary_matches_published_name =
      std::find(config.published_output_names.begin(), config.published_output_names.end(),
                config.primary_output_name) != config.published_output_names.end();
  if (!primary_matches_published_name) {
    throw std::invalid_argument(
        "processcvu runtime config primary_output_name must name a published output");
  }

  require_non_empty_runtime_token(config.input_dtype, "input_dtype");
  require_non_empty_runtime_token(config.output_dtype, "output_dtype");
  if (config.input_tensors.empty() || config.output_tensors.empty()) {
    throw std::invalid_argument(
        "processcvu runtime config requires explicit typed input_tensors/output_tensors");
  }
  if (config.input_tensors.size() != config.output_tensors.size()) {
    throw std::invalid_argument(
        "processcvu runtime config explicit input_tensors/output_tensors count mismatch");
  }
  if (has_any_runtime_output_arrays) {
    for (std::size_t i = 0; i < runtime_output_count; ++i) {
      if (i < resolved_output_shapes.size() && resolved_output_shapes[i].empty()) {
        throw std::invalid_argument(
            "processcvu runtime config field 'output_shapes' contains an empty shape");
      }
      if (!config.runtime_output_dtype_list.empty()) {
        require_non_empty_runtime_token(config.runtime_output_dtype_list[i],
                                        "runtime_output_dtype_list");
      }
    }
  }
}

ProcessCvuStagePayload
build_processcvu_payload_from_runtime_config_common(const CompiledProcessCvuRuntimeConfig& config,
                                                    bool validate) {
  if (validate) {
    validate_runtime_output_config_strict(config);
  }
  const std::string family = canonical_family_name_internal(config.graph_family);

  ProcessCvuStagePayload payload;
  payload.canonical_contract = true;
  payload.graph_family = family;
  payload.graph_family_enum = family_enum_from_name_internal(payload.graph_family);
  payload.graph_name =
      !config.graph_name.empty()
          ? config.graph_name
          : (!payload.graph_family.empty() ? payload.graph_family : std::string("processcvu"));
  payload.default_input_name = config.default_input_name;
  payload.default_output_names = config.runtime_output_names;
  payload.primary_output_name = config.primary_output_name;
  payload.primary_output_transport_kind = config.primary_output_transport_kind;
  payload.primary_output_semantic_kind = config.primary_output_semantic_kind;

  payload.graph_id = config.graph_id;
  payload.scaled_width = config.scaled_width;
  payload.scaled_height = config.scaled_height;
  payload.input_stride = config.input_stride;
  payload.output_stride = config.output_stride;
  payload.input_offset = config.input_offset;
  payload.batch_size = config.batch_size > 0 ? config.batch_size : 1;
  payload.round_off = config.round_off;
  payload.byte_align = config.byte_align;
  payload.opt_flags = config.opt_flags;
  payload.pad_value = config.pad_value;
  payload.aspect_ratio = config.aspect_ratio;
  payload.normalize = config.normalize;
  payload.tessellate = config.tessellate;
  payload.preproc_single_output_handoff = config.single_output_handoff;

  payload.has_q_scale = config.has_q_scale;
  payload.q_scale = config.q_scale;
  payload.has_q_zp = config.has_q_zp;
  payload.q_zp = static_cast<int>(config.q_zp);
  payload.q_scale_list = config.q_scale_list;
  payload.q_zp_list.assign(config.q_zp_list.begin(), config.q_zp_list.end());
  payload.dq_scale_list = config.dq_scale_list;
  payload.dq_zp_list.assign(config.dq_zp_list.begin(), config.dq_zp_list.end());
  std::string tensor_desc_error;
  if (!processcvu_build_runtime_tensor_descs(config, &payload.input_tensors,
                                             &payload.output_tensors, &tensor_desc_error)) {
    throw std::invalid_argument(
        std::string("processcvu runtime config could not build tensor descriptors") +
        (tensor_desc_error.empty() ? std::string() : std::string(": ") + tensor_desc_error));
  }
  payload.input_shapes = processcvu_resolved_input_shapes_runtime(config);
  payload.num_in_tensor = !payload.input_tensors.empty()
                              ? static_cast<int>(payload.input_tensors.size())
                              : static_cast<int>(processcvu_semantic_input_count(config));
  payload.slice_shapes = processcvu_resolved_slice_shapes_runtime(config);
  payload.output_shapes = processcvu_resolved_output_shapes_runtime(config);
  payload.runtime_output_logical_index_list = config.runtime_output_logical_index_list;
  payload.runtime_output_output_slot_list = config.runtime_output_output_slot_list;
  payload.runtime_output_physical_index_list = config.runtime_output_physical_index_list;
  payload.runtime_output_dtype_list = config.runtime_output_dtype_list;
  payload.runtime_output_transport_kind_list = config.runtime_output_transport_kind_list;
  payload.runtime_output_semantic_kind_list = config.runtime_output_semantic_kind_list;
  payload.runtime_output_logical_shapes =
      processcvu_resolved_runtime_output_logical_shapes_runtime(config);
  payload.runtime_output_logical_layout_list = config.runtime_output_logical_layout_list;
  payload.channel_mean.assign(config.channel_mean.begin(), config.channel_mean.end());
  payload.channel_stddev.assign(config.channel_stddev.begin(), config.channel_stddev.end());

  payload.input_img_type = config.input_img_type;
  payload.output_img_type = config.output_img_type;
  payload.input_dtype = config.input_dtype;
  payload.output_dtype = config.output_dtype;
  payload.out_dtype = config.out_dtype;
  payload.scaling_type = config.scaling_type;
  payload.padding_type = config.padding_type;
  synthesize_runtime_output_arrays_from_payload_internal(&payload);
  canonicalize_preproc_single_handoff_payload_internal(&payload);
  return payload;
}

ProcessCvuStagePayload
build_processcvu_payload_from_runtime_config_strict(const CompiledProcessCvuRuntimeConfig& config) {
  return build_processcvu_payload_from_runtime_config_common(config, true);
}

ProcessCvuStagePayload build_processcvu_payload_from_runtime_config_unchecked(
    const CompiledProcessCvuRuntimeConfig& config) {
  return build_processcvu_payload_from_runtime_config_common(config, false);
}

ProcessCvuCanonicalFacts
build_processcvu_facts_from_runtime_config_impl(const CompiledProcessCvuRuntimeConfig& config) {
  const auto payload = build_processcvu_payload_from_runtime_config_strict(config);
  const std::string family = canonical_family_name_internal(config.graph_family);
  const std::size_t semantic_input_count = processcvu_semantic_input_count(config);
  const std::size_t semantic_output_count = processcvu_semantic_output_count(config);
  const std::size_t physical_input_count = processcvu_physical_input_names(config).size();
  const std::size_t physical_output_count = processcvu_physical_output_names(config).size();
  if (family == "preproc") {
    return build_preproc_facts_from_payload_internal(payload);
  }
  if (semantic_input_count > physical_input_count ||
      semantic_output_count > physical_output_count) {
    throw std::invalid_argument("processcvu runtime config requires explicit packed-route facts "
                                "for semantic multi-io over packed transport");
  }
  if (semantic_input_count > 1U || semantic_output_count > 1U) {
    return build_multi_io_processcvu_facts_from_payload_internal(payload,
                                                                 config.runtime_input_names);
  }
  return build_single_io_processcvu_facts_from_payload_internal(payload);
}

ProcessCvuStagePayload
build_processcvu_payload_from_runtime_config_impl(const CompiledProcessCvuRuntimeConfig& config) {
  auto payload = build_processcvu_payload_from_runtime_config_strict(config);
  payload.canonical_contract = true;
  return payload;
}

} // namespace

ProcessCvuStagePayload build_processcvu_payload_from_runtime_config_internal(
    const CompiledProcessCvuRuntimeConfig& config) {
  return build_processcvu_payload_from_runtime_config_impl(config);
}

ProcessCvuStagePayload build_processcvu_payload_from_runtime_config_unchecked_internal(
    const CompiledProcessCvuRuntimeConfig& config) {
  auto payload = build_processcvu_payload_from_runtime_config_unchecked(config);
  payload.canonical_contract = true;
  return payload;
}

ProcessCvuCanonicalFacts
build_processcvu_facts_from_runtime_config_internal(const CompiledProcessCvuRuntimeConfig& config) {
  return build_processcvu_facts_from_runtime_config_impl(config);
}

ProcessCvuCanonicalCompileInputs
build_processcvu_compile_inputs_from_runtime_config(const CompiledProcessCvuRuntimeConfig& config) {
  ProcessCvuCanonicalCompileInputs out;
  out.payload = build_processcvu_payload_from_runtime_config_internal(config);
  out.facts = build_processcvu_facts_from_runtime_config_internal(config);
  return out;
}

CompiledProcessCvuContract build_processcvu_compiled_contract_from_runtime_config(
    const CompiledProcessCvuRuntimeConfig& config) {
  return build_processcvu_compiled_contract(
      build_processcvu_compile_inputs_from_runtime_config(config));
}

} // namespace simaai::neat::pipeline_internal::sima::stagesemantics
