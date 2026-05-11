#pragma once
#ifndef SIMA_NEAT_INTERNAL
#error "Internal header. Not part of the public API."
#endif

#include "builder/NodeGroup.h"
#include "pipeline/internal/RenderedStageQueryTypes.h"
#include "pipeline/internal/contract/ContractApply.h"
#include "pipeline/internal/contract/ContractCompiler.h"
#include "pipeline/internal/sima/CompiledProcessCvuContractQuery.h"
#include "pipeline/internal/sima/ContractRender.h"
#include "pipeline/internal/sima/TensorSemanticsUtil.h"

#include <cctype>
#include <functional>
#include <initializer_list>
#include <memory>
#include <optional>
#include <string>
#include <vector>

namespace simaai::neat::pipeline_internal::rendered_stage_query {

inline std::optional<sima::SimaPluginStaticManifest> rendered_manifest_from_group(
    const NodeGroup& group,
    const char* pipeline_label = "RenderedStageQuery") {
  ContractCompileInput input;
  input.pipeline_label = pipeline_label ? pipeline_label : "RenderedStageQuery";
  sima::ManifestBuildDiagnostics diagnostics;
  const CompiledPipelineContracts compiled = compile_node_contracts(group.nodes(), input, &diagnostics);
  if (!diagnostics.errors.empty()) {
    return std::nullopt;
  }

  std::vector<std::shared_ptr<Node>> applied_nodes = group.nodes();
  std::string apply_error;
  apply_compiled_contracts(&applied_nodes, compiled, &apply_error);
  if (!apply_error.empty()) {
    return std::nullopt;
  }
  return render_manifest_from_compiled_contracts(compiled, input, &diagnostics);
}

inline const sima::StageStaticSpec* find_stage_in_manifest(
    const sima::SimaPluginStaticManifest& manifest,
    const std::function<bool(const sima::StageStaticSpec&)>& predicate) {
  for (const auto& stage : manifest.stages) {
    if (predicate(stage)) {
      return &stage;
    }
  }
  return nullptr;
}

inline const sima::StageStaticSpec* find_preproc_stage(
    const sima::SimaPluginStaticManifest& manifest) {
  return find_stage_in_manifest(manifest, [](const sima::StageStaticSpec& stage) {
    return stage.payload_kind == sima::StagePayloadKind::ProcessCvu &&
           stage.processcvu.graph_family == "preproc";
  });
}

inline const sima::StageStaticSpec* find_mla_stage(
    const sima::SimaPluginStaticManifest& manifest) {
  return find_stage_in_manifest(manifest, [](const sima::StageStaticSpec& stage) {
    return stage.payload_kind == sima::StagePayloadKind::ProcessMla;
  });
}

inline const sima::StageStaticSpec* find_processcvu_stage(
    const sima::SimaPluginStaticManifest& manifest,
    const std::string& family) {
  return find_stage_in_manifest(manifest, [&](const sima::StageStaticSpec& stage) {
    return stage.payload_kind == sima::StagePayloadKind::ProcessCvu &&
           stage.processcvu.graph_family == family;
  });
}

inline std::string first_non_empty(std::initializer_list<std::string> values) {
  for (const auto& value : values) {
    if (!value.empty()) {
      return value;
    }
  }
  return {};
}

// Boundary/query helper only. This is a compatibility projection from rendered
// contract format text, not a tensor semantic authority.
inline TensorLayout layout_projection_from_contract_format(const std::string& fmt) {
  const std::string layout = sima::tensorsemantics::normalize_layout_token(fmt);
  if (layout == "CHW") {
    return TensorLayout::CHW;
  }
  if (layout == "HWC") {
    return TensorLayout::HWC;
  }
  if (layout == "HW") {
    return TensorLayout::HW;
  }
  return TensorLayout::Unknown;
}

inline TensorDType dtype_from_contract_token(const std::string& raw) {
  std::string up;
  up.reserve(raw.size());
  for (char c : raw) {
    up.push_back(static_cast<char>(std::toupper(static_cast<unsigned char>(c))));
  }
  if (up.find("BF16") != std::string::npos || up.find("BFLOAT16") != std::string::npos) {
    return TensorDType::BFloat16;
  }
  if (up.find("FLOAT32") != std::string::npos || up.find("FP32") != std::string::npos) {
    return TensorDType::Float32;
  }
  if (up.find("UINT8") != std::string::npos || up == "U8") {
    return TensorDType::UInt8;
  }
  if (up.find("INT8") != std::string::npos) {
    return TensorDType::Int8;
  }
  if (up.find("UINT16") != std::string::npos) {
    return TensorDType::UInt16;
  }
  if (up.find("INT16") != std::string::npos) {
    return TensorDType::Int16;
  }
  if (up.find("INT32") != std::string::npos) {
    return TensorDType::Int32;
  }
  return TensorDType::Float32;
}

// Boundary/query helper only. This projects width/height/depth for callers
// that still need legacy dims views; it must not be treated as semantic truth.
inline stages::TensorDims tensor_dims_projection_from_contract_shape(std::vector<int64_t> shape,
                                                                     TensorLayout layout) {
  if (shape.size() >= 4U && shape.front() == 1) {
    shape.erase(shape.begin());
  }

  stages::TensorDims dims;
  if (shape.size() >= 3U && layout != TensorLayout::CHW && layout != TensorLayout::HWC) {
    return dims;
  }
  const bool chw_like = layout == TensorLayout::CHW;
  if (shape.size() >= 3U) {
    const int64_t a = shape[shape.size() - 3U];
    const int64_t b = shape[shape.size() - 2U];
    const int64_t c = shape[shape.size() - 1U];
    if (chw_like) {
      dims.depth = static_cast<int>(a);
      dims.height = static_cast<int>(b);
      dims.width = static_cast<int>(c);
    } else {
      dims.height = static_cast<int>(a);
      dims.width = static_cast<int>(b);
      dims.depth = static_cast<int>(c);
    }
  } else if (shape.size() == 2U) {
    dims.height = static_cast<int>(shape[0]);
    dims.width = static_cast<int>(shape[1]);
    dims.depth = 1;
  } else if (shape.size() == 1U) {
    dims.width = static_cast<int>(shape[0]);
    dims.height = 1;
    dims.depth = 1;
  }
  return dims;
}

inline std::vector<int64_t> maybe_strip_batch_axis(std::vector<int64_t> shape,
                                                   const std::string& format,
                                                   bool include_batch_axis) {
  if (include_batch_axis || shape.size() <= 1U) {
    return shape;
  }
  std::string up;
  up.reserve(format.size());
  for (char c : format) {
    up.push_back(static_cast<char>(std::toupper(static_cast<unsigned char>(c))));
  }
  const bool explicit_batch_axis = !up.empty() && up.front() == 'N';
  if (explicit_batch_axis && !shape.empty() && shape.front() == 1) {
    shape.erase(shape.begin());
  }
  return shape;
}

inline std::vector<int64_t> maybe_strip_batch_axis_from_aligned_values(
    std::vector<int64_t> values,
    const std::vector<int64_t>& original_shape,
    const std::string& format,
    bool include_batch_axis) {
  if (include_batch_axis || values.size() <= 1U || values.size() != original_shape.size()) {
    return values;
  }
  const auto stripped_shape = maybe_strip_batch_axis(original_shape, format, include_batch_axis);
  if (stripped_shape.size() + 1U == values.size()) {
    values.erase(values.begin());
  }
  return values;
}

inline const sima::LogicalTensorStaticSpec* find_logical_output(
    const sima::StageStaticSpec& stage,
    int logical_output_index,
    int output_slot) {
  for (const auto& logical : stage.logical_outputs) {
    if (logical_output_index >= 0 && logical.logical_index == logical_output_index) {
      return &logical;
    }
    if (output_slot >= 0 && logical.output_slot == output_slot) {
      return &logical;
    }
  }
  return nullptr;
}

inline bool is_packed_blob_logical_output(const sima::LogicalTensorStaticSpec& logical) {
  return logical.shape.size() == 1U && logical.size_bytes > 0U &&
         layout_projection_from_contract_format(logical.layout) == TensorLayout::HW;
}

inline stages::PreprocOutputTransportKind preproc_transport_kind_from_manifest(
    sima::ProcessCvuOutputTransportKind kind) {
  switch (kind) {
  case sima::ProcessCvuOutputTransportKind::Dense:
    return stages::PreprocOutputTransportKind::Dense;
  case sima::ProcessCvuOutputTransportKind::Packed:
    return stages::PreprocOutputTransportKind::Packed;
  case sima::ProcessCvuOutputTransportKind::Unknown:
  default:
    return stages::PreprocOutputTransportKind::Unknown;
  }
}

inline stages::PreprocOutputSemanticKind preproc_semantic_kind_from_manifest(
    sima::ProcessCvuOutputSemanticKind kind) {
  switch (kind) {
  case sima::ProcessCvuOutputSemanticKind::Image:
    return stages::PreprocOutputSemanticKind::Image;
  case sima::ProcessCvuOutputSemanticKind::TessellatedImage:
    return stages::PreprocOutputSemanticKind::TessellatedImage;
  case sima::ProcessCvuOutputSemanticKind::QuantizedTensor:
    return stages::PreprocOutputSemanticKind::QuantizedTensor;
  case sima::ProcessCvuOutputSemanticKind::QuantTessTensor:
    return stages::PreprocOutputSemanticKind::QuantTessTensor;
  case sima::ProcessCvuOutputSemanticKind::Tensor:
    return stages::PreprocOutputSemanticKind::Tensor;
  case sima::ProcessCvuOutputSemanticKind::Unknown:
  default:
    return stages::PreprocOutputSemanticKind::Unknown;
  }
}

inline stages::PreprocOutputInfo preproc_output_info(const NodeGroup& group) {
  const auto manifest = rendered_manifest_from_group(group);
  if (!manifest.has_value()) {
    return {};
  }
  const auto* stage = find_preproc_stage(*manifest);
  if (!stage) {
    return {};
  }

  stages::PreprocOutputInfo info;
  info.transport_kind =
      preproc_transport_kind_from_manifest(stage->processcvu.primary_output_transport_kind);
  info.semantic_kind =
      preproc_semantic_kind_from_manifest(stage->processcvu.primary_output_semantic_kind);
  info.output_dtype =
      first_non_empty({stage->processcvu.output_dtype, stage->processcvu.out_dtype});
  info.primary_output_name = stage->processcvu.primary_output_name;
  info.output_memory_order = stage->processcvu.default_output_names;

  if (info.primary_output_name.empty()) {
    return {};
  }

  const auto* primary_logical = [&]() -> const sima::LogicalTensorStaticSpec* {
    for (const auto& route : stage->output_order) {
      if (sima::output_name_from_route(route) != info.primary_output_name || route.output_slot < 0) {
        continue;
      }
      info.primary_route_slot = route.output_slot;
      return find_logical_output(*stage, route.logical_output_index, route.output_slot);
    }
    return nullptr;
  }();

  if (!primary_logical) {
    return {};
  }

  std::size_t primary_output_index = std::string::npos;
  for (std::size_t i = 0; i < stage->processcvu.default_output_names.size(); ++i) {
    if (stage->processcvu.default_output_names[i] == info.primary_output_name) {
      primary_output_index = i;
      break;
    }
  }

  if (primary_output_index != std::string::npos) {
    if (primary_output_index < stage->processcvu.runtime_output_transport_kind_list.size()) {
      info.transport_kind = preproc_transport_kind_from_manifest(
          stage->processcvu.runtime_output_transport_kind_list[primary_output_index]);
    }
    if (primary_output_index < stage->processcvu.runtime_output_semantic_kind_list.size()) {
      info.semantic_kind = preproc_semantic_kind_from_manifest(
          stage->processcvu.runtime_output_semantic_kind_list[primary_output_index]);
    }
    if (primary_output_index < stage->processcvu.runtime_output_logical_shapes.size()) {
      const auto& shape = stage->processcvu.runtime_output_logical_shapes[primary_output_index];
      info.logical_dims.height = shape.size() >= 1 ? shape[0] : 0;
      info.logical_dims.width = shape.size() >= 2 ? shape[1] : 0;
      info.logical_dims.depth = shape.size() >= 3 ? shape[2] : 0;
    }
    if (primary_output_index < stage->processcvu.runtime_output_logical_layout_list.size()) {
      info.logical_layout = layout_projection_from_contract_format(
          stage->processcvu.runtime_output_logical_layout_list[primary_output_index]);
    }
  }

  if (info.logical_dims.width <= 0 || info.logical_dims.height <= 0 || info.logical_dims.depth <= 0) {
    info.logical_dims = tensor_dims_projection_from_contract_shape(
        primary_logical->shape,
        layout_projection_from_contract_format(primary_logical->layout));
  }
  if (info.logical_layout == TensorLayout::Unknown) {
    info.logical_layout = layout_projection_from_contract_format(primary_logical->layout);
  }
  if (info.transport_kind == stages::PreprocOutputTransportKind::Unknown) {
    info.transport_kind = is_packed_blob_logical_output(*primary_logical)
                              ? stages::PreprocOutputTransportKind::Packed
                              : stages::PreprocOutputTransportKind::Dense;
  }
  if (info.semantic_kind == stages::PreprocOutputSemanticKind::Unknown) {
    info.semantic_kind =
        (info.transport_kind == stages::PreprocOutputTransportKind::Packed &&
         info.primary_output_name == "output_tessellated_image")
            ? stages::PreprocOutputSemanticKind::TessellatedImage
            : stages::PreprocOutputSemanticKind::Image;
  }
  if (info.output_dtype.empty()) {
    info.output_dtype = primary_logical->dtype;
  }
  return info;
}

inline std::string primary_input_buffer_name(const NodeGroup& group) {
  const auto manifest = rendered_manifest_from_group(group);
  if (!manifest.has_value()) {
    return {};
  }
  for (const auto& stage : manifest->stages) {
    if (stage.input_bindings.empty()) {
      continue;
    }
    for (const auto& binding : stage.input_bindings) {
      if (!binding.source_segment_name.empty() && binding.source_segment_name != "parent") {
        return binding.source_segment_name;
      }
    }
    for (const auto& binding : stage.input_bindings) {
      if (!binding.cm_input_name.empty()) {
        return binding.cm_input_name;
      }
      if (!binding.source_segment_name.empty()) {
        return binding.source_segment_name;
      }
    }
  }
  return {};
}

inline std::vector<stages::MlaOutputTensorInfo> mla_output_tensors(const NodeGroup& group) {
  std::vector<stages::MlaOutputTensorInfo> out;
  const auto manifest = rendered_manifest_from_group(group);
  if (!manifest.has_value()) {
    return out;
  }
  const auto* stage = find_mla_stage(*manifest);
  if (!stage || stage->logical_outputs.empty()) {
    return out;
  }

  out.reserve(stage->logical_outputs.size());
  for (std::size_t i = 0; i < stage->logical_outputs.size(); ++i) {
    const auto& logical = stage->logical_outputs[i];
    stages::MlaOutputTensorInfo info;
    info.shape = logical.shape;
    info.stride_bytes = logical.stride_bytes;
    info.byte_offset = logical.byte_offset;
    info.memory_index = (logical.physical_index >= 0) ? logical.physical_index : 0;
    info.data_type = logical.dtype;
    info.output_format = logical.layout;
    info.layout = layout_projection_from_contract_format(info.output_format);
    info.name = first_non_empty(
        {logical.logical_name, logical.backend_name, logical.segment_name,
         std::string("ofm") + std::to_string(i)});
    info.segment_name = logical.segment_name;
    if (info.segment_name.empty() && logical.physical_index >= 0 &&
        static_cast<std::size_t>(logical.physical_index) < stage->physical_outputs.size()) {
      info.segment_name =
          stage->physical_outputs[static_cast<std::size_t>(logical.physical_index)].segment_name;
    }
    if (info.segment_name.empty()) {
      info.segment_name = info.name;
    }
    info.size_bytes = static_cast<int64_t>(logical.size_bytes);
    if (info.size_bytes <= 0 && logical.physical_index >= 0 &&
        static_cast<std::size_t>(logical.physical_index) < stage->physical_outputs.size()) {
      info.size_bytes = static_cast<int64_t>(
          stage->physical_outputs[static_cast<std::size_t>(logical.physical_index)].size_bytes);
    }
    out.push_back(std::move(info));
  }
  return out;
}

inline stages::MlaOutputInfo mla_output_info(const NodeGroup& group) {
  stages::MlaOutputInfo info;
  const auto outputs = mla_output_tensors(group);
  if (outputs.empty()) {
    return info;
  }

  const auto& first = outputs.front();
  info.data_type = first.data_type;
  info.logical_data_type = first.data_type;
  info.output_format = first.output_format;
  info.layout = first.layout;
  info.dims = tensor_dims_projection_from_contract_shape(first.shape, first.layout);
  info.size_bytes = first.size_bytes;
  return info;
}

inline stages::MlaInputTensorInfo mla_input_tensor_info(const NodeGroup& group) {
  const auto manifest = rendered_manifest_from_group(group);
  if (!manifest.has_value()) {
    return {};
  }
  const auto* stage = find_mla_stage(*manifest);
  if (!stage || stage->logical_inputs.empty()) {
    return {};
  }

  stages::MlaInputTensorInfo info;
  const auto& logical = stage->logical_inputs.front();
  info.logical_shape = logical.shape;
  info.logical_layout = layout_projection_from_contract_format(logical.layout);
  info.logical_dtype = logical.dtype;
  info.logical_format = logical.layout;
  info.media_type = "application/vnd.simaai.tensor";
  info.segment_name =
      first_non_empty({logical.segment_name, logical.backend_name, logical.logical_name});
  if (logical.physical_index >= 0 &&
      static_cast<std::size_t>(logical.physical_index) < stage->physical_inputs.size()) {
    const auto& physical = stage->physical_inputs[static_cast<std::size_t>(logical.physical_index)];
    info.span_byte_offset = physical.source_byte_offset;
    info.span_size_bytes = static_cast<int64_t>(physical.size_bytes);
    if (!physical.segment_name.empty()) {
      info.segment_name = physical.segment_name;
    }
  }

  if (!stage->input_bindings.empty()) {
    const auto* binding = &stage->input_bindings.front();
    for (const auto& candidate : stage->input_bindings) {
      if (candidate.local_logical_input_index == logical.logical_index) {
        binding = &candidate;
        break;
      }
    }
    if (!binding->source_segment_name.empty()) {
      info.segment_name = binding->source_segment_name;
    }
  }

  if (info.span_size_bytes <= 0) {
    info.span_byte_offset = logical.byte_offset;
    info.span_size_bytes = static_cast<int64_t>(logical.size_bytes);
  }

  return info;
}

inline stages::MlaInputInfo mla_input_info(const NodeGroup& group) {
  const stages::MlaInputTensorInfo input = mla_input_tensor_info(group);
  stages::MlaInputInfo info;
  info.size_bytes = input.span_size_bytes;
  info.input_media_type = input.media_type;
  info.input_dtype = input.logical_dtype;
  info.input_format = input.logical_format;
  info.layout = input.logical_layout;
  info.logical_input_dtype = input.logical_dtype;
  info.logical_input_format = input.logical_format;
  info.logical_layout = input.logical_layout;
  info.logical_dims =
      tensor_dims_projection_from_contract_shape(input.logical_shape, input.logical_layout);
  info.dims = input.physical_shape.has_value()
                  ? tensor_dims_projection_from_contract_shape(*input.physical_shape,
                                                               input.logical_layout)
                  : info.logical_dims;
  return info;
}

inline std::optional<stages::PackedTensorOutputInfo> packed_processcvu_output_info(
    const sima::StageStaticSpec& stage,
    bool include_batch_axis) {
  if (stage.logical_outputs.empty()) {
    return std::nullopt;
  }

  stages::PackedTensorOutputInfo info;
  info.dtype = dtype_from_contract_token(stage.logical_outputs.front().dtype);
  info.output_format = stage.logical_outputs.front().layout;
  info.layout = layout_projection_from_contract_format(info.output_format);
  info.outputs.reserve(stage.logical_outputs.size());
  for (const auto& logical : stage.logical_outputs) {
    stages::PackedTensorOutput out;
    out.shape = maybe_strip_batch_axis(logical.shape, logical.layout, include_batch_axis);
    out.stride_bytes = maybe_strip_batch_axis_from_aligned_values(
        logical.stride_bytes, logical.shape, logical.layout, include_batch_axis);
    out.byte_offset = logical.byte_offset;
    info.outputs.push_back(std::move(out));
  }
  return info;
}

inline std::optional<stages::PackedTensorOutputInfo> packed_stage_output_info(
    const sima::StageStaticSpec& stage,
    bool include_batch_axis) {
  if (stage.logical_outputs.empty()) {
    return std::nullopt;
  }

  stages::PackedTensorOutputInfo info;
  info.dtype = dtype_from_contract_token(stage.logical_outputs.front().dtype);
  info.output_format = stage.logical_outputs.front().layout;
  info.layout = layout_projection_from_contract_format(info.output_format);
  info.outputs.reserve(stage.logical_outputs.size());
  for (const auto& logical : stage.logical_outputs) {
    stages::PackedTensorOutput out;
    out.shape = maybe_strip_batch_axis(logical.shape, logical.layout, include_batch_axis);
    out.stride_bytes = maybe_strip_batch_axis_from_aligned_values(
        logical.stride_bytes, logical.shape, logical.layout, include_batch_axis);
    out.byte_offset = logical.byte_offset;
    info.outputs.push_back(std::move(out));
  }
  return info;
}

inline stages::PackedTensorOutputInfo terminal_output_info(const NodeGroup& group,
                                                           bool include_batch_axis = true) {
  const auto manifest = rendered_manifest_from_group(group);
  if (!manifest.has_value()) {
    return {};
  }
  for (auto it = manifest->stages.rbegin(); it != manifest->stages.rend(); ++it) {
    const auto info = packed_stage_output_info(*it, include_batch_axis);
    if (info.has_value()) {
      return *info;
    }
  }
  return {};
}

inline stages::DetessDequantOutputInfo detessdequant_output_info(const NodeGroup& group,
                                                                 bool include_batch_axis = true) {
  const auto manifest = rendered_manifest_from_group(group);
  if (!manifest.has_value()) {
    return {};
  }
  const auto* stage = find_processcvu_stage(*manifest, "detessdequant");
  if (!stage) {
    return {};
  }
  const auto info = packed_processcvu_output_info(*stage, include_batch_axis);
  return info.has_value() ? *info : stages::DetessDequantOutputInfo{};
}

inline stages::DequantOutputInfo dequant_output_info(const NodeGroup& group,
                                                     bool include_batch_axis = true) {
  const auto manifest = rendered_manifest_from_group(group);
  if (!manifest.has_value()) {
    return {};
  }
  if (const auto* stage = find_processcvu_stage(*manifest, "dequantize"); stage) {
    const auto info = packed_processcvu_output_info(*stage, include_batch_axis);
    if (info.has_value()) {
      return *info;
    }
  }
  if (const auto* stage = find_stage_in_manifest(*manifest, [](const sima::StageStaticSpec& stage) {
        return stage.payload_kind == sima::StagePayloadKind::Dequant;
      });
      stage) {
    const auto info = packed_processcvu_output_info(*stage, include_batch_axis);
    if (info.has_value()) {
      return *info;
    }
  }
  return {};
}

} // namespace simaai::neat::pipeline_internal::rendered_stage_query
