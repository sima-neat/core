#include "pipeline/internal/sima/StageRecipePreparation.h"
#include "pipeline/internal/sima/KernelStageRecipe.h"
#include "pipeline/internal/sima/KernelTensorLayout.h"
#include "pipeline/internal/TensorTessellationMetadata.h"
#include "pipeline/internal/sima/CanonicalKernelLowering.h"
#include "pipeline/internal/sima/CanonicalGraphStageContract.h"
#include "pipeline/internal/sima/PreprocFrameAdapter.h"
#include <simaai/scheduler/client/frame_layout.h>

#include "model/internal/ModelPack.h"
#include "pipeline/internal/SimaaiGstCompat.h"
#include "pipeline/internal/EnvUtil.h"
#include "pipeline/internal/sima/InternalEdgeContractResolver.h"
#include "pipeline/internal/sima/MpkContract.h"
#include "pipeline/internal/sima/static_contract/MpkDecoder.h"
#include "pipeline/internal/sima/TensorSemanticsUtil.h"

#include <algorithm>
#include <array>
#include <cctype>
#include <cstddef>
#include <cstdarg>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <limits>
#include <optional>
#include <set>
#include <sstream>
#include <unordered_map>
#include <utility>

namespace simaai::neat::pipeline_internal::sima {
namespace {

struct CapsTensorSpec {
  int tensor_index = -1;
  std::vector<std::int64_t> shape;
  std::string dtype;
  std::string layout;
  MpkShapeSemantics shape_semantics = MpkShapeSemantics::Unknown;
  int max_w = 0;
  int max_h = 0;
  int max_stride = 0;
  std::string semantic_tag;
  std::uint64_t size_bytes = 0U;
};

struct GraphProcessCvuIoData {
  std::vector<MpkTensorContract> input_tensors;
  std::vector<MpkTensorContract> output_tensors;
  std::vector<std::vector<int>> slice_shapes;
  std::string canonical_input_dtype;
  std::string canonical_output_dtype;
  std::string input_slot_name;
  std::vector<std::string> runtime_output_slot_names;
  // True when the MPK contract sets align_c16 or cblock for this stage.
  // Propagates the C16 storage fact while cblock below preserves its encoding.
  bool c16_packed = false;
  // Distinguishes CBlock16 from align_c16's padded pixel-major HWC storage.
  bool cblock = false;
};

constexpr std::uint32_t
processcvu_tiled_encoding_flags_local(const GraphTiledChannelEncoding encoding) {
  if (encoding == GraphTiledChannelEncoding::CBlock16) {
    return SIMA_EV_TILED_FLAG_CBLOCK16;
  }
  return encoding == GraphTiledChannelEncoding::PaddedHwcC16 ? SIMA_EV_TILED_FLAG_PADDED_HWC_C16
                                                             : SIMA_EV_TILED_FLAG_COMPACT_CHANNELS;
}

bool checked_add_u64_local(const std::uint64_t a, const std::uint64_t b, std::uint64_t* out);

std::string processcvu_canonical_graph_name_local(std::string graph_name);
bool processcvu_is_native_visual_graph_local(const std::string& graph_name);

bool processcvu_graph_family_uses_packed_input_transport_local(const std::string& graph_family);

bool processcvu_build_stage_tensor_descs_local(const StageStaticSpec& stage,
                                               ProcessCvuStagePayload* cfg,
                                               std::string* error_message);

const LogicalInputStaticSpec*
processcvu_find_logical_input_by_index_local(const StageStaticSpec& stage, int logical_index);

std::string prepared_input_layout_token_local(const ProcessCvuStagePayload& cfg) {
  if (!cfg.input_tensors.empty()) {
    return tensorsemantics::layout_token_from_ev_tensor_desc(cfg.input_tensors.front());
  }
  return {};
}

std::string prepared_output_layout_token_local(const ProcessCvuStagePayload& cfg,
                                               std::size_t index = 0U) {
  if (index < cfg.output_tensors.size()) {
    return tensorsemantics::layout_token_from_ev_tensor_desc(cfg.output_tensors[index]);
  }
  if (!cfg.output_tensors.empty()) {
    return tensorsemantics::layout_token_from_ev_tensor_desc(cfg.output_tensors.front());
  }
  return {};
}

std::vector<std::int64_t> shape_vector_from_ev_shape_local(const sima_ev_shape_desc& shape) {
  std::vector<std::int64_t> out;
  const auto rank = std::min<std::uint32_t>(shape.rank, SIMA_EV_MAX_RANK);
  out.reserve(rank);
  for (std::uint32_t i = 0; i < rank; ++i) {
    out.push_back(shape.sizes[i]);
  }
  return out;
}

std::optional<MpkContract> load_graph_contract_from_manifest_local(const SimaPluginStaticManifest&,
                                                                   std::filesystem::path*,
                                                                   std::string*);
std::optional<MpkContract>
load_graph_contract_from_pipeline_elements_local(const std::vector<PipelineElementSpec>&,
                                                 std::filesystem::path*, std::string*);
std::optional<MpkContract>
load_graph_contract_from_model_sources_local(const std::vector<std::string>&,
                                             std::filesystem::path*, std::string*);

std::string upper_copy_local(std::string value) {
  std::transform(value.begin(), value.end(), value.begin(),
                 [](unsigned char c) { return static_cast<char>(std::toupper(c)); });
  return value;
}

std::string stage_key_from_stage_spec_local(const StageStaticSpec& stage) {
  if (!stage.logical_stage_id.empty()) {
    return stage.logical_stage_id;
  }
  return stage.element_name;
}

bool stage_is_processmla_local(const StageStaticSpec& stage) {
  return stage.payload_kind == StagePayloadKind::ProcessMla ||
         upper_copy_local(stage.plugin_kind) == "PROCESSMLA" ||
         upper_copy_local(stage.kernel_kind) == "PROCESSMLA" ||
         upper_copy_local(stage.kernel_kind) == "MLA";
}

bool stage_is_processcvu_local(const StageStaticSpec& stage) {
  return stage.payload_kind == StagePayloadKind::ProcessCvu ||
         upper_copy_local(stage.plugin_kind) == "NEATPROCESSCVU" ||
         upper_copy_local(stage.plugin_kind) == "PROCESSCVU";
}

bool processcvu_stage_is_manifest_substitution_local(const StageStaticSpec& stage) {
  if (!stage_is_processcvu_local(stage)) {
    return false;
  }
  const std::string canonical_family = processcvu_canonical_graph_name_local(
      !stage.processcvu.graph_family.empty() ? stage.processcvu.graph_family
                                             : stage.processcvu.graph_name);
  return canonical_family == "preproc" || canonical_family == "quantize" ||
         canonical_family == "quanttess" || canonical_family == "tessellate" ||
         canonical_family == "detessellate" || canonical_family == "dequantize" ||
         canonical_family == "detessdequant" || canonical_family == "detesscast" ||
         canonical_family == "cast" || canonical_family == "casttess" ||
         canonical_family == "feature_histogram" || canonical_family == "grider_fast" ||
         canonical_family == "track_descriptor" || canonical_family == "track_klt" ||
         canonical_family == "simor_depth_map";
}

bool stage_is_graph_owned_local(const StageStaticSpec& stage) {
  return stage_is_processcvu_local(stage) &&
         !processcvu_stage_is_manifest_substitution_local(stage);
}

bool graph_node_has_member_local(const MpkGraphNode& node, const std::string& member_name);

bool graph_node_matches_stage_key_local(const MpkGraphNode& node, const std::string& stage_key) {
  if (stage_key.empty()) {
    return false;
  }
  if (node.name == stage_key || graph_node_has_member_local(node, stage_key)) {
    return true;
  }
  const std::string prefix = node.name + "_";
  return !node.name.empty() && stage_key.rfind(prefix, 0) == 0;
}

bool prepared_runtime_debug_enabled_local() {
  return pipeline_internal::env_bool("SIMA_PREPARED_RUNTIME_BUILD_DEBUG", false);
}

void prepared_runtime_debug_log_local(const char* fmt, ...) {
  if (!prepared_runtime_debug_enabled_local() || !fmt) {
    return;
  }
  std::fprintf(stderr, "[prepared-runtime-build] ");
  va_list args;
  va_start(args, fmt);
  std::vfprintf(stderr, fmt, args);
  va_end(args);
  std::fprintf(stderr, "\n");
}

int tensorbuffer_dtype_from_token_local(const std::string& raw) {
  const std::string up = upper_copy_local(raw);
  if (up == "INT8" || up == "EVXX_INT8") {
    return SIMA_TENSOR_SET_DTYPE_INT8_V1;
  }
  if (up == "UINT8" || up == "U8" || up == "EVXX_UINT8") {
    return SIMA_TENSOR_SET_DTYPE_UINT8_V1;
  }
  if (up == "BF16" || up == "BFLOAT16" || up == "EVXX_BFLOAT16") {
    return SIMA_TENSOR_SET_DTYPE_BF16_V1;
  }
  if (up == "FP32" || up == "FLOAT32" || up == "EVXX_FP32" || up == "EVXX_FLOAT32") {
    return SIMA_TENSOR_SET_DTYPE_FP32_V1;
  }
  if (up == "INT16" || up == "EVXX_INT16") {
    return SIMA_TENSOR_SET_DTYPE_INT16_V1;
  }
  if (up == "UINT16" || up == "EVXX_UINT16") {
    return SIMA_TENSOR_SET_DTYPE_UINT16_V1;
  }
  if (up == "INT32" || up == "EVXX_INT32") {
    return SIMA_TENSOR_SET_DTYPE_INT32_V1;
  }
  if (up == "UINT32") {
    return SIMA_TENSOR_SET_DTYPE_UNKNOWN_V1;
  }
  if (up == "FP64" || up == "FLOAT64" || up == "EVXX_FP64" || up == "EVXX_FLOAT64") {
    return SIMA_TENSOR_SET_DTYPE_FP64_V1;
  }
  return SIMA_TENSOR_SET_DTYPE_UNKNOWN_V1;
}

int tensorbuffer_layout_from_token_local(const std::string& raw) {
  const std::string normalized = tensorsemantics::normalize_layout_token(raw);
  if (normalized == "HWC") {
    return SIMA_TENSOR_SET_LAYOUT_HWC_V1;
  }
  if (normalized == "CHW") {
    return SIMA_TENSOR_SET_LAYOUT_CHW_V1;
  }
  if (normalized == "HW") {
    return SIMA_TENSOR_SET_LAYOUT_HW_V1;
  }
  return SIMA_TENSOR_SET_LAYOUT_UNKNOWN_V1;
}

std::size_t dtype_size_bytes_local(const std::string& raw) {
  const std::string up = upper_copy_local(raw);
  if (up == "BF16" || up == "BFLOAT16" || up == "FP16" || up == "FLOAT16" || up == "INT16" ||
      up == "UINT16") {
    return 2U;
  }
  if (up == "FP32" || up == "FLOAT32" || up == "INT32" || up == "UINT32") {
    return 4U;
  }
  if (up == "FP64" || up == "FLOAT64") {
    return 8U;
  }
  return 1U;
}

std::vector<std::int64_t> contiguous_stride_bytes_local(const std::vector<std::int64_t>& shape,
                                                        const std::string& dtype) {
  std::vector<std::int64_t> strides(shape.size(), 0);
  const std::size_t elem_bytes = dtype_size_bytes_local(dtype);
  if (elem_bytes == 0U) {
    return strides;
  }
  std::int64_t stride = static_cast<std::int64_t>(elem_bytes);
  for (std::size_t i = shape.size(); i-- > 0U;) {
    strides[i] = stride;
    if (shape[i] > 0 && stride <= std::numeric_limits<std::int64_t>::max() / shape[i]) {
      stride *= shape[i];
    }
  }
  return strides;
}

std::uint64_t shape_size_bytes_local(const std::vector<std::int64_t>& shape,
                                     const std::string& dtype) {
  if (shape.empty()) {
    return 0U;
  }
  std::uint64_t total = static_cast<std::uint64_t>(dtype_size_bytes_local(dtype));
  for (const auto dim : shape) {
    if (dim <= 0) {
      return 0U;
    }
    const auto u_dim = static_cast<std::uint64_t>(dim);
    if (u_dim > 0U && total > (std::numeric_limits<std::uint64_t>::max() / u_dim)) {
      return 0U;
    }
    total *= u_dim;
  }
  return total;
}

std::string processmla_caps_format_from_dtype_local(const std::string& dtype,
                                                    const std::string& fallback_format) {
  std::string out = fallback_format.empty() ? std::string("EVXX_INT8") : fallback_format;
  const std::string up = upper_copy_local(dtype);
  if (up.empty()) {
    return out;
  }
  if (up == "INT8") {
    return "EVXX_INT8";
  }
  if (up == "UINT8") {
    return "UINT8";
  }
  if (up == "INT16") {
    return "INT16";
  }
  if (up == "UINT16") {
    return "UINT16";
  }
  if (up == "INT32") {
    return "INT32";
  }
  if (up == "FP32" || up == "FLOAT32") {
    return "FP32";
  }
  if (up == "FP64" || up == "FLOAT64") {
    return "FP64";
  }
  if (up == "BF16" || up == "BFLOAT16") {
    return "EVXX_BFLOAT16";
  }
  return up;
}

std::string processcvu_caps_format_from_dtype_local(const std::string& value,
                                                    const std::string& fallback) {
  std::string out = fallback.empty() ? std::string("EVXX_INT8") : fallback;
  const std::string up = upper_copy_local(value);
  if (up.empty()) {
    return out;
  }
  if (up.rfind("EVXX_", 0) == 0) {
    return up;
  }
  if (up == "INT8") {
    return "EVXX_INT8";
  }
  if (up == "INT16") {
    return "EVXX_INT16";
  }
  if (up == "INT32") {
    return "EVXX_INT32";
  }
  if (up == "FP16" || up == "FLOAT16" || up == "HALF") {
    return "EVXX_FLOAT16";
  }
  if (up == "FP32" || up == "FLOAT32") {
    return "EVXX_FLOAT32";
  }
  if (up == "BF16" || up == "BFLOAT16") {
    return "EVXX_BFLOAT16";
  }
  return up;
}

void processmla_add_shape_fields_to_caps_local(GstCaps* caps, const CapsTensorSpec& tensor);

GstCaps* make_generic_tensor_set_caps_local() {
  GstCaps* caps = gst_caps_new_empty_simple("application/vnd.simaai.tensor");
  if (!caps) {
    return gst_caps_new_any();
  }
  gst_caps_set_simple(caps, "representation", G_TYPE_STRING, "tensor-set", "storage", G_TYPE_STRING,
                      "tensorbuffer", nullptr);
  return caps;
}

GstCaps* processmla_make_tensor_caps_from_tensor_local(const CapsTensorSpec& tensor,
                                                       const std::string& fallback_format,
                                                       bool prefer_tensor_dtype) {
  const std::string format = processmla_caps_format_from_dtype_local(tensor.dtype, fallback_format);
  GstCaps* caps = gst_caps_new_simple("application/vnd.simaai.tensor", "format", G_TYPE_STRING,
                                      format.c_str(), nullptr);
  if (!caps) {
    return gst_caps_new_any();
  }
  const char* dtype_token =
      prefer_tensor_dtype && !tensor.dtype.empty() ? tensor.dtype.c_str() : format.c_str();
  if (dtype_token && *dtype_token) {
    gst_caps_set_simple(caps, "dtype", G_TYPE_STRING, dtype_token, nullptr);
  }
  processmla_add_shape_fields_to_caps_local(caps, tensor);
  return caps;
}

CapsTensorSpec caps_tensor_from_publish_logical_output_local(
    const simaai::gst::TensorBufferPublishLogicalOutput& output) {
  CapsTensorSpec out;
  out.tensor_index = output.backend_output_index;
  out.shape = output.shape;
  out.size_bytes = output.size_bytes;
  out.semantic_tag = !output.logical_name.empty() ? output.logical_name : output.segment_name;
  switch (output.dtype) {
  case SIMA_TENSOR_SET_DTYPE_BF16_V1:
    out.dtype = "BF16";
    break;
  case SIMA_TENSOR_SET_DTYPE_FP32_V1:
    out.dtype = "FP32";
    break;
  case SIMA_TENSOR_SET_DTYPE_INT8_V1:
    out.dtype = "INT8";
    break;
  case SIMA_TENSOR_SET_DTYPE_UINT8_V1:
    out.dtype = "UINT8";
    break;
  case SIMA_TENSOR_SET_DTYPE_INT16_V1:
    out.dtype = "INT16";
    break;
  case SIMA_TENSOR_SET_DTYPE_UINT16_V1:
    out.dtype = "UINT16";
    break;
  case SIMA_TENSOR_SET_DTYPE_INT32_V1:
    out.dtype = "INT32";
    break;
  default:
    out.dtype.clear();
    break;
  }
  switch (output.layout) {
  case SIMA_TENSOR_SET_LAYOUT_CHW_V1:
    out.layout = "CHW";
    break;
  case SIMA_TENSOR_SET_LAYOUT_HWC_V1:
    out.layout = "HWC";
    break;
  case SIMA_TENSOR_SET_LAYOUT_HW_V1:
    out.layout = "HW";
    break;
  default:
    out.layout.clear();
    break;
  }
  if (out.size_bytes == 0U) {
    out.size_bytes = shape_size_bytes_local(out.shape, out.dtype);
  }
  return out;
}

void processmla_add_shape_fields_to_caps_local(GstCaps* caps, const CapsTensorSpec& tensor) {
  if (!caps || tensor.shape.empty()) {
    return;
  }
  GstStructure* s = gst_caps_get_structure(caps, 0);
  if (!s) {
    return;
  }
  gst_structure_set(s, "rank", G_TYPE_INT, static_cast<gint>(tensor.shape.size()), nullptr);
  std::ostringstream csv;
  for (std::size_t i = 0; i < tensor.shape.size(); ++i) {
    const auto dim = tensor.shape[i];
    if (dim <= 0 || dim > static_cast<std::int64_t>(std::numeric_limits<gint>::max())) {
      return;
    }
    const std::string key = "dim" + std::to_string(i);
    gst_structure_set(s, key.c_str(), G_TYPE_INT, static_cast<gint>(dim), nullptr);
    if (i > 0U) {
      csv << ",";
    }
    csv << dim;
  }
  gst_structure_set(s, "shape", G_TYPE_STRING, csv.str().c_str(), nullptr);
}

bool build_publish_contract_from_manifest_stage_local(const StageStaticSpec& stage,
                                                      simaai::gst::TensorBufferPublishContract* out,
                                                      std::string* error_message) {
  if (!out) {
    if (error_message) {
      *error_message = "prepared publish contract requires output storage";
    }
    return false;
  }
  *out = simaai::gst::TensorBufferPublishContract{};
  out->stage_key = stage_key_from_stage_spec_local(stage);
  out->preserve_physical_segments = stage.consumer_keeps_distinct_physical_inputs;
  if (stage.logical_outputs.empty()) {
    if (error_message) {
      *error_message = "prepared publish contract requires logical_outputs";
    }
    return false;
  }
  if (stage.frame_layout) {
    for (std::size_t index = 0; index < stage.frame_layout->allocations.size(); ++index) {
      simaai::gst::TensorBufferPublishPhysicalOutput output;
      output.physical_index = static_cast<int>(index);
      output.size_bytes = stage.frame_layout->allocations[index].size_bytes;
      output.segment_name = "allocation_" + std::to_string(index);
      out->physical_outputs.push_back(std::move(output));
    }
  } else {
    out->physical_outputs.reserve(stage.physical_outputs.size());
    for (const auto& physical : stage.physical_outputs) {
      if (stage.native_kernel &&
          std::any_of(stage.native_kernel->private_buffers.begin(),
                      stage.native_kernel->private_buffers.end(), [&](const auto& buffer) {
                        return buffer.physical_index == physical.physical_index;
                      }))
        continue;
      simaai::gst::TensorBufferPublishPhysicalOutput output;
      output.physical_index = physical.physical_index;
      output.size_bytes = physical.size_bytes;
      output.segment_name = physical.segment_name;
      out->physical_outputs.push_back(std::move(output));
    }
  }
  out->logical_outputs.reserve(stage.logical_outputs.size());
  for (const auto& logical : stage.logical_outputs) {
    simaai::gst::TensorBufferPublishLogicalOutput output;
    output.logical_index = logical.logical_index;
    output.physical_index = logical.physical_index;
    output.memory_index = logical.physical_index;
    output.backend_output_index = logical.backend_output_index;
    output.route_slot = logical.output_slot;
    output.logical_name = logical.logical_name;
    output.backend_name = logical.backend_name;
    output.segment_name = logical.segment_name;
    output.byte_offset = logical.byte_offset;
    output.size_bytes = logical.size_bytes;
    if (stage.frame_layout) {
      const auto binding = std::find_if(
          stage.frame_layout->outputs.begin(), stage.frame_layout->outputs.end(),
          [&](const auto& candidate) {
            return candidate.tensor == static_cast<std::size_t>(logical.backend_output_index);
          });
      const auto relative = stage.native_kernel && stage.native_kernel->image_publication
                                ? std::int64_t{0}
                                : logical.byte_offset;
      if (binding == stage.frame_layout->outputs.end() ||
          binding->span.allocation_index >= out->physical_outputs.size() ||
          logical.byte_offset < 0 || std::uint64_t(relative) > binding->span.size_bytes ||
          logical.size_bytes > binding->span.size_bytes - std::uint64_t(relative)) {
        if (error_message)
          *error_message = "publication view exceeds its final physical binding";
        return false;
      }
      output.physical_index = static_cast<int>(binding->span.allocation_index);
      output.memory_index = static_cast<int>(binding->span.allocation_index);
      output.segment_name = out->physical_outputs[binding->span.allocation_index].segment_name;
      output.byte_offset = static_cast<std::int64_t>(binding->span.byte_offset) + relative;
    }
    output.shape = logical.shape;
    output.stride_bytes = logical.stride_bytes;
    output.dtype = tensorbuffer_dtype_from_token_local(logical.dtype);
    output.layout = tensorbuffer_layout_from_token_local(logical.layout);
    if (output.stride_bytes.empty() && !output.shape.empty() &&
        output.dtype != SIMA_TENSOR_SET_DTYPE_UNKNOWN_V1) {
      output.stride_bytes = contiguous_stride_bytes_local(output.shape, logical.dtype);
    }
    if (logical.quant.has_value()) {
      simaai::gst::TensorBufferQuantView quant;
      quant.granularity =
          static_cast<int>(logical.quant->granularity == QuantGranularity::PerAxis ? 1 : 0);
      quant.axis = logical.quant->axis;
      quant.scales = logical.quant->scales;
      quant.zero_points = logical.quant->zero_points;
      output.quant = std::move(quant);
    }
    out->logical_outputs.push_back(std::move(output));
  }
  out->output_order.reserve(stage.output_order.size());
  for (const auto& route : stage.output_order) {
    simaai::gst::TensorBufferPublishOutputRoute output_route;
    output_route.output_slot = route.output_slot;
    output_route.logical_output_index = route.logical_output_index;
    output_route.cm_output_name = route.cm_output_name;
    output_route.segment_name = route.segment_name;
    if (stage.frame_layout) {
      const auto logical = std::find_if(
          out->logical_outputs.begin(), out->logical_outputs.end(),
          [&](const auto& output) { return output.logical_index == route.logical_output_index; });
      if (logical == out->logical_outputs.end()) {
        if (error_message)
          *error_message = "publication route has no final logical output";
        return false;
      }
      output_route.segment_name = logical->segment_name;
    }
    out->output_order.push_back(std::move(output_route));
  }
  return true;
}

bool processcvu_stage_is_dequant_like_local(const StageStaticSpec& stage) {
  if (stage.payload_kind != StagePayloadKind::ProcessCvu) {
    return false;
  }
  if (stage.processcvu.graph_family_enum == ProcessCvuGraphFamily::Dequant ||
      stage.processcvu.graph_family_enum == ProcessCvuGraphFamily::DetessDequant) {
    return true;
  }
  auto contains_dequant = [](std::string value) {
    std::transform(value.begin(), value.end(), value.begin(),
                   [](unsigned char c) { return static_cast<char>(std::tolower(c)); });
    return value.find("dequant") != std::string::npos;
  };
  return contains_dequant(stage.processcvu.graph_family) ||
         contains_dequant(stage.processcvu.graph_name) || contains_dequant(stage.kernel_kind);
}

void attach_processcvu_dequant_qparams_to_publish_contract_local(
    const StageStaticSpec& stage, const ProcessCvuStagePayload& typed_config,
    simaai::gst::TensorBufferPublishContract* contract) {
  if (!contract || contract->logical_outputs.empty() ||
      !processcvu_stage_is_dequant_like_local(stage)) {
    return;
  }
  const std::size_t qparam_count =
      std::min(typed_config.dq_scale_list.size(), typed_config.dq_zp_list.size());
  if (qparam_count == 0U) {
    return;
  }

  for (std::size_t i = 0; i < contract->logical_outputs.size(); ++i) {
    auto& logical = contract->logical_outputs[i];
    std::size_t q_index = i;
    if (logical.route_slot >= 0 && static_cast<std::size_t>(logical.route_slot) < qparam_count) {
      q_index = static_cast<std::size_t>(logical.route_slot);
    } else if (logical.logical_index >= 0 &&
               static_cast<std::size_t>(logical.logical_index) < qparam_count) {
      q_index = static_cast<std::size_t>(logical.logical_index);
    }
    if (q_index >= qparam_count) {
      continue;
    }
    simaai::gst::TensorBufferQuantView quant;
    quant.granularity = 0;
    quant.axis = -1;
    quant.scales = {static_cast<double>(static_cast<float>(typed_config.dq_scale_list[q_index]))};
    quant.zero_points = {static_cast<std::int64_t>(typed_config.dq_zp_list[q_index])};
    logical.quant = std::move(quant);
  }
}

bool processcvu_primary_output_uses_packed_caps_from_payload_local(
    const ProcessCvuStagePayload& payload) {
  return payload.primary_output_transport_kind == ProcessCvuOutputTransportKind::Packed ||
         payload.primary_output_semantic_kind == ProcessCvuOutputSemanticKind::TessellatedImage ||
         payload.primary_output_semantic_kind == ProcessCvuOutputSemanticKind::QuantTessTensor;
}

bool checked_add_u64_local(const std::uint64_t a, const std::uint64_t b, std::uint64_t* out);

std::string processcvu_canonical_graph_name_local(std::string graph_name) {
  graph_name = upper_copy_local(std::move(graph_name));
  if (graph_name == "PREPROC" || graph_name == "PREPROCESS") {
    return "preproc";
  }
  if (graph_name == "QUANT" || graph_name == "QUANTIZE") {
    return "quantize";
  }
  if (graph_name == "QUANTIZETENSOR" || graph_name == "QUANTIZE_TENSOR" ||
      graph_name == "QUANTIZEGENERIC" || graph_name == "QUANTIZE_GENERIC") {
    return "quantizetensor";
  }
  if (graph_name == "QUANTIZETESSELLATE" || graph_name == "QUANTIZE_TESSELLATE") {
    return "quantizetessellate";
  }
  if (graph_name == "TESS" || graph_name == "TESSELLATE") {
    return "tessellate";
  }
  if (graph_name == "QUANTTESS" || graph_name == "QUANT_TESS") {
    return "quanttess";
  }
  if (graph_name == "CASTTESS" || graph_name == "CAST_TESS" || graph_name == "CASTTESSELLATE") {
    return "casttess";
  }
  if (graph_name == "DEQUANT" || graph_name == "DEQUANTIZE") {
    return "dequantize";
  }
  if (graph_name == "DETESS" || graph_name == "DETESSELLATE") {
    return "detessellate";
  }
  if (graph_name == "DETESSDEQUANT" || graph_name == "DETESS_DEQUANT") {
    return "detessdequant";
  }
  if (graph_name == "DETESSCAST" || graph_name == "DETESS_CAST" ||
      graph_name == "DETESSELLATECAST") {
    return "detesscast";
  }
  std::transform(graph_name.begin(), graph_name.end(), graph_name.begin(),
                 [](unsigned char c) { return static_cast<char>(std::tolower(c)); });
  if (graph_name == "FEATUREHISTOGRAM" || graph_name == "FEATURE_HISTOGRAM" ||
      graph_name == "featurehistogram" || graph_name == "feature_histogram") {
    return "feature_histogram";
  }
  if (graph_name == "GRIDERFAST" || graph_name == "GRIDER_FAST" || graph_name == "griderfast" ||
      graph_name == "grider_fast") {
    return "grider_fast";
  }
  if (graph_name == "TRACKDESCRIPTOR" || graph_name == "TRACK_DESCRIPTOR" ||
      graph_name == "trackdescriptor" || graph_name == "track_descriptor") {
    return "track_descriptor";
  }
  if (graph_name == "TRACKKLT" || graph_name == "TRACK_KLT" || graph_name == "trackklt" ||
      graph_name == "track_klt") {
    return "track_klt";
  }
  return graph_name;
}

bool processcvu_is_native_visual_graph_local(const std::string& graph_name) {
  const std::string canonical = processcvu_canonical_graph_name_local(graph_name);
  return canonical == "feature_histogram" || canonical == "grider_fast" ||
         canonical == "track_descriptor" || canonical == "track_klt" ||
         canonical == "simor_depth_map";
}

bool processcvu_graph_family_uses_packed_input_transport_local(const std::string& graph_family) {
  const std::string canonical = processcvu_canonical_graph_name_local(graph_family);
  return canonical == "detessellate" || canonical == "detesscast" || canonical == "detessdequant";
}

namespace {

bool processcvu_dtype_token_to_ev_local(const std::string& raw_dtype, std::uint32_t* out_dtype) {
  if (!out_dtype) {
    return false;
  }
  const std::string token = upper_copy_local(raw_dtype);
  if (token == "FP32" || token == "FLOAT32" || token == "EVXX_FLOAT32") {
    *out_dtype = SIMA_EV_DTYPE_FP32;
    return true;
  }
  if (token == "BF16" || token == "BFLOAT16" || token == "EVXX_BFLOAT16") {
    *out_dtype = SIMA_EV_DTYPE_BF16;
    return true;
  }
  if (token == "INT16" || token == "EVXX_INT16") {
    *out_dtype = SIMA_EV_DTYPE_INT16;
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

bool processcvu_normalize_tile_shape_local(const std::vector<std::int64_t>& shape,
                                           const std::vector<int>& raw_tile_shape,
                                           std::vector<std::int64_t>* out,
                                           std::string* error_message) {
  if (!out) {
    if (error_message) {
      *error_message = "processcvu tile shape requires output storage";
    }
    return false;
  }
  out->clear();
  std::vector<int> shape_int;
  shape_int.reserve(shape.size());
  for (const auto dim : shape) {
    if (dim <= 0 || dim > static_cast<std::int64_t>(std::numeric_limits<int>::max())) {
      if (error_message) {
        *error_message = "processcvu tile shape invalid tensor shape";
      }
      return false;
    }
    shape_int.push_back(static_cast<int>(dim));
  }
  std::vector<int> normalized;
  if (!tensorsemantics::normalize_tile_shape(
          shape_int, raw_tile_shape, &normalized, error_message, "processcvu tile shape missing",
          "tile_shape_rank_prefix_invalid", "processcvu tile shape invalid")) {
    return false;
  }
  out->assign(normalized.begin(), normalized.end());
  return true;
}

bool processcvu_build_dense_tensor_desc_local(const std::vector<std::int64_t>& shape,
                                              const std::string& dtype_token,
                                              const std::string& layout_token,
                                              const std::uint64_t size_bytes,
                                              sima_ev_tensor_desc* out,
                                              std::string* error_message) {
  if (!out) {
    if (error_message) {
      *error_message = "processcvu dense tensor desc requires output storage";
    }
    return false;
  }
  const std::string normalized_layout = tensorsemantics::normalize_layout_token(layout_token);
  if (!layout_token.empty() && normalized_layout.empty()) {
    if (error_message) {
      *error_message = "processcvu dense tensor layout invalid";
    }
    return false;
  }
  if (normalized_layout.empty()) {
    if (!tensorsemantics::build_generic_dense_tensor_desc(
            shape, dtype_token, out, error_message,
            "processcvu dense tensor desc requires output storage", "processcvu shape rank invalid",
            "processcvu shape dim invalid", "processcvu dense tensor dtype invalid",
            "processcvu dense stride desc requires output storage")) {
      return false;
    }
    out->storage.nbytes =
        size_bytes != 0U ? size_bytes : shape_size_bytes_local(shape, dtype_token);
    return true;
  }
  std::memset(out, 0, sizeof(*out));
  if (!tensorsemantics::fill_shape_desc(shape, normalized_layout, &out->shape, error_message,
                                        "processcvu shape desc requires output storage",
                                        "processcvu shape rank invalid",
                                        "processcvu shape dim invalid")) {
    return false;
  }
  std::uint32_t dtype = 0;
  if (!processcvu_dtype_token_to_ev_local(dtype_token, &dtype)) {
    if (error_message) {
      *error_message = "processcvu dense tensor dtype invalid";
    }
    return false;
  }
  out->dtype = dtype;
  out->layout_kind = SIMA_EV_LAYOUT_STRIDED;
  out->storage.nbytes = size_bytes != 0U ? size_bytes : shape_size_bytes_local(shape, dtype_token);
  return tensorsemantics::fill_dense_strides(out->shape, normalized_layout, out->dtype,
                                             &out->layout.strided, error_message,
                                             "processcvu dense stride desc requires output storage",
                                             "processcvu dense stride dtype invalid");
}

bool processcvu_compute_tiled_storage_nbytes_local(const sima_ev_tensor_desc& desc,
                                                   const GraphTiledChannelEncoding encoding,
                                                   std::uint64_t* out) {
  if (!out || desc.layout_kind != SIMA_EV_LAYOUT_TILED || desc.shape.rank == 0U ||
      desc.shape.rank > SIMA_EV_MAX_RANK) {
    return false;
  }
  const int element_bytes = sima_ev_elem_size_bytes(desc.dtype);
  const auto checked_mul_u64_local = [](std::uint64_t a, std::uint64_t b, std::uint64_t* output) {
    if (!output || (a != 0 && b > UINT64_MAX / a))
      return false;
    *output = a * b;
    return true;
  };
  if (element_bytes <= 0) {
    return false;
  }
  const bool c16_channel_storage = encoding == GraphTiledChannelEncoding::PaddedHwcC16 ||
                                   encoding == GraphTiledChannelEncoding::CBlock16;
  const int channel_axis = tensorsemantics::find_shape_axis(desc.shape, SIMA_EV_AXIS_C);
  if (c16_channel_storage && channel_axis < 0) {
    return false;
  }

  const std::uint64_t tile_align =
      desc.layout.tiled.tile_align_bytes != 0U ? desc.layout.tiled.tile_align_bytes : 1U;
  std::uint64_t total = 0U;
  const auto walk = [&](const auto& self, const std::uint32_t axis,
                        const std::uint64_t tile_elements,
                        const std::uint64_t tile_multiplicity) -> bool {
    if (axis == desc.shape.rank) {
      std::uint64_t payload = 0U;
      if (!checked_mul_u64_local(tile_elements, static_cast<std::uint64_t>(element_bytes),
                                 &payload)) {
        return false;
      }
      std::uint64_t padded_payload = 0U;
      if (!checked_add_u64_local(payload, tile_align - 1U, &padded_payload)) {
        return false;
      }
      const std::uint64_t aligned_payload = padded_payload / tile_align * tile_align;
      std::uint64_t class_bytes = 0U;
      return aligned_payload != 0U &&
             checked_mul_u64_local(aligned_payload, tile_multiplicity, &class_bytes) &&
             checked_add_u64_local(total, class_bytes, &total);
    }

    const std::int64_t dim_i = desc.shape.sizes[axis];
    const std::int64_t tile_i = desc.layout.tiled.tile_sizes[axis];
    if (dim_i <= 0 || tile_i <= 0 || tile_i > dim_i) {
      return false;
    }
    const std::uint64_t dim = static_cast<std::uint64_t>(dim_i);
    const std::uint64_t tile = static_cast<std::uint64_t>(tile_i);
    const auto visit_class = [&](const std::uint64_t logical_extent,
                                 const std::uint64_t count) -> bool {
      if (count == 0U) {
        return true;
      }
      std::uint64_t stored_extent = logical_extent;
      if (c16_channel_storage && static_cast<int>(axis) == channel_axis) {
        std::uint64_t rounded = 0U;
        if (!checked_add_u64_local(logical_extent, 15U, &rounded)) {
          return false;
        }
        stored_extent = rounded / 16U * 16U;
      }
      std::uint64_t next_elements = 0U;
      std::uint64_t next_multiplicity = 0U;
      if (!checked_mul_u64_local(tile_elements, stored_extent, &next_elements) ||
          !checked_mul_u64_local(tile_multiplicity, count, &next_multiplicity) ||
          !self(self, axis + 1U, next_elements, next_multiplicity)) {
        return false;
      }
      return true;
    };
    const std::uint64_t full_count = dim / tile;
    const std::uint64_t remainder = dim % tile;
    if (!visit_class(tile, full_count) || (remainder != 0U && !visit_class(remainder, 1U))) {
      return false;
    }
    return true;
  };

  // Each axis contributes at most a full-tile and a remainder-tile class, so
  // exact partial-tile accounting is bounded by 2^rank rather than tensor size.
  if (!walk(walk, 0U, 1U, 1U) || total == 0U) {
    return false;
  }
  *out = total;
  return true;
}

bool processcvu_build_tiled_tensor_desc_local(
    const std::vector<std::int64_t>& shape, const std::vector<std::int64_t>& tile_shape,
    const std::string& dtype_token, const std::string& layout_token,
    const std::uint32_t tile_align_bytes, const std::uint64_t size_bytes,
    const GraphTiledChannelEncoding encoding, sima_ev_tensor_desc* out,
    std::string* error_message) {
  if (!out) {
    if (error_message) {
      *error_message = "processcvu tiled tensor desc requires output storage";
    }
    return false;
  }
  std::memset(out, 0, sizeof(*out));
  if (shape.size() != tile_shape.size()) {
    if (error_message) {
      *error_message = "processcvu tiled tensor rank mismatch";
    }
    return false;
  }
  const std::string normalized_layout = tensorsemantics::normalize_layout_token(layout_token);
  if (!layout_token.empty() && normalized_layout.empty()) {
    if (error_message) {
      *error_message = "processcvu tiled tensor layout invalid";
    }
    return false;
  }
  if (normalized_layout.empty()) {
    if (!tensorsemantics::build_generic_tiled_tensor_desc(
            shape, tile_shape, dtype_token, tile_align_bytes, out, error_message,
            "processcvu tiled tensor desc requires output storage", "processcvu shape rank invalid",
            "processcvu shape dim invalid", "processcvu tiled tensor dtype invalid",
            "processcvu tiled tensor rank mismatch",
            "processcvu tiled tensor tile shape invalid")) {
      return false;
    }
    if (tensorsemantics::find_shape_axis(out->shape, SIMA_EV_AXIS_C) >= 0) {
      out->layout.tiled.flags = processcvu_tiled_encoding_flags_local(encoding);
    }
    if (size_bytes != 0U) {
      out->storage.nbytes = size_bytes;
      return true;
    }
    std::uint64_t bytes = 0;
    if (!processcvu_compute_tiled_storage_nbytes_local(*out, encoding, &bytes))
      return false;
    out->storage.nbytes = bytes;
    return true;
  }
  if (!tensorsemantics::fill_shape_desc(shape, normalized_layout, &out->shape, error_message,
                                        "processcvu shape desc requires output storage",
                                        "processcvu shape rank invalid",
                                        "processcvu shape dim invalid")) {
    return false;
  }
  std::uint32_t dtype = 0;
  if (!processcvu_dtype_token_to_ev_local(dtype_token, &dtype)) {
    if (error_message) {
      *error_message = "processcvu tiled tensor dtype invalid";
    }
    return false;
  }
  out->dtype = dtype;
  out->layout_kind = SIMA_EV_LAYOUT_TILED;
  out->storage.nbytes = size_bytes;
  for (std::size_t i = 0; i < tile_shape.size(); ++i) {
    out->layout.tiled.tile_sizes[i] = tile_shape[i];
  }
  out->layout.tiled.tile_align_bytes = tile_align_bytes;
  if (tensorsemantics::find_shape_axis(out->shape, SIMA_EV_AXIS_C) >= 0) {
    out->layout.tiled.flags = processcvu_tiled_encoding_flags_local(encoding);
  } else {
    out->layout.tiled.flags = SIMA_EV_TILED_FLAG_NONE;
  }
  if (size_bytes == 0U) {
    std::uint64_t bytes = 0;
    if (!processcvu_compute_tiled_storage_nbytes_local(*out, encoding, &bytes)) {
      if (error_message)
        *error_message = "model tiled tensor physical storage size invalid";
      return false;
    }
    out->storage.nbytes = bytes;
  }
  return true;
}

std::vector<std::int64_t> processcvu_tensor_shape_local(const MpkTensorContract& tensor) {
  if (!tensor.logical_shape.empty()) {
    return tensor.logical_shape;
  }
  return tensor.mpk_shape;
}

std::string processcvu_tensor_dtype_local(const MpkTensorContract& tensor,
                                          const std::string& fallback = std::string()) {
  if (!fallback.empty()) {
    return fallback;
  }
  if (!tensor.logical_dtype.empty()) {
    return tensor.logical_dtype;
  }
  return tensor.dtype;
}

std::string processcvu_tensor_layout_token_local(const std::vector<std::int64_t>& shape,
                                                 const std::string& fallback = std::string()) {
  const std::string normalized = tensorsemantics::normalize_layout_token(fallback);
  if (!normalized.empty()) {
    return normalized;
  }
  if (shape.size() == 2U) {
    return "HW";
  }
  if (shape.size() == 3U || shape.size() == 4U) {
    return "HWC";
  }
  return {};
}

} // namespace

bool processcvu_build_stage_tensor_descs_local(const StageStaticSpec& stage,
                                               ProcessCvuStagePayload* cfg,
                                               std::string* error_message) {
  if (!cfg) {
    if (error_message) {
      *error_message = "processcvu stage tensor descs require config output";
    }
    return false;
  }
  const auto& payload = stage.processcvu;
  if (!payload.input_tensors.empty() && !payload.output_tensors.empty()) {
    cfg->input_tensors = payload.input_tensors;
    cfg->output_tensors = payload.output_tensors;
    cfg->num_in_tensor = static_cast<int32_t>(cfg->input_tensors.size());
    return true;
  }
  if (!payload.input_tensors.empty() || !payload.output_tensors.empty()) {
    if (error_message)
      *error_message = "processcvu stage has incomplete authored tensor ports";
    return false;
  }
  cfg->input_tensors.clear();
  cfg->output_tensors.clear();
  for (const auto& logical : stage.logical_inputs) {
    sima_ev_tensor_desc desc{};
    if (!processcvu_build_dense_tensor_desc_local(logical.shape, logical.dtype, logical.layout,
                                                  logical.size_bytes, &desc, error_message)) {
      return false;
    }
    cfg->input_tensors.emplace_back(desc, logical.dtype);
  }
  for (const auto& logical : stage.logical_outputs) {
    sima_ev_tensor_desc desc{};
    if (!processcvu_build_dense_tensor_desc_local(logical.shape, logical.dtype, logical.layout,
                                                  logical.size_bytes, &desc, error_message)) {
      return false;
    }
    cfg->output_tensors.emplace_back(desc, logical.dtype);
  }
  if (cfg->input_tensors.empty() || cfg->output_tensors.empty()) {
    if (error_message) {
      *error_message = "processcvu stage tensor descs missing logical tensors";
    }
    return false;
  }
  cfg->num_in_tensor = static_cast<int32_t>(cfg->input_tensors.size());
  return true;
}

bool processcvu_build_graph_io_tensor_descs_local(const std::string& graph_name,
                                                  const GraphProcessCvuIoData& io,
                                                  ProcessCvuStagePayload* cfg,
                                                  std::string* error_message) {
  if (!cfg) {
    if (error_message) {
      *error_message = "processcvu graph IO tensor descs require config output";
    }
    return false;
  }
  cfg->input_tensors.clear();
  cfg->output_tensors.clear();
  const std::string canonical = processcvu_canonical_graph_name_local(graph_name);
  GraphTiledChannelEncoding resolved_encoding;
  if (!resolve_graph_processcvu_tiled_channel_encoding(canonical, io.c16_packed, io.cblock,
                                                       &resolved_encoding, error_message)) {
    return false;
  }
  const bool input_is_tiled =
      canonical == "detessellate" || canonical == "detesscast" || canonical == "detessdequant";
  const bool output_is_tiled = canonical == "tessellate" || canonical == "casttess" ||
                               canonical == "quanttess" || canonical == "quantizetessellate";
  const bool native_visual_graph = processcvu_is_native_visual_graph_local(canonical);
  if (native_visual_graph && io.input_tensors.size() != io.output_tensors.size()) {
    cfg->num_in_tensor = static_cast<int32_t>(io.input_tensors.size());
    return true;
  }
  if (io.input_tensors.empty() || io.output_tensors.empty()) {
    if (error_message) {
      *error_message = "processcvu graph IO missing tensors";
    }
    return false;
  }
  if (io.input_tensors.size() != io.output_tensors.size()) {
    if (error_message) {
      *error_message = "processcvu graph IO tensor count mismatch";
    }
    return false;
  }

  auto pick_slice_shape = [&](std::size_t index) -> std::vector<int> {
    if (io.slice_shapes.empty()) {
      return {};
    }
    return io.slice_shapes[io.slice_shapes.size() == 1U ? 0U : index];
  };

  const std::uint32_t output_tile_align =
      cfg->byte_align > 0 ? static_cast<std::uint32_t>(cfg->byte_align) : 0U;
  // Only DDQ historically defaults to 16-byte tile alignment. Detess and
  // other model graphs retain tight packing; unsupported native layouts fail.
  const std::uint32_t input_tile_align = canonical == "detessdequant" ? 16U : 0U;
  for (std::size_t i = 0; i < io.input_tensors.size(); ++i) {
    const auto& input = io.input_tensors[i];
    const auto& output = io.output_tensors[i];
    const auto input_shape = processcvu_tensor_shape_local(input);
    const auto output_shape = processcvu_tensor_shape_local(output);
    const std::string input_dtype = processcvu_tensor_dtype_local(input, io.canonical_input_dtype);
    const std::string output_dtype =
        processcvu_tensor_dtype_local(output, io.canonical_output_dtype);
    const std::string input_layout =
        processcvu_tensor_layout_token_local(input_shape, prepared_input_layout_token_local(*cfg));
    const std::string output_layout = processcvu_tensor_layout_token_local(
        output_shape, prepared_output_layout_token_local(*cfg));
    sima_ev_tensor_desc input_desc{};
    sima_ev_tensor_desc output_desc{};
    if (input_is_tiled) {
      std::vector<std::int64_t> tile_shape;
      if (!processcvu_normalize_tile_shape_local(input_shape, pick_slice_shape(i), &tile_shape,
                                                 error_message) ||
          !processcvu_build_tiled_tensor_desc_local(
              input_shape, tile_shape, input_dtype, input_layout, input_tile_align,
              input.size_bytes, resolved_encoding, &input_desc, error_message)) {
        return false;
      }
    } else if (!processcvu_build_dense_tensor_desc_local(input_shape, input_dtype, input_layout,
                                                         input.size_bytes, &input_desc,
                                                         error_message)) {
      return false;
    }
    if (output_is_tiled) {
      std::vector<std::int64_t> tile_shape;
      if (!processcvu_normalize_tile_shape_local(output_shape, pick_slice_shape(i), &tile_shape,
                                                 error_message) ||
          !processcvu_build_tiled_tensor_desc_local(
              output_shape, tile_shape, output_dtype, output_layout, output_tile_align,
              output.size_bytes, resolved_encoding, &output_desc, error_message)) {
        return false;
      }
    } else if (!processcvu_build_dense_tensor_desc_local(output_shape, output_dtype, output_layout,
                                                         output.size_bytes, &output_desc,
                                                         error_message)) {
      return false;
    }
    cfg->input_tensors.emplace_back(input_desc, input_dtype);
    cfg->output_tensors.emplace_back(output_desc, output_dtype);
  }
  cfg->num_in_tensor = static_cast<int32_t>(cfg->input_tensors.size());
  return true;
}

const LogicalInputStaticSpec*
processcvu_find_logical_input_by_index_local(const StageStaticSpec& stage, int logical_index) {
  const auto it = std::find_if(stage.logical_inputs.begin(), stage.logical_inputs.end(),
                               [&](const LogicalInputStaticSpec& logical) {
                                 return logical.logical_index == logical_index;
                               });
  return it == stage.logical_inputs.end() ? nullptr : &*it;
}

bool complete_kernel_stage_from_graph_io_local(const StageStaticSpec& original_stage,
                                               const MpkGraphNode& graph_node,
                                               const GraphProcessCvuIoData& io,
                                               StageStaticSpec* out, std::string* error_message) {
  if (!out || io.input_tensors.empty() || io.output_tensors.empty()) {
    if (error_message)
      *error_message = "graph numerical IO requires nonempty tensors";
    return false;
  }
  *out = original_stage;
  auto& payload = out->processcvu;
  payload.graph_name = processcvu_canonical_graph_name_local(
      !payload.graph_name.empty()
          ? payload.graph_name
          : (!payload.graph_family.empty() ? payload.graph_family : graph_node.canonical_op));
  if (!processcvu_build_graph_io_tensor_descs_local(payload.graph_name, io, &payload,
                                                    error_message))
    return false;
  return complete_graph_kernel_contract(
      out, io.input_tensors, io.output_tensors, io.canonical_input_dtype, io.canonical_output_dtype,
      io.input_slot_name, io.runtime_output_slot_names, error_message);
}

std::optional<std::filesystem::path>
discover_pack_root_from_model_path_local(const std::string& model_path) {
  if (model_path.empty()) {
    return std::nullopt;
  }
  std::filesystem::path path(model_path);
  std::error_code ec;
  auto inspect = [&](std::filesystem::path candidate) -> std::optional<std::filesystem::path> {
    if (candidate.empty()) {
      return std::nullopt;
    }
    auto cur = candidate;
    if (!std::filesystem::is_directory(cur, ec)) {
      cur = cur.parent_path();
    }
    while (!cur.empty()) {
      if (std::filesystem::exists(cur / "etc" / "mpk.json", ec) ||
          std::filesystem::exists(cur / "mpk.json", ec)) {
        return cur;
      }
      const auto parent = cur.parent_path();
      if (parent == cur) {
        break;
      }
      cur = parent;
    }
    return std::nullopt;
  };
  if (auto found = inspect(path); found.has_value()) {
    return found;
  }
  if (path.has_parent_path()) {
    auto parent = path.parent_path();
    const auto leaf = upper_copy_local(parent.filename().string());
    if ((leaf == "BIN" || leaf == "ETC") && !parent.parent_path().empty()) {
      if (auto found = inspect(parent.parent_path()); found.has_value()) {
        return found;
      }
      return parent.parent_path();
    }
    return parent;
  }
  return std::nullopt;
}

bool checked_add_u64_local(const std::uint64_t a, const std::uint64_t b, std::uint64_t* out) {
  if (!out) {
    return false;
  }
  if (a > (std::numeric_limits<std::uint64_t>::max() - b)) {
    return false;
  }
  *out = a + b;
  return true;
}

bool graph_node_has_member_local(const MpkGraphNode& node, const std::string& member_name) {
  return std::find(node.member_node_ids.begin(), node.member_node_ids.end(), member_name) !=
         node.member_node_ids.end();
}

bool graph_kernel_argument_matches_tensor_local(const MpkTensorContract& tensor,
                                                const std::string& value) {
  if (value.empty()) {
    return false;
  }
  return (!tensor.name.empty() && tensor.name == value) ||
         (!tensor.segment_name.empty() && tensor.segment_name == value);
}

std::string
graph_kernel_input_slot_name_local(const MpkGraphNode& node,
                                   const std::vector<MpkTensorContract>& input_tensors) {
  if (input_tensors.empty()) {
    return {};
  }
  for (const auto& field : node.kernel_contract.fields) {
    if (field.kind != MpkGraphKernelFieldKind::Argument || field.name == "metadata") {
      continue;
    }
    for (const auto& tensor : input_tensors) {
      if (graph_kernel_argument_matches_tensor_local(tensor, field.value)) {
        return field.name;
      }
    }
  }
  return {};
}

std::vector<std::string>
graph_kernel_output_slot_names_local(const MpkGraphNode& node,
                                     const std::vector<MpkTensorContract>& output_tensors) {
  std::vector<std::string> names;
  names.reserve(output_tensors.size());
  std::set<std::string> used_slots;
  for (const auto& tensor : output_tensors) {
    std::string slot_name;
    for (const auto& field : node.kernel_contract.fields) {
      if (field.kind != MpkGraphKernelFieldKind::Argument || field.name == "metadata" ||
          used_slots.find(field.name) != used_slots.end()) {
        continue;
      }
      if (graph_kernel_argument_matches_tensor_local(tensor, field.value)) {
        slot_name = field.name;
        break;
      }
    }
    if (slot_name.empty() && output_tensors.size() == 1U) {
      // Do not guess a kernel slot name when the graph contract does not
      // prove which argument owns this output tensor. The prepared typed
      // contract still carries the legacy runtime slot as a fallback.
    }
    if (!slot_name.empty()) {
      used_slots.insert(slot_name);
    }
    names.push_back(std::move(slot_name));
  }
  return names;
}

bool graph_node_tensor_name_matches_local(const MpkTensorContract& tensor,
                                          const std::string& value) {
  if (value.empty()) {
    return false;
  }
  return (!tensor.name.empty() && tensor.name == value) ||
         (!tensor.segment_name.empty() && tensor.segment_name == value);
}

std::vector<MpkTensorContract>
graph_node_select_tensors_local(const std::vector<MpkTensorContract>& tensors,
                                const std::vector<std::string>& node_names) {
  if (tensors.empty() || node_names.empty()) {
    return {};
  }
  std::vector<MpkTensorContract> selected;
  selected.reserve(std::min(tensors.size(), node_names.size()));
  for (const auto& node_name : node_names) {
    auto it = std::find_if(tensors.begin(), tensors.end(), [&](const MpkTensorContract& tensor) {
      return graph_node_tensor_name_matches_local(tensor, node_name);
    });
    if (it != tensors.end()) {
      selected.push_back(*it);
    }
  }
  return selected;
}

std::vector<MpkTensorContract>
graph_node_select_tensors_from_kernel_fields_local(const std::vector<MpkTensorContract>& tensors,
                                                   const MpkGraphNode& node) {
  if (tensors.empty()) {
    return {};
  }
  std::vector<MpkTensorContract> selected;
  selected.reserve(tensors.size());
  std::set<std::string> used_tensor_names;
  for (const auto& field : node.kernel_contract.fields) {
    if (field.kind != MpkGraphKernelFieldKind::Argument || field.name == "metadata" ||
        field.value.empty()) {
      continue;
    }
    auto it = std::find_if(tensors.begin(), tensors.end(), [&](const MpkTensorContract& tensor) {
      return graph_node_tensor_name_matches_local(tensor, field.value);
    });
    if (it == tensors.end()) {
      continue;
    }
    const std::string key =
        !it->name.empty() ? it->name : (!it->segment_name.empty() ? it->segment_name : field.value);
    if (!used_tensor_names.insert(key).second) {
      continue;
    }
    selected.push_back(*it);
  }
  return selected;
}

bool complete_kernel_stage_from_graph_local(const StageStaticSpec& original_stage,
                                            const MpkContract& contract,
                                            const MpkGraphNode& graph_node, StageStaticSpec* out,
                                            std::string* error_message) {
  const auto* stage = get_stage_io_contract(contract, graph_node.name);
  if (!stage) {
    if (error_message) {
      *error_message = "graph processcvu stage missing IO contract";
    }
    return false;
  }
  GraphProcessCvuIoData io;
  const std::string canonical = processcvu_canonical_graph_name_local(
      !original_stage.processcvu.graph_family.empty() ? original_stage.processcvu.graph_family
                                                      : original_stage.processcvu.graph_name);
  if (canonical == "detesscast" && stage->input_tensors.size() > 1U &&
      !stage->output_tensors.empty()) {
    io.input_tensors =
        graph_node_select_tensors_local(stage->input_tensors, graph_node.input_tensor_names);
    io.output_tensors =
        graph_node_select_tensors_local(stage->output_tensors, graph_node.output_tensor_names);
    if (io.input_tensors.empty()) {
      io.input_tensors =
          graph_node_select_tensors_from_kernel_fields_local(stage->input_tensors, graph_node);
    }
    if (io.output_tensors.empty()) {
      io.output_tensors =
          graph_node_select_tensors_from_kernel_fields_local(stage->output_tensors, graph_node);
    }
    if (io.input_tensors.size() != 1U || io.output_tensors.size() != 1U) {
      if (error_message) {
        *error_message =
            "graph detesscast stage did not resolve a single tensor pair for graph node";
      }
      return false;
    }
  } else {
    io.input_tensors = stage->input_tensors;
    io.output_tensors = stage->output_tensors;
  }
  io.canonical_input_dtype = stage->canonical_input_dtype;
  io.canonical_output_dtype = stage->canonical_output_dtype;
  io.c16_packed =
      (stage->has_align_c16 && stage->align_c16) || (stage->has_cblock && stage->cblock);
  io.cblock = stage->has_cblock && stage->cblock;
  if (!stage->slice_shape.empty()) {
    io.slice_shapes.push_back(
        std::vector<int>(stage->slice_shape.begin(), stage->slice_shape.end()));
  }
  io.input_slot_name = graph_kernel_input_slot_name_local(graph_node, io.input_tensors);
  io.runtime_output_slot_names =
      graph_kernel_output_slot_names_local(graph_node, io.output_tensors);
  return complete_kernel_stage_from_graph_io_local(original_stage, graph_node, io, out,
                                                   error_message);
}

std::string resolve_exact_processcvu_graph_stage_key_local(const StageStaticSpec* stage) {
  if (!stage || stage->payload_kind != StagePayloadKind::ProcessCvu) {
    return {};
  }
  return stage->processcvu.exact_stage_name_or_id;
}

const StageStaticSpec*
find_original_stage_for_transformed_key_local(const SimaPluginStaticManifest& original_manifest,
                                              const NameTransform& name_transform,
                                              const StageStaticSpec& transformed_stage) {
  const std::string transformed_stage_key = stage_key_from_stage_spec_local(transformed_stage);
  if (transformed_stage_key.empty()) {
    return nullptr;
  }
  for (const auto& candidate : original_manifest.stages) {
    const std::string original_stage_key = stage_key_from_stage_spec_local(candidate);
    if (original_stage_key.empty()) {
      continue;
    }
    if (original_stage_key == transformed_stage_key ||
        apply_name_transform(name_transform, original_stage_key) == transformed_stage_key) {
      return &candidate;
    }
  }
  return nullptr;
}

std::vector<std::string> graph_stage_candidate_keys_local(const StageStaticSpec& transformed_stage,
                                                          const StageStaticSpec* original_stage,
                                                          const std::string& exact_stage_key = {}) {
  std::vector<std::string> keys;
  auto append = [&](const std::string& key) {
    if (key.empty()) {
      return;
    }
    if (std::find(keys.begin(), keys.end(), key) == keys.end()) {
      keys.push_back(key);
    }
  };
  append(exact_stage_key);
  if (original_stage) {
    append(stage_key_from_stage_spec_local(*original_stage));
  }
  append(stage_key_from_stage_spec_local(transformed_stage));
  return keys;
}

std::vector<const MpkGraphNode*>
find_graph_nodes_matching_stage_keys_local(const std::vector<MpkGraphNode>& nodes,
                                           const std::vector<std::string>& stage_keys,
                                           const std::string& canonical_op) {
  std::vector<const MpkGraphNode*> matches;
  const std::string wanted = upper_copy_local(canonical_op);
  for (const auto& node : nodes) {
    if (!wanted.empty() && upper_copy_local(node.canonical_op) != wanted) {
      continue;
    }
    bool matched = stage_keys.empty();
    for (const auto& key : stage_keys) {
      if (graph_node_matches_stage_key_local(node, key)) {
        matched = true;
        break;
      }
    }
    if (matched) {
      matches.push_back(&node);
    }
  }
  std::sort(matches.begin(), matches.end(), [](const MpkGraphNode* a, const MpkGraphNode* b) {
    if (a->sequence != b->sequence) {
      return a->sequence < b->sequence;
    }
    return a->name < b->name;
  });
  return matches;
}

std::optional<MpkContract>
discover_graph_contract_local(const SimaPluginStaticManifest& transformed_manifest,
                              const std::optional<SimaPluginStaticManifest>& original_manifest,
                              const std::vector<PipelineElementSpec>& pipeline_elements,
                              const std::vector<std::string>& model_source_paths,
                              std::filesystem::path* pack_root_out, std::string* error_message) {
  if (pack_root_out) {
    *pack_root_out = std::filesystem::path();
  }
  if (error_message) {
    error_message->clear();
  }

  std::filesystem::path pack_root;
  std::string load_error;
  std::optional<MpkContract> contract_opt;

  if (original_manifest.has_value()) {
    contract_opt =
        load_graph_contract_from_manifest_local(*original_manifest, &pack_root, &load_error);
  }
  if (!contract_opt.has_value()) {
    std::string transformed_error;
    contract_opt = load_graph_contract_from_manifest_local(transformed_manifest, &pack_root,
                                                           &transformed_error);
    if (!contract_opt.has_value() && !transformed_error.empty()) {
      load_error = transformed_error;
    }
  }
  if (!contract_opt.has_value()) {
    std::string pipeline_error;
    contract_opt = load_graph_contract_from_pipeline_elements_local(pipeline_elements, &pack_root,
                                                                    &pipeline_error);
    if (!contract_opt.has_value() && !pipeline_error.empty()) {
      load_error = pipeline_error;
    }
  }
  if (!contract_opt.has_value()) {
    std::string source_error;
    contract_opt =
        load_graph_contract_from_model_sources_local(model_source_paths, &pack_root, &source_error);
    if (!contract_opt.has_value() && !source_error.empty()) {
      load_error = source_error;
    }
  }
  if (!contract_opt.has_value()) {
    if (error_message && !load_error.empty()) {
      *error_message = load_error;
    }
    return std::nullopt;
  }
  if (pack_root_out) {
    *pack_root_out = pack_root;
  }
  return contract_opt;
}

std::optional<MpkContract>
load_graph_contract_from_manifest_local(const SimaPluginStaticManifest& manifest,
                                        std::filesystem::path* pack_root_out,
                                        std::string* error_message) {
  for (const auto& stage : manifest.stages) {
    if (!stage_is_processmla_local(stage)) {
      continue;
    }
    prepared_runtime_debug_log_local("graph overlay inspect mla stage=%s model_path=%s",
                                     stage_key_from_stage_spec_local(stage).c_str(),
                                     stage.processmla.model_path.c_str());
    if (stage.processmla.model_path.empty()) {
      continue;
    }
    const auto pack_root = discover_pack_root_from_model_path_local(stage.processmla.model_path);
    if (!pack_root.has_value()) {
      prepared_runtime_debug_log_local(
          "graph overlay pack_root discovery failed stage=%s model_path=%s",
          stage_key_from_stage_spec_local(stage).c_str(), stage.processmla.model_path.c_str());
      continue;
    }
    std::string load_error;
    auto loaded = static_contract::MpkDecoder::load_package_root(*pack_root, &load_error);
    if (loaded.has_value()) {
      if (pack_root_out) {
        *pack_root_out = *pack_root;
      }
      return std::move(loaded->contract);
    }
    prepared_runtime_debug_log_local("graph overlay pack_root load failed root=%s error=%s",
                                     pack_root->string().c_str(), load_error.c_str());
    if (error_message && error_message->empty()) {
      *error_message = load_error;
    }
  }
  if (error_message && error_message->empty()) {
    *error_message = "unable to discover pack root for graph-native prepared runtime overlay";
  }
  return std::nullopt;
}

std::optional<MpkContract> load_graph_contract_from_pipeline_elements_local(
    const std::vector<PipelineElementSpec>& pipeline_elements, std::filesystem::path* pack_root_out,
    std::string* error_message) {
  for (const auto& element : pipeline_elements) {
    const std::string plugin = upper_copy_local(element.plugin);
    if (plugin.find("PROCESSMLA") == std::string::npos) {
      continue;
    }
    if (!element.model_path_property.has_value() || element.model_path_property->empty()) {
      prepared_runtime_debug_log_local(
          "graph overlay inspect pipeline mla element=%s stage_id=%s model_path=<empty>",
          element.element_name.c_str(), element.stage_id.c_str());
      continue;
    }
    prepared_runtime_debug_log_local(
        "graph overlay inspect pipeline mla element=%s stage_id=%s model_path=%s",
        element.element_name.c_str(), element.stage_id.c_str(),
        element.model_path_property->c_str());
    const auto pack_root = discover_pack_root_from_model_path_local(*element.model_path_property);
    if (!pack_root.has_value()) {
      prepared_runtime_debug_log_local(
          "graph overlay pipeline pack_root discovery failed element=%s model_path=%s",
          element.element_name.c_str(), element.model_path_property->c_str());
      continue;
    }
    std::string load_error;
    auto loaded = static_contract::MpkDecoder::load_package_root(*pack_root, &load_error);
    if (loaded.has_value()) {
      if (pack_root_out) {
        *pack_root_out = *pack_root;
      }
      return std::move(loaded->contract);
    }
    prepared_runtime_debug_log_local(
        "graph overlay pipeline pack_root load failed root=%s error=%s",
        pack_root->string().c_str(), load_error.c_str());
    if (error_message && error_message->empty()) {
      *error_message = load_error;
    }
  }
  if (error_message && error_message->empty()) {
    *error_message = "unable to discover pack root from pipeline element properties";
  }
  return std::nullopt;
}

std::optional<MpkContract>
load_graph_contract_from_model_sources_local(const std::vector<std::string>& model_source_paths,
                                             std::filesystem::path* pack_root_out,
                                             std::string* error_message) {
  for (const auto& source_path : model_source_paths) {
    if (source_path.empty()) {
      continue;
    }
    prepared_runtime_debug_log_local("graph overlay inspect model source_path=%s",
                                     source_path.c_str());
    try {
      internal::ModelPack pack(source_path);
      if (!pack.mpk_contract().has_value()) {
        prepared_runtime_debug_log_local(
            "graph overlay model source missing mpk contract source_path=%s", source_path.c_str());
        continue;
      }
      MpkContract contract = *pack.mpk_contract();
      std::filesystem::path pack_root;
      if (!pack.etc_dir().empty()) {
        pack_root = std::filesystem::path(pack.etc_dir()).parent_path();
      } else if (!contract.mpk_json_path.empty()) {
        pack_root = std::filesystem::path(contract.mpk_json_path).parent_path();
        if (pack_root.filename() == "etc") {
          pack_root = pack_root.parent_path();
        }
      }
      if (pack_root_out) {
        *pack_root_out = pack_root;
      }
      return contract;
    } catch (const std::exception& e) {
      prepared_runtime_debug_log_local(
          "graph overlay model source load failed source_path=%s error=%s", source_path.c_str(),
          e.what());
      if (error_message && error_message->empty()) {
        *error_message = e.what();
      }
    }
  }
  if (error_message && error_message->empty()) {
    *error_message = "unable to discover pack root from model source paths";
  }
  return std::nullopt;
}

bool image_publication_matches_local(
    const std::shared_ptr<const ::simaai::scheduler::client::TensorStageImagePublication>& authored,
    const std::shared_ptr<const ::simaai::scheduler::client::TensorStageImagePublication>&
        checked) {
  if (!authored || !checked)
    return !authored && !checked;
  if (authored->format != checked->format || authored->width != checked->width ||
      authored->height != checked->height || authored->planes.size() != checked->planes.size())
    return false;
  for (std::size_t i = 0; i < authored->planes.size(); ++i) {
    const auto& a = authored->planes[i];
    const auto& b = checked->planes[i];
    if (a.backend_output_index != b.backend_output_index || a.byte_offset != b.byte_offset ||
        a.stride != b.stride || a.visible_rows != b.visible_rows ||
        a.visible_row_bytes != b.visible_row_bytes)
      return false;
  }
  return true;
}

::simaai::scheduler::client::Status lower_kernel_stage_local(StageStaticSpec* stage) {
  namespace rt = ::simaai::scheduler::client;
  namespace gs = ::simaai::scheduler::client;
  const auto invalid = [](std::string field, std::string message) -> rt::Status {
    return rt::Error{rt::ErrorCode::InvalidRequest, std::move(field), std::move(message)};
  };
  auto& payload = stage->processcvu;
  if (payload.image_transform && payload.image_fps_num > 0 && payload.image_fps_den <= 0)
    return invalid("stage.framerate", "a known image rate requires a positive denominator");
  if (payload.requested_run_target.empty())
    payload.requested_run_target = payload.run_target;
  if (payload.run_target.empty())
    payload.run_target = "AUTO";
  if (payload.resolved_exec_backend.empty())
    payload.resolved_exec_backend = "EVXX";
  payload.graph_name = processcvu_canonical_graph_name_local(
      payload.graph_name.empty() ? payload.graph_family : payload.graph_name);
  if (!payload.canonical_contract || payload.graph_name.empty() ||
      payload.default_input_name.empty() || payload.primary_output_name.empty())
    return invalid("stage.contract", "Core numerical stage has incomplete semantic contract");
  const bool image = payload.graph_name == "preproc";
  std::string error;
  if (image && !processcvu_build_stage_tensor_descs_local(*stage, &payload, &error))
    return invalid("stage.tensors", error);
  payload.num_in_tensor = static_cast<int>(payload.input_tensors.size());
  std::shared_ptr<KernelStageRecipe> recipe;
  if (image) {
    if (payload.preproc_single_output_handoff &&
        std::find(payload.default_output_names.begin(), payload.default_output_names.end(),
                  payload.primary_output_name) == payload.default_output_names.end())
      return invalid("stage.output", "Preproc primary handoff is not a runtime output");
    auto checked = kernel_lowering::PreprocFrameAdapter::check_policy(payload);
    if (const auto* failure = std::get_if<rt::Error>(&checked))
      return *failure;
    auto policy =
        std::get<std::shared_ptr<const kernel_lowering::PreprocFrameAdapter::CheckedPolicy>>(
            std::move(checked));
    recipe = std::make_shared<KernelStageRecipe>();
    recipe->target = rt::Hardware::EV74;
    gs::TensorStageMember member;
    member.executable = kernel_lowering::PreprocFrameAdapter::checked_executable(*policy);
    recipe->image_publication =
        kernel_lowering::PreprocFrameAdapter::checked_image_publication(*policy);
    const auto output_count =
        recipe->image_publication ? recipe->image_publication->planes.size() : 1;
    for (std::size_t i = 0; i < output_count; ++i)
      member.outputs.push_back({i, 0, gs::TensorStagePortSource::Output});
    recipe->members.push_back(std::move(member));
    recipe->frame_adapter_factory =
        [policy = std::move(policy)]() -> rt::Result<std::shared_ptr<gs::TensorStageFrameAdapter>> {
      auto instance = kernel_lowering::PreprocFrameAdapter::instantiate(policy);
      if (const auto* failure = std::get_if<rt::Error>(&instance))
        return *failure;
      return std::shared_ptr<gs::TensorStageFrameAdapter>(
          std::get<std::shared_ptr<kernel_lowering::PreprocFrameAdapter>>(std::move(instance)));
    };
  } else {
    auto checked = kernel_lowering::visual_family(payload.graph_name)
                       ? kernel_lowering::check_visual_stage(*stage)
                       : kernel_lowering::check_transform_stage(*stage);
    if (const auto* failure = std::get_if<rt::Error>(&checked))
      return *failure;
    recipe = std::get<std::shared_ptr<KernelStageRecipe>>(std::move(checked));
  }
  if (!image_publication_matches_local(stage->image_publication, recipe->image_publication))
    return invalid("stage.image_publication",
                   "authored image publication differs from the checked output contract");
  recipe->image_publication = stage->image_publication;
  if (image && !recipe->image_publication) {
    const LogicalTensorStaticSpec* selected = nullptr;
    for (const auto& logical : stage->logical_outputs) {
      const bool named = logical.logical_name == payload.primary_output_name ||
                         logical.backend_name == payload.primary_output_name ||
                         std::any_of(stage->output_order.begin(), stage->output_order.end(),
                                     [&](const auto& route) {
                                       return route.cm_output_name == payload.primary_output_name &&
                                              route.logical_output_index == logical.logical_index;
                                     });
      if (!named)
        continue;
      if (selected)
        return invalid("stage.output", "Preproc primary publication is ambiguous");
      selected = &logical;
    }
    if (!selected && stage->logical_outputs.size() == 1)
      selected = &stage->logical_outputs.front();
    if (!selected || selected->backend_output_index < 0)
      return invalid("stage.output", "Preproc selected publication has no backend output identity");
    // A selected runtime alternative may retain backend index one even when it
    // is the sole exposed output. Route by authored identity, never vector slot.
    recipe->members.front().outputs.front().tensor =
        static_cast<std::size_t>(selected->backend_output_index);
  }
  stage->native_kernel = std::move(recipe);
  return {};
}

::simaai::scheduler::client::Status
apply_kernel_publication_views(const StageStaticSpec& stage,
                               ::simaai::scheduler::client::TensorStagePublication& publication,
                               bool include_identity_views = false) {
  namespace rt = ::simaai::scheduler::client;
  namespace gs = ::simaai::scheduler::client;
  const auto invalid = [](std::string field, std::string message) -> rt::Status {
    return rt::Error{rt::ErrorCode::InvalidRequest, std::move(field), std::move(message)};
  };
  if (!stage.native_kernel)
    return {};
  std::set<std::size_t> layouts;
  for (const auto& member : stage.native_kernel->members) {
    auto inspected = ::simaai::scheduler::client::inspect_kernel(member.executable);
    if (const auto* failure = std::get_if<rt::Error>(&inspected))
      return *failure;
    const auto& checked = std::get<rt::KernelInspection>(inspected);
    for (std::size_t i = 0; i < member.outputs.size(); ++i) {
      const auto backend = member.outputs[i].tensor;
      if (std::none_of(stage.logical_outputs.begin(), stage.logical_outputs.end(),
                       [&](const auto& output) {
                         return output.backend_output_index == static_cast<int>(backend);
                       }))
        continue;
      if (i >= checked.outputs().size())
        return invalid("stage.publication", "checked producer output is missing");
      const auto& spec = checked.outputs()[i];
      if (layouts.insert(backend).second) {
        gs::TensorStagePhysicalView view{static_cast<int>(backend), spec.dtype, spec.layout};
        // Inspection supplies the execution dtype/layout, never permission to
        // reshape. Only the originating semantic/compiler contract can author
        // a publication relation, selected by backend identity rather than a
        // logical vector position or the current number of exposed outputs.
        std::optional<PublicationViewRelation> relation;
        for (const auto& logical : stage.logical_outputs) {
          if (logical.backend_output_index != static_cast<int>(backend))
            continue;
          if (relation && *relation != logical.publication_relation)
            return invalid("stage.publication", "conflicting publication relations for output " +
                                                    std::to_string(backend));
          relation = logical.publication_relation;
          switch (logical.publication_relation) {
          case PublicationViewRelation::Identity:
            break;
          case PublicationViewRelation::SingletonAxes:
            view.relation = gs::TensorStagePublicationRelation::SingletonAxes;
            break;
          case PublicationViewRelation::PackedStorage:
            view.relation = gs::TensorStagePublicationRelation::PackedStorage;
            break;
          case PublicationViewRelation::BatchSlices:
            view.relation = gs::TensorStagePublicationRelation::BatchSlices;
            break;
          default:
            return invalid("stage.publication", "unknown publication relation");
          }
        }
        if (publication.output_caps && relation == PublicationViewRelation::PackedStorage &&
            kernel_tensor_requires_detessellation(spec)) {
          if (const auto* tiled = std::get_if<rt::MlaTiled>(&spec.layout)) {
            const char* dtype = spec.dtype == rt::DType::BFloat16 ? "BF16"
                                : spec.dtype == rt::DType::Int8   ? "INT8"
                                : spec.dtype == rt::DType::UInt8  ? "UINT8"
                                                                  : nullptr;
            if (!dtype || i >= checked.output_requirements().size())
              return invalid("stage.publication", "unsupported checked tiled output metadata");
            TessSpec tess;
            tess.format =
                std::string("MLA_TILED_") + dtype +
                (tiled->channels == rt::MlaTiled::Channels::Compact ? "_COMPACT" : "_PAD16");
            tess.slice_shape.assign(tiled->tile_shape.begin(), tiled->tile_shape.end());
            tess.axes.assign(tiled->axes.begin(), tiled->axes.end());
            tess.storage_bytes = checked.output_requirements()[i].storage_bytes;
            for (const auto& logical : stage.logical_outputs) {
              // PackedStorage also represents deliberately opaque byte vectors
              // (for example legacy Tess output). Annotate only the numerical
              // view, never reinterpret a raw-byte alias of the same carrier.
              if (logical.backend_output_index == static_cast<int>(backend) &&
                  same_numerical_tensor_view(spec.shape, tensorbuffer_dtype_from_token_local(dtype),
                                             logical.shape,
                                             tensorbuffer_dtype_from_token_local(logical.dtype)))
                set_tessellation_caps(publication.output_caps.get(), logical.logical_index, tess);
            }
          }
        }
        if (include_identity_views || (relation && *relation != PublicationViewRelation::Identity))
          publication.physical_views.push_back(std::move(view));
      }
    }
  }
  return {};
}

::simaai::scheduler::client::Status finalize_processcvu_publication(StageStaticSpec* stage) {
  namespace rt = ::simaai::scheduler::client;
  namespace gs = ::simaai::scheduler::client;
  const auto invalid = [](std::string field, std::string message) -> rt::Status {
    return rt::Error{rt::ErrorCode::InvalidRequest, std::move(field), std::move(message)};
  };
  auto recipe = std::make_shared<KernelStageRecipe>(*stage->native_kernel);
  const auto& payload = stage->processcvu;
  const bool image = bool(recipe->frame_adapter_factory);
  std::string error;
  auto contract = std::make_shared<simaai::gst::TensorBufferPublishContract>();
  if (!build_publish_contract_from_manifest_stage_local(*stage, contract.get(), &error))
    return invalid("stage.publication", error);
  attach_processcvu_dequant_qparams_to_publish_contract_local(*stage, payload, contract.get());
  gs::TensorStagePublication publication;
  publication.contract = contract;
  publication.output_name = payload.primary_output_name;
  GstCaps* sink = nullptr;
  if (image) {
    sink = gst_caps_new_empty_simple("video/x-raw");
    if (!payload.input_img_type.empty())
      gst_caps_set_simple(sink, "format", G_TYPE_STRING, payload.input_img_type.c_str(), nullptr);
  } else if (stage->input_bindings.size() == 1 && payload.input_tensors.size() <= 1 &&
             !processcvu_graph_family_uses_packed_input_transport_local(payload.graph_name)) {
    CapsTensorSpec input;
    if (!stage->logical_inputs.empty()) {
      input.dtype = stage->logical_inputs.front().dtype;
      input.shape = stage->logical_inputs.front().shape;
    }
    if (input.shape.empty() && !payload.input_tensors.empty())
      input.shape =
          shape_vector_from_ev_shape_local(payload.input_tensors.front().descriptor().shape);
    input.dtype = processcvu_caps_format_from_dtype_local(input.dtype, payload.input_dtype);
    if (!input.shape.empty())
      sink = processmla_make_tensor_caps_from_tensor_local(input, input.dtype, false);
  }
  publication.input_caps =
      std::shared_ptr<GstCaps>(sink ? sink : make_generic_tensor_set_caps_local(), gst_caps_unref);
  GstCaps* source = nullptr;
  if (recipe->image_publication) {
    const auto& published = *recipe->image_publication;
    source = gst_caps_new_simple("video/x-raw", "format", G_TYPE_STRING,
                                 gst_video_format_to_string(published.format), "width", G_TYPE_INT,
                                 gint(published.width), "height", G_TYPE_INT,
                                 gint(published.height), nullptr);
    if (payload.image_fps_num > 0)
      gst_caps_set_simple(source, "framerate", GST_TYPE_FRACTION, payload.image_fps_num,
                          payload.image_fps_den, nullptr);
    if (payload.image_transform && published.planes.size() > 1) {
      GstVideoColorimetry color{};
      color.range = payload.image_transform->color.range == ImageYuvRange::Full
                        ? GST_VIDEO_COLOR_RANGE_0_255
                        : GST_VIDEO_COLOR_RANGE_16_235;
      color.matrix = payload.image_transform->color.matrix == ImageColorMatrix::Bt601
                         ? GST_VIDEO_COLOR_MATRIX_BT601
                         : GST_VIDEO_COLOR_MATRIX_BT709;
      gchar* name = gst_video_colorimetry_to_string(&color);
      if (name) {
        gst_caps_set_simple(source, "colorimetry", G_TYPE_STRING, name, nullptr);
        g_free(name);
      }
      const auto& siting = payload.image_transform->color.output_siting;
      auto site = GST_VIDEO_CHROMA_SITE_UNKNOWN;
      if (siting[0] == ImageChromaSiting::Cosited)
        site = GstVideoChromaSite(site | GST_VIDEO_CHROMA_SITE_H_COSITED);
      if (siting[1] == ImageChromaSiting::Cosited)
        site = GstVideoChromaSite(site | GST_VIDEO_CHROMA_SITE_V_COSITED);
      if (site == GST_VIDEO_CHROMA_SITE_UNKNOWN)
        site = GST_VIDEO_CHROMA_SITE_NONE;
      gchar* chroma = gst_video_chroma_site_to_string(site);
      if (chroma) {
        gst_caps_set_simple(source, "chroma-site", G_TYPE_STRING, chroma, nullptr);
        g_free(chroma);
      }
    }
  }
  if (!source && !processcvu_primary_output_uses_packed_caps_from_payload_local(payload) &&
      contract->logical_outputs.size() == 1 && contract->output_order.size() <= 1) {
    const auto& logical = contract->logical_outputs.front();
    auto output = caps_tensor_from_publish_logical_output_local(logical);
    const auto found = std::find_if(
        stage->logical_outputs.begin(), stage->logical_outputs.end(),
        [&](const auto& value) { return value.logical_index == logical.logical_index; });
    const auto dtype = found == stage->logical_outputs.end()
                           ? (payload.out_dtype.empty() ? payload.output_dtype : payload.out_dtype)
                           : found->dtype;
    output.dtype = processcvu_caps_format_from_dtype_local(dtype, payload.output_dtype);
    if (!output.shape.empty()) {
      source = processmla_make_tensor_caps_from_tensor_local(output, output.dtype, false);
      processmla_add_shape_fields_to_caps_local(source, output);
    }
  }
  publication.output_caps = std::shared_ptr<GstCaps>(
      source ? source : make_generic_tensor_set_caps_local(), gst_caps_unref);
  if (auto failure = apply_kernel_publication_views(*stage, publication, true))
    return failure;
  if (recipe->image_publication) {
    auto image_view = std::make_shared<gs::TensorStageImagePublication>(*recipe->image_publication);
    for (auto& plane : image_view->planes) {
      const auto logical = std::find_if(
          stage->logical_outputs.begin(), stage->logical_outputs.end(), [&](const auto& value) {
            return value.backend_output_index == plane.backend_output_index;
          });
      if (logical == stage->logical_outputs.end() || logical->byte_offset < 0)
        return invalid("stage.image", "raw image plane lacks an exact publication route");
      const auto binding = std::find_if(
          stage->frame_layout->outputs.begin(), stage->frame_layout->outputs.end(),
          [&](const auto& candidate) {
            return candidate.tensor == static_cast<std::size_t>(logical->backend_output_index);
          });
      if (binding == stage->frame_layout->outputs.end())
        return invalid("stage.image", "raw image plane has no final allocation binding");
      plane.byte_offset = binding->span.byte_offset;
    }
    recipe->image_publication = std::move(image_view);
  }
  stage->tensor_publication =
      std::make_shared<const gs::TensorStagePublication>(std::move(publication));
  stage->image_publication = std::move(recipe->image_publication);
  return {};
}

} // namespace

bool resolve_graph_processcvu_tiled_channel_encoding(const std::string& canonical_graph_name,
                                                     const bool c16_packed, const bool cblock,
                                                     GraphTiledChannelEncoding* out,
                                                     std::string* error_message) {
  if (error_message) {
    error_message->clear();
  }
  if (!out) {
    if (error_message) {
      *error_message = "graph processcvu V2 tiled channel encoding requires output storage";
    }
    return false;
  }
  if (cblock && canonical_graph_name != "detessdequant") {
    if (error_message) {
      *error_message = "graph processcvu V2 CBlock16 encoding is supported only for detessdequant";
    }
    return false;
  }
  *out = cblock ? GraphTiledChannelEncoding::CBlock16
                : (c16_packed ? GraphTiledChannelEncoding::PaddedHwcC16
                              : GraphTiledChannelEncoding::CompactPixelMajor);
  return true;
}

bool build_kernel_stage_publication(const StageStaticSpec& stage,
                                    ::simaai::scheduler::client::TensorStagePublication* output,
                                    std::string* error) {
  if (output && stage.tensor_publication) {
    *output = *stage.tensor_publication;
    // A reusable checked recipe can serve differently named graph instances.
    // Only transport identity changes; preserve checked views/qparams and never
    // mutate the reusable publication or repeat numerical policy admission.
    const auto key = stage_key_from_stage_spec_local(stage);
    if (output->contract && output->contract->stage_key != key) {
      auto contract = std::make_shared<simaai::gst::TensorBufferPublishContract>(*output->contract);
      contract->stage_key = key;
      output->contract = std::move(contract);
    }
    return true;
  }

  if (!output) {
    if (error)
      *error = "native kernel publication output is null";
    return false;
  }
  if (stage.logical_outputs.empty()) {
    if (error)
      *error = "native kernel stage requires a public output";
    return false;
  }
  auto contract = std::make_shared<simaai::gst::TensorBufferPublishContract>();
  if (!build_publish_contract_from_manifest_stage_local(stage, contract.get(), error))
    return false;
  output->input_caps = std::shared_ptr<GstCaps>(
      gst_caps_new_empty_simple("application/vnd.simaai.tensor"), gst_caps_unref);
  output->output_caps =
      std::shared_ptr<GstCaps>(make_generic_tensor_set_caps_local(), gst_caps_unref);
  if ((stage.mla_program || stage.tvm_program) && contract->logical_outputs.size() == 1 &&
      contract->output_order.size() <= 1) {
    const auto tensor =
        caps_tensor_from_publish_logical_output_local(contract->logical_outputs.front());
    if (!tensor.dtype.empty() && !tensor.shape.empty()) {
      const auto format = processcvu_caps_format_from_dtype_local(tensor.dtype, {});
      gst_caps_set_simple(output->output_caps.get(), "format", G_TYPE_STRING, format.c_str(),
                          "dtype", G_TYPE_STRING, format.c_str(), nullptr);
      processmla_add_shape_fields_to_caps_local(output->output_caps.get(), tensor);
    }
  }
  if (stage.native_kernel && !stage.native_kernel->output_format.empty()) {
    gst_caps_set_simple(output->output_caps.get(), "format", G_TYPE_STRING,
                        stage.native_kernel->output_format.c_str(), nullptr);
  }
  output->output_name = stage.logical_outputs.front().logical_name;
  output->contract = std::move(contract);
  output->physical_views.clear();
  if (auto failure = apply_kernel_publication_views(stage, *output)) {
    if (error)
      *error = failure->message;
    return false;
  }
  return true;
}

::simaai::scheduler::client::Status complete_kernel_input_routes(SimaPluginStaticManifest* manifest,
                                                                 std::size_t stage_index) {
  const auto invalid = [](std::string message) -> ::simaai::scheduler::client::Status {
    return ::simaai::scheduler::client::Error{
        ::simaai::scheduler::client::ErrorCode::InvalidRequest, "stage.input_routes",
        std::move(message)};
  };
  if (!manifest || stage_index >= manifest->stages.size())
    return invalid("input route completion requires an existing manifest stage");
  auto& stage = manifest->stages[stage_index];
  auto bindings = stage.input_bindings;
  for (std::size_t index = 0; index < bindings.size(); ++index) {
    auto& binding = bindings[index];
    const int logical_index = binding.local_logical_input_index >= 0
                                  ? binding.local_logical_input_index
                                  : binding.sink_pad_index;
    const auto* logical = processcvu_find_logical_input_by_index_local(stage, logical_index);
    if (!logical)
      return invalid("logical input missing for binding");
    const auto resolved =
        edgecontract::resolve_edge_contract_for_binding(*manifest, stage_index, index);
    // Preserve the former preparation precedence: unresolved/external edges
    // keep their authored fields; only the same consumer logical view can fill
    // missing coordinates. Complete explicit overrides are never replaced.
    if (!resolved || resolved->consumer_logical_input != logical)
      continue;
    const auto* source = resolved->producer_logical_output;
    const auto* physical = resolved->producer_physical_output;
    const bool incomplete = binding.src_logical_output_index < 0 || binding.src_output_slot < 0 ||
                            binding.src_physical_output_index < 0 ||
                            binding.src_physical_size_bytes == 0 ||
                            binding.source_segment_name.empty();
    if (binding.src_logical_output_index < 0 && source)
      binding.src_logical_output_index = source->logical_index;
    if (binding.src_output_slot < 0 && source)
      binding.src_output_slot = source->output_slot;
    if (binding.src_physical_output_index < 0) {
      if (physical)
        binding.src_physical_output_index = physical->physical_index;
      else if (source)
        binding.src_physical_output_index = source->physical_index;
    }
    if (!binding.src_physical_size_bytes && physical)
      binding.src_physical_size_bytes = physical->size_bytes;
    if (!binding.src_physical_size_bytes && source)
      binding.src_physical_size_bytes = source->size_bytes;
    if (!binding.src_physical_byte_offset && incomplete) {
      if (logical->byte_offset > 0)
        binding.src_physical_byte_offset = logical->byte_offset;
      else if (source && source->byte_offset > 0)
        binding.src_physical_byte_offset = source->byte_offset;
      // Otherwise the relative offset remains zero. Unlike the removed CVU
      // read-request representation, the generic engine already adds the
      // consumer physical carrier's source_byte_offset exactly once.
    }
    if (binding.source_segment_name.empty() && source)
      binding.source_segment_name = source->segment_name;
    if (binding.source_segment_name.empty() && physical)
      binding.source_segment_name = physical->segment_name;
  }
  stage.input_bindings = std::move(bindings);
  return {};
}

::simaai::scheduler::client::Status
prepare_kernel_stage_recipes(SimaPluginStaticManifest* manifest,
                             const std::optional<SimaPluginStaticManifest>& original_manifest,
                             const std::vector<PipelineElementSpec>& pipeline_elements,
                             const std::vector<std::string>& model_source_paths,
                             const NameTransform& name_transform) {
  namespace rt = ::simaai::scheduler::client;
  const auto invalid = [](std::string message) -> rt::Status {
    return rt::Error{rt::ErrorCode::InvalidRequest, "stage.contract", std::move(message)};
  };
  if (!manifest)
    return invalid("canonical lowering requires a manifest");
  // Commit together: a failed cold check never leaves a partly admitted graph.
  auto candidate = *manifest;
  std::optional<MpkContract> graph;
  for (std::size_t stage_index = 0; stage_index < candidate.stages.size(); ++stage_index) {
    auto& stage = candidate.stages[stage_index];
    if (stage.native_kernel) {
      if (stage_is_processcvu_local(stage))
        if (auto failure = complete_kernel_input_routes(&candidate, stage_index))
          return failure;
      // Reusable recipes retain immutable checked policy only. Mutable image
      // and resolver state is instantiated once at the cold request boundary.
      continue;
    }
    if (!stage_is_processcvu_local(stage))
      continue;
    if (stage_is_graph_owned_local(stage)) {
      const auto canonical = processcvu_canonical_graph_name_local(
          stage.processcvu.graph_family.empty() ? stage.processcvu.graph_name
                                                : stage.processcvu.graph_family);
      const bool region = stage.input_bindings.size() > 1 || stage.logical_inputs.size() > 1 ||
                          stage.logical_outputs.size() > 1 || stage.output_order.size() > 1;
      if (canonical != "detessellate" && canonical != "detesscast" &&
          canonical != "detessdequant" && canonical != "dequantize" && !region) {
        if (!graph) {
          std::filesystem::path root;
          std::string error;
          graph = discover_graph_contract_local(candidate, original_manifest, pipeline_elements,
                                                model_source_paths, &root, &error);
          if (!graph)
            return invalid(error.empty() ? "graph-owned stage requires MPK manifest" : error);
        }
        const auto* original = original_manifest ? find_original_stage_for_transformed_key_local(
                                                       *original_manifest, name_transform, stage)
                                                 : nullptr;
        const auto exact = !stage.processcvu.exact_stage_name_or_id.empty()
                               ? stage.processcvu.exact_stage_name_or_id
                               : resolve_exact_processcvu_graph_stage_key_local(original);
        const auto keys = graph_stage_candidate_keys_local(stage, original, exact);
        const auto matches =
            find_graph_nodes_matching_stage_keys_local(graph->graph.raw_nodes, keys, canonical);
        if (matches.size() != 1)
          return invalid("graph-owned numerical stage could not resolve a unique graph node: " +
                         stage_key_from_stage_spec_local(stage));
        StageStaticSpec completed;
        std::string error;
        if (!complete_kernel_stage_from_graph_local(stage, *graph, *matches.front(), &completed,
                                                    &error))
          return invalid(error);
        stage = std::move(completed);
      }
    }
    if (auto failure = complete_kernel_input_routes(&candidate, stage_index))
      return failure;
    if (auto failure = lower_kernel_stage_local(&stage))
      return failure;
  }
  *manifest = std::move(candidate);
  return {};
}

::simaai::scheduler::client::Status
finalize_stage_publications(SimaPluginStaticManifest* manifest) {
  if (!manifest)
    return ::simaai::scheduler::client::Error{
        ::simaai::scheduler::client::ErrorCode::InvalidRequest, "stage.publication",
        "manifest required"};
  for (auto& stage : manifest->stages) {
    if (!stage.native_kernel && !stage.mla_program && !stage.tvm_program)
      continue;
    if (stage.native_kernel && stage_is_processcvu_local(stage)) {
      if (auto error = finalize_processcvu_publication(&stage))
        return error;
    } else {
      ::simaai::scheduler::client::TensorStagePublication publication;
      std::string error;
      if (!build_kernel_stage_publication(stage, &publication, &error))
        return ::simaai::scheduler::client::Error{
            ::simaai::scheduler::client::ErrorCode::InvalidRequest, "stage.publication",
            std::move(error)};
      stage.tensor_publication =
          std::make_shared<const ::simaai::scheduler::client::TensorStagePublication>(
              std::move(publication));
    }
  }
  return {};
}

} // namespace simaai::neat::pipeline_internal::sima
