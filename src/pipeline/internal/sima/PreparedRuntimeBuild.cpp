#include "pipeline/internal/sima/PreparedRuntimeBuild.h"

#include "model/internal/ModelPack.h"
#include "pipeline/internal/SimaaiGstCompat.h"
#include "pipeline/internal/EnvUtil.h"
#include "pipeline/internal/sima/MpkContract.h"
#include "pipeline/internal/sima/TensorSemanticsUtil.h"

#include "gst/SimaPreparedRuntimeAbi.h"

#include <algorithm>
#include <array>
#include <cctype>
#include <cstddef>
#include <cstdarg>
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

struct MpkTensorDims {
  int width = 0;
  int height = 0;
  int depth = 0;
  std::string layout;
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
  // Propagates to GraphProcessCvuStageRequest::c16_packed_io so the EV
  // tile descriptor builder clears COMPACT_CHANNELS for c16-padded buffers.
  bool c16_packed = false;
};

bool checked_add_u64_local(const std::uint64_t a, const std::uint64_t b, std::uint64_t* out);

MpkTensorDims dims_from_mpk_shape(std::vector<std::int64_t> shape) {
  if (shape.size() >= 4U && shape.front() == 1) {
    shape.erase(shape.begin());
  }

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

std::string processcvu_canonical_graph_name_local(std::string graph_name);
GstCaps* processmla_make_transport_caps_from_tensor_local(const CapsTensorSpec& tensor,
                                                          const std::string& transport_format);
bool processcvu_graph_family_uses_packed_input_transport_local(const std::string& graph_family);
bool processcvu_tensor_dims_whc_from_shape_layout_local(const std::vector<std::int64_t>& shape,
                                                        const std::string& layout, int* w, int* h,
                                                        int* c);
bool processcvu_build_stage_tensor_descs_local(const StageStaticSpec& stage,
                                               simaai::gst::PreparedProcessCvuTypedConfig* cfg,
                                               std::string* error_message);

bool expand_processmla_output_descs_from_logical_outputs_local(
    std::vector<ProcessMlaOutputDesc>* outputs,
    const std::vector<ProcessMlaLogicalOutputDesc>& logical_outputs);
bool build_processmla_output_descs_from_physical_specs_local(
    const std::vector<PhysicalBufferStaticSpec>& physical_outputs,
    const std::vector<ProcessMlaLogicalOutputDesc>& logical_outputs,
    std::vector<ProcessMlaOutputDesc>* outputs);
bool build_processmla_output_descs_from_physical_tensors_local(
    const std::vector<MpkTensorContract>& physical_tensors,
    const std::vector<ProcessMlaLogicalOutputDesc>& logical_outputs,
    std::vector<ProcessMlaOutputDesc>* outputs);
const LogicalInputStaticSpec*
processcvu_find_logical_input_by_index_local(const StageStaticSpec& stage, int logical_index);
const LogicalTensorStaticSpec*
processcvu_find_logical_output_by_index_local(const StageStaticSpec& stage, int logical_index);
const LogicalTensorStaticSpec*
processcvu_find_logical_output_by_index_or_slot_local(const StageStaticSpec& stage,
                                                      int logical_index, int output_slot);
const PhysicalBufferStaticSpec*
processcvu_find_physical_input_by_index_local(const StageStaticSpec& stage, int physical_index);
const PhysicalBufferStaticSpec*
processcvu_find_physical_output_by_index_local(const StageStaticSpec& stage, int physical_index);
std::string preferred_tensor_name_local(const MpkTensorContract& tensor, const std::size_t index,
                                        const std::string& fallback_prefix);
std::string graph_tensor_semantic_name_local(const MpkTensorContract& tensor);

std::string
prepared_input_layout_token_local(const simaai::gst::PreparedProcessCvuTypedConfig& cfg) {
  if (!cfg.input_tensors.empty()) {
    return tensorsemantics::layout_token_from_ev_tensor_desc(cfg.input_tensors.front());
  }
  return {};
}

std::string
prepared_output_layout_token_local(const simaai::gst::PreparedProcessCvuTypedConfig& cfg,
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

std::optional<std::filesystem::path>
discover_pack_root_from_model_path_local(const std::string& model_path);
std::string resolve_model_path_from_pack_root_local(const std::filesystem::path& pack_root,
                                                    const std::string& executable);
bool build_processmla_prepared_stage_from_graph_local(const MpkContract& contract,
                                                      const MpkGraphNode& graph_node,
                                                      const std::string& stage_key,
                                                      simaai::gst::ProcessMlaPreparedStage* out,
                                                      std::string* error_message);
std::optional<MpkContract>
load_graph_contract_from_manifest_local(const SimaPluginStaticManifest& manifest,
                                        std::filesystem::path* pack_root_out,
                                        std::string* error_message);
std::optional<MpkContract> load_graph_contract_from_pipeline_elements_local(
    const std::vector<PipelineElementSpec>& pipeline_elements, std::filesystem::path* pack_root_out,
    std::string* error_message);
std::optional<MpkContract>
load_graph_contract_from_model_sources_local(const std::vector<std::string>& model_source_paths,
                                             std::filesystem::path* pack_root_out,
                                             std::string* error_message);
std::string resolve_exact_processcvu_graph_stage_key_local(const StageStaticSpec* stage);
bool build_processcvu_prepared_stage_from_graph_local(const StageStaticSpec& original_stage,
                                                      const MpkContract& contract,
                                                      const MpkGraphNode& graph_node,
                                                      const std::string& stage_key,
                                                      simaai::gst::ProcessCvuPreparedStage* out,
                                                      std::string* error_message);
bool build_processcvu_prepared_stage_from_stage_contract_local(
    const StageStaticSpec& stage, simaai::gst::ProcessCvuPreparedStage* out,
    std::string* error_message);
bool build_dequant_prepared_stage_from_stage_contract_local(const StageStaticSpec& stage,
                                                            simaai::gst::DequantPreparedStage* out,
                                                            std::string* error_message);

std::string graph_tensor_semantic_name_local(const MpkTensorContract& tensor) {
  return !tensor.name.empty() ? tensor.name : tensor.segment_name;
}

simaai::neat::GraphTensorContract
bridge_graph_tensor_contract_from_mpk_local(const MpkTensorContract& tensor) {
  simaai::neat::GraphTensorContract out;
  out.tensor_index = tensor.tensor_index;
  out.physical_index = tensor.physical_index;
  out.source_physical_index = tensor.source_physical_index;
  out.name = tensor.name;
  out.segment_name = tensor.segment_name;
  out.dtype = !tensor.logical_dtype.empty() ? tensor.logical_dtype : tensor.dtype;
  out.shape = !tensor.logical_shape.empty() ? tensor.logical_shape : tensor.mpk_shape;
  out.size_bytes = tensor.size_bytes;
  out.byte_offset = tensor.byte_offset;
  out.source_byte_offset = tensor.source_byte_offset;
  out.stride_bytes = tensor.stride_bytes;
  switch (tensor.materialization_kind) {
  case MpkTensorMaterializationKind::OffsetView:
    out.materialization_kind = simaai::neat::GraphTensorMaterializationKind::OffsetView;
    break;
  case MpkTensorMaterializationKind::Bf16LaneSplitRepack:
    out.materialization_kind = simaai::neat::GraphTensorMaterializationKind::Bf16LaneSplitRepack;
    break;
  case MpkTensorMaterializationKind::Unknown:
    out.materialization_kind = simaai::neat::GraphTensorMaterializationKind::Unknown;
    break;
  case MpkTensorMaterializationKind::Direct:
  default:
    out.materialization_kind = simaai::neat::GraphTensorMaterializationKind::Direct;
    break;
  }
  return out;
}

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

bool stage_is_cast_local(const StageStaticSpec& stage) {
  const std::string plugin_kind = upper_copy_local(stage.plugin_kind);
  if (stage.payload_kind == StagePayloadKind::ProcessCvu || plugin_kind == "NEATPROCESSCVU" ||
      plugin_kind == "PROCESSCVU") {
    return false;
  }
  return plugin_kind == "NEATCAST" || upper_copy_local(stage.kernel_kind) == "CAST";
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

bool stage_is_dequant_local(const StageStaticSpec& stage) {
  return stage.payload_kind == StagePayloadKind::Dequant ||
         upper_copy_local(stage.plugin_kind) == "NEATDEQUANT" ||
         upper_copy_local(stage.plugin_kind) == "DEQUANT" ||
         upper_copy_local(stage.kernel_kind) == "DEQUANT" ||
         upper_copy_local(stage.kernel_kind) == "DEQUANTIZE";
}

std::optional<simaai::gst::PreparedStageKind>
prepared_stage_kind_for_stage_local(const StageStaticSpec& stage) {
  if (stage_is_processcvu_local(stage)) {
    return simaai::gst::PreparedStageKind::ProcessCvu;
  }
  if (stage_is_cast_local(stage)) {
    return simaai::gst::PreparedStageKind::Cast;
  }
  if (stage_is_processmla_local(stage)) {
    return simaai::gst::PreparedStageKind::ProcessMla;
  }
  if (stage_is_dequant_local(stage)) {
    return simaai::gst::PreparedStageKind::Dequant;
  }
  return std::nullopt;
}

bool processcvu_stage_is_manifest_substitution_local(const StageStaticSpec& stage) {
  if (!stage_is_processcvu_local(stage)) {
    return false;
  }
  const std::string canonical_family = processcvu_canonical_graph_name_local(
      !stage.processcvu.graph_family.empty() ? stage.processcvu.graph_family
                                             : stage.processcvu.graph_name);
  return canonical_family == "preproc" || canonical_family == "quantize" ||
         canonical_family == "quanttess" || canonical_family == "cast" ||
         canonical_family == "casttess";
}

bool stage_is_graph_owned_local(const StageStaticSpec& stage) {
  if (!prepared_stage_kind_for_stage_local(stage).has_value()) {
    return false;
  }
  if (stage_is_processcvu_local(stage)) {
    return !processcvu_stage_is_manifest_substitution_local(stage);
  }
  if (stage_is_cast_local(stage)) {
    return false;
  }
  return true;
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

void prepared_runtime_warn_log_local(const char* fmt, ...) {
  if (!fmt) {
    return;
  }
  std::fprintf(stderr, "[prepared-runtime-build][warn] ");
  va_list args;
  va_start(args, fmt);
  std::vfprintf(stderr, fmt, args);
  va_end(args);
  std::fprintf(stderr, "\n");
}

std::optional<std::size_t> find_contract_stage_index_local(const MpkContract& contract,
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

bool contract_stage_output_feeds_mla_local(const MpkContract& contract,
                                           const std::size_t stage_index, const int output_index) {
  const auto* mla_stage = get_mla_stage_io_contract(contract);
  const auto mla_stage_index = find_contract_stage_index_local(contract, mla_stage);
  if (!mla_stage_index.has_value()) {
    return false;
  }
  for (const auto& edge : contract.edges) {
    if (edge.src_plugin_index != stage_index || edge.src_output_index != output_index) {
      continue;
    }
    if (edge.dst_plugin_index == *mla_stage_index) {
      return true;
    }
  }
  return false;
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

std::string tensorbuffer_layout_token_local(const int layout) {
  switch (layout) {
  case SIMA_TENSOR_SET_LAYOUT_HWC_V1:
  case SIMA_TENSOR_SET_LAYOUT_NHWC_V1:
    return "HWC";
  case SIMA_TENSOR_SET_LAYOUT_CHW_V1:
  case SIMA_TENSOR_SET_LAYOUT_NCHW_V1:
    return "CHW";
  case SIMA_TENSOR_SET_LAYOUT_HW_V1:
    return "HW";
  default:
    return {};
  }
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

bool tensor_dims_whd_local(const CapsTensorSpec& tensor, std::int64_t* w, std::int64_t* h,
                           std::int64_t* d) {
  if (!w || !h || !d) {
    return false;
  }
  *w = 0;
  *h = 0;
  *d = 0;
  if (tensor.shape.empty()) {
    return false;
  }
  const std::string layout = upper_copy_local(tensor.layout);
  if (tensor.shape.size() >= 3U) {
    const std::int64_t a = tensor.shape[tensor.shape.size() - 3U];
    const std::int64_t b = tensor.shape[tensor.shape.size() - 2U];
    const std::int64_t c = tensor.shape[tensor.shape.size() - 1U];
    if (layout.find("CHW") != std::string::npos) {
      *d = a;
      *h = b;
      *w = c;
    } else {
      *h = a;
      *w = b;
      *d = c;
    }
  } else if (tensor.shape.size() == 2U) {
    *h = tensor.shape[0];
    *w = tensor.shape[1];
    *d = 1;
  } else {
    *w = tensor.shape[0];
    *h = 1;
    *d = 1;
  }
  return *w > 0 && *h > 0 && *d > 0;
}

CapsTensorSpec caps_tensor_from_logical_input_local(const LogicalInputStaticSpec& input) {
  CapsTensorSpec out;
  out.tensor_index = input.backend_input_index;
  out.shape = input.shape;
  out.dtype = input.dtype;
  out.layout = input.layout;
  out.semantic_tag = !input.logical_name.empty() ? input.logical_name : input.segment_name;
  out.size_bytes = input.size_bytes;
  return out;
}

CapsTensorSpec caps_tensor_from_logical_output_local(const LogicalTensorStaticSpec& output) {
  CapsTensorSpec out;
  out.tensor_index = output.tensor_index;
  out.shape = output.shape;
  out.dtype = output.dtype;
  out.layout = output.layout;
  out.max_stride = static_cast<int>(std::min<std::uint64_t>(
      output.size_bytes, static_cast<std::uint64_t>(std::numeric_limits<int>::max())));
  out.semantic_tag = !output.logical_name.empty() ? output.logical_name : output.segment_name;
  out.size_bytes = output.size_bytes;
  return out;
}

CapsTensorSpec
caps_tensor_from_mpk_tensor_local(const MpkTensorContract& tensor,
                                  const std::string& semantic_dtype_override = std::string()) {
  CapsTensorSpec out;
  out.tensor_index = tensor.tensor_index;
  const bool packed_extent =
      tensor.shape_semantics == MpkShapeSemantics::PackedExtent && tensor.logical_shape.empty();
  out.shape = packed_extent
                  ? std::vector<std::int64_t>{}
                  : (!tensor.logical_shape.empty() ? tensor.logical_shape : tensor.mpk_shape);
  out.dtype = !semantic_dtype_override.empty()
                  ? semantic_dtype_override
                  : (!tensor.logical_dtype.empty() ? tensor.logical_dtype : tensor.dtype);
  if (out.dtype.empty() && packed_extent) {
    out.dtype = "INT8";
  }
  // Graph-owned contracts do not currently carry an explicit layout token for every
  // tensor boundary. Keep layout unspecified here instead of fabricating one from rank.
  out.shape_semantics = tensor.shape_semantics;
  out.semantic_tag = !tensor.name.empty() ? tensor.name : tensor.segment_name;
  out.layout.clear();
  out.size_bytes = tensor.size_bytes;
  if (out.size_bytes == 0U) {
    out.size_bytes = shape_size_bytes_local(out.shape, out.dtype);
  }
  return out;
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

bool processmla_dtype_is_float_like_local(const std::string& dtype) {
  const std::string up = upper_copy_local(dtype);
  return up.find("BF16") != std::string::npos || up.find("BFLOAT16") != std::string::npos ||
         up.find("FP32") != std::string::npos || up.find("FLOAT32") != std::string::npos ||
         up.find("FP64") != std::string::npos || up.find("FLOAT64") != std::string::npos;
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

bool processmla_is_video_format_token_local(const std::string& token) {
  const std::string up = upper_copy_local(token);
  return up == "RGB" || up == "BGR" || up == "NV12" || up == "I420" || up == "IYUV" ||
         up == "GRAY" || up == "GRAY8";
}

void processmla_add_shape_fields_to_caps_local(GstCaps* caps, const CapsTensorSpec& tensor);

GstCaps* processmla_make_video_caps_from_tensor_local(const CapsTensorSpec& tensor,
                                                      const std::string& fallback_format) {
  std::string format = fallback_format.empty() ? std::string("BGR") : fallback_format;
  if (!tensor.dtype.empty()) {
    format = upper_copy_local(tensor.dtype);
  }
  if (format == "IYUV") {
    format = "I420";
  } else if (format == "GRAY") {
    format = "GRAY8";
  }
  GstCaps* caps =
      gst_caps_new_simple("video/x-raw", "format", G_TYPE_STRING, format.c_str(), nullptr);
  if (!caps) {
    return gst_caps_new_any();
  }
  std::int64_t w = 0;
  std::int64_t h = 0;
  std::int64_t d = 0;
  if (tensor_dims_whd_local(tensor, &w, &h, &d)) {
    if (w > 0) {
      gst_caps_set_simple(caps, "width", G_TYPE_INT, static_cast<gint>(w), nullptr);
    }
    if (h > 0) {
      gst_caps_set_simple(caps, "height", G_TYPE_INT, static_cast<gint>(h), nullptr);
    }
  }
  return caps;
}

std::string common_string_or_empty_local(const std::vector<std::string>& values) {
  std::string common;
  bool initialized = false;
  for (const auto& value : values) {
    if (value.empty()) {
      return {};
    }
    if (!initialized) {
      common = value;
      initialized = true;
      continue;
    }
    if (common != value) {
      return {};
    }
  }
  return initialized ? common : std::string{};
}

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

const simaai::gst::TensorBufferPublishLogicalOutput* single_publish_logical_output_local(
    const simaai::gst::TensorBufferPublishContract& publish_contract) {
  if (publish_contract.logical_outputs.size() != 1U) {
    return nullptr;
  }
  if (publish_contract.output_order.empty()) {
    return &publish_contract.logical_outputs[0];
  }
  if (publish_contract.output_order.size() != 1U) {
    return nullptr;
  }
  const auto logical_pos = publish_contract.output_order[0].logical_output_index;
  if (logical_pos < 0 ||
      static_cast<std::size_t>(logical_pos) >= publish_contract.logical_outputs.size()) {
    return nullptr;
  }
  return &publish_contract.logical_outputs[static_cast<std::size_t>(logical_pos)];
}

std::string
common_logical_input_dtype_local(const std::vector<LogicalInputStaticSpec>& logical_inputs) {
  std::vector<std::string> values;
  values.reserve(logical_inputs.size());
  for (const auto& logical : logical_inputs) {
    values.push_back(logical.dtype);
  }
  return common_string_or_empty_local(values);
}

std::string
common_logical_input_layout_local(const std::vector<LogicalInputStaticSpec>& logical_inputs) {
  std::vector<std::string> values;
  values.reserve(logical_inputs.size());
  for (const auto& logical : logical_inputs) {
    values.push_back(logical.layout);
  }
  return common_string_or_empty_local(values);
}

std::string
common_logical_output_dtype_local(const std::vector<LogicalTensorStaticSpec>& logical_outputs) {
  std::vector<std::string> values;
  values.reserve(logical_outputs.size());
  for (const auto& logical : logical_outputs) {
    values.push_back(logical.dtype);
  }
  return common_string_or_empty_local(values);
}

std::string
common_logical_output_layout_local(const std::vector<LogicalTensorStaticSpec>& logical_outputs) {
  std::vector<std::string> values;
  values.reserve(logical_outputs.size());
  for (const auto& logical : logical_outputs) {
    values.push_back(logical.layout);
  }
  return common_string_or_empty_local(values);
}

std::vector<std::int64_t>
common_logical_input_shape_local(const std::vector<LogicalInputStaticSpec>& logical_inputs) {
  if (logical_inputs.empty()) {
    return {};
  }
  const auto& first = logical_inputs[0].shape;
  for (std::size_t i = 1; i < logical_inputs.size(); ++i) {
    if (logical_inputs[i].shape != first) {
      return {};
    }
  }
  return first;
}

std::vector<std::int64_t>
common_logical_output_shape_local(const std::vector<LogicalTensorStaticSpec>& logical_outputs) {
  if (logical_outputs.empty()) {
    return {};
  }
  const auto& first = logical_outputs[0].shape;
  for (std::size_t i = 1; i < logical_outputs.size(); ++i) {
    if (logical_outputs[i].shape != first) {
      return {};
    }
  }
  return first;
}

GstCaps* processmla_make_mla_transport_caps_local() {
  return gst_caps_new_simple("application/vnd.simaai.tensor", "format", G_TYPE_STRING, "MLA",
                             nullptr);
}

GstCaps* build_exact_mla_transport_caps_from_size_bytes_local(const std::uint64_t size_bytes) {
  (void)size_bytes;
  return processmla_make_mla_transport_caps_local();
}

template <typename PhysicalCarrier>
GstCaps* build_mla_transport_caps_from_prepared_physical_carriers_local(
    const std::vector<PhysicalCarrier>& physicals) {
  (void)physicals;
  return processmla_make_mla_transport_caps_local();
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

GstCaps* processmla_make_transport_caps_from_tensor_local(const CapsTensorSpec& tensor,
                                                          const std::string& transport_format) {
  GstCaps* caps = gst_caps_new_simple("application/vnd.simaai.tensor", "format", G_TYPE_STRING,
                                      transport_format.c_str(), nullptr);
  if (!caps) {
    return gst_caps_new_any();
  }

  const std::string dtype = processmla_caps_format_from_dtype_local(tensor.dtype, std::string{});
  if (!dtype.empty()) {
    gst_caps_set_simple(caps, "dtype", G_TYPE_STRING, dtype.c_str(), nullptr);
  }
  processmla_add_shape_fields_to_caps_local(caps, tensor);
  return caps;
}

bool processmla_publish_contract_has_precise_single_output_local(
    const simaai::gst::TensorBufferPublishContract& publish_contract, CapsTensorSpec* out) {
  if (!out) {
    return false;
  }
  const auto* logical = single_publish_logical_output_local(publish_contract);
  if (!logical) {
    return false;
  }
  *out = caps_tensor_from_publish_logical_output_local(*logical);
  return !out->shape.empty();
}

bool build_processmla_caps_from_graph_contract_local(
    const std::vector<MpkTensorContract>& logical_inputs,
    const simaai::gst::TensorBufferReadRequest& input_request,
    const simaai::gst::TensorBufferPublishContract& output_publish_contract,
    GstCaps** sink_caps_out, GstCaps** src_caps_out) {
  if (!sink_caps_out || !src_caps_out) {
    return false;
  }
  *sink_caps_out = gst_caps_new_any();
  *src_caps_out = gst_caps_new_any();
  if (logical_inputs.empty()) {
    return true;
  }

  GstCaps* sink_caps = nullptr;
  std::string sink_dtype_up;
  if (logical_inputs.size() == 1U && input_request.entries.size() == 1U) {
    CapsTensorSpec input_tensor = caps_tensor_from_mpk_tensor_local(logical_inputs[0]);
    if (!input_tensor.shape.empty()) {
      sink_dtype_up = processmla_caps_format_from_dtype_local(input_tensor.dtype, std::string{});
      if (processmla_is_video_format_token_local(sink_dtype_up)) {
        sink_caps = processmla_make_video_caps_from_tensor_local(input_tensor, sink_dtype_up);
      } else {
        sink_caps =
            processmla_make_tensor_caps_from_tensor_local(input_tensor, std::string{}, false);
        processmla_add_shape_fields_to_caps_local(sink_caps, input_tensor);
        if (!sink_dtype_up.empty()) {
          gst_caps_set_simple(sink_caps, "dtype", G_TYPE_STRING, sink_dtype_up.c_str(), nullptr);
        }
      }
    } else {
      sink_caps = make_generic_tensor_set_caps_local();
    }
  } else {
    sink_caps = make_generic_tensor_set_caps_local();
  }

  GstCaps* src_caps = nullptr;
  CapsTensorSpec output_tensor;
  if (processmla_publish_contract_has_precise_single_output_local(output_publish_contract,
                                                                  &output_tensor)) {
    src_caps = processmla_make_tensor_caps_from_tensor_local(output_tensor, std::string{}, false);
    processmla_add_shape_fields_to_caps_local(src_caps, output_tensor);
    const std::string output_dtype_up =
        processmla_caps_format_from_dtype_local(output_tensor.dtype, "");
    if (!output_dtype_up.empty() && output_dtype_up != "MLA") {
      gst_caps_set_simple(src_caps, "dtype", G_TYPE_STRING, output_dtype_up.c_str(), nullptr);
    }
    const bool output_dtype_unspecified = output_dtype_up.empty() || output_dtype_up == "MLA";
    if (src_caps && output_dtype_unspecified && !sink_dtype_up.empty() &&
        !processmla_is_video_format_token_local(sink_dtype_up)) {
      gst_caps_set_simple(src_caps, "format", G_TYPE_STRING, sink_dtype_up.c_str(), "dtype",
                          G_TYPE_STRING, sink_dtype_up.c_str(), nullptr);
    }
  } else {
    src_caps = make_generic_tensor_set_caps_local();
  }

  if (sink_caps) {
    gst_caps_unref(*sink_caps_out);
    *sink_caps_out = sink_caps;
  }
  if (src_caps) {
    gst_caps_unref(*src_caps_out);
    *src_caps_out = src_caps;
  }
  return true;
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
  out->physical_outputs.reserve(stage.physical_outputs.size());
  for (const auto& physical : stage.physical_outputs) {
    simaai::gst::TensorBufferPublishPhysicalOutput output;
    output.physical_index = physical.physical_index;
    output.size_bytes = physical.size_bytes;
    output.segment_name = physical.segment_name;
    out->physical_outputs.push_back(std::move(output));
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
  for (const auto& logical : out->logical_outputs) {
    if (logical.physical_index < 0 ||
        static_cast<std::size_t>(logical.physical_index) >= out->physical_outputs.size()) {
      continue;
    }
    const auto elem_bytes = [&]() -> std::uint64_t {
      switch (logical.dtype) {
      case SIMA_TENSOR_SET_DTYPE_BF16_V1:
        return 2U;
      case SIMA_TENSOR_SET_DTYPE_FP32_V1:
      case SIMA_TENSOR_SET_DTYPE_INT32_V1:
        return 4U;
      case SIMA_TENSOR_SET_DTYPE_INT16_V1:
      case SIMA_TENSOR_SET_DTYPE_UINT16_V1:
        return 2U;
      case SIMA_TENSOR_SET_DTYPE_INT8_V1:
      case SIMA_TENSOR_SET_DTYPE_UINT8_V1:
      default:
        return 1U;
      }
    }();
    auto stride_bytes = logical.stride_bytes;
    if (stride_bytes.empty()) {
      stride_bytes.resize(logical.shape.size(), 0);
      std::int64_t stride = static_cast<std::int64_t>(elem_bytes);
      for (std::size_t dim = logical.shape.size(); dim-- > 0U;) {
        stride_bytes[dim] = stride;
        if (logical.shape[dim] > 0 &&
            stride <= (std::numeric_limits<std::int64_t>::max() / logical.shape[dim])) {
          stride *= logical.shape[dim];
        }
      }
    }
    std::uint64_t physical_span = logical.size_bytes;
    if (!logical.shape.empty() && logical.shape.size() == stride_bytes.size()) {
      std::uint64_t max_offset = 0U;
      bool span_ok = true;
      for (std::size_t dim = 0; dim < logical.shape.size(); ++dim) {
        if (logical.shape[dim] <= 0 || stride_bytes[dim] < 0) {
          span_ok = false;
          break;
        }
        const auto dim_extent = static_cast<std::uint64_t>(logical.shape[dim] - 1);
        const auto stride_u64 = static_cast<std::uint64_t>(stride_bytes[dim]);
        if (dim_extent > 0U &&
            stride_u64 > (std::numeric_limits<std::uint64_t>::max() / dim_extent)) {
          span_ok = false;
          break;
        }
        const auto term = dim_extent * stride_u64;
        if (!checked_add_u64_local(max_offset, term, &max_offset)) {
          span_ok = false;
          break;
        }
      }
      if (span_ok) {
        std::uint64_t span_end = 0U;
        if (checked_add_u64_local(max_offset, elem_bytes, &span_end)) {
          physical_span = std::max(physical_span, span_end);
        }
      }
    }
    const std::uint64_t required_span =
        static_cast<std::uint64_t>(logical.byte_offset) + physical_span;
    auto& physical = out->physical_outputs[static_cast<std::size_t>(logical.physical_index)];
    if (physical.size_bytes < required_span) {
      physical.size_bytes = required_span;
    }
  }
  out->output_order.reserve(stage.output_order.size());
  for (const auto& route : stage.output_order) {
    simaai::gst::TensorBufferPublishOutputRoute output_route;
    output_route.output_slot = route.output_slot;
    output_route.logical_output_index = route.logical_output_index;
    output_route.cm_output_name = route.cm_output_name;
    output_route.segment_name = route.segment_name;
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
    const StageStaticSpec& stage, const simaai::gst::PreparedProcessCvuTypedConfig& typed_config,
    simaai::gst::TensorBufferPublishContract* contract) {
  if (!contract || contract->logical_outputs.empty() ||
      !processcvu_stage_is_dequant_like_local(stage)) {
    return;
  }
  const std::size_t qparam_count =
      std::min(typed_config.dq_scale_array.size(), typed_config.dq_zp_array.size());
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
    quant.scales = {static_cast<double>(typed_config.dq_scale_array[q_index])};
    quant.zero_points = {static_cast<std::int64_t>(typed_config.dq_zp_array[q_index])};
    logical.quant = std::move(quant);
  }
}

simaai::gst::TensorBufferPublishContract
build_publish_contract_from_runtime_config_local(const ProcessMlaRuntimeConfig& runtime_cfg);

const InputBindingStaticSpec*
find_stage_input_binding_for_logical_index_local(const StageStaticSpec& stage,
                                                 std::size_t logical_request_index) {
  const InputBindingStaticSpec* fallback = nullptr;
  for (const auto& candidate : stage.input_bindings) {
    if (candidate.local_logical_input_index >= 0 &&
        static_cast<std::size_t>(candidate.local_logical_input_index) == logical_request_index) {
      return &candidate;
    }
    if (!fallback && candidate.sink_pad_index >= 0 &&
        static_cast<std::size_t>(candidate.sink_pad_index) == logical_request_index) {
      fallback = &candidate;
    }
  }
  return fallback;
}

bool build_processmla_read_request_entry_from_logical_input_local(
    const LogicalInputStaticSpec& logical_input, const InputBindingStaticSpec* binding,
    std::size_t request_index, simaai::gst::TensorBufferReadRequestEntry* out,
    std::string* error_message) {
  if (!out) {
    if (error_message) {
      *error_message = "processmla read request entry requires output storage";
    }
    return false;
  }

  *out = simaai::gst::TensorBufferReadRequestEntry{};
  out->request_index = request_index;
  out->logical_index = logical_input.logical_index >= 0 ? logical_input.logical_index
                                                        : static_cast<int>(request_index);
  out->logical_name = logical_input.logical_name;
  out->segment_name = logical_input.segment_name;
  out->expected_size_bytes = logical_input.size_bytes;
  out->byte_offset = logical_input.byte_offset >= 0 ? logical_input.byte_offset : -1;
  out->physical_index = logical_input.physical_index;

  if (binding) {
    if (binding->src_logical_output_index >= 0) {
      out->logical_index = binding->src_logical_output_index;
    }
    if (binding->src_output_slot >= 0) {
      out->route_slot = binding->src_output_slot;
    }
    if (binding->src_physical_output_index >= 0) {
      out->physical_index = binding->src_physical_output_index;
    }
    out->source_physical_size_bytes = binding->src_physical_size_bytes;
    out->source_physical_byte_offset =
        binding->src_physical_byte_offset >= 0 ? binding->src_physical_byte_offset : -1;
    if (!binding->source_segment_name.empty()) {
      out->segment_name = binding->source_segment_name;
    }
    if (out->source_physical_byte_offset >= 0) {
      out->byte_offset = out->source_physical_byte_offset;
    }
  }

  if (out->expected_size_bytes == 0U) {
    if (error_message) {
      std::ostringstream oss;
      oss << "processmla logical input requires positive size_bytes"
          << " request_index=" << request_index;
      *error_message = oss.str();
    }
    return false;
  }
  return true;
}

bool build_processmla_read_request_entry_from_physical_input_local(
    const PhysicalBufferStaticSpec& physical_input, const LogicalInputStaticSpec* logical_input,
    std::size_t request_index, simaai::gst::TensorBufferReadRequestEntry* out,
    std::string* error_message) {
  if (!out) {
    if (error_message) {
      *error_message = "processmla physical carrier request entry requires output storage";
    }
    return false;
  }

  *out = simaai::gst::TensorBufferReadRequestEntry{};
  out->request_index = request_index;
  out->physical_carrier = true;
  out->logical_index = logical_input ? ((logical_input->backend_input_index >= 0)
                                            ? logical_input->backend_input_index
                                            : logical_input->logical_index)
                                     : static_cast<int>(request_index);
  out->logical_name =
      (logical_input && !logical_input->logical_name.empty())
          ? logical_input->logical_name
          : (!physical_input.segment_name.empty() ? physical_input.segment_name
                                                  : ("ifm" + std::to_string(request_index)));
  out->segment_name = physical_input.segment_name;
  out->expected_size_bytes = physical_input.size_bytes;
  out->byte_offset = physical_input.source_byte_offset >= 0 ? physical_input.source_byte_offset : 0;
  out->physical_index = physical_input.source_physical_index >= 0
                            ? physical_input.source_physical_index
                            : physical_input.physical_index;
  out->source_physical_size_bytes = physical_input.size_bytes;
  out->source_physical_byte_offset =
      physical_input.source_byte_offset >= 0 ? physical_input.source_byte_offset : 0;

  if (out->expected_size_bytes == 0U) {
    if (error_message) {
      std::ostringstream oss;
      oss << "processmla physical carrier requires positive size_bytes"
          << " request_index=" << request_index;
      *error_message = oss.str();
    }
    return false;
  }
  return true;
}

std::string processmla_carrier_segment_name_from_inputs_local(
    const std::vector<MpkTensorContract>& logical_inputs) {
  for (const auto& input : logical_inputs) {
    if (!input.segment_name.empty()) {
      return input.segment_name;
    }
  }
  return {};
}

int processmla_carrier_physical_index_from_inputs_local(
    const std::vector<MpkTensorContract>& logical_inputs) {
  for (const auto& input : logical_inputs) {
    if (input.source_physical_index >= 0) {
      return input.source_physical_index;
    }
    if (input.physical_index >= 0) {
      return input.physical_index;
    }
  }
  return -1;
}

std::int64_t processmla_carrier_byte_offset_from_inputs_local(
    const std::vector<MpkTensorContract>& logical_inputs) {
  std::optional<std::int64_t> min_offset;
  for (const auto& input : logical_inputs) {
    const std::int64_t candidate = input.source_byte_offset >= 0
                                       ? input.source_byte_offset
                                       : (input.byte_offset >= 0 ? input.byte_offset : 0);
    if (!min_offset.has_value() || candidate < *min_offset) {
      min_offset = candidate;
    }
  }
  return min_offset.value_or(0);
}

std::size_t
processmla_carrier_size_from_inputs_local(const std::vector<MpkTensorContract>& logical_inputs) {
  std::uint64_t total = 0U;
  for (const auto& input : logical_inputs) {
    if (input.size_bytes == 0U) {
      return 0U;
    }
    if (total > std::numeric_limits<std::uint64_t>::max() - input.size_bytes) {
      return 0U;
    }
    total += input.size_bytes;
  }
  if (total > std::numeric_limits<std::size_t>::max()) {
    return 0U;
  }
  return static_cast<std::size_t>(total);
}

bool build_processmla_read_request_entry_from_dispatcher_tensor_local(
    const MpkTensorContract& request_input, const std::vector<MpkTensorContract>& logical_inputs,
    std::size_t request_index, simaai::gst::TensorBufferReadRequestEntry* out,
    std::string* error_message) {
  if (!out) {
    if (error_message) {
      *error_message = "graph processmla carrier request entry requires output storage";
    }
    return false;
  }

  *out = simaai::gst::TensorBufferReadRequestEntry{};
  out->request_index = request_index;
  out->physical_carrier = true;
  out->logical_index = request_input.tensor_index >= 0
                           ? request_input.tensor_index
                           : (!logical_inputs.empty() && logical_inputs.front().tensor_index >= 0
                                  ? logical_inputs.front().tensor_index
                                  : static_cast<int>(request_index));
  out->logical_name = graph_tensor_semantic_name_local(request_input);
  if (out->logical_name.empty() && !logical_inputs.empty()) {
    out->logical_name = graph_tensor_semantic_name_local(logical_inputs.front());
  }
  out->segment_name = !request_input.segment_name.empty()
                          ? request_input.segment_name
                          : processmla_carrier_segment_name_from_inputs_local(logical_inputs);
  if (out->segment_name.empty() && !out->logical_name.empty()) {
    out->segment_name = out->logical_name;
  }
  out->expected_size_bytes = request_input.size_bytes > 0U
                                 ? request_input.size_bytes
                                 : processmla_carrier_size_from_inputs_local(logical_inputs);
  out->byte_offset = request_input.source_byte_offset >= 0
                         ? request_input.source_byte_offset
                         : (request_input.byte_offset >= 0
                                ? request_input.byte_offset
                                : processmla_carrier_byte_offset_from_inputs_local(logical_inputs));
  out->physical_index =
      request_input.source_physical_index >= 0
          ? request_input.source_physical_index
          : (request_input.physical_index >= 0
                 ? request_input.physical_index
                 : processmla_carrier_physical_index_from_inputs_local(logical_inputs));
  out->source_physical_size_bytes = out->expected_size_bytes;
  out->source_physical_byte_offset = out->byte_offset;

  if (out->logical_index < 0 || out->logical_name.empty() ||
      (out->segment_name.empty() && out->physical_index < 0) || out->expected_size_bytes == 0U ||
      out->byte_offset < 0) {
    if (error_message) {
      std::ostringstream oss;
      oss << "graph processmla dispatcher input missing required carrier identity"
          << " request_index=" << request_index << " logical_name=" << out->logical_name
          << " segment_name=" << (out->segment_name.empty() ? "<empty>" : out->segment_name)
          << " physical_index=" << out->physical_index
          << " expected_size_bytes=" << out->expected_size_bytes
          << " byte_offset=" << out->byte_offset;
      *error_message = oss.str();
    }
    return false;
  }
  return true;
}

bool build_cast_prepared_stage_from_manifest_stage_local(const StageStaticSpec& stage,
                                                         simaai::gst::CastPreparedStage* out,
                                                         std::string* error_message) {
  if (!out) {
    if (error_message) {
      *error_message = "cast prepared stage requires output storage";
    }
    return false;
  }
  if (!stage_is_cast_local(stage)) {
    if (error_message) {
      *error_message = "stage is not cast";
    }
    return false;
  }

  simaai::gst::CastPreparedStage prepared;
  prepared.stage_key = stage_key_from_stage_spec_local(stage);
  std::string input_dtype = common_logical_input_dtype_local(stage.logical_inputs);
  std::string output_dtype = common_logical_output_dtype_local(stage.logical_outputs);
  input_dtype = upper_copy_local(input_dtype);
  output_dtype = upper_copy_local(output_dtype);
  prepared.direction = (output_dtype.find("BF16") != std::string::npos &&
                        input_dtype.find("FP32") != std::string::npos)
                           ? 1
                           : 0;
  if (!build_publish_contract_from_manifest_stage_local(stage, &prepared.identity_publish_contract,
                                                        error_message)) {
    return false;
  }
  simaai::gst::TensorBufferPreparedMetaTemplate meta_template;
  if (!simaai::gst::tensor_buffer_prepare_meta_template_from_contract(
          prepared.identity_publish_contract, &meta_template, error_message)) {
    return false;
  }
  prepared.prepared_meta_template = std::move(meta_template);
  *out = std::move(prepared);
  return true;
}

bool processcvu_primary_output_uses_packed_caps_from_payload_local(
    const ProcessCvuStagePayload& payload) {
  return payload.primary_output_transport_kind == ProcessCvuOutputTransportKind::Packed ||
         payload.primary_output_semantic_kind == ProcessCvuOutputSemanticKind::TessellatedImage ||
         payload.primary_output_semantic_kind == ProcessCvuOutputSemanticKind::QuantTessTensor;
}

bool processcvu_set_input_binding_contract_from_packed_transport_local(
    simaai::gst::CvuInputMemoryBinding* binding, const std::uint64_t size_bytes,
    const std::string& dtype, std::string* error_message) {
  if (!binding || size_bytes == 0U) {
    if (error_message) {
      *error_message = "packed input transport requires non-zero size";
    }
    return false;
  }
  const std::size_t elem_bytes = dtype_size_bytes_local(dtype);
  if (elem_bytes == 0U || (size_bytes % elem_bytes) != 0U) {
    if (error_message) {
      *error_message = "packed input transport size is not aligned to dtype";
    }
    return false;
  }
  binding->dtype = dtype;
  return true;
}

bool build_processcvu_typed_config_from_manifest_stage_local(
    const StageStaticSpec& stage, simaai::gst::PreparedProcessCvuTypedConfig* out,
    std::string* error_message) {
  if (!out) {
    if (error_message) {
      *error_message = "processcvu typed config requires output storage";
    }
    return false;
  }
  if (!stage_is_processcvu_local(stage)) {
    if (error_message) {
      *error_message = "stage is not processcvu";
    }
    return false;
  }
  const auto& payload = stage.processcvu;
  if (!payload.canonical_contract || payload.graph_name.empty() ||
      payload.default_input_name.empty() || payload.primary_output_name.empty()) {
    if (error_message) {
      *error_message = "processcvu canonical payload is incomplete";
    }
    return false;
  }

  simaai::gst::PreparedProcessCvuTypedConfig cfg;
  cfg.graph_name = payload.graph_name;
  cfg.cpu = "CVU";
  cfg.requested_run_target =
      payload.requested_run_target.empty() ? payload.run_target : payload.requested_run_target;
  cfg.run_target = payload.run_target.empty() ? std::string("AUTO") : payload.run_target;
  cfg.resolved_exec_backend =
      payload.resolved_exec_backend.empty() ? std::string("EVXX") : payload.resolved_exec_backend;
  cfg.run_target_resolution_reason = payload.run_target_resolution_reason;
  cfg.graph_id = payload.graph_id;
  cfg.default_input_name = payload.default_input_name;
  cfg.primary_output_name = payload.primary_output_name;
  cfg.single_output_handoff = payload.preproc_single_output_handoff;
  const std::string canonical_graph_family = processcvu_canonical_graph_name_local(
      !payload.graph_family.empty() ? payload.graph_family : payload.graph_name);
  cfg.runtime_output_names = payload.default_output_names;
  if (canonical_graph_family == "preproc" && payload.preproc_single_output_handoff) {
    if (payload.primary_output_name.empty() ||
        std::find(payload.default_output_names.begin(), payload.default_output_names.end(),
                  payload.primary_output_name) == payload.default_output_names.end()) {
      if (error_message) {
        *error_message =
            "processcvu preproc single-output handoff primary output is not a runtime output";
      }
      return false;
    }
  }
  cfg.published_output_names.clear();
  for (const auto& route : stage.output_order) {
    if (!route.cm_output_name.empty()) {
      cfg.published_output_names.push_back(route.cm_output_name);
    }
  }
  if (cfg.published_output_names.empty()) {
    for (const auto& logical : stage.logical_outputs) {
      if (!logical.logical_name.empty()) {
        cfg.published_output_names.push_back(logical.logical_name);
      } else if (!logical.backend_name.empty()) {
        cfg.published_output_names.push_back(logical.backend_name);
      } else if (!logical.segment_name.empty()) {
        cfg.published_output_names.push_back(logical.segment_name);
      }
    }
  }

  cfg.input_dtype = payload.input_dtype;
  cfg.output_dtype = payload.output_dtype;
  cfg.out_dtype = payload.out_dtype;
  cfg.input_img_type = payload.input_img_type;
  cfg.output_img_type = payload.output_img_type;
  cfg.scaling_type = payload.scaling_type;
  cfg.padding_type = payload.padding_type;
  cfg.scaled_width = payload.scaled_width;
  cfg.scaled_height = payload.scaled_height;
  cfg.input_stride = payload.input_stride;
  cfg.output_stride = payload.output_stride;
  cfg.input_offset = payload.input_offset;
  cfg.batch_size = payload.batch_size;
  cfg.round_off = payload.round_off;
  cfg.byte_align = payload.byte_align;
  cfg.opt_flags = payload.opt_flags;
  cfg.aspect_ratio = payload.aspect_ratio;
  cfg.normalize = payload.normalize;
  cfg.tessellate = payload.tessellate;
  cfg.num_in_tensor = payload.num_in_tensor;
  cfg.has_q_scale = payload.has_q_scale;
  cfg.q_scale = static_cast<float>(payload.q_scale);
  cfg.has_q_zp = payload.has_q_zp;
  cfg.q_zp = payload.q_zp;
  for (const auto value : payload.q_scale_list) {
    cfg.q_scale_array.push_back(static_cast<float>(value));
  }
  for (const auto value : payload.q_zp_list) {
    cfg.q_zp_array.push_back(value);
  }
  for (const auto value : payload.dq_scale_list) {
    cfg.dq_scale_array.push_back(static_cast<float>(value));
  }
  for (const auto value : payload.dq_zp_list) {
    cfg.dq_zp_array.push_back(value);
  }
  if (payload.channel_mean.size() == 3U) {
    cfg.has_channel_mean = true;
    for (std::size_t i = 0; i < 3U; ++i) {
      cfg.channel_mean[i] = static_cast<float>(payload.channel_mean[i]);
    }
  }
  if (payload.channel_stddev.size() == 3U) {
    cfg.has_channel_stddev = true;
    for (std::size_t i = 0; i < 3U; ++i) {
      cfg.channel_stddev[i] = static_cast<float>(payload.channel_stddev[i]);
    }
  }
  for (const auto& dtype : payload.runtime_output_dtype_list) {
    cfg.output_dtype_array.push_back(dtype);
  }
  cfg.out_dtype_array = cfg.output_dtype_array;
  cfg.input_materialization_kind_array.clear();
  cfg.input_materialization_kind_array.reserve(stage.logical_inputs.size());
  for (const auto& logical : stage.logical_inputs) {
    cfg.input_materialization_kind_array.push_back(
        static_cast<int32_t>(logical.materialization_kind));
  }
  if (canonical_graph_family == "preproc") {
    if (!processcvu_build_stage_tensor_descs_local(stage, &cfg, error_message)) {
      return false;
    }
  } else {
    if (payload.input_tensors.empty() || payload.output_tensors.empty()) {
      if (error_message) {
        *error_message = "generic EV processcvu manifest requires direct tensor descriptors";
      }
      return false;
    }
    if (payload.input_tensors.size() != payload.output_tensors.size()) {
      if (error_message) {
        *error_message = "generic EV processcvu manifest tensor descriptor count mismatch";
      }
      return false;
    }
    cfg.input_tensors = payload.input_tensors;
    cfg.output_tensors = payload.output_tensors;
    cfg.num_in_tensor = static_cast<int32_t>(cfg.input_tensors.size());
  }
  if (!cfg.input_tensors.empty()) {
    (void)tensorsemantics::layout_token_from_ev_tensor_desc(cfg.input_tensors.front());
  }
  *out = std::move(cfg);
  return true;
}

bool build_processcvu_routing_contract_from_manifest_stage_local(
    const StageStaticSpec& stage, simaai::gst::CvuRoutingContract* out,
    std::string* error_message) {
  if (!out) {
    if (error_message) {
      *error_message = "processcvu routing contract requires output storage";
    }
    return false;
  }
  *out = simaai::gst::CvuRoutingContract{};
  if (!stage_is_processcvu_local(stage)) {
    if (error_message) {
      *error_message = "stage is not processcvu";
    }
    return false;
  }

  const auto& payload = stage.processcvu;
  const std::string graph_family =
      !payload.graph_family.empty() ? payload.graph_family : payload.graph_name;
  const std::string canonical_graph_family = processcvu_canonical_graph_name_local(graph_family);
  const bool packed_input_transport =
      processcvu_graph_family_uses_packed_input_transport_local(graph_family);

  out->input_bindings.reserve(stage.input_bindings.size());
  for (const auto& binding_spec : stage.input_bindings) {
    const int logical_index = binding_spec.local_logical_input_index >= 0
                                  ? binding_spec.local_logical_input_index
                                  : binding_spec.sink_pad_index;
    const auto* logical = processcvu_find_logical_input_by_index_local(stage, logical_index);
    if (!logical) {
      if (error_message) {
        *error_message = "processcvu logical input missing for binding";
      }
      return false;
    }
    simaai::gst::CvuInputMemoryBinding binding;
    binding.sink_pad_index = binding_spec.sink_pad_index;
    binding.logical_input_index = logical_index;
    binding.local_physical_index = logical->physical_index;
    binding.source_logical_index = binding_spec.src_logical_output_index;
    binding.source_output_slot = binding_spec.src_output_slot;
    binding.source_physical_index = binding_spec.src_physical_output_index;
    binding.source_size_bytes = binding_spec.src_physical_size_bytes;
    binding.source_byte_offset = binding_spec.src_physical_byte_offset;
    binding.group_name = "sink_pad_" + std::to_string(binding_spec.sink_pad_index);
    binding.segment_name =
        !binding_spec.source_segment_name.empty() ? binding_spec.source_segment_name : "parent";
    binding.graph_input_name =
        !binding_spec.cm_input_name.empty()
            ? binding_spec.cm_input_name
            : (!logical->backend_name.empty() ? logical->backend_name : logical->logical_name);
    if (binding.graph_input_name.empty()) {
      binding.graph_input_name = "input_" + std::to_string(binding.sink_pad_index);
    }
    int verify_w = 0;
    int verify_h = 0;
    int verify_c = 0;
    if (!processcvu_tensor_dims_whc_from_shape_layout_local(logical->shape, logical->layout,
                                                            &verify_w, &verify_h, &verify_c)) {
      if (error_message) {
        *error_message = "processcvu logical input geometry missing";
      }
      return false;
    }
    binding.shape = logical->shape;
    binding.dtype = logical->dtype;
    binding.layout = logical->layout;
    if (packed_input_transport) {
      if (!processcvu_set_input_binding_contract_from_packed_transport_local(
              &binding, logical->size_bytes, logical->dtype, error_message)) {
        return false;
      }
    }
    out->input_bindings.push_back(std::move(binding));
  }

  std::vector<std::string> manifest_runtime_output_names = payload.default_output_names;
  const bool strict_preproc_single_output =
      canonical_graph_family == "preproc" && payload.preproc_single_output_handoff;
  if (strict_preproc_single_output) {
    if (payload.primary_output_name.empty() ||
        std::find(manifest_runtime_output_names.begin(), manifest_runtime_output_names.end(),
                  payload.primary_output_name) == manifest_runtime_output_names.end()) {
      if (error_message) {
        *error_message =
            "processcvu preproc single-output handoff primary output is not a runtime output";
      }
      return false;
    }
    // The prepared-runtime contract must describe the same active CM/runtime
    // outputs as ConfigManager.  For graph-200 preproc single-output handoff,
    // only the selected output is active; the other default preproc output is
    // an internal/inactive alternative and must not be advertised as a runtime
    // binding.
    manifest_runtime_output_names = {payload.primary_output_name};
  }

  auto find_payload_runtime_output_index =
      [&](const std::string& output_name) -> std::optional<std::size_t> {
    for (std::size_t i = 0; i < payload.default_output_names.size(); ++i) {
      if (payload.default_output_names[i] == output_name) {
        return i;
      }
    }
    return std::nullopt;
  };

  auto find_manifest_output_route = [&](const std::string& output_name) -> const StageOutputRoute* {
    for (const auto& route : stage.output_order) {
      if (route.cm_output_name == output_name) {
        return &route;
      }
    }
    return nullptr;
  };

  out->runtime_output_bindings.reserve(manifest_runtime_output_names.size());
  for (const auto& runtime_output_name : manifest_runtime_output_names) {
    const auto payload_index = find_payload_runtime_output_index(runtime_output_name);
    if (!payload_index.has_value()) {
      if (error_message) {
        *error_message = "processcvu runtime output missing from manifest payload";
      }
      return false;
    }
    const std::size_t i = *payload_index;
    const auto* manifest_route = find_manifest_output_route(runtime_output_name);
    simaai::gst::CvuOutputBinding binding;
    binding.dispatcher_name = runtime_output_name;
    binding.output_slot = i < payload.runtime_output_output_slot_list.size()
                              ? payload.runtime_output_output_slot_list[i]
                              : static_cast<int>(i);
    binding.logical_output_index = i < payload.runtime_output_logical_index_list.size()
                                       ? payload.runtime_output_logical_index_list[i]
                                       : binding.output_slot;
    binding.physical_output_index = i < payload.runtime_output_physical_index_list.size()
                                        ? payload.runtime_output_physical_index_list[i]
                                        : binding.logical_output_index;
    if (manifest_route) {
      if (manifest_route->output_slot >= 0) {
        binding.output_slot = manifest_route->output_slot;
      }
      if (manifest_route->logical_output_index >= 0) {
        binding.logical_output_index = manifest_route->logical_output_index;
      }
    }
    binding.dtype = i < payload.runtime_output_dtype_list.size()
                        ? payload.runtime_output_dtype_list[i]
                        : payload.output_dtype;
    binding.layout = i < payload.runtime_output_logical_layout_list.size()
                         ? payload.runtime_output_logical_layout_list[i]
                         : payload.logical_output_layout_token(i);
    const auto transport_kind = i < payload.runtime_output_transport_kind_list.size()
                                    ? payload.runtime_output_transport_kind_list[i]
                                    : ProcessCvuOutputTransportKind::Dense;
    auto fill_binding_from_payload_desc = [&]() -> bool {
      if (i >= payload.output_tensors.size()) {
        return false;
      }
      const auto& desc = payload.output_tensors[i];
      binding.shape = shape_vector_from_ev_shape_local(desc.shape);
      if (desc.storage.addr <=
          static_cast<std::uint64_t>(std::numeric_limits<std::int64_t>::max())) {
        binding.byte_offset = static_cast<std::int64_t>(desc.storage.addr);
      }
      return !binding.shape.empty();
    };
    if (transport_kind == ProcessCvuOutputTransportKind::Packed) {
      binding.contract_kind = simaai::gst::CvuOutputBinding::ContractKind::Packed;
      if (const auto* physical = processcvu_find_physical_output_by_index_local(
              stage, binding.physical_output_index)) {
        binding.segment_name = physical->segment_name;
        binding.size_bytes = physical->size_bytes;
      }
      const auto* logical =
          strict_preproc_single_output
              ? processcvu_find_logical_output_by_index_local(stage, binding.logical_output_index)
              : processcvu_find_logical_output_by_index_or_slot_local(
                    stage, binding.logical_output_index, binding.output_slot);
      if (logical) {
        binding.byte_offset = logical->byte_offset >= 0 ? logical->byte_offset : 0;
        int verify_w = 0;
        int verify_h = 0;
        int verify_c = 0;
        if (!processcvu_tensor_dims_whc_from_shape_layout_local(logical->shape, logical->layout,
                                                                &verify_w, &verify_h, &verify_c)) {
          if (error_message) {
            *error_message = "processcvu runtime logical packed output geometry missing";
          }
          return false;
        }
        binding.shape = logical->shape;
        binding.layout = logical->layout;
      } else if (!fill_binding_from_payload_desc()) {
        if (error_message) {
          *error_message = "processcvu runtime output descriptor missing";
        }
        return false;
      }
    } else {
      const auto* logical =
          strict_preproc_single_output
              ? processcvu_find_logical_output_by_index_local(stage, binding.logical_output_index)
              : processcvu_find_logical_output_by_index_or_slot_local(
                    stage, binding.logical_output_index, binding.output_slot);
      if (!logical) {
        if (const auto* physical = processcvu_find_physical_output_by_index_local(
                stage, binding.physical_output_index)) {
          binding.segment_name = physical->segment_name;
          binding.size_bytes = physical->size_bytes;
        }
        if (!fill_binding_from_payload_desc()) {
          if (error_message) {
            *error_message = "processcvu runtime output descriptor missing";
          }
          return false;
        }
      } else {
        binding.segment_name = logical->segment_name;
        binding.byte_offset = logical->byte_offset >= 0 ? logical->byte_offset : 0;
        int verify_w = 0;
        int verify_h = 0;
        int verify_c = 0;
        if (!processcvu_tensor_dims_whc_from_shape_layout_local(logical->shape, logical->layout,
                                                                &verify_w, &verify_h, &verify_c)) {
          if (error_message) {
            *error_message = "processcvu runtime logical output geometry missing";
          }
          return false;
        }
        binding.shape = logical->shape;
      }
    }
    int verify_w = 0;
    int verify_h = 0;
    int verify_c = 0;
    if (!processcvu_tensor_dims_whc_from_shape_layout_local(binding.shape, binding.layout,
                                                            &verify_w, &verify_h, &verify_c)) {
      if (error_message) {
        *error_message = "processcvu runtime output geometry missing";
      }
      return false;
    }
    if (binding.segment_name.empty()) {
      if (const auto* physical = processcvu_find_physical_output_by_index_local(
              stage, binding.physical_output_index)) {
        binding.segment_name = physical->segment_name;
      }
    }
    if (runtime_output_name == "output_tessellated_image") {
      if (const auto* physical = processcvu_find_physical_output_by_index_local(
              stage, binding.physical_output_index)) {
        const auto dense_size = shape_size_bytes_local(binding.shape, binding.dtype);
        if (physical->size_bytes > 0U && (transport_kind == ProcessCvuOutputTransportKind::Packed ||
                                          dense_size == 0U || physical->size_bytes > dense_size)) {
          binding.contract_kind = simaai::gst::CvuOutputBinding::ContractKind::Packed;
          binding.size_bytes = physical->size_bytes;
          if (binding.segment_name.empty()) {
            binding.segment_name = physical->segment_name;
          }
        }
      }
    }
    if (strict_preproc_single_output && binding.dispatcher_name == payload.primary_output_name) {
      binding.output_slot = 0;
      binding.logical_output_index = 0;
      binding.physical_output_index = 0;
    }
    out->runtime_output_bindings.push_back(std::move(binding));
  }

  out->exposed_output_bindings.reserve(stage.output_order.size());
  for (const auto& route : stage.output_order) {
    const auto* logical =
        strict_preproc_single_output
            ? processcvu_find_logical_output_by_index_local(stage, route.logical_output_index)
            : processcvu_find_logical_output_by_index_or_slot_local(
                  stage, route.logical_output_index, route.output_slot);
    if (!logical) {
      if (error_message) {
        *error_message = "processcvu exposed logical output missing";
      }
      return false;
    }
    simaai::gst::CvuOutputBinding binding;
    binding.output_slot = route.output_slot;
    binding.logical_output_index = logical->logical_index;
    binding.physical_output_index = logical->physical_index;
    binding.byte_offset = logical->byte_offset >= 0 ? logical->byte_offset : 0;
    binding.dispatcher_name =
        !route.cm_output_name.empty()
            ? route.cm_output_name
            : (!logical->logical_name.empty() ? logical->logical_name : logical->backend_name);
    binding.segment_name = !route.segment_name.empty() ? route.segment_name : logical->segment_name;
    binding.dtype = logical->dtype;
    binding.layout = logical->layout;
    int verify_w = 0;
    int verify_h = 0;
    int verify_c = 0;
    if (!processcvu_tensor_dims_whc_from_shape_layout_local(logical->shape, logical->layout,
                                                            &verify_w, &verify_h, &verify_c)) {
      if (error_message) {
        *error_message = "processcvu exposed logical output geometry missing";
      }
      return false;
    }
    binding.shape = logical->shape;
    out->exposed_output_bindings.push_back(std::move(binding));
  }
  if (strict_preproc_single_output && out->exposed_output_bindings.size() != 1U) {
    if (error_message) {
      *error_message = "processcvu preproc single-output handoff built unexpected exposed outputs";
    }
    return false;
  }
  out->is_preproc_graph = processcvu_canonical_graph_name_local(graph_family) == "preproc";
  out->preproc_single_output_handoff = payload.preproc_single_output_handoff;
  out->single_input_mode = out->input_bindings.size() <= 1U;
  if (!out->input_bindings.empty()) {
    bool consistent = true;
    int first_w = 0;
    int first_h = 0;
    int first_c = 0;
    if (!processcvu_tensor_dims_whc_from_shape_layout_local(out->input_bindings[0].shape,
                                                            out->input_bindings[0].layout, &first_w,
                                                            &first_h, &first_c)) {
      consistent = false;
    }
    for (const auto& binding : out->input_bindings) {
      int binding_w = 0;
      int binding_h = 0;
      int binding_c = 0;
      if (!processcvu_tensor_dims_whc_from_shape_layout_local(binding.shape, binding.layout,
                                                              &binding_w, &binding_h, &binding_c) ||
          binding_w != first_w || binding_h != first_h) {
        consistent = false;
        break;
      }
    }
    if (consistent) {
      out->global_input_dims = simaai::gst::Dims{first_w, first_h};
    }
  }
  return true;
}

GstCaps* build_processcvu_sink_caps_local(const simaai::gst::PreparedProcessCvuTypedConfig& cfg,
                                          const simaai::gst::CvuRoutingContract& routing) {
  if (routing.is_preproc_graph) {
    GstCaps* caps = gst_caps_new_empty_simple("video/x-raw");
    if (!cfg.input_img_type.empty()) {
      gst_caps_set_simple(caps, "format", G_TYPE_STRING, cfg.input_img_type.c_str(), nullptr);
    }
    return caps;
  }
  if (routing.input_bindings.size() != 1U || cfg.input_tensors.size() > 1U ||
      processcvu_graph_family_uses_packed_input_transport_local(cfg.graph_name)) {
    return make_generic_tensor_set_caps_local();
  }
  const auto* primary_input = routing.input_bindings.empty() ? nullptr : &routing.input_bindings[0];
  CapsTensorSpec tensor;
  tensor.dtype = primary_input ? primary_input->dtype : cfg.input_dtype;
  if (primary_input && !primary_input->shape.empty()) {
    tensor.shape = primary_input->shape;
  }
  if (tensor.shape.empty() && !cfg.input_tensors.empty()) {
    const auto& input_desc = cfg.input_tensors[0];
    const auto rank = std::min<std::uint32_t>(input_desc.shape.rank, SIMA_EV_MAX_RANK);
    tensor.shape.reserve(rank);
    for (std::uint32_t i = 0; i < rank; ++i) {
      tensor.shape.push_back(input_desc.shape.sizes[i]);
    }
  }
  const std::string caps_dtype =
      processcvu_caps_format_from_dtype_local(tensor.dtype, cfg.input_dtype);
  tensor.dtype = caps_dtype;
  if (tensor.shape.empty()) {
    return make_generic_tensor_set_caps_local();
  }
  return processmla_make_tensor_caps_from_tensor_local(tensor, caps_dtype, false);
}

GstCaps*
build_processcvu_src_caps_local(const simaai::gst::PreparedProcessCvuTypedConfig& cfg,
                                const simaai::gst::CvuRoutingContract& routing,
                                const simaai::gst::TensorBufferPublishContract& publish_contract,
                                const std::string& primary_output_name,
                                const bool primary_output_packed_caps) {
  const simaai::gst::CvuOutputBinding* primary = nullptr;
  for (const auto& output : routing.exposed_output_bindings) {
    if (output.dispatcher_name == primary_output_name) {
      primary = &output;
      break;
    }
  }
  if (primary_output_packed_caps || routing.exposed_output_bindings.size() != 1U ||
      publish_contract.logical_outputs.size() != 1U || publish_contract.output_order.size() > 1U) {
    return make_generic_tensor_set_caps_local();
  }
  const std::string dtype = (primary && !primary->dtype.empty())
                                ? primary->dtype
                                : (!cfg.out_dtype.empty() ? cfg.out_dtype : cfg.output_dtype);
  const auto* logical = single_publish_logical_output_local(publish_contract);
  if (!logical) {
    return make_generic_tensor_set_caps_local();
  }
  CapsTensorSpec tensor = caps_tensor_from_publish_logical_output_local(*logical);
  if (tensor.dtype.empty()) {
    tensor.dtype = dtype;
  }
  const std::string caps_dtype = processcvu_caps_format_from_dtype_local(dtype, cfg.output_dtype);
  tensor.dtype = caps_dtype;
  if (tensor.shape.empty()) {
    return make_generic_tensor_set_caps_local();
  }
  // Match the plugin-side ProcessCVU caps builder: tensor pad caps should advertise the
  // tokenized transport/format string rather than a semantic alias such as BF16.
  GstCaps* caps = processmla_make_tensor_caps_from_tensor_local(tensor, caps_dtype, false);
  processmla_add_shape_fields_to_caps_local(caps, tensor);
  return caps;
}

bool build_processcvu_prepared_stage_from_manifest_stage_local(
    const StageStaticSpec& stage, simaai::gst::ProcessCvuPreparedStage* out,
    std::string* error_message) {
  if (!out) {
    if (error_message) {
      *error_message = "processcvu prepared stage requires output storage";
    }
    return false;
  }
  simaai::gst::ProcessCvuPreparedStage prepared;
  prepared.stage_key = stage_key_from_stage_spec_local(stage);
  if (!build_processcvu_typed_config_from_manifest_stage_local(stage, &prepared.typed_config,
                                                               error_message)) {
    return false;
  }
  for (const auto& logical : stage.logical_inputs) {
    simaai::gst::ProcessCvuPreparedLogicalInput input;
    input.logical_index = logical.logical_index;
    input.backend_input_index = logical.backend_input_index;
    input.physical_index = logical.physical_index;
    input.shape = logical.shape;
    input.stride_bytes = logical.stride_bytes;
    input.byte_offset = logical.byte_offset;
    input.size_bytes = logical.size_bytes;
    input.dtype = logical.dtype;
    input.layout = logical.layout;
    input.logical_name = logical.logical_name;
    input.backend_name = logical.backend_name;
    input.segment_name = logical.segment_name;
    switch (logical.materialization_kind) {
    case TensorMaterializationKind::OffsetView:
      input.materialization_kind = simaai::gst::PreparedTensorMaterializationKind::OffsetView;
      break;
    case TensorMaterializationKind::Bf16LaneSplitRepack:
      input.materialization_kind =
          simaai::gst::PreparedTensorMaterializationKind::Bf16LaneSplitRepack;
      break;
    case TensorMaterializationKind::Unknown:
    case TensorMaterializationKind::Direct:
    default:
      input.materialization_kind = simaai::gst::PreparedTensorMaterializationKind::Direct;
      break;
    }
    if (logical.quant.has_value()) {
      simaai::gst::TensorBufferQuantView quant;
      quant.granularity =
          static_cast<int>(logical.quant->granularity == QuantGranularity::PerAxis ? 1 : 0);
      quant.axis = logical.quant->axis;
      quant.scales = logical.quant->scales;
      quant.zero_points = logical.quant->zero_points;
      input.quant = std::move(quant);
    }
    prepared.logical_inputs.push_back(std::move(input));
  }
  for (const auto& physical : stage.physical_inputs) {
    simaai::gst::ProcessCvuPreparedPhysicalInput input;
    input.physical_index = physical.physical_index;
    input.allocator_index = physical.allocator_index;
    input.source_physical_index = physical.source_physical_index;
    input.size_bytes = physical.size_bytes;
    input.source_byte_offset = physical.source_byte_offset;
    input.segment_name = physical.segment_name;
    prepared.physical_inputs.push_back(std::move(input));
  }
  if (!build_processcvu_routing_contract_from_manifest_stage_local(
          stage, &prepared.routing_contract, error_message)) {
    return false;
  }
  if (!build_publish_contract_from_manifest_stage_local(stage, &prepared.output_publish_contract,
                                                        error_message)) {
    return false;
  }
  attach_processcvu_dequant_qparams_to_publish_contract_local(stage, prepared.typed_config,
                                                              &prepared.output_publish_contract);
  simaai::gst::TensorBufferPreparedMetaTemplate meta_template;
  if (!simaai::gst::tensor_buffer_prepare_meta_template_from_contract(
          prepared.output_publish_contract, &meta_template, error_message)) {
    return false;
  }
  prepared.output_meta_template = std::move(meta_template);
  prepared.typed_config.runtime_output_names.clear();
  prepared.typed_config.runtime_output_names.reserve(
      prepared.routing_contract.runtime_output_bindings.size());
  for (const auto& binding : prepared.routing_contract.runtime_output_bindings) {
    prepared.typed_config.runtime_output_names.push_back(binding.dispatcher_name);
  }
  prepared.typed_config.published_output_names.clear();
  prepared.typed_config.published_output_names.reserve(
      prepared.output_publish_contract.logical_outputs.size());
  for (const auto& logical : prepared.output_publish_contract.logical_outputs) {
    prepared.typed_config.published_output_names.push_back(logical.logical_name);
  }
  prepared.primary_output_name = stage.processcvu.primary_output_name;
  prepared.primary_output_packed_caps =
      processcvu_primary_output_uses_packed_caps_from_payload_local(stage.processcvu);
  prepared.sink_caps =
      build_processcvu_sink_caps_local(prepared.typed_config, prepared.routing_contract);
  prepared.src_caps = build_processcvu_src_caps_local(
      prepared.typed_config, prepared.routing_contract, prepared.output_publish_contract,
      prepared.primary_output_name, prepared.primary_output_packed_caps);
  *out = std::move(prepared);
  return true;
}

bool build_processcvu_prepared_stage_from_stage_contract_local(
    const StageStaticSpec& stage, simaai::gst::ProcessCvuPreparedStage* out,
    std::string* error_message) {
  return build_processcvu_prepared_stage_from_manifest_stage_local(stage, out, error_message);
}

constexpr const char* kPreparedRuntimePackedSegmentNameLocal = "__tensorbuffer_packed_parent__";

bool checked_add_u64_local(const std::uint64_t a, const std::uint64_t b, std::uint64_t* out);

std::uint64_t non_negative_u64_local(const std::int64_t value) {
  return value > 0 ? static_cast<std::uint64_t>(value) : 0U;
}

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
  return graph_name;
}

bool processcvu_graph_family_uses_packed_input_transport_local(const std::string& graph_family) {
  const std::string canonical = processcvu_canonical_graph_name_local(graph_family);
  return canonical == "detessellate" || canonical == "detesscast" || canonical == "detessdequant";
}

bool processcvu_tensor_dims_whc_from_shape_layout_local(const std::vector<std::int64_t>& shape,
                                                        const std::string& layout, int* w, int* h,
                                                        int* c) {
  if (!w || !h || !c || shape.empty()) {
    return false;
  }
  *w = 0;
  *h = 0;
  *c = 0;
  const std::string norm = upper_copy_local(layout);
  if (shape.size() >= 4U && shape.front() == 1) {
    std::vector<std::int64_t> trimmed(shape.begin() + 1, shape.end());
    return processcvu_tensor_dims_whc_from_shape_layout_local(trimmed, layout, w, h, c);
  }
  if (shape.size() >= 3U) {
    const auto a = shape[shape.size() - 3U];
    const auto b = shape[shape.size() - 2U];
    const auto d = shape[shape.size() - 1U];
    if (a <= 0 || b <= 0 || d <= 0 || a > std::numeric_limits<int>::max() ||
        b > std::numeric_limits<int>::max() || d > std::numeric_limits<int>::max()) {
      return false;
    }
    if (norm.find("CHW") != std::string::npos) {
      *c = static_cast<int>(a);
      *h = static_cast<int>(b);
      *w = static_cast<int>(d);
    } else {
      *h = static_cast<int>(a);
      *w = static_cast<int>(b);
      *c = static_cast<int>(d);
    }
    return true;
  }
  if (shape.size() == 2U) {
    if (shape[0] <= 0 || shape[1] <= 0 || shape[0] > std::numeric_limits<int>::max() ||
        shape[1] > std::numeric_limits<int>::max()) {
      return false;
    }
    *h = static_cast<int>(shape[0]);
    *w = static_cast<int>(shape[1]);
    *c = 1;
    return true;
  }
  if (shape[0] <= 0 || shape[0] > std::numeric_limits<int>::max()) {
    return false;
  }
  *w = static_cast<int>(shape[0]);
  *h = 1;
  *c = 1;
  return true;
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
  if (shape.empty()) {
    if (error_message) {
      *error_message = "processcvu tile shape missing";
    }
    return false;
  }
  if (raw_tile_shape.empty()) {
    *out = shape;
    return true;
  }
  std::vector<std::int64_t> normalized(raw_tile_shape.begin(), raw_tile_shape.end());
  if (normalized.size() > shape.size()) {
    const std::size_t extra = normalized.size() - shape.size();
    for (std::size_t i = 0; i < extra; ++i) {
      if (normalized[i] != 1) {
        if (error_message) {
          *error_message = "tile_shape_rank_prefix_invalid";
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
      if (error_message) {
        *error_message = "processcvu tile shape invalid";
      }
      return false;
    }
  }
  *out = std::move(normalized);
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
  if (!processcvu_dtype_token_to_ev_local(dtype_token, &out->dtype)) {
    if (error_message) {
      *error_message = "processcvu dense tensor dtype invalid";
    }
    return false;
  }
  out->layout_kind = SIMA_EV_LAYOUT_STRIDED;
  out->storage.nbytes = size_bytes != 0U ? size_bytes : shape_size_bytes_local(shape, dtype_token);
  return tensorsemantics::fill_dense_strides(out->shape, normalized_layout, out->dtype,
                                             &out->layout.strided, error_message,
                                             "processcvu dense stride desc requires output storage",
                                             "processcvu dense stride dtype invalid");
}

bool processcvu_build_tiled_tensor_desc_local(
    const std::vector<std::int64_t>& shape, const std::vector<std::int64_t>& tile_shape,
    const std::string& dtype_token, const std::string& layout_token,
    const std::uint32_t tile_align_bytes, const std::uint64_t size_bytes, const bool c16_packed,
    sima_ev_tensor_desc* out, std::string* error_message) {
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
    if (tensorsemantics::find_shape_axis(out->shape, SIMA_EV_AXIS_C) >= 0 && c16_packed) {
      out->layout.tiled.flags &= ~static_cast<std::uint32_t>(SIMA_EV_TILED_FLAG_COMPACT_CHANNELS);
    }
    out->storage.nbytes = size_bytes != 0U ? size_bytes
                                           : tensorsemantics::generic_fixed_slot_tiled_size_bytes(
                                                 shape, tile_shape, dtype_token, tile_align_bytes);
    return out->storage.nbytes != 0U;
  }
  if (!tensorsemantics::fill_shape_desc(shape, normalized_layout, &out->shape, error_message,
                                        "processcvu shape desc requires output storage",
                                        "processcvu shape rank invalid",
                                        "processcvu shape dim invalid")) {
    return false;
  }
  if (!processcvu_dtype_token_to_ev_local(dtype_token, &out->dtype)) {
    if (error_message) {
      *error_message = "processcvu tiled tensor dtype invalid";
    }
    return false;
  }
  out->layout_kind = SIMA_EV_LAYOUT_TILED;
  out->storage.nbytes = size_bytes != 0U ? size_bytes : shape_size_bytes_local(shape, dtype_token);
  for (std::size_t i = 0; i < tile_shape.size(); ++i) {
    out->layout.tiled.tile_sizes[i] = tile_shape[i];
  }
  out->layout.tiled.tile_align_bytes = tile_align_bytes;
  if (tensorsemantics::find_shape_axis(out->shape, SIMA_EV_AXIS_C) >= 0) {
    out->layout.tiled.flags =
        c16_packed ? SIMA_EV_TILED_FLAG_NONE : SIMA_EV_TILED_FLAG_COMPACT_CHANNELS;
  } else {
    out->layout.tiled.flags = SIMA_EV_TILED_FLAG_NONE;
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
  (void)shape;
  return tensorsemantics::normalize_layout_token(fallback);
}

std::string processcvu_tensor_layout_token_from_desc_local(const sima_ev_tensor_desc& desc) {
  return tensorsemantics::layout_token_from_ev_tensor_desc(desc);
}

bool tensorbuffer_read_request_complete_local(const simaai::gst::TensorBufferReadRequest& request,
                                              std::string* error_message) {
  for (std::size_t i = 0; i < request.entries.size(); ++i) {
    const auto& entry = request.entries[i];
    if (entry.expected_size_bytes == 0U) {
      if (error_message) {
        *error_message = "graph-owned read request entry missing expected_size_bytes";
      }
      return false;
    }
    if (entry.physical_index < 0 && entry.segment_name.empty()) {
      if (error_message) {
        *error_message = "graph-owned read request entry missing carrier identity";
      }
      return false;
    }
  }
  return true;
}

bool tensorbuffer_publish_contract_complete_local(
    const simaai::gst::TensorBufferPublishContract& contract, std::string* error_message) {
  if (contract.logical_outputs.empty()) {
    if (error_message) {
      *error_message = "graph-owned publish contract missing logical outputs";
    }
    return false;
  }
  for (const auto& physical : contract.physical_outputs) {
    if (physical.physical_index < 0 || physical.segment_name.empty() || physical.size_bytes == 0U) {
      if (error_message) {
        *error_message = "graph-owned publish contract physical output incomplete";
      }
      return false;
    }
  }
  for (const auto& logical : contract.logical_outputs) {
    const bool has_explicit_logical_view =
        logical.layout != SIMA_TENSOR_SET_LAYOUT_UNKNOWN_V1 ||
        (!logical.axis_semantics.empty() && logical.axis_semantics_match_shape()) ||
        (!logical.stride_bytes.empty() && logical.shape.size() == logical.stride_bytes.size());
    if (logical.physical_index < 0 || logical.segment_name.empty() || logical.size_bytes == 0U ||
        logical.shape.empty() || logical.dtype == SIMA_TENSOR_SET_DTYPE_UNKNOWN_V1 ||
        !has_explicit_logical_view) {
      if (error_message) {
        *error_message = "graph-owned publish contract logical output incomplete";
      }
      return false;
    }
  }
  return true;
}

bool processmla_prepared_stage_complete_local(const simaai::gst::ProcessMlaPreparedStage& prepared,
                                              std::string* error_message) {
  if (prepared.runtime_cfg.model_path.empty()) {
    if (error_message) {
      *error_message = "graph-owned processmla missing model_path";
    }
    return false;
  }
  if (!tensorbuffer_read_request_complete_local(prepared.input_request, error_message)) {
    return false;
  }
  if (!tensorbuffer_publish_contract_complete_local(prepared.output_publish_contract,
                                                    error_message)) {
    return false;
  }
  if (!prepared.output_meta_template.has_value()) {
    if (error_message) {
      *error_message = "graph-owned processmla missing prepared meta template";
    }
    return false;
  }
  return true;
}

bool processcvu_prepared_stage_complete_local(const simaai::gst::ProcessCvuPreparedStage& prepared,
                                              std::string* error_message) {
  if (prepared.typed_config.graph_name.empty()) {
    if (error_message) {
      *error_message = "graph-owned processcvu missing graph_name";
    }
    return false;
  }
  if (prepared.routing_contract.input_bindings.empty() ||
      prepared.routing_contract.exposed_output_bindings.empty()) {
    if (error_message) {
      *error_message = "graph-owned processcvu routing contract incomplete";
    }
    return false;
  }
  for (const auto& binding : prepared.routing_contract.input_bindings) {
    if (binding.source_physical_index < 0 || binding.segment_name.empty() ||
        binding.shape.empty() || binding.dtype.empty()) {
      if (error_message) {
        *error_message = "graph-owned processcvu input binding incomplete";
      }
      return false;
    }
  }
  for (const auto& binding : prepared.routing_contract.exposed_output_bindings) {
    if (binding.physical_output_index < 0 || binding.segment_name.empty() ||
        !binding.size_bytes.has_value() || *binding.size_bytes == 0U || binding.shape.empty() ||
        binding.dtype.empty()) {
      if (error_message) {
        *error_message = "graph-owned processcvu output binding incomplete";
      }
      return false;
    }
  }
  if (!tensorbuffer_publish_contract_complete_local(prepared.output_publish_contract,
                                                    error_message)) {
    return false;
  }
  if (!prepared.output_meta_template.has_value()) {
    if (error_message) {
      *error_message = "graph-owned processcvu missing prepared meta template";
    }
    return false;
  }
  return true;
}

bool dequant_prepared_stage_complete_local(const simaai::gst::DequantPreparedStage& prepared,
                                           std::string* error_message) {
  if (prepared.input_dtype.empty() || prepared.input_elem_bytes == 0U ||
      prepared.tensor_shape.empty() || prepared.tensor_layout.empty() ||
      prepared.required_input_bytes == 0U || prepared.required_output_bytes == 0U ||
      prepared.quant_spans.empty()) {
    if (error_message) {
      *error_message = "graph-owned dequant prepared stage incomplete";
    }
    return false;
  }
  if (!tensorbuffer_publish_contract_complete_local(prepared.identity_publish_contract,
                                                    error_message)) {
    return false;
  }
  if (!prepared.prepared_meta_template.has_value()) {
    if (error_message) {
      *error_message = "graph-owned dequant missing prepared meta template";
    }
    return false;
  }
  return true;
}

} // namespace

bool processcvu_build_stage_tensor_descs_local(const StageStaticSpec& stage,
                                               simaai::gst::PreparedProcessCvuTypedConfig* cfg,
                                               std::string* error_message) {
  if (!cfg) {
    if (error_message) {
      *error_message = "processcvu stage tensor descs require config output";
    }
    return false;
  }
  cfg->input_tensors.clear();
  cfg->output_tensors.clear();
  const auto& payload = stage.processcvu;
  if (!payload.input_tensors.empty() && !payload.output_tensors.empty()) {
    cfg->input_tensors = payload.input_tensors;
    cfg->output_tensors = payload.output_tensors;
    cfg->num_in_tensor = static_cast<int32_t>(cfg->input_tensors.size());
    return true;
  }
  for (const auto& logical : stage.logical_inputs) {
    sima_ev_tensor_desc desc{};
    if (!processcvu_build_dense_tensor_desc_local(logical.shape, logical.dtype, logical.layout,
                                                  logical.size_bytes, &desc, error_message)) {
      return false;
    }
    cfg->input_tensors.push_back(desc);
  }
  for (const auto& logical : stage.logical_outputs) {
    sima_ev_tensor_desc desc{};
    if (!processcvu_build_dense_tensor_desc_local(logical.shape, logical.dtype, logical.layout,
                                                  logical.size_bytes, &desc, error_message)) {
      return false;
    }
    cfg->output_tensors.push_back(desc);
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
                                                  simaai::gst::PreparedProcessCvuTypedConfig* cfg,
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
  const bool input_is_tiled =
      canonical == "detessellate" || canonical == "detesscast" || canonical == "detessdequant";
  const bool output_is_tiled = canonical == "tessellate" || canonical == "casttess" ||
                               canonical == "quanttess" || canonical == "quantizetessellate";
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
          !processcvu_build_tiled_tensor_desc_local(input_shape, tile_shape, input_dtype,
                                                    input_layout, 0U, input.size_bytes,
                                                    io.c16_packed, &input_desc, error_message)) {
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
              output.size_bytes, io.c16_packed, &output_desc, error_message)) {
        return false;
      }
    } else if (!processcvu_build_dense_tensor_desc_local(output_shape, output_dtype, output_layout,
                                                         output.size_bytes, &output_desc,
                                                         error_message)) {
      return false;
    }
    cfg->input_tensors.push_back(input_desc);
    cfg->output_tensors.push_back(output_desc);
  }
  cfg->num_in_tensor = static_cast<int32_t>(cfg->input_tensors.size());
  return true;
}

bool expand_processmla_output_descs_from_logical_outputs_local(
    std::vector<ProcessMlaOutputDesc>* outputs,
    const std::vector<ProcessMlaLogicalOutputDesc>& logical_outputs) {
  if (!outputs) {
    return false;
  }
  outputs->clear();
  outputs->reserve(logical_outputs.size());
  for (std::size_t i = 0; i < logical_outputs.size(); ++i) {
    const auto& logical = logical_outputs[i];
    ProcessMlaOutputDesc output;
    output.name =
        !logical.backend_name.empty()
            ? logical.backend_name
            : (!logical.logical_name.empty() ? logical.logical_name : logical.segment_name);
    if (output.name.empty()) {
      output.name = "mla_output_" + std::to_string(i);
    }
    output.size = logical.size_bytes;
    output.source_output_index = logical.backend_output_index >= 0
                                     ? static_cast<std::size_t>(logical.backend_output_index)
                                     : i;
    output.source_byte_offset =
        logical.byte_offset > 0 ? static_cast<std::uint64_t>(logical.byte_offset) : 0U;
    outputs->push_back(std::move(output));
  }
  return true;
}

bool build_processmla_output_descs_from_physical_specs_local(
    const std::vector<PhysicalBufferStaticSpec>& physical_outputs,
    const std::vector<ProcessMlaLogicalOutputDesc>& logical_outputs,
    std::vector<ProcessMlaOutputDesc>* outputs) {
  if (!outputs) {
    return false;
  }
  outputs->clear();
  if (!physical_outputs.empty()) {
    std::size_t slot_count = physical_outputs.size();
    for (const auto& physical : physical_outputs) {
      if (physical.physical_index >= 0) {
        slot_count = std::max<std::size_t>(slot_count,
                                           static_cast<std::size_t>(physical.physical_index) + 1U);
      }
    }
    outputs->assign(slot_count, ProcessMlaOutputDesc{});
    for (std::size_t i = 0; i < physical_outputs.size(); ++i) {
      const auto& physical = physical_outputs[i];
      const std::size_t slot =
          physical.physical_index >= 0 ? static_cast<std::size_t>(physical.physical_index) : i;
      auto& output = (*outputs)[slot];
      output.name = !physical.segment_name.empty() ? physical.segment_name
                                                   : ("mla_output_" + std::to_string(slot));
      output.size = static_cast<std::size_t>(physical.size_bytes);
      output.source_output_index = physical.source_physical_index >= 0
                                       ? static_cast<std::size_t>(physical.source_physical_index)
                                       : slot;
      output.source_byte_offset = physical.source_byte_offset > 0
                                      ? static_cast<std::uint64_t>(physical.source_byte_offset)
                                      : 0U;
    }
  }
  for (std::size_t i = 0; i < logical_outputs.size(); ++i) {
    const auto& logical = logical_outputs[i];
    const std::size_t slot =
        logical.physical_index >= 0 ? static_cast<std::size_t>(logical.physical_index) : i;
    if (!physical_outputs.empty() && slot >= outputs->size()) {
      return false;
    }
    if (slot >= outputs->size()) {
      outputs->resize(slot + 1U);
    }
    auto& output = (*outputs)[slot];
    if (output.name.empty()) {
      output.name =
          !logical.segment_name.empty()
              ? logical.segment_name
              : (!logical.backend_name.empty()
                     ? logical.backend_name
                     : (!logical.logical_name.empty() ? logical.logical_name
                                                      : ("mla_output_" + std::to_string(slot))));
    }
    if (output.source_output_index == 0U && logical.backend_output_index >= 0 &&
        output.size == 0U) {
      output.source_output_index = static_cast<std::size_t>(logical.backend_output_index);
    }
    const std::uint64_t logical_offset =
        logical.byte_offset > 0 ? static_cast<std::uint64_t>(logical.byte_offset) : 0U;
    std::uint64_t span_end = logical_offset;
    if (checked_add_u64_local(logical_offset, logical.size_bytes, &span_end)) {
      output.size = std::max<std::size_t>(output.size, static_cast<std::size_t>(span_end));
    }
  }
  if (outputs->empty()) {
    return expand_processmla_output_descs_from_logical_outputs_local(outputs, logical_outputs);
  }
  return true;
}

bool build_processmla_output_descs_from_physical_tensors_local(
    const std::vector<MpkTensorContract>& physical_tensors,
    const std::vector<ProcessMlaLogicalOutputDesc>& logical_outputs,
    std::vector<ProcessMlaOutputDesc>* outputs) {
  if (!outputs) {
    return false;
  }
  outputs->clear();
  if (!physical_tensors.empty()) {
    std::size_t slot_count = physical_tensors.size();
    for (const auto& tensor : physical_tensors) {
      if (tensor.physical_index >= 0) {
        slot_count =
            std::max<std::size_t>(slot_count, static_cast<std::size_t>(tensor.physical_index) + 1U);
      }
    }
    outputs->assign(slot_count, ProcessMlaOutputDesc{});
    for (std::size_t i = 0; i < physical_tensors.size(); ++i) {
      const auto& tensor = physical_tensors[i];
      const std::size_t slot =
          tensor.physical_index >= 0 ? static_cast<std::size_t>(tensor.physical_index) : i;
      auto& output = (*outputs)[slot];
      output.name =
          !tensor.segment_name.empty()
              ? tensor.segment_name
              : (!tensor.name.empty() ? tensor.name : ("mla_output_" + std::to_string(slot)));
      output.size = tensor.size_bytes;
      output.source_output_index = tensor.source_physical_index >= 0
                                       ? static_cast<std::size_t>(tensor.source_physical_index)
                                       : slot;
      output.source_byte_offset = tensor.source_byte_offset > 0
                                      ? static_cast<std::uint64_t>(tensor.source_byte_offset)
                                      : 0U;
    }
  }
  for (std::size_t i = 0; i < logical_outputs.size(); ++i) {
    const auto& logical = logical_outputs[i];
    const std::size_t slot =
        logical.physical_index >= 0 ? static_cast<std::size_t>(logical.physical_index) : i;
    if (!physical_tensors.empty() && slot >= outputs->size()) {
      return false;
    }
    if (slot >= outputs->size()) {
      outputs->resize(slot + 1U);
    }
    auto& output = (*outputs)[slot];
    if (output.name.empty()) {
      output.name =
          !logical.segment_name.empty()
              ? logical.segment_name
              : (!logical.backend_name.empty()
                     ? logical.backend_name
                     : (!logical.logical_name.empty() ? logical.logical_name
                                                      : ("mla_output_" + std::to_string(slot))));
    }
    if (output.source_output_index == 0U && logical.backend_output_index >= 0 &&
        output.size == 0U) {
      output.source_output_index = static_cast<std::size_t>(logical.backend_output_index);
    }
    const std::uint64_t logical_offset =
        logical.byte_offset > 0 ? static_cast<std::uint64_t>(logical.byte_offset) : 0U;
    std::uint64_t span_end = logical_offset;
    if (checked_add_u64_local(logical_offset, logical.size_bytes, &span_end)) {
      output.size = std::max<std::size_t>(output.size, static_cast<std::size_t>(span_end));
    }
  }
  if (outputs->empty()) {
    return expand_processmla_output_descs_from_logical_outputs_local(outputs, logical_outputs);
  }
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

const LogicalTensorStaticSpec*
processcvu_find_logical_output_by_index_local(const StageStaticSpec& stage, int logical_index) {
  const auto it = std::find_if(stage.logical_outputs.begin(), stage.logical_outputs.end(),
                               [&](const LogicalTensorStaticSpec& logical) {
                                 return logical.logical_index == logical_index;
                               });
  return it == stage.logical_outputs.end() ? nullptr : &*it;
}

const LogicalTensorStaticSpec*
processcvu_find_logical_output_by_index_or_slot_local(const StageStaticSpec& stage,
                                                      int logical_index, int output_slot) {
  if (const auto* logical = processcvu_find_logical_output_by_index_local(stage, logical_index)) {
    return logical;
  }
  const auto it = std::find_if(
      stage.logical_outputs.begin(), stage.logical_outputs.end(),
      [&](const LogicalTensorStaticSpec& logical) { return logical.output_slot == output_slot; });
  return it == stage.logical_outputs.end() ? nullptr : &*it;
}

const PhysicalBufferStaticSpec*
processcvu_find_physical_input_by_index_local(const StageStaticSpec& stage, int physical_index) {
  const auto it = std::find_if(stage.physical_inputs.begin(), stage.physical_inputs.end(),
                               [&](const PhysicalBufferStaticSpec& physical) {
                                 return physical.physical_index == physical_index;
                               });
  return it == stage.physical_inputs.end() ? nullptr : &*it;
}

const PhysicalBufferStaticSpec*
processcvu_find_physical_output_by_index_local(const StageStaticSpec& stage, int physical_index) {
  const auto it = std::find_if(stage.physical_outputs.begin(), stage.physical_outputs.end(),
                               [&](const PhysicalBufferStaticSpec& physical) {
                                 return physical.physical_index == physical_index;
                               });
  return it == stage.physical_outputs.end() ? nullptr : &*it;
}

std::string preferred_tensor_name_local(const MpkTensorContract& tensor, const std::size_t index,
                                        const std::string& fallback_prefix) {
  if (!tensor.name.empty()) {
    return tensor.name;
  }
  if (!tensor.segment_name.empty()) {
    return tensor.segment_name;
  }
  return fallback_prefix + std::to_string(index);
}

simaai::gst::TensorBufferPublishContract
build_publish_contract_from_runtime_config_local(const ProcessMlaRuntimeConfig& runtime_cfg) {
  simaai::gst::TensorBufferPublishContract contract;
  contract.stage_key = runtime_cfg.stage_key;
  std::unordered_map<int, std::size_t> physical_index_to_slot;
  contract.physical_outputs.reserve(runtime_cfg.outputs.size());
  for (std::size_t i = 0; i < runtime_cfg.outputs.size(); ++i) {
    const auto& runtime_output = runtime_cfg.outputs[i];
    simaai::gst::TensorBufferPublishPhysicalOutput physical;
    physical.physical_index = static_cast<int>(i);
    physical.size_bytes = runtime_output.size;
    physical.segment_name =
        !runtime_output.name.empty() ? runtime_output.name : ("mla_output_" + std::to_string(i));
    physical_index_to_slot.emplace(physical.physical_index, contract.physical_outputs.size());
    contract.physical_outputs.push_back(std::move(physical));
  }
  for (std::size_t i = 0; i < runtime_cfg.logical_outputs.size(); ++i) {
    const auto& logical = runtime_cfg.logical_outputs[i];
    simaai::gst::TensorBufferPublishLogicalOutput output;
    output.logical_index = logical.logical_index >= 0 ? logical.logical_index : static_cast<int>(i);
    output.physical_index =
        logical.physical_index >= 0 ? logical.physical_index : static_cast<int>(i);
    output.memory_index = logical.memory_index >= 0 ? logical.memory_index : output.physical_index;
    output.backend_output_index =
        logical.backend_output_index >= 0 ? logical.backend_output_index : static_cast<int>(i);
    output.route_slot = logical.route_slot >= 0 ? logical.route_slot : static_cast<int>(i);
    output.logical_name = logical.logical_name;
    output.backend_name = logical.backend_name;
    output.segment_name =
        !logical.segment_name.empty()
            ? logical.segment_name
            : (!logical.logical_name.empty()
                   ? logical.logical_name
                   : (!logical.backend_name.empty() ? logical.backend_name
                                                    : ("mla_output_" + std::to_string(i))));
    output.byte_offset = logical.byte_offset;
    output.shape = logical.shape;
    output.stride_bytes = logical.stride_bytes;
    output.dtype = tensorbuffer_dtype_from_token_local(logical.dtype);
    output.layout = tensorbuffer_layout_from_token_local(logical.layout);
    if (output.stride_bytes.empty() && !output.shape.empty() &&
        output.dtype != SIMA_TENSOR_SET_DTYPE_UNKNOWN_V1) {
      output.stride_bytes = contiguous_stride_bytes_local(output.shape, logical.dtype);
    }
    const auto logical_size_bytes = shape_size_bytes_local(logical.shape, logical.dtype);
    output.size_bytes = logical_size_bytes != 0U ? logical_size_bytes : logical.size_bytes;
    if (logical.has_quant) {
      simaai::gst::TensorBufferQuantView quant;
      quant.granularity = logical.quant_granularity;
      quant.axis = logical.quant_axis;
      quant.scales = logical.quant_scales;
      quant.zero_points = logical.quant_zero_points;
      output.quant = std::move(quant);
    }
    const auto physical_index = output.physical_index;
    const std::uint64_t span_end = non_negative_u64_local(output.byte_offset) + output.size_bytes;
    auto it = physical_index_to_slot.find(physical_index);
    if (it == physical_index_to_slot.end()) {
      simaai::gst::TensorBufferPublishPhysicalOutput physical;
      physical.physical_index = physical_index;
      physical.size_bytes = span_end;
      physical.segment_name = output.segment_name;
      physical_index_to_slot.emplace(physical_index, contract.physical_outputs.size());
      contract.physical_outputs.push_back(std::move(physical));
    } else {
      auto& physical = contract.physical_outputs[it->second];
      physical.size_bytes = std::max<std::uint64_t>(physical.size_bytes, span_end);
    }
    contract.logical_outputs.push_back(std::move(output));
  }
  contract.output_order.reserve(contract.logical_outputs.size());
  for (std::size_t i = 0; i < contract.logical_outputs.size(); ++i) {
    const auto& logical = contract.logical_outputs[i];
    simaai::gst::TensorBufferPublishOutputRoute route;
    route.output_slot = logical.route_slot;
    route.logical_output_index = static_cast<int>(i);
    route.cm_output_name =
        !logical.logical_name.empty() ? logical.logical_name : logical.backend_name;
    route.segment_name = logical.segment_name;
    contract.output_order.push_back(std::move(route));
  }
  return contract;
}

bool build_processcvu_prepared_stage_from_graph_io_local(const StageStaticSpec& original_stage,
                                                         const MpkGraphNode& graph_node,
                                                         const std::string& stage_key,
                                                         const GraphProcessCvuIoData& io,
                                                         simaai::gst::ProcessCvuPreparedStage* out,
                                                         std::string* error_message) {
  if (!out) {
    if (error_message) {
      *error_message = "graph processcvu prepared stage requires output storage";
    }
    return false;
  }
  if (io.input_tensors.empty() || io.output_tensors.empty()) {
    if (error_message) {
      *error_message = "graph processcvu IO synthesis requires non-empty tensors";
    }
    return false;
  }

  const auto& payload = original_stage.processcvu;
  simaai::neat::GraphProcessCvuStageRequest request;
  request.stage_key = stage_key;
  request.graph_name = processcvu_canonical_graph_name_local(
      !payload.graph_name.empty()
          ? payload.graph_name
          : (!payload.graph_family.empty() ? payload.graph_family : graph_node.canonical_op));
  request.requested_run_target =
      payload.requested_run_target.empty() ? payload.run_target : payload.requested_run_target;
  request.run_target = payload.run_target.empty() ? std::string("AUTO") : payload.run_target;
  request.resolved_exec_backend =
      payload.resolved_exec_backend.empty() ? std::string("EVXX") : payload.resolved_exec_backend;
  request.run_target_resolution_reason = payload.run_target_resolution_reason;
  request.graph_id = payload.graph_id;
  request.batch_size = payload.batch_size;
  request.round_off = payload.round_off;
  request.byte_align = payload.byte_align;
  request.aspect_ratio = payload.aspect_ratio;
  request.normalize = payload.normalize;
  request.tessellate = payload.tessellate;
  request.scaled_width = payload.scaled_width;
  request.scaled_height = payload.scaled_height;
  request.input_stride = payload.input_stride;
  request.output_stride = payload.output_stride;
  request.input_offset = payload.input_offset;
  request.input_img_type = payload.input_img_type;
  request.output_img_type = payload.output_img_type;
  request.scaling_type = payload.scaling_type;
  request.padding_type = payload.padding_type;
  request.input_dtype = payload.input_dtype;
  request.output_dtype = payload.output_dtype;
  request.canonical_input_dtype = io.canonical_input_dtype;
  request.canonical_output_dtype = io.canonical_output_dtype;
  request.input_slot_name = io.input_slot_name;
  request.runtime_output_slot_names = io.runtime_output_slot_names;
  request.runtime_output_logical_layout_list = payload.runtime_output_logical_layout_list;
  for (const auto value : payload.dq_scale_list) {
    request.dq_scale_array.push_back(static_cast<float>(value));
  }
  for (const auto value : payload.dq_zp_list) {
    request.dq_zp_array.push_back(value);
  }
  request.slice_shapes = io.slice_shapes;
  request.c16_packed_io = io.c16_packed;
  request.input_tensors.reserve(io.input_tensors.size());
  for (const auto& tensor : io.input_tensors) {
    request.input_tensors.push_back(bridge_graph_tensor_contract_from_mpk_local(tensor));
  }
  request.output_tensors.reserve(io.output_tensors.size());
  for (const auto& tensor : io.output_tensors) {
    request.output_tensors.push_back(bridge_graph_tensor_contract_from_mpk_local(tensor));
  }

  return simaai::neat::build_graph_processcvu_prepared_stage(request, out, error_message);
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

std::string resolve_model_path_from_pack_root_local(const std::filesystem::path& pack_root,
                                                    const std::string& executable) {
  if (executable.empty()) {
    return pack_root.string();
  }
  std::filesystem::path exec_path(executable);
  if (exec_path.is_absolute()) {
    return exec_path.string();
  }
  std::error_code ec;
  const auto direct = pack_root / exec_path;
  if (std::filesystem::exists(direct, ec)) {
    return direct.string();
  }
  const auto share_path = pack_root / "share" / exec_path;
  if (std::filesystem::exists(share_path, ec)) {
    return share_path.string();
  }
  const auto bin_path = pack_root / "bin" / exec_path;
  if (std::filesystem::exists(bin_path, ec)) {
    return bin_path.string();
  }
  return direct.string();
}

bool build_processmla_prepared_stage_from_graph_local(const MpkContract& contract,
                                                      const MpkGraphNode& graph_node,
                                                      const std::string& stage_key,
                                                      simaai::gst::ProcessMlaPreparedStage* out,
                                                      std::string* error_message) {
  if (!out) {
    if (error_message) {
      *error_message = "graph processmla prepared stage requires output storage";
    }
    return false;
  }
  const auto* stage = get_stage_io_contract(contract, graph_node.name);
  const auto* mla_stage = stage ? stage : get_mla_stage_io_contract(contract);
  if (!mla_stage) {
    if (error_message) {
      *error_message = "graph processmla stage missing MLA contract";
    }
    return false;
  }
  auto logical_inputs = get_mla_boundary_logical_inputs_contract(contract);
  if (logical_inputs.empty()) {
    logical_inputs = mla_stage->input_tensors;
  }
  auto physical_inputs = get_mla_boundary_physical_inputs_contract(contract);
  if (physical_inputs.empty()) {
    physical_inputs = mla_stage->input_tensors;
  }
  auto logical_outputs = get_mla_logical_outputs_contract(contract);
  if (logical_outputs.empty()) {
    logical_outputs = mla_stage->output_tensors;
  }
  auto physical_outputs = get_mla_boundary_physical_outputs_contract(contract);
  if (physical_outputs.empty()) {
    physical_outputs = mla_stage->output_tensors;
  }
  if (logical_inputs.empty() || physical_inputs.empty() || logical_outputs.empty()) {
    if (error_message) {
      *error_message = "graph processmla stage missing logical tensors";
    }
    return false;
  }

  simaai::neat::GraphProcessMlaStageRequest request;
  request.stage_key = stage_key;
  request.model_path = mla_stage->executable;
  request.batch_size = mla_stage->batch_size;
  request.batch_model = mla_stage->batch_sz_model;

  const auto& dispatcher_inputs =
      !mla_stage->input_tensors.empty() ? mla_stage->input_tensors : logical_inputs;
  request.dispatcher_inputs.reserve(dispatcher_inputs.size());
  for (const auto& tensor : dispatcher_inputs) {
    request.dispatcher_inputs.push_back(bridge_graph_tensor_contract_from_mpk_local(tensor));
  }
  request.logical_inputs.reserve(logical_inputs.size());
  for (const auto& tensor : logical_inputs) {
    request.logical_inputs.push_back(bridge_graph_tensor_contract_from_mpk_local(tensor));
  }
  request.physical_inputs.reserve(physical_inputs.size());
  for (const auto& tensor : physical_inputs) {
    request.physical_inputs.push_back(bridge_graph_tensor_contract_from_mpk_local(tensor));
  }
  request.stage_outputs.reserve(physical_outputs.size());
  for (const auto& tensor : physical_outputs) {
    request.stage_outputs.push_back(bridge_graph_tensor_contract_from_mpk_local(tensor));
  }
  request.logical_outputs.reserve(logical_outputs.size());
  for (const auto& tensor : logical_outputs) {
    request.logical_outputs.push_back(bridge_graph_tensor_contract_from_mpk_local(tensor));
  }
  if (mla_stage->quant.has_value()) {
    simaai::neat::GraphQuantContract quant;
    quant.scales = mla_stage->quant->scales;
    quant.zero_points = mla_stage->quant->zero_points;
    quant.axis = mla_stage->quant->axis;
    request.output_quant = std::move(quant);
  }

  return simaai::neat::build_graph_processmla_prepared_stage(request, out, error_message);
}

bool build_physical_group_offsets_from_tensor_views_local(
    const std::vector<MpkTensorContract>& tensors,
    std::unordered_map<int, std::uint64_t>* base_offsets, std::uint64_t* total_size_bytes,
    std::size_t* physical_group_count, std::string* error_message) {
  if (!base_offsets || !total_size_bytes || !physical_group_count) {
    if (error_message) {
      *error_message = "tensor view collapse requires output storage";
    }
    return false;
  }
  base_offsets->clear();
  *total_size_bytes = 0U;
  *physical_group_count = 0U;
  if (tensors.empty()) {
    return true;
  }

  std::vector<int> ordered_keys;
  std::unordered_map<int, std::uint64_t> span_bytes_by_key;
  for (std::size_t i = 0; i < tensors.size(); ++i) {
    const auto& tensor = tensors[i];
    if (tensor.size_bytes == 0U) {
      if (error_message) {
        *error_message = "batched MLA collapse requires non-zero tensor sizes";
      }
      return false;
    }
    const int key = tensor.physical_index >= 0 ? tensor.physical_index : static_cast<int>(i);
    const std::uint64_t local_offset = non_negative_u64_local(tensor.byte_offset);
    std::uint64_t span_end = 0U;
    if (!checked_add_u64_local(local_offset, tensor.size_bytes, &span_end)) {
      if (error_message) {
        *error_message = "batched MLA collapse span overflow";
      }
      return false;
    }
    if (span_bytes_by_key.emplace(key, span_end).second) {
      ordered_keys.push_back(key);
    } else {
      span_bytes_by_key[key] = std::max(span_bytes_by_key[key], span_end);
    }
  }

  for (const int key : ordered_keys) {
    (*base_offsets)[key] = *total_size_bytes;
    if (!checked_add_u64_local(*total_size_bytes, span_bytes_by_key[key], total_size_bytes)) {
      if (error_message) {
        *error_message = "batched MLA collapse total size overflow";
      }
      return false;
    }
  }
  *physical_group_count = ordered_keys.size();
  return true;
}

bool collapse_batched_mla_tensor_views_to_single_physical_local(
    const int batch_size, const std::string& stage_key, const char* label,
    std::vector<MpkTensorContract>* tensors, std::string* error_message) {
  if (!tensors) {
    if (error_message) {
      *error_message = "batched MLA tensor view collapse requires tensor storage";
    }
    return false;
  }
  if (batch_size <= 1 || tensors->empty()) {
    return true;
  }

  std::unordered_map<int, std::uint64_t> base_offsets;
  std::uint64_t total_size_bytes = 0U;
  std::size_t physical_group_count = 0U;
  if (!build_physical_group_offsets_from_tensor_views_local(
          *tensors, &base_offsets, &total_size_bytes, &physical_group_count, error_message)) {
    return false;
  }
  if (physical_group_count <= 1U) {
    return true;
  }

  prepared_runtime_warn_log_local(
      "collapsing batched MLA tensor views stage=%s label=%s batch=%d physical_groups=%zu "
      "total_bytes=%llu",
      stage_key.c_str(), label ? label : "unknown", batch_size, physical_group_count,
      static_cast<unsigned long long>(total_size_bytes));

  for (std::size_t i = 0; i < tensors->size(); ++i) {
    auto& tensor = (*tensors)[i];
    const int key =
        tensor.source_physical_index >= 0
            ? tensor.source_physical_index
            : (tensor.physical_index >= 0 ? tensor.physical_index : static_cast<int>(i));
    const auto it = base_offsets.find(key);
    if (it == base_offsets.end()) {
      if (error_message) {
        *error_message = "batched MLA tensor collapse missing source physical offset";
      }
      return false;
    }
    const std::uint64_t local_offset = non_negative_u64_local(tensor.source_byte_offset) +
                                       non_negative_u64_local(tensor.byte_offset);
    std::uint64_t collapsed_offset = 0U;
    if (!checked_add_u64_local(it->second, local_offset, &collapsed_offset) ||
        collapsed_offset > static_cast<std::uint64_t>(std::numeric_limits<std::int64_t>::max())) {
      if (error_message) {
        *error_message = "batched MLA tensor collapse offset overflow";
      }
      return false;
    }
    tensor.physical_index = 0;
    tensor.source_physical_index = 0;
    tensor.byte_offset = static_cast<std::int64_t>(collapsed_offset);
    tensor.source_byte_offset = 0;
    tensor.segment_name = kPreparedRuntimePackedSegmentNameLocal;
  }
  return true;
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

std::uint64_t shape_elements_or_zero_local(const std::vector<std::int64_t>& shape) {
  if (shape.empty()) {
    return 0U;
  }
  std::uint64_t elements = 1U;
  for (const auto dim : shape) {
    if (dim <= 0) {
      return 0U;
    }
    if (elements > (std::numeric_limits<std::uint64_t>::max() / static_cast<std::uint64_t>(dim))) {
      return 0U;
    }
    elements *= static_cast<std::uint64_t>(dim);
  }
  return elements;
}

std::vector<std::int64_t>
contiguous_stride_bytes_for_elem_local(const std::vector<std::int64_t>& shape,
                                       const std::int64_t elem_bytes) {
  std::vector<std::int64_t> strides(shape.size(), 0);
  if (shape.empty() || elem_bytes <= 0) {
    return strides;
  }
  std::int64_t stride = elem_bytes;
  for (std::size_t i = shape.size(); i-- > 0U;) {
    strides[i] = stride;
    if (shape[i] > 0 && stride <= (std::numeric_limits<std::int64_t>::max() / shape[i])) {
      stride *= shape[i];
    }
  }
  return strides;
}

std::vector<std::int64_t>
strides_or_contiguous_for_shape_local(const std::vector<std::int64_t>& stride_bytes,
                                      const std::vector<std::int64_t>& shape,
                                      const std::int64_t elem_bytes) {
  if (!shape.empty() && stride_bytes.size() == shape.size()) {
    return stride_bytes;
  }
  if (!shape.empty()) {
    const auto normalized =
        pipeline_internal::normalize_strides_rank_to_shape(stride_bytes, {}, shape, true);
    if (normalized.size() == shape.size()) {
      return normalized;
    }
  }
  return contiguous_stride_bytes_for_elem_local(shape, elem_bytes);
}

bool strides_match_contiguous_local(const std::vector<std::int64_t>& shape,
                                    const std::vector<std::int64_t>& strides,
                                    const std::int64_t elem_bytes) {
  return strides == contiguous_stride_bytes_for_elem_local(shape, elem_bytes);
}

bool shape_is_subset_compatible_local(const std::vector<std::int64_t>& input_shape,
                                      const std::vector<std::int64_t>& output_shape) {
  if (input_shape.empty() || output_shape.empty() || input_shape.size() != output_shape.size()) {
    return false;
  }
  for (std::size_t i = 0; i < input_shape.size(); ++i) {
    if (input_shape[i] <= 0 || output_shape[i] <= 0 || output_shape[i] > input_shape[i]) {
      return false;
    }
  }
  return true;
}

bool tensor_physical_span_bytes_local(const std::vector<std::int64_t>& shape,
                                      const std::vector<std::int64_t>& stride_bytes,
                                      const std::uint64_t elem_bytes,
                                      const std::uint64_t logical_size_bytes,
                                      std::uint64_t* out_span_bytes) {
  if (!out_span_bytes) {
    return false;
  }
  *out_span_bytes = logical_size_bytes;
  if (shape.empty()) {
    return true;
  }
  if (shape.size() != stride_bytes.size()) {
    return false;
  }
  std::uint64_t max_offset = 0U;
  for (std::size_t i = 0; i < shape.size(); ++i) {
    if (shape[i] <= 0 || stride_bytes[i] < 0) {
      return false;
    }
    const auto dim_extent = static_cast<std::uint64_t>(shape[i] - 1);
    const auto stride = static_cast<std::uint64_t>(stride_bytes[i]);
    if (dim_extent > 0U && stride > (std::numeric_limits<std::uint64_t>::max() / dim_extent)) {
      return false;
    }
    const auto term = dim_extent * stride;
    if (!checked_add_u64_local(max_offset, term, &max_offset)) {
      return false;
    }
  }
  std::uint64_t span_end = 0U;
  if (!checked_add_u64_local(max_offset, elem_bytes, &span_end)) {
    return false;
  }
  *out_span_bytes = std::max(logical_size_bytes, span_end);
  return true;
}

bool build_packed_physical_offsets_from_static_specs_local(
    const std::vector<PhysicalBufferStaticSpec>& physicals,
    std::unordered_map<int, std::uint64_t>* offsets, std::uint64_t* total_size_bytes,
    std::string* error_message) {
  if (!offsets || !total_size_bytes) {
    if (error_message) {
      *error_message = "packed physical offsets require output storage";
    }
    return false;
  }
  offsets->clear();
  *total_size_bytes = 0U;
  for (std::size_t i = 0; i < physicals.size(); ++i) {
    const auto& physical = physicals[i];
    const int key = physical.physical_index >= 0 ? physical.physical_index : static_cast<int>(i);
    if (!offsets->emplace(key, *total_size_bytes).second) {
      if (error_message) {
        *error_message = "duplicate physical_index in packed physical offsets";
      }
      return false;
    }
    if (!checked_add_u64_local(*total_size_bytes, physical.size_bytes, total_size_bytes)) {
      if (error_message) {
        *error_message = "packed physical offsets overflow";
      }
      return false;
    }
  }
  return true;
}

bool resolve_packed_logical_offset_local(
    const std::unordered_map<int, std::uint64_t>& physical_offsets, const int physical_index,
    const int fallback_index, const std::int64_t byte_offset, std::uint64_t* out_byte_offset,
    std::string* error_message) {
  if (!out_byte_offset) {
    if (error_message) {
      *error_message = "packed logical offset requires output storage";
    }
    return false;
  }

  std::uint64_t base_offset = 0U;
  if (!physical_offsets.empty()) {
    auto it = physical_offsets.find(physical_index);
    if (it == physical_offsets.end() && fallback_index >= 0) {
      it = physical_offsets.find(fallback_index);
    }
    if (it == physical_offsets.end()) {
      if (error_message) {
        *error_message = "missing physical index for packed logical offset";
      }
      return false;
    }
    base_offset = it->second;
  }

  const auto local_offset = byte_offset >= 0 ? static_cast<std::uint64_t>(byte_offset) : 0U;
  if (!checked_add_u64_local(base_offset, local_offset, out_byte_offset)) {
    if (error_message) {
      *error_message = "packed logical offset overflow";
    }
    return false;
  }
  return true;
}

bool pack_publish_contract_for_single_runtime_segment_local(
    const simaai::gst::TensorBufferPublishContract& contract,
    simaai::gst::TensorBufferPublishContract* out, std::string* error_message) {
  if (!out) {
    if (error_message) {
      *error_message = "missing packed publish contract output";
    }
    return false;
  }

  *out = contract;
  if (contract.physical_outputs.size() <= 1U) {
    return true;
  }
  // Same gate as the runtime joiner: when the consumer's I/O contract demands
  // distinct physical segments (multi-IFM MLA dispatch with native
  // data.ifm.persistent.input_NN slots), do NOT coalesce. The publish contract
  // is already a copy of the input; leaving physical_outputs as-is preserves
  // the per-segment identity the dispatcher needs.
  if (contract.preserve_physical_segments) {
    return true;
  }

  std::unordered_map<int, std::uint64_t> physical_offsets;
  std::uint64_t total_size_bytes = 0U;
  for (std::size_t i = 0; i < contract.physical_outputs.size(); ++i) {
    const auto& physical = contract.physical_outputs[i];
    const int key = physical.physical_index >= 0 ? physical.physical_index : static_cast<int>(i);
    if (!physical_offsets.emplace(key, total_size_bytes).second) {
      if (error_message) {
        *error_message = "duplicate physical index in packed publish contract";
      }
      return false;
    }
    if (!checked_add_u64_local(total_size_bytes, physical.size_bytes, &total_size_bytes)) {
      if (error_message) {
        *error_message = "packed publish contract size overflow";
      }
      return false;
    }
  }

  out->physical_outputs.clear();
  out->physical_outputs.push_back(simaai::gst::TensorBufferPublishPhysicalOutput{
      0, total_size_bytes, kPreparedRuntimePackedSegmentNameLocal});

  for (auto& logical : out->logical_outputs) {
    std::uint64_t packed_offset = 0U;
    if (!resolve_packed_logical_offset_local(physical_offsets, logical.physical_index,
                                             logical.memory_index, logical.byte_offset,
                                             &packed_offset, error_message)) {
      return false;
    }
    if (packed_offset > static_cast<std::uint64_t>(std::numeric_limits<std::int64_t>::max())) {
      if (error_message) {
        *error_message = "packed publish contract offset overflow";
      }
      return false;
    }
    logical.physical_index = 0;
    logical.memory_index = 0;
    logical.segment_name = kPreparedRuntimePackedSegmentNameLocal;
    logical.byte_offset = static_cast<std::int64_t>(packed_offset);
  }

  for (auto& route : out->output_order) {
    route.segment_name = kPreparedRuntimePackedSegmentNameLocal;
  }
  return true;
}

GstCaps* build_dequant_caps_from_tensor_local(const CapsTensorSpec& tensor,
                                              const std::string& dtype_format) {
  GstCaps* caps = processmla_make_tensor_caps_from_tensor_local(tensor, dtype_format, false);
  processmla_add_shape_fields_to_caps_local(caps, tensor);
  return caps;
}

bool dequant_manifest_input_has_precise_single_tensor_local(const StageStaticSpec& stage,
                                                            CapsTensorSpec* out) {
  if (!out || stage.logical_inputs.size() != 1U || stage.input_bindings.size() > 1U ||
      stage.physical_inputs.size() > 1U) {
    return false;
  }
  *out = caps_tensor_from_logical_input_local(stage.logical_inputs[0]);
  return !out->shape.empty();
}

bool dequant_publish_contract_has_precise_single_output_local(
    const simaai::gst::TensorBufferPublishContract& publish_contract, CapsTensorSpec* out) {
  if (!out) {
    return false;
  }
  const auto* logical = single_publish_logical_output_local(publish_contract);
  if (!logical) {
    return false;
  }
  *out = caps_tensor_from_publish_logical_output_local(*logical);
  return !out->shape.empty();
}

bool build_dequant_prepared_stage_from_stage_contract_local(const StageStaticSpec& stage,
                                                            simaai::gst::DequantPreparedStage* out,
                                                            std::string* error_message) {
  if (!out) {
    if (error_message) {
      *error_message = "dequant prepared stage requires output storage";
    }
    return false;
  }
  if (!stage_is_dequant_local(stage)) {
    if (error_message) {
      *error_message = "stage is not dequant";
    }
    return false;
  }
  if (stage.logical_inputs.empty() || stage.logical_outputs.empty()) {
    if (error_message) {
      *error_message = "dequant stage requires logical inputs and outputs";
    }
    return false;
  }
  if (stage.logical_inputs.size() != stage.logical_outputs.size()) {
    if (error_message) {
      *error_message = "dequant stage logical input/output count mismatch";
    }
    return false;
  }

  simaai::gst::DequantPreparedStage prepared;
  prepared.stage_key = stage_key_from_stage_spec_local(stage);
  prepared.input_dtype = upper_copy_local(common_logical_input_dtype_local(stage.logical_inputs));
  if (prepared.input_dtype.empty()) {
    if (error_message) {
      *error_message = "dequant stage requires homogeneous input dtype";
    }
    return false;
  }
  prepared.input_elem_bytes = dtype_size_bytes_local(prepared.input_dtype);
  if (prepared.input_elem_bytes == 0U) {
    if (error_message) {
      *error_message = "dequant stage requires valid input element size";
    }
    return false;
  }
  prepared.tensor_shape = common_logical_output_shape_local(stage.logical_outputs);
  if (prepared.tensor_shape.empty()) {
    prepared.tensor_shape = stage.logical_outputs.front().shape;
  }
  prepared.tensor_layout = common_logical_output_layout_local(stage.logical_outputs);
  if (prepared.tensor_layout.empty()) {
    prepared.tensor_layout = !stage.logical_outputs.front().layout.empty()
                                 ? stage.logical_outputs.front().layout
                                 : common_logical_input_layout_local(stage.logical_inputs);
  }

  std::unordered_map<int, std::uint64_t> input_physical_offsets;
  std::uint64_t total_input_physical_size_bytes = 0U;
  if (!build_packed_physical_offsets_from_static_specs_local(
          stage.physical_inputs, &input_physical_offsets, &total_input_physical_size_bytes,
          error_message)) {
    return false;
  }
  std::unordered_map<int, std::uint64_t> output_physical_offsets;
  std::uint64_t total_output_physical_size_bytes = 0U;
  if (!build_packed_physical_offsets_from_static_specs_local(
          stage.physical_outputs, &output_physical_offsets, &total_output_physical_size_bytes,
          error_message)) {
    return false;
  }

  prepared.required_input_bytes = total_input_physical_size_bytes;
  prepared.required_output_bytes = total_output_physical_size_bytes;
  prepared.quant_spans.reserve(stage.logical_inputs.size());

  for (std::size_t i = 0; i < stage.logical_inputs.size(); ++i) {
    const auto& logical_input = stage.logical_inputs[i];
    const auto& logical_output = stage.logical_outputs[i];

    const QuantStaticSpec* quant =
        logical_input.quant.has_value() ? &*logical_input.quant : nullptr;
    if (!quant && i < stage.output_quant.size()) {
      quant = &stage.output_quant[i];
    }
    if (!quant && stage.output_quant.size() == 1U) {
      quant = &stage.output_quant.front();
    }
    if (!quant || quant->scales.empty() || quant->zero_points.empty()) {
      if (error_message) {
        *error_message = "dequant stage missing quant contract";
      }
      return false;
    }

    const auto input_shape = logical_input.shape;
    const auto output_shape = logical_output.shape;
    const auto input_stride_bytes =
        strides_or_contiguous_for_shape_local(logical_input.stride_bytes, input_shape,
                                              static_cast<std::int64_t>(prepared.input_elem_bytes));
    const auto output_stride_bytes = strides_or_contiguous_for_shape_local(
        logical_output.stride_bytes, output_shape, static_cast<std::int64_t>(sizeof(float)));
    const std::uint64_t input_size_bytes =
        logical_input.size_bytes > 0U ? logical_input.size_bytes
                                      : shape_size_bytes_local(input_shape, prepared.input_dtype);
    const std::string output_dtype = !logical_output.dtype.empty() ? logical_output.dtype : "FP32";
    const std::uint64_t output_size_bytes =
        logical_output.size_bytes > 0U ? logical_output.size_bytes
                                       : shape_size_bytes_local(output_shape, output_dtype);
    if (input_size_bytes == 0U || output_size_bytes == 0U) {
      if (error_message) {
        *error_message = "dequant stage requires non-zero logical tensor sizes";
      }
      return false;
    }
    if ((input_size_bytes % prepared.input_elem_bytes) != 0U ||
        (output_size_bytes % sizeof(float)) != 0U) {
      if (error_message) {
        *error_message = "dequant stage logical tensor sizes are not element aligned";
      }
      return false;
    }

    std::vector<std::int64_t> iteration_shape = !output_shape.empty() ? output_shape : input_shape;
    std::uint64_t input_iteration_size_bytes = input_size_bytes;
    std::uint64_t output_iteration_size_bytes = output_size_bytes;
    std::uint64_t input_elem_count = input_size_bytes / prepared.input_elem_bytes;
    std::uint64_t output_elem_count = output_size_bytes / sizeof(float);
    if (input_elem_count != output_elem_count) {
      if (!shape_is_subset_compatible_local(input_shape, output_shape)) {
        if (error_message) {
          *error_message = "dequant stage input/output element count mismatch";
        }
        return false;
      }
      const std::uint64_t sliced_elems = shape_elements_or_zero_local(output_shape);
      if (sliced_elems == 0U) {
        if (error_message) {
          *error_message = "dequant stage output shape has zero elements";
        }
        return false;
      }
      input_iteration_size_bytes =
          sliced_elems * static_cast<std::uint64_t>(prepared.input_elem_bytes);
      output_iteration_size_bytes = sliced_elems * static_cast<std::uint64_t>(sizeof(float));
      input_elem_count = sliced_elems;
      output_elem_count = sliced_elems;
      iteration_shape = output_shape;
    }
    if (iteration_shape.empty() || input_stride_bytes.size() != iteration_shape.size() ||
        output_stride_bytes.size() != iteration_shape.size()) {
      if (error_message) {
        *error_message = "dequant stage rank/stride mismatch";
      }
      return false;
    }

    std::uint64_t input_offset_bytes = 0U;
    if (!resolve_packed_logical_offset_local(input_physical_offsets, logical_input.physical_index,
                                             -1, logical_input.byte_offset, &input_offset_bytes,
                                             error_message)) {
      return false;
    }
    std::uint64_t output_offset_bytes = 0U;
    if (!resolve_packed_logical_offset_local(output_physical_offsets, logical_output.physical_index,
                                             -1, logical_output.byte_offset, &output_offset_bytes,
                                             error_message)) {
      return false;
    }
    if ((input_offset_bytes % prepared.input_elem_bytes) != 0U ||
        (output_offset_bytes % sizeof(float)) != 0U) {
      if (error_message) {
        *error_message = "dequant stage logical tensor offsets are not element aligned";
      }
      return false;
    }

    std::uint64_t input_physical_span_bytes = 0U;
    if (!tensor_physical_span_bytes_local(iteration_shape, input_stride_bytes,
                                          static_cast<std::uint64_t>(prepared.input_elem_bytes),
                                          input_iteration_size_bytes, &input_physical_span_bytes)) {
      if (error_message) {
        *error_message = "dequant stage input physical span overflow";
      }
      return false;
    }
    std::uint64_t output_physical_span_bytes = 0U;
    if (!tensor_physical_span_bytes_local(
            iteration_shape, output_stride_bytes, static_cast<std::uint64_t>(sizeof(float)),
            output_iteration_size_bytes, &output_physical_span_bytes)) {
      if (error_message) {
        *error_message = "dequant stage output physical span overflow";
      }
      return false;
    }
    std::uint64_t input_span_end = 0U;
    std::uint64_t output_span_end = 0U;
    if (!checked_add_u64_local(input_offset_bytes, input_physical_span_bytes, &input_span_end) ||
        !checked_add_u64_local(output_offset_bytes, output_physical_span_bytes, &output_span_end)) {
      if (error_message) {
        *error_message = "dequant stage tensor span overflow";
      }
      return false;
    }
    if (input_span_end > prepared.required_input_bytes ||
        output_span_end > prepared.required_output_bytes) {
      if (error_message) {
        *error_message = "dequant stage tensor span exceeds packed physical buffers";
      }
      return false;
    }

    simaai::gst::DequantPreparedSpan span;
    span.input_elem_offset =
        static_cast<std::size_t>(input_offset_bytes / prepared.input_elem_bytes);
    span.output_elem_offset =
        static_cast<std::size_t>(output_offset_bytes / static_cast<std::uint64_t>(sizeof(float)));
    span.elem_count = static_cast<std::size_t>(input_elem_count);
    span.input_byte_offset = input_offset_bytes;
    span.output_byte_offset = output_offset_bytes;
    span.q_scale = quant->scales.front();
    span.q_zp = quant->zero_points.front();
    span.shape = std::move(iteration_shape);
    span.input_stride_bytes = input_stride_bytes;
    span.output_stride_bytes = output_stride_bytes;
    span.input_contiguous = strides_match_contiguous_local(
        span.shape, span.input_stride_bytes, static_cast<std::int64_t>(prepared.input_elem_bytes));
    span.output_contiguous = strides_match_contiguous_local(
        span.shape, span.output_stride_bytes, static_cast<std::int64_t>(sizeof(float)));
    prepared.quant_spans.push_back(std::move(span));
  }

  if (!prepared.quant_spans.empty()) {
    prepared.has_scalar_quant = prepared.quant_spans.size() == 1U;
    prepared.q_scale = prepared.quant_spans.front().q_scale;
    prepared.q_zp = prepared.quant_spans.front().q_zp;
  }

  if (!build_publish_contract_from_manifest_stage_local(stage, &prepared.identity_publish_contract,
                                                        error_message)) {
    return false;
  }
  simaai::gst::TensorBufferPreparedMetaTemplate meta_template;
  if (!simaai::gst::tensor_buffer_prepare_meta_template_from_contract(
          prepared.identity_publish_contract, &meta_template, error_message)) {
    return false;
  }
  prepared.prepared_meta_template = std::move(meta_template);

  CapsTensorSpec sink_tensor;
  if (dequant_manifest_input_has_precise_single_tensor_local(stage, &sink_tensor)) {
    sink_tensor.layout = !sink_tensor.layout.empty() ? sink_tensor.layout : prepared.tensor_layout;
    prepared.sink_caps = build_dequant_caps_from_tensor_local(
        sink_tensor, processmla_caps_format_from_dtype_local(prepared.input_dtype, "EVXX_INT8"));
  } else {
    prepared.sink_caps = make_generic_tensor_set_caps_local();
  }
  CapsTensorSpec src_tensor;
  if (dequant_publish_contract_has_precise_single_output_local(prepared.identity_publish_contract,
                                                               &src_tensor)) {
    src_tensor.layout = !src_tensor.layout.empty() ? src_tensor.layout : prepared.tensor_layout;
    prepared.src_caps = build_dequant_caps_from_tensor_local(src_tensor, "FP32");
  } else {
    prepared.src_caps = make_generic_tensor_set_caps_local();
  }
  if (!dequant_prepared_stage_complete_local(prepared, error_message)) {
    return false;
  }

  *out = std::move(prepared);
  return true;
}

const MpkGraphNode* find_graph_node_by_name_local(const std::vector<MpkGraphNode>& nodes,
                                                  const std::string& name,
                                                  const std::string& canonical_op) {
  for (const auto& node : nodes) {
    if (node.name != name) {
      continue;
    }
    if (!canonical_op.empty() &&
        upper_copy_local(node.canonical_op) != upper_copy_local(canonical_op)) {
      continue;
    }
    return &node;
  }
  return nullptr;
}

std::vector<const MpkGraphNode*>
find_graph_raw_nodes_by_op_local(const std::vector<MpkGraphNode>& nodes,
                                 const std::string& canonical_op) {
  std::vector<const MpkGraphNode*> out;
  const auto wanted = upper_copy_local(canonical_op);
  for (const auto& node : nodes) {
    if (!wanted.empty() && upper_copy_local(node.canonical_op) != wanted) {
      continue;
    }
    out.push_back(&node);
  }
  std::sort(out.begin(), out.end(), [](const MpkGraphNode* a, const MpkGraphNode* b) {
    if (a->sequence != b->sequence) {
      return a->sequence < b->sequence;
    }
    return a->name < b->name;
  });
  return out;
}

bool graph_node_has_member_local(const MpkGraphNode& node, const std::string& member_name) {
  return std::find(node.member_node_ids.begin(), node.member_node_ids.end(), member_name) !=
         node.member_node_ids.end();
}

const MpkGraphKernelField* find_graph_kernel_field_local(const MpkGraphNode& node,
                                                         const MpkGraphKernelFieldKind kind,
                                                         const std::string& name) {
  for (const auto& field : node.kernel_contract.fields) {
    if (field.kind == kind && field.name == name) {
      return &field;
    }
  }
  return nullptr;
}

std::string graph_kernel_field_value_local(const MpkGraphNode& node,
                                           const MpkGraphKernelFieldKind kind,
                                           const std::string& name) {
  const auto* field = find_graph_kernel_field_local(node, kind, name);
  return field ? field->value : std::string();
}

bool parse_graph_int_field_local(const MpkGraphNode& node, const MpkGraphKernelFieldKind kind,
                                 const std::string& name, int32_t* value) {
  if (!value) {
    return false;
  }
  const std::string raw = graph_kernel_field_value_local(node, kind, name);
  if (raw.empty()) {
    return false;
  }
  char* end = nullptr;
  const long parsed = std::strtol(raw.c_str(), &end, 10);
  if (end == raw.c_str() || (end && *end != '\0') ||
      parsed < static_cast<long>(std::numeric_limits<int32_t>::min()) ||
      parsed > static_cast<long>(std::numeric_limits<int32_t>::max())) {
    return false;
  }
  *value = static_cast<int32_t>(parsed);
  return true;
}

bool parse_graph_int_array_field_local(const MpkGraphNode& node, const MpkGraphKernelFieldKind kind,
                                       const std::string& name, std::vector<int32_t>* values) {
  if (!values) {
    return false;
  }
  const std::string raw = graph_kernel_field_value_local(node, kind, name);
  if (raw.empty()) {
    return false;
  }
  std::size_t pos = 0U;
  auto skip_ws = [&]() {
    while (pos < raw.size() && std::isspace(static_cast<unsigned char>(raw[pos]))) {
      ++pos;
    }
  };
  skip_ws();
  if (pos >= raw.size() || raw[pos] != '[') {
    return false;
  }
  ++pos;
  values->clear();
  while (pos < raw.size()) {
    skip_ws();
    if (pos < raw.size() && raw[pos] == ']') {
      ++pos;
      skip_ws();
      return pos == raw.size();
    }
    char* end = nullptr;
    const long parsed = std::strtol(raw.c_str() + pos, &end, 10);
    if (end == raw.c_str() + pos ||
        parsed < static_cast<long>(std::numeric_limits<int32_t>::min()) ||
        parsed > static_cast<long>(std::numeric_limits<int32_t>::max())) {
      return false;
    }
    values->push_back(static_cast<int32_t>(parsed));
    pos = static_cast<std::size_t>(end - raw.c_str());
    skip_ws();
    if (pos < raw.size() && raw[pos] == ',') {
      ++pos;
      continue;
    }
    if (pos < raw.size() && raw[pos] == ']') {
      ++pos;
      skip_ws();
      return pos == raw.size();
    }
    return false;
  }
  return false;
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

bool processcvu_build_graph_io_tensor_descs_local(const ProcessCvuStagePayload& payload,
                                                  const std::string& graph_name,
                                                  const GraphProcessCvuIoData& io,
                                                  simaai::gst::PreparedProcessCvuTypedConfig* cfg,
                                                  std::string* error_message);
bool checked_add_u64_local(const std::uint64_t a, const std::uint64_t b, std::uint64_t* out);
bool tensor_physical_span_bytes_local(const std::vector<std::int64_t>& shape,
                                      const std::vector<std::int64_t>& stride_bytes,
                                      const std::uint64_t elem_bytes,
                                      const std::uint64_t logical_size_bytes,
                                      std::uint64_t* out_span_bytes);
bool build_processcvu_prepared_stage_from_graph_io_local(const StageStaticSpec& original_stage,
                                                         const MpkGraphNode& graph_node,
                                                         const std::string& stage_key,
                                                         const GraphProcessCvuIoData& io,
                                                         simaai::gst::ProcessCvuPreparedStage* out,
                                                         std::string* error_message);
bool build_processcvu_prepared_stage_from_graph_local(const StageStaticSpec& original_stage,
                                                      const MpkContract& contract,
                                                      const MpkGraphNode& graph_node,
                                                      const std::string& stage_key,
                                                      simaai::gst::ProcessCvuPreparedStage* out,
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
  if (!stage->slice_shape.empty()) {
    io.slice_shapes.push_back(
        std::vector<int>(stage->slice_shape.begin(), stage->slice_shape.end()));
  }
  io.input_slot_name = graph_kernel_input_slot_name_local(graph_node, io.input_tensors);
  io.runtime_output_slot_names =
      graph_kernel_output_slot_names_local(graph_node, io.output_tensors);
  return build_processcvu_prepared_stage_from_graph_io_local(original_stage, graph_node, stage_key,
                                                             io, out, error_message);
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

const MpkGraphNode* find_unique_graph_node_by_op_local(const std::vector<MpkGraphNode>& nodes,
                                                       const std::string& canonical_op) {
  std::vector<const MpkGraphNode*> matches;
  const std::string wanted = upper_copy_local(canonical_op);
  for (const auto& node : nodes) {
    if (!wanted.empty() && upper_copy_local(node.canonical_op) != wanted) {
      continue;
    }
    matches.push_back(&node);
  }
  std::sort(matches.begin(), matches.end(), [](const MpkGraphNode* a, const MpkGraphNode* b) {
    if (a->sequence != b->sequence) {
      return a->sequence < b->sequence;
    }
    return a->name < b->name;
  });
  return matches.size() == 1U ? matches[0] : nullptr;
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

bool build_graph_owned_prepared_stage_local(const StageStaticSpec& transformed_stage,
                                            const StageStaticSpec* original_stage,
                                            const MpkContract& contract,
                                            const std::filesystem::path& pack_root,
                                            simaai::gst::PreparedStageSpec* out,
                                            std::string* error_message) {
  if (!out) {
    if (error_message) {
      *error_message = "graph-owned prepared stage requires output storage";
    }
    return false;
  }

  const std::string transformed_stage_key = stage_key_from_stage_spec_local(transformed_stage);

  if (stage_is_processmla_local(transformed_stage) ||
      (original_stage && stage_is_processmla_local(*original_stage))) {
    const auto keys = graph_stage_candidate_keys_local(transformed_stage, original_stage);
    auto matches = find_graph_nodes_matching_stage_keys_local(contract.graph.nodes, keys, "mla");
    if (matches.empty()) {
      if (const auto* unique_mla =
              find_unique_graph_node_by_op_local(contract.graph.nodes, "mla")) {
        matches.push_back(unique_mla);
      }
    }
    if (matches.size() != 1U) {
      if (error_message) {
        *error_message = "graph-owned processmla stage could not resolve a unique graph node";
      }
      return false;
    }
    simaai::gst::ProcessMlaPreparedStage processmla_stage;
    if (!build_processmla_prepared_stage_from_graph_local(
            contract, *matches[0], transformed_stage_key, &processmla_stage, error_message)) {
      return false;
    }
    if (!processmla_stage.runtime_cfg.model_path.empty()) {
      processmla_stage.runtime_cfg.model_path = resolve_model_path_from_pack_root_local(
          pack_root, processmla_stage.runtime_cfg.model_path);
    }
    out->stage_key = transformed_stage_key;
    out->kind = simaai::gst::PreparedStageKind::ProcessMla;
    out->processmla = std::move(processmla_stage);
    return true;
  }

  const bool transformed_is_processcvu = stage_is_processcvu_local(transformed_stage);
  const bool original_is_processcvu = original_stage && stage_is_processcvu_local(*original_stage);
  if (!transformed_is_processcvu && !original_is_processcvu &&
      (stage_is_dequant_local(transformed_stage) ||
       (original_stage && stage_is_dequant_local(*original_stage)))) {
    simaai::gst::DequantPreparedStage dequant_stage;
    if (!build_dequant_prepared_stage_from_stage_contract_local(transformed_stage, &dequant_stage,
                                                                error_message)) {
      return false;
    }
    out->stage_key = transformed_stage_key;
    out->kind = simaai::gst::PreparedStageKind::Dequant;
    out->dequant = std::move(dequant_stage);
    return true;
  }

  if (stage_is_processcvu_local(transformed_stage) ||
      (original_stage && stage_is_processcvu_local(*original_stage))) {
    const StageStaticSpec& processcvu_stage_spec = transformed_stage;
    const std::string graph_family = !processcvu_stage_spec.processcvu.graph_family.empty()
                                         ? processcvu_stage_spec.processcvu.graph_family
                                         : processcvu_stage_spec.processcvu.graph_name;
    const std::string canonical_family = processcvu_canonical_graph_name_local(graph_family);
    const std::string exact_stage_key =
        !resolve_exact_processcvu_graph_stage_key_local(&processcvu_stage_spec).empty()
            ? resolve_exact_processcvu_graph_stage_key_local(&processcvu_stage_spec)
            : resolve_exact_processcvu_graph_stage_key_local(original_stage);
    const bool processcvu_region_contract = transformed_stage.input_bindings.size() > 1U ||
                                            transformed_stage.logical_inputs.size() > 1U ||
                                            transformed_stage.logical_outputs.size() > 1U ||
                                            transformed_stage.output_order.size() > 1U;

    simaai::gst::ProcessCvuPreparedStage processcvu_stage;
    if (canonical_family == "detesscast" || canonical_family == "detessdequant" ||
        canonical_family == "dequantize" || processcvu_region_contract) {
      if (!build_processcvu_prepared_stage_from_stage_contract_local(
              transformed_stage, &processcvu_stage, error_message)) {
        return false;
      }
    } else {
      const auto keys =
          graph_stage_candidate_keys_local(transformed_stage, original_stage, exact_stage_key);
      const auto matches = find_graph_nodes_matching_stage_keys_local(contract.graph.raw_nodes,
                                                                      keys, canonical_family);
      if (matches.size() != 1U) {
        if (error_message) {
          *error_message = "graph-owned processcvu stage could not resolve a unique graph node";
        }
        return false;
      }
      if (!build_processcvu_prepared_stage_from_graph_local(transformed_stage, contract,
                                                            *matches[0], transformed_stage_key,
                                                            &processcvu_stage, error_message)) {
        return false;
      }
    }
    out->stage_key = transformed_stage_key;
    out->kind = simaai::gst::PreparedStageKind::ProcessCvu;
    out->processcvu = std::move(processcvu_stage);
    return true;
  }

  if (error_message) {
    *error_message = "unsupported graph-owned stage kind";
  }
  return false;
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
    auto contract = load_mpk_contract_from_pack_root(pack_root->string(), &load_error);
    if (contract.has_value()) {
      if (pack_root_out) {
        *pack_root_out = *pack_root;
      }
      return contract;
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
    auto contract = load_mpk_contract_from_pack_root(pack_root->string(), &load_error);
    if (contract.has_value()) {
      if (pack_root_out) {
        *pack_root_out = *pack_root;
      }
      return contract;
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

bool prepared_runtime_graph_dump_enabled_local() {
  const auto enabled = [](const char* key) {
    const char* raw = std::getenv(key);
    return raw && *raw && std::strcmp(raw, "0") != 0;
  };
  const char* explicit_path = std::getenv("SIMA_PREPARED_RUNTIME_GRAPH_OUTPUT_PATH");
  return (explicit_path && *explicit_path) || enabled("SIMA_PREPARED_RUNTIME_GRAPH_DUMP") ||
         enabled("SIMA_MPK_GRAPH_DUMP");
}

std::string runtime_graph_token_local(std::string raw) {
  for (char& ch : raw) {
    const unsigned char uch = static_cast<unsigned char>(ch);
    if (!std::isalnum(uch)) {
      ch = '_';
    }
  }
  while (!raw.empty() && raw.front() == '_') {
    raw.erase(raw.begin());
  }
  while (!raw.empty() && raw.back() == '_') {
    raw.pop_back();
  }
  return raw.empty() ? std::string("mpk") : raw;
}

std::filesystem::path
prepared_runtime_graph_output_path_local(const std::filesystem::path& pack_root,
                                         const MpkContract& contract) {
  if (const char* raw = std::getenv("SIMA_PREPARED_RUNTIME_GRAPH_OUTPUT_PATH"); raw && *raw) {
    return std::filesystem::path(raw);
  }
  if (const char* raw = std::getenv("SIMA_MPK_GRAPH_OUTPUT_PATH"); raw && *raw) {
    std::filesystem::path graph_path(raw);
    const std::string stem = graph_path.stem().string();
    std::string runtime_stem = stem;
    if (runtime_stem.size() > 6U && runtime_stem.rfind("_graph") == runtime_stem.size() - 6U) {
      runtime_stem.erase(runtime_stem.size() - 6U);
    }
    runtime_stem += "_runtime_graph";
    return graph_path.parent_path() / (runtime_stem + graph_path.extension().string());
  }
  std::string base_name;
  if (!contract.model_name.empty()) {
    base_name = runtime_graph_token_local(contract.model_name);
  } else if (!contract.mpk_json_path.empty()) {
    base_name =
        runtime_graph_token_local(std::filesystem::path(contract.mpk_json_path).stem().string());
  } else {
    base_name = "mpk";
  }
  return pack_root / (base_name + "_runtime_graph.md");
}

std::string ints_dbg_runtime_local(const std::vector<int32_t>& values) {
  std::ostringstream oss;
  oss << "[";
  for (std::size_t i = 0; i < values.size(); ++i) {
    if (i != 0U) {
      oss << ",";
    }
    oss << values[i];
  }
  oss << "]";
  return oss.str();
}

std::string ints64_dbg_runtime_local(const std::vector<std::int64_t>& values) {
  std::ostringstream oss;
  oss << "[";
  for (std::size_t i = 0; i < values.size(); ++i) {
    if (i != 0U) {
      oss << ",";
    }
    oss << values[i];
  }
  oss << "]";
  return oss.str();
}

std::string strings_dbg_runtime_local(const std::vector<std::string>& values) {
  std::ostringstream oss;
  oss << "[";
  for (std::size_t i = 0; i < values.size(); ++i) {
    if (i != 0U) {
      oss << ",";
    }
    oss << "\"" << values[i] << "\"";
  }
  oss << "]";
  return oss.str();
}

std::string caps_dbg_runtime_local(GstCaps* caps) {
  if (!caps) {
    return "<null>";
  }
  gchar* raw = gst_caps_to_string(caps);
  const std::string value = raw ? std::string(raw) : std::string("<null>");
  if (raw) {
    g_free(raw);
  }
  return value;
}

std::string ev_shape_dbg_runtime_local(const sima_ev_shape_desc& shape) {
  std::ostringstream oss;
  const guint rank = std::min<guint>(shape.rank, SIMA_EV_MAX_RANK);
  oss << "[";
  for (guint i = 0; i < rank; ++i) {
    if (i != 0U) {
      oss << ",";
    }
    oss << shape.sizes[i];
  }
  oss << "]";
  return oss.str();
}

std::string ev_tensor_desc_dbg_runtime_local(const sima_ev_tensor_desc& desc) {
  std::ostringstream oss;
  oss << "{dtype=" << static_cast<int>(desc.dtype)
      << ",layout_kind=" << static_cast<int>(desc.layout_kind)
      << ",shape=" << ev_shape_dbg_runtime_local(desc.shape);
  if (desc.layout_kind == SIMA_EV_LAYOUT_TILED) {
    oss << ",tile=[";
    const guint rank = std::min<guint>(desc.shape.rank, SIMA_EV_MAX_RANK);
    for (guint i = 0; i < rank; ++i) {
      if (i != 0U) {
        oss << ",";
      }
      oss << desc.layout.tiled.tile_sizes[i];
    }
    oss << "]";
  }
  oss << "}";
  return oss.str();
}

std::string ev_tensor_descs_dbg_runtime_local(const std::vector<sima_ev_tensor_desc>& descs) {
  std::ostringstream oss;
  oss << "[";
  for (std::size_t i = 0; i < descs.size(); ++i) {
    if (i != 0U) {
      oss << ",";
    }
    oss << ev_tensor_desc_dbg_runtime_local(descs[i]);
  }
  oss << "]";
  return oss.str();
}

MpkGraphNode* find_runtime_graph_node_by_name_local(std::vector<MpkGraphNode>* nodes,
                                                    const std::string& name) {
  if (!nodes || name.empty()) {
    return nullptr;
  }
  for (auto& node : *nodes) {
    if (node.name == name) {
      return &node;
    }
  }
  return nullptr;
}

MpkGraphNode* find_runtime_graph_node_local(std::vector<MpkGraphNode>* nodes,
                                            const MpkGraphNode& target) {
  if (!nodes) {
    return nullptr;
  }
  if (!target.node_id.empty()) {
    for (auto& node : *nodes) {
      if (node.node_id == target.node_id) {
        return &node;
      }
    }
  }
  if (!target.name.empty()) {
    if (auto* node = find_runtime_graph_node_by_name_local(nodes, target.name)) {
      return node;
    }
  }
  if (!target.label.empty()) {
    for (auto& node : *nodes) {
      if (node.label == target.label && node.kind == target.kind) {
        return &node;
      }
    }
  }
  return nullptr;
}

void set_runtime_graph_kernel_field_local(MpkGraphNode* node, const MpkGraphKernelFieldKind kind,
                                          const std::string& name, const std::string& value,
                                          const bool add_if_missing) {
  if (!node || name.empty()) {
    return;
  }
  for (auto& field : node->kernel_contract.fields) {
    if (field.kind == kind && field.name == name) {
      field.value = value;
      field.known = true;
      return;
    }
  }
  if (!add_if_missing) {
    return;
  }
  node->kernel_contract.fields.push_back(
      MpkGraphKernelField{.name = name, .value = value, .kind = kind, .known = true});
}

void overlay_processcvu_runtime_graph_node_local(
    MpkGraphNode* node, const simaai::gst::ProcessCvuPreparedStage& stage) {
  if (!node) {
    return;
  }
  const auto& cfg = stage.typed_config;
  const std::string canonical_graph_family = processcvu_canonical_graph_name_local(cfg.graph_name);
  if (canonical_graph_family != "detessdequant") {
    set_runtime_graph_kernel_field_local(node, MpkGraphKernelFieldKind::Value, "num_in_tensor",
                                         std::to_string(cfg.num_in_tensor), false);
  }

  set_runtime_graph_kernel_field_local(node, MpkGraphKernelFieldKind::Parameter,
                                       "runtime.stage_key", stage.stage_key, true);
  set_runtime_graph_kernel_field_local(node, MpkGraphKernelFieldKind::Parameter,
                                       "runtime.input_layout",
                                       prepared_input_layout_token_local(cfg), true);
  set_runtime_graph_kernel_field_local(node, MpkGraphKernelFieldKind::Parameter,
                                       "runtime.output_layout",
                                       prepared_output_layout_token_local(cfg), true);
  set_runtime_graph_kernel_field_local(node, MpkGraphKernelFieldKind::Parameter,
                                       "runtime.default_input_name", cfg.default_input_name, true);
  set_runtime_graph_kernel_field_local(node, MpkGraphKernelFieldKind::Parameter,
                                       "runtime.input_tensors",
                                       ev_tensor_descs_dbg_runtime_local(cfg.input_tensors), true);
  set_runtime_graph_kernel_field_local(node, MpkGraphKernelFieldKind::Parameter,
                                       "runtime.output_tensors",
                                       ev_tensor_descs_dbg_runtime_local(cfg.output_tensors), true);
  set_runtime_graph_kernel_field_local(node, MpkGraphKernelFieldKind::Parameter,
                                       "runtime.runtime_output_names",
                                       strings_dbg_runtime_local(cfg.runtime_output_names), true);
  set_runtime_graph_kernel_field_local(node, MpkGraphKernelFieldKind::Parameter,
                                       "runtime.published_output_names",
                                       strings_dbg_runtime_local(cfg.published_output_names), true);
  set_runtime_graph_kernel_field_local(node, MpkGraphKernelFieldKind::Parameter,
                                       "runtime.primary_output_name", stage.primary_output_name,
                                       true);
  set_runtime_graph_kernel_field_local(node, MpkGraphKernelFieldKind::Parameter,
                                       "runtime.primary_output_packed_caps",
                                       stage.primary_output_packed_caps ? "true" : "false", true);
  set_runtime_graph_kernel_field_local(node, MpkGraphKernelFieldKind::Parameter,
                                       "runtime.sink_caps", caps_dbg_runtime_local(stage.sink_caps),
                                       true);
  set_runtime_graph_kernel_field_local(node, MpkGraphKernelFieldKind::Parameter, "runtime.src_caps",
                                       caps_dbg_runtime_local(stage.src_caps), true);

  if (const auto* logical = single_publish_logical_output_local(stage.output_publish_contract)) {
    node->mpk_shape = logical->shape;
    node->size_bytes = logical->size_bytes;
  }
  if (!cfg.output_dtype.empty()) {
    node->dtype = cfg.output_dtype;
  } else if (!cfg.out_dtype.empty()) {
    node->dtype = cfg.out_dtype;
  }
}

void overlay_processmla_runtime_graph_node_local(
    MpkGraphNode* node, const simaai::gst::ProcessMlaPreparedStage& stage) {
  if (!node) {
    return;
  }
  std::vector<std::string> dispatcher_outputs;
  dispatcher_outputs.reserve(stage.runtime_cfg.dispatcher_outputs.size());
  for (const auto& output : stage.runtime_cfg.dispatcher_outputs) {
    dispatcher_outputs.push_back(output.name + ":" + std::to_string(output.size));
  }
  std::vector<std::string> logical_outputs;
  logical_outputs.reserve(stage.runtime_cfg.logical_outputs.size());
  for (const auto& logical : stage.runtime_cfg.logical_outputs) {
    logical_outputs.push_back(logical.logical_name + ":" + std::to_string(logical.size_bytes));
  }
  set_runtime_graph_kernel_field_local(node, MpkGraphKernelFieldKind::Parameter,
                                       "runtime.stage_key", stage.stage_key, true);
  set_runtime_graph_kernel_field_local(node, MpkGraphKernelFieldKind::Parameter,
                                       "runtime.model_path", stage.runtime_cfg.model_path, true);
  set_runtime_graph_kernel_field_local(
      node, MpkGraphKernelFieldKind::Parameter, "runtime.dispatcher_inputs",
      strings_dbg_runtime_local(stage.runtime_cfg.dispatcher_input_names), true);
  set_runtime_graph_kernel_field_local(node, MpkGraphKernelFieldKind::Parameter,
                                       "runtime.dispatcher_outputs",
                                       strings_dbg_runtime_local(dispatcher_outputs), true);
  set_runtime_graph_kernel_field_local(node, MpkGraphKernelFieldKind::Parameter,
                                       "runtime.logical_outputs",
                                       strings_dbg_runtime_local(logical_outputs), true);
  set_runtime_graph_kernel_field_local(node, MpkGraphKernelFieldKind::Parameter,
                                       "runtime.sink_caps", caps_dbg_runtime_local(stage.sink_caps),
                                       true);
  set_runtime_graph_kernel_field_local(node, MpkGraphKernelFieldKind::Parameter, "runtime.src_caps",
                                       caps_dbg_runtime_local(stage.src_caps), true);
  if (const auto* logical = single_publish_logical_output_local(stage.output_publish_contract)) {
    node->mpk_shape = logical->shape;
    node->size_bytes = logical->size_bytes;
    if (logical->dtype != SIMA_TENSOR_SET_DTYPE_UNKNOWN_V1 &&
        stage.runtime_cfg.logical_outputs.size() == 1U) {
      node->dtype = stage.runtime_cfg.logical_outputs[0].dtype;
    }
  }
}

} // namespace

std::optional<simaai::neat::PreparedRuntimeDescriptor>
build_prepared_runtime_context(const GstContext* static_manifest_context,
                               const SimaPluginStaticManifest& transformed_manifest,
                               const std::optional<SimaPluginStaticManifest>& original_manifest,
                               const std::vector<PipelineElementSpec>& pipeline_elements,
                               const std::vector<std::string>& model_source_paths,
                               const NameTransform& name_transform, std::string* error_message) {
  if (error_message) {
    error_message->clear();
  }

  simaai::neat::PreparedRuntimeDescriptor context;
  context.session_id = transformed_manifest.session_id;
  context.model_id = transformed_manifest.model_id;
  context.stages.reserve(transformed_manifest.stages.size());

  const bool needs_graph_contract =
      std::any_of(transformed_manifest.stages.begin(), transformed_manifest.stages.end(),
                  [](const StageStaticSpec& stage) { return stage_is_graph_owned_local(stage); });
  std::optional<MpkContract> graph_contract;
  std::filesystem::path graph_pack_root;
  if (needs_graph_contract) {
    std::string graph_error;
    graph_contract =
        discover_graph_contract_local(transformed_manifest, original_manifest, pipeline_elements,
                                      model_source_paths, &graph_pack_root, &graph_error);
    if (!graph_contract.has_value()) {
      if (error_message && error_message->empty()) {
        *error_message =
            graph_error.empty() ? "graph-owned stages require mpk graph contract" : graph_error;
      }
      return std::nullopt;
    }
  }

  for (const auto& stage : transformed_manifest.stages) {
    const StageStaticSpec* original_stage_ptr =
        (original_manifest.has_value() && !stage.logical_stage_id.empty())
            ? find_original_stage_for_transformed_key_local(*original_manifest, name_transform,
                                                            stage)
            : nullptr;

    if (stage_is_graph_owned_local(stage)) {
      if (!graph_contract.has_value()) {
        if (error_message) {
          *error_message = "graph-owned stage missing discovered graph contract";
        }
        return std::nullopt;
      }
      simaai::gst::PreparedStageSpec prepared;
      if (!build_graph_owned_prepared_stage_local(stage, original_stage_ptr, *graph_contract,
                                                  graph_pack_root, &prepared, error_message)) {
        return std::nullopt;
      }
      context.stages.push_back(std::move(prepared));
      continue;
    }

    if (stage_is_cast_local(stage)) {
      if (static_manifest_context) {
        simaai::gst::PreparedStageSpec prepared;
        const char* stage_id_or_name =
            stage.logical_stage_id.empty() ? nullptr : stage.logical_stage_id.c_str();
        const char* element_name_fallback =
            stage.element_name.empty() ? nullptr : stage.element_name.c_str();
        if (!simaai::neat::build_prepared_stage_from_manifest_context(
                static_manifest_context, stage_id_or_name, element_name_fallback, &prepared,
                error_message)) {
          return std::nullopt;
        }
        context.stages.push_back(std::move(prepared));
        continue;
      }

      simaai::gst::CastPreparedStage cast_stage;
      if (!build_cast_prepared_stage_from_manifest_stage_local(stage, &cast_stage, error_message)) {
        return std::nullopt;
      }
      simaai::gst::PreparedStageSpec prepared;
      prepared.stage_key = cast_stage.stage_key;
      prepared.kind = simaai::gst::PreparedStageKind::Cast;
      prepared.cast = std::move(cast_stage);
      context.stages.push_back(std::move(prepared));
      continue;
    }

    if (stage_is_processcvu_local(stage)) {
      if (static_manifest_context) {
        simaai::gst::PreparedStageSpec prepared;
        const char* stage_id_or_name =
            stage.logical_stage_id.empty() ? nullptr : stage.logical_stage_id.c_str();
        const char* element_name_fallback =
            stage.element_name.empty() ? nullptr : stage.element_name.c_str();
        if (!simaai::neat::build_prepared_stage_from_manifest_context(
                static_manifest_context, stage_id_or_name, element_name_fallback, &prepared,
                error_message)) {
          return std::nullopt;
        }
        context.stages.push_back(std::move(prepared));
        continue;
      }

      simaai::gst::ProcessCvuPreparedStage processcvu_stage;
      if (!build_processcvu_prepared_stage_from_manifest_stage_local(stage, &processcvu_stage,
                                                                     error_message)) {
        return std::nullopt;
      }
      simaai::gst::PreparedStageSpec prepared;
      prepared.stage_key = processcvu_stage.stage_key;
      prepared.kind = simaai::gst::PreparedStageKind::ProcessCvu;
      prepared.processcvu = std::move(processcvu_stage);
      context.stages.push_back(std::move(prepared));
      continue;
    }
  }

  return context;
}

} // namespace simaai::neat::pipeline_internal::sima
