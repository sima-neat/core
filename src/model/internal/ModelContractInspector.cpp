#include "model/internal/ModelContractInspector.h"

#include "model/internal/ModelInternal.h"
#include "model/internal/ModelParser.h"

#include "pipeline/internal/contract/PluginCompiledContracts.h"
#include "pipeline/internal/sima/MpkContract.h"
#include "pipeline/internal/sima/SimaPluginStaticManifest.h"
#include "pipeline/internal/sima/TensorSemanticsUtil.h"

#include <algorithm>
#include <cctype>
#include <functional>
#include <optional>
#include <sstream>
#include <unordered_map>

namespace simaai::neat::internal {
namespace {

using pipeline_internal::sima::BoxDecodePhysicalInputStaticContract;
using pipeline_internal::sima::BoxDecodeStaticContract;
using pipeline_internal::sima::BoxDecodeTensorStaticContract;
using pipeline_internal::sima::InputBindingStaticSpec;
using pipeline_internal::sima::LogicalInputStaticSpec;
using pipeline_internal::sima::LogicalTensorStaticSpec;
using pipeline_internal::sima::MpkContract;
using pipeline_internal::sima::MpkContractEdge;
using pipeline_internal::sima::MpkPluginIoContract;
using pipeline_internal::sima::MpkQuantContract;
using pipeline_internal::sima::MpkShapeSemantics;
using pipeline_internal::sima::MpkTensorContract;
using pipeline_internal::sima::PhysicalBufferStaticSpec;
using pipeline_internal::sima::QuantGranularity;
using pipeline_internal::sima::QuantStaticSpec;
using pipeline_internal::sima::StageOutputRoute;
using pipeline_internal::sima::TensorStaticSpec;
namespace tensorsemantics = pipeline_internal::sima::tensorsemantics;

struct ReportStats {
  bool raw_items = false;
  bool planned_items = false;
  bool typed_items = false;
  bool gst_items = false;
};

std::string lower_copy(std::string value) {
  std::transform(value.begin(), value.end(), value.begin(),
                 [](unsigned char c) { return static_cast<char>(std::tolower(c)); });
  return value;
}

std::string bool_name(const bool value) {
  return value ? "true" : "false";
}

std::string csv_or_none(const std::vector<std::string>& values) {
  if (values.empty()) {
    return "none";
  }
  std::ostringstream oss;
  for (std::size_t i = 0; i < values.size(); ++i) {
    if (i > 0U) {
      oss << ",";
    }
    oss << values[i];
  }
  return oss.str();
}

std::string shape_string(const std::vector<std::int64_t>& shape) {
  if (shape.empty()) {
    return "[]";
  }
  std::ostringstream oss;
  oss << "[";
  for (std::size_t i = 0; i < shape.size(); ++i) {
    if (i > 0U) {
      oss << "x";
    }
    oss << shape[i];
  }
  oss << "]";
  return oss.str();
}

std::string int_vector_string(const std::vector<std::int64_t>& values) {
  if (values.empty()) {
    return "[]";
  }
  std::ostringstream oss;
  oss << "[";
  for (std::size_t i = 0; i < values.size(); ++i) {
    if (i > 0U) {
      oss << ",";
    }
    oss << values[i];
  }
  oss << "]";
  return oss.str();
}

std::string uint_vector_string(const std::vector<std::size_t>& values) {
  if (values.empty()) {
    return "[]";
  }
  std::ostringstream oss;
  oss << "[";
  for (std::size_t i = 0; i < values.size(); ++i) {
    if (i > 0U) {
      oss << ",";
    }
    oss << values[i];
  }
  oss << "]";
  return oss.str();
}

std::string double_vector_string(const std::vector<double>& values) {
  if (values.empty()) {
    return "[]";
  }
  std::ostringstream oss;
  oss << "[";
  for (std::size_t i = 0; i < values.size(); ++i) {
    if (i > 0U) {
      oss << ",";
    }
    oss << values[i];
  }
  oss << "]";
  return oss.str();
}

std::string quant_zero_points_string(const std::vector<std::int64_t>& values) {
  return int_vector_string(values);
}

std::string shapes_2d_string(const std::vector<std::vector<int>>& shapes) {
  if (shapes.empty()) {
    return "[]";
  }
  std::ostringstream oss;
  oss << "[";
  for (std::size_t i = 0; i < shapes.size(); ++i) {
    if (i > 0U) {
      oss << ",";
    }
    oss << "[";
    for (std::size_t j = 0; j < shapes[i].size(); ++j) {
      if (j > 0U) {
        oss << "x";
      }
      oss << shapes[i][j];
    }
    oss << "]";
  }
  oss << "]";
  return oss.str();
}

std::string shape_descs_string(const std::vector<sima_ev_shape_desc>& shapes) {
  if (shapes.empty()) {
    return "[]";
  }
  std::ostringstream oss;
  oss << "[";
  for (std::size_t i = 0; i < shapes.size(); ++i) {
    if (i > 0U) {
      oss << ",";
    }
    oss << "[";
    const auto rank = std::min<std::uint32_t>(shapes[i].rank, SIMA_EV_MAX_RANK);
    for (std::uint32_t dim = 0; dim < rank; ++dim) {
      if (dim > 0U) {
        oss << "x";
      }
      oss << shapes[i].sizes[dim];
    }
    oss << "]";
  }
  oss << "]";
  return oss.str();
}

std::string stage_filter_name(const ModelContractStageFilter filter) {
  switch (filter) {
  case ModelContractStageFilter::All:
    return "all";
  case ModelContractStageFilter::Pre:
    return "pre";
  case ModelContractStageFilter::Infer:
    return "infer";
  case ModelContractStageFilter::Post:
    return "post";
  }
  return "all";
}

std::string execution_stage_kind_name(const ExecutionStageKind kind) {
  switch (kind) {
  case ExecutionStageKind::Preproc:
    return "preproc";
  case ExecutionStageKind::Quant:
    return "quant";
  case ExecutionStageKind::Tess:
    return "tess";
  case ExecutionStageKind::QuantTess:
    return "quanttess";
  case ExecutionStageKind::Mla:
    return "mla";
  case ExecutionStageKind::Detess:
    return "detess";
  case ExecutionStageKind::DetessDequant:
    return "detessdequant";
  case ExecutionStageKind::Dequant:
    return "dequant";
  case ExecutionStageKind::BoxDecode:
    return "boxdecode";
  case ExecutionStageKind::Cast:
    return "cast";
  case ExecutionStageKind::Unknown:
    break;
  }
  return "unknown";
}

std::string device_kind_name(const pipeline_internal::sima::DeviceKind kind) {
  switch (kind) {
  case pipeline_internal::sima::DeviceKind::Cpu:
    return "cpu";
  case pipeline_internal::sima::DeviceKind::Mla:
    return "mla";
  case pipeline_internal::sima::DeviceKind::Evxx:
    return "evxx";
  case pipeline_internal::sima::DeviceKind::Unknown:
    break;
  }
  return "unknown";
}

std::string stage_payload_kind_name(const pipeline_internal::sima::StagePayloadKind kind) {
  switch (kind) {
  case pipeline_internal::sima::StagePayloadKind::ProcessCvu:
    return "processcvu";
  case pipeline_internal::sima::StagePayloadKind::ProcessMla:
    return "processmla";
  case pipeline_internal::sima::StagePayloadKind::BoxDecode:
    return "boxdecode";
  case pipeline_internal::sima::StagePayloadKind::DetessDequant:
    return "detessdequant";
  case pipeline_internal::sima::StagePayloadKind::Quant:
    return "quant";
  case pipeline_internal::sima::StagePayloadKind::Tess:
    return "tess";
  case pipeline_internal::sima::StagePayloadKind::Dequant:
    return "dequant";
  case pipeline_internal::sima::StagePayloadKind::QuantTess:
    return "quanttess";
  case pipeline_internal::sima::StagePayloadKind::None:
    break;
  }
  return "none";
}

std::string
output_transport_kind_name(const pipeline_internal::sima::ProcessCvuOutputTransportKind kind) {
  switch (kind) {
  case pipeline_internal::sima::ProcessCvuOutputTransportKind::Dense:
    return "dense";
  case pipeline_internal::sima::ProcessCvuOutputTransportKind::Packed:
    return "packed";
  case pipeline_internal::sima::ProcessCvuOutputTransportKind::Unknown:
    break;
  }
  return "unknown";
}

std::string
output_semantic_kind_name(const pipeline_internal::sima::ProcessCvuOutputSemanticKind kind) {
  switch (kind) {
  case pipeline_internal::sima::ProcessCvuOutputSemanticKind::Image:
    return "image";
  case pipeline_internal::sima::ProcessCvuOutputSemanticKind::TessellatedImage:
    return "tessellated_image";
  case pipeline_internal::sima::ProcessCvuOutputSemanticKind::QuantizedTensor:
    return "quantized_tensor";
  case pipeline_internal::sima::ProcessCvuOutputSemanticKind::QuantTessTensor:
    return "quanttess_tensor";
  case pipeline_internal::sima::ProcessCvuOutputSemanticKind::Tensor:
    return "tensor";
  case pipeline_internal::sima::ProcessCvuOutputSemanticKind::Unknown:
    break;
  }
  return "unknown";
}

std::string quant_granularity_name(const QuantGranularity granularity) {
  switch (granularity) {
  case QuantGranularity::PerTensor:
    return "per_tensor";
  case QuantGranularity::PerAxis:
    return "per_axis";
  }
  return "unknown";
}

std::string mpk_shape_semantics_name(const MpkShapeSemantics semantics) {
  switch (semantics) {
  case MpkShapeSemantics::Geometry:
    return "geometry";
  case MpkShapeSemantics::PackedExtent:
    return "packed_extent";
  case MpkShapeSemantics::Unknown:
    break;
  }
  return "unknown";
}

std::string input_kind_name(const InputKind kind) {
  switch (kind) {
  case InputKind::Image:
    return "image";
  case InputKind::Tensor:
    return "tensor";
  case InputKind::Auto:
    break;
  }
  return "auto";
}

std::string preprocess_graph_family_name_local(const PreprocessGraphFamily family) {
  switch (family) {
  case PreprocessGraphFamily::Disabled:
    return "disabled";
  case PreprocessGraphFamily::Preproc:
    return "preproc";
  case PreprocessGraphFamily::Quant:
    return "quant";
  case PreprocessGraphFamily::Tess:
    return "tess";
  case PreprocessGraphFamily::QuantTess:
    return "quanttess";
  }
  return "unknown";
}

std::string stage_group_name(const ModelStage stage) {
  switch (stage) {
  case ModelStage::Preprocess:
    return "pre";
  case ModelStage::MlaOnly:
    return "infer";
  case ModelStage::Postprocess:
    return "post";
  case ModelStage::Full:
    break;
  }
  return "all";
}

void append_section_header(std::ostringstream& oss, const std::string& title) {
  oss << "== " << title << " ==\n";
}

void append_kv(std::ostringstream& oss, const int indent, const std::string& key,
               const std::string& value) {
  oss << std::string(static_cast<std::size_t>(std::max(indent, 0)), ' ') << key << "=" << value
      << "\n";
}

template <typename T>
void append_kv_num(std::ostringstream& oss, const int indent, const std::string& key, T value) {
  std::ostringstream tmp;
  tmp << value;
  append_kv(oss, indent, key, tmp.str());
}

bool contains_case_insensitive(const std::string& haystack, const std::string& needle) {
  if (needle.empty()) {
    return true;
  }
  return lower_copy(haystack).find(lower_copy(needle)) != std::string::npos;
}

bool matches_plugin_filters(const std::vector<std::string>& filters,
                            const std::initializer_list<std::string>& fields) {
  if (filters.empty()) {
    return true;
  }
  for (const auto& filter : filters) {
    bool matched = false;
    for (const auto& field : fields) {
      if (contains_case_insensitive(field, filter)) {
        matched = true;
        break;
      }
    }
    if (!matched) {
      return false;
    }
  }
  return true;
}

bool stage_group_selected(const ModelContractStageFilter filter, const ModelStage stage) {
  switch (filter) {
  case ModelContractStageFilter::All:
    return true;
  case ModelContractStageFilter::Pre:
    return stage == ModelStage::Preprocess;
  case ModelContractStageFilter::Infer:
    return stage == ModelStage::MlaOnly;
  case ModelContractStageFilter::Post:
    return stage == ModelStage::Postprocess;
  }
  return true;
}

bool raw_plugin_selected(const ModelContractStageFilter filter, const std::size_t plugin_index,
                         const std::optional<std::size_t>& mla_plugin_index) {
  switch (filter) {
  case ModelContractStageFilter::All:
    return true;
  case ModelContractStageFilter::Pre:
    return !mla_plugin_index.has_value() || plugin_index < *mla_plugin_index;
  case ModelContractStageFilter::Infer:
    return mla_plugin_index.has_value() && plugin_index == *mla_plugin_index;
  case ModelContractStageFilter::Post:
    return mla_plugin_index.has_value() && plugin_index > *mla_plugin_index;
  }
  return true;
}

std::string execution_plan_chain_string(const std::vector<ExecutionStage>& stages) {
  std::vector<std::string> names;
  names.reserve(stages.size());
  for (const auto& stage : stages) {
    names.push_back(execution_stage_kind_name(stage.kind));
  }
  return csv_or_none(names);
}

void append_quant(std::ostringstream& oss, const int indent, const QuantStaticSpec& quant) {
  append_kv(oss, indent, "quant.granularity", quant_granularity_name(quant.granularity));
  append_kv_num(oss, indent, "quant.axis", quant.axis);
  append_kv(oss, indent, "quant.scales", double_vector_string(quant.scales));
  append_kv(oss, indent, "quant.zero_points", quant_zero_points_string(quant.zero_points));
}

void append_quant_with_prefix(std::ostringstream& oss, const int indent, const std::string& prefix,
                              const QuantStaticSpec& quant) {
  append_kv(oss, indent, prefix + ".granularity", quant_granularity_name(quant.granularity));
  append_kv_num(oss, indent, prefix + ".axis", quant.axis);
  append_kv(oss, indent, prefix + ".scales", double_vector_string(quant.scales));
  append_kv(oss, indent, prefix + ".zero_points", quant_zero_points_string(quant.zero_points));
}

void append_optional_quant(std::ostringstream& oss, const int indent,
                           const std::optional<QuantStaticSpec>& quant) {
  if (!quant.has_value()) {
    append_kv(oss, indent, "quant", "none");
    return;
  }
  append_quant(oss, indent, *quant);
}

void append_mpk_quant(std::ostringstream& oss, const int indent,
                      const std::optional<MpkQuantContract>& quant) {
  if (!quant.has_value()) {
    append_kv(oss, indent, "quant", "none");
    return;
  }
  append_kv_num(oss, indent, "quant.axis", quant->axis);
  append_kv(oss, indent, "quant.scales", double_vector_string(quant->scales));
  append_kv(oss, indent, "quant.zero_points", quant_zero_points_string(quant->zero_points));
}

void append_tensor_static_spec(std::ostringstream& oss, const int indent, const std::string& prefix,
                               const TensorStaticSpec& tensor) {
  append_kv_num(oss, indent, prefix + ".tensor_index", tensor.tensor_index);
  append_kv(oss, indent, prefix + ".shape", shape_string(tensor.shape));
  append_kv(oss, indent, prefix + ".dtype", tensor.dtype.empty() ? "<empty>" : tensor.dtype);
  append_kv(oss, indent, prefix + ".layout", tensor.layout.empty() ? "<empty>" : tensor.layout);
  append_kv_num(oss, indent, prefix + ".max_w", tensor.max_w);
  append_kv_num(oss, indent, prefix + ".max_h", tensor.max_h);
  append_kv_num(oss, indent, prefix + ".max_stride", tensor.max_stride);
  append_kv(oss, indent, prefix + ".semantic_tag",
            tensor.semantic_tag.empty() ? "<empty>" : tensor.semantic_tag);
}

void append_physical_buffer(std::ostringstream& oss, const int indent, const std::string& prefix,
                            const PhysicalBufferStaticSpec& buffer) {
  append_kv_num(oss, indent, prefix + ".physical_index", buffer.physical_index);
  append_kv_num(oss, indent, prefix + ".allocator_index", buffer.allocator_index);
  append_kv_num(oss, indent, prefix + ".source_physical_index", buffer.source_physical_index);
  append_kv_num(oss, indent, prefix + ".size_bytes", buffer.size_bytes);
  append_kv_num(oss, indent, prefix + ".source_byte_offset", buffer.source_byte_offset);
  append_kv(oss, indent, prefix + ".device_kind", device_kind_name(buffer.device_kind));
  append_kv_num(oss, indent, prefix + ".memory_flags", buffer.memory_flags);
  append_kv(oss, indent, prefix + ".segment_name",
            buffer.segment_name.empty() ? "<empty>" : buffer.segment_name);
}

void append_logical_tensor(std::ostringstream& oss, const int indent, const std::string& prefix,
                           const LogicalTensorStaticSpec& tensor) {
  append_kv_num(oss, indent, prefix + ".logical_index", tensor.logical_index);
  append_kv_num(oss, indent, prefix + ".backend_output_index", tensor.backend_output_index);
  append_kv_num(oss, indent, prefix + ".physical_index", tensor.physical_index);
  append_kv_num(oss, indent, prefix + ".output_slot", tensor.output_slot);
  append_kv_num(oss, indent, prefix + ".tensor_index", tensor.tensor_index);
  append_kv_num(oss, indent, prefix + ".byte_offset", tensor.byte_offset);
  append_kv_num(oss, indent, prefix + ".size_bytes", tensor.size_bytes);
  append_kv(oss, indent, prefix + ".shape", shape_string(tensor.shape));
  append_kv(oss, indent, prefix + ".stride_bytes", int_vector_string(tensor.stride_bytes));
  append_kv(oss, indent, prefix + ".dtype", tensor.dtype.empty() ? "<empty>" : tensor.dtype);
  append_kv(oss, indent, prefix + ".layout", tensor.layout.empty() ? "<empty>" : tensor.layout);
  append_kv(oss, indent, prefix + ".logical_name",
            tensor.logical_name.empty() ? "<empty>" : tensor.logical_name);
  append_kv(oss, indent, prefix + ".backend_name",
            tensor.backend_name.empty() ? "<empty>" : tensor.backend_name);
  append_kv(oss, indent, prefix + ".segment_name",
            tensor.segment_name.empty() ? "<empty>" : tensor.segment_name);
  append_optional_quant(oss, indent, tensor.quant);
}

void append_logical_input(std::ostringstream& oss, const int indent, const std::string& prefix,
                          const LogicalInputStaticSpec& tensor) {
  append_kv_num(oss, indent, prefix + ".logical_index", tensor.logical_index);
  append_kv_num(oss, indent, prefix + ".backend_input_index", tensor.backend_input_index);
  append_kv_num(oss, indent, prefix + ".physical_index", tensor.physical_index);
  append_kv_num(oss, indent, prefix + ".byte_offset", tensor.byte_offset);
  append_kv_num(oss, indent, prefix + ".size_bytes", tensor.size_bytes);
  append_kv(oss, indent, prefix + ".shape", shape_string(tensor.shape));
  append_kv(oss, indent, prefix + ".stride_bytes", int_vector_string(tensor.stride_bytes));
  append_kv(oss, indent, prefix + ".dtype", tensor.dtype.empty() ? "<empty>" : tensor.dtype);
  append_kv(oss, indent, prefix + ".layout", tensor.layout.empty() ? "<empty>" : tensor.layout);
  append_kv(oss, indent, prefix + ".logical_name",
            tensor.logical_name.empty() ? "<empty>" : tensor.logical_name);
  append_kv(oss, indent, prefix + ".backend_name",
            tensor.backend_name.empty() ? "<empty>" : tensor.backend_name);
  append_kv(oss, indent, prefix + ".segment_name",
            tensor.segment_name.empty() ? "<empty>" : tensor.segment_name);
  append_optional_quant(oss, indent, tensor.quant);
}

void append_input_binding(std::ostringstream& oss, const int indent, const std::string& prefix,
                          const InputBindingStaticSpec& binding) {
  append_kv_num(oss, indent, prefix + ".sink_pad_index", binding.sink_pad_index);
  append_kv_num(oss, indent, prefix + ".local_logical_input_index",
                binding.local_logical_input_index);
  append_kv_num(oss, indent, prefix + ".src_stage_index", binding.src_stage_index);
  append_kv(oss, indent, prefix + ".src_stage_id",
            binding.src_stage_id.empty() ? "<empty>" : binding.src_stage_id);
  append_kv_num(oss, indent, prefix + ".src_logical_output_index",
                binding.src_logical_output_index);
  append_kv_num(oss, indent, prefix + ".src_output_slot", binding.src_output_slot);
  append_kv_num(oss, indent, prefix + ".src_physical_output_index",
                binding.src_physical_output_index);
  append_kv_num(oss, indent, prefix + ".src_physical_size_bytes", binding.src_physical_size_bytes);
  append_kv_num(oss, indent, prefix + ".src_physical_byte_offset",
                binding.src_physical_byte_offset);
  append_kv(oss, indent, prefix + ".required", bool_name(binding.required));
  append_kv(oss, indent, prefix + ".cm_input_name",
            binding.cm_input_name.empty() ? "<empty>" : binding.cm_input_name);
  append_kv(oss, indent, prefix + ".source_segment_name",
            binding.source_segment_name.empty() ? "<empty>" : binding.source_segment_name);
}

void append_stage_output_route(std::ostringstream& oss, const int indent, const std::string& prefix,
                               const StageOutputRoute& route) {
  append_kv_num(oss, indent, prefix + ".output_slot", route.output_slot);
  append_kv_num(oss, indent, prefix + ".logical_output_index", route.logical_output_index);
  append_kv_num(oss, indent, prefix + ".tensor_index", route.tensor_index);
  append_kv(oss, indent, prefix + ".cm_output_name",
            route.cm_output_name.empty() ? "<empty>" : route.cm_output_name);
  append_kv(oss, indent, prefix + ".segment_name",
            route.segment_name.empty() ? "<empty>" : route.segment_name);
}

void append_mpk_tensor(std::ostringstream& oss, const int indent, const std::string& prefix,
                       const MpkTensorContract& tensor) {
  append_kv_num(oss, indent, prefix + ".tensor_index", tensor.tensor_index);
  append_kv_num(oss, indent, prefix + ".physical_index", tensor.physical_index);
  append_kv_num(oss, indent, prefix + ".source_physical_index", tensor.source_physical_index);
  append_kv(oss, indent, prefix + ".name", tensor.name.empty() ? "<empty>" : tensor.name);
  append_kv(oss, indent, prefix + ".segment_name",
            tensor.segment_name.empty() ? "<empty>" : tensor.segment_name);
  append_kv(oss, indent, prefix + ".kind", tensor.kind.empty() ? "<empty>" : tensor.kind);
  append_kv(oss, indent, prefix + ".dtype", tensor.dtype.empty() ? "<empty>" : tensor.dtype);
  append_kv(oss, indent, prefix + ".mpk_shape", shape_string(tensor.mpk_shape));
  append_kv(oss, indent, prefix + ".shape_semantics",
            mpk_shape_semantics_name(tensor.shape_semantics));
  append_kv_num(oss, indent, prefix + ".size_bytes", tensor.size_bytes);
  append_kv_num(oss, indent, prefix + ".byte_offset", tensor.byte_offset);
  append_kv_num(oss, indent, prefix + ".source_byte_offset", tensor.source_byte_offset);
  append_kv(oss, indent, prefix + ".stride_bytes", int_vector_string(tensor.stride_bytes));
  append_kv(oss, indent, prefix + ".logical_shape", shape_string(tensor.logical_shape));
  append_kv(oss, indent, prefix + ".logical_dtype",
            tensor.logical_dtype.empty() ? "<empty>" : tensor.logical_dtype);
  append_kv(oss, indent, prefix + ".logical_source_plugin",
            tensor.logical_source_plugin.empty() ? "<empty>" : tensor.logical_source_plugin);
  append_kv(oss, indent, prefix + ".logical_source_kernel",
            tensor.logical_source_kernel.empty() ? "<empty>" : tensor.logical_source_kernel);
  append_kv_num(oss, indent, prefix + ".logical_source_sequence", tensor.logical_source_sequence);
}

std::string ev_axis_semantics_string(const sima_ev_shape_desc& shape) {
  std::ostringstream oss;
  oss << "[";
  const auto rank = std::min<std::uint32_t>(shape.rank, SIMA_EV_MAX_RANK);
  for (std::uint32_t i = 0; i < rank; ++i) {
    if (i) {
      oss << ",";
    }
    switch (static_cast<sima_ev_axis_semantic>(shape.axis_semantics[i])) {
    case SIMA_EV_AXIS_N:
      oss << "N";
      break;
    case SIMA_EV_AXIS_D:
      oss << "D";
      break;
    case SIMA_EV_AXIS_H:
      oss << "H";
      break;
    case SIMA_EV_AXIS_W:
      oss << "W";
      break;
    case SIMA_EV_AXIS_C:
      oss << "C";
      break;
    case SIMA_EV_AXIS_UNKNOWN:
    default:
      oss << "?";
      break;
    }
  }
  oss << "]";
  return oss.str();
}

void append_ev_tensor_desc(std::ostringstream& oss, const int indent, const std::string& prefix,
                           const sima_ev_tensor_desc& desc) {
  std::vector<std::int64_t> shape;
  const auto rank = std::min<std::uint32_t>(desc.shape.rank, SIMA_EV_MAX_RANK);
  shape.reserve(rank);
  for (std::uint32_t i = 0; i < rank; ++i) {
    shape.push_back(static_cast<std::int64_t>(desc.shape.sizes[i]));
  }
  append_kv(oss, indent, prefix + ".shape", shape_string(shape));
  append_kv(oss, indent, prefix + ".axis_semantics", ev_axis_semantics_string(desc.shape));
  append_kv(oss, indent, prefix + ".layout_hint",
            tensorsemantics::layout_token_from_ev_tensor_desc(desc));
  append_kv_num(oss, indent, prefix + ".storage.nbytes", desc.storage.nbytes);
}

void append_mpk_plugin_metadata(std::ostringstream& oss, const int indent,
                                const MpkPluginIoContract& plugin) {
  append_kv(oss, indent, "slice_shape", shape_string(plugin.slice_shape));
  append_kv(oss, indent, "slice_begin", int_vector_string(plugin.slice_begin));
  append_kv(oss, indent, "slice_end", int_vector_string(plugin.slice_end));
  append_kv(oss, indent, "frame_shape", shape_string(plugin.frame_shape));
  append_kv(oss, indent, "frame_type", plugin.frame_type.empty() ? "<empty>" : plugin.frame_type);
  append_kv(oss, indent, "out_shape_raw", shape_string(plugin.out_shape_raw));
  append_kv(oss, indent, "canonical_input_dtype",
            plugin.canonical_input_dtype.empty() ? "<empty>" : plugin.canonical_input_dtype);
  append_kv(oss, indent, "canonical_output_dtype",
            plugin.canonical_output_dtype.empty() ? "<empty>" : plugin.canonical_output_dtype);
  append_kv(oss, indent, "has_canonical_processcvu_contract",
            bool_name(plugin.has_canonical_processcvu_contract));
  // slice_d/h/w/c removed; individual dims now live in slice_shape
  // (already printed above as "slice_shape")
  append_kv(oss, indent, "has_align_c16", bool_name(plugin.has_align_c16));
  append_kv(oss, indent, "align_c16", bool_name(plugin.align_c16));
  append_kv(oss, indent, "has_cblock", bool_name(plugin.has_cblock));
  append_kv(oss, indent, "cblock", bool_name(plugin.cblock));
  append_mpk_quant(oss, indent, plugin.quant);
}

void append_archive_summary(std::ostringstream& oss, const mpk::MpKManifest& manifest,
                            const std::string& mpk_json_path = {},
                            const std::string& model_name = {}) {
  append_section_header(oss, "Archive Summary");
  append_kv(oss, 0, "archive_path", manifest.archive_path);
  append_kv(oss, 0, "package_name", manifest.package_name);
  append_kv(oss, 0, "version", manifest.version);
  append_kv_num(oss, 0, "archive_size_bytes", manifest.archive_size_bytes);
  append_kv_num(oss, 0, "entry_count", manifest.entries.size());
  append_kv(oss, 0, "has_pipeline_sequence", bool_name(manifest.has_pipeline_sequence));
  append_kv(oss, 0, "has_model_binary", bool_name(manifest.has_model_binary));
  if (!mpk_json_path.empty()) {
    append_kv(oss, 0, "mpk_json_path", mpk_json_path);
  }
  if (!model_name.empty()) {
    append_kv(oss, 0, "model_name", model_name);
  }
  oss << "\n";
}

void append_route_summary(std::ostringstream& oss, const ModelPack& pack,
                          const ModelContractReportContext& context) {
  append_section_header(oss, "Route Summary");
  const ParsedModelInfo parsed = parse_model_from_pack(pack);
  if (context.model_info != nullptr) {
    const auto& info = *context.model_info;
    append_kv(oss, 0, "route_summary_available", "true");
    append_kv(oss, 0, "include_preprocess_stage",
              bool_name(info.selection.include_preprocess_stage));
    append_kv(oss, 0, "include_postprocess_stage",
              bool_name(info.selection.include_postprocess_stage));
    append_kv(oss, 0, "infer_only", bool_name(info.selection.infer_only));
    append_kv(oss, 0, "preprocess_graph",
              info.selection.preprocess_graph.empty() ? "<empty>"
                                                      : info.selection.preprocess_graph);
    append_kv(oss, 0, "selected_post_kind",
              context.selected_post_kind.empty() ? "<unknown>" : context.selected_post_kind);
    append_kv_num(oss, 0, "output_topology.physical", info.output_topology.physical_outputs);
    append_kv_num(oss, 0, "output_topology.logical", info.output_topology.logical_outputs);
    append_kv(oss, 0, "output_topology.packed", bool_name(info.output_topology.packed_outputs));
    append_kv(oss, 0, "pre_kernels", csv_or_none(info.pre_kernels));
    append_kv(oss, 0, "post_kernels", csv_or_none(info.post_kernels));
  } else {
    append_kv(oss, 0, "route_summary_available", "false");
    append_kv(oss, 0, "selected_post_kind", "<unavailable>");
    append_kv_num(oss, 0, "output_topology.physical", parsed.outputs.physical.size());
    append_kv_num(oss, 0, "output_topology.logical", parsed.outputs.logical.size());
    append_kv(oss, 0, "output_topology.packed", bool_name(parsed.outputs.packed_output));
    append_kv(oss, 0, "pre_kernels", csv_or_none(parsed.pre_kernels));
    append_kv(oss, 0, "post_kernels", csv_or_none(parsed.post_kernels));
  }
  if (context.preprocess_plan != nullptr) {
    append_kv(oss, 0, "resolved_input_kind",
              input_kind_name(context.preprocess_plan->resolved_kind));
    append_kv(oss, 0, "resolved_graph_family",
              preprocess_graph_family_name_local(context.preprocess_plan->graph_family));
    append_kv_num(oss, 0, "ingress.count", context.preprocess_plan->ingress_contracts.size());
    for (std::size_t i = 0; i < context.preprocess_plan->ingress_contracts.size(); ++i) {
      const auto& ingress = context.preprocess_plan->ingress_contracts[i];
      const std::string prefix = "ingress[" + std::to_string(i) + "]";
      append_kv(oss, 0, prefix + ".media_type",
                ingress.media_type.empty() ? "<empty>" : ingress.media_type);
      append_kv(oss, 0, prefix + ".format", ingress.format.empty() ? "<empty>" : ingress.format);
      append_kv_num(oss, 0, prefix + ".width", ingress.width);
      append_kv_num(oss, 0, prefix + ".height", ingress.height);
      append_kv_num(oss, 0, prefix + ".depth", ingress.depth);
    }
    append_kv(oss, 0, "mla.media_type",
              context.preprocess_plan->mla_contract.media_type.empty()
                  ? "<empty>"
                  : context.preprocess_plan->mla_contract.media_type);
    append_kv(oss, 0, "mla.format",
              context.preprocess_plan->mla_contract.format.empty()
                  ? "<empty>"
                  : context.preprocess_plan->mla_contract.format);
  }
  const ExecutionPlan plan = pack.execution_plan();
  append_kv(oss, 0, "planned_pre_chain", execution_plan_chain_string(plan.pre));
  append_kv(oss, 0, "planned_infer_chain", execution_plan_chain_string(plan.infer));
  append_kv(oss, 0, "planned_post_chain", execution_plan_chain_string(plan.post));
  for (std::size_t i = 0; i < parsed.warnings.size(); ++i) {
    append_kv(oss, 0, "warning[" + std::to_string(i) + "]", parsed.warnings[i]);
  }
  oss << "\n";
}

void append_raw_mpk_graph(std::ostringstream& oss, const ModelPack& pack,
                          const ModelContractReportOptions& options, ReportStats* stats) {
  const auto& maybe_contract = pack.mpk_contract();
  if (!maybe_contract.has_value()) {
    return;
  }
  const MpkContract& contract = *maybe_contract;
  const auto ordered = pipeline_internal::sima::plugins_in_execution_order(contract);
  const ParsedModelInfo parsed = parse_model_from_pack(pack);
  std::optional<std::size_t> mla_plugin_index;
  if (parsed.mla_plugin_index >= 0) {
    mla_plugin_index = static_cast<std::size_t>(parsed.mla_plugin_index);
  }

  std::unordered_map<std::size_t, std::vector<const MpkContractEdge*>> incoming;
  std::unordered_map<std::size_t, std::vector<const MpkContractEdge*>> outgoing;
  for (const auto& edge : contract.edges) {
    incoming[edge.dst_plugin_index].push_back(&edge);
    outgoing[edge.src_plugin_index].push_back(&edge);
  }

  append_section_header(oss, "Raw MPK Contract Graph");
  append_kv_num(oss, 0, "plugin_count", contract.plugins.size());
  append_kv_num(oss, 0, "edge_count", contract.edges.size());
  append_kv(oss, 0, "execution_order", uint_vector_string(ordered));

  for (std::size_t rank = 0; rank < ordered.size(); ++rank) {
    const std::size_t plugin_index = ordered[rank];
    if (plugin_index >= contract.plugins.size()) {
      continue;
    }
    const auto& plugin = contract.plugins[plugin_index];
    if (!raw_plugin_selected(options.stage_filter, plugin_index, mla_plugin_index)) {
      continue;
    }
    if (!matches_plugin_filters(options.plugin_filters,
                                {plugin.name, plugin.plugin_id, plugin.processor, plugin.kernel})) {
      continue;
    }
    if (stats) {
      stats->raw_items = true;
    }
    oss << "[plugin " << rank << "] index=" << plugin_index << "\n";
    append_kv(oss, 2, "name", plugin.name.empty() ? "<empty>" : plugin.name);
    append_kv(oss, 2, "plugin_id", plugin.plugin_id.empty() ? "<empty>" : plugin.plugin_id);
    append_kv(oss, 2, "processor", plugin.processor.empty() ? "<empty>" : plugin.processor);
    append_kv(oss, 2, "kernel", plugin.kernel.empty() ? "<empty>" : plugin.kernel);
    append_kv_num(oss, 2, "sequence", plugin.sequence);
    append_mpk_plugin_metadata(oss, 2, plugin);
    for (std::size_t i = 0; i < plugin.input_tensors.size(); ++i) {
      append_mpk_tensor(oss, 2, "input[" + std::to_string(i) + "]", plugin.input_tensors[i]);
    }
    for (std::size_t i = 0; i < plugin.output_tensors.size(); ++i) {
      append_mpk_tensor(oss, 2, "output[" + std::to_string(i) + "]", plugin.output_tensors[i]);
    }
    const auto in_it = incoming.find(plugin_index);
    if (in_it != incoming.end()) {
      for (std::size_t i = 0; i < in_it->second.size(); ++i) {
        const auto& edge = *in_it->second[i];
        append_kv(oss, 2, "incoming[" + std::to_string(i) + "]",
                  edge.src_plugin + ":" + std::to_string(edge.src_output_index) + " -> " +
                      edge.dst_plugin + ":" + std::to_string(edge.dst_input_index) +
                      " tensor=" + edge.tensor_name);
      }
    }
    const auto out_it = outgoing.find(plugin_index);
    if (out_it != outgoing.end()) {
      for (std::size_t i = 0; i < out_it->second.size(); ++i) {
        const auto& edge = *out_it->second[i];
        append_kv(oss, 2, "outgoing[" + std::to_string(i) + "]",
                  edge.src_plugin + ":" + std::to_string(edge.src_output_index) + " -> " +
                      edge.dst_plugin + ":" + std::to_string(edge.dst_input_index) +
                      " tensor=" + edge.tensor_name);
      }
    }
    oss << "\n";
  }
}

template <typename T>
void append_named_collection(
    std::ostringstream& oss, const int indent, const std::string& base, const std::vector<T>& items,
    const std::function<void(std::ostringstream&, int, const std::string&, const T&)>& fn) {
  for (std::size_t i = 0; i < items.size(); ++i) {
    fn(oss, indent, base + "[" + std::to_string(i) + "]", items[i]);
  }
}

void append_compiled_runtime_contract(std::ostringstream& oss, const int indent,
                                      const CompiledRuntimeContract& runtime) {
  append_kv(oss, indent, "runtime.plugin_kind",
            runtime.plugin_kind.empty() ? "<empty>" : runtime.plugin_kind);
  append_named_collection<LogicalInputStaticSpec>(oss, indent, "runtime.logical_input",
                                                  runtime.logical_inputs, append_logical_input);
  append_named_collection<InputBindingStaticSpec>(oss, indent, "runtime.input_binding",
                                                  runtime.input_bindings, append_input_binding);
  append_named_collection<PhysicalBufferStaticSpec>(
      oss, indent, "runtime.physical_input", runtime.physical_inputs, append_physical_buffer);
  append_named_collection<PhysicalBufferStaticSpec>(
      oss, indent, "runtime.physical_output", runtime.physical_outputs, append_physical_buffer);
  append_named_collection<LogicalTensorStaticSpec>(oss, indent, "runtime.logical_output",
                                                   runtime.logical_outputs, append_logical_tensor);
  append_named_collection<StageOutputRoute>(oss, indent, "runtime.output_route",
                                            runtime.output_order, append_stage_output_route);
  for (std::size_t i = 0; i < runtime.output_quant.size(); ++i) {
    append_quant_with_prefix(oss, indent, "runtime.output_quant[" + std::to_string(i) + "]",
                             runtime.output_quant[i]);
  }
  append_kv(oss, indent, "runtime.required_preprocess_meta_fields",
            csv_or_none(runtime.required_preprocess_meta_fields));
}

void append_processcvu_contract(std::ostringstream& oss, const int indent,
                                const CompiledProcessCvuContract& compiled) {
  const auto& payload = compiled.payload;
  append_kv(oss, indent, "contract_type", "processcvu");
  append_kv(oss, indent, "payload.graph_family",
            payload.graph_family.empty() ? "<empty>" : payload.graph_family);
  append_kv(oss, indent, "payload.graph_name",
            payload.graph_name.empty() ? "<empty>" : payload.graph_name);
  append_kv(oss, indent, "payload.default_input_name",
            payload.default_input_name.empty() ? "<empty>" : payload.default_input_name);
  append_kv(oss, indent, "payload.default_output_names", csv_or_none(payload.default_output_names));
  append_kv(oss, indent, "payload.primary_output_name",
            payload.primary_output_name.empty() ? "<empty>" : payload.primary_output_name);
  append_kv(oss, indent, "payload.primary_output_transport_kind",
            output_transport_kind_name(payload.primary_output_transport_kind));
  append_kv(oss, indent, "payload.primary_output_semantic_kind",
            output_semantic_kind_name(payload.primary_output_semantic_kind));
  append_kv(oss, indent, "payload.input_dtype",
            payload.input_dtype.empty() ? "<empty>" : payload.input_dtype);
  append_kv(oss, indent, "payload.output_dtype",
            payload.output_dtype.empty() ? "<empty>" : payload.output_dtype);
  append_kv(oss, indent, "payload.out_dtype",
            payload.out_dtype.empty() ? "<empty>" : payload.out_dtype);
  for (std::size_t i = 0; i < payload.input_tensors.size(); ++i) {
    append_ev_tensor_desc(oss, indent, "payload.input_tensor[" + std::to_string(i) + "]",
                          payload.input_tensors[i]);
  }
  for (std::size_t i = 0; i < payload.output_tensors.size(); ++i) {
    append_ev_tensor_desc(oss, indent, "payload.output_tensor[" + std::to_string(i) + "]",
                          payload.output_tensors[i]);
  }
  append_kv(oss, indent, "payload.canonical_contract", bool_name(payload.canonical_contract));
  append_kv(oss, indent, "payload.slice_shape_raw", shape_string(payload.slice_shape_raw));
  append_kv(oss, indent, "payload.out_shape_raw", shape_string(payload.out_shape_raw));
  append_kv(oss, indent, "payload.align_c16", bool_name(payload.align_c16));
  append_kv(oss, indent, "payload.cblock", bool_name(payload.cblock));
  append_kv(oss, indent, "preproc_single_output_handoff",
            bool_name(compiled.preproc_single_output_handoff));
  append_compiled_runtime_contract(oss, indent, compiled.runtime_contract);
  append_kv(oss, indent, "exposed.primary_output_name",
            compiled.exposed_view.primary_output_name.empty()
                ? "<empty>"
                : compiled.exposed_view.primary_output_name);
  append_named_collection<StageOutputRoute>(oss, indent, "exposed.output_route",
                                            compiled.exposed_view.exposed_output_order,
                                            append_stage_output_route);
  append_named_collection<LogicalTensorStaticSpec>(oss, indent, "exposed.logical_output",
                                                   compiled.exposed_view.exposed_logical_outputs,
                                                   append_logical_tensor);
}

void append_mla_contract(std::ostringstream& oss, const int indent,
                         const CompiledMlaContract& compiled) {
  append_kv(oss, indent, "contract_type", "processmla");
  append_kv(oss, indent, "payload.model_path",
            compiled.payload.model_path.empty() ? "<empty>" : compiled.payload.model_path);
  append_kv_num(oss, indent, "payload.batch_size", compiled.payload.batch_size);
  append_kv_num(oss, indent, "payload.batch_sz_model", compiled.payload.batch_sz_model);
  for (std::size_t i = 0; i < compiled.payload.dispatcher_output_names.size(); ++i) {
    append_kv(oss, indent, "payload.dispatcher_output_name[" + std::to_string(i) + "]",
              compiled.payload.dispatcher_output_names[i]);
  }
  for (std::size_t i = 0; i < compiled.payload.dispatcher_output_sizes.size(); ++i) {
    append_kv_num(oss, indent, "payload.dispatcher_output_size[" + std::to_string(i) + "]",
                  compiled.payload.dispatcher_output_sizes[i]);
  }
  append_named_collection<TensorStaticSpec>(oss, indent, "inputs", compiled.inputs,
                                            append_tensor_static_spec);
  append_named_collection<PhysicalBufferStaticSpec>(oss, indent, "dispatcher_physical_output",
                                                    compiled.dispatcher_physical_outputs,
                                                    append_physical_buffer);
  append_compiled_runtime_contract(oss, indent, compiled.runtime_contract);
}

void append_boxdecode_contract(std::ostringstream& oss, const int indent,
                               const CompiledBoxDecodeContract& compiled) {
  append_kv(oss, indent, "contract_type", "boxdecode");
  append_kv(oss, indent, "payload.input_dtype",
            compiled.payload.input_dtype.empty() ? "<empty>" : compiled.payload.input_dtype);
  append_kv(oss, indent, "payload.tess_needed", bool_name(compiled.payload.tess_needed));
  append_kv(oss, indent, "payload.quant_needed", bool_name(compiled.payload.quant_needed));
  append_kv(oss, indent, "payload.model_owned_flags",
            bool_name(compiled.payload.model_owned_flags));
  append_kv(oss, indent, "payload.quant_contract_required",
            bool_name(compiled.payload.quant_contract_required));
  append_kv_num(oss, indent, "payload.detection_threshold", compiled.payload.detection_threshold);
  append_kv_num(oss, indent, "payload.nms_iou_threshold", compiled.payload.nms_iou_threshold);
  append_kv_num(oss, indent, "payload.topk", compiled.payload.topk);
  append_kv(oss, indent, "payload.slice_shapes", shape_descs_string(compiled.payload.slice_shapes));
  append_compiled_runtime_contract(oss, indent, compiled.runtime_contract);
}

void append_dequant_contract(std::ostringstream& oss, const int indent,
                             const CompiledDequantContract& compiled) {
  append_kv(oss, indent, "contract_type", "dequant");
  append_compiled_runtime_contract(oss, indent, compiled.runtime_contract);
}

void append_transport_contract(std::ostringstream& oss, const int indent,
                               const CompiledTransportContract& compiled) {
  append_kv(oss, indent, "contract_type", "transport");
  append_kv(oss, indent, "transport.plugin_kind",
            compiled.plugin_kind.empty() ? "<empty>" : compiled.plugin_kind);
  append_kv(oss, indent, "transport.kernel_kind",
            compiled.kernel_kind.empty() ? "<empty>" : compiled.kernel_kind);
  append_kv(oss, indent, "transport.payload_kind", stage_payload_kind_name(compiled.payload_kind));
  append_kv(oss, indent, "transport.model_managed_stage", bool_name(compiled.model_managed_stage));
  append_compiled_runtime_contract(oss, indent, compiled.runtime_contract);
}

template <typename StageVec>
void append_stage_plan_group(std::ostringstream& oss, const std::string& group_name,
                             const StageVec& stages, const ModelContractReportOptions& options,
                             ReportStats* stats) {
  oss << "[" << group_name << "]\n";
  std::size_t emitted = 0U;
  for (std::size_t i = 0; i < stages.size(); ++i) {
    const auto& stage = stages[i];
    if (!matches_plugin_filters(options.plugin_filters,
                                {stage.stage_name, stage.plugin_id, stage.processor, stage.kernel,
                                 stage.factory_name})) {
      continue;
    }
    if (stats) {
      stats->planned_items = true;
    }
    oss << "  [stage " << emitted++ << "]\n";
    append_kv_num(oss, 4, "order_index", stage.order_index);
    append_kv(oss, 4, "stage_name", stage.stage_name.empty() ? "<empty>" : stage.stage_name);
    if (stage.mpk_plugin_index.has_value()) {
      append_kv_num(oss, 4, "mpk_plugin_index", *stage.mpk_plugin_index);
    } else {
      append_kv(oss, 4, "mpk_plugin_index", "<none>");
    }
    append_kv(oss, 4, "factory_name", stage.factory_name.empty() ? "<empty>" : stage.factory_name);
    append_kv(oss, 4, "plugin_id", stage.plugin_id.empty() ? "<empty>" : stage.plugin_id);
    append_kv(oss, 4, "processor", stage.processor.empty() ? "<empty>" : stage.processor);
    append_kv(oss, 4, "kernel", stage.kernel.empty() ? "<empty>" : stage.kernel);
    append_kv(oss, 4, "kind", execution_stage_kind_name(stage.kind));
  }
  if (emitted == 0U) {
    append_kv(oss, 2, "note", "no matching stages");
  }
}

template <typename StageVec>
void append_typed_contract_group(std::ostringstream& oss, const std::string& group_name,
                                 const StageVec& stages,
                                 const std::vector<ModelFragment::StageFacts>& facts,
                                 const ModelContractReportOptions& options, ReportStats* stats) {
  oss << "[" << group_name << "]\n";
  std::size_t emitted = 0U;
  for (std::size_t i = 0; i < stages.size(); ++i) {
    const auto& stage = stages[i];
    const auto* fact = (i < facts.size()) ? &facts[i] : nullptr;
    if (!matches_plugin_filters(options.plugin_filters,
                                {stage.stage_name, stage.plugin_id, stage.processor, stage.kernel,
                                 stage.factory_name})) {
      continue;
    }
    if (stats) {
      stats->typed_items = true;
    }
    oss << "  [stage " << emitted++ << "]\n";
    append_kv_num(oss, 4, "order_index", stage.order_index);
    append_kv(oss, 4, "stage_name", stage.stage_name.empty() ? "<empty>" : stage.stage_name);
    if (stage.mpk_plugin_index.has_value()) {
      append_kv_num(oss, 4, "mpk_plugin_index", *stage.mpk_plugin_index);
    } else {
      append_kv(oss, 4, "mpk_plugin_index", "<none>");
    }
    append_kv(oss, 4, "kind", execution_stage_kind_name(stage.kind));
    if (!fact) {
      append_kv(oss, 4, "stage_facts", "<missing>");
      continue;
    }
    append_kv(oss, 4, "stage_fact.stage_name",
              fact->stage_name.empty() ? "<empty>" : fact->stage_name);
    append_kv_num(oss, 4, "stage_fact.stage_order", fact->stage_order);
    if (fact->processcvu_contract.has_value()) {
      append_processcvu_contract(oss, 4, *fact->processcvu_contract);
    }
    if (fact->mla_compiled.has_value()) {
      append_mla_contract(oss, 4, *fact->mla_compiled);
    }
    if (fact->boxdecode_compiled.has_value()) {
      append_boxdecode_contract(oss, 4, *fact->boxdecode_compiled);
    }
    if (fact->dequant_compiled.has_value()) {
      append_dequant_contract(oss, 4, *fact->dequant_compiled);
    }
    if (fact->transport_compiled.has_value()) {
      append_transport_contract(oss, 4, *fact->transport_compiled);
    }
  }
  if (emitted == 0U) {
    append_kv(oss, 2, "note", "no matching stages");
  }
}

void append_planned_model_stages(std::ostringstream& oss, const ModelPack& pack,
                                 const ModelContractReportOptions& options, ReportStats* stats) {
  append_section_header(oss, "Planned Model Stages");
  const ExecutionPlan plan = pack.execution_plan();
  if (stage_group_selected(options.stage_filter, ModelStage::Preprocess)) {
    append_stage_plan_group(oss, "pre", plan.pre, options, stats);
  }
  if (stage_group_selected(options.stage_filter, ModelStage::MlaOnly)) {
    append_stage_plan_group(oss, "infer", plan.infer, options, stats);
  }
  if (stage_group_selected(options.stage_filter, ModelStage::Postprocess)) {
    append_stage_plan_group(oss, "post", plan.post, options, stats);
  }
  oss << "\n";
}

void append_typed_runtime_contracts(std::ostringstream& oss, const ModelPack& pack,
                                    const ModelContractReportOptions& options, ReportStats* stats) {
  append_section_header(oss, "Typed Runtime/Static Contracts");
  const ExecutionPlan plan = pack.execution_plan();
  if (stage_group_selected(options.stage_filter, ModelStage::Preprocess)) {
    const auto facts = pack.stage_facts_for_model_stage(ModelStage::Preprocess);
    append_typed_contract_group(oss, "pre", plan.pre, facts, options, stats);
  }
  if (stage_group_selected(options.stage_filter, ModelStage::MlaOnly)) {
    const auto facts = pack.stage_facts_for_model_stage(ModelStage::MlaOnly);
    append_typed_contract_group(oss, "infer", plan.infer, facts, options, stats);
  }
  if (stage_group_selected(options.stage_filter, ModelStage::Postprocess)) {
    const auto facts = pack.stage_facts_for_model_stage(ModelStage::Postprocess);
    append_typed_contract_group(oss, "post", plan.post, facts, options, stats);
  }
  oss << "\n";
}

void append_gst_fragments(std::ostringstream& oss, const ModelPack& pack,
                          const ModelContractReportOptions& options, ReportStats* stats) {
  if (!options.show_gst) {
    return;
  }
  append_section_header(oss, "GST Fragments");
  auto append_fragment = [&](const ModelStage stage) {
    if (!stage_group_selected(options.stage_filter, stage)) {
      return;
    }
    const std::string label = stage_group_name(stage);
    try {
      const auto fragment = pack.fragment(stage);
      if (!fragment.gst.empty() || !fragment.elements.empty()) {
        if (stats) {
          stats->gst_items = true;
        }
      }
      oss << "[" << label << "]\n";
      append_kv(oss, 2, "gst", fragment.gst.empty() ? "<empty>" : fragment.gst);
      append_kv(oss, 2, "elements", csv_or_none(fragment.elements));
    } catch (const std::exception& e) {
      oss << "[" << label << "]\n";
      append_kv(oss, 2, "error", e.what());
    }
  };
  append_fragment(ModelStage::Preprocess);
  append_fragment(ModelStage::MlaOnly);
  append_fragment(ModelStage::Postprocess);
  oss << "\n";
}

void append_filtered_note_if_needed(std::ostringstream& oss, const ReportStats& stats) {
  if (stats.raw_items || stats.planned_items || stats.typed_items || stats.gst_items) {
    return;
  }
  append_section_header(oss, "Filtered Result");
  append_kv(oss, 0, "note", "no plugins or stages matched the requested filters");
  oss << "\n";
}

std::string selected_post_kind_name_from_model(const Model& model) {
  switch (ModelAccess::resolved_post_kind(model)) {
  case PostRouteStageKind::Detess:
    return "detess";
  case PostRouteStageKind::DetessDequant:
    return "detessdequant";
  case PostRouteStageKind::Dequantize:
    return "dequantize";
  case PostRouteStageKind::BoxDecode:
    return "boxdecode";
  case PostRouteStageKind::Cast:
    return "cast";
  case PostRouteStageKind::Unknown:
    return "unknown";
  case PostRouteStageKind::None:
    break;
  }
  return "none";
}

std::string build_raw_only_report(const mpk::MpKManifest& manifest, const ModelPack& pack,
                                  const ModelContractReportOptions& options,
                                  const std::string& error_section_title,
                                  const std::string& error_message) {
  std::ostringstream oss;
  const auto& contract = pack.mpk_contract();
  append_archive_summary(oss, manifest,
                         contract.has_value() ? contract->mpk_json_path : std::string{},
                         contract.has_value() ? contract->model_name : std::string{});
  ReportStats stats;
  append_raw_mpk_graph(oss, pack, options, &stats);
  append_section_header(oss, error_section_title);
  append_kv(oss, 0, "error", error_message);
  oss << "\n";
  append_filtered_note_if_needed(oss, stats);
  return oss.str();
}

} // namespace

std::string build_model_contract_report(const ModelPack& pack,
                                        const ModelContractReportOptions& options,
                                        const ModelContractReportContext& context) {
  std::ostringstream oss;
  if (context.manifest != nullptr) {
    const auto& contract = pack.mpk_contract();
    append_archive_summary(oss, *context.manifest,
                           contract.has_value() ? contract->mpk_json_path : std::string{},
                           contract.has_value() ? contract->model_name : std::string{});
  }
  append_route_summary(oss, pack, context);
  ReportStats stats;
  append_raw_mpk_graph(oss, pack, options, &stats);
  append_planned_model_stages(oss, pack, options, &stats);
  append_typed_runtime_contracts(oss, pack, options, &stats);
  append_gst_fragments(oss, pack, options, &stats);
  append_filtered_note_if_needed(oss, stats);
  return oss.str();
}

ModelContractInspectionResult
inspect_model_contract_archive(const std::string& tar_gz,
                               const ModelContractReportOptions& options) {
  ModelContractInspectionResult result;
  mpk::MpKManifest manifest;
  try {
    mpk::MpKLoaderOptions loader_options;
    loader_options.reject_unsupported_file_types = false;
    loader_options.require_pipeline_sequence = false;
    manifest = mpk::MpKLoader::inspect(tar_gz, loader_options);
    result.archive_ok = true;
  } catch (const std::exception& e) {
    std::ostringstream oss;
    append_section_header(oss, "Archive Error");
    append_kv(oss, 0, "archive_path", tar_gz);
    append_kv(oss, 0, "error", e.what());
    oss << "\n";
    result.report = oss.str();
    return result;
  }

  std::optional<ModelPack> raw_pack;
  try {
    raw_pack.emplace(tar_gz);
    result.raw_contract_ok = true;
  } catch (const std::exception& e) {
    std::ostringstream oss;
    append_archive_summary(oss, manifest);
    append_section_header(oss, "MPK Contract Error");
    append_kv(oss, 0, "error", e.what());
    oss << "\n";
    result.report = oss.str();
    return result;
  }

  try {
    Model model(tar_gz, Model::Options{});
    result.model_ok = true;
    ModelContractReportContext context;
    context.manifest = &manifest;
    const Model::ModelInfo model_info = model.info();
    const ResolvedPreprocessPlan preprocess_plan = model.resolved_preprocess_plan();
    context.model_info = &model_info;
    context.preprocess_plan = &preprocess_plan;
    context.selected_post_kind = selected_post_kind_name_from_model(model);
    result.report = build_model_contract_report(ModelAccess::pack(model), options, context);
    return result;
  } catch (const std::exception& e) {
    result.report =
        build_raw_only_report(manifest, *raw_pack, options, "Model Planning Error", e.what());
    return result;
  }
}

} // namespace simaai::neat::internal
