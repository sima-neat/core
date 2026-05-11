#include "nodes/sima/Cast.h"

#include "gst/GstHelpers.h"
#include "pipeline/internal/SampleUtil.h"
#include "pipeline/internal/TensorMath.h"
#include "pipeline/internal/contract/CompiledNodeContractQuery.h"
#include "pipeline/internal/contract/ContractFacts.h"
#include "pipeline/internal/sima/StaticSpecBuilders.h"
#include "pipeline/internal/sima/stagesemantics/ProcessCvuStageSemantics.h"
#include "pipeline/internal/sima/stagesemantics/TransportStageSemantics.h"

#include <algorithm>
#include <cctype>
#include <cstdint>
#include <optional>
#include <sstream>
#include <string>

namespace simaai::neat {
namespace {

std::string upper_copy_local(std::string value) {
  std::transform(value.begin(), value.end(), value.begin(),
                 [](unsigned char c) { return static_cast<char>(std::toupper(c)); });
  return value;
}

std::string default_logical_stage_id_local(const Node& node, const std::string& element_name) {
  const std::string user_label = node.user_label();
  return user_label.empty() ? element_name : user_label;
}

std::string tensor_name_from_logical_output_local(
    const pipeline_internal::sima::LogicalTensorStaticSpec& logical,
    std::size_t index) {
  if (!logical.backend_name.empty()) {
    return logical.backend_name;
  }
  if (!logical.logical_name.empty()) {
    return logical.logical_name;
  }
  if (!logical.segment_name.empty()) {
    return logical.segment_name;
  }
  return "output_tensor_" + std::to_string(index);
}

std::string tensor_name_from_sample_local(const Tensor& tensor, std::size_t index) {
  if (!tensor.route.backend_name.empty()) {
    return tensor.route.backend_name;
  }
  if (!tensor.route.name.empty()) {
    return tensor.route.name;
  }
  if (!tensor.route.segment_name.empty()) {
    return tensor.route.segment_name;
  }
  return "output_tensor_" + std::to_string(index);
}

std::string normalize_runtime_dtype_local(std::string token) {
  token = upper_copy_local(std::move(token));
  if (token.find("BF16") != std::string::npos || token.find("BFLOAT16") != std::string::npos) {
    return "BF16";
  }
  if (token.find("FP32") != std::string::npos || token.find("FLOAT32") != std::string::npos) {
    return "FP32";
  }
  return token;
}

std::size_t runtime_dtype_bytes_local(const std::string& token) {
  const std::string up = normalize_runtime_dtype_local(token);
  if (up == "BF16") {
    return 2U;
  }
  if (up == "FP32") {
    return 4U;
  }
  return 0U;
}

std::string cast_output_dtype_local(CastDirection direction) {
  return (direction == CastDirection::Fp32ToBf16) ? std::string("BF16") : std::string("FP32");
}

bool scale_byte_count_local(std::int64_t in_value,
                            std::size_t in_elem_bytes,
                            std::size_t out_elem_bytes,
                            std::int64_t* out_value) {
  if (!out_value || in_elem_bytes == 0U || out_elem_bytes == 0U || in_value < 0) {
    return false;
  }
  const std::int64_t numerator = in_value * static_cast<std::int64_t>(out_elem_bytes);
  if (numerator % static_cast<std::int64_t>(in_elem_bytes) != 0) {
    return false;
  }
  *out_value = numerator / static_cast<std::int64_t>(in_elem_bytes);
  return true;
}

bool scale_size_bytes_local(std::uint64_t in_value,
                            std::size_t in_elem_bytes,
                            std::size_t out_elem_bytes,
                            std::uint64_t* out_value) {
  if (!out_value || in_elem_bytes == 0U || out_elem_bytes == 0U) {
    return false;
  }
  const std::uint64_t numerator = in_value * static_cast<std::uint64_t>(out_elem_bytes);
  if (numerator % static_cast<std::uint64_t>(in_elem_bytes) != 0U) {
    return false;
  }
  *out_value = numerator / static_cast<std::uint64_t>(in_elem_bytes);
  return true;
}

bool transform_logical_output_for_cast(
    const pipeline_internal::sima::LogicalTensorStaticSpec& src,
    CastDirection direction,
    pipeline_internal::sima::LogicalTensorStaticSpec* out,
    std::string* err) {
  if (!out) {
    if (err) {
      *err = "cast contract transform requires output logical tensor";
    }
    return false;
  }
  *out = src;
  const std::size_t in_elem_bytes = runtime_dtype_bytes_local(src.dtype);
  const std::size_t out_elem_bytes = runtime_dtype_bytes_local(cast_output_dtype_local(direction));
  if (in_elem_bytes == 0U || out_elem_bytes == 0U) {
    if (err) {
      *err = "cast contract transform requires FP32/BF16 upstream dtype facts";
    }
    return false;
  }
  std::int64_t scaled_offset = 0;
  std::uint64_t scaled_size = 0U;
  if (!scale_byte_count_local(src.byte_offset, in_elem_bytes, out_elem_bytes, &scaled_offset) ||
      !scale_size_bytes_local(src.size_bytes, in_elem_bytes, out_elem_bytes, &scaled_size)) {
    if (err) {
      *err = "cast contract transform encountered non-aligned logical byte facts";
    }
    return false;
  }
  out->dtype = cast_output_dtype_local(direction);
  out->byte_offset = scaled_offset;
  out->size_bytes = scaled_size;
  for (auto& stride : out->stride_bytes) {
    if (stride == 0) {
      continue;
    }
    std::int64_t scaled_stride = 0;
    if (!scale_byte_count_local(stride, in_elem_bytes, out_elem_bytes, &scaled_stride)) {
      if (err) {
        *err = "cast contract transform encountered non-aligned stride bytes";
      }
      return false;
    }
    stride = scaled_stride;
  }
  out->quant.reset();
  return true;
}

std::uint64_t logical_tensor_physical_span_bytes_local(
    const pipeline_internal::sima::LogicalTensorStaticSpec& tensor) {
  const std::size_t elem_bytes = runtime_dtype_bytes_local(tensor.dtype);
  if (tensor.shape.empty() || tensor.stride_bytes.empty() || tensor.shape.size() != tensor.stride_bytes.size() ||
      elem_bytes == 0U) {
    return tensor.size_bytes;
  }
  std::uint64_t max_offset = 0U;
  for (std::size_t i = 0; i < tensor.shape.size(); ++i) {
    if (tensor.shape[i] <= 0 || tensor.stride_bytes[i] < 0) {
      return tensor.size_bytes;
    }
    if (tensor.shape[i] == 1) {
      continue;
    }
    const auto dim = static_cast<std::uint64_t>(tensor.shape[i] - 1);
    const auto stride = static_cast<std::uint64_t>(tensor.stride_bytes[i]);
    const std::uint64_t delta = dim * stride;
    if (dim > 0U && delta / dim != stride) {
      return tensor.size_bytes;
    }
    if (max_offset > (std::numeric_limits<std::uint64_t>::max() - delta)) {
      return tensor.size_bytes;
    }
    max_offset += delta;
  }
  if (max_offset > (std::numeric_limits<std::uint64_t>::max() - elem_bytes)) {
    return tensor.size_bytes;
  }
  return std::max<std::uint64_t>(tensor.size_bytes, max_offset + elem_bytes);
}

bool can_pack_cast_fanout_to_single_parent_local(const CompiledRuntimeContract& upstream,
                                                 std::string* segment_name_out) {
  if (upstream.logical_outputs.empty()) {
    return false;
  }
  const auto& first = upstream.logical_outputs.front();
  const int physical_index = first.physical_index;
  const std::string segment_name = first.segment_name;
  for (const auto& logical : upstream.logical_outputs) {
    if (logical.physical_index != physical_index) {
      return false;
    }
    if (!segment_name.empty() && !logical.segment_name.empty() && logical.segment_name != segment_name) {
      return false;
    }
  }
  if (segment_name_out) {
    *segment_name_out = segment_name;
  }
  return true;
}

std::uint64_t tensor_logical_size_bytes_local(const Tensor& tensor);
std::string resolved_tensor_segment_name_local(const Tensor& tensor, std::size_t index);

bool can_pack_cast_fanout_from_sample_local(const TensorList& tensors,
                                            int* physical_index_out,
                                            std::string* segment_name_out) {
  if (tensors.empty()) {
    return false;
  }

  const auto& first = tensors.front();
  if (!first.storage) {
    return false;
  }

  const std::string segment_name = resolved_tensor_segment_name_local(first, 0U);
  if (segment_name.empty()) {
    return false;
  }

  std::size_t running_offset = 0U;
  for (std::size_t i = 0; i < tensors.size(); ++i) {
    const auto& tensor = tensors[i];
    if (!tensor.storage || tensor.storage != first.storage) {
      return false;
    }
    if (resolved_tensor_segment_name_local(tensor, i) != segment_name) {
      return false;
    }

    const std::uint64_t logical_size = tensor_logical_size_bytes_local(tensor);
    const std::int64_t byte_offset =
        tensor.route.physical_byte_offset >= 0 ? tensor.route.physical_byte_offset : tensor.byte_offset;
    if (logical_size == 0U || byte_offset < 0) {
      return false;
    }
    if (static_cast<std::size_t>(byte_offset) != running_offset) {
      return false;
    }
    running_offset += static_cast<std::size_t>(logical_size);
  }

  if (running_offset == 0U) {
    return false;
  }

  if (physical_index_out) {
    *physical_index_out = 0;
  }
  if (segment_name_out) {
    *segment_name_out = segment_name;
  }
  return true;
}

std::string runtime_dtype_token_from_tensor_local(TensorDType dtype) {
  switch (dtype) {
  case TensorDType::UInt8:
    return "UINT8";
  case TensorDType::Int8:
    return "INT8";
  case TensorDType::UInt16:
    return "UINT16";
  case TensorDType::Int16:
    return "INT16";
  case TensorDType::Int32:
    return "INT32";
  case TensorDType::BFloat16:
    return "BF16";
  case TensorDType::Float32:
    return "FP32";
  case TensorDType::Float64:
    return "FP64";
  }
  return {};
}

std::string runtime_layout_token_from_tensor_local(const Tensor& tensor) {
  if (!tensor.axis_semantics.empty()) {
    const auto is_exact = [&](std::initializer_list<TensorAxisSemantic> expected) {
      return tensor.axis_semantics.size() == expected.size() &&
             std::equal(tensor.axis_semantics.begin(), tensor.axis_semantics.end(), expected.begin(),
                        expected.end());
    };
    if (is_exact({TensorAxisSemantic::H, TensorAxisSemantic::W}) ||
        is_exact({TensorAxisSemantic::N, TensorAxisSemantic::H, TensorAxisSemantic::W})) {
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
  }
  switch (tensor.layout) {
  case TensorLayout::CHW:
    return "CHW";
  case TensorLayout::HW:
    return "HW";
  case TensorLayout::HWC:
    return "HWC";
  case TensorLayout::Unknown:
  default:
    return {};
  }
}

std::uint64_t tensor_logical_size_bytes_local(const Tensor& tensor) {
  const std::size_t elem_bytes = pipeline_internal::dtype_bytes(tensor.dtype);
  if (elem_bytes == 0U) {
    return 0U;
  }
  std::uint64_t total = static_cast<std::uint64_t>(elem_bytes);
  for (const auto dim : tensor.shape) {
    if (dim <= 0) {
      return 0U;
    }
    total *= static_cast<std::uint64_t>(dim);
  }
  return total;
}

int resolved_tensor_physical_index_local(const Tensor& tensor, std::size_t index) {
  if (tensor.route.physical_index >= 0) {
    return tensor.route.physical_index;
  }
  if (tensor.route.memory_index >= 0) {
    return tensor.route.memory_index;
  }
  return static_cast<int>(index);
}

std::string resolved_tensor_segment_name_local(const Tensor& tensor, std::size_t index) {
  if (tensor.storage && !tensor.storage->sima_segments.empty()) {
    std::size_t segment_index = 0U;
    if (tensor.route.memory_index >= 0 &&
        static_cast<std::size_t>(tensor.route.memory_index) < tensor.storage->sima_segments.size()) {
      segment_index = static_cast<std::size_t>(tensor.route.memory_index);
    } else if (tensor.route.physical_index >= 0 &&
               static_cast<std::size_t>(tensor.route.physical_index) <
                   tensor.storage->sima_segments.size()) {
      segment_index = static_cast<std::size_t>(tensor.route.physical_index);
    }
    const auto& segment = tensor.storage->sima_segments[segment_index];
    if (!segment.name.empty()) {
      return segment.name;
    }
  }
  if (!tensor.route.segment_name.empty()) {
    return tensor.route.segment_name;
  }
  return tensor_name_from_sample_local(tensor, index);
}

std::uint64_t resolved_tensor_segment_size_bytes_local(
    const Tensor& tensor,
    int physical_index,
    std::uint64_t logical_size_bytes) {
  if (tensor.storage && !tensor.storage->sima_segments.empty()) {
    std::size_t segment_index = 0U;
    if (tensor.route.memory_index >= 0 &&
        static_cast<std::size_t>(tensor.route.memory_index) < tensor.storage->sima_segments.size()) {
      segment_index = static_cast<std::size_t>(tensor.route.memory_index);
    } else if (physical_index >= 0 &&
               static_cast<std::size_t>(physical_index) < tensor.storage->sima_segments.size()) {
      segment_index = static_cast<std::size_t>(physical_index);
    }
    return static_cast<std::uint64_t>(tensor.storage->sima_segments[segment_index].size_bytes);
  }
  if (tensor.storage && tensor.storage->size_bytes > 0U) {
    return static_cast<std::uint64_t>(tensor.storage->size_bytes);
  }
  return logical_size_bytes;
}

CompiledRuntimeContract build_cast_runtime_contract_from_upstream_local(
    const CompiledRuntimeContract& upstream,
    CastDirection direction,
    std::string* err) {
  CompiledRuntimeContract runtime;
  runtime.plugin_kind = "cast";
  runtime.logical_inputs.reserve(upstream.logical_outputs.size());
  runtime.logical_outputs.reserve(upstream.logical_outputs.size());
  runtime.output_order = upstream.output_order;

  for (std::size_t i = 0; i < upstream.logical_outputs.size(); ++i) {
    const auto& logical = upstream.logical_outputs[i];
    pipeline_internal::sima::LogicalInputStaticSpec input_logical;
    input_logical.logical_index = logical.logical_index;
    input_logical.backend_input_index = logical.backend_output_index;
    input_logical.physical_index = logical.physical_index;
    input_logical.shape = logical.shape;
    input_logical.stride_bytes = logical.stride_bytes;
    input_logical.byte_offset = logical.byte_offset;
    input_logical.size_bytes = logical.size_bytes;
    input_logical.dtype = logical.dtype;
    input_logical.layout = logical.layout;
    input_logical.logical_name = logical.logical_name;
    input_logical.backend_name = logical.backend_name;
    input_logical.segment_name = logical.segment_name;
    input_logical.quant = logical.quant;
    runtime.logical_inputs.push_back(std::move(input_logical));

    pipeline_internal::sima::LogicalTensorStaticSpec transformed;
    if (!transform_logical_output_for_cast(logical, direction, &transformed, err)) {
      return {};
    }
    runtime.logical_outputs.push_back(std::move(transformed));
  }

  std::string packed_segment_name;
  const bool packed_single_parent =
      can_pack_cast_fanout_to_single_parent_local(upstream, &packed_segment_name);
  if (packed_single_parent) {
    const auto source_physical =
        !upstream.physical_outputs.empty() ? upstream.physical_outputs.front()
                                           : pipeline_internal::sima::PhysicalBufferStaticSpec{};
    std::uint64_t packed_input_size_bytes = 0U;
    std::uint64_t packed_output_size_bytes = 0U;
    for (auto& logical : runtime.logical_inputs) {
      logical.physical_index = 0;
      if (!packed_segment_name.empty()) {
        logical.segment_name = packed_segment_name;
      }
      packed_input_size_bytes =
          std::max<std::uint64_t>(packed_input_size_bytes,
                                  static_cast<std::uint64_t>(std::max<std::int64_t>(logical.byte_offset, 0)) +
                                      logical_tensor_physical_span_bytes_local(
                                          pipeline_internal::sima::LogicalTensorStaticSpec{
                                              .logical_index = logical.logical_index,
                                              .backend_output_index = logical.backend_input_index,
                                              .physical_index = logical.physical_index,
                                              .byte_offset = logical.byte_offset,
                                              .size_bytes = logical.size_bytes,
                                              .shape = logical.shape,
                                              .stride_bytes = logical.stride_bytes,
                                              .dtype = logical.dtype,
                                              .layout = logical.layout,
                                              .logical_name = logical.logical_name,
                                              .backend_name = logical.backend_name,
                                              .segment_name = logical.segment_name,
                                              .quant = logical.quant}));
    }
    for (auto& logical : runtime.logical_outputs) {
      logical.physical_index = 0;
      if (!packed_segment_name.empty()) {
        logical.segment_name = packed_segment_name;
      }
      packed_output_size_bytes =
          std::max<std::uint64_t>(packed_output_size_bytes,
                                  static_cast<std::uint64_t>(std::max<std::int64_t>(logical.byte_offset, 0)) +
                                      logical_tensor_physical_span_bytes_local(logical));
    }
    for (auto& route : runtime.output_order) {
      if (!packed_segment_name.empty()) {
        route.segment_name = packed_segment_name;
      }
    }
    runtime.physical_inputs = {pipeline_internal::sima::specbuilders::build_physical_buffer_static_spec(
        0, 0, packed_input_size_bytes, source_physical.device_kind,
        packed_segment_name.empty() ? source_physical.segment_name : packed_segment_name,
        source_physical.source_physical_index, source_physical.source_byte_offset)};
    runtime.physical_outputs = {pipeline_internal::sima::specbuilders::build_physical_buffer_static_spec(
        0, 0, packed_output_size_bytes, source_physical.device_kind,
        packed_segment_name.empty() ? source_physical.segment_name : packed_segment_name,
        source_physical.source_physical_index, source_physical.source_byte_offset)};
  } else {
    runtime.physical_inputs = upstream.physical_outputs;
    runtime.physical_outputs = upstream.physical_outputs;
  }
  for (auto& physical : runtime.physical_outputs) {
    const std::size_t in_elem_bytes =
        runtime_dtype_bytes_local(!upstream.logical_outputs.empty() ? upstream.logical_outputs.front().dtype
                                                                    : std::string());
    const std::size_t out_elem_bytes = runtime_dtype_bytes_local(cast_output_dtype_local(direction));
    if (in_elem_bytes == 0U || out_elem_bytes == 0U) {
      continue;
    }
    std::int64_t scaled_offset = 0;
    std::uint64_t scaled_size = 0U;
    if (!scale_byte_count_local(physical.source_byte_offset, in_elem_bytes, out_elem_bytes,
                                &scaled_offset) ||
        (!packed_single_parent &&
         !scale_size_bytes_local(physical.size_bytes, in_elem_bytes, out_elem_bytes,
                                 &scaled_size))) {
      if (err) {
        *err = "cast contract transform encountered non-aligned physical byte facts";
      }
      return {};
    }
    physical.source_byte_offset = scaled_offset;
    if (!packed_single_parent) {
      physical.size_bytes = scaled_size;
    }
  }
  runtime.output_quant.clear();
  return runtime;
}

CompiledRuntimeContract build_cast_runtime_contract_from_sample_local(
    const Sample& sample,
    CastDirection direction,
    std::string* err) {
  struct PhysicalSeed {
    int physical_index = -1;
    std::string segment_name;
    std::uint64_t input_size_bytes = 0U;
    std::uint64_t output_size_bytes = 0U;
  };

  CompiledRuntimeContract runtime;
  runtime.plugin_kind = "cast";
  TensorList tensors = tensors_from_sample(sample, true);
  std::stable_sort(tensors.begin(), tensors.end(), [](const Tensor& lhs, const Tensor& rhs) {
    return lhs.route.logical_index < rhs.route.logical_index;
  });

  std::vector<PhysicalSeed> physical_seeds;
  auto find_or_create_physical_seed = [&](int physical_index,
                                          const std::string& segment_name) -> PhysicalSeed& {
    auto it = std::find_if(physical_seeds.begin(), physical_seeds.end(),
                           [&](const PhysicalSeed& seed) {
                             return seed.physical_index == physical_index;
                           });
    if (it != physical_seeds.end()) {
      if (it->segment_name.empty()) {
        it->segment_name = segment_name;
      }
      return *it;
    }
    physical_seeds.push_back(PhysicalSeed{physical_index, segment_name, 0U, 0U});
    return physical_seeds.back();
  };

  runtime.logical_inputs.reserve(tensors.size());
  runtime.logical_outputs.reserve(tensors.size());
  runtime.output_order.reserve(tensors.size());
  int packed_physical_index = 0;
  std::string packed_segment_name;
  const bool packed_single_parent = can_pack_cast_fanout_from_sample_local(
      tensors, &packed_physical_index, &packed_segment_name);
  for (std::size_t i = 0; i < tensors.size(); ++i) {
    const auto& tensor = tensors[i];
    const std::size_t in_elem_bytes = pipeline_internal::dtype_bytes(tensor.dtype);
    const std::size_t out_elem_bytes = runtime_dtype_bytes_local(cast_output_dtype_local(direction));
    if (in_elem_bytes == 0U || out_elem_bytes == 0U) {
      if (err) {
        *err = "cast contract compile requires typed ingress tensor element size";
      }
      return {};
    }

    std::uint64_t logical_size = 1U;
    for (const auto dim : tensor.shape) {
      if (dim <= 0) {
        if (err) {
          *err = "cast contract compile requires positive ingress tensor shape";
        }
        return {};
      }
      logical_size *= static_cast<std::uint64_t>(dim);
    }
    const std::uint64_t input_logical_size =
        tensor_logical_size_bytes_local(tensor) != 0U
            ? tensor_logical_size_bytes_local(tensor)
            : (logical_size * static_cast<std::uint64_t>(in_elem_bytes));
    logical_size *= static_cast<std::uint64_t>(out_elem_bytes);

    const std::int64_t input_byte_offset =
        tensor.route.physical_byte_offset >= 0 ? tensor.route.physical_byte_offset : tensor.byte_offset;
    std::int64_t scaled_offset = 0;
    if (!scale_byte_count_local(input_byte_offset, in_elem_bytes, out_elem_bytes, &scaled_offset)) {
      if (err) {
        *err = "cast contract compile encountered non-aligned ingress tensor byte_offset";
      }
      return {};
    }
    const int physical_index =
        packed_single_parent ? packed_physical_index : resolved_tensor_physical_index_local(tensor, i);
    const std::string logical_name = tensor_name_from_sample_local(tensor, i);
    const std::string backend_name =
        !tensor.route.backend_name.empty() ? tensor.route.backend_name : logical_name;
    const std::string segment_name =
        packed_single_parent ? packed_segment_name : resolved_tensor_segment_name_local(tensor, i);
    const std::string layout = runtime_layout_token_from_tensor_local(tensor);
    if (layout.empty()) {
      if (err) {
        *err = "cast contract compile missing explicit tensor layout semantics";
      }
      return {};
    }
    const std::string input_dtype = runtime_dtype_token_from_tensor_local(tensor.dtype);

    pipeline_internal::sima::LogicalInputStaticSpec input_logical;
    input_logical.logical_index =
        tensor.route.logical_index >= 0 ? tensor.route.logical_index : static_cast<int>(i);
    input_logical.backend_input_index =
        tensor.route.backend_output_index >= 0 ? tensor.route.backend_output_index
                                               : static_cast<int>(i);
    input_logical.physical_index = physical_index;
    input_logical.shape = tensor.shape;
    input_logical.stride_bytes = tensor.strides_bytes;
    input_logical.byte_offset = input_byte_offset;
    input_logical.size_bytes = input_logical_size;
    input_logical.dtype = input_dtype;
    input_logical.layout = layout;
    input_logical.logical_name = logical_name;
    input_logical.backend_name = backend_name;
    input_logical.segment_name = segment_name;
    runtime.logical_inputs.push_back(std::move(input_logical));

    pipeline_internal::sima::LogicalTensorStaticSpec logical;
    logical.logical_index = runtime.logical_inputs.back().logical_index;
    logical.backend_output_index =
        tensor.route.backend_output_index >= 0 ? tensor.route.backend_output_index
                                               : static_cast<int>(i);
    logical.physical_index = physical_index;
    logical.output_slot =
        tensor.route.route_slot >= 0 ? tensor.route.route_slot : static_cast<int>(i);
    logical.tensor_index = static_cast<int>(i);
    logical.byte_offset = scaled_offset;
    logical.size_bytes = logical_size;
    logical.shape = tensor.shape;
    logical.stride_bytes = tensor.strides_bytes;
    for (auto& stride : logical.stride_bytes) {
      if (stride == 0) {
        continue;
      }
      std::int64_t scaled_stride = 0;
      if (!scale_byte_count_local(stride, in_elem_bytes, out_elem_bytes, &scaled_stride)) {
        if (err) {
          *err = "cast contract compile encountered non-aligned ingress tensor stride bytes";
        }
        return {};
      }
      stride = scaled_stride;
    }
    logical.dtype = cast_output_dtype_local(direction);
    logical.layout = layout;
    logical.logical_name = logical_name;
    logical.backend_name = backend_name;
    logical.segment_name = segment_name;
    runtime.logical_outputs.push_back(logical);
    runtime.output_order.push_back(pipeline_internal::sima::StageOutputRoute{
        logical.output_slot,
        logical.logical_index,
        logical.tensor_index,
        logical.backend_name,
        logical.segment_name,
    });

    auto& physical = find_or_create_physical_seed(physical_index, segment_name);
    const std::uint64_t input_segment_size =
        resolved_tensor_segment_size_bytes_local(tensor, physical_index, input_logical_size);
    physical.input_size_bytes =
        std::max(physical.input_size_bytes,
                 std::max(input_segment_size,
                          static_cast<std::uint64_t>(input_byte_offset) + input_logical_size));
    physical.output_size_bytes =
        std::max(physical.output_size_bytes,
                 static_cast<std::uint64_t>(scaled_offset) + logical.size_bytes);
  }

  runtime.physical_inputs.reserve(physical_seeds.size());
  runtime.physical_outputs.reserve(physical_seeds.size());
  for (std::size_t i = 0; i < physical_seeds.size(); ++i) {
    const auto& physical = physical_seeds[i];
    const int physical_index =
        physical.physical_index >= 0 ? physical.physical_index : static_cast<int>(i);
    runtime.physical_inputs.push_back(
        pipeline_internal::sima::specbuilders::build_physical_buffer_static_spec(
            physical_index, physical_index, physical.input_size_bytes,
            pipeline_internal::sima::DeviceKind::Cpu, physical.segment_name, physical_index, 0));
    runtime.physical_outputs.push_back(
        pipeline_internal::sima::specbuilders::build_physical_buffer_static_spec(
            physical_index, physical_index, physical.output_size_bytes,
            pipeline_internal::sima::DeviceKind::Cpu, physical.segment_name, physical_index, 0));
  }

  runtime.output_quant.clear();
  return runtime;
}

} // namespace

Cast::Cast(CastOptions opt) : opt_(std::move(opt)) {}

std::string Cast::backend_fragment(int node_index) const {
  require_element("neatprocesscvu", "Cast::backend_fragment");
  const std::string name =
      opt_.element_name.empty() ? ("n" + std::to_string(node_index) + "_cast") : opt_.element_name;
  const std::string stage_id = default_logical_stage_id_local(*this, name);
  std::ostringstream ss;
  ss << "neatprocesscvu name=" << name << " stage-id=" << stage_id;
  if (opt_.num_buffers > 0) {
    ss << " num-buffers=" << opt_.num_buffers;
  }
  return ss.str();
}

std::vector<std::string> Cast::element_names(int node_index) const {
  if (!opt_.element_name.empty()) {
    return {opt_.element_name};
  }
  return {"n" + std::to_string(node_index) + "_cast"};
}

OutputSpec Cast::output_spec(const OutputSpec& input) const {
  OutputSpec out = input;
  const std::string format = upper_copy_local(out.format);
  const std::string dtype = upper_copy_local(out.dtype);

  if (opt_.direction == CastDirection::Bf16ToFp32) {
    if (dtype.empty() || dtype.find("BF16") != std::string::npos ||
        dtype.find("BFLOAT16") != std::string::npos) {
      out.dtype = "Float32";
    }
    if (format.empty() || format.find("BF16") != std::string::npos ||
        format.find("BFLOAT16") != std::string::npos) {
      out.format = "FP32";
    }
    if (out.byte_size > 0U) {
      out.byte_size *= 2U;
    }
  } else {
    if (dtype.empty() || dtype.find("FP32") != std::string::npos ||
        dtype.find("FLOAT32") != std::string::npos) {
      out.dtype = "BFloat16";
    }
    if (format.empty() || format.find("FP32") != std::string::npos ||
        format.find("FLOAT32") != std::string::npos) {
      out.format = "BF16";
    }
    if (out.byte_size > 1U) {
      out.byte_size /= 2U;
    }
  }
  out.certainty = SpecCertainty::Hint;
  out.note = "cast";
  return out;
}

NodeContractDefinition Cast::contract_definition() const {
  NodeContractDefinition def;
  def.node_kind = kind();
  def.plugin_kind = "processcvu";

  ContractPortSpec input;
  input.port_id = "input_tensor";
  input.media_type = "application/vnd.simaai.tensor";
  input.required_segment_names = {"input_tensor"};
  def.inputs.push_back(std::move(input));

  ContractPortSpec output;
  output.port_id = "output_tensor";
  output.media_type = "application/vnd.simaai.tensor";
  output.required_segment_names = {"output_tensor"};
  def.outputs.push_back(std::move(output));
  return def;
}

bool Cast::compile_node_contract(const ContractCompileInput& input,
                                 CompiledNodeContract* out,
                                 std::string* err) const {
  const std::string element_name = element_names(input.node_index).empty()
                                       ? std::string("cast")
                                       : element_names(input.node_index).front();
  const std::string logical_stage_id = default_logical_stage_id_local(*this, element_name);
  if (!opt_.compiled_contract) {
    CompiledRuntimeContract runtime;
    if (const auto* upstream_runtime = compiled_runtime_contract_from_stage(input.immediate_upstream);
        upstream_runtime && !upstream_runtime->logical_outputs.empty()) {
      runtime = build_cast_runtime_contract_from_upstream_local(*upstream_runtime, opt_.direction,
                                                                err);
    } else if (input.ingress.ingress_sample.has_value()) {
      runtime = build_cast_runtime_contract_from_sample_local(*input.ingress.ingress_sample,
                                                              opt_.direction, err);
    } else {
      if (err) {
        *err = "Cast: compiled processcvu contract or typed upstream/ingress sample is required";
      }
      return false;
    }
    if (runtime.logical_outputs.empty()) {
      if (err && err->empty()) {
        *err = "Cast: failed to derive transport runtime contract";
      }
      return false;
    }
    pipeline_internal::sima::stagesemantics::TransportCanonicalFacts facts;
    facts.plugin_kind = "cast";
    facts.kernel_kind = "cast";
    facts.runtime_contract = std::move(runtime);
    const auto compiled =
        pipeline_internal::sima::stagesemantics::build_transport_compiled_contract_from_facts(
            facts);
    return pipeline_internal::sima::stagesemantics::build_transport_node_contract(
        kind(), element_name, logical_stage_id, contract_definition(), compiled, out, err);
  }
  return pipeline_internal::sima::stagesemantics::build_processcvu_node_contract(
      kind(), element_name, logical_stage_id, contract_definition(), *opt_.compiled_contract,
      out, err);
}

void Cast::apply_compiled_contract(const CompiledNodeContract&, std::string* err) {
  if (err) {
    err->clear();
  }
}

} // namespace simaai::neat

namespace simaai::neat::nodes {

std::shared_ptr<simaai::neat::Node> Cast(CastOptions opt) {
  return std::make_shared<simaai::neat::Cast>(std::move(opt));
}

} // namespace simaai::neat::nodes
