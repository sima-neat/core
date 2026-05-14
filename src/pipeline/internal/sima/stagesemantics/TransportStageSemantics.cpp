#include "pipeline/internal/sima/stagesemantics/TransportStageSemantics.h"

#include "pipeline/internal/TensorMath.h"
#include "pipeline/internal/sima/StaticSpecBuilders.h"

#include <algorithm>
#include <cctype>
#include <cstdint>

namespace simaai::neat::pipeline_internal::sima::stagesemantics {
namespace {

std::string upper_copy_local(std::string value) {
  std::transform(value.begin(), value.end(), value.begin(),
                 [](unsigned char c) { return static_cast<char>(std::toupper(c)); });
  return value;
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

std::string cast_output_dtype_local(TransportCastDirection direction) {
  return (direction == TransportCastDirection::Fp32ToBf16) ? std::string("BF16")
                                                           : std::string("FP32");
}

bool scale_byte_count_local(std::int64_t in_value, std::size_t in_elem_bytes,
                            std::size_t out_elem_bytes, std::int64_t* out_value) {
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

bool scale_size_bytes_local(std::uint64_t in_value, std::size_t in_elem_bytes,
                            std::size_t out_elem_bytes, std::uint64_t* out_value) {
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

std::uint64_t logical_tensor_physical_span_bytes_local(
    const pipeline_internal::sima::LogicalTensorStaticSpec& tensor) {
  const std::size_t elem_bytes = runtime_dtype_bytes_local(tensor.dtype);
  if (tensor.shape.empty() || tensor.stride_bytes.empty() ||
      tensor.shape.size() != tensor.stride_bytes.size() || elem_bytes == 0U) {
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
    if (!segment_name.empty() && !logical.segment_name.empty() &&
        logical.segment_name != segment_name) {
      return false;
    }
  }
  if (segment_name_out) {
    *segment_name_out = segment_name;
  }
  return true;
}

bool transform_logical_output_for_cast(const pipeline_internal::sima::LogicalTensorStaticSpec& src,
                                       TransportCastDirection direction,
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

} // namespace

CompiledTransportContract
build_transport_compiled_contract_from_facts(const TransportCanonicalFacts& facts) {
  CompiledTransportContract transport;
  transport.plugin_kind = facts.plugin_kind;
  transport.kernel_kind = facts.kernel_kind;
  if (facts.runtime_contract.has_value()) {
    transport.runtime_contract = *facts.runtime_contract;
  }
  transport.runtime_contract.plugin_kind = transport.runtime_contract.plugin_kind.empty()
                                               ? facts.plugin_kind
                                               : transport.runtime_contract.plugin_kind;
  transport.payload_kind = facts.payload_kind;
  transport.processcvu_payload = facts.processcvu_payload;
  transport.model_managed_stage = facts.model_managed_stage;
  return transport;
}

CompiledRuntimeContract build_transport_runtime_contract_from_processcvu_compiled(
    const CompiledProcessCvuContract& compiled) {
  CompiledRuntimeContract runtime = compiled.runtime_contract;
  if (runtime.plugin_kind.empty()) {
    runtime.plugin_kind = compiled.payload.graph_family.empty() ? compiled.payload.graph_name
                                                                : compiled.payload.graph_family;
  }
  return runtime;
}

std::optional<TransportCastDirection>
infer_cast_direction_from_upstream_contract(const CompiledRuntimeContract& upstream) {
  if (upstream.logical_outputs.empty()) {
    return std::nullopt;
  }
  const std::string dtype = normalize_runtime_dtype_local(upstream.logical_outputs.front().dtype);
  if (dtype == "BF16") {
    return TransportCastDirection::Bf16ToFp32;
  }
  if (dtype == "FP32") {
    return TransportCastDirection::Fp32ToBf16;
  }
  return std::nullopt;
}

CompiledRuntimeContract build_cast_runtime_contract_from_external_input(
    int width, int height, int depth, const std::string& layout, TransportCastDirection direction,
    std::string* err) {
  CompiledRuntimeContract runtime;
  runtime.plugin_kind = "cast";

  if (width <= 0 || height <= 0 || depth <= 0) {
    if (err) {
      *err = "cast external-input contract requires positive width/height/depth";
    }
    return {};
  }

  const std::string input_dtype =
      (direction == TransportCastDirection::Fp32ToBf16) ? std::string("FP32") : std::string("BF16");
  const std::string output_dtype = cast_output_dtype_local(direction);
  const std::string resolved_layout = layout.empty() ? std::string("HWC") : layout;
  const std::vector<std::int64_t> shape = {static_cast<std::int64_t>(height),
                                           static_cast<std::int64_t>(width),
                                           static_cast<std::int64_t>(depth)};
  const std::uint64_t input_size_bytes =
      specbuilders::tensor_size_bytes_from_shape_dtype(shape, input_dtype);
  const std::uint64_t output_size_bytes =
      specbuilders::tensor_size_bytes_from_shape_dtype(shape, output_dtype);
  if (input_size_bytes == 0U || output_size_bytes == 0U) {
    if (err) {
      *err = "cast external-input contract failed to derive tensor byte size";
    }
    return {};
  }

  runtime.logical_inputs.push_back(specbuilders::build_logical_input_static_spec(
      0, 0, 0, shape, input_dtype, resolved_layout, "input_tensor", "input_tensor", "parent"));
  runtime.physical_inputs.push_back(specbuilders::build_physical_buffer_static_spec(
      0, 0, input_size_bytes, pipeline_internal::sima::DeviceKind::Cpu, "parent"));
  runtime.physical_outputs.push_back(specbuilders::build_physical_buffer_static_spec(
      0, 0, output_size_bytes, pipeline_internal::sima::DeviceKind::Cpu, "parent"));
  runtime.logical_outputs.push_back(specbuilders::build_logical_output_static_spec(
      0, 0, 0, 0, 0, shape, output_dtype, resolved_layout, "output_tensor", "output_tensor",
      "parent"));
  runtime.output_order.push_back(
      specbuilders::build_output_route_static_spec(0, 0, 0, "output_tensor", "parent"));
  runtime.output_quant.clear();
  if (err) {
    err->clear();
  }
  return runtime;
}

CompiledRuntimeContract
build_cast_runtime_contract_from_upstream(const CompiledRuntimeContract& upstream,
                                          TransportCastDirection direction, std::string* err) {
  CompiledRuntimeContract runtime;
  runtime.plugin_kind = "cast";
  runtime.logical_inputs.reserve(upstream.logical_outputs.size());
  runtime.logical_outputs.reserve(upstream.logical_outputs.size());
  runtime.output_order = upstream.output_order;

  for (const auto& logical : upstream.logical_outputs) {
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
  if (can_pack_cast_fanout_to_single_parent_local(upstream, &packed_segment_name)) {
    const auto source_physical = !upstream.physical_outputs.empty()
                                     ? upstream.physical_outputs.front()
                                     : pipeline_internal::sima::PhysicalBufferStaticSpec{};
    std::uint64_t packed_input_size_bytes = 0U;
    std::uint64_t packed_output_size_bytes = 0U;
    for (auto& logical : runtime.logical_inputs) {
      logical.physical_index = 0;
      if (!packed_segment_name.empty()) {
        logical.segment_name = packed_segment_name;
      }
      packed_input_size_bytes = std::max<std::uint64_t>(
          packed_input_size_bytes,
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
      packed_output_size_bytes = std::max<std::uint64_t>(
          packed_output_size_bytes,
          static_cast<std::uint64_t>(std::max<std::int64_t>(logical.byte_offset, 0)) +
              logical_tensor_physical_span_bytes_local(logical));
    }
    for (auto& route : runtime.output_order) {
      if (!packed_segment_name.empty()) {
        route.segment_name = packed_segment_name;
      }
    }
    runtime.physical_inputs = {specbuilders::build_physical_buffer_static_spec(
        0, 0, packed_input_size_bytes, source_physical.device_kind,
        packed_segment_name.empty() ? source_physical.segment_name : packed_segment_name,
        source_physical.source_physical_index, source_physical.source_byte_offset)};
    runtime.physical_outputs = {specbuilders::build_physical_buffer_static_spec(
        0, 0, packed_output_size_bytes, source_physical.device_kind,
        packed_segment_name.empty() ? source_physical.segment_name : packed_segment_name,
        source_physical.source_physical_index, source_physical.source_byte_offset)};
  } else {
    runtime.physical_inputs = upstream.physical_outputs;
    runtime.physical_outputs = upstream.physical_outputs;
  }
  for (auto& physical : runtime.physical_outputs) {
    const std::size_t in_elem_bytes = runtime_dtype_bytes_local(
        !upstream.logical_outputs.empty() ? upstream.logical_outputs.front().dtype : std::string());
    const std::size_t out_elem_bytes =
        runtime_dtype_bytes_local(cast_output_dtype_local(direction));
    if (in_elem_bytes == 0U || out_elem_bytes == 0U) {
      continue;
    }
    std::int64_t scaled_offset = 0;
    std::uint64_t scaled_size = 0U;
    if (!scale_byte_count_local(physical.source_byte_offset, in_elem_bytes, out_elem_bytes,
                                &scaled_offset) ||
        !scale_size_bytes_local(physical.size_bytes, in_elem_bytes, out_elem_bytes, &scaled_size)) {
      if (err) {
        *err = "cast contract transform encountered non-aligned physical byte facts";
      }
      return {};
    }
    physical.source_byte_offset = scaled_offset;
    physical.size_bytes = scaled_size;
  }
  runtime.output_quant.clear();
  return runtime;
}

bool build_transport_node_contract(const std::string& node_kind, const std::string& element_name,
                                   const std::string& logical_stage_id,
                                   const NodeContractDefinition& definition,
                                   const CompiledTransportContract& compiled,
                                   CompiledNodeContract* out, std::string* error_message) {
  if (!out) {
    if (error_message) {
      *error_message = node_kind + " contract compile: output is null";
    }
    return false;
  }
  auto stored = compiled;
  out->node_kind = node_kind;
  out->plugin_kind =
      !stored.plugin_kind.empty()
          ? stored.plugin_kind
          : (!definition.plugin_kind.empty() ? definition.plugin_kind : std::string("transport"));
  out->element_name = element_name;
  out->logical_stage_id = logical_stage_id.empty() ? element_name : logical_stage_id;
  out->definition = definition;
  stored.runtime_contract.plugin_kind = out->plugin_kind;
  out->transport = std::move(stored);
  out->renderable = true;
  if (error_message) {
    error_message->clear();
  }
  return true;
}

} // namespace simaai::neat::pipeline_internal::sima::stagesemantics
