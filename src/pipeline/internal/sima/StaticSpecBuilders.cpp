#include "pipeline/internal/sima/StaticSpecBuilders.h"

#include "pipeline/internal/TensorMath.h"
#include "pipeline/internal/EnvUtil.h"

#include <algorithm>
#include <cinttypes>
#include <cctype>
#include <cstring>
#include <cstdio>
#include <cstdlib>

namespace simaai::neat::pipeline_internal::sima::specbuilders {
namespace {

std::string upper_copy_local(std::string value) {
  std::transform(value.begin(), value.end(), value.begin(),
                 [](unsigned char c) { return static_cast<char>(std::toupper(c)); });
  return value;
}

bool detess_layout_debug_enabled_local() {
  return pipeline_internal::env_bool("SIMA_DETESS_LAYOUT_DEBUG", false);
}

std::string join_shape_debug_local(const std::vector<std::int64_t>& shape) {
  if (shape.empty()) {
    return "<none>";
  }
  std::string out = "[";
  for (std::size_t i = 0; i < shape.size(); ++i) {
    if (i > 0U) {
      out += "x";
    }
    out += std::to_string(shape[i]);
  }
  out += "]";
  return out;
}

} // namespace

std::uint64_t dtype_size_bytes_from_token(const std::string& raw_dtype) {
  const std::string token = upper_copy_local(raw_dtype);
  if (token.find("BF16") != std::string::npos || token.find("FLOAT16") != std::string::npos ||
      token.find("FP16") != std::string::npos || token == "INT16" || token == "EVXX_INT16" ||
      token == "UINT16") {
    return 2U;
  }
  if (token.find("INT32") != std::string::npos || token.find("FLOAT32") != std::string::npos ||
      token.find("FP32") != std::string::npos) {
    return 4U;
  }
  return 1U;
}

std::uint64_t tensor_size_bytes_from_shape_dtype(const std::vector<std::int64_t>& shape,
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
  return elems * dtype_size_bytes_from_token(dtype);
}

std::vector<std::int64_t> dense_shape_from_dims(int width, int height, int depth,
                                                const std::string& layout) {
  const std::string normalized = upper_copy_local(layout);
  if (width <= 0 || height <= 0 || depth <= 0) {
    return {};
  }
  if (normalized == "CHW" || normalized == "NCHW") {
    return {depth, height, width};
  }
  if (normalized == "HW") {
    return {height, width};
  }
  return {height, width, depth};
}

LogicalInputStaticSpec build_logical_input_static_spec(
    int logical_index, int backend_input_index, int physical_index,
    const std::vector<std::int64_t>& shape, const std::string& dtype, const std::string& layout,
    const std::string& logical_name, const std::string& backend_name,
    const std::string& segment_name, std::int64_t byte_offset, std::uint64_t size_bytes_override,
    TensorMaterializationKind materialization_kind, const std::optional<QuantStaticSpec>& quant) {
  LogicalInputStaticSpec logical;
  logical.logical_index = logical_index;
  logical.backend_input_index = backend_input_index;
  logical.physical_index = physical_index;
  logical.shape = shape;
  logical.stride_bytes = contiguous_strides_bytes(shape, dtype_size_bytes_from_token(dtype));
  logical.byte_offset = byte_offset;
  logical.size_bytes = size_bytes_override > 0U ? size_bytes_override
                                                : tensor_size_bytes_from_shape_dtype(shape, dtype);
  logical.dtype = dtype;
  logical.layout = layout;
  logical.logical_name = logical_name;
  logical.backend_name = backend_name;
  logical.segment_name = segment_name;
  logical.materialization_kind = materialization_kind;
  logical.quant = quant;
  return logical;
}

LogicalTensorStaticSpec build_logical_output_static_spec(
    int logical_index, int backend_output_index, int physical_index, int output_slot,
    int tensor_index, const std::vector<std::int64_t>& shape, const std::string& dtype,
    const std::string& layout, const std::string& logical_name, const std::string& backend_name,
    const std::string& segment_name, std::int64_t byte_offset, std::uint64_t size_bytes_override,
    const std::optional<QuantStaticSpec>& quant) {
  LogicalTensorStaticSpec logical;
  logical.logical_index = logical_index;
  logical.backend_output_index = backend_output_index;
  logical.physical_index = physical_index;
  logical.output_slot = output_slot;
  logical.tensor_index = tensor_index;
  logical.byte_offset = byte_offset;
  logical.shape = shape;
  logical.stride_bytes = contiguous_strides_bytes(shape, dtype_size_bytes_from_token(dtype));
  logical.size_bytes = size_bytes_override > 0U ? size_bytes_override
                                                : tensor_size_bytes_from_shape_dtype(shape, dtype);
  logical.dtype = dtype;
  logical.layout = layout;
  logical.logical_name = logical_name;
  logical.backend_name = backend_name;
  logical.segment_name = segment_name;
  logical.quant = quant;
  if (detess_layout_debug_enabled_local()) {
    std::fprintf(stderr,
                 "[detess-layout-debug] where=specbuilder.build_logical_output logical=%d "
                 "backend=%d physical=%d "
                 "slot=%d tensor=%d layout=%s shape=%s size=%" PRIu64 " segment=%s name=%s\n",
                 logical.logical_index, logical.backend_output_index, logical.physical_index,
                 logical.output_slot, logical.tensor_index, logical.layout.c_str(),
                 join_shape_debug_local(logical.shape).c_str(), logical.size_bytes,
                 logical.segment_name.c_str(), logical.logical_name.c_str());
  }
  return logical;
}

PhysicalBufferStaticSpec
build_physical_buffer_static_spec(int physical_index, int allocator_index, std::uint64_t size_bytes,
                                  DeviceKind device_kind, const std::string& segment_name,
                                  int source_physical_index, std::int64_t source_byte_offset) {
  PhysicalBufferStaticSpec physical;
  physical.physical_index = physical_index;
  physical.allocator_index = allocator_index;
  physical.size_bytes = size_bytes;
  physical.device_kind = device_kind;
  physical.segment_name = segment_name;
  physical.source_physical_index = source_physical_index;
  physical.source_byte_offset = source_byte_offset;
  return physical;
}

InputBindingStaticSpec build_input_binding_static_spec(
    int sink_pad_index, int local_logical_input_index, const std::string& cm_input_name,
    const std::string& source_segment_name, int src_logical_output_index, int src_output_slot,
    int src_physical_output_index, std::uint64_t src_physical_size_bytes,
    std::int64_t src_physical_byte_offset, bool required) {
  InputBindingStaticSpec binding;
  binding.sink_pad_index = sink_pad_index;
  binding.local_logical_input_index = local_logical_input_index;
  binding.src_logical_output_index = src_logical_output_index;
  binding.src_output_slot = src_output_slot;
  binding.src_physical_output_index = src_physical_output_index;
  binding.src_physical_size_bytes = src_physical_size_bytes;
  binding.src_physical_byte_offset = src_physical_byte_offset;
  binding.required = required;
  binding.cm_input_name = cm_input_name;
  binding.source_segment_name = source_segment_name;
  return binding;
}

StageOutputRoute build_output_route_static_spec(int output_slot, int logical_output_index,
                                                int tensor_index, const std::string& cm_output_name,
                                                const std::string& segment_name) {
  StageOutputRoute route;
  route.output_slot = output_slot;
  route.logical_output_index = logical_output_index;
  route.tensor_index = tensor_index;
  route.cm_output_name = cm_output_name;
  route.segment_name = segment_name;
  return route;
}

void finalize_logical_input_spec(LogicalInputStaticSpec* logical, std::size_t index,
                                 const std::vector<std::string>& physical_input_names) {
  if (!logical) {
    return;
  }
  if (logical->logical_index < 0) {
    logical->logical_index = static_cast<int>(index);
  }
  if (logical->backend_input_index < 0) {
    logical->backend_input_index = static_cast<int>(index);
  }
  if (logical->physical_index < 0) {
    logical->physical_index = 0;
  }
  if (logical->segment_name.empty() && logical->physical_index >= 0 &&
      static_cast<std::size_t>(logical->physical_index) < physical_input_names.size()) {
    logical->segment_name = physical_input_names[static_cast<std::size_t>(logical->physical_index)];
  }
  if (logical->logical_name.empty()) {
    logical->logical_name =
        !logical->backend_name.empty()
            ? logical->backend_name
            : (!logical->segment_name.empty() ? logical->segment_name
                                              : ("input_" + std::to_string(index)));
  }
  if (logical->backend_name.empty()) {
    logical->backend_name =
        !logical->logical_name.empty() ? logical->logical_name : logical->segment_name;
  }
  if (logical->stride_bytes.empty() && !logical->shape.empty()) {
    logical->stride_bytes =
        contiguous_strides_bytes(logical->shape, dtype_size_bytes_from_token(logical->dtype));
  }
  if (logical->size_bytes == 0U) {
    logical->size_bytes = tensor_size_bytes_from_shape_dtype(logical->shape, logical->dtype);
  }
}

void finalize_logical_output_spec(LogicalTensorStaticSpec* logical, std::size_t index,
                                  const std::vector<std::string>& physical_output_names) {
  if (!logical) {
    return;
  }
  if (logical->logical_index < 0) {
    logical->logical_index = static_cast<int>(index);
  }
  if (logical->backend_output_index < 0) {
    logical->backend_output_index = static_cast<int>(index);
  }
  if (logical->output_slot < 0) {
    logical->output_slot = static_cast<int>(index);
  }
  if (logical->tensor_index < 0) {
    logical->tensor_index = static_cast<int>(index);
  }
  if (logical->physical_index < 0) {
    logical->physical_index = 0;
  }
  if (logical->segment_name.empty() && logical->physical_index >= 0 &&
      static_cast<std::size_t>(logical->physical_index) < physical_output_names.size()) {
    logical->segment_name =
        physical_output_names[static_cast<std::size_t>(logical->physical_index)];
  }
  if (logical->logical_name.empty()) {
    logical->logical_name =
        !logical->backend_name.empty()
            ? logical->backend_name
            : (!logical->segment_name.empty() ? logical->segment_name
                                              : ("output_" + std::to_string(index)));
  }
  if (logical->backend_name.empty()) {
    logical->backend_name =
        !logical->logical_name.empty() ? logical->logical_name : logical->segment_name;
  }
  if (logical->stride_bytes.empty() && !logical->shape.empty()) {
    logical->stride_bytes =
        contiguous_strides_bytes(logical->shape, dtype_size_bytes_from_token(logical->dtype));
  }
  if (logical->size_bytes == 0U) {
    logical->size_bytes = tensor_size_bytes_from_shape_dtype(logical->shape, logical->dtype);
  }
  if (detess_layout_debug_enabled_local()) {
    std::fprintf(stderr,
                 "[detess-layout-debug] where=specbuilder.finalize_logical_output index=%zu "
                 "logical=%d physical=%d "
                 "slot=%d layout=%s shape=%s size=%" PRIu64 " segment=%s name=%s\n",
                 index, logical->logical_index, logical->physical_index, logical->output_slot,
                 logical->layout.c_str(), join_shape_debug_local(logical->shape).c_str(),
                 logical->size_bytes, logical->segment_name.c_str(), logical->logical_name.c_str());
  }
}

} // namespace simaai::neat::pipeline_internal::sima::specbuilders
