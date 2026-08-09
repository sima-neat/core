#define SIMA_NEAT_INTERNAL 1
#include "pipeline/internal/sima/static_contract/LegacyAfeMpkDecoder.h"

#include "pipeline/internal/sima/static_contract/KernelRegistry.h"

#include <nlohmann/json.hpp>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <fstream>
#include <initializer_list>
#include <limits>
#include <sstream>
#include <stdexcept>
#include <unordered_map>
#include <unordered_set>
#include <utility>

namespace simaai::neat::pipeline_internal::sima::static_contract {
namespace {

using Json = nlohmann::json;

struct DecodeAbort final {
  LegacyAfeDecodeErrorCode code;
  std::string path;
  std::string detail;
};

[[noreturn]] void reject(const LegacyAfeDecodeErrorCode code, std::string path,
                         std::string detail) {
  throw DecodeAbort{code, std::move(path), std::move(detail)};
}

const Json* required_member_ptr(const Json& object, const char* key, std::string path) {
  if (!object.is_object() || !object.contains(key)) {
    reject(LegacyAfeDecodeErrorCode::MissingRequiredField, path + "." + key,
           "required MPK member is absent");
  }
  return &object.at(key);
}

std::string required_string(const Json& object, const char* key, const std::string& path) {
  const auto& value = *required_member_ptr(object, key, path);
  if (!value.is_string() || value.get_ref<const std::string&>().empty()) {
    reject(LegacyAfeDecodeErrorCode::InvalidField, path + "." + key, "expected a non-empty string");
  }
  return value.get<std::string>();
}

std::uint64_t positive_u64(const Json& value, const std::string& path) {
  std::uint64_t result = 0;
  if (value.is_number_unsigned()) {
    result = value.get<std::uint64_t>();
  } else if (value.is_number_integer()) {
    const auto signed_value = value.get<std::int64_t>();
    if (signed_value > 0) {
      result = static_cast<std::uint64_t>(signed_value);
    }
  }
  if (result == 0U) {
    reject(LegacyAfeDecodeErrorCode::InvalidField, path, "expected a positive integer");
  }
  return result;
}

std::int64_t integer(const Json& value, const std::string& path) {
  if (!value.is_number_integer()) {
    reject(LegacyAfeDecodeErrorCode::InvalidField, path, "expected an integer");
  }
  return value.get<std::int64_t>();
}

bool required_bool(const Json& object, const char* key, const std::string& path) {
  const auto& value = *required_member_ptr(object, key, path);
  if (!value.is_boolean()) {
    reject(LegacyAfeDecodeErrorCode::InvalidField, path + "." + key, "expected a boolean");
  }
  return value.get<bool>();
}

void require_exact_keys(const Json& object, const std::initializer_list<const char*> keys,
                        const std::string& path) {
  std::unordered_set<std::string> expected;
  for (const char* key : keys) {
    expected.emplace(key);
  }
  if (!object.is_object() || object.size() != expected.size()) {
    reject(LegacyAfeDecodeErrorCode::InvalidField, path,
           "object members do not match the exact legacy typed contract");
  }
  for (const auto& [key, value] : object.items()) {
    (void)value;
    if (!expected.contains(key)) {
      reject(LegacyAfeDecodeErrorCode::InvalidField, path + "." + key,
             "member is not part of the exact legacy typed contract");
    }
  }
}

TensorShape shape(const Json& value, const std::string& path, const bool allow_zero = false) {
  if (!value.is_array() || value.empty()) {
    reject(LegacyAfeDecodeErrorCode::InvalidField, path,
           "expected a non-empty integer shape array");
  }
  TensorShape result;
  result.reserve(value.size());
  for (std::size_t index = 0; index < value.size(); ++index) {
    const auto dimension = integer(value[index], path + "[" + std::to_string(index) + "]");
    if (dimension < 0 || (!allow_zero && dimension == 0)) {
      reject(LegacyAfeDecodeErrorCode::InvalidField, path + "[" + std::to_string(index) + "]",
             allow_zero ? "shape element must be non-negative"
                        : "shape dimension must be positive");
    }
    result.push_back(dimension);
  }
  return result;
}

std::vector<TensorShape> shapes(const Json& object, const char* key, const std::string& path,
                                const bool required) {
  if (!object.contains(key)) {
    if (required) {
      reject(LegacyAfeDecodeErrorCode::MissingRequiredField, path + "." + key,
             "required shape list is absent");
    }
    return {};
  }
  const auto& value = object.at(key);
  if (!value.is_array()) {
    reject(LegacyAfeDecodeErrorCode::InvalidField, path + "." + key, "expected an array of shapes");
  }
  std::vector<TensorShape> result;
  result.reserve(value.size());
  for (std::size_t index = 0; index < value.size(); ++index) {
    result.push_back(shape(value[index], path + "." + key + "[" + std::to_string(index) + "]"));
  }
  return result;
}

std::vector<QuantizationSpec> quantization(const Json& object, const char* key,
                                           const std::string& path) {
  const auto& value = *required_member_ptr(object, key, path);
  if (!value.is_array() || value.empty()) {
    reject(LegacyAfeDecodeErrorCode::InvalidField, path + "." + key,
           "expected a non-empty channel-parameter array");
  }
  std::vector<QuantizationSpec> result;
  result.reserve(value.size());
  for (std::size_t index = 0; index < value.size(); ++index) {
    const auto& pair = value[index];
    const std::string pair_path = path + "." + key + "[" + std::to_string(index) + "]";
    if (!pair.is_array() || pair.size() != 2U || !pair[0].is_number() ||
        !pair[1].is_number_integer()) {
      reject(LegacyAfeDecodeErrorCode::InvalidField, pair_path,
             "expected [positive scale, integer zero-point]");
    }
    const double scale_value = pair[0].get<double>();
    if (!std::isfinite(scale_value) || scale_value <= 0.0) {
      reject(LegacyAfeDecodeErrorCode::InvalidField, pair_path + "[0]",
             "quantization scale must be finite and positive");
    }
    result.push_back({scale_value, pair[1].get<std::int64_t>()});
  }
  return result;
}

struct Node {
  std::string name;
  std::uint64_t bytes = 0;
};

std::vector<Node> nodes(const Json& object, const char* key, const std::string& path) {
  const auto& value = *required_member_ptr(object, key, path);
  if (!value.is_array() || value.empty()) {
    reject(LegacyAfeDecodeErrorCode::InvalidField, path + "." + key,
           "expected a non-empty node array");
  }
  std::vector<Node> result;
  result.reserve(value.size());
  for (std::size_t index = 0; index < value.size(); ++index) {
    const auto& node = value[index];
    const std::string node_path = path + "." + key + "[" + std::to_string(index) + "]";
    if (!node.is_object()) {
      reject(LegacyAfeDecodeErrorCode::InvalidField, node_path, "expected a node object");
    }
    result.push_back(
        {required_string(node, "name", node_path),
         positive_u64(*required_member_ptr(node, "size", node_path), node_path + ".size")});
  }
  return result;
}

std::optional<std::uint64_t> element_width(const std::string& dtype) {
  if (dtype == "int8") {
    return 1U;
  }
  if (dtype == "bfloat16") {
    return 2U;
  }
  if (dtype == "float32") {
    return 4U;
  }
  return std::nullopt;
}

std::optional<std::uint64_t> dense_bytes(const TensorShape& value_shape, const std::string& dtype) {
  const auto width = element_width(dtype);
  if (!width.has_value()) {
    return std::nullopt;
  }
  std::uint64_t product = *width;
  for (const auto dimension : value_shape) {
    if (dimension <= 0 || product > std::numeric_limits<std::uint64_t>::max() /
                                        static_cast<std::uint64_t>(dimension)) {
      return std::nullopt;
    }
    product *= static_cast<std::uint64_t>(dimension);
  }
  return product;
}

std::optional<std::vector<std::int64_t>>
contiguous_stride_bytes(const TensorShape& value_shape, const std::uint64_t element_bytes) {
  if (value_shape.empty() || element_bytes == 0U ||
      element_bytes > static_cast<std::uint64_t>(std::numeric_limits<std::int64_t>::max())) {
    return std::nullopt;
  }
  std::vector<std::int64_t> result(value_shape.size(), 0);
  std::uint64_t stride = element_bytes;
  for (std::size_t reverse = value_shape.size(); reverse > 0U; --reverse) {
    const std::size_t axis = reverse - 1U;
    if (value_shape[axis] <= 0 ||
        stride > static_cast<std::uint64_t>(std::numeric_limits<std::int64_t>::max())) {
      return std::nullopt;
    }
    result[axis] = static_cast<std::int64_t>(stride);
    if (axis != 0U &&
        stride > std::numeric_limits<std::uint64_t>::max() /
                     static_cast<std::uint64_t>(value_shape[axis])) {
      return std::nullopt;
    }
    stride *= static_cast<std::uint64_t>(value_shape[axis]);
  }
  return result;
}

std::optional<std::uint64_t> exact_element_width(const std::uint64_t bytes,
                                                 const TensorShape& value_shape) {
  std::uint64_t elements = 1U;
  for (const auto dimension : value_shape) {
    if (dimension <= 0 ||
        elements > std::numeric_limits<std::uint64_t>::max() /
                       static_cast<std::uint64_t>(dimension)) {
      return std::nullopt;
    }
    elements *= static_cast<std::uint64_t>(dimension);
  }
  if (elements == 0U || bytes % elements != 0U || bytes / elements == 0U) {
    return std::nullopt;
  }
  return bytes / elements;
}

void merge_dtype(ValueSpec& value, const std::string& dtype, const std::string& path) {
  if (value.logical_dtype.has_value() && *value.logical_dtype != dtype) {
    reject(LegacyAfeDecodeErrorCode::ConfigurationMismatch, path,
           "conflicting exact dtype evidence for value '" + value.name + "'");
  }
  value.logical_dtype = dtype;
}

void merge_shape(ValueSpec& value, const TensorShape& value_shape, const std::string& path) {
  if (value.logical_shape.has_value() && *value.logical_shape != value_shape) {
    reject(LegacyAfeDecodeErrorCode::ConfigurationMismatch, path,
           "conflicting exact shape evidence for value '" + value.name + "'");
  }
  value.logical_shape = value_shape;
}

void merge_quantization(ValueSpec& value, const std::vector<QuantizationSpec>& q,
                        const std::string& path) {
  if (!value.quantization.empty()) {
    if (value.quantization.size() != q.size()) {
      reject(LegacyAfeDecodeErrorCode::ConfigurationMismatch, path,
             "conflicting quantization evidence");
    }
    for (std::size_t index = 0; index < q.size(); ++index) {
      if (value.quantization[index].scale != q[index].scale ||
          value.quantization[index].zero_point != q[index].zero_point) {
        reject(LegacyAfeDecodeErrorCode::ConfigurationMismatch, path,
               "conflicting quantization evidence");
      }
    }
  }
  value.quantization = q;
}

std::string topology_error_detail(const MlaElfIoTopologyValidation& validation) {
  std::string detail = validation.detail;
  if (validation.expected != validation.actual) {
    detail += " (expected " + std::to_string(validation.expected) + ", actual " +
              std::to_string(validation.actual) + ")";
  }
  return detail;
}

struct PluginRef {
  std::uint64_t sequence = 0;
  const Json* plugin = nullptr;
  std::size_t manifest_index = 0;
};

OpConfig parse_typed_config(const OpKind kind, const Json& plugin, const Json& config,
                            const Json& params, const std::string& path) {
  switch (kind) {
  case OpKind::Cast: {
    require_exact_keys(params, {"out_dtype", "input_shapes", "output_shapes"},
                       path + ".config_params.params");
    CastOpConfig result{required_string(params, "out_dtype", path + ".params")};
    if (result.output_dtype != "bfloat16" && result.output_dtype != "float32") {
      reject(LegacyAfeDecodeErrorCode::ConfigurationMismatch,
             path + ".config_params.params.out_dtype",
             "legacy cast has no exact registered transition for this dtype");
    }
    return result;
  }
  case OpKind::Quantize: {
    require_exact_keys(params,
                       {"channel_params", "num_bits", "rounding", "output_data_type",
                        "input_shapes", "output_shapes"},
                       path + ".config_params.params");
    QuantizeOpConfig result{required_string(params, "output_data_type", path + ".params"),
                            integer(*required_member_ptr(params, "num_bits", path + ".params"),
                                    path + ".params.num_bits"),
                            required_string(params, "rounding", path + ".params"),
                            quantization(params, "channel_params", path + ".params")};
    if (result.output_dtype != "int8" || result.num_bits != 8 || result.rounding != "TONEAREST") {
      reject(LegacyAfeDecodeErrorCode::ConfigurationMismatch, path + ".config_params.params",
             "legacy quantization must be exact signed INT8/TONEAREST");
    }
    return result;
  }
  case OpKind::Tessellate: {
    require_exact_keys(
        params,
        {"slice_shape", "align_c16", "cblock", "frame_type", "input_shapes", "output_shapes"},
        path + ".config_params.params");
    TessellateOpConfig result{shape(*required_member_ptr(params, "slice_shape", path + ".params"),
                                    path + ".params.slice_shape"),
                              required_bool(params, "align_c16", path + ".params"),
                              required_bool(params, "cblock", path + ".params"),
                              required_string(params, "frame_type", path + ".params")};
    if (result.frame_type != "int8" && result.frame_type != "bfloat16") {
      reject(LegacyAfeDecodeErrorCode::ConfigurationMismatch,
             path + ".config_params.params.frame_type",
             "legacy tessellation frame type is not registered");
    }
    return result;
  }
  case OpKind::Pack:
    require_exact_keys(params, {"input_shapes", "output_shapes"}, path + ".config_params.params");
    return PackOpConfig{};
  case OpKind::Mla: {
    const auto& resources = *required_member_ptr(plugin, "resources", path);
    if (!resources.is_object()) {
      reject(LegacyAfeDecodeErrorCode::InvalidField, path + ".resources", "expected an object");
    }
    const auto quads =
        integer(*required_member_ptr(config, "number_of_quads_to_user", path + ".config_params"),
                path + ".config_params.number_of_quads_to_user");
    if (quads <= 0) {
      reject(LegacyAfeDecodeErrorCode::InvalidField,
             path + ".config_params.number_of_quads_to_user",
             "number of MLA quads must be positive");
    }
    return MlaOpConfig{required_string(resources, "executable", path + ".resources"), quads};
  }
  case OpKind::Unpack: {
    require_exact_keys(params, {"tensor_types", "tensor_shapes", "input_shapes", "output_shapes"},
                       path + ".config_params.params");
    const auto& types = *required_member_ptr(params, "tensor_types", path + ".params");
    if (!types.is_array() || types.empty()) {
      reject(LegacyAfeDecodeErrorCode::InvalidField, path + ".params.tensor_types",
             "expected a non-empty tensor type array");
    }
    std::vector<std::string> tensor_types;
    tensor_types.reserve(types.size());
    for (std::size_t index = 0; index < types.size(); ++index) {
      if (!types[index].is_string() || types[index].get_ref<const std::string&>().empty()) {
        reject(LegacyAfeDecodeErrorCode::InvalidField,
               path + ".params.tensor_types[" + std::to_string(index) + "]",
               "expected a non-empty tensor dtype");
      }
      const std::string tensor_type = types[index].get<std::string>();
      if (tensor_type != "int8" && tensor_type != "bfloat16") {
        reject(LegacyAfeDecodeErrorCode::ConfigurationMismatch,
               path + ".params.tensor_types[" + std::to_string(index) + "]",
               "legacy unpack carrier type is not registered");
      }
      tensor_types.push_back(tensor_type);
    }
    return UnpackOpConfig{std::move(tensor_types),
                          shapes(params, "tensor_shapes", path + ".params", true)};
  }
  case OpKind::Slice: {
    require_exact_keys(
        params, {"begin", "end", "input_shape", "output_shape", "input_shapes", "output_shapes"},
        path + ".config_params.params");
    SliceOpConfig result{
        shape(*required_member_ptr(params, "begin", path + ".params"), path + ".params.begin",
              true),
        shape(*required_member_ptr(params, "end", path + ".params"), path + ".params.end"),
        shape(*required_member_ptr(params, "input_shape", path + ".params"),
              path + ".params.input_shape"),
        shape(*required_member_ptr(params, "output_shape", path + ".params"),
              path + ".params.output_shape")};
    if (result.begin.size() != result.end.size() ||
        result.begin.size() != result.input_shape.size() ||
        result.begin.size() != result.output_shape.size()) {
      reject(LegacyAfeDecodeErrorCode::ConfigurationMismatch, path + ".config_params.params",
             "slice ranks disagree");
    }
    for (std::size_t index = 0; index < result.begin.size(); ++index) {
      if (result.begin[index] > result.end[index] ||
          result.end[index] > result.input_shape[index] ||
          result.end[index] - result.begin[index] != result.output_shape[index]) {
        reject(LegacyAfeDecodeErrorCode::ConfigurationMismatch, path + ".config_params.params",
               "slice bounds do not prove output shape");
      }
    }
    return result;
  }
  case OpKind::Detessellate: {
    require_exact_keys(params,
                       {"slice_shape", "frame_shape", "align_c16", "cblock", "frame_type",
                        "input_shapes", "output_shapes"},
                       path + ".config_params.params");
    DetessellateOpConfig result{shape(*required_member_ptr(params, "slice_shape", path + ".params"),
                                      path + ".params.slice_shape"),
                                shape(*required_member_ptr(params, "frame_shape", path + ".params"),
                                      path + ".params.frame_shape"),
                                required_bool(params, "align_c16", path + ".params"),
                                required_bool(params, "cblock", path + ".params"),
                                required_string(params, "frame_type", path + ".params")};
    if (result.frame_type != "int8" && result.frame_type != "bfloat16") {
      reject(LegacyAfeDecodeErrorCode::ConfigurationMismatch,
             path + ".config_params.params.frame_type",
             "legacy detessellation frame type is not registered");
    }
    return result;
  }
  case OpKind::Dequantize: {
    require_exact_keys(params,
                       {"channel_params", "input_data_type", "input_shapes", "output_shapes"},
                       path + ".config_params.params");
    DequantizeOpConfig result{required_string(params, "input_data_type", path + ".params"),
                              quantization(params, "channel_params", path + ".params")};
    if (result.input_dtype != "int8") {
      reject(LegacyAfeDecodeErrorCode::ConfigurationMismatch,
             path + ".config_params.params.input_data_type",
             "legacy dequantization input must be exact int8");
    }
    return result;
  }
  case OpKind::PassThrough:
    require_exact_keys(params, {}, path + ".config_params.params");
    return PassThroughOpConfig{};
  }
  reject(LegacyAfeDecodeErrorCode::ConfigurationMismatch, path, "unhandled exact operation kind");
}

void apply_input_evidence(ModelExecutionPlanData& data, const OpSpec& op, const std::string& path) {
  if (!op.input_shapes.empty()) {
    if (op.input_shapes.size() != op.inputs.size()) {
      reject(LegacyAfeDecodeErrorCode::ConfigurationMismatch,
             path + ".config_params.params.input_shapes",
             "input shape count does not equal input node count");
    }
    for (std::size_t index = 0; index < op.inputs.size(); ++index) {
      merge_shape(data.values[op.inputs[index]], op.input_shapes[index], path);
    }
  }

  switch (op.kind) {
  case OpKind::Cast: {
    const auto& config = std::get<CastOpConfig>(op.config);
    if (config.output_dtype == "bfloat16") {
      merge_dtype(data.values[op.inputs.front()], "float32", path);
    } else if (config.output_dtype == "float32") {
      merge_dtype(data.values[op.inputs.front()], "bfloat16", path);
    } else {
      reject(LegacyAfeDecodeErrorCode::ConfigurationMismatch, path,
             "legacy cast_transform has an unsupported exact dtype transition");
    }
    break;
  }
  case OpKind::Quantize:
    merge_dtype(data.values[op.inputs.front()], "float32", path);
    break;
  case OpKind::Tessellate:
    merge_dtype(data.values[op.inputs.front()], std::get<TessellateOpConfig>(op.config).frame_type,
                path);
    break;
  case OpKind::Detessellate:
    merge_dtype(data.values[op.inputs.front()],
                std::get<DetessellateOpConfig>(op.config).frame_type, path);
    break;
  case OpKind::Dequantize: {
    const auto& config = std::get<DequantizeOpConfig>(op.config);
    merge_dtype(data.values[op.inputs.front()], config.input_dtype, path);
    merge_quantization(data.values[op.inputs.front()], config.channel_params, path);
    break;
  }
  default:
    break;
  }
}

ValueSpec make_output_value(const ValueId id, const Node& node, const OpSpec& op,
                            const std::size_t output_index, const ModelExecutionPlanData& data) {
  ValueSpec value;
  value.id = id;
  value.name = node.name;
  value.required_bytes = node.bytes;
  if (!op.output_shapes.empty()) {
    value.logical_shape = op.output_shapes.at(output_index);
  }

  switch (op.kind) {
  case OpKind::Cast:
    value.logical_dtype = std::get<CastOpConfig>(op.config).output_dtype;
    break;
  case OpKind::Quantize: {
    const auto& config = std::get<QuantizeOpConfig>(op.config);
    value.logical_dtype = config.output_dtype;
    value.quantization = config.channel_params;
    break;
  }
  case OpKind::Tessellate:
    value.logical_dtype = std::get<TessellateOpConfig>(op.config).frame_type;
    value.representation = ValueRepresentation::Tessellated;
    break;
  case OpKind::Pack:
    value.representation = ValueRepresentation::Packed;
    break;
  case OpKind::Mla:
    value.representation =
        op.outputs.size() == 1U ? ValueRepresentation::Packed : ValueRepresentation::BackendNative;
    break;
  case OpKind::Unpack: {
    // Legacy `tensor_types` describes the unpack carrier units, not always the
    // logical MLA tensor precision (BF16 EV-tess packages legitimately label
    // these byte carriers int8). Preserve it in UnpackOpConfig and let the
    // exact downstream detess/slice/precision operation prove logical facts.
    value.logical_shape.reset();
    value.representation = ValueRepresentation::BackendNative;
    break;
  }
  case OpKind::Slice:
    value.logical_shape = std::get<SliceOpConfig>(op.config).output_shape;
    if (const auto& input = data.values[op.inputs.front()]; input.logical_dtype.has_value()) {
      value.logical_dtype = input.logical_dtype;
      value.quantization = input.quantization;
    }
    break;
  case OpKind::Detessellate:
    value.logical_dtype = std::get<DetessellateOpConfig>(op.config).frame_type;
    break;
  case OpKind::Dequantize:
    value.logical_dtype = "float32";
    break;
  case OpKind::PassThrough: {
    const auto& input = data.values[op.inputs.at(output_index)];
    value.logical_dtype = input.logical_dtype;
    value.logical_shape = input.logical_shape;
    value.logical_layout = input.logical_layout;
    value.quantization = input.quantization;
    value.representation = input.representation;
    break;
  }
  }
  return value;
}

void propagate_identity_evidence(ModelExecutionPlanData& data) {
  // Slice and PassThrough preserve dtype/quantization. Later operations often
  // provide the only exact input-type evidence, so converge both directions.
  bool changed = true;
  while (changed) {
    changed = false;
    for (const auto& op : data.ops) {
      if (op.kind != OpKind::Slice && op.kind != OpKind::PassThrough) {
        continue;
      }
      for (std::size_t index = 0; index < op.outputs.size(); ++index) {
        auto& input = data.values[op.inputs[op.kind == OpKind::PassThrough ? index : 0U]];
        auto& output = data.values[op.outputs[index]];
        if (!input.logical_dtype.has_value() && output.logical_dtype.has_value()) {
          input.logical_dtype = output.logical_dtype;
          changed = true;
        }
        if (!output.logical_dtype.has_value() && input.logical_dtype.has_value()) {
          output.logical_dtype = input.logical_dtype;
          changed = true;
        }
        if (input.quantization.empty() && !output.quantization.empty()) {
          input.quantization = output.quantization;
          changed = true;
        }
        if (output.quantization.empty() && !input.quantization.empty()) {
          output.quantization = input.quantization;
          changed = true;
        }
      }
    }
  }
}

void validate_dense_byte_equations(const ModelExecutionPlanData& data) {
  for (const auto& value : data.values) {
    if (value.representation != ValueRepresentation::Dense || !value.logical_dtype.has_value() ||
        !value.logical_shape.has_value()) {
      continue;
    }
    const auto expected = dense_bytes(*value.logical_shape, *value.logical_dtype);
    if (!expected.has_value()) {
      reject(LegacyAfeDecodeErrorCode::ConfigurationMismatch,
             "$.values[" + std::to_string(value.id) + "]",
             "unsupported or overflowing dense byte equation for value '" + value.name + "'");
    }
    if (*expected != value.required_bytes) {
      reject(
          LegacyAfeDecodeErrorCode::ValueSizeMismatch, "$.values[" + std::to_string(value.id) + "]",
          "dense byte equation disagrees for value '" + value.name + "': expected " +
              std::to_string(*expected) + ", MPK declares " + std::to_string(value.required_bytes));
    }
  }
}

void lower_read_expressions(ModelExecutionPlanData& data,
                            std::vector<LegacyAfeProofFact>* proof) {
  for (const auto& op : data.ops) {
    if (op.kind == OpKind::Unpack) {
      if (op.inputs.size() != 1U || op.outputs.empty()) {
        reject(LegacyAfeDecodeErrorCode::ConfigurationMismatch,
               "$.ops[" + std::to_string(op.id) + "]",
               "unpack read expression requires one carrier and at least one view");
      }
      const auto& source = data.values[op.inputs.front()];
      if (source.read_expression.has_value()) {
        reject(LegacyAfeDecodeErrorCode::ConfigurationMismatch,
               "$.ops[" + std::to_string(op.id) + "]",
               "unpack carrier must already be a materialized root buffer");
      }
      const auto& config = std::get<UnpackOpConfig>(op.config);
      if (config.tensor_types.size() != op.outputs.size() ||
          config.tensor_shapes.size() != op.outputs.size()) {
        reject(LegacyAfeDecodeErrorCode::ConfigurationMismatch,
               "$.ops[" + std::to_string(op.id) + "]",
               "unpack read-expression arity is inconsistent");
      }

      std::uint64_t offset = 0U;
      for (std::size_t index = 0; index < op.outputs.size(); ++index) {
        auto& output = data.values[op.outputs[index]];
        const auto width = element_width(config.tensor_types[index]);
        const auto strides = width.has_value()
                                 ? contiguous_stride_bytes(config.tensor_shapes[index], *width)
                                 : std::nullopt;
        if (!strides.has_value() || offset > source.required_bytes ||
            output.required_bytes > source.required_bytes - offset) {
          reject(LegacyAfeDecodeErrorCode::ValueSizeMismatch,
                 "$.ops[" + std::to_string(op.id) + "]",
                 "unpack view does not fit its exact packed carrier");
        }
        output.read_expression = ReadExpression{source.id, offset, *strides};
        if (proof) {
          proof->push_back(
              {"read[" + std::to_string(output.id) + "]",
               "unpack lowers to root value '" + source.name + "' + " +
                   std::to_string(offset) + " bytes; no runtime operation is scheduled"});
        }
        offset += output.required_bytes;
      }
      if (offset != source.required_bytes) {
        reject(LegacyAfeDecodeErrorCode::ValueSizeMismatch,
               "$.ops[" + std::to_string(op.id) + "]",
               "unpack views do not partition the exact packed carrier");
      }
      continue;
    }

    if (op.kind == OpKind::Slice) {
      if (op.inputs.size() != 1U || op.outputs.size() != 1U) {
        reject(LegacyAfeDecodeErrorCode::ConfigurationMismatch,
               "$.ops[" + std::to_string(op.id) + "]",
               "slice read expression requires one input and one output");
      }
      const auto& input = data.values[op.inputs.front()];
      auto& output = data.values[op.outputs.front()];
      const auto& config = std::get<SliceOpConfig>(op.config);

      ValueId root = input.id;
      std::uint64_t base_offset = 0U;
      std::vector<std::int64_t> strides;
      if (input.read_expression.has_value()) {
        root = input.read_expression->source_value_id;
        base_offset = input.read_expression->byte_offset;
        strides = input.read_expression->stride_bytes;
      } else {
        const auto width = exact_element_width(input.required_bytes, config.input_shape);
        const auto dense_strides = width.has_value()
                                       ? contiguous_stride_bytes(config.input_shape, *width)
                                       : std::nullopt;
        if (!dense_strides.has_value()) {
          reject(LegacyAfeDecodeErrorCode::ConfigurationMismatch,
                 "$.ops[" + std::to_string(op.id) + "]",
                 "slice input has no exact dense carrier stride");
        }
        strides = *dense_strides;
      }
      if (strides.size() != config.begin.size()) {
        reject(LegacyAfeDecodeErrorCode::ConfigurationMismatch,
               "$.ops[" + std::to_string(op.id) + "]",
               "slice input view rank disagrees with its byte strides");
      }
      for (std::size_t axis = 0; axis < config.begin.size(); ++axis) {
        const auto begin = static_cast<std::uint64_t>(config.begin[axis]);
        const auto stride = static_cast<std::uint64_t>(strides[axis]);
        if (begin != 0U && stride > std::numeric_limits<std::uint64_t>::max() / begin) {
          reject(LegacyAfeDecodeErrorCode::ConfigurationMismatch,
                 "$.ops[" + std::to_string(op.id) + "]",
                 "slice read-expression offset overflows");
        }
        const auto axis_offset = begin * stride;
        if (base_offset > std::numeric_limits<std::uint64_t>::max() - axis_offset) {
          reject(LegacyAfeDecodeErrorCode::ConfigurationMismatch,
                 "$.ops[" + std::to_string(op.id) + "]",
                 "slice read-expression offset overflows");
        }
        base_offset += axis_offset;
      }
      output.read_expression = ReadExpression{root, base_offset, std::move(strides)};
      if (proof) {
        proof->push_back(
            {"read[" + std::to_string(output.id) + "]",
             "slice composes to root value '" + data.values[root].name + "' + " +
                 std::to_string(base_offset) + " bytes; no runtime operation is scheduled"});
      }
      continue;
    }

    // PassThrough is publication metadata.  Preserve an already-compiled view
    // without turning that metadata edge into work for any backend.
    if (op.kind == OpKind::PassThrough) {
      for (std::size_t index = 0; index < op.outputs.size(); ++index) {
        const auto& input = data.values[op.inputs[index]];
        if (input.read_expression.has_value()) {
          data.values[op.outputs[index]].read_expression = input.read_expression;
        }
      }
    }
  }
}

LegacyAfeDecodeResult decode_impl(const std::string_view text, const MlaElfIoTopology& topology,
                                  const std::string& source) {
  LegacyAfeDecodeResult result;
  try {
    Json root = Json::parse(text.begin(), text.end(), nullptr, false);
    if (root.is_discarded() || !root.is_object()) {
      reject(LegacyAfeDecodeErrorCode::InvalidJson, "$",
             "manifest is not valid JSON object syntax");
    }

    ModelExecutionPlanData data;
    data.contract_version = required_string(root, "model_sdk_version", "$");
    if (data.contract_version != "2.0.0") {
      reject(LegacyAfeDecodeErrorCode::UnsupportedContractVersion, "$.model_sdk_version",
             "LegacyAfeMpkDecoder supports only exact contract version 2.0.0");
    }
    result.proof.push_back({"contract.version", "MPK $.model_sdk_version exactly equals '2.0.0'"});

    const auto& input_array = *required_member_ptr(root, "input_nodes", "$");
    if (!input_array.is_array() || input_array.empty()) {
      reject(LegacyAfeDecodeErrorCode::InvalidField, "$.input_nodes",
             "expected at least one model input");
    }

    std::unordered_map<std::string, ValueId> value_by_name;
    for (std::size_t index = 0; index < input_array.size(); ++index) {
      const auto& input = input_array[index];
      const std::string path = "$.input_nodes[" + std::to_string(index) + "]";
      if (!input.is_object()) {
        reject(LegacyAfeDecodeErrorCode::InvalidField, path, "expected a node object");
      }
      ValueSpec value;
      value.id = static_cast<ValueId>(data.values.size());
      value.name = required_string(input, "name", path);
      value.required_bytes =
          positive_u64(*required_member_ptr(input, "size", path), path + ".size");
      if (!value_by_name.emplace(value.name, value.id).second) {
        reject(LegacyAfeDecodeErrorCode::DuplicateProducer, path + ".name",
               "model input name has a duplicate producer");
      }
      data.model_inputs.push_back(value.id);
      result.proof.push_back({"model.input[" + std::to_string(index) + "]",
                              "MPK input node '" + value.name + "' declares " +
                                  std::to_string(value.required_bytes) + " bytes"});
      data.values.push_back(std::move(value));
    }

    const auto& plugins = *required_member_ptr(root, "plugins", "$");
    if (!plugins.is_array() || plugins.empty()) {
      reject(LegacyAfeDecodeErrorCode::InvalidField, "$.plugins",
             "expected a non-empty plugin array");
    }
    std::vector<PluginRef> ordered;
    ordered.reserve(plugins.size());
    std::unordered_set<std::uint64_t> sequences;
    for (std::size_t index = 0; index < plugins.size(); ++index) {
      const auto& plugin = plugins[index];
      const std::string path = "$.plugins[" + std::to_string(index) + "]";
      if (!plugin.is_object()) {
        reject(LegacyAfeDecodeErrorCode::InvalidField, path, "expected a plugin object");
      }
      const auto sequence =
          positive_u64(*required_member_ptr(plugin, "sequence", path), path + ".sequence");
      if (!sequences.emplace(sequence).second) {
        reject(LegacyAfeDecodeErrorCode::DuplicateSequence, path + ".sequence",
               "plugin sequence is duplicated");
      }
      ordered.push_back({sequence, &plugin, index});
    }
    std::sort(ordered.begin(), ordered.end(), [](const PluginRef& left, const PluginRef& right) {
      return left.sequence < right.sequence;
    });
    for (std::size_t index = 0; index < ordered.size(); ++index) {
      if (ordered[index].sequence != index + 1U) {
        reject(LegacyAfeDecodeErrorCode::InvalidField,
               "$.plugins[" + std::to_string(ordered[index].manifest_index) + "].sequence",
               "plugin sequences must be contiguous starting at one");
      }
    }

    std::size_t mla_count = 0U;
    std::size_t pass_count = 0U;
    std::size_t mla_op_index = 0U;
    std::size_t pass_op_index = 0U;
    std::unordered_set<std::string> operation_names;
    for (const auto& ordered_plugin : ordered) {
      const Json& plugin = *ordered_plugin.plugin;
      const std::string path = "$.plugins[" + std::to_string(ordered_plugin.manifest_index) + "]";
      const std::string name = required_string(plugin, "name", path);
      if (!operation_names.emplace(name).second) {
        reject(LegacyAfeDecodeErrorCode::InvalidField, path + ".name",
               "operation name is duplicated");
      }
      const std::string processor = required_string(plugin, "processor", path);
      if (required_string(plugin, "type", path) != "sgpProcess") {
        reject(LegacyAfeDecodeErrorCode::InvalidField, path + ".type",
               "legacy static-plan plugin must be exactly 'sgpProcess'");
      }
      const auto& config = *required_member_ptr(plugin, "config_params", path);
      if (!config.is_object()) {
        reject(LegacyAfeDecodeErrorCode::InvalidField, path + ".config_params",
               "expected an object");
      }
      if (processor == "MLA") {
        require_exact_keys(config,
                           {"desired_batch_size", "actual_batch_size", "number_of_quads_to_user"},
                           path + ".config_params");
      } else {
        require_exact_keys(config, {"desired_batch_size", "actual_batch_size", "kernel", "params"},
                           path + ".config_params");
      }
      const auto desired_batch =
          positive_u64(*required_member_ptr(config, "desired_batch_size", path + ".config_params"),
                       path + ".config_params.desired_batch_size");
      const auto actual_batch =
          positive_u64(*required_member_ptr(config, "actual_batch_size", path + ".config_params"),
                       path + ".config_params.actual_batch_size");
      if (desired_batch != actual_batch) {
        reject(LegacyAfeDecodeErrorCode::ConfigurationMismatch, path + ".config_params",
               "desired and actual batch sizes disagree");
      }

      std::string kernel;
      if (config.contains("kernel")) {
        if (!config.at("kernel").is_string()) {
          reject(LegacyAfeDecodeErrorCode::InvalidField, path + ".config_params.kernel",
                 "expected a string");
        }
        kernel = config.at("kernel").get<std::string>();
      } else if (processor != "MLA") {
        reject(LegacyAfeDecodeErrorCode::MissingRequiredField, path + ".config_params.kernel",
               "non-MLA plugin has no exact kernel token");
      }
      const auto descriptor = lookup_exact_kernel(data.contract_version, processor, kernel);
      if (!descriptor.has_value()) {
        reject(LegacyAfeDecodeErrorCode::UnsupportedKernel, path + ".config_params.kernel",
               "no exact registry entry for ('" + data.contract_version + "', '" + processor +
                   "', '" + kernel + "')");
      }

      const auto input_nodes = nodes(plugin, "input_nodes", path);
      const auto output_nodes = nodes(plugin, "output_nodes", path);
      if (!exact_kernel_arity_is_valid(*descriptor, input_nodes.size(), output_nodes.size())) {
        reject(LegacyAfeDecodeErrorCode::InvalidKernelArity, path,
               "node arity violates the exact kernel registry entry");
      }

      OpSpec op;
      op.id = static_cast<OpId>(data.ops.size());
      op.sequence = ordered_plugin.sequence;
      op.name = name;
      op.kind = descriptor->kind;
      op.processor = processor;
      op.kernel = kernel;
      for (std::size_t index = 0; index < input_nodes.size(); ++index) {
        const auto found = value_by_name.find(input_nodes[index].name);
        if (found == value_by_name.end()) {
          reject(LegacyAfeDecodeErrorCode::MissingProducer,
                 path + ".input_nodes[" + std::to_string(index) + "].name",
                 "no earlier exact producer for tensor '" + input_nodes[index].name + "'");
        }
        if (data.values[found->second].required_bytes != input_nodes[index].bytes) {
          reject(LegacyAfeDecodeErrorCode::ValueSizeMismatch,
                 path + ".input_nodes[" + std::to_string(index) + "].size",
                 "consumer byte extent disagrees with exact producer for tensor '" +
                     input_nodes[index].name + "'");
        }
        op.inputs.push_back(found->second);
      }

      Json empty_params = Json::object();
      const Json* params = &empty_params;
      if (op.kind != OpKind::Mla) {
        const auto& params_member = *required_member_ptr(config, "params", path + ".config_params");
        if (!params_member.is_object()) {
          reject(LegacyAfeDecodeErrorCode::InvalidField, path + ".config_params.params",
                 "expected an object");
        }
        params = &params_member;
      }
      const bool shape_lists_required = op.kind != OpKind::Mla && op.kind != OpKind::PassThrough;
      op.input_shapes =
          shapes(*params, "input_shapes", path + ".config_params.params", shape_lists_required);
      op.output_shapes =
          shapes(*params, "output_shapes", path + ".config_params.params", shape_lists_required);
      if ((!op.input_shapes.empty() && op.input_shapes.size() != input_nodes.size()) ||
          (!op.output_shapes.empty() && op.output_shapes.size() != output_nodes.size())) {
        reject(LegacyAfeDecodeErrorCode::ConfigurationMismatch, path + ".config_params.params",
               "typed shape-list arity disagrees with MPK node arity");
      }
      op.config = parse_typed_config(op.kind, plugin, config, *params, path);
      if (op.kind == OpKind::Pack) {
        if (actual_batch != 1U) {
          reject(LegacyAfeDecodeErrorCode::ConfigurationMismatch,
                 path + ".config_params.actual_batch_size",
                 "legacy Pack placement is supported only for exact batch one");
        }
        auto& pack = std::get<PackOpConfig>(op.config);
        std::uint64_t parent_offset = 0U;
        pack.components.reserve(op.inputs.size());
        for (const auto input_id : op.inputs) {
          const auto input_bytes = data.values[input_id].required_bytes;
          if (input_bytes >
              std::numeric_limits<std::uint64_t>::max() - 15U) {
            reject(LegacyAfeDecodeErrorCode::ConfigurationMismatch,
                   path + ".config_params.params.input_shapes",
                   "legacy Pack component alignment overflows");
          }
          const auto stored_bytes = (input_bytes + 15U) & ~std::uint64_t{15U};
          if (parent_offset >
              std::numeric_limits<std::uint64_t>::max() - stored_bytes) {
            reject(LegacyAfeDecodeErrorCode::ConfigurationMismatch,
                   path + ".config_params.params.input_shapes",
                   "legacy Pack parent placement overflows");
          }
          pack.components.push_back(
              {input_id, parent_offset, stored_bytes});
          parent_offset += stored_bytes;
        }
      }
      if (op.kind == OpKind::Unpack) {
        const auto& unpack = std::get<UnpackOpConfig>(op.config);
        if (unpack.tensor_types.size() != output_nodes.size() ||
            unpack.tensor_shapes.size() != output_nodes.size() ||
            unpack.tensor_shapes != op.output_shapes) {
          reject(LegacyAfeDecodeErrorCode::ConfigurationMismatch, path + ".config_params.params",
                 "unpack tensor metadata disagrees with output nodes/shapes");
        }
        for (std::size_t index = 0; index < output_nodes.size(); ++index) {
          const auto expected =
              dense_bytes(unpack.tensor_shapes[index], unpack.tensor_types[index]);
          if (!expected.has_value() || *expected != output_nodes[index].bytes) {
            reject(LegacyAfeDecodeErrorCode::ValueSizeMismatch,
                   path + ".output_nodes[" + std::to_string(index) + "].size",
                   "unpack carrier byte equation disagrees with MPK output size");
          }
        }
      }
      if (op.kind == OpKind::Slice) {
        const auto& slice_config = std::get<SliceOpConfig>(op.config);
        if (slice_config.input_shape != op.input_shapes.front() ||
            slice_config.output_shape != op.output_shapes.front() ||
            slice_config.begin.size() != slice_config.end.size() ||
            slice_config.begin.size() != slice_config.input_shape.size() ||
            slice_config.end.size() != slice_config.output_shape.size()) {
          reject(LegacyAfeDecodeErrorCode::ConfigurationMismatch, path + ".config_params.params",
                 "slice typed fields are internally inconsistent");
        }
      }
      if (op.kind == OpKind::PassThrough) {
        for (std::size_t index = 0; index < input_nodes.size(); ++index) {
          if (input_nodes[index].bytes != output_nodes[index].bytes) {
            reject(LegacyAfeDecodeErrorCode::ValueSizeMismatch,
                   path + ".output_nodes[" + std::to_string(index) + "].size",
                   "PassThrough must preserve each exact byte extent");
          }
        }
      }

      apply_input_evidence(data, op, path);
      std::unordered_set<std::string> pending_output_names;
      for (std::size_t index = 0; index < output_nodes.size(); ++index) {
        const ValueId value_id = static_cast<ValueId>(data.values.size() + index);
        if (value_by_name.contains(output_nodes[index].name) ||
            !pending_output_names.emplace(output_nodes[index].name).second) {
          reject(LegacyAfeDecodeErrorCode::DuplicateProducer,
                 path + ".output_nodes[" + std::to_string(index) + "].name",
                 "tensor '" + output_nodes[index].name + "' has duplicate producers");
        }
        op.outputs.push_back(value_id);
        // make_output_value needs the complete ordered output-id list for MLA
        // representation classification, which is why ids are prepared first.
      }
      for (std::size_t index = 0; index < output_nodes.size(); ++index) {
        auto value = make_output_value(op.outputs[index], output_nodes[index], op, index, data);
        value_by_name.emplace(value.name, value.id);
        data.values.push_back(std::move(value));
      }

      result.proof.push_back({"op[" + std::to_string(op.id) + "]",
                              "exact registry key ('" + data.contract_version + "', '" + processor +
                                  "', '" + kernel + "') resolves plugin '" + name + "'"});
      if (op.kind == OpKind::Mla) {
        ++mla_count;
        mla_op_index = data.ops.size();
      }
      if (op.kind == OpKind::PassThrough) {
        ++pass_count;
        pass_op_index = data.ops.size();
      }
      data.ops.push_back(std::move(op));
    }

    if (mla_count == 0U) {
      reject(LegacyAfeDecodeErrorCode::MissingMlaStage, "$.plugins",
             "no exact MLA operation exists");
    }
    if (mla_count != 1U) {
      reject(LegacyAfeDecodeErrorCode::MultipleMlaStages, "$.plugins",
             "legacy EVO decoder accepts exactly one MLA stage");
    }
    if (pass_count == 0U) {
      reject(LegacyAfeDecodeErrorCode::MissingPublicationStage, "$.plugins",
             "no exact PassThrough publication operation exists");
    }
    if (pass_count != 1U || pass_op_index + 1U != data.ops.size()) {
      reject(LegacyAfeDecodeErrorCode::InvalidPublicationStage, "$.plugins",
             "PassThrough must be the unique terminal operation");
    }

    const auto& mla = data.ops[mla_op_index];
    const auto topology_validation =
        reconcile_mla_elf_io_topology_strict(topology, mla.inputs.size(), mla.outputs.size());
    if (!topology_validation.ok) {
      const bool mismatch =
          topology_validation.code == MlaElfIoTopologyError::IfmPortCountMismatch ||
          topology_validation.code == MlaElfIoTopologyError::OfmPortCountMismatch;
      reject(mismatch ? LegacyAfeDecodeErrorCode::ElfTopologyMismatch
                      : LegacyAfeDecodeErrorCode::ElfTopologyInvalid,
             "$.plugins[" + std::to_string(mla_op_index) + "]",
             topology_error_detail(topology_validation));
    }

    for (std::size_t index = 0; index < mla.inputs.size(); ++index) {
      const std::string symbol =
          topology.monolithic_ifm ? "data.ifm.b0" : topology.ifm_symbol_names.at(index);
      const auto& value = data.values[mla.inputs[index]];
      data.backend_ports.push_back({0U, BackendPortDirection::Input, index, symbol, value.id,
                                    value.required_bytes, kLegacyEvoCmaRegionAlignmentBytes,
                                    BackendPortAlignmentAuthority::LegacyPolicy,
                                    BackendPortAccess::ReadOnly});
      result.proof.push_back({"MLA.IFM[" + std::to_string(index) + "]",
                              "ELF symbol '" + symbol + "' and MPK MLA input[" +
                                  std::to_string(index) +
                                  "] agree; 4096-byte alignment is explicit legacy CMA policy"});
    }
    for (std::size_t index = 0; index < mla.outputs.size(); ++index) {
      const std::string symbol =
          topology.monolithic_ofm ? "data.ofm.b0" : topology.ofm_symbol_names.at(index);
      const auto& value = data.values[mla.outputs[index]];
      data.backend_ports.push_back({0U, BackendPortDirection::Output, index, symbol, value.id,
                                    value.required_bytes, kLegacyEvoCmaRegionAlignmentBytes,
                                    BackendPortAlignmentAuthority::LegacyPolicy,
                                    BackendPortAccess::WriteOnly});
      result.proof.push_back({"MLA.OFM[" + std::to_string(index) + "]",
                              "ELF symbol '" + symbol + "' and MPK MLA output[" +
                                  std::to_string(index) +
                                  "] agree; 4096-byte alignment is explicit legacy CMA policy"});
    }

    const auto& publication = data.ops[pass_op_index];
    for (std::size_t index = 0; index < publication.outputs.size(); ++index) {
      const auto value_id = publication.outputs[index];
      data.model_outputs.push_back({index, data.values[value_id].name, value_id});
      result.proof.push_back(
          {"model.output[" + std::to_string(index) + "]",
           "terminal exact PassThrough output preserves tuple index and full name '" +
               data.values[value_id].name + "'"});
    }

    std::unordered_set<ValueId> consumed_values;
    for (const auto& op : data.ops) {
      consumed_values.insert(op.inputs.begin(), op.inputs.end());
    }
    std::unordered_set<ValueId> public_values;
    for (const auto& output : data.model_outputs) {
      public_values.emplace(output.value_id);
    }
    for (const auto& value : data.values) {
      if (!consumed_values.contains(value.id) && !public_values.contains(value.id)) {
        reject(LegacyAfeDecodeErrorCode::ConfigurationMismatch,
               "$.values[" + std::to_string(value.id) + "]",
               "value '" + value.name + "' is neither consumed nor publicly published");
      }
    }

    propagate_identity_evidence(data);
    validate_dense_byte_equations(data);
    lower_read_expressions(data, &result.proof);

    std::string plan_error;
    result.plan = ModelExecutionPlan::create(std::move(data), &plan_error);
    if (!result.plan.has_value()) {
      reject(LegacyAfeDecodeErrorCode::PlanValidationFailed, "$", plan_error);
    }
    return result;
  } catch (const DecodeAbort& failure) {
    result.plan.reset();
    result.error = LegacyAfeDecodeError{failure.code, source, failure.path, failure.detail};
    return result;
  } catch (const std::exception& failure) {
    result.plan.reset();
    result.error =
        LegacyAfeDecodeError{LegacyAfeDecodeErrorCode::InvalidJson, source, "$", failure.what()};
    return result;
  } catch (...) {
    result.plan.reset();
    result.error = LegacyAfeDecodeError{LegacyAfeDecodeErrorCode::InvalidJson, source, "$",
                                        "unknown decoder failure"};
    return result;
  }
}

} // namespace

LegacyAfeDecodeResult LegacyAfeMpkDecoder::decode_json(const std::string_view mpk_json,
                                                       const MlaElfIoTopology& elf_topology,
                                                       std::string source_label) const noexcept {
  return decode_impl(mpk_json, elf_topology, source_label);
}

LegacyAfeDecodeResult
LegacyAfeMpkDecoder::decode_file(const std::filesystem::path& mpk_manifest,
                                 const MlaElfIoTopology& elf_topology) const noexcept {
  std::ifstream input(mpk_manifest, std::ios::binary);
  if (!input.is_open()) {
    LegacyAfeDecodeResult result;
    result.error = LegacyAfeDecodeError{LegacyAfeDecodeErrorCode::IoError, mpk_manifest.string(),
                                        "$", "cannot open MPK manifest"};
    return result;
  }
  std::ostringstream contents;
  contents << input.rdbuf();
  if (!input.good() && !input.eof()) {
    LegacyAfeDecodeResult result;
    result.error = LegacyAfeDecodeError{LegacyAfeDecodeErrorCode::IoError, mpk_manifest.string(),
                                        "$", "cannot read MPK manifest"};
    return result;
  }
  return decode_impl(contents.str(), elf_topology, mpk_manifest.string());
}

} // namespace simaai::neat::pipeline_internal::sima::static_contract
