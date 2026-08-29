#define SIMA_NEAT_INTERNAL 1
#include "pipeline/internal/sima/static_contract/AfeMpkV2Decoder.h"

#include "pipeline/internal/sima/static_contract/AfePublicationLedger.h"
#include "pipeline/internal/sima/static_contract/KernelRegistry.h"

#include <glib.h>
#include <nlohmann/json.hpp>

#include <algorithm>
#include <array>
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
  AfeMpkV2DecodeErrorCode code;
  std::string path;
  std::string detail;
};

std::string sha256_text(const std::string_view text) {
  gchar* digest = g_compute_checksum_for_data(
      G_CHECKSUM_SHA256, reinterpret_cast<const guchar*>(text.data()), text.size());
  if (!digest) {
    return {};
  }
  std::string result(digest);
  g_free(digest);
  return result;
}

[[noreturn]] void reject(const AfeMpkV2DecodeErrorCode code, std::string path, std::string detail) {
  throw DecodeAbort{code, std::move(path), std::move(detail)};
}

const Json* required_member_ptr(const Json& object, const char* key, std::string path) {
  if (!object.is_object() || !object.contains(key)) {
    reject(AfeMpkV2DecodeErrorCode::MissingRequiredField, path + "." + key,
           "required MPK member is absent");
  }
  return &object.at(key);
}

std::string required_string(const Json& object, const char* key, const std::string& path) {
  const auto& value = *required_member_ptr(object, key, path);
  if (!value.is_string() || value.get_ref<const std::string&>().empty()) {
    reject(AfeMpkV2DecodeErrorCode::InvalidField, path + "." + key, "expected a non-empty string");
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
    reject(AfeMpkV2DecodeErrorCode::InvalidField, path, "expected a positive integer");
  }
  return result;
}

std::int64_t integer(const Json& value, const std::string& path) {
  if (!value.is_number_integer()) {
    reject(AfeMpkV2DecodeErrorCode::InvalidField, path, "expected an integer");
  }
  return value.get<std::int64_t>();
}

bool required_bool(const Json& object, const char* key, const std::string& path) {
  const auto& value = *required_member_ptr(object, key, path);
  if (!value.is_boolean()) {
    reject(AfeMpkV2DecodeErrorCode::InvalidField, path + "." + key, "expected a boolean");
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
    reject(AfeMpkV2DecodeErrorCode::InvalidField, path,
           "object members do not match the exact legacy typed contract");
  }
  for (const auto& [key, value] : object.items()) {
    (void)value;
    if (!expected.contains(key)) {
      reject(AfeMpkV2DecodeErrorCode::InvalidField, path + "." + key,
             "member is not part of the exact legacy typed contract");
    }
  }
}

TensorShape shape(const Json& value, const std::string& path, const bool allow_zero = false) {
  if (!value.is_array() || value.empty()) {
    reject(AfeMpkV2DecodeErrorCode::InvalidField, path, "expected a non-empty integer shape array");
  }
  TensorShape result;
  result.reserve(value.size());
  for (std::size_t index = 0; index < value.size(); ++index) {
    const auto dimension = integer(value[index], path + "[" + std::to_string(index) + "]");
    if (dimension < 0 || (!allow_zero && dimension == 0)) {
      reject(AfeMpkV2DecodeErrorCode::InvalidField, path + "[" + std::to_string(index) + "]",
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
      reject(AfeMpkV2DecodeErrorCode::MissingRequiredField, path + "." + key,
             "required shape list is absent");
    }
    return {};
  }
  const auto& value = object.at(key);
  if (!value.is_array()) {
    reject(AfeMpkV2DecodeErrorCode::InvalidField, path + "." + key, "expected an array of shapes");
  }
  std::vector<TensorShape> result;
  result.reserve(value.size());
  for (std::size_t index = 0; index < value.size(); ++index) {
    result.push_back(shape(value[index], path + "." + key + "[" + std::to_string(index) + "]"));
  }
  return result;
}

std::optional<std::uint64_t> element_width(const std::string& dtype);

std::vector<HostTensorTypeSpec> host_tensor_types(const Json& object, const char* key,
                                                  const std::string& path) {
  const auto& value = *required_member_ptr(object, key, path);
  if (!value.is_array() || value.empty()) {
    reject(AfeMpkV2DecodeErrorCode::InvalidField, path + "." + key,
           "expected a non-empty host tensor type array");
  }
  std::vector<HostTensorTypeSpec> result;
  result.reserve(value.size());
  for (std::size_t index = 0; index < value.size(); ++index) {
    const auto item_path = path + "." + key + "[" + std::to_string(index) + "]";
    const auto& item = value[index];
    require_exact_keys(item, {"scalar", "shape"}, item_path);
    HostTensorTypeSpec type{
        required_string(item, "scalar", item_path),
        shape(*required_member_ptr(item, "shape", item_path), item_path + ".shape")};
    if (!element_width(type.scalar).has_value()) {
      reject(AfeMpkV2DecodeErrorCode::ConfigurationMismatch, item_path + ".scalar",
             "host tensor scalar type has no exact registered DLPack mapping");
    }
    result.push_back(std::move(type));
  }
  return result;
}

std::vector<std::string> non_empty_strings(const Json& object, const char* key,
                                           const std::string& path) {
  const auto& value = *required_member_ptr(object, key, path);
  if (!value.is_array() || value.empty()) {
    reject(AfeMpkV2DecodeErrorCode::InvalidField, path + "." + key,
           "expected a non-empty string array");
  }
  std::vector<std::string> result;
  result.reserve(value.size());
  std::unordered_set<std::string> unique;
  for (std::size_t index = 0; index < value.size(); ++index) {
    const auto item_path = path + "." + key + "[" + std::to_string(index) + "]";
    if (!value[index].is_string() || value[index].get_ref<const std::string&>().empty()) {
      reject(AfeMpkV2DecodeErrorCode::InvalidField, item_path, "expected a non-empty string");
    }
    auto item = value[index].get<std::string>();
    if (!unique.emplace(item).second) {
      reject(AfeMpkV2DecodeErrorCode::ConfigurationMismatch, item_path,
             "host input name is duplicated");
    }
    result.push_back(std::move(item));
  }
  return result;
}

std::vector<QuantizationSpec> quantization(const Json& object, const char* key,
                                           const std::string& path) {
  const auto& value = *required_member_ptr(object, key, path);
  if (!value.is_array() || value.empty()) {
    reject(AfeMpkV2DecodeErrorCode::InvalidField, path + "." + key,
           "expected a non-empty channel-parameter array");
  }
  std::vector<QuantizationSpec> result;
  result.reserve(value.size());
  for (std::size_t index = 0; index < value.size(); ++index) {
    const auto& pair = value[index];
    const std::string pair_path = path + "." + key + "[" + std::to_string(index) + "]";
    if (!pair.is_array() || pair.size() != 2U || !pair[0].is_number() ||
        !pair[1].is_number_integer()) {
      reject(AfeMpkV2DecodeErrorCode::InvalidField, pair_path,
             "expected [positive scale, integer zero-point]");
    }
    const double scale_value = pair[0].get<double>();
    if (!std::isfinite(scale_value) || scale_value <= 0.0) {
      reject(AfeMpkV2DecodeErrorCode::InvalidField, pair_path + "[0]",
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
    reject(AfeMpkV2DecodeErrorCode::InvalidField, path + "." + key,
           "expected a non-empty node array");
  }
  std::vector<Node> result;
  result.reserve(value.size());
  for (std::size_t index = 0; index < value.size(); ++index) {
    const auto& node = value[index];
    const std::string node_path = path + "." + key + "[" + std::to_string(index) + "]";
    if (!node.is_object()) {
      reject(AfeMpkV2DecodeErrorCode::InvalidField, node_path, "expected a node object");
    }
    result.push_back(
        {required_string(node, "name", node_path),
         positive_u64(*required_member_ptr(node, "size", node_path), node_path + ".size")});
  }
  return result;
}

std::optional<std::uint64_t> element_width(const std::string& dtype) {
  if (dtype == "bool" || dtype == "int8" || dtype == "uint8") {
    return 1U;
  }
  if (dtype == "int16" || dtype == "uint16" || dtype == "float16" || dtype == "bfloat16") {
    return 2U;
  }
  if (dtype == "int32" || dtype == "uint32" || dtype == "float32") {
    return 4U;
  }
  if (dtype == "int64" || dtype == "uint64" || dtype == "float64") {
    return 8U;
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
    if (axis != 0U && stride > std::numeric_limits<std::uint64_t>::max() /
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
    if (dimension <= 0 || elements > std::numeric_limits<std::uint64_t>::max() /
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

std::optional<std::uint64_t> align_up_16(const std::uint64_t value) {
  if (value > std::numeric_limits<std::uint64_t>::max() - 15U) {
    return std::nullopt;
  }
  return (value + 15U) & ~std::uint64_t{15U};
}

bool is_qmla_dense_output_symbol(const std::string& symbol) {
  return symbol.rfind("data.ofm.persistent.afe_mla_output_", 0U) == 0U &&
         symbol.ends_with(".b0");
}

std::optional<std::vector<std::int64_t>> qmla_dense_last_axis_strides(
    const ValueSpec& value, const std::uint64_t physical_extent) {
  if (!value.logical_dtype || !value.logical_shape || value.logical_shape->empty()) {
    return std::nullopt;
  }
  if (value.representation != ValueRepresentation::Dense ||
      value.logical_layout != std::optional<std::string>{"normal"}) {
    return std::nullopt;
  }
  const auto width = element_width(*value.logical_dtype);
  const auto logical_bytes = dense_bytes(*value.logical_shape, *value.logical_dtype);
  if (!width || !logical_bytes || *logical_bytes != value.required_bytes) {
    return std::nullopt;
  }
  const auto last = value.logical_shape->back();
  if (last <= 0 || static_cast<std::uint64_t>(last) >
                       std::numeric_limits<std::uint64_t>::max() / *width) {
    return std::nullopt;
  }
  const auto padded_row = align_up_16(static_cast<std::uint64_t>(last) * *width);
  if (!padded_row) {
    return std::nullopt;
  }
  std::uint64_t prefix = 1U;
  for (std::size_t axis = 0U; axis + 1U < value.logical_shape->size(); ++axis) {
    const auto dimension = value.logical_shape->at(axis);
    if (dimension <= 0 || prefix > std::numeric_limits<std::uint64_t>::max() /
                                      static_cast<std::uint64_t>(dimension)) {
      return std::nullopt;
    }
    prefix *= static_cast<std::uint64_t>(dimension);
  }
  if (prefix > std::numeric_limits<std::uint64_t>::max() / *padded_row ||
      prefix * *padded_row != physical_extent) {
    return std::nullopt;
  }
  std::vector<std::int64_t> strides(value.logical_shape->size(), 0);
  std::uint64_t stride = *width;
  for (std::size_t reverse = value.logical_shape->size(); reverse > 0U; --reverse) {
    const auto axis = reverse - 1U;
    if (axis + 1U == value.logical_shape->size()) {
      stride = *width;
    } else if (axis + 2U == value.logical_shape->size()) {
      stride = *padded_row;
    } else {
      const auto inner = value.logical_shape->at(axis + 1U);
      if (inner <= 0 || stride > std::numeric_limits<std::uint64_t>::max() /
                                    static_cast<std::uint64_t>(inner)) {
        return std::nullopt;
      }
      stride *= static_cast<std::uint64_t>(inner);
    }
    if (stride > static_cast<std::uint64_t>(std::numeric_limits<std::int64_t>::max())) {
      return std::nullopt;
    }
    strides[axis] = static_cast<std::int64_t>(stride);
  }
  return strides;
}

std::optional<std::uint64_t> affine_touched_span(
    const ValueSpec& value, const std::vector<std::int64_t>& strides) {
  const auto width = value.logical_dtype ? element_width(*value.logical_dtype) : std::nullopt;
  if (!width || !value.logical_shape || value.logical_shape->size() != strides.size()) {
    return std::nullopt;
  }
  std::uint64_t span = *width;
  for (std::size_t axis = 0U; axis < strides.size(); ++axis) {
    const auto dimension = value.logical_shape->at(axis);
    if (dimension <= 0 || strides[axis] <= 0 ||
        static_cast<std::uint64_t>(dimension - 1) >
            (std::numeric_limits<std::uint64_t>::max() - span) /
                static_cast<std::uint64_t>(strides[axis])) {
      return std::nullopt;
    }
    span += static_cast<std::uint64_t>(dimension - 1) *
            static_cast<std::uint64_t>(strides[axis]);
  }
  return span;
}

void author_mla_output_storage(ModelExecutionPlanData& data, const std::size_t mla_op_index,
                               const std::size_t output_index, const std::string& symbol,
                               const std::uint64_t physical_extent,
                               std::vector<AfeMpkV2ProofFact>* proof) {
  auto& value = data.values.at(data.ops.at(mla_op_index).outputs.at(output_index));
  const std::string path = "$.plugins[" + std::to_string(mla_op_index) + "].output_nodes[" +
                           std::to_string(output_index) + "]";
  if (physical_extent < value.required_bytes) {
    reject(AfeMpkV2DecodeErrorCode::ValueSizeMismatch, path + ".size",
           "QMLA OFM physical extent is smaller than its logical tensor");
  }
  if (physical_extent == value.required_bytes) {
    return;
  }
  if (!is_qmla_dense_output_symbol(symbol)) {
    reject(AfeMpkV2DecodeErrorCode::ConfigurationMismatch, path,
           "larger MLA output carrier has no registered QMLA dense layout ABI");
  }
  const auto& mla = std::get<MlaOpConfig>(data.ops.at(mla_op_index).config);
  if (output_index >= mla.output_types.size() || !value.logical_dtype ||
      !value.logical_shape || *value.logical_dtype != mla.output_types[output_index].scalar ||
      *value.logical_shape != mla.output_types[output_index].shape) {
    reject(AfeMpkV2DecodeErrorCode::ConfigurationMismatch, path,
           "larger MLA output carrier has no exact typed dense QMLA contract");
  }
  value.representation = ValueRepresentation::Dense;
  value.logical_layout = "normal";
  StorageBinding binding;
  binding.kind = StorageBindingKind::Root;
  binding.carrier_id = value.id;
  binding.access = StorageAccess::ReadWrite;
  if (const auto strides = qmla_dense_last_axis_strides(value, physical_extent)) {
    const auto touched = affine_touched_span(value, *strides);
    if (!touched || *touched > physical_extent) {
      reject(AfeMpkV2DecodeErrorCode::ConfigurationMismatch, path,
             "QMLA row-padded output has an invalid affine span");
    }
    binding.physical_span = *touched;
    binding.stride_bytes = *strides;
    if (proof) {
      proof->push_back({"MLA.OFM[" + std::to_string(output_index) + "].layout",
                        "QMLA SHT_DATA extent exactly proves a 16-byte dense-last-axis pitch"});
    }
  } else {
    reject(AfeMpkV2DecodeErrorCode::ConfigurationMismatch, path,
           "larger MLA output carrier does not match the exact dense-last-axis QMLA ABI");
  }
  value.storage_binding = std::move(binding);
}

void merge_dtype(ValueSpec& value, const std::string& dtype, const std::string& path) {
  if (value.logical_dtype.has_value() && *value.logical_dtype != dtype) {
    reject(AfeMpkV2DecodeErrorCode::ConfigurationMismatch, path,
           "conflicting exact dtype evidence for value '" + value.name + "'");
  }
  value.logical_dtype = dtype;
}

void merge_shape(ValueSpec& value, const TensorShape& value_shape, const std::string& path) {
  if (value.logical_shape.has_value() && *value.logical_shape != value_shape) {
    reject(AfeMpkV2DecodeErrorCode::ConfigurationMismatch, path,
           "conflicting exact shape evidence for value '" + value.name + "'");
  }
  value.logical_shape = value_shape;
}

void merge_layout(ValueSpec& value, const std::string& layout, const std::string& path) {
  if (value.logical_layout.has_value() && *value.logical_layout != layout) {
    reject(AfeMpkV2DecodeErrorCode::ConfigurationMismatch, path,
           "conflicting exact layout evidence for value '" + value.name + "'");
  }
  value.logical_layout = layout;
}

void merge_quantization(ValueSpec& value, const std::vector<QuantizationSpec>& q,
                        const std::string& path) {
  if (!value.quantization.empty()) {
    if (value.quantization.size() != q.size()) {
      reject(AfeMpkV2DecodeErrorCode::ConfigurationMismatch, path,
             "conflicting quantization evidence");
    }
    for (std::size_t index = 0; index < q.size(); ++index) {
      if (value.quantization[index].scale != q[index].scale ||
          value.quantization[index].zero_point != q[index].zero_point) {
        reject(AfeMpkV2DecodeErrorCode::ConfigurationMismatch, path,
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

OpConfig parse_typed_config(const OpKind kind, const std::string_view kernel, const Json& plugin,
                            const Json& config, const Json& params, const std::string& path) {
  switch (kind) {
  case OpKind::Cast: {
    require_exact_keys(params, {"out_dtype", "input_shapes", "output_shapes"},
                       path + ".config_params.params");
    CastOpConfig result{required_string(params, "out_dtype", path + ".params")};
    if (result.output_dtype != "bfloat16" && result.output_dtype != "float32") {
      reject(AfeMpkV2DecodeErrorCode::ConfigurationMismatch,
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
      reject(AfeMpkV2DecodeErrorCode::ConfigurationMismatch, path + ".config_params.params",
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
      reject(AfeMpkV2DecodeErrorCode::ConfigurationMismatch,
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
      reject(AfeMpkV2DecodeErrorCode::InvalidField, path + ".resources", "expected an object");
    }
    const auto quads =
        integer(*required_member_ptr(config, "number_of_quads_to_user", path + ".config_params"),
                path + ".config_params.number_of_quads_to_user");
    if (quads <= 0) {
      reject(AfeMpkV2DecodeErrorCode::InvalidField, path + ".config_params.number_of_quads_to_user",
             "number of MLA quads must be positive");
    }
    MlaOpConfig result;
    result.executable = required_string(resources, "executable", path + ".resources");
    result.number_of_quads = quads;
    if (config.contains("input_types") || config.contains("output_types")) {
      if (!config.contains("input_types") || !config.contains("output_types")) {
        reject(AfeMpkV2DecodeErrorCode::MissingRequiredField, path + ".config_params",
               "typed MLA grammar requires both input_types and output_types");
      }
      result.input_types = host_tensor_types(config, "input_types", path + ".config_params");
      result.output_types = host_tensor_types(config, "output_types", path + ".config_params");
    }
    return result;
  }
  case OpKind::Unpack: {
    require_exact_keys(params, {"tensor_types", "tensor_shapes", "input_shapes", "output_shapes"},
                       path + ".config_params.params");
    const auto& types = *required_member_ptr(params, "tensor_types", path + ".params");
    if (!types.is_array() || types.empty()) {
      reject(AfeMpkV2DecodeErrorCode::InvalidField, path + ".params.tensor_types",
             "expected a non-empty tensor type array");
    }
    std::vector<std::string> tensor_types;
    tensor_types.reserve(types.size());
    for (std::size_t index = 0; index < types.size(); ++index) {
      if (!types[index].is_string() || types[index].get_ref<const std::string&>().empty()) {
        reject(AfeMpkV2DecodeErrorCode::InvalidField,
               path + ".params.tensor_types[" + std::to_string(index) + "]",
               "expected a non-empty tensor dtype");
      }
      const std::string tensor_type = types[index].get<std::string>();
      if (tensor_type != "int8" && tensor_type != "bfloat16") {
        reject(AfeMpkV2DecodeErrorCode::ConfigurationMismatch,
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
      reject(AfeMpkV2DecodeErrorCode::ConfigurationMismatch, path + ".config_params.params",
             "slice ranks disagree");
    }
    for (std::size_t index = 0; index < result.begin.size(); ++index) {
      if (result.begin[index] > result.end[index] ||
          result.end[index] > result.input_shape[index] ||
          result.end[index] - result.begin[index] != result.output_shape[index]) {
        reject(AfeMpkV2DecodeErrorCode::ConfigurationMismatch, path + ".config_params.params",
               "slice bounds do not prove output shape");
      }
    }
    return result;
  }
  case OpKind::Reshape: {
    if (kernel == "batch_flatten_transform") {
      require_exact_keys(params, {"input_shapes", "output_shapes"},
                         path + ".config_params.params");
      auto output_shapes = shapes(params, "output_shapes", path + ".params", true);
      if (output_shapes.size() != 1U) {
        reject(AfeMpkV2DecodeErrorCode::ConfigurationMismatch,
               path + ".config_params.params.output_shapes",
               "batch flatten requires exactly one output shape");
      }
      return ReshapeOpConfig{std::move(output_shapes.front())};
    }
    if (kernel != "reshape_transform") {
      reject(AfeMpkV2DecodeErrorCode::UnsupportedKernel, path + ".config_params.kernel",
             "reshape operation has no exact typed grammar for kernel '" +
                 std::string(kernel) + "'");
    }
    require_exact_keys(params, {"newshape", "input_shapes", "output_shapes"},
                       path + ".config_params.params");
    return ReshapeOpConfig{shape(*required_member_ptr(params, "newshape", path + ".params"),
                                 path + ".params.newshape")};
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
      reject(AfeMpkV2DecodeErrorCode::ConfigurationMismatch,
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
      reject(AfeMpkV2DecodeErrorCode::ConfigurationMismatch,
             path + ".config_params.params.input_data_type",
             "legacy dequantization input must be exact int8");
    }
    return result;
  }
  case OpKind::HostTvm: {
    require_exact_keys(config, {"input_names", "input_types", "output_types"},
                       path + ".config_params");
    const auto& resources = *required_member_ptr(plugin, "resources", path);
    require_exact_keys(resources, {"executable"}, path + ".resources");
    return HostTvmOpConfig{required_string(resources, "executable", path + ".resources"),
                           non_empty_strings(config, "input_names", path + ".config_params"),
                           host_tensor_types(config, "input_types", path + ".config_params"),
                           host_tensor_types(config, "output_types", path + ".config_params"),
                           {}};
  }
  case OpKind::PassThrough:
    require_exact_keys(params, {}, path + ".config_params.params");
    return PassThroughOpConfig{};
  }
  reject(AfeMpkV2DecodeErrorCode::ConfigurationMismatch, path, "unhandled exact operation kind");
}

void apply_input_evidence(ModelExecutionPlanData& data, const OpSpec& op, const std::string& path) {
  if (!op.input_shapes.empty()) {
    if (op.input_shapes.size() != op.inputs.size()) {
      reject(AfeMpkV2DecodeErrorCode::ConfigurationMismatch,
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
      reject(AfeMpkV2DecodeErrorCode::ConfigurationMismatch, path,
             "legacy cast_transform has an unsupported exact dtype transition");
    }
    break;
  }
  case OpKind::Mla: {
    const auto& config = std::get<MlaOpConfig>(op.config);
    for (std::size_t index = 0; index < config.input_types.size(); ++index) {
      merge_dtype(data.values[op.inputs[index]], config.input_types[index].scalar, path);
      merge_shape(data.values[op.inputs[index]], config.input_types[index].shape, path);
    }
    break;
  }
  case OpKind::Quantize:
    merge_dtype(data.values[op.inputs.front()], "float32", path);
    // Graph 222's registered dense quantize ABI (and graph 226's dense
    // ingress) consumes canonical HWC geometry.  Seed that exact target ABI
    // evidence here so standalone Quantize does not depend on a following
    // Tessellate operation to recover its semantic axes.
    merge_layout(data.values[op.inputs.front()], "HWC", path);
    break;
  case OpKind::Tessellate:
    merge_dtype(data.values[op.inputs.front()], std::get<TessellateOpConfig>(op.config).frame_type,
                path);
    // Graph 2's registered tensor-transform ABI consumes canonical HWC
    // geometry. This is target ABI evidence, not a shape/name heuristic.
    merge_layout(data.values[op.inputs.front()], "HWC", path);
    break;
  case OpKind::Detessellate:
    merge_dtype(data.values[op.inputs.front()],
                std::get<DetessellateOpConfig>(op.config).frame_type, path);
    // Graph 3 is the inverse of the same canonical HWC transform. Retain the
    // semantic axes even though its input bytes are backend-native/tiled.
    merge_layout(data.values[op.inputs.front()], "HWC", path);
    break;
  case OpKind::Dequantize: {
    const auto& config = std::get<DequantizeOpConfig>(op.config);
    merge_dtype(data.values[op.inputs.front()], config.input_dtype, path);
    merge_quantization(data.values[op.inputs.front()], config.channel_params, path);
    break;
  }
  case OpKind::HostTvm: {
    const auto& config = std::get<HostTvmOpConfig>(op.config);
    for (std::size_t index = 0; index < op.inputs.size(); ++index) {
      merge_dtype(data.values[op.inputs[index]], config.input_types[index].scalar, path);
      merge_shape(data.values[op.inputs[index]], config.input_types[index].shape, path);
    }
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
    value.logical_layout = data.values[op.inputs.front()].logical_layout;
    break;
  case OpKind::Quantize: {
    const auto& config = std::get<QuantizeOpConfig>(op.config);
    value.logical_dtype = config.output_dtype;
    value.logical_layout = data.values[op.inputs.front()].logical_layout;
    value.quantization = config.channel_params;
    break;
  }
  case OpKind::Tessellate:
    // AFE's tessellate output_shapes field describes the flattened packed
    // carrier (for example [1, 1228800]), not a new logical tensor geometry.
    // Graph 2/226 preserves the exact semantic frame authored on its input;
    // keep that N/H/W/C shape on the materialized ValueSpec while the output
    // node byte extent and Tessellated representation remain the independent
    // physical-storage authority.
    value.logical_shape = op.input_shapes.front();
    value.logical_dtype = std::get<TessellateOpConfig>(op.config).frame_type;
    value.logical_layout = "HWC";
    value.representation = ValueRepresentation::Tessellated;
    break;
  case OpKind::Pack:
    value.representation = ValueRepresentation::Packed;
    break;
  case OpKind::Mla: {
    const auto& config = std::get<MlaOpConfig>(op.config);
    if (!config.output_types.empty()) {
      value.logical_dtype = config.output_types.at(output_index).scalar;
      value.logical_shape = config.output_types.at(output_index).shape;
    }
    // Neither output count nor ELF topology proves a logical representation.
    // Exact physical-layout admission may refine a typed QMLA output later.
    value.representation = ValueRepresentation::BackendNative;
    break;
  }
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
      value.logical_layout = input.logical_layout;
    }
    break;
  case OpKind::Reshape: {
    const auto& input = data.values[op.inputs.front()];
    value.logical_shape = std::get<ReshapeOpConfig>(op.config).new_shape;
    value.logical_dtype = input.logical_dtype;
    value.quantization = input.quantization;
    value.representation = input.representation;
    break;
  }
  case OpKind::Detessellate:
    value.logical_dtype = std::get<DetessellateOpConfig>(op.config).frame_type;
    value.logical_layout = "HWC";
    break;
  case OpKind::Dequantize:
    value.logical_dtype = "float32";
    value.logical_layout = data.values[op.inputs.front()].logical_layout;
    break;
  case OpKind::HostTvm: {
    const auto& type = std::get<HostTvmOpConfig>(op.config).output_types.at(output_index);
    value.logical_dtype = type.scalar;
    value.logical_shape = type.shape;
    break;
  }
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
  // Slice and PassThrough preserve dtype/quantization. Cast, Quantize, and
  // Dequantize preserve semantic axes only when both exact endpoint shapes
  // prove that the edge is shape-preserving. Later operations often provide
  // the only exact evidence, so converge these facts in both directions.
  bool changed = true;
  while (changed) {
    changed = false;
    for (const auto& op : data.ops) {
      if (op.kind == OpKind::Cast || op.kind == OpKind::Quantize ||
          op.kind == OpKind::Dequantize) {
        if (op.inputs.size() != 1U || op.outputs.size() != 1U) {
          continue;
        }
        auto& input = data.values[op.inputs.front()];
        auto& output = data.values[op.outputs.front()];
        if (!input.logical_shape.has_value() || !output.logical_shape.has_value()) {
          continue;
        }
        if (*input.logical_shape != *output.logical_shape) {
          reject(AfeMpkV2DecodeErrorCode::ConfigurationMismatch,
                 "$.ops[" + std::to_string(op.id) + "]",
                 "shape-preserving transform '" + op.name +
                     "' has contradictory exact endpoint shapes");
        }
        if (input.logical_layout.has_value() && output.logical_layout.has_value() &&
            *input.logical_layout != *output.logical_layout) {
          reject(AfeMpkV2DecodeErrorCode::ConfigurationMismatch,
                 "$.ops[" + std::to_string(op.id) + "]",
                 "shape-preserving transform '" + op.name +
                     "' has contradictory exact endpoint layouts");
        }
        if (!input.logical_layout.has_value() && output.logical_layout.has_value()) {
          input.logical_layout = output.logical_layout;
          changed = true;
        }
        if (!output.logical_layout.has_value() && input.logical_layout.has_value()) {
          output.logical_layout = input.logical_layout;
          changed = true;
        }
        continue;
      }
      if (op.kind != OpKind::Slice && op.kind != OpKind::Reshape &&
          op.kind != OpKind::PassThrough) {
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
        // Slice and publication retain axis meaning. Reshape is deliberately
        // excluded: a byte-preserving reshape does not by itself prove how a
        // changed rank maps to semantic axes.
        if (op.kind != OpKind::Reshape) {
          if (!input.logical_layout.has_value() && output.logical_layout.has_value()) {
            input.logical_layout = output.logical_layout;
            changed = true;
          }
          if (!output.logical_layout.has_value() && input.logical_layout.has_value()) {
            output.logical_layout = input.logical_layout;
            changed = true;
          }
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
      reject(AfeMpkV2DecodeErrorCode::ConfigurationMismatch,
             "$.values[" + std::to_string(value.id) + "]",
             "unsupported or overflowing dense byte equation for value '" + value.name + "'");
    }
    if (*expected != value.required_bytes) {
      reject(
          AfeMpkV2DecodeErrorCode::ValueSizeMismatch, "$.values[" + std::to_string(value.id) + "]",
          "dense byte equation disagrees for value '" + value.name + "': expected " +
              std::to_string(*expected) + ", MPK declares " + std::to_string(value.required_bytes));
    }
  }
}

void lower_read_expressions(ModelExecutionPlanData& data, std::vector<AfeMpkV2ProofFact>* proof) {
  for (const auto& op : data.ops) {
    if (op.kind == OpKind::Unpack) {
      if (op.inputs.size() != 1U || op.outputs.empty()) {
        reject(AfeMpkV2DecodeErrorCode::ConfigurationMismatch,
               "$.ops[" + std::to_string(op.id) + "]",
               "unpack read expression requires one carrier and at least one view");
      }
      const auto& source = data.values[op.inputs.front()];
      if (source.read_expression.has_value()) {
        reject(AfeMpkV2DecodeErrorCode::ConfigurationMismatch,
               "$.ops[" + std::to_string(op.id) + "]",
               "unpack carrier must already be a materialized root buffer");
      }
      const auto& config = std::get<UnpackOpConfig>(op.config);
      if (config.tensor_types.size() != op.outputs.size() ||
          config.tensor_shapes.size() != op.outputs.size()) {
        reject(AfeMpkV2DecodeErrorCode::ConfigurationMismatch,
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
          reject(AfeMpkV2DecodeErrorCode::ValueSizeMismatch, "$.ops[" + std::to_string(op.id) + "]",
                 "unpack view does not fit its exact packed carrier");
        }
        output.read_expression = ReadExpression{source.id, offset, *strides};
        if (proof) {
          proof->push_back({"read[" + std::to_string(output.id) + "]",
                            "unpack lowers to root value '" + source.name + "' + " +
                                std::to_string(offset) +
                                " bytes; no runtime operation is scheduled"});
        }
        offset += output.required_bytes;
      }
      if (offset != source.required_bytes) {
        reject(AfeMpkV2DecodeErrorCode::ValueSizeMismatch, "$.ops[" + std::to_string(op.id) + "]",
               "unpack views do not partition the exact packed carrier");
      }
      continue;
    }

    if (op.kind == OpKind::Slice) {
      if (op.inputs.size() != 1U || op.outputs.size() != 1U) {
        reject(AfeMpkV2DecodeErrorCode::ConfigurationMismatch,
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
        const auto dense_strides =
            width.has_value() ? contiguous_stride_bytes(config.input_shape, *width) : std::nullopt;
        if (!dense_strides.has_value()) {
          reject(AfeMpkV2DecodeErrorCode::ConfigurationMismatch,
                 "$.ops[" + std::to_string(op.id) + "]",
                 "slice input has no exact dense carrier stride");
        }
        strides = *dense_strides;
      }
      if (strides.size() != config.begin.size()) {
        reject(AfeMpkV2DecodeErrorCode::ConfigurationMismatch,
               "$.ops[" + std::to_string(op.id) + "]",
               "slice input view rank disagrees with its byte strides");
      }
      for (std::size_t axis = 0; axis < config.begin.size(); ++axis) {
        const auto begin = static_cast<std::uint64_t>(config.begin[axis]);
        const auto stride = static_cast<std::uint64_t>(strides[axis]);
        if (begin != 0U && stride > std::numeric_limits<std::uint64_t>::max() / begin) {
          reject(AfeMpkV2DecodeErrorCode::ConfigurationMismatch,
                 "$.ops[" + std::to_string(op.id) + "]", "slice read-expression offset overflows");
        }
        const auto axis_offset = begin * stride;
        if (base_offset > std::numeric_limits<std::uint64_t>::max() - axis_offset) {
          reject(AfeMpkV2DecodeErrorCode::ConfigurationMismatch,
                 "$.ops[" + std::to_string(op.id) + "]", "slice read-expression offset overflows");
        }
        base_offset += axis_offset;
      }
      output.read_expression = ReadExpression{root, base_offset, std::move(strides)};
      if (proof) {
        proof->push_back({"read[" + std::to_string(output.id) + "]",
                          "slice composes to root value '" + data.values[root].name + "' + " +
                              std::to_string(base_offset) +
                              " bytes; no runtime operation is scheduled"});
      }
      continue;
    }

    if (op.kind == OpKind::Reshape) {
      if (op.inputs.size() != 1U || op.outputs.size() != 1U) {
        reject(AfeMpkV2DecodeErrorCode::ConfigurationMismatch,
               "$.ops[" + std::to_string(op.id) + "]",
               "reshape address view requires one input and one output");
      }
      const auto& input = data.values[op.inputs.front()];
      auto& output = data.values[op.outputs.front()];
      if (input.required_bytes != output.required_bytes || !output.logical_shape.has_value()) {
        reject(AfeMpkV2DecodeErrorCode::ValueSizeMismatch, "$.ops[" + std::to_string(op.id) + "]",
               "reshape does not preserve the exact byte extent");
      }
      const auto width = exact_element_width(output.required_bytes, *output.logical_shape);
      const auto strides =
          width ? contiguous_stride_bytes(*output.logical_shape, *width) : std::nullopt;
      if (!strides) {
        reject(AfeMpkV2DecodeErrorCode::ConfigurationMismatch,
               "$.ops[" + std::to_string(op.id) + "]",
               "reshape output has no exact contiguous byte equation");
      }
      if (input.read_expression.has_value()) {
        if (!input.logical_shape.has_value()) {
          reject(AfeMpkV2DecodeErrorCode::ConfigurationMismatch,
                 "$.ops[" + std::to_string(op.id) + "]",
                 "reshape input view has no exact logical shape");
        }
        const auto input_width = exact_element_width(input.required_bytes, *input.logical_shape);
        const auto input_dense = input_width
                                     ? contiguous_stride_bytes(*input.logical_shape, *input_width)
                                     : std::nullopt;
        if (!input_dense || input.read_expression->stride_bytes != *input_dense) {
          reject(AfeMpkV2DecodeErrorCode::ConfigurationMismatch,
                 "$.ops[" + std::to_string(op.id) + "]",
                 "reshape cannot reinterpret a non-contiguous input view");
        }
      }
      const auto root = input.read_expression ? input.read_expression->source_value_id : input.id;
      const auto offset = input.read_expression ? input.read_expression->byte_offset : 0U;
      output.read_expression = ReadExpression{root, offset, *strides};
      if (proof) {
        proof->push_back(
            {"read[" + std::to_string(output.id) + "]",
             "reshape lowers to an exact address view; no runtime operation is scheduled"});
      }
      continue;
    }

    if (op.kind == OpKind::HostTvm) {
      const auto& host = std::get<HostTvmOpConfig>(op.config);
      for (std::size_t output_index = 0; output_index < op.outputs.size(); ++output_index) {
        const auto input_index = host.output_alias_input[output_index];
        if (input_index < 0) {
          continue;
        }
        const auto& input = data.values[op.inputs[static_cast<std::size_t>(input_index)]];
        auto& output = data.values[op.outputs[output_index]];
        const auto width =
            exact_element_width(output.required_bytes, host.output_types[output_index].shape);
        const auto strides =
            width ? contiguous_stride_bytes(host.output_types[output_index].shape, *width)
                  : std::nullopt;
        if (!strides || input.required_bytes != output.required_bytes) {
          reject(AfeMpkV2DecodeErrorCode::ValueSizeMismatch, "$.ops[" + std::to_string(op.id) + "]",
                 "TVM __nop output does not preserve an exact dense byte extent");
        }
        const auto root = input.read_expression ? input.read_expression->source_value_id : input.id;
        const auto offset = input.read_expression ? input.read_expression->byte_offset : 0U;
        output.read_expression = ReadExpression{root, offset, *strides};
        if (proof) {
          proof->push_back({"read[" + std::to_string(output.id) + "]",
                            "embedded TVM GraphExecutor __nop + shared storage_id lowers to an "
                            "exact address view"});
        }
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
        } else if (input.storage_binding.has_value() &&
                   input.storage_binding->physical_span > input.required_bytes &&
                   input.logical_shape.has_value() && input.logical_dtype.has_value()) {
          auto strides = input.storage_binding->stride_bytes;
          if (strides.empty()) {
            const auto width = element_width(*input.logical_dtype);
            const auto dense = width ? contiguous_stride_bytes(*input.logical_shape, *width)
                                     : std::nullopt;
            if (!dense) {
              reject(AfeMpkV2DecodeErrorCode::ConfigurationMismatch,
                     "$.ops[" + std::to_string(op.id) + "]",
                     "padded MLA publication has no exact logical stride");
            }
            strides = *dense;
          }
          data.values[op.outputs[index]].read_expression =
              ReadExpression{input.id, 0U, std::move(strides)};
        }
      }
    }
  }
}

void validate_view_consumers(const ModelExecutionPlanData& data) {
  for (const auto& value : data.values) {
    if (!value.read_expression.has_value()) {
      continue;
    }
    bool contiguous = false;
    if (value.logical_shape.has_value()) {
      const auto width = exact_element_width(value.required_bytes, *value.logical_shape);
      const auto dense = width ? contiguous_stride_bytes(*value.logical_shape, *width)
                               : std::nullopt;
      contiguous = dense && value.read_expression->stride_bytes == *dense;
    }
    if (contiguous) {
      continue;
    }
    for (const auto& consumer : data.ops) {
      if (std::find(consumer.inputs.begin(), consumer.inputs.end(), value.id) ==
          consumer.inputs.end()) {
        continue;
      }
      // These two retained graph ABIs have exact positive-stride descriptor
      // domains. Dense-only graph 222/2, MLA ports, and GraphExecutor inputs do
      // not acquire a strided contract from a stock MPK or ELF filename.
      if (consumer.kind == OpKind::Cast || consumer.kind == OpKind::Dequantize ||
          consumer.kind == OpKind::Slice) {
        continue;
      }
      reject(AfeMpkV2DecodeErrorCode::ConfigurationMismatch,
             "$.values[" + std::to_string(value.id) + "]",
             "non-contiguous view '" + value.name + "' is consumed by dense-only operation '" +
                 consumer.name + "'");
    }
  }
}

AfeMpkV2DecodeResult
decode_impl(const std::string_view text,
            const std::span<const MlaStageExecutableEvidence> executable_evidence,
            const std::span<const HostTvmExecutableEvidence> host_evidence,
            const std::string& source) {
  AfeMpkV2DecodeResult result;
  try {
    Json root = Json::parse(text.begin(), text.end(), nullptr, false);
    if (root.is_discarded() || !root.is_object()) {
      reject(AfeMpkV2DecodeErrorCode::InvalidJson, "$", "manifest is not valid JSON object syntax");
    }
    if (root.contains("execution_contract")) {
      reject(AfeMpkV2DecodeErrorCode::InvalidField, "$.execution_contract",
             "stock-AFE admission does not accept a second execution authority");
    }
    const std::string manifest_sha256 = sha256_text(text);
    if (manifest_sha256.empty()) {
      reject(AfeMpkV2DecodeErrorCode::IoError, "$", "cannot hash the exact MPK manifest bytes");
    }

    ModelExecutionPlanData data;
    data.contract_version = required_string(root, "model_sdk_version", "$");
    if (data.contract_version != "2.0.0" && data.contract_version != "2.1.0") {
      reject(AfeMpkV2DecodeErrorCode::UnsupportedContractVersion, "$.model_sdk_version",
             "strict decoder supports only exact contract versions 2.0.0 and 2.1.0");
    }
    result.proof.push_back({"contract.version", "MPK $.model_sdk_version exactly equals '" +
                                                    data.contract_version + "'"});

    const auto& input_array = *required_member_ptr(root, "input_nodes", "$");
    if (!input_array.is_array() || input_array.empty()) {
      reject(AfeMpkV2DecodeErrorCode::InvalidField, "$.input_nodes",
             "expected at least one model input");
    }

    std::unordered_map<std::string, ValueId> value_by_name;
    for (std::size_t index = 0; index < input_array.size(); ++index) {
      const auto& input = input_array[index];
      const std::string path = "$.input_nodes[" + std::to_string(index) + "]";
      if (!input.is_object()) {
        reject(AfeMpkV2DecodeErrorCode::InvalidField, path, "expected a node object");
      }
      ValueSpec value;
      value.id = static_cast<ValueId>(data.values.size());
      value.name = required_string(input, "name", path);
      value.required_bytes =
          positive_u64(*required_member_ptr(input, "size", path), path + ".size");
      if (!value_by_name.emplace(value.name, value.id).second) {
        reject(AfeMpkV2DecodeErrorCode::DuplicateProducer, path + ".name",
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
      reject(AfeMpkV2DecodeErrorCode::InvalidField, "$.plugins",
             "expected a non-empty plugin array");
    }
    std::vector<PluginRef> ordered;
    ordered.reserve(plugins.size());
    std::unordered_set<std::uint64_t> sequences;
    for (std::size_t index = 0; index < plugins.size(); ++index) {
      const auto& plugin = plugins[index];
      const std::string path = "$.plugins[" + std::to_string(index) + "]";
      if (!plugin.is_object()) {
        reject(AfeMpkV2DecodeErrorCode::InvalidField, path, "expected a plugin object");
      }
      const auto sequence =
          positive_u64(*required_member_ptr(plugin, "sequence", path), path + ".sequence");
      if (!sequences.emplace(sequence).second) {
        reject(AfeMpkV2DecodeErrorCode::DuplicateSequence, path + ".sequence",
               "plugin sequence is duplicated");
      }
      ordered.push_back({sequence, &plugin, index});
    }
    std::sort(ordered.begin(), ordered.end(), [](const PluginRef& left, const PluginRef& right) {
      return left.sequence < right.sequence;
    });
    for (std::size_t index = 0; index < ordered.size(); ++index) {
      if (ordered[index].sequence != index + 1U) {
        reject(AfeMpkV2DecodeErrorCode::InvalidField,
               "$.plugins[" + std::to_string(ordered[index].manifest_index) + "].sequence",
               "plugin sequences must be contiguous starting at one");
      }
    }

    std::size_t mla_count = 0U;
    std::size_t pass_count = 0U;
    std::vector<std::size_t> mla_op_indices;
    std::vector<std::size_t> host_op_indices;
    std::size_t pass_op_index = 0U;
    std::unordered_set<std::string> operation_names;
    for (const auto& ordered_plugin : ordered) {
      const Json& plugin = *ordered_plugin.plugin;
      const std::string path = "$.plugins[" + std::to_string(ordered_plugin.manifest_index) + "]";
      const std::string name = required_string(plugin, "name", path);
      if (!operation_names.emplace(name).second) {
        reject(AfeMpkV2DecodeErrorCode::InvalidField, path + ".name",
               "operation name is duplicated");
      }
      const std::string processor = required_string(plugin, "processor", path);
      if (required_string(plugin, "type", path) != "sgpProcess") {
        reject(AfeMpkV2DecodeErrorCode::InvalidField, path + ".type",
               "legacy static-plan plugin must be exactly 'sgpProcess'");
      }
      const auto& config = *required_member_ptr(plugin, "config_params", path);
      if (!config.is_object()) {
        reject(AfeMpkV2DecodeErrorCode::InvalidField, path + ".config_params",
               "expected an object");
      }
      if (processor == "MLA") {
        const bool typed_mla = config.contains("input_types") || config.contains("output_types");
        if (typed_mla) {
          require_exact_keys(config,
                             {"desired_batch_size", "actual_batch_size",
                              "number_of_quads_to_user", "input_types", "output_types"},
                             path + ".config_params");
        } else {
          require_exact_keys(
              config, {"desired_batch_size", "actual_batch_size", "number_of_quads_to_user"},
              path + ".config_params");
        }
      } else if (processor == "A65") {
        if (data.contract_version != "2.1.0") {
          reject(AfeMpkV2DecodeErrorCode::UnsupportedHostModule, path + ".config_params",
                 "A65 host modules require the exact typed 2.1.0 contract");
        }
        require_exact_keys(config, {"input_names", "input_types", "output_types"},
                           path + ".config_params");
      } else {
        require_exact_keys(config, {"desired_batch_size", "actual_batch_size", "kernel", "params"},
                           path + ".config_params");
      }
      std::uint64_t actual_batch = 1U;
      if (processor != "A65") {
        const auto desired_batch = positive_u64(
            *required_member_ptr(config, "desired_batch_size", path + ".config_params"),
            path + ".config_params.desired_batch_size");
        actual_batch =
            positive_u64(*required_member_ptr(config, "actual_batch_size", path + ".config_params"),
                         path + ".config_params.actual_batch_size");
        if (desired_batch != actual_batch) {
          reject(AfeMpkV2DecodeErrorCode::ConfigurationMismatch, path + ".config_params",
                 "desired and actual batch sizes disagree");
        }
      }

      std::string kernel;
      if (config.contains("kernel")) {
        if (!config.at("kernel").is_string()) {
          reject(AfeMpkV2DecodeErrorCode::InvalidField, path + ".config_params.kernel",
                 "expected a string");
        }
        kernel = config.at("kernel").get<std::string>();
      } else if (processor != "MLA" && processor != "A65") {
        reject(AfeMpkV2DecodeErrorCode::MissingRequiredField, path + ".config_params.kernel",
               "non-MLA plugin has no exact kernel token");
      }
      const auto descriptor = lookup_exact_kernel(data.contract_version, processor, kernel);
      if (!descriptor.has_value()) {
        reject(AfeMpkV2DecodeErrorCode::UnsupportedKernel, path + ".config_params.kernel",
               "no exact registry entry for ('" + data.contract_version + "', '" + processor +
                   "', '" + kernel + "')");
      }

      const auto input_nodes = nodes(plugin, "input_nodes", path);
      const auto output_nodes = nodes(plugin, "output_nodes", path);
      if (!exact_kernel_arity_is_valid(*descriptor, input_nodes.size(), output_nodes.size())) {
        reject(AfeMpkV2DecodeErrorCode::InvalidKernelArity, path,
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
          reject(AfeMpkV2DecodeErrorCode::MissingProducer,
                 path + ".input_nodes[" + std::to_string(index) + "].name",
                 "no earlier exact producer for tensor '" + input_nodes[index].name + "'");
        }
        if (data.values[found->second].required_bytes != input_nodes[index].bytes) {
          reject(AfeMpkV2DecodeErrorCode::ValueSizeMismatch,
                 path + ".input_nodes[" + std::to_string(index) + "].size",
                 "consumer byte extent disagrees with exact producer for tensor '" +
                     input_nodes[index].name + "'");
        }
        op.inputs.push_back(found->second);
      }

      Json empty_params = Json::object();
      const Json* params = &empty_params;
      if (op.kind != OpKind::Mla && op.kind != OpKind::HostTvm) {
        const auto& params_member = *required_member_ptr(config, "params", path + ".config_params");
        if (!params_member.is_object()) {
          reject(AfeMpkV2DecodeErrorCode::InvalidField, path + ".config_params.params",
                 "expected an object");
        }
        params = &params_member;
      }
      op.config = parse_typed_config(op.kind, op.kernel, plugin, config, *params, path);
      if (op.kind == OpKind::Mla) {
        const auto& mla = std::get<MlaOpConfig>(op.config);
        for (const auto& type : mla.input_types) {
          op.input_shapes.push_back(type.shape);
        }
        for (const auto& type : mla.output_types) {
          op.output_shapes.push_back(type.shape);
        }
      } else if (op.kind == OpKind::HostTvm) {
        const auto& host = std::get<HostTvmOpConfig>(op.config);
        for (const auto& type : host.input_types) {
          op.input_shapes.push_back(type.shape);
        }
        for (const auto& type : host.output_types) {
          op.output_shapes.push_back(type.shape);
        }
      } else {
        const bool shape_lists_required = op.kind != OpKind::Mla && op.kind != OpKind::PassThrough;
        op.input_shapes =
            shapes(*params, "input_shapes", path + ".config_params.params", shape_lists_required);
        op.output_shapes =
            shapes(*params, "output_shapes", path + ".config_params.params", shape_lists_required);
      }
      if ((!op.input_shapes.empty() && op.input_shapes.size() != input_nodes.size()) ||
          (!op.output_shapes.empty() && op.output_shapes.size() != output_nodes.size())) {
        reject(AfeMpkV2DecodeErrorCode::ConfigurationMismatch, path + ".config_params.params",
               "typed shape-list arity disagrees with MPK node arity");
      }
      if (op.kind == OpKind::HostTvm) {
        const auto& host = std::get<HostTvmOpConfig>(op.config);
        if (host.input_names.size() != input_nodes.size() ||
            host.input_types.size() != input_nodes.size() ||
            host.output_types.size() != output_nodes.size()) {
          reject(AfeMpkV2DecodeErrorCode::ConfigurationMismatch, path + ".config_params",
                 "host-module typed port arity disagrees with MPK nodes");
        }
        for (std::size_t index = 0; index < input_nodes.size(); ++index) {
          const auto bytes =
              dense_bytes(host.input_types[index].shape, host.input_types[index].scalar);
          if (!bytes || *bytes != input_nodes[index].bytes) {
            reject(AfeMpkV2DecodeErrorCode::ValueSizeMismatch,
                   path + ".config_params.input_types[" + std::to_string(index) + "]",
                   "host input dense byte equation disagrees with its MPK node");
          }
        }
        for (std::size_t index = 0; index < output_nodes.size(); ++index) {
          const auto bytes =
              dense_bytes(host.output_types[index].shape, host.output_types[index].scalar);
          if (!bytes || *bytes != output_nodes[index].bytes) {
            reject(AfeMpkV2DecodeErrorCode::ValueSizeMismatch,
                   path + ".config_params.output_types[" + std::to_string(index) + "]",
                   "host output dense byte equation disagrees with its MPK node");
          }
        }
      }
      if (op.kind == OpKind::Mla) {
        const auto& mla = std::get<MlaOpConfig>(op.config);
        if ((!mla.input_types.empty() && mla.input_types.size() != input_nodes.size()) ||
            (!mla.output_types.empty() && mla.output_types.size() != output_nodes.size())) {
          reject(AfeMpkV2DecodeErrorCode::ConfigurationMismatch, path + ".config_params",
                 "typed MLA port arity disagrees with MPK nodes");
        }
        for (std::size_t index = 0; index < mla.input_types.size(); ++index) {
          const auto bytes = dense_bytes(mla.input_types[index].shape, mla.input_types[index].scalar);
          if (!bytes || *bytes != input_nodes[index].bytes) {
            reject(AfeMpkV2DecodeErrorCode::ValueSizeMismatch,
                   path + ".config_params.input_types[" + std::to_string(index) + "]",
                   "typed MLA input byte equation disagrees with its MPK node");
          }
        }
        for (std::size_t index = 0; index < mla.output_types.size(); ++index) {
          const auto bytes =
              dense_bytes(mla.output_types[index].shape, mla.output_types[index].scalar);
          if (!bytes || *bytes != output_nodes[index].bytes) {
            reject(AfeMpkV2DecodeErrorCode::ValueSizeMismatch,
                   path + ".config_params.output_types[" + std::to_string(index) + "]",
                   "typed MLA output byte equation disagrees with its MPK node");
          }
        }
      }
      if (op.kind == OpKind::Pack) {
        if (actual_batch != 1U) {
          reject(AfeMpkV2DecodeErrorCode::ConfigurationMismatch,
                 path + ".config_params.actual_batch_size",
                 "legacy Pack placement is supported only for exact batch one");
        }
        auto& pack = std::get<PackOpConfig>(op.config);
        std::uint64_t parent_offset = 0U;
        pack.components.reserve(op.inputs.size());
        for (const auto input_id : op.inputs) {
          const auto input_bytes = data.values[input_id].required_bytes;
          if (input_bytes > std::numeric_limits<std::uint64_t>::max() - 15U) {
            reject(AfeMpkV2DecodeErrorCode::ConfigurationMismatch,
                   path + ".config_params.params.input_shapes",
                   "legacy Pack component alignment overflows");
          }
          const auto stored_bytes = (input_bytes + 15U) & ~std::uint64_t{15U};
          if (parent_offset > std::numeric_limits<std::uint64_t>::max() - stored_bytes) {
            reject(AfeMpkV2DecodeErrorCode::ConfigurationMismatch,
                   path + ".config_params.params.input_shapes",
                   "legacy Pack parent placement overflows");
          }
          pack.components.push_back({input_id, parent_offset, stored_bytes});
          parent_offset += stored_bytes;
        }
      }
      if (op.kind == OpKind::Unpack) {
        const auto& unpack = std::get<UnpackOpConfig>(op.config);
        if (unpack.tensor_types.size() != output_nodes.size() ||
            unpack.tensor_shapes.size() != output_nodes.size() ||
            unpack.tensor_shapes != op.output_shapes) {
          reject(AfeMpkV2DecodeErrorCode::ConfigurationMismatch, path + ".config_params.params",
                 "unpack tensor metadata disagrees with output nodes/shapes");
        }
        for (std::size_t index = 0; index < output_nodes.size(); ++index) {
          const auto expected =
              dense_bytes(unpack.tensor_shapes[index], unpack.tensor_types[index]);
          if (!expected.has_value() || *expected != output_nodes[index].bytes) {
            reject(AfeMpkV2DecodeErrorCode::ValueSizeMismatch,
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
          reject(AfeMpkV2DecodeErrorCode::ConfigurationMismatch, path + ".config_params.params",
                 "slice typed fields are internally inconsistent");
        }
      }
      if (op.kind == OpKind::PassThrough) {
        for (std::size_t index = 0; index < input_nodes.size(); ++index) {
          if (input_nodes[index].bytes != output_nodes[index].bytes) {
            reject(AfeMpkV2DecodeErrorCode::ValueSizeMismatch,
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
          reject(AfeMpkV2DecodeErrorCode::DuplicateProducer,
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
        mla_op_indices.push_back(data.ops.size());
      }
      if (op.kind == OpKind::PassThrough) {
        ++pass_count;
        pass_op_index = data.ops.size();
      }
      if (op.kind == OpKind::HostTvm) {
        host_op_indices.push_back(data.ops.size());
      }
      data.ops.push_back(std::move(op));
    }

    if (mla_count == 0U) {
      reject(AfeMpkV2DecodeErrorCode::MissingMlaStage, "$.plugins",
             "no exact MLA operation exists");
    }
    if (pass_count > 1U ||
        (pass_count == 1U && pass_op_index + 1U != data.ops.size())) {
      reject(AfeMpkV2DecodeErrorCode::InvalidPublicationStage, "$.plugins",
             "PassThrough, when present, must be the unique terminal publication operation");
    }

    if (executable_evidence.size() != mla_op_indices.size()) {
      reject(executable_evidence.size() < mla_op_indices.size()
                 ? AfeMpkV2DecodeErrorCode::MissingMlaExecutableEvidence
                 : AfeMpkV2DecodeErrorCode::UnexpectedMlaExecutableEvidence,
             "$.plugins",
             "MLA executable evidence cardinality does not equal exact MLA operation count");
    }
    std::vector<bool> evidence_used(executable_evidence.size(), false);
    for (std::size_t stage_index = 0; stage_index < mla_op_indices.size(); ++stage_index) {
      const auto mla_op_index = mla_op_indices[stage_index];
      const auto& mla = data.ops[mla_op_index];
      auto& config = std::get<MlaOpConfig>(data.ops[mla_op_index].config);
      const MlaStageExecutableEvidence* evidence = nullptr;
      std::size_t evidence_index = 0U;
      for (std::size_t index = 0; index < executable_evidence.size(); ++index) {
        const auto& candidate = executable_evidence[index];
        if (candidate.logical_stage_id != mla.name || candidate.executable != config.executable) {
          continue;
        }
        if (evidence != nullptr) {
          reject(AfeMpkV2DecodeErrorCode::AmbiguousMlaExecutableEvidence,
                 "$.plugins[" + std::to_string(mla_op_index) + "].resources.executable",
                 "more than one ELF evidence item matches MLA stage '" + mla.name + "'");
        }
        evidence = &candidate;
        evidence_index = index;
      }
      if (!evidence) {
        reject(AfeMpkV2DecodeErrorCode::MissingMlaExecutableEvidence,
               "$.plugins[" + std::to_string(mla_op_index) + "].resources.executable",
               "no ELF evidence matches both logical stage '" + mla.name + "' and executable '" +
                   config.executable + "'");
      }
      evidence_used[evidence_index] = true;
      config.executable_bytes = evidence->byte_length;
      config.executable_sha256 = evidence->sha256;
      const auto& topology = evidence->topology;
      const auto topology_validation =
          reconcile_mla_elf_io_topology_strict(topology, mla.inputs.size(), mla.outputs.size());
      if (!topology_validation.ok) {
        const bool mismatch =
            topology_validation.code == MlaElfIoTopologyError::IfmPortCountMismatch ||
            topology_validation.code == MlaElfIoTopologyError::OfmPortCountMismatch;
        reject(mismatch ? AfeMpkV2DecodeErrorCode::ElfTopologyMismatch
                        : AfeMpkV2DecodeErrorCode::ElfTopologyInvalid,
               "$.plugins[" + std::to_string(mla_op_index) + "]",
               topology_error_detail(topology_validation));
      }

      result.proof.push_back(
          {"MLA[" + std::to_string(stage_index) + "].identity",
           "MPK logical stage '" + mla.name + "' and executable '" + config.executable +
               "' exactly select one ELF topology; no filename/order inference"});
      for (std::size_t index = 0; index < mla.inputs.size(); ++index) {
        const std::string symbol =
            topology.monolithic_ifm ? "data.ifm.b0" : topology.ifm_symbol_names.at(index);
        const auto& value = data.values[mla.inputs[index]];
        const auto physical_extent = mla_elf_ifm_extent_bytes(topology, index);
        if (physical_extent != value.required_bytes) {
          reject(AfeMpkV2DecodeErrorCode::ValueSizeMismatch,
                 "$.plugins[" + std::to_string(mla_op_index) + "].input_nodes[" +
                     std::to_string(index) + "].size",
                 "QMLA IFM extent does not exactly equal the logical tensor");
        }
        data.backend_ports.push_back(
            {stage_index, BackendPortDirection::Input, index, symbol, value.id,
             physical_extent, kLegacyEvoCmaRegionAlignmentBytes,
             BackendPortAlignmentAuthority::LegacyPolicy, BackendPortAccess::ReadOnly});
        result.proof.push_back(
            {"MLA[" + std::to_string(stage_index) + "].IFM[" + std::to_string(index) + "]",
             "ELF symbol '" + symbol + "' and exact stage MPK input agree"});
      }
      for (std::size_t index = 0; index < mla.outputs.size(); ++index) {
        const std::string symbol =
            topology.monolithic_ofm ? "data.ofm.b0" : topology.ofm_symbol_names.at(index);
        const auto physical_extent = mla_elf_ofm_extent_bytes(topology, index);
        author_mla_output_storage(data, mla_op_index, index, symbol, physical_extent,
                                  &result.proof);
        const auto& value = data.values[mla.outputs[index]];
        data.backend_ports.push_back(
            {stage_index, BackendPortDirection::Output, index, symbol, value.id,
             physical_extent, kLegacyEvoCmaRegionAlignmentBytes,
             BackendPortAlignmentAuthority::LegacyPolicy, BackendPortAccess::WriteOnly});
        result.proof.push_back(
            {"MLA[" + std::to_string(stage_index) + "].OFM[" + std::to_string(index) + "]",
             "ELF symbol '" + symbol + "' and exact stage MPK output agree"});
      }
    }
    if (std::find(evidence_used.begin(), evidence_used.end(), false) != evidence_used.end()) {
      reject(AfeMpkV2DecodeErrorCode::UnexpectedMlaExecutableEvidence, "$.plugins",
             "an ELF evidence item is not selected by any exact MLA stage identity");
    }

    if (host_evidence.size() != host_op_indices.size()) {
      reject(AfeMpkV2DecodeErrorCode::UnsupportedHostModule, "$.plugins",
             "A65 host-module evidence cardinality does not equal exact A65 operation count");
    }
    std::vector<bool> host_evidence_used(host_evidence.size(), false);
    for (const auto host_op_index : host_op_indices) {
      auto& op = data.ops[host_op_index];
      auto& config = std::get<HostTvmOpConfig>(op.config);
      const HostTvmExecutableEvidence* evidence = nullptr;
      std::size_t evidence_index = 0U;
      for (std::size_t index = 0; index < host_evidence.size(); ++index) {
        const auto& candidate = host_evidence[index];
        if (candidate.logical_stage_id != op.name || candidate.executable != config.executable) {
          continue;
        }
        if (evidence != nullptr) {
          reject(AfeMpkV2DecodeErrorCode::UnsupportedHostModule,
                 "$.plugins[" + std::to_string(host_op_index) + "].resources.executable",
                 "more than one A65 module evidence item matches the exact stage");
        }
        evidence = &candidate;
        evidence_index = index;
      }
      if (!evidence) {
        reject(AfeMpkV2DecodeErrorCode::UnsupportedHostModule,
               "$.plugins[" + std::to_string(host_op_index) + "].resources.executable",
               "no structural A65 module evidence matches logical identity and executable");
      }
      if (evidence->input_names != config.input_names ||
          evidence->input_types.size() != config.input_types.size() ||
          evidence->output_types.size() != config.output_types.size() ||
          evidence->output_alias_input.size() != config.output_types.size()) {
        reject(AfeMpkV2DecodeErrorCode::ConfigurationMismatch,
               "$.plugins[" + std::to_string(host_op_index) + "].config_params",
               "embedded TVM graph port contract disagrees with the MPK host contract");
      }
      const auto types_equal = [](const auto& left, const auto& right) {
        return left.scalar == right.scalar && left.shape == right.shape;
      };
      for (std::size_t index = 0; index < config.input_types.size(); ++index) {
        if (!types_equal(config.input_types[index], evidence->input_types[index])) {
          reject(AfeMpkV2DecodeErrorCode::ConfigurationMismatch,
                 "$.plugins[" + std::to_string(host_op_index) + "].config_params.input_types[" +
                     std::to_string(index) + "]",
                 "embedded TVM graph input type disagrees with the MPK");
        }
      }
      for (std::size_t index = 0; index < config.output_types.size(); ++index) {
        if (!types_equal(config.output_types[index], evidence->output_types[index])) {
          reject(AfeMpkV2DecodeErrorCode::ConfigurationMismatch,
                 "$.plugins[" + std::to_string(host_op_index) + "].config_params.output_types[" +
                     std::to_string(index) + "]",
                 "embedded TVM graph output type disagrees with the MPK");
        }
        const auto alias = evidence->output_alias_input[index];
        if (alias >= 0 && static_cast<std::size_t>(alias) >= config.input_types.size()) {
          reject(AfeMpkV2DecodeErrorCode::ConfigurationMismatch,
                 "$.plugins[" + std::to_string(host_op_index) + "]",
                 "embedded TVM graph output alias index is out of range");
        }
      }
      config.output_alias_input = evidence->output_alias_input;
      config.executable_bytes = evidence->byte_length;
      config.executable_sha256 = evidence->sha256;
      host_evidence_used[evidence_index] = true;
      result.proof.push_back({"A65[" + std::to_string(host_op_index) + "].identity",
                              "MPK logical stage and executable exactly match embedded "
                              "GraphExecutor JSON; model code was not loaded during admission"});
    }
    if (std::find(host_evidence_used.begin(), host_evidence_used.end(), false) !=
        host_evidence_used.end()) {
      reject(AfeMpkV2DecodeErrorCode::UnsupportedHostModule, "$.plugins",
             "an A65 module evidence item is not selected by any exact host stage identity");
    }

    if (pass_count == 1U) {
      const auto publication_inputs = data.ops[pass_op_index].inputs;
      const auto publication_output_count = data.ops[pass_op_index].outputs.size();
      for (std::size_t index = 0; index < publication_inputs.size(); ++index) {
        const auto value_id = publication_inputs[index];
        data.model_outputs.push_back({index, data.values[value_id].name, value_id});
        result.proof.push_back(
            {"model.output[" + std::to_string(index) + "]",
             "terminal PassThrough is removed as zero work; ordered input value '" +
                 data.values[value_id].name + "' is published directly"});
      }

      // PassThrough is publication metadata, not a physical operation and not
      // a second set of output carriers.  It is terminal, so its values form a
      // suffix and can be removed without renumbering any retained identity.
      data.ops.pop_back();
      if (publication_output_count > data.values.size()) {
        reject(AfeMpkV2DecodeErrorCode::InvalidPublicationStage, "$.plugins",
               "terminal PassThrough output suffix is malformed");
      }
      data.values.resize(data.values.size() - publication_output_count);
    } else {
      std::unordered_set<ValueId> consumed;
      for (const auto& op : data.ops) {
        consumed.insert(op.inputs.begin(), op.inputs.end());
      }
      std::vector<ValueId> terminal_values;
      for (const auto& op : data.ops) {
        for (const auto output : op.outputs) {
          if (!consumed.contains(output)) {
            terminal_values.push_back(output);
          }
        }
      }
      if (terminal_values.empty()) {
        reject(AfeMpkV2DecodeErrorCode::MissingPublicationStage, "$.plugins",
               "stock graph has no terminal produced value to publish");
      }

      if (terminal_values.size() == 1U) {
        const auto value_id = terminal_values.front();
        data.model_outputs.push_back({0U, data.values[value_id].name, value_id});
        result.proof.push_back(
            {"model.output[0]",
             "the unique unconsumed terminal produced value '" + data.values[value_id].name +
                 "' is published directly"});
      } else {
        const auto contract = lookup_afe_publication_contract(manifest_sha256);
        if (!contract) {
          reject(AfeMpkV2DecodeErrorCode::MissingPublicationStage, "$.plugins",
                 "multiple terminal values require an exact digest-bound ordered output contract; "
                 "manifest sha256=" + manifest_sha256);
        }
        std::unordered_set<ValueId> terminal_set(terminal_values.begin(), terminal_values.end());
        std::unordered_set<ValueId> selected;
        for (std::size_t index = 0; index < contract->ordered_value_names.size(); ++index) {
          const std::string wanted(contract->ordered_value_names[index]);
          const auto found = value_by_name.find(wanted);
          if (found == value_by_name.end() || !terminal_set.contains(found->second) ||
              !selected.emplace(found->second).second) {
            reject(AfeMpkV2DecodeErrorCode::InvalidPublicationStage, "$.plugins",
                   "digest-bound output contract contains a missing, nonterminal, or duplicate "
                   "value '" + wanted + "'");
          }
          data.model_outputs.push_back({index, wanted, found->second});
          result.proof.push_back(
              {"model.output[" + std::to_string(index) + "]",
               "exact MPK sha256 " + manifest_sha256 + " publishes terminal value '" + wanted +
                   "'"});
        }
        if (selected.size() != terminal_set.size()) {
          reject(AfeMpkV2DecodeErrorCode::InvalidPublicationStage, "$.plugins",
                 "digest-bound output contract does not enumerate every terminal value exactly "
                 "once");
        }
      }
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
        reject(AfeMpkV2DecodeErrorCode::ConfigurationMismatch,
               "$.values[" + std::to_string(value.id) + "]",
               "value '" + value.name + "' is neither consumed nor publicly published");
      }
    }

    propagate_identity_evidence(data);
    validate_dense_byte_equations(data);
    lower_read_expressions(data, &result.proof);
    validate_view_consumers(data);

    std::string plan_error;
    result.plan = ModelExecutionPlan::create(std::move(data), &plan_error);
    if (!result.plan.has_value()) {
      reject(AfeMpkV2DecodeErrorCode::PlanValidationFailed, "$", plan_error);
    }
    return result;
  } catch (const DecodeAbort& failure) {
    result.plan.reset();
    result.error = AfeMpkV2DecodeError{failure.code, source, failure.path, failure.detail};
    return result;
  } catch (const std::exception& failure) {
    result.plan.reset();
    result.error =
        AfeMpkV2DecodeError{AfeMpkV2DecodeErrorCode::InvalidJson, source, "$", failure.what()};
    return result;
  } catch (...) {
    result.plan.reset();
    result.error = AfeMpkV2DecodeError{AfeMpkV2DecodeErrorCode::InvalidJson, source, "$",
                                       "unknown decoder failure"};
    return result;
  }
}

} // namespace

AfeMpkV2DecodeResult AfeMpkV2Decoder::decode_json(const std::string_view mpk_json,
                                                  const MlaElfIoTopology& elf_topology,
                                                  std::string source_label) const noexcept {
  // Compatibility is intentionally unambiguous: discover the sole MLA identity
  // from the manifest, then route through the exact multi-evidence decoder.
  try {
    const Json root = Json::parse(mpk_json.begin(), mpk_json.end());
    const auto& plugins = root.at("plugins");
    std::vector<MlaStageExecutableEvidence> evidence;
    for (const auto& plugin : plugins) {
      if (plugin.value("processor", std::string{}) != "MLA") {
        continue;
      }
      evidence.push_back({plugin.at("name").get<std::string>(),
                          plugin.at("resources").at("executable").get<std::string>(),
                          elf_topology});
    }
    if (evidence.size() != 1U) {
      AfeMpkV2DecodeResult result;
      result.error = AfeMpkV2DecodeError{
          AfeMpkV2DecodeErrorCode::MultipleMlaStages, std::move(source_label), "$.plugins",
          "single-topology compatibility overload requires exactly one MLA stage"};
      return result;
    }
    return decode_impl(mpk_json, evidence, {}, source_label);
  } catch (const std::exception& failure) {
    AfeMpkV2DecodeResult result;
    result.error = AfeMpkV2DecodeError{AfeMpkV2DecodeErrorCode::InvalidJson,
                                       std::move(source_label), "$", failure.what()};
    return result;
  }
}

AfeMpkV2DecodeResult
AfeMpkV2Decoder::decode_json(const std::string_view mpk_json,
                             const std::span<const MlaStageExecutableEvidence> executable_evidence,
                             std::string source_label) const noexcept {
  return decode_impl(mpk_json, executable_evidence, {}, source_label);
}

AfeMpkV2DecodeResult
AfeMpkV2Decoder::decode_json(const std::string_view mpk_json,
                             const std::span<const MlaStageExecutableEvidence> executable_evidence,
                             const std::span<const HostTvmExecutableEvidence> host_evidence,
                             std::string source_label) const noexcept {
  return decode_impl(mpk_json, executable_evidence, host_evidence, source_label);
}

AfeMpkV2DecodeResult
AfeMpkV2Decoder::decode_file(const std::filesystem::path& mpk_manifest,
                             const MlaElfIoTopology& elf_topology) const noexcept {
  std::ifstream input(mpk_manifest, std::ios::binary);
  if (!input.is_open()) {
    AfeMpkV2DecodeResult result;
    result.error = AfeMpkV2DecodeError{AfeMpkV2DecodeErrorCode::IoError, mpk_manifest.string(), "$",
                                       "cannot open MPK manifest"};
    return result;
  }
  std::ostringstream contents;
  contents << input.rdbuf();
  if (!input.good() && !input.eof()) {
    AfeMpkV2DecodeResult result;
    result.error = AfeMpkV2DecodeError{AfeMpkV2DecodeErrorCode::IoError, mpk_manifest.string(), "$",
                                       "cannot read MPK manifest"};
    return result;
  }
  return decode_json(contents.str(), elf_topology, mpk_manifest.string());
}

AfeMpkV2DecodeResult AfeMpkV2Decoder::decode_file(
    const std::filesystem::path& mpk_manifest,
    const std::span<const MlaStageExecutableEvidence> executable_evidence) const noexcept {
  return decode_file(mpk_manifest, executable_evidence,
                     std::span<const HostTvmExecutableEvidence>{});
}

AfeMpkV2DecodeResult AfeMpkV2Decoder::decode_file(
    const std::filesystem::path& mpk_manifest,
    const std::span<const MlaStageExecutableEvidence> executable_evidence,
    const std::span<const HostTvmExecutableEvidence> host_evidence) const noexcept {
  std::ifstream input(mpk_manifest, std::ios::binary);
  if (!input.is_open()) {
    AfeMpkV2DecodeResult result;
    result.error = AfeMpkV2DecodeError{AfeMpkV2DecodeErrorCode::IoError, mpk_manifest.string(), "$",
                                       "cannot open MPK manifest"};
    return result;
  }
  std::ostringstream contents;
  contents << input.rdbuf();
  if (!input.good() && !input.eof()) {
    AfeMpkV2DecodeResult result;
    result.error = AfeMpkV2DecodeError{AfeMpkV2DecodeErrorCode::IoError, mpk_manifest.string(), "$",
                                       "cannot read MPK manifest"};
    return result;
  }
  return decode_impl(contents.str(), executable_evidence, host_evidence, mpk_manifest.string());
}

} // namespace simaai::neat::pipeline_internal::sima::static_contract
