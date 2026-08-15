#include "HostPcieTensorSetMeta.h"

#include "gst/SimaTensorSetMetaAbi.h"

#include <cctype>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace simaai::neat::pcie::internal {
namespace {

int tensor_set_dtype(const TensorDType dtype) {
  switch (dtype) {
  case TensorDType::UInt8:
    return SIMA_TENSOR_SET_DTYPE_UINT8_V1;
  case TensorDType::Int8:
    return SIMA_TENSOR_SET_DTYPE_INT8_V1;
  case TensorDType::UInt16:
    return SIMA_TENSOR_SET_DTYPE_UINT16_V1;
  case TensorDType::Int16:
    return SIMA_TENSOR_SET_DTYPE_INT16_V1;
  case TensorDType::Int32:
    return SIMA_TENSOR_SET_DTYPE_INT32_V1;
  case TensorDType::BFloat16:
    return SIMA_TENSOR_SET_DTYPE_BF16_V1;
  case TensorDType::Float32:
    return SIMA_TENSOR_SET_DTYPE_FP32_V1;
  case TensorDType::Float64:
    return SIMA_TENSOR_SET_DTYPE_FP64_V1;
  }
  return SIMA_TENSOR_SET_DTYPE_UNKNOWN_V1;
}

std::string canonical_dtype(std::string dtype) {
  std::string out;
  out.reserve(dtype.size());
  for (const unsigned char c : dtype) {
    if (std::isalnum(c)) {
      out.push_back(static_cast<char>(std::tolower(c)));
    }
  }
  if (out == "evxxint8" || out == "ev74int8")
    return "int8";
  if (out == "bfloat16" || out == "evxxbfloat16" || out == "ev74bfloat16")
    return "bf16";
  if (out == "float32")
    return "fp32";
  return out;
}

const char* tensor_dtype_name(const TensorDType dtype) {
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
  return "UNKNOWN";
}

std::string shape_string(const std::vector<std::int64_t>& shape) {
  std::ostringstream out;
  out << '[';
  for (std::size_t i = 0; i < shape.size(); ++i) {
    if (i != 0U)
      out << ", ";
    out << shape[i];
  }
  out << ']';
  return out.str();
}

int tensor_set_layout(const Tensor& tensor) {
  if (tensor.layout == TensorLayout::HW)
    return SIMA_TENSOR_SET_LAYOUT_HW_V1;
  if (tensor.layout == TensorLayout::HWC)
    return SIMA_TENSOR_SET_LAYOUT_HWC_V1;
  if (tensor.layout == TensorLayout::NHWC)
    return SIMA_TENSOR_SET_LAYOUT_NHWC_V1;
  if (tensor.shape.size() == 2)
    return SIMA_TENSOR_SET_LAYOUT_HW_V1;
  if (tensor.shape.size() == 3)
    return SIMA_TENSOR_SET_LAYOUT_HWC_V1;
  if (tensor.shape.size() == 4)
    return SIMA_TENSOR_SET_LAYOUT_NHWC_V1;
  return SIMA_TENSOR_SET_LAYOUT_UNKNOWN_V1;
}

} // namespace

MappedSample::~MappedSample() {
  if (mapped && buffer) {
    gst_buffer_unmap(buffer, &map);
  }
  if (sample) {
    gst_sample_unref(sample);
  }
}

void attach_tensor_set_meta(GstBuffer* buffer, const std::vector<TensorMetaSpan>& spans,
                            const std::vector<PcieTensorFact>& input_facts) {
  if (!buffer || spans.empty()) {
    throw std::runtime_error("tensor-set metadata requires a buffer and at least one tensor");
  }
  if (input_facts.size() != spans.size()) {
    throw std::runtime_error("submitted tensor count does not match the model input contract");
  }
  if (gst_meta_get_info(SIMA_TENSOR_SET_META_NAME) == nullptr) {
    static const gchar* tags[] = {"memory", "tensor", nullptr};
    if (gst_meta_register_custom(SIMA_TENSOR_SET_META_NAME, tags, nullptr, nullptr, nullptr) ==
        nullptr) {
      throw std::runtime_error("failed to register GstSimaTensorSetMeta");
    }
  }

  std::vector<std::string> names;
  names.reserve(spans.size());
  std::vector<SimaTensorDescriptorV2> descriptors;
  descriptors.reserve(spans.size());

  for (std::size_t i = 0; i < spans.size(); ++i) {
    const auto& span = spans[i];
    if (!span.tensor || span.byte_offset < 0) {
      throw std::runtime_error("invalid tensor-set metadata span");
    }
    const auto& tensor = *span.tensor;
    const auto& fact = input_facts[i];
    if (canonical_dtype(tensor_dtype_name(tensor.dtype)) != canonical_dtype(fact.dtype)) {
      throw std::invalid_argument("PCIe input tensor " + std::to_string(i) + " ('" + fact.name +
                                  "') has dtype " + tensor_dtype_name(tensor.dtype) +
                                  "; expected " + fact.dtype);
    }
    if (tensor.shape != fact.shape) {
      throw std::invalid_argument("PCIe input tensor " + std::to_string(i) + " ('" + fact.name +
                                  "') has shape " + shape_string(tensor.shape) + "; expected " +
                                  shape_string(fact.shape));
    }
    if (tensor.shape.size() > SIMA_TENSOR_SET_MAX_RANK) {
      throw std::runtime_error("tensor-set metadata supports tensor ranks up to " +
                               std::to_string(SIMA_TENSOR_SET_MAX_RANK));
    }
    const auto& declared_strides =
        span.strides_bytes_override.empty() ? tensor.strides_bytes : span.strides_bytes_override;
    if (!declared_strides.empty() && declared_strides.size() != tensor.shape.size()) {
      throw std::runtime_error("tensor-set metadata strides must match tensor rank");
    }
    names.push_back(tensor.route.name.empty() ? "tensor_" + std::to_string(i) : tensor.route.name);

    SimaTensorDescriptorV2 desc{};
    desc.logical_index = static_cast<gint>(i);
    desc.physical_index = input_facts[i].physical_index >= 0
                              ? input_facts[i].physical_index
                              : (tensor.route.physical_index >= 0 ? tensor.route.physical_index
                                                                  : static_cast<int>(i));
    desc.backend_output_index = tensor.route.backend_output_index;
    desc.route_slot = tensor.route.route_slot >= 0 ? tensor.route.route_slot : static_cast<gint>(i);
    desc.memory_index = tensor.route.memory_index;
    desc.logical_name_id = static_cast<gint>(i);
    desc.backend_name_id = static_cast<gint>(i);
    desc.segment_name_id = static_cast<gint>(i);
    desc.byte_offset = span.byte_offset;
    desc.size_bytes = span.size_bytes;
    desc.dtype = tensor_set_dtype(tensor.dtype);
    desc.layout = tensor_set_layout(tensor);
    desc.rank = static_cast<guint>(tensor.shape.size());
    for (guint d = 0; d < desc.rank; ++d) {
      desc.shape[d] = tensor.shape[d];
    }
    const std::vector<std::int64_t> strides =
        declared_strides.empty()
            ? contiguous_tensor_strides(tensor.shape, tensor_dtype_bytes(tensor.dtype))
            : declared_strides;
    for (guint d = 0; d < desc.rank && d < strides.size(); ++d) {
      desc.stride_bytes[d] = strides[d];
    }
    descriptors.push_back(desc);
  }

  std::vector<gchar*> name_table;
  name_table.reserve(names.size() + 1U);
  for (const auto& name : names) {
    name_table.push_back(const_cast<gchar*>(name.c_str()));
  }
  name_table.push_back(nullptr);

  GstCustomMeta* meta = gst_buffer_add_custom_meta(buffer, SIMA_TENSOR_SET_META_NAME);
  if (!meta) {
    throw std::runtime_error("failed to attach GstSimaTensorSetMeta");
  }

  GBytes* descriptor_bytes =
      g_bytes_new(descriptors.data(), descriptors.size() * sizeof(SimaTensorDescriptorV2));
  GstStructure* structure = gst_custom_meta_get_structure(meta);
  gst_structure_set(
      structure, SIMA_TENSOR_SET_META_FIELD_VERSION, G_TYPE_UINT, SIMA_TENSOR_SET_META_VERSION,
      SIMA_TENSOR_SET_META_FIELD_TENSOR_COUNT, G_TYPE_UINT, static_cast<guint>(descriptors.size()),
      SIMA_TENSOR_SET_META_FIELD_DESCRIPTOR_SIZE, G_TYPE_UINT,
      static_cast<guint>(sizeof(SimaTensorDescriptorV2)), SIMA_TENSOR_SET_META_FIELD_DESCRIPTORS,
      G_TYPE_BYTES, descriptor_bytes, SIMA_TENSOR_SET_META_FIELD_STAGE_KEY, G_TYPE_STRING,
      "pcie-input", SIMA_TENSOR_SET_META_FIELD_NAME_TABLE, G_TYPE_STRV, name_table.data(), nullptr);
  g_bytes_unref(descriptor_bytes);
}

} // namespace simaai::neat::pcie::internal
