#include "HostPcieTensorSetMeta.h"

#include "gst/SimaTensorSetMetaAbi.h"

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

TensorDType dtype_from_tensor_set(const gint dtype) {
  switch (dtype) {
  case SIMA_TENSOR_SET_DTYPE_UINT8_V1:
    return TensorDType::UInt8;
  case SIMA_TENSOR_SET_DTYPE_INT8_V1:
    return TensorDType::Int8;
  case SIMA_TENSOR_SET_DTYPE_UINT16_V1:
    return TensorDType::UInt16;
  case SIMA_TENSOR_SET_DTYPE_INT16_V1:
    return TensorDType::Int16;
  case SIMA_TENSOR_SET_DTYPE_BF16_V1:
    return TensorDType::BFloat16;
  case SIMA_TENSOR_SET_DTYPE_INT32_V1:
    return TensorDType::Int32;
  case SIMA_TENSOR_SET_DTYPE_FP32_V1:
    return TensorDType::Float32;
  case SIMA_TENSOR_SET_DTYPE_FP64_V1:
    return TensorDType::Float64;
  default:
    throw std::runtime_error("unsupported GstSimaTensorSetMeta dtype");
  }
}

TensorLayout layout_from_tensor_set(const gint layout) {
  switch (layout) {
  case SIMA_TENSOR_SET_LAYOUT_HW_V1:
    return TensorLayout::HW;
  case SIMA_TENSOR_SET_LAYOUT_HWC_V1:
    return TensorLayout::HWC;
  case SIMA_TENSOR_SET_LAYOUT_NHWC_V1:
    return TensorLayout::NHWC;
  default:
    return TensorLayout::Unknown;
  }
}

struct GBytesOwner {
  GBytes* value = nullptr;

  ~GBytesOwner() {
    if (value) {
      g_bytes_unref(value);
    }
  }
};

struct GStrvOwner {
  gchar** value = nullptr;

  ~GStrvOwner() {
    if (value) {
      g_strfreev(value);
    }
  }
};

std::string tensor_name_from_table(gchar** names, const std::size_t name_count, const gint name_id,
                                   const std::size_t fallback_index) {
  if (names && name_id >= 0) {
    const auto index = static_cast<std::size_t>(name_id);
    if (index < name_count && names[index] && names[index][0] != '\0') {
      return names[index];
    }
  }
  return "tensor_" + std::to_string(fallback_index);
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

void attach_tensor_set_meta(GstBuffer* buffer, const std::vector<TensorMetaSpan>& spans) {
  if (!buffer || spans.empty()) {
    throw std::runtime_error("tensor-set metadata requires a buffer and at least one tensor");
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
    if (tensor.shape.size() > SIMA_TENSOR_SET_MAX_RANK) {
      throw std::runtime_error("tensor-set metadata supports tensor ranks up to " +
                               std::to_string(SIMA_TENSOR_SET_MAX_RANK));
    }
    if (!tensor.strides_bytes.empty() && tensor.strides_bytes.size() != tensor.shape.size()) {
      throw std::runtime_error("tensor-set metadata strides must match tensor rank");
    }
    names.push_back(tensor.route.name.empty() ? "tensor_" + std::to_string(i) : tensor.route.name);

    SimaTensorDescriptorV2 desc{};
    desc.logical_index = static_cast<gint>(i);
    desc.physical_index =
        tensor.route.physical_index >= 0 ? tensor.route.physical_index : static_cast<int>(i);
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
        tensor.strides_bytes.empty()
            ? contiguous_tensor_strides(tensor.shape, tensor_dtype_bytes(tensor.dtype))
            : tensor.strides_bytes;
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

TensorList tensors_from_tensor_set_meta(const std::shared_ptr<MappedSample>& owner) {
  TensorList out;
  if (!owner || !owner->buffer || !owner->map.data) {
    return out;
  }

  GstCustomMeta* meta = gst_buffer_get_custom_meta(owner->buffer, SIMA_TENSOR_SET_META_NAME);
  GstStructure* structure = meta ? gst_custom_meta_get_structure(meta) : nullptr;
  if (!structure) {
    return out;
  }

  guint version = 0;
  guint tensor_count = 0;
  guint descriptor_size = 0;
  GBytesOwner descriptor_bytes;
  if (!gst_structure_get_uint(structure, SIMA_TENSOR_SET_META_FIELD_VERSION, &version) ||
      !gst_structure_get_uint(structure, SIMA_TENSOR_SET_META_FIELD_TENSOR_COUNT, &tensor_count) ||
      !gst_structure_get_uint(structure, SIMA_TENSOR_SET_META_FIELD_DESCRIPTOR_SIZE,
                              &descriptor_size) ||
      !gst_structure_get(structure, SIMA_TENSOR_SET_META_FIELD_DESCRIPTORS, G_TYPE_BYTES,
                         &descriptor_bytes.value, nullptr)) {
    throw std::runtime_error("output GstSimaTensorSetMeta is missing required fields");
  }

  if (version != SIMA_TENSOR_SET_META_VERSION || tensor_count == 0 ||
      descriptor_size != sizeof(SimaTensorDescriptorV2) || !descriptor_bytes.value) {
    throw std::runtime_error("output GstSimaTensorSetMeta has incompatible descriptor header");
  }

  gsize descriptor_blob_size = 0;
  const auto* descriptor_blob = static_cast<const std::uint8_t*>(
      g_bytes_get_data(descriptor_bytes.value, &descriptor_blob_size));
  if (!descriptor_blob ||
      descriptor_blob_size != static_cast<gsize>(tensor_count) * descriptor_size) {
    throw std::runtime_error("output GstSimaTensorSetMeta descriptor blob size is invalid");
  }

  GStrvOwner name_table;
  gst_structure_get(structure, SIMA_TENSOR_SET_META_FIELD_NAME_TABLE, G_TYPE_STRV,
                    &name_table.value, nullptr);
  const std::size_t name_count = name_table.value ? g_strv_length(name_table.value) : 0;

  const auto* descriptors = reinterpret_cast<const SimaTensorDescriptorV2*>(descriptor_blob);
  out.reserve(tensor_count);
  for (guint i = 0; i < tensor_count; ++i) {
    const auto& desc = descriptors[i];
    if (desc.byte_offset < 0) {
      throw std::runtime_error("output GstSimaTensorSetMeta has negative tensor offset");
    }
    const auto offset = static_cast<std::size_t>(desc.byte_offset);
    const auto size = static_cast<std::size_t>(desc.size_bytes);
    if (offset > owner->map.size || size > owner->map.size - offset) {
      throw std::runtime_error("output GstSimaTensorSetMeta tensor range exceeds payload");
    }
    if (desc.rank > SIMA_TENSOR_SET_MAX_RANK) {
      throw std::runtime_error("output GstSimaTensorSetMeta tensor rank is invalid");
    }

    Tensor tensor;
    tensor.owner = owner;
    tensor.data = static_cast<std::uint8_t*>(owner->map.data) + offset;
    tensor.size_bytes = size;
    tensor.byte_offset = 0;
    tensor.dtype = dtype_from_tensor_set(desc.dtype);
    tensor.layout = layout_from_tensor_set(desc.layout);
    tensor.shape.assign(desc.shape, desc.shape + desc.rank);
    tensor.strides_bytes.assign(desc.stride_bytes, desc.stride_bytes + desc.rank);
    tensor.read_only = true;
    tensor.route.name =
        tensor_name_from_table(name_table.value, name_count, desc.logical_name_id, i);
    tensor.route.logical_index = desc.logical_index;
    tensor.route.backend_output_index = desc.backend_output_index;
    tensor.route.physical_index = desc.physical_index;
    tensor.route.route_slot = desc.route_slot;
    tensor.route.memory_index = desc.memory_index;
    tensor.route.physical_byte_offset = desc.byte_offset;
    out.push_back(std::move(tensor));
  }
  return out;
}

} // namespace simaai::neat::pcie::internal
