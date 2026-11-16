#include "pipeline/FeatureTypes.h"

#include "pipeline/GraphOptions.h"
#include "pipeline/internal/TensorMath.h"

#include <algorithm>
#include <cstring>
#include <limits>
#include <stdexcept>

namespace simaai::neat {
namespace {

std::size_t checked_mul(std::size_t a, std::size_t b, const char* what) {
  if (a != 0U && b > std::numeric_limits<std::size_t>::max() / a) {
    throw std::runtime_error(std::string("SuperPoint payload ") + what + " overflows size_t");
  }
  return a * b;
}

std::size_t checked_add(std::size_t a, std::size_t b, const char* what) {
  if (b > std::numeric_limits<std::size_t>::max() - a) {
    throw std::runtime_error(std::string("SuperPoint payload ") + what + " overflows size_t");
  }
  return a + b;
}

std::size_t value_bytes(FeatureValueDType dtype) {
  switch (dtype) {
  case FeatureValueDType::Int8:
    return 1U;
  case FeatureValueDType::BFloat16:
    return 2U;
  case FeatureValueDType::Float32:
    return 4U;
  default:
    throw std::runtime_error("SuperPoint payload contains an unknown dtype code");
  }
}

TensorDType tensor_dtype(FeatureValueDType dtype) {
  switch (dtype) {
  case FeatureValueDType::Int8:
    return TensorDType::Int8;
  case FeatureValueDType::BFloat16:
    return TensorDType::BFloat16;
  case FeatureValueDType::Float32:
    return TensorDType::Float32;
  default:
    throw std::runtime_error("SuperPoint payload contains an unknown dtype code");
  }
}

Tensor copy_strided_rows(const std::vector<std::uint8_t>& bytes, std::size_t offset,
                         std::size_t count, std::size_t columns, std::size_t stride,
                         FeatureValueDType dtype) {
  const std::size_t elem = value_bytes(dtype);
  const std::size_t row_bytes = checked_mul(columns, elem, "row size");
  const std::size_t output_bytes = checked_mul(count, row_bytes, "output size");
  auto storage = make_cpu_owned_storage(output_bytes);
  if (output_bytes != 0U) {
    Mapping mapping = storage->map(MapMode::Write);
    auto* dst = static_cast<std::uint8_t*>(mapping.data);
    for (std::size_t row = 0; row < count; ++row) {
      std::memcpy(dst + row * row_bytes, bytes.data() + offset + row * stride, row_bytes);
    }
  }
  Tensor out;
  out.storage = std::move(storage);
  out.device = {DeviceType::CPU, 0};
  out.read_only = false;
  out.dtype = tensor_dtype(dtype);
  out.layout = TensorLayout::Unknown;
  if (columns == 1U) {
    out.shape = {static_cast<std::int64_t>(count)};
    out.strides_bytes = {static_cast<std::int64_t>(elem)};
  } else {
    out.shape = {static_cast<std::int64_t>(count), static_cast<std::int64_t>(columns)};
    out.strides_bytes = {static_cast<std::int64_t>(row_bytes), static_cast<std::int64_t>(elem)};
  }
  return out;
}

struct Section {
  std::size_t begin = 0;
  std::size_t end = 0;
};

Section validate_section(std::size_t offset, std::size_t capacity, std::size_t stride,
                         std::size_t row_bytes, std::size_t header_bytes, std::size_t total_bytes,
                         const char* name) {
  if (stride < row_bytes || (offset % 4U) != 0U) {
    throw std::runtime_error(std::string("SuperPoint V1 invalid ") + name + " stride/alignment");
  }
  const std::size_t span = capacity == 0U
                               ? 0U
                               : checked_add(checked_mul(capacity - 1U, stride, "section span"),
                                             row_bytes, "section span");
  if (offset < header_bytes || offset > total_bytes || span > total_bytes - offset) {
    throw std::runtime_error(std::string("SuperPoint V1 ") + name + " section is out of bounds");
  }
  return {offset, checked_add(offset, span, "section end")};
}

FeaturePointTensors parse_v1(const std::vector<std::uint8_t>& bytes) {
  if (bytes.size() < sizeof(FeaturePointsHeaderV1)) {
    throw std::runtime_error("SuperPoint V1 payload is smaller than its fixed header");
  }
  FeaturePointsHeaderV1 h{};
  std::memcpy(&h, bytes.data(), sizeof(h));
  if (h.magic != kFeaturePointsMagicV1 || h.version != 1U ||
      h.header_bytes < sizeof(FeaturePointsHeaderV1) || h.header_bytes > h.total_bytes ||
      h.total_bytes > bytes.size()) {
    throw std::runtime_error("SuperPoint V1 header magic/version/size is invalid");
  }
  if (h.count > h.capacity || h.descriptor_dim == 0U ||
      h.layout != static_cast<std::uint8_t>(FeaturePointsLayout::StructureOfArrays)) {
    throw std::runtime_error("SuperPoint V1 count, descriptor dimension, or layout is invalid");
  }
  const auto coord_dtype = static_cast<FeatureValueDType>(h.coordinate_dtype);
  const auto score_dtype = static_cast<FeatureValueDType>(h.score_dtype);
  const auto desc_dtype = static_cast<FeatureValueDType>(h.descriptor_dtype);
  if (coord_dtype != FeatureValueDType::Float32 || score_dtype != FeatureValueDType::Float32) {
    throw std::runtime_error("SuperPoint V1 coordinates and scores must be FP32");
  }
  const std::size_t coord_row = 2U * value_bytes(coord_dtype);
  const std::size_t score_row = value_bytes(score_dtype);
  const std::size_t desc_row = checked_mul(static_cast<std::size_t>(h.descriptor_dim),
                                           value_bytes(desc_dtype), "descriptor row");
  const Section sections[] = {
      validate_section(h.keypoints_offset, h.capacity, h.keypoints_stride, coord_row,
                       h.header_bytes, h.total_bytes, "keypoints"),
      validate_section(h.scores_offset, h.capacity, h.scores_stride, score_row, h.header_bytes,
                       h.total_bytes, "scores"),
      validate_section(h.descriptors_offset, h.capacity, h.descriptor_stride, desc_row,
                       h.header_bytes, h.total_bytes, "descriptors"),
  };
  for (std::size_t i = 0; i < 3U; ++i) {
    for (std::size_t j = i + 1U; j < 3U; ++j) {
      if (sections[i].begin < sections[j].end && sections[j].begin < sections[i].end) {
        throw std::runtime_error("SuperPoint V1 payload sections overlap");
      }
    }
  }
  FeaturePointTensors out;
  out.keypoints =
      copy_strided_rows(bytes, h.keypoints_offset, h.count, 2U, h.keypoints_stride, coord_dtype);
  out.scores = copy_strided_rows(bytes, h.scores_offset, h.count, 1U, h.scores_stride, score_dtype);
  out.descriptors = copy_strided_rows(bytes, h.descriptors_offset, h.count, h.descriptor_dim,
                                      h.descriptor_stride, desc_dtype);
  return out;
}

FeaturePointTensors parse_legacy_v0(const std::vector<std::uint8_t>& bytes) {
  constexpr std::size_t kRecordBytes = 264U;
  constexpr std::size_t kDescriptorDim = 256U;
  if (bytes.size() < sizeof(std::int32_t) ||
      ((bytes.size() - sizeof(std::int32_t)) % kRecordBytes) != 0U) {
    throw std::runtime_error("SuperPoint legacy V0 payload size is invalid");
  }
  std::int32_t signed_count = 0;
  std::memcpy(&signed_count, bytes.data(), sizeof(signed_count));
  const std::size_t capacity = (bytes.size() - sizeof(std::int32_t)) / kRecordBytes;
  if (signed_count < 0 || static_cast<std::size_t>(signed_count) > capacity) {
    throw std::runtime_error("SuperPoint legacy V0 count exceeds capacity");
  }
  const std::size_t count = static_cast<std::size_t>(signed_count);
  auto key_storage = make_cpu_owned_storage(count * 2U * sizeof(float));
  auto score_storage = make_cpu_owned_storage(count * sizeof(float));
  auto desc_storage = make_cpu_owned_storage(count * kDescriptorDim);
  if (count != 0U) {
    Mapping key_map = key_storage->map(MapMode::Write);
    Mapping score_map = score_storage->map(MapMode::Write);
    Mapping desc_map = desc_storage->map(MapMode::Write);
    auto* keypoints = static_cast<float*>(key_map.data);
    auto* scores = static_cast<float*>(score_map.data);
    auto* descriptors = static_cast<std::uint8_t*>(desc_map.data);
    for (std::size_t i = 0; i < count; ++i) {
      const auto* record = bytes.data() + sizeof(std::int32_t) + i * kRecordBytes;
      std::uint16_t x = 0;
      std::uint16_t y = 0;
      std::memcpy(&x, record, sizeof(x));
      std::memcpy(&y, record + 2U, sizeof(y));
      keypoints[2U * i] = static_cast<float>(x);
      keypoints[2U * i + 1U] = static_cast<float>(y);
      std::memcpy(scores + i, record + 4U, sizeof(float));
      std::memcpy(descriptors + i * kDescriptorDim, record + 8U, kDescriptorDim);
    }
  }
  auto make_tensor = [](std::shared_ptr<TensorBuffer> storage, TensorDType dtype,
                        std::vector<std::int64_t> shape, std::vector<std::int64_t> strides) {
    Tensor t;
    t.storage = std::move(storage);
    t.device = {DeviceType::CPU, 0};
    t.read_only = false;
    t.dtype = dtype;
    t.layout = TensorLayout::Unknown;
    t.shape = std::move(shape);
    t.strides_bytes = std::move(strides);
    return t;
  };
  FeaturePointTensors out;
  out.keypoints = make_tensor(std::move(key_storage), TensorDType::Float32,
                              {static_cast<std::int64_t>(count), 2}, {8, 4});
  out.scores = make_tensor(std::move(score_storage), TensorDType::Float32,
                           {static_cast<std::int64_t>(count)}, {4});
  out.descriptors = make_tensor(std::move(desc_storage), TensorDType::Int8,
                                {static_cast<std::int64_t>(count), 256}, {256, 1});
  return out;
}

} // namespace

FeaturePointTensors decode_superpoint_tensor(const Tensor& tensor) {
  if (!tensor.storage || !tensor.is_dense()) {
    throw std::runtime_error("SuperPoint output tensor must have dense storage");
  }
  const std::vector<std::uint8_t> bytes = tensor.copy_payload_bytes();
  std::uint32_t magic = 0U;
  if (bytes.size() >= sizeof(magic)) {
    std::memcpy(&magic, bytes.data(), sizeof(magic));
  }
  if (magic == kFeaturePointsMagicV1) {
    return parse_v1(bytes);
  }
  if (read_feature_format(tensor) == kFeatureFormatLegacyA65V0) {
    return parse_legacy_v0(bytes);
  }
  throw std::runtime_error(
      "SuperPoint output is neither FEATURE_POINTS_V1 nor explicitly tagged legacy V0");
}

FeaturePointTensorList decode_superpoint(const TensorList& tensors) {
  FeaturePointTensorList out;
  out.reserve(tensors.size());
  for (std::size_t i = 0; i < tensors.size(); ++i) {
    try {
      out.push_back(decode_superpoint_tensor(tensors[i]));
    } catch (const std::runtime_error& e) {
      throw std::runtime_error("decode_superpoint: input tensor " + std::to_string(i) + ": " +
                               e.what());
    }
  }
  return out;
}

void tag_feature_format(Tensor& tensor, std::string format) {
  if (!tensor.semantic.feature.has_value()) {
    tensor.semantic.feature = FeatureSpec{};
  }
  tensor.semantic.feature->format = std::move(format);
}

std::string read_feature_format(const Tensor& tensor) {
  return tensor.semantic.feature.has_value() ? tensor.semantic.feature->format : std::string{};
}

void tag_feature_format_in_sample(Sample& sample) {
  if (sample.kind == SampleKind::TensorSet && sample.tensors.size() == 1U) {
    std::string format = !sample.payload_tag.empty() ? sample.payload_tag : sample.format;
    if (format == kFeatureFormatPointsV1 || format == kFeatureFormatLegacyA65V0) {
      tag_feature_format(sample.tensors.front(), std::move(format));
    }
  } else if (sample.kind == SampleKind::Bundle) {
    for (auto& field : sample.fields) {
      tag_feature_format_in_sample(field);
    }
  }
}

} // namespace simaai::neat
