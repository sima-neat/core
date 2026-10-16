#include "pipeline/internal/sima/BoxDecodePublication.h"
#include "pipeline/FeatureTypes.h"

#include <simaai/scheduler/host_task.h>

#include <algorithm>
#include <arm_neon.h>
#include <array>
#include <cmath>
#include <cstring>
#include <limits>
#include <stdexcept>
#include <vector>

namespace simaai::neat::pipeline_internal::sima {
namespace {
namespace k = ::simaai::kernels;
namespace host = ::simaai::scheduler::host;
struct WirePlan {
  BoxDecodeWireFormat format;
  std::size_t capacity = 0, bytes = 0, auxiliary_row = 0;
  std::vector<k::TensorRequirements> inputs;
  FeaturePointsHeaderV1 feature_header;
};
std::size_t checked_add(std::size_t a, std::size_t b) {
  if (b > std::numeric_limits<std::size_t>::max() - a)
    throw std::invalid_argument("BoxDecode wire extent overflows");
  return a + b;
}
std::size_t checked_mul(std::size_t a, std::size_t b) {
  if (a && b > std::numeric_limits<std::size_t>::max() / a)
    throw std::invalid_argument("BoxDecode wire extent overflows");
  return a * b;
}
std::size_t aligned16(std::size_t n) {
  return checked_add(n, 15) & ~std::size_t{15};
}
std::size_t scalar_bytes(k::DType dtype) {
  switch (dtype) {
  case k::DType::Float32:
  case k::DType::Int32:
    return 4;
  case k::DType::BFloat16:
    return 2;
  case k::DType::Int8:
  case k::DType::UInt8:
    return 1;
  default:
    throw std::invalid_argument("BoxDecode wire dtype is unsupported");
  }
}
void require_spec(const k::TensorSpec& spec, k::DType dtype,
                  std::initializer_list<std::int64_t> shape) {
  if (spec.dtype != dtype || spec.shape != k::Shape(shape) ||
      !std::holds_alternative<k::Contiguous>(spec.layout))
    throw std::invalid_argument("BoxDecode wire input does not match canonical output contract");
}
template <class T> T load(const std::byte* source) {
  T value;
  std::memcpy(&value, source, sizeof(value));
  return value;
}
template <class T> void store(std::byte* target, const T& value) {
  std::memcpy(target, &value, sizeof(value));
}
bool overlaps(const void* a, std::size_t na, const void* b, std::size_t nb) {
  const auto aa = reinterpret_cast<std::uintptr_t>(a), bb = reinterpret_cast<std::uintptr_t>(b);
  return aa <= bb ? bb - aa < na : aa - bb < nb;
}
k::Status publish(const WirePlan& plan, std::span<const k::ConstHostBuffer> inputs,
                  std::span<const k::HostBuffer> outputs) {
  if (inputs.size() != plan.inputs.size() || outputs.size() != 1)
    return k::Error{k::ErrorCode::InvalidArity, "ports", "wire publication port count changed"};
  if (!outputs[0].data || outputs[0].size_bytes < plan.bytes)
    return k::Error{k::ErrorCode::BufferTooSmall, "output", "wire publication output is too small"};
  if (reinterpret_cast<std::uintptr_t>(outputs[0].data) > UINTPTR_MAX - plan.bytes)
    return k::Error{k::ErrorCode::SizeOverflow, "output",
                    "wire publication output address overflows"};
  for (std::size_t i = 0; i < inputs.size(); ++i) {
    const auto bytes = plan.inputs[i].storage_bytes;
    if (!inputs[i].data || inputs[i].size_bytes < bytes)
      return k::Error{k::ErrorCode::BufferTooSmall, "input", "wire publication input is too small"};
    if (reinterpret_cast<std::uintptr_t>(inputs[i].data) > UINTPTR_MAX - bytes)
      return k::Error{k::ErrorCode::SizeOverflow, "input",
                      "wire publication input address overflows"};
    if (overlaps(inputs[i].data, bytes, outputs[0].data, plan.bytes))
      return k::Error{k::ErrorCode::AliasedBuffers, "output",
                      "wire publication input/output spans overlap"};
  }
  const auto count = load<std::int32_t>(inputs[3].data);
  if (count < 0 || static_cast<std::size_t>(count) > plan.capacity)
    return k::Error{k::ErrorCode::InvalidParameter, "count",
                    "numerical result count exceeds capacity"};
  auto* output = outputs[0].data;
  const auto n = static_cast<std::size_t>(count);
  if (plan.format == BoxDecodeWireFormat::RcnnProposals) {
    const auto per_level = plan.capacity / 4;
    const auto group_bytes = 4 + per_level * 12;
    std::array<std::int32_t, 4> counts{};
    // Validate every bucket before modifying the public buffer. Level selection
    // belongs to the numerical kernel; this only packs its FP32 proposal rows.
    for (std::size_t i = 0; i < n; ++i) {
      const auto level = load<std::int32_t>(inputs[4].data + i * 4);
      if (level < 0 || level >= 4 || ++counts[level] > static_cast<std::int32_t>(per_level))
        return k::Error{k::ErrorCode::InvalidParameter, "levels",
                        "RCNN proposal level or per-level count is invalid"};
    }
    std::memset(output, 0, plan.bytes);
    for (std::size_t level = 0; level < 4; ++level)
      store(output + level * group_bytes, counts[level]);
    counts.fill(0);
    for (std::size_t i = 0; i < n; ++i) {
      const auto level = load<std::int32_t>(inputs[4].data + i * 4);
      auto* row = output + level * group_bytes + 4 + counts[level]++ * 12;
      const auto* from = inputs[0].data + i * 16;
      const float x = load<float>(from), y = load<float>(from + 4);
      const std::array<float, 6> values{
          load<float>(inputs[1].data + i * 4), 0.f, x, y, x + load<float>(from + 8),
          y + load<float>(from + 12)};
      for (std::size_t field = 0; field < values.size(); ++field)
        store(row + field * 2, static_cast<float16_t>(values[field]));
    }
    return {};
  }
  if (plan.format == BoxDecodeWireFormat::FeaturePointsV1) {
    // Copy valid rows and clear wire padding/tails; retained carriers cannot
    // leak a preceding frame, even when the current result count shrinks.
    auto header = plan.feature_header;
    header.count = static_cast<std::uint32_t>(count);
    std::memset(output, 0, header.keypoints_offset);
    store(output, header);
    std::memcpy(output + header.keypoints_offset, inputs[0].data, n * 8);
    std::memset(output + header.keypoints_offset + n * 8, 0,
                header.scores_offset - header.keypoints_offset - n * 8);
    std::memcpy(output + header.scores_offset, inputs[1].data, n * 4);
    std::memset(output + header.scores_offset + n * 4, 0,
                header.descriptors_offset - header.scores_offset - n * 4);
    for (std::size_t i = 0; i < n; ++i) {
      auto* row = output + header.descriptors_offset + i * header.descriptor_stride;
      std::memcpy(row, inputs[2].data + i * plan.auxiliary_row, plan.auxiliary_row);
      std::memset(row + plan.auxiliary_row, 0, header.descriptor_stride - plan.auxiliary_row);
    }
    const auto tail = header.descriptors_offset + n * header.descriptor_stride;
    std::memset(output + tail, 0, plan.bytes - tail);
    return {};
  }
  if (plan.format == BoxDecodeWireFormat::LegacyFeaturePointsV0) {
    // The legacy API saturates integer image coordinates into uint16 fields.
    // Canonical SuperPoint coordinates are integral FP32 pixel positions.
    for (std::size_t i = 0; i < n; ++i) {
      const auto x = load<float>(inputs[0].data + i * 8);
      const auto y = load<float>(inputs[0].data + i * 8 + 4);
      if (!std::isfinite(x) || !std::isfinite(y))
        return k::Error{k::ErrorCode::InvalidParameter, "coordinates",
                        "nonfinite feature coordinate"};
    }
    store(output, count);
    for (std::size_t i = 0; i < n; ++i) {
      auto* row = output + 4 + i * 264;
      store(row, static_cast<std::uint16_t>(
                     std::clamp(load<float>(inputs[0].data + i * 8), 0.f, 65535.f)));
      store(row + 2, static_cast<std::uint16_t>(
                         std::clamp(load<float>(inputs[0].data + i * 8 + 4), 0.f, 65535.f)));
      std::memcpy(row + 4, inputs[1].data + i * 4, 4);
      std::memcpy(row + 8, inputs[2].data + i * 256, 256);
    }
    std::memset(output + 4 + n * 264, 0, (plan.capacity - n) * 264);
    return {};
  }
  if (plan.format == BoxDecodeWireFormat::Pose) {
    // Coordinates were already rounded/clipped by the numerical kernel. Check
    // representability before writing; converting is wire encoding, not math.
    for (std::size_t i = 0; i < n * 17; ++i)
      for (std::size_t axis = 0; axis < 2; ++axis) {
        const float value = load<float>(inputs[4].data + (i * 3 + axis) * 4);
        if (!std::isfinite(value) || value < 0.f || double(value) > UINT32_MAX)
          return k::Error{k::ErrorCode::InvalidParameter, "coordinates",
                          "pose coordinate cannot be encoded"};
      }
  }
  store(output, count);
  for (std::size_t i = 0; i < n; ++i) {
    auto* row = output + 4 + i * 24;
    std::memcpy(row, inputs[0].data + i * 16, 16);
    std::memcpy(row + 16, inputs[1].data + i * 4, 4);
    std::memcpy(row + 20, inputs[2].data + i * 4, 4);
  }
  std::memset(output + 4 + n * 24, 0, (plan.capacity - n) * 24);
  auto* auxiliary = output + 4 + plan.capacity * 24;
  if (plan.format == BoxDecodeWireFormat::Masks) {
    std::memcpy(auxiliary, inputs[4].data, n * plan.auxiliary_row);
    std::memset(auxiliary + n * plan.auxiliary_row, 0, (plan.capacity - n) * plan.auxiliary_row);
  } else if (plan.format == BoxDecodeWireFormat::Pose) {
    for (std::size_t i = 0; i < n * 17; ++i) {
      const auto* from = inputs[4].data + i * 12;
      auto* to = auxiliary + i * 12;
      store(to, static_cast<std::uint32_t>(load<float>(from)));
      store(to + 4, static_cast<std::uint32_t>(load<float>(from + 4)));
      std::memcpy(to + 8, from + 8, 4);
    }
    std::memset(auxiliary + n * 204, 0, (plan.capacity - n) * 204);
  }
  return {};
}
} // namespace

BoxDecodePublication prepare_boxdecode_publication(BoxDecodeWireFormat format,
                                                   std::span<const k::TensorSpec> inputs) {
  const bool features = format == BoxDecodeWireFormat::FeaturePointsV1 ||
                        format == BoxDecodeWireFormat::LegacyFeaturePointsV0;
  const bool rcnn = format == BoxDecodeWireFormat::RcnnProposals;
  const bool auxiliary =
      rcnn || format == BoxDecodeWireFormat::Pose || format == BoxDecodeWireFormat::Masks;
  if ((!features && !auxiliary && format != BoxDecodeWireFormat::Boxes) ||
      inputs.size() != (auxiliary ? 5u : 4u) || inputs[0].shape.size() != 2)
    throw std::invalid_argument("BoxDecode wire format or port count is invalid");
  const auto capacity = inputs[0].shape.dimensions()[0];
  if (capacity < 1 || capacity > 65536)
    throw std::invalid_argument("BoxDecode wire capacity must be 1..65536");
  auto plan = std::make_shared<WirePlan>();
  plan->format = format;
  plan->capacity = static_cast<std::size_t>(capacity);
  require_spec(inputs[0], (features || rcnn) ? k::DType::Float32 : k::DType::Int32,
               {capacity, features ? 2 : 4});
  require_spec(inputs[1], k::DType::Float32, {capacity});
  require_spec(inputs[3], k::DType::Int32, {1});
  if (features) {
    if (inputs[2].shape.size() != 2)
      throw std::invalid_argument("SuperPoint descriptors must have shape [K,D]");
    const auto dim = inputs[2].shape.dimensions()[1];
    if (dim < 1 || dim > UINT16_MAX ||
        (inputs[2].dtype != k::DType::Int8 && inputs[2].dtype != k::DType::BFloat16 &&
         inputs[2].dtype != k::DType::Float32))
      throw std::invalid_argument("SuperPoint descriptor dimension or dtype is unsupported");
    require_spec(inputs[2], inputs[2].dtype, {capacity, dim});
    plan->auxiliary_row = checked_mul(static_cast<std::size_t>(dim), scalar_bytes(inputs[2].dtype));
    if (format == BoxDecodeWireFormat::LegacyFeaturePointsV0) {
      if (dim != 256 || inputs[2].dtype != k::DType::Int8)
        throw std::invalid_argument("legacy SuperPoint wire requires 256 INT8 descriptors");
      plan->bytes = checked_add(4, checked_mul(plan->capacity, 264));
    } else {
      auto& h = plan->feature_header;
      h.capacity = static_cast<std::uint32_t>(capacity);
      h.descriptor_dim = static_cast<std::uint16_t>(dim);
      h.descriptor_dtype = static_cast<std::uint8_t>(
          inputs[2].dtype == k::DType::Int8       ? FeatureValueDType::Int8
          : inputs[2].dtype == k::DType::BFloat16 ? FeatureValueDType::BFloat16
                                                  : FeatureValueDType::Float32);
      h.keypoints_offset = aligned16(sizeof(h));
      h.keypoints_stride = 8;
      h.scores_offset = aligned16(checked_add(h.keypoints_offset, checked_mul(plan->capacity, 8)));
      h.scores_stride = 4;
      h.descriptors_offset =
          aligned16(checked_add(h.scores_offset, checked_mul(plan->capacity, 4)));
      h.descriptor_stride = aligned16(plan->auxiliary_row);
      plan->bytes = aligned16(
          checked_add(h.descriptors_offset, checked_mul(plan->capacity, h.descriptor_stride)));
      if (plan->bytes > UINT32_MAX)
        throw std::invalid_argument("SuperPoint wire payload exceeds uint32 header extent");
      h.total_bytes = static_cast<std::uint32_t>(plan->bytes);
    }
  } else {
    require_spec(inputs[2], k::DType::Int32, {capacity});
    if (format == BoxDecodeWireFormat::Pose) {
      require_spec(inputs[4], k::DType::Float32, {capacity, 17, 3});
      plan->auxiliary_row = 204;
    } else if (format == BoxDecodeWireFormat::Masks) {
      if (inputs[4].shape.size() != 3)
        throw std::invalid_argument("BoxDecode masks must have shape [K,H,W]");
      const auto shape = inputs[4].shape.dimensions();
      if (shape[1] < 1 || shape[2] < 1)
        throw std::invalid_argument("BoxDecode mask extents must be positive");
      require_spec(inputs[4], k::DType::UInt8, {capacity, shape[1], shape[2]});
      plan->auxiliary_row = checked_mul(shape[1], shape[2]);
    }
    if (rcnn) {
      if (capacity % 4)
        throw std::invalid_argument("RCNN wire capacity must be four times per-level top-K");
      require_spec(inputs[4], k::DType::Int32, {capacity});
      plan->bytes = checked_mul(4, checked_add(4, checked_mul(plan->capacity / 4, 12)));
    } else
      plan->bytes =
          checked_add(4, checked_mul(plan->capacity, checked_add(24, plan->auxiliary_row)));
  }
  for (const auto& input : inputs) {
    auto bytes = scalar_bytes(input.dtype);
    for (const auto dim : input.shape.dimensions())
      bytes = checked_mul(bytes, dim);
    plan->inputs.push_back({bytes, scalar_bytes(input.dtype)});
  }
  if (plan->bytes > static_cast<std::uint64_t>(INT64_MAX))
    throw std::invalid_argument("BoxDecode wire shape exceeds int64");
  const k::TensorSpec output{{static_cast<std::int64_t>(plan->bytes)}, k::DType::UInt8};
  const std::array<k::TensorRequirements, 1> requirements{{{plan->bytes, 1}}};
  auto contract = host::make_contract(inputs, std::span(&output, 1), plan->inputs, requirements);
  if (const auto* error = std::get_if<k::Error>(&contract))
    throw std::invalid_argument(error->message);
  const auto bytes = plan->bytes;
  auto executable = host::stateless_host<WirePlan, publish>(
      std::shared_ptr<const WirePlan>(std::move(plan)),
      std::get<std::shared_ptr<const host::Contract>>(std::move(contract)), {.elements = bytes});
  return {std::move(executable), output, bytes};
}
} // namespace simaai::neat::pipeline_internal::sima
