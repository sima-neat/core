#include "pipeline/internal/sima/BoxDecodePublication.h"
#include "pipeline/FeatureTypes.h"
#include "pipeline/DetectionTypes.h"

#include <simaai/scheduler/client/host_task.h>

#include <algorithm>
#if defined(__aarch64__)
#include <arm_neon.h>
#endif
#include <array>
#include <cmath>
#include <cstring>
#include <limits>
#include <stdexcept>
#include <vector>

namespace simaai::neat::pipeline_internal::sima {
namespace {
namespace k = ::simaai::scheduler::client;
// Wire encoding is IEEE binary16 on both host-contract and A65 builds. Keep
// the A65 conversion type unchanged; this is not a numerical kernel fallback.
#if defined(__aarch64__)
using WireFloat16 = float16_t;
#else
using WireFloat16 = _Float16;
#endif
static_assert(sizeof(WireFloat16) == 2);

struct WirePlan {
  BoxDecodeWireFormat format;
  std::size_t capacity = 0, bytes = 0, auxiliary_row = 0;
  std::size_t pose_points = 0, pose_input = 0;
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
  if (spec.dtype != dtype || spec.shape != std::vector<std::int64_t>(shape) ||
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
k::HostStatus publish(const WirePlan& plan, k::HostBuffers buffers) {
  if (buffers.input_count() != plan.inputs.size() || buffers.input_count() > 6 ||
      buffers.output_count() != 1)
    return k::HostError{k::ProviderErrorCode::InvalidArity, "ports",
                        "wire publication port count changed"};
  // Resolve the borrowed port descriptors once. The packing loops remain
  // local arithmetic and loads/stores, not a DSO call per scalar or row.
  std::array<k::ConstBytes, 6> inputs{};
  const auto input_count = buffers.input_count();
  for (std::size_t i = 0; i < input_count; ++i)
    inputs[i] = buffers.input(i);
  const auto target = buffers.output(0);
  if (!target.data || target.size_bytes < plan.bytes)
    return k::HostError{k::ProviderErrorCode::BufferTooSmall, "output",
                        "wire publication output is too small"};
  if (reinterpret_cast<std::uintptr_t>(target.data) > UINTPTR_MAX - plan.bytes)
    return k::HostError{k::ProviderErrorCode::SizeOverflow, "output",
                        "wire publication output address overflows"};
  for (std::size_t i = 0; i < input_count; ++i) {
    const auto bytes = plan.inputs[i].storage_bytes;
    if (!inputs[i].data || inputs[i].size_bytes < bytes)
      return k::HostError{k::ProviderErrorCode::BufferTooSmall, "input",
                          "wire publication input is too small"};
    if (reinterpret_cast<std::uintptr_t>(inputs[i].data) > UINTPTR_MAX - bytes)
      return k::HostError{k::ProviderErrorCode::SizeOverflow, "input",
                          "wire publication input address overflows"};
    if (overlaps(inputs[i].data, bytes, target.data, plan.bytes))
      return k::HostError{k::ProviderErrorCode::AliasedBuffers, "output",
                          "wire publication input/output spans overlap"};
  }
  const auto count = load<std::int32_t>(inputs[3].data);
  if (count < 0 || static_cast<std::size_t>(count) > plan.capacity)
    return k::HostError{k::ProviderErrorCode::InvalidParameter, "count",
                        "numerical result count exceeds capacity"};
  auto* output = target.data;
  const auto n = static_cast<std::size_t>(count);
  if (plan.format == BoxDecodeWireFormat::OrientedBoxesV1) {
    // Validate the complete selected result before touching the public carrier.
    for (std::size_t i = 0; i < n; ++i) {
      for (std::size_t axis = 0; axis < 5; ++axis)
        if (!std::isfinite(load<float>(inputs[0].data + (i * 5 + axis) * 4)))
          return k::HostError{k::ProviderErrorCode::InvalidParameter, "geometry",
                              "oriented geometry is nonfinite"};
      const auto score = load<float>(inputs[1].data + i * 4);
      if (!std::isfinite(score) || score < 0 || score > 1 ||
          load<std::int32_t>(inputs[2].data + i * 4) < 0)
        return k::HostError{k::ProviderErrorCode::InvalidParameter, "records",
                            "oriented score or class is invalid"};
    }
    OrientedBoxesHeaderV1 header;
    header.total_bytes = static_cast<std::uint32_t>(plan.bytes);
    header.capacity = static_cast<std::uint32_t>(plan.capacity);
    header.count = static_cast<std::uint32_t>(n);
    std::memset(output, 0, plan.bytes);
    store(output, header);
    for (std::size_t i = 0; i < n; ++i) {
      auto* row = output + sizeof(header) + i * sizeof(OrientedBoxRecordV1);
      std::memcpy(row, inputs[0].data + i * 20, 20);
      std::memcpy(row + 20, inputs[1].data + i * 4, 4);
      std::memcpy(row + 24, inputs[2].data + i * 4, 4);
    }
    return {};
  }
  if (plan.format == BoxDecodeWireFormat::RcnnProposals) {
    const auto per_level = plan.capacity / 4;
    const auto group_bytes = 4 + per_level * 12;
    std::array<std::int32_t, 4> counts{};
    // Validate every bucket before modifying the public buffer. Level selection
    // belongs to the numerical kernel; this only packs its FP32 proposal rows.
    for (std::size_t i = 0; i < n; ++i) {
      const auto level = load<std::int32_t>(inputs[4].data + i * 4);
      if (level < 0 || level >= 4 || ++counts[level] > static_cast<std::int32_t>(per_level))
        return k::HostError{k::ProviderErrorCode::InvalidParameter, "levels",
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
        store(row + field * 2, static_cast<WireFloat16>(values[field]));
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
        return k::HostError{k::ProviderErrorCode::InvalidParameter, "coordinates",
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
  if (plan.pose_points) {
    // Coordinates were already rounded/clipped by the numerical kernel. Check
    // representability before writing; converting is wire encoding, not math.
    for (std::size_t i = 0; i < n * plan.pose_points; ++i)
      for (std::size_t axis = 0; axis < 2; ++axis) {
        const float value = load<float>(inputs[plan.pose_input].data + (i * 3 + axis) * 4);
        if (!std::isfinite(value) || value < 0.f || double(value) > UINT32_MAX)
          return k::HostError{k::ProviderErrorCode::InvalidParameter, "coordinates",
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
  if (plan.format == BoxDecodeWireFormat::Masks || plan.format == BoxDecodeWireFormat::MasksPose) {
    std::memcpy(auxiliary, inputs[4].data, n * plan.auxiliary_row);
    std::memset(auxiliary + n * plan.auxiliary_row, 0, (plan.capacity - n) * plan.auxiliary_row);
    auxiliary += plan.capacity * plan.auxiliary_row;
  }
  if (plan.pose_points) {
    // Public wire capacity is fixed at 17 points, independently of canonical P.
    // Copy only the shared survivor rows and clear every unused point/tail.
    for (std::size_t i = 0; i < n; ++i) {
      for (std::size_t point = 0; point < plan.pose_points; ++point) {
        const auto* from = inputs[plan.pose_input].data + (i * plan.pose_points + point) * 12;
        auto* to = auxiliary + i * 204 + point * 12;
        store(to, static_cast<std::uint32_t>(load<float>(from)));
        store(to + 4, static_cast<std::uint32_t>(load<float>(from + 4)));
        std::memcpy(to + 8, from + 8, 4);
      }
      std::memset(auxiliary + i * 204 + plan.pose_points * 12, 0, 204 - plan.pose_points * 12);
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
  const bool oriented = format == BoxDecodeWireFormat::OrientedBoxesV1;
  const bool combined = format == BoxDecodeWireFormat::MasksPose;
  const bool pose = format == BoxDecodeWireFormat::Pose || combined;
  const bool masks = format == BoxDecodeWireFormat::Masks || combined;
  const bool auxiliary = rcnn || pose || masks;
  if ((!features && !auxiliary && !oriented && format != BoxDecodeWireFormat::Boxes) ||
      inputs.size() != (combined    ? 6u
                        : auxiliary ? 5u
                                    : 4u) ||
      inputs[0].shape.size() != 2)
    throw std::invalid_argument("BoxDecode wire format or port count is invalid");
  const auto capacity = inputs[0].shape[0];
  if (capacity < 1 || capacity > 65536)
    throw std::invalid_argument("BoxDecode wire capacity must be 1..65536");
  auto plan = std::make_shared<WirePlan>();
  plan->format = format;
  plan->capacity = static_cast<std::size_t>(capacity);
  require_spec(inputs[0], (features || rcnn || oriented) ? k::DType::Float32 : k::DType::Int32,
               {capacity, features   ? 2
                          : oriented ? 5
                                     : 4});
  require_spec(inputs[1], k::DType::Float32, {capacity});
  require_spec(inputs[3], k::DType::Int32, {1});
  if (features) {
    if (inputs[2].shape.size() != 2)
      throw std::invalid_argument("SuperPoint descriptors must have shape [K,D]");
    const auto dim = inputs[2].shape[1];
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
    if (pose) {
      plan->pose_input = combined ? 5 : 4;
      const auto& points = inputs[plan->pose_input];
      if (points.shape.size() != 3)
        throw std::invalid_argument("BoxDecode keypoints must have shape [K,P,3]");
      const auto count = points.shape[1];
      if (count < 1 || count > 17 || (!combined && count != 17))
        throw std::invalid_argument("BoxDecode wire requires 1..17 combined or 17 pose points");
      require_spec(points, k::DType::Float32, {capacity, count, 3});
      plan->pose_points = static_cast<std::size_t>(count);
    }
    if (masks) {
      if (inputs[4].shape.size() != 3)
        throw std::invalid_argument("BoxDecode masks must have shape [K,H,W]");
      const auto shape = inputs[4].shape;
      if (shape[1] < 1 || shape[2] < 1 || (combined && (shape[1] != 160 || shape[2] != 160)))
        throw std::invalid_argument(
            "BoxDecode mask extents must be positive; combined wire requires 160x160");
      require_spec(inputs[4], k::DType::UInt8, {capacity, shape[1], shape[2]});
      plan->auxiliary_row = checked_mul(shape[1], shape[2]);
    }
    if (oriented) {
      plan->bytes = checked_add(32, checked_mul(plan->capacity, 32));
    } else if (rcnn) {
      if (capacity % 4)
        throw std::invalid_argument("RCNN wire capacity must be four times per-level top-K");
      require_spec(inputs[4], k::DType::Int32, {capacity});
      plan->bytes = checked_mul(4, checked_add(4, checked_mul(plan->capacity / 4, 12)));
    } else
      plan->bytes = checked_add(
          4, checked_mul(plan->capacity,
                         checked_add(checked_add(24, plan->auxiliary_row), pose ? 204 : 0)));
  }
  for (const auto& input : inputs) {
    auto bytes = scalar_bytes(input.dtype);
    for (const auto dim : input.shape)
      bytes = checked_mul(bytes, dim);
    plan->inputs.push_back({bytes, scalar_bytes(input.dtype)});
  }
  if (plan->bytes > static_cast<std::uint64_t>(INT64_MAX))
    throw std::invalid_argument("BoxDecode wire shape exceeds int64");
  const k::TensorSpec output{{static_cast<std::int64_t>(plan->bytes)}, k::DType::UInt8};
  const std::array<k::TensorRequirements, 1> requirements{{{plan->bytes, 1}}};
  const auto bytes = plan->bytes;
  k::HostTaskRequest request;
  request.inputs = inputs;
  request.outputs = std::span(&output, 1);
  request.input_requirements = plan->inputs;
  request.output_requirements = requirements;
  request.elements = bytes;
  request.payload = std::shared_ptr<const WirePlan>(std::move(plan));
  request.callback = +[](const void* payload, k::HostBuffers buffers) -> k::HostStatus {
    return publish(*static_cast<const WirePlan*>(payload), buffers);
  };
  auto checked = k::make_host_task(request);
  if (const auto* error = std::get_if<k::Error>(&checked))
    throw std::invalid_argument(error->message);
  return {std::get<k::KernelExecutable>(std::move(checked)), output, bytes};
}
} // namespace simaai::neat::pipeline_internal::sima
