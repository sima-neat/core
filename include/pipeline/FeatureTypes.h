/**
 * @file
 * @ingroup pipeline
 * @brief Versioned, bounds-checked SuperPoint feature-point wire format and parsers.
 */
#pragma once

#include "pipeline/SuperPointTypes.h"
#include "pipeline/Tensor.h"
#include "pipeline/TensorCore.h"

#include <cstdint>
#include <string>
#include <vector>

namespace simaai::neat {

inline constexpr char kFeatureFormatPointsV1[] = "FEATURE_POINTS_V1";
inline constexpr char kFeatureFormatLegacyA65V0[] = "FEATURE_POINTS_LEGACY_A65_V0";
inline constexpr std::uint32_t kFeaturePointsMagicV1 = 0x31565046U; // `FPV1`, little endian.

enum class FeatureValueDType : std::uint8_t {
  Int8 = 1,
  BFloat16 = 2,
  Float32 = 3,
};

enum class FeaturePointsLayout : std::uint8_t {
  StructureOfArrays = 1,
};

/** Fixed prefix of the FEATURE_POINTS_V1 payload. All offsets are from payload byte zero. */
struct FeaturePointsHeaderV1 {
  std::uint32_t magic = kFeaturePointsMagicV1;
  std::uint16_t version = 1;
  std::uint16_t header_bytes = sizeof(FeaturePointsHeaderV1);
  std::uint32_t total_bytes = 0;
  std::uint32_t count = 0;
  std::uint32_t capacity = 0;
  std::uint16_t descriptor_dim = 0;
  std::uint8_t coordinate_dtype = static_cast<std::uint8_t>(FeatureValueDType::Float32);
  std::uint8_t score_dtype = static_cast<std::uint8_t>(FeatureValueDType::Float32);
  std::uint8_t descriptor_dtype = static_cast<std::uint8_t>(FeatureValueDType::Float32);
  std::uint8_t layout = static_cast<std::uint8_t>(FeaturePointsLayout::StructureOfArrays);
  std::uint16_t flags = 0;
  std::uint32_t keypoints_offset = 0;
  std::uint32_t keypoints_stride = 0;
  std::uint32_t scores_offset = 0;
  std::uint32_t scores_stride = 0;
  std::uint32_t descriptors_offset = 0;
  std::uint32_t descriptor_stride = 0;
  std::uint32_t reserved[4]{};
};

static_assert(sizeof(FeaturePointsHeaderV1) == 68,
              "FEATURE_POINTS_V1 fixed header ABI must remain 68 bytes");

struct FeaturePointTensors {
  Tensor keypoints;   ///< FP32 `[N,2]` image coordinates.
  Tensor scores;      ///< FP32 `[N]` detector scores.
  Tensor descriptors; ///< INT8/BF16/FP32 `[N,D]` normalized descriptors.
};

using FeaturePointTensorList = std::vector<FeaturePointTensors>;

/// Parse a V1 tensor, or the explicit legacy V0 representation when its feature tag says so.
FeaturePointTensors decode_superpoint_tensor(const Tensor& tensor);

/// Positional one-to-one list form of `decode_superpoint_tensor`.
FeaturePointTensorList decode_superpoint(const TensorList& tensors);

void tag_feature_format(Tensor& tensor, std::string format);
std::string read_feature_format(const Tensor& tensor);

struct Sample;
void tag_feature_format_in_sample(Sample& sample);

} // namespace simaai::neat
