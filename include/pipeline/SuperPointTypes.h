/**
 * @file
 * @ingroup pipeline
 * @brief Stable SuperPoint numerical-profile and output-encoding selections for BoxDecode.
 */
#pragma once

#include "pipeline/TensorCore.h"

#include <cstdint>

namespace simaai::neat {

/** Numerical contract used by `BoxDecodeType::SuperPoint`.
 *
 * `Auto` is a construction-time sentinel. It selects authoritative MPK metadata when present
 * and otherwise resolves to `A65V1`; it is never inferred from tensor geometry or values and
 * must not reach the runtime backend.
 */
enum class SuperPointProfile : std::int32_t {
  Auto = 0,
  LightGlueV1 = 1,
  MagicLeapDemoV1 = 2,
  PaperBicubicV1 = 3,
  A65V1 = 4,
};

/// Byte representation emitted by the SuperPoint BoxDecode backend.
enum class SuperPointOutputFormat : std::int32_t {
  FeaturePointsV1 = 1,
  LegacyA65InterleavedV0 = 2,
};

/// SuperPoint-specific BoxDecode options. Negative spatial knobs mean "use profile/MPK default".
struct SuperPointOptions {
  // Auto preserves Model/node inheritance and authoritative MPK selection. If
  // neither supplies a profile, contract finalization selects A65V1.
  SuperPointProfile profile = SuperPointProfile::Auto;
  int nms_radius = -1;
  int border_margin = -1;
  TensorDType descriptor_output_dtype = TensorDType::Float32;
  SuperPointOutputFormat output_format = SuperPointOutputFormat::FeaturePointsV1;
};

constexpr const char* superpoint_profile_token(SuperPointProfile profile) {
  switch (profile) {
  case SuperPointProfile::Auto:
    return "auto";
  case SuperPointProfile::LightGlueV1:
    return "lightglue-v1";
  case SuperPointProfile::MagicLeapDemoV1:
    return "magic-leap-demo-v1";
  case SuperPointProfile::PaperBicubicV1:
    return "paper-bicubic-v1";
  case SuperPointProfile::A65V1:
    return "a65-v1";
  default:
    return "auto";
  }
}

constexpr const char* superpoint_output_format_token(SuperPointOutputFormat format) {
  switch (format) {
  case SuperPointOutputFormat::FeaturePointsV1:
    return "feature-points-v1";
  case SuperPointOutputFormat::LegacyA65InterleavedV0:
    return "legacy-a65-interleaved-v0";
  default:
    return "feature-points-v1";
  }
}

} // namespace simaai::neat
