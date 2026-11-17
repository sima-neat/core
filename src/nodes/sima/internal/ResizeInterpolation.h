#pragma once

#include "nodes/sima/PreprocMath.h"

namespace simaai::neat::internal {
// One token boundary for Model and standalone Resize. Invalid enum values
// remain errors instead of silently falling back to a different sampler.
constexpr const char* resize_interpolation_token(ResizeInterpolation interpolation) noexcept {
  switch (interpolation) {
  case ResizeInterpolation::Nearest:
    return "NEAREST";
  case ResizeInterpolation::Bilinear:
    return "BILINEAR";
  case ResizeInterpolation::Bicubic:
    return "BICUBIC";
  case ResizeInterpolation::BoxAverage:
    return "INTERAREA";
  case ResizeInterpolation::BilinearAntialiased:
    return "BILINEAR_ANTIALIASED";
  }
  return nullptr;
}
} // namespace simaai::neat::internal
