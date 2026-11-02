#pragma once

#include <linux/videodev2.h>
#include <cstdint>

namespace simaai::neat::internal::uvc {
// V4L2 intervals are reciprocal frame rates. Widen before multiplication: a
// step test can contain three independent 32-bit factors. No floating-point
// rounding or integer-fps approximation participates in mode admission.
inline bool interval_contains(const v4l2_frmivalenum& interval, std::uint32_t fps_num,
                              std::uint32_t fps_den) {
  if (!fps_num || !fps_den)
    return false;
  const auto valid = [](v4l2_fract f) { return f.numerator && f.denominator; };
  const auto equal = [](v4l2_fract a, v4l2_fract b) {
    return std::uint64_t(a.numerator) * b.denominator == std::uint64_t(b.numerator) * a.denominator;
  };
  const v4l2_fract requested{fps_den, fps_num};
  if (interval.type == V4L2_FRMIVAL_TYPE_DISCRETE)
    return valid(interval.discrete) && equal(requested, interval.discrete);
  if (interval.type != V4L2_FRMIVAL_TYPE_STEPWISE && interval.type != V4L2_FRMIVAL_TYPE_CONTINUOUS)
    return false;
  const auto& s = interval.stepwise;
  if (!valid(s.min) || !valid(s.max) ||
      std::uint64_t(fps_den) * s.min.denominator < std::uint64_t(fps_num) * s.min.numerator ||
      std::uint64_t(fps_den) * s.max.denominator > std::uint64_t(fps_num) * s.max.numerator)
    return false;
  if (interval.type == V4L2_FRMIVAL_TYPE_CONTINUOUS)
    return true;
  if (!valid(s.step))
    return false;
  const __uint128_t delta =
      __uint128_t(fps_den) * s.min.denominator - __uint128_t(fps_num) * s.min.numerator;
  return (delta * s.step.denominator) %
             (__uint128_t(fps_num) * s.min.denominator * s.step.numerator) ==
         0;
}
} // namespace simaai::neat::internal::uvc
