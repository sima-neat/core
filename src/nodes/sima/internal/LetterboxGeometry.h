#pragma once

#include <cstdint>

namespace simaai::neat::internal {
// A Model-level top-left letterbox resizes like Segment Anything: the limiting
// axis fills the canvas and the other axis is int(extent * canvas / limiting + 0.5)
// in double. The volatile product keeps the multiply and the add separately
// rounded (no fused multiply-add), as the SAM BoxDecode kernel's crop does.
inline std::int64_t nearest_letterbox_extent(std::int64_t extent, std::int64_t canvas,
                                             std::int64_t limiting) noexcept {
  const double scale = double(canvas) / double(limiting);
  const volatile double product = double(extent) * scale;
  return std::int64_t(product + 0.5);
}
} // namespace simaai::neat::internal
