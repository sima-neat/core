#pragma once

#include <array>
#include <cstdint>
#include <optional>
#include <variant>

namespace simaai::neat::internal {
// Core geometry policy only. Pixel sampling and numerical admission remain in Kernels.
struct CropGeometry {
  std::array<std::int64_t, 4> window; // Virtual resize width/height, center X/Y.
  std::array<double, 6> model_to_source;
};
inline std::int64_t center_crop_offset(std::int64_t difference) noexcept {
  const auto half = difference / 2;
  return half + ((difference % 2 != 0 && half % 2 != 0) ? 1 : 0);
}
inline std::variant<CropGeometry, const char*>
checked_crop_geometry(std::int64_t source_width, std::int64_t source_height,
                      std::int64_t output_width, std::int64_t output_height,
                      std::optional<int> shorter_side) {
  // Bounds precede all cross-products. Preserve the native image/window limit.
  constexpr std::int64_t limit = 4096;
  for (const auto size : {source_width, source_height, output_width, output_height})
    if (size <= 0 || size > limit)
      return "source and output crop extents must be 1..4096";
  std::int64_t width, height;
  if (shorter_side) {
    if (*shorter_side <= 0 || *shorter_side > limit)
      return "shorter_side must be 1..4096";
    if (source_width <= source_height) {
      width = *shorter_side;
      height = width * source_height / source_width;
    } else {
      height = *shorter_side;
      width = height * source_width / source_height;
    }
  } else if (output_width * source_height >= output_height * source_width) {
    width = output_width;
    height = (source_height * width + source_width - 1) / source_width;
  } else {
    height = output_height;
    width = (source_width * height + source_height - 1) / source_height;
  }
  if (width > limit || height > limit)
    return "virtual resize extents exceed 4096";
  if (width < output_width || height < output_height)
    return "shorter_side resize is smaller than the crop; padding is not permitted";
  const auto x = center_crop_offset(width - output_width);
  const auto y = center_crop_offset(height - output_height);
  const double scale_x = double(source_width) / double(width);
  const double scale_y = double(source_height) / double(height);
  return CropGeometry{{width, height, x, y},
                      {scale_x, 0., double(x) * scale_x, 0., scale_y, double(y) * scale_y}};
}
} // namespace simaai::neat::internal
