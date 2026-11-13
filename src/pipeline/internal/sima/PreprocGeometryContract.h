#pragma once
#ifndef SIMA_NEAT_INTERNAL
#error "Internal header. Not part of the public API."
#endif
#include <algorithm>
#include <array>
#include <cstddef>
#include <span>
#include <string_view>

namespace simaai::neat::pipeline_internal::sima::kernel_lowering {
inline constexpr std::size_t preproc_legacy_geometry_field_count = 4;
inline constexpr std::array<std::string_view, 5> preproc_geometry_fields{
    "source_roi", "source_rois", "destination", "padding", "resize_window"};

// Legacy four-slot updates clear the optional trailing window. Unknown schemas
// fail closed; Crop requires the full five-slot capability.
inline bool supports_preproc_geometry_fields(std::span<const std::string_view> fields,
                                             bool require_window = false) {
  if (fields.size() != preproc_geometry_fields.size() &&
      (require_window || fields.size() != preproc_legacy_geometry_field_count))
    return false;
  return std::equal(fields.begin(), fields.end(), preproc_geometry_fields.begin());
}
} // namespace simaai::neat::pipeline_internal::sima::kernel_lowering
