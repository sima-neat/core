#pragma once
#ifndef SIMA_NEAT_INTERNAL
#error "Internal header. Not part of the public API."
#endif
#include <algorithm>
#include <array>
#include <span>
#include <string_view>

namespace simaai::neat::pipeline_internal::sima::kernel_lowering {
inline constexpr std::array<std::string_view, 4> preproc_geometry_fields{
    "source_roi", "source_rois", "destination", "padding"};

// The only supported extension documents an omittable trailing resize window.
// Legacy four-slot updates clear that window. Unknown extensions fail closed.
inline bool supports_preproc_geometry_fields(std::span<const std::string_view> fields) {
  return (fields.size() == preproc_geometry_fields.size() ||
          (fields.size() == preproc_geometry_fields.size() + 1 &&
           fields.back() == "resize_window")) &&
         std::equal(preproc_geometry_fields.begin(), preproc_geometry_fields.end(), fields.begin());
}
} // namespace simaai::neat::pipeline_internal::sima::kernel_lowering
