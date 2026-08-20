#pragma once
#ifndef SIMA_NEAT_INTERNAL
#error "Internal header. Not part of the public API."
#endif

#include <cstdint>

namespace simaai::neat::pipeline_internal::sima::stagesemantics {

/// Core-side identity of an exact prepared SSD head contract.
///
/// This value never crosses the object-decode ABI. The deployed decoder accepts the stable
/// family token `ssd` and selects its matching implementation from the already validated head
/// geometry.
enum class SsdRecipeId : std::uint8_t {
  Unknown = 0,
  Ssd300V1 = 1,
  SsdMobile300V1 = 2,
  SsdMobile320V1 = 3,
  SsdLiteMobile320V1 = 4,
};

enum class SsdClassSelectionKind : std::uint8_t {
  Exact = 0,
  PrefixFromZero = 1,
};

/// Separates the confidence depth encoded by the prepared heads from the class
/// prefix requested by the application. The legacy runtime ABI continues to
/// receive `selected_count` as `num_classes`.
struct SsdClassSelection {
  int encoded_count = 0;
  int selected_count = 0;
  SsdClassSelectionKind kind = SsdClassSelectionKind::Exact;
};

constexpr const char* ssd_recipe_id_token(SsdRecipeId id) {
  switch (id) {
  case SsdRecipeId::Ssd300V1:
    return "ssd300-v1";
  case SsdRecipeId::SsdMobile300V1:
    return "ssd-mobile-300-v1";
  case SsdRecipeId::SsdMobile320V1:
    return "ssd-mobile-320-v1";
  case SsdRecipeId::SsdLiteMobile320V1:
    return "ssdlite-mobile-320-v1";
  case SsdRecipeId::Unknown:
  default:
    return "unknown";
  }
}

} // namespace simaai::neat::pipeline_internal::sima::stagesemantics
