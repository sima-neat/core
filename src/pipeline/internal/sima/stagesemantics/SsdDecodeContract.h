#pragma once
#ifndef SIMA_NEAT_INTERNAL
#error "Internal header. Not part of the public API."
#endif

#include "model/PreprocessPlan.h"
#include "pipeline/internal/sima/BoxDecodeStaticContractExtractor.h"
#include "pipeline/internal/sima/stagesemantics/SsdRecipeId.h"

#include <cstdint>
#include <span>
#include <string>

namespace simaai::neat::pipeline_internal::sima::stagesemantics {

struct SsdLevelSpec {
  int height = 0;
  int width = 0;
  int localization_channels = 0;
  int confidence_channels = 0;
};

enum class SsdLocalizationChannelOrder : std::uint8_t {
  AnchorMajorCoordinates,
};

enum class SsdConfidenceChannelOrder : std::uint8_t {
  ClassMajorAnchors,
  AnchorMajorClasses,
};

enum class SsdClassCountPolicy : std::uint8_t {
  Exact,
  AllowPrefixNarrowing,
};

/**
 * @brief One SSD contract implemented by both Core validation and the runtime decoder.
 *
 * The ordered level shapes describe the prepared MLA outputs, not an upstream model family.
 * All spans point at static constexpr storage owned by the registry implementation.
 */
struct SsdRecipeDescriptor {
  SsdRecipeId id = SsdRecipeId::Unknown;
  std::span<const SsdLevelSpec> ordered_levels;
  int model_width = 0;
  int model_height = 0;
  ResizeMode required_resize = ResizeMode::Stretch;
  BoxDecodeScoreActivation activation = BoxDecodeScoreActivation::Unknown;
  SsdLocalizationChannelOrder localization_order =
      SsdLocalizationChannelOrder::AnchorMajorCoordinates;
  SsdConfidenceChannelOrder confidence_order = SsdConfidenceChannelOrder::ClassMajorAnchors;
  int background_class = 0;
  int encoded_class_count = 0;
  SsdClassCountPolicy class_count_policy = SsdClassCountPolicy::Exact;
};

/// Return the descriptor keyed by a resolved Core-side recipe ID, or nullptr for Unknown.
const SsdRecipeDescriptor* find_ssd_recipe_descriptor(SsdRecipeId id);

/// Resolve and validate the complete ordered prepared-head signature. Throws on any mismatch.
const SsdRecipeDescriptor& resolve_ssd_recipe_descriptor(const BoxDecodeStaticContract& contract);

/// Human-readable ordered signature used by diagnostics and tests.
std::string ssd_observed_signature(const BoxDecodeStaticContract& contract);

} // namespace simaai::neat::pipeline_internal::sima::stagesemantics
