#include "pipeline/internal/sima/stagesemantics/SsdDecodeContract.h"

#include <array>
#include <optional>
#include <sstream>
#include <stdexcept>
#include <string>

namespace simaai::neat::pipeline_internal::sima::stagesemantics {
namespace {

constexpr std::array<SsdLevelSpec, 6> kSsd300V1Levels = {{
    {38, 38, 16, 324},
    {19, 19, 24, 486},
    {10, 10, 24, 486},
    {5, 5, 24, 486},
    {3, 3, 16, 324},
    {1, 1, 16, 324},
}};

constexpr std::array<SsdLevelSpec, 6> kSsdMobile300V1Levels = {{
    {19, 19, 12, 273},
    {10, 10, 24, 546},
    {5, 5, 24, 546},
    {3, 3, 24, 546},
    {2, 2, 24, 546},
    {1, 1, 24, 546},
}};

constexpr std::array<SsdLevelSpec, 6> kSsdMobile320V1Levels = {{
    {20, 20, 12, 273},
    {10, 10, 24, 546},
    {5, 5, 24, 546},
    {3, 3, 24, 546},
    {2, 2, 24, 546},
    {1, 1, 24, 546},
}};

constexpr std::array<SsdRecipeDescriptor, 3> kSsdRecipes = {{
    {SsdRecipeId::Ssd300V1, kSsd300V1Levels, 300, 300, ResizeMode::Stretch,
     BoxDecodeScoreActivation::Softmax, SsdLocalizationChannelOrder::AnchorMajorCoordinates,
     SsdConfidenceChannelOrder::ClassMajorAnchors, 0, 81,
     SsdClassCountPolicy::AllowPrefixNarrowing},
    {SsdRecipeId::SsdMobile300V1, kSsdMobile300V1Levels, 300, 300, ResizeMode::Stretch,
     BoxDecodeScoreActivation::Sigmoid, SsdLocalizationChannelOrder::AnchorMajorCoordinates,
     SsdConfidenceChannelOrder::AnchorMajorClasses, 0, 91, SsdClassCountPolicy::Exact},
    {SsdRecipeId::SsdMobile320V1, kSsdMobile320V1Levels, 320, 320, ResizeMode::Stretch,
     BoxDecodeScoreActivation::Sigmoid, SsdLocalizationChannelOrder::AnchorMajorCoordinates,
     SsdConfidenceChannelOrder::AnchorMajorClasses, 0, 91, SsdClassCountPolicy::Exact},
}};

constexpr bool same_ordered_signature(const SsdRecipeDescriptor& lhs,
                                      const SsdRecipeDescriptor& rhs) {
  if (lhs.encoded_class_count != rhs.encoded_class_count ||
      lhs.ordered_levels.size() != rhs.ordered_levels.size()) {
    return false;
  }
  for (std::size_t i = 0; i < lhs.ordered_levels.size(); ++i) {
    const auto& a = lhs.ordered_levels[i];
    const auto& b = rhs.ordered_levels[i];
    if (a.height != b.height || a.width != b.width ||
        a.localization_channels != b.localization_channels ||
        a.confidence_channels != b.confidence_channels) {
      return false;
    }
  }
  return true;
}

constexpr bool valid_registry() {
  for (std::size_t i = 0; i < kSsdRecipes.size(); ++i) {
    const auto& recipe = kSsdRecipes[i];
    if (recipe.id == SsdRecipeId::Unknown || recipe.ordered_levels.empty() ||
        recipe.model_width <= 0 || recipe.model_height <= 0 ||
        recipe.required_resize != ResizeMode::Stretch || recipe.encoded_class_count <= 0 ||
        recipe.background_class < 0 || recipe.background_class >= recipe.encoded_class_count ||
        recipe.activation == BoxDecodeScoreActivation::Unknown) {
      return false;
    }
    for (const auto& level : recipe.ordered_levels) {
      if (level.height <= 0 || level.width <= 0 || level.localization_channels <= 0 ||
          (level.localization_channels % 4) != 0 || level.confidence_channels <= 0 ||
          level.confidence_channels !=
              (level.localization_channels / 4) * recipe.encoded_class_count) {
        return false;
      }
    }
    for (std::size_t j = i + 1U; j < kSsdRecipes.size(); ++j) {
      if (recipe.id == kSsdRecipes[j].id || same_ordered_signature(recipe, kSsdRecipes[j])) {
        return false;
      }
    }
  }
  return true;
}

static_assert(valid_registry(), "SSD recipe registry contains an invalid or ambiguous profile");

struct TensorHwc {
  int height = 0;
  int width = 0;
  int channels = 0;
};

std::optional<TensorHwc> logical_hwc(const BoxDecodeTensorStaticContract& tensor) {
  if (tensor.input_shape.size() < 3U) {
    return std::nullopt;
  }
  const std::size_t rank = tensor.input_shape.size();
  const int height = tensor.input_shape[rank - 3U];
  const int width = tensor.input_shape[rank - 2U];
  const int physical_channels = tensor.input_shape[rank - 1U];
  if (height <= 0 || width <= 0 || physical_channels <= 0) {
    return std::nullopt;
  }

  int logical_channels = physical_channels;
  if (!tensor.slice_shape.empty()) {
    if (tensor.slice_shape.size() < 3U) {
      return std::nullopt;
    }
    const int sliced_channels = tensor.slice_shape.back();
    if (sliced_channels <= 0 || sliced_channels > physical_channels) {
      return std::nullopt;
    }
    logical_channels = sliced_channels;
  }
  // slice_shape H/W describe the packed tensor's tile/stripe geometry (captured SSD heads
  // commonly use slice_height=1), not a smaller logical feature grid. input_shape H/W are the
  // complete per-frame grid; only the sliced channel depth removes physical channel padding.
  return TensorHwc{height, width, logical_channels};
}

std::string format_levels(std::span<const SsdLevelSpec> levels, bool localization) {
  std::ostringstream out;
  out << '[';
  for (std::size_t i = 0; i < levels.size(); ++i) {
    if (i != 0U) {
      out << ',';
    }
    const auto& level = levels[i];
    out << level.height << 'x' << level.width << 'x'
        << (localization ? level.localization_channels : level.confidence_channels);
  }
  out << ']';
  return out.str();
}

std::string supported_signatures() {
  std::ostringstream out;
  for (std::size_t i = 0; i < kSsdRecipes.size(); ++i) {
    if (i != 0U) {
      out << "; ";
    }
    const auto& recipe = kSsdRecipes[i];
    out << ssd_recipe_id_token(recipe.id) << " loc=" << format_levels(recipe.ordered_levels, true)
        << " conf=" << format_levels(recipe.ordered_levels, false);
  }
  return out.str();
}

struct ObservedSsdSignature {
  std::array<SsdLevelSpec, 6> levels{};
  std::size_t level_count = 0;
  int encoded_class_count = 0;
};

[[noreturn]] void throw_malformed(const std::string& detail,
                                  const BoxDecodeStaticContract& contract) {
  throw std::invalid_argument(
      "SSD BoxDecode: malformed grouped-by-role head contract: " + detail + ". Observed " +
      ssd_observed_signature(contract) +
      ". Expected binding order is loc[0..5] followed by confidence[0..5].");
}

ObservedSsdSignature observe_ssd_signature(const BoxDecodeStaticContract& contract) {
  if (contract.tensors.size() != 12U) {
    throw_malformed("expected exactly 12 tensors (six localization plus six confidence), got " +
                        std::to_string(contract.tensors.size()),
                    contract);
  }

  ObservedSsdSignature observed;
  observed.level_count = contract.tensors.size() / 2U;
  std::optional<int> encoded_classes;
  for (std::size_t i = 0; i < observed.level_count; ++i) {
    const auto loc = logical_hwc(contract.tensors[i]);
    const auto conf = logical_hwc(contract.tensors[i + observed.level_count]);
    if (!loc.has_value()) {
      throw_malformed("localization tensor " + std::to_string(i) +
                          " does not have a positive logical HWC shape",
                      contract);
    }
    if (!conf.has_value()) {
      throw_malformed("confidence tensor " + std::to_string(i) +
                          " does not have a positive logical HWC shape",
                      contract);
    }
    if (loc->height != conf->height || loc->width != conf->width) {
      throw_malformed("level " + std::to_string(i) + " localization/confidence grids differ (" +
                          std::to_string(loc->height) + "x" + std::to_string(loc->width) +
                          " versus " + std::to_string(conf->height) + "x" +
                          std::to_string(conf->width) + ')',
                      contract);
    }
    if (loc->channels < 4 || (loc->channels % 4) != 0) {
      throw_malformed("localization tensor " + std::to_string(i) + " channel depth " +
                          std::to_string(loc->channels) + " is not 4 * priors_per_cell",
                      contract);
    }
    const int priors_per_cell = loc->channels / 4;
    if (conf->channels <= 0 || (conf->channels % priors_per_cell) != 0) {
      throw_malformed("confidence tensor " + std::to_string(i) + " channel depth " +
                          std::to_string(conf->channels) + " is not divisible by " +
                          std::to_string(priors_per_cell) + " priors_per_cell",
                      contract);
    }
    const int classes = conf->channels / priors_per_cell;
    if (!encoded_classes.has_value()) {
      encoded_classes = classes;
    } else if (*encoded_classes != classes) {
      throw_malformed("confidence tensor " + std::to_string(i) + " encodes " +
                          std::to_string(classes) + " classes but earlier levels encode " +
                          std::to_string(*encoded_classes),
                      contract);
    }
    observed.levels[i] = SsdLevelSpec{loc->height, loc->width, loc->channels, conf->channels};
  }
  observed.encoded_class_count = encoded_classes.value_or(0);
  return observed;
}

bool matches(const ObservedSsdSignature& observed, const SsdRecipeDescriptor& recipe) {
  if (observed.level_count != recipe.ordered_levels.size() ||
      observed.encoded_class_count != recipe.encoded_class_count) {
    return false;
  }
  for (std::size_t i = 0; i < observed.level_count; ++i) {
    const auto& actual = observed.levels[i];
    const auto& expected = recipe.ordered_levels[i];
    if (actual.height != expected.height || actual.width != expected.width ||
        actual.localization_channels != expected.localization_channels ||
        actual.confidence_channels != expected.confidence_channels) {
      return false;
    }
  }
  return true;
}

std::string format_observed(const ObservedSsdSignature& observed) {
  return "loc=" +
         format_levels(std::span<const SsdLevelSpec>(observed.levels.data(), observed.level_count),
                       true) +
         " conf=" +
         format_levels(std::span<const SsdLevelSpec>(observed.levels.data(), observed.level_count),
                       false) +
         " encoded_classes=" + std::to_string(observed.encoded_class_count);
}

} // namespace

const SsdRecipeDescriptor* find_ssd_recipe_descriptor(SsdRecipeId id) {
  for (const auto& recipe : kSsdRecipes) {
    if (recipe.id == id) {
      return &recipe;
    }
  }
  return nullptr;
}

const SsdRecipeDescriptor& resolve_ssd_recipe_descriptor(const BoxDecodeStaticContract& contract) {
  if (!box_decode_type_is_ssd_family(contract.decode_type)) {
    throw std::invalid_argument("SSD BoxDecode resolver requires an SSD-family decode type");
  }

  const ObservedSsdSignature observed = observe_ssd_signature(contract);
  if (const auto* requested = find_ssd_recipe_descriptor(contract.ssd_recipe_id)) {
    if (matches(observed, *requested)) {
      return *requested;
    }
    throw std::invalid_argument(
        "SSD BoxDecode: ordered head signature does not match requested profile '" +
        std::string(ssd_recipe_id_token(requested->id)) + "'. Observed " +
        format_observed(observed) +
        "; expected loc=" + format_levels(requested->ordered_levels, true) +
        " conf=" + format_levels(requested->ordered_levels, false) + '.');
  }

  for (const auto& recipe : kSsdRecipes) {
    if (matches(observed, recipe)) {
      return recipe;
    }
  }

  throw std::invalid_argument(
      "SSD BoxDecode: unsupported ordered head signature. Observed " + format_observed(observed) +
      ". Supported profiles: " + supported_signatures() +
      ". Expected binding order is loc[0..5] followed by confidence[0..5]; levels are not "
      "reordered.");
}

std::string ssd_observed_signature(const BoxDecodeStaticContract& contract) {
  if (contract.tensors.empty()) {
    return "loc=[] conf=[]";
  }
  const std::size_t levels = contract.tensors.size() / 2U;
  std::ostringstream loc;
  std::ostringstream conf;
  loc << "loc=[";
  conf << "conf=[";
  for (std::size_t i = 0; i < contract.tensors.size(); ++i) {
    const auto shape = logical_hwc(contract.tensors[i]);
    std::ostringstream* target = i < levels ? &loc : &conf;
    const std::size_t role_index = i < levels ? i : i - levels;
    if (role_index != 0U) {
      *target << ',';
    }
    if (!shape.has_value()) {
      *target << "invalid";
    } else {
      *target << shape->height << 'x' << shape->width << 'x' << shape->channels;
    }
  }
  loc << ']';
  conf << ']';
  return loc.str() + ' ' + conf.str();
}

} // namespace simaai::neat::pipeline_internal::sima::stagesemantics
