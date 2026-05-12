/**
 * @file
 * @ingroup builder
 * @brief Stage-declared preprocess metadata runtime requirements.
 *
 * Preprocess Nodes (and similar transforms) declare what they need to find on
 * incoming GstBuffer's `GstSimaMeta` to operate correctly: which named fields
 * must be present, and whether specific transforms are expected to be active
 * upstream (resize, normalize, quantize, tessellate). The runtime checks
 * these requirements against actual buffer metadata before processing.
 */
#pragma once

#include <optional>
#include <string>
#include <vector>

namespace simaai::neat {

/**
 * @brief Per-stage declaration of GstSimaMeta fields and upstream transforms required at runtime.
 *
 * Populated by Nodes that depend on metadata from earlier preprocess stages
 * (e.g., a tessellate stage needing the quantization params from the
 * upstream quantize stage). The runtime uses this to assert the pipeline is
 * wired correctly before processing the first buffer.
 *
 * @ingroup builder
 * @see PreprocessMetaRequirementProvider
 */
struct PreprocessMetaRequirement {
  std::string stage_name;                       ///< Logical stage name reporting the requirement.
  std::string plugin_name;                      ///< Backing GStreamer plugin name.
  std::vector<std::string> required_fields;     ///< Names of GstSimaMeta fields that must be present.
  std::optional<bool> expect_resize;            ///< If set, expects an upstream resize stage.
  std::optional<bool> expect_normalize;         ///< If set, expects an upstream normalize stage.
  std::optional<bool> expect_quantize;          ///< If set, expects an upstream quantize stage.
  std::optional<bool> expect_tessellate;        ///< If set, expects an upstream tessellate stage.
};

/**
 * @brief Mixin interface implemented by Nodes that declare preprocess-meta requirements.
 *
 * Nodes that don't have any requirement return `std::nullopt`. The Builder
 * collects requirements across the pipeline and validates them as part of
 * contract checks.
 *
 * @ingroup builder
 * @see PreprocessMetaRequirement
 */
class PreprocessMetaRequirementProvider {
public:
  virtual ~PreprocessMetaRequirementProvider() = default;

  /// @brief Return this Node's metadata requirement, or `std::nullopt` if none.
  virtual std::optional<PreprocessMetaRequirement> preprocess_meta_requirement() const = 0;
};

} // namespace simaai::neat
