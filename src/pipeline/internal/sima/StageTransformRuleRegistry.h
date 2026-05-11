/**
 * @file
 * @ingroup internal_sima
 * @brief **Framework internal — reach-through tier.** Registry of stage-transform rules.
 *
 * The route planner consults this registry to know how a given kernel-kind stage feeds off the
 * MLA's inputs vs. its outputs, and whether the rule propagates the MLA's output-quant params
 * downstream. The registry's key is the kernel-kind token (e.g., `"detess_dequant"`), and each
 * rule says where the stage's tensors come from in the surrounding route.
 */
#pragma once
#ifndef SIMA_NEAT_INTERNAL
#error "Internal header. Not part of the public API."
#endif

#include <optional>
#include <string>

namespace simaai::neat::pipeline_internal::sima {

/// Where a stage's tensors come from relative to the MLA core.
enum class StageTensorSource {
  None = 0,        ///< Stage is not anchored to the MLA.
  MlaInputs = 1,   ///< Stage feeds (or is fed by) the MLA's input tensors.
  MlaOutputs = 2,  ///< Stage feeds (or is fed by) the MLA's output tensors.
};

/**
 * @brief One entry in the stage-transform rule table.
 *
 * Tells the planner where this stage's input and output tensors live relative to the MLA, and
 * whether to propagate the MLA's output quant parameters across the stage.
 */
struct StageTransformRule {
  StageTensorSource input_source = StageTensorSource::None;   ///< Where this stage's inputs come from.
  StageTensorSource output_source = StageTensorSource::None;  ///< Where this stage's outputs go.
  bool propagate_output_quant = false;                        ///< Carry MLA output-quant params across.
};

/**
 * @brief Lookup table mapping kernel-kind tokens to their `StageTransformRule`.
 *
 * The default instance (see `default_stage_transform_rules`) covers the well-known pre- and
 * post-MLA adapter stages. New rules can be added by extending the registry singleton.
 */
class StageTransformRuleRegistry {
public:
  /// Look up the rule for `kernel_kind`; nullopt if no rule registered.
  std::optional<StageTransformRule> lookup(const std::string& kernel_kind) const;

  /// True when `kernel_kind` is a registered pre-MLA adapter (its outputs feed the MLA).
  bool is_pre_adapter(const std::string& kernel_kind) const;

  /// True when `kernel_kind` is a registered post-MLA adapter (its inputs come from the MLA).
  bool is_post_adapter(const std::string& kernel_kind) const;
};

/// Process-wide default rule registry covering all framework-recognized adapter kernels.
const StageTransformRuleRegistry& default_stage_transform_rules();

} // namespace simaai::neat::pipeline_internal::sima
