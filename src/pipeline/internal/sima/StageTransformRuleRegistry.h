#pragma once
#ifndef SIMA_NEAT_INTERNAL
#error "Internal header. Not part of the public API."
#endif

#include <optional>
#include <string>

namespace simaai::neat::pipeline_internal::sima {

enum class StageTensorSource {
  None = 0,
  MlaInputs = 1,
  MlaOutputs = 2,
};

struct StageTransformRule {
  StageTensorSource input_source = StageTensorSource::None;
  StageTensorSource output_source = StageTensorSource::None;
  bool propagate_output_quant = false;
};

class StageTransformRuleRegistry {
public:
  std::optional<StageTransformRule> lookup(const std::string& kernel_kind) const;
  bool is_pre_adapter(const std::string& kernel_kind) const;
  bool is_post_adapter(const std::string& kernel_kind) const;
};

const StageTransformRuleRegistry& default_stage_transform_rules();

} // namespace simaai::neat::pipeline_internal::sima
