#include "pipeline/internal/sima/StageTransformRuleRegistry.h"

#include "pipeline/internal/TensorMath.h"

#include <optional>
#include <unordered_set>
#include <unordered_map>

namespace simaai::neat::pipeline_internal::sima {
namespace {

const std::unordered_set<std::string> kPreRules = {"preproc", "quanttess", "quantize",
                                                   "tessellate"};
const std::unordered_set<std::string> kPostRules = {"detessdequant", "detessellate", "dequantize",
                                                    "boxdecode"};
const std::unordered_map<std::string, StageTransformRule> kRules = {
    {"preproc", StageTransformRule{StageTensorSource::None, StageTensorSource::MlaInputs, false}},
    {"quanttess", StageTransformRule{StageTensorSource::None, StageTensorSource::MlaInputs, false}},
    {"quantize", StageTransformRule{StageTensorSource::None, StageTensorSource::MlaInputs, false}},
    {"tessellate",
     StageTransformRule{StageTensorSource::None, StageTensorSource::MlaInputs, false}},
    {"detessdequant",
     StageTransformRule{StageTensorSource::MlaOutputs, StageTensorSource::None, true}},
    {"detessellate",
     StageTransformRule{StageTensorSource::MlaOutputs, StageTensorSource::None, true}},
    {"dequantize",
     StageTransformRule{StageTensorSource::MlaOutputs, StageTensorSource::None, true}},
    {"boxdecode", StageTransformRule{StageTensorSource::MlaOutputs, StageTensorSource::None, true}},
};

} // namespace

std::optional<StageTransformRule>
StageTransformRuleRegistry::lookup(const std::string& kernel_kind) const {
  const auto it = kRules.find(lower_copy(kernel_kind));
  if (it == kRules.end()) {
    return std::nullopt;
  }
  return it->second;
}

bool StageTransformRuleRegistry::is_pre_adapter(const std::string& kernel_kind) const {
  return kPreRules.find(lower_copy(kernel_kind)) != kPreRules.end();
}

bool StageTransformRuleRegistry::is_post_adapter(const std::string& kernel_kind) const {
  return kPostRules.find(lower_copy(kernel_kind)) != kPostRules.end();
}

const StageTransformRuleRegistry& default_stage_transform_rules() {
  static const StageTransformRuleRegistry rules;
  return rules;
}

} // namespace simaai::neat::pipeline_internal::sima
