#pragma once

#include "builder/SpecProvider.h"
#include "builder/NodeGroup.h"
#include "pipeline/TensorConversion.h"

#include <string>
#include <vector>

namespace simaai::neat {

struct NegotiationResult {
  bool ok = true;
  std::string error;
  std::vector<ConversionTrace> trace;
};

NegotiationResult validate_tensor_pipeline(const simaai::neat::NodeGroup& group,
                                           const TensorConstraint& input, ConversionPolicy policy);

} // namespace simaai::neat
