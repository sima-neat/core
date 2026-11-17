#pragma once
#ifndef SIMA_NEAT_INTERNAL
#error "Internal header. Not part of the public API."
#endif

#include "pipeline/ExecutionPlan.h"
#include "pipeline/internal/SegmentBuildPlan.h"

#include <string>

namespace simaai::neat::internal {

struct ExecutionPlanAccess {
  /// Describe the Model build recorded in `build` as an execution plan.
  static ::simaai::neat::ExecutionPlan make(const pipeline_internal::SegmentBuildPlan& build,
                                            const std::string& model_name);
};

} // namespace simaai::neat::internal
