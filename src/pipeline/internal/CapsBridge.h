#pragma once
#ifndef SIMA_NEAT_INTERNAL
#error "Internal header. Not part of the public API."
#endif

#include "pipeline/TensorSpec.h"

#include <gst/gst.h>

namespace simaai::neat::pipeline_internal {

simaai::neat::TensorConstraint tensor_constraint_from_caps(GstCaps* caps);
std::string tensor_constraint_debug_string(const simaai::neat::TensorConstraint& constraint);

} // namespace simaai::neat::pipeline_internal
