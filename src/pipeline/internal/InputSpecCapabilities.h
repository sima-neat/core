/**
 * @file
 * @brief Pipeline-owned discovery of input-specialization capabilities.
 */
#pragma once

#include "builder/internal/InputSpecSpecialization.h"

namespace simaai::neat::pipeline_internal {

/**
 * Probe process-local backend capabilities once for a pipeline compile/build.
 *
 * Keeping discovery here preserves the dependency direction: semantic Nodes
 * consume generic facts and never inspect GStreamer themselves.
 */
simaai::neat::internal::InputSpecSpecializationContext discover_input_spec_specialization_context();

} // namespace simaai::neat::pipeline_internal
