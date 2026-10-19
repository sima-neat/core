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

bool node_has_input_spec_specialization(const Node& node);

internal::SpecializedNodeSequence
specialize_pipeline_nodes_for_input(std::span<const std::shared_ptr<Node>> nodes,
                                    const OutputSpec& input,
                                    const internal::InputSpecSpecializationContext& context);

} // namespace simaai::neat::pipeline_internal
