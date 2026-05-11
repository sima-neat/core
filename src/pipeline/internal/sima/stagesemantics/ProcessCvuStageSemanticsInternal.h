#pragma once
#ifndef SIMA_NEAT_INTERNAL
#error "Internal header. Not part of the public API."
#endif

#include "pipeline/internal/sima/stagesemantics/ProcessCvuStageSemantics.h"

#include <string>
#include <vector>

namespace simaai::neat::pipeline_internal::sima::stagesemantics {

std::string canonical_family_name_internal(std::string graph_family);
ProcessCvuGraphFamily family_enum_from_name_internal(const std::string& graph_family);
void synthesize_runtime_output_arrays_from_payload_internal(ProcessCvuStagePayload* payload);
void canonicalize_preproc_single_handoff_payload_internal(ProcessCvuStagePayload* payload);
ProcessCvuCanonicalFacts build_preproc_facts_from_payload_internal(
    const ProcessCvuStagePayload& payload);
ProcessCvuCanonicalFacts build_single_io_processcvu_facts_from_payload_internal(
    const ProcessCvuStagePayload& payload);
ProcessCvuCanonicalFacts build_multi_io_processcvu_facts_from_payload_internal(
    const ProcessCvuStagePayload& payload,
    const std::vector<std::string>& runtime_input_names);

} // namespace simaai::neat::pipeline_internal::sima::stagesemantics
