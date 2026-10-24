#pragma once
#ifndef SIMA_NEAT_INTERNAL
#error "Internal header. Not part of the public API."
#endif

#include "model/Model.h"
#include "pipeline/internal/sima/MpkContract.h"

#include <vector>

namespace simaai::neat::internal {

// Measures each MLA stage's ELF with Scheduler's mla-rt client and returns its mean hardware
// latency. Throws with the stage and reason when the model or a measurement is unusable.
std::vector<MlaStage> collect_mla_stage_stats(const pipeline_internal::sima::MpkContract& contract);

} // namespace simaai::neat::internal
