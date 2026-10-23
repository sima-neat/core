#pragma once
#ifndef SIMA_NEAT_INTERNAL
#error "Internal header. Not part of the public API."
#endif

#include "model/Model.h"
#include "pipeline/internal/sima/MpkContract.h"

#include <vector>

namespace simaai::neat::internal {

// Measures each MLA stage's ELF one layer at a time with Scheduler's mla-rt client and returns
// each layer's hardware time. Throws with the stage and reason when the model or a measurement
// is unusable.
std::vector<MlaLayer> collect_mla_layer_stats(const pipeline_internal::sima::MpkContract& contract);

} // namespace simaai::neat::internal
