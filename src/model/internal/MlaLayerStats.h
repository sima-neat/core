#pragma once
#ifndef SIMA_NEAT_INTERNAL
#error "Internal header. Not part of the public API."
#endif

#include "model/Model.h"
#include "pipeline/internal/sima/MpkContract.h"

#include <filesystem>
#include <vector>

namespace simaai::neat::internal {

// Runs each MLA stage's ELF in per-layer mode with mla-rt and returns each layer's hardware time.
// Throws with the reason when the model, mla-rt, or its stats files are unusable.
std::vector<MlaLayerStats>
collect_mla_layer_stats(const pipeline_internal::sima::MpkContract& contract,
                        const std::filesystem::path& mla_rt);

} // namespace simaai::neat::internal
