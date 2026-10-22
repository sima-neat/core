#pragma once
#ifndef SIMA_NEAT_INTERNAL
#error "Internal header. Not part of the public API."
#endif

#include "model/Model.h"
#include "pipeline/internal/sima/MpkContract.h"

#include <filesystem>
#include <vector>

namespace simaai::neat::internal {

// Runs each MLA stage's ELF uninterrupted with mla-rt and returns its mean hardware latency.
// Throws with the reason when mla-rt fails or its stats file is unusable.
std::vector<MlaStage> collect_mla_stage_stats(const pipeline_internal::sima::MpkContract& contract,
                                              const std::filesystem::path& mla_rt);

} // namespace simaai::neat::internal
