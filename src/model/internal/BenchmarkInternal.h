#pragma once
#ifndef SIMA_NEAT_INTERNAL
#error "Internal header. Not part of the public API."
#endif

#include "model/Model.h"
#include "pipeline/internal/sima/MpkContract.h"

#include <chrono>
#include <filesystem>
#include <vector>

namespace simaai::neat::internal {

struct LayerProfileStage {
  std::size_t stage_index = 0;
  std::filesystem::path executable;
  std::filesystem::path stats_yaml;
};

std::vector<LayerProfileStage>
resolve_layer_profile_stages(const pipeline_internal::sima::MpkContract& contract);

std::vector<LayerStats>
collect_mla_layer_stats(const std::vector<LayerProfileStage>& stages,
                        const std::filesystem::path& mla_rt,
                        std::chrono::milliseconds timeout = std::chrono::seconds(120));

} // namespace simaai::neat::internal
