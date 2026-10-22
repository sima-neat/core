#pragma once
#ifndef SIMA_NEAT_INTERNAL
#error "Internal header. Not part of the public API."
#endif

#include "model/Model.h"
#include "pipeline/internal/sima/MpkContract.h"

#include <filesystem>
#include <cstdint>
#include <map>
#include <vector>

namespace simaai::neat::internal {

struct LayerProfileRow {
  std::string name;
  std::uint64_t start = 0;
  std::uint64_t end = 0;
  double run_time = 0;
};

struct LayerProfileStage {
  std::size_t stage_index = 0;
  std::filesystem::path executable;
  std::filesystem::path stats_yaml;
  std::map<std::uint64_t, LayerProfileRow> expected_layers; ///< Validated before mla-rt runs.
};

std::vector<LayerProfileStage>
resolve_layer_profile_stages(const pipeline_internal::sima::MpkContract& contract);

std::vector<MlaLayerStats> collect_mla_layer_stats(const std::vector<LayerProfileStage>& stages,
                                                   const std::filesystem::path& mla_rt);

} // namespace simaai::neat::internal
