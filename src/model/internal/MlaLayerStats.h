#pragma once
#ifndef SIMA_NEAT_INTERNAL
#error "Internal header. Not part of the public API."
#endif

#include "model/Model.h"
#include "pipeline/internal/sima/MpkContract.h"

#include <chrono>
#include <filesystem>
#include <cstdint>
#include <map>
#include <span>
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
  std::map<std::uint64_t, LayerProfileRow>
      expected_layers; ///< Validated once, before benchmarking.
};

std::vector<LayerProfileStage>
resolve_layer_profile_stages(const pipeline_internal::sima::MpkContract& contract);
// Select exact MPK plugin identities before opening any executable or stats file.
std::vector<LayerProfileStage>
resolve_layer_profile_stages(const pipeline_internal::sima::MpkContract& contract,
                             std::span<const std::size_t> selected_plugins);

// Check Scheduler's offline diagnostic capability before benchmark inference.
// Profiling requires an idle DevKit; this does not arbitrate unrelated workloads.
void require_owned_mla_layer_profiling();

// Scheduler owns the cold provider session. Core keeps stage and report policy.
std::vector<LayerStats>
collect_mla_layer_stats(const std::vector<LayerProfileStage>& stages,
                        std::chrono::milliseconds timeout = std::chrono::milliseconds(120000));

// Pure report validation. The caller supplies an already collected report;
// parsing does not establish execution or DMA-retirement evidence.
std::vector<LayerStats> read_mla_layer_statistics(const LayerProfileStage& stage,
                                                  const std::filesystem::path& output_yaml);

std::vector<LayerStats> parse_mla_layer_statistics(const LayerProfileStage& stage,
                                                   const std::string& output_yaml);

} // namespace simaai::neat::internal
