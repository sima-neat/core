#include "model/internal/MlaStageStats.h"
#include "pipeline/internal/sima/static_contract/MpkDecoder.h"

#include <simaai/scheduler/client/mla_profile.h>

#include <glib.h>

#include <stdexcept>
#include <string>
#include <variant>

namespace simaai::neat {
namespace {

[[noreturn]] void stage_stats_error(const std::string& message) {
  throw std::runtime_error(message);
}

struct MlaStageStatsStage {
  std::size_t stage_index = 0;
  std::filesystem::path executable;
};

// Stage indices count MLA and A65 plugins in execution order, as MLA layer stats does.
// Every stage is validated before mla-rt runs.
std::vector<MlaStageStatsStage>
resolve_mla_stage_stats_stages(const pipeline_internal::sima::MpkContract& contract) {
  std::vector<MlaStageStatsStage> stages;
  std::size_t stage_index = 0;
  for (const auto plugin_index : pipeline_internal::sima::plugins_in_execution_order(contract)) {
    const auto& plugin = contract.plugins.at(plugin_index);
    const bool is_mla = g_ascii_strcasecmp(plugin.processor.c_str(), "MLA") == 0;
    if (!is_mla && g_ascii_strcasecmp(plugin.processor.c_str(), "A65") != 0)
      continue;
    const auto current_index = stage_index++;
    if (!is_mla)
      continue;
    const auto executable = pipeline_internal::sima::static_contract::resolve_mla_executable(
        contract.mpk_json_path, plugin.executable);
    if (!std::filesystem::is_regular_file(executable))
      stage_stats_error("missing MLA ELF for " + plugin.name + ": " + executable.string());
    stages.push_back({current_index, executable});
  }
  if (stages.empty())
    stage_stats_error("model has no MLA stages");
  return stages;
}

} // namespace

std::vector<MlaStage>
internal::collect_mla_stage_stats(const pipeline_internal::sima::MpkContract& contract) {
  std::vector<MlaStage> result;
  for (const auto& stage : resolve_mla_stage_stats_stages(contract)) {
    const auto latency = simaai::scheduler::client::measure_mla_stage_stats(stage.executable);
    if (const auto* error = std::get_if<simaai::scheduler::client::Error>(&latency))
      stage_stats_error("MLA stage " + std::to_string(stage.stage_index) + " (" +
                        stage.executable.filename().string() + "): " + error->message);
    result.push_back({stage.stage_index, std::get<double>(latency)});
  }
  return result;
}

} // namespace simaai::neat
