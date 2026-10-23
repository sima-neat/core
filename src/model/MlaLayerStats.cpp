#include "model/internal/MlaLayerStats.h"
#include "pipeline/internal/sima/static_contract/MpkDecoder.h"

#include <simaai/scheduler/client/mla_profile.h>

#include <glib.h>

#include <set>
#include <stdexcept>
#include <string>
#include <utility>
#include <variant>

namespace simaai::neat {
namespace {

[[noreturn]] void layer_stats_error(const std::string& message) {
  throw std::runtime_error(message);
}

struct MlaLayerStatsStage {
  std::size_t stage_index = 0;
  std::filesystem::path executable;
  std::filesystem::path stats_yaml;
};

// Stage indices count MLA and A65 plugins in execution order, as MLA stage stats does, so a
// selected stage keeps its full-model index. Only selected stages' files are opened, and they
// are all checked before mla-rt runs.
std::vector<MlaLayerStatsStage>
resolve_mla_layer_stats_stages(const pipeline_internal::sima::MpkContract& contract,
                               std::span<const std::size_t> selected_plugins) {
  std::set<std::size_t> selected(selected_plugins.begin(), selected_plugins.end());
  if (selected.size() != selected_plugins.size())
    layer_stats_error("duplicate selected MLA stage identity");
  for (const auto index : selected)
    if (index >= contract.plugins.size() ||
        g_ascii_strcasecmp(contract.plugins[index].processor.c_str(), "MLA") != 0)
      layer_stats_error("invalid selected MLA stage identity");
  std::vector<MlaLayerStatsStage> stages;
  std::size_t stage_index = 0;
  for (const auto plugin_index : pipeline_internal::sima::plugins_in_execution_order(contract)) {
    const auto& plugin = contract.plugins.at(plugin_index);
    const bool is_mla = g_ascii_strcasecmp(plugin.processor.c_str(), "MLA") == 0;
    if (!is_mla && g_ascii_strcasecmp(plugin.processor.c_str(), "A65") != 0)
      continue;
    const auto current_index = stage_index++;
    if (!is_mla || !selected.erase(plugin_index))
      continue;
    const auto executable = pipeline_internal::sima::static_contract::resolve_mla_executable(
        contract.mpk_json_path, plugin.executable);
    if (!std::filesystem::is_regular_file(executable))
      layer_stats_error("missing MLA ELF for " + plugin.name + ": " + executable.string());
    const auto stats_yaml = std::filesystem::path(contract.mpk_json_path).parent_path() /
                            (executable.stem().string() + "_stats.yaml");
    if (!std::filesystem::is_regular_file(stats_yaml))
      layer_stats_error("missing matching stats YAML: " + stats_yaml.string());
    stages.push_back({current_index, executable, stats_yaml});
  }
  if (!selected.empty())
    layer_stats_error("selected MLA stage is absent from execution order");
  if (stages.empty())
    layer_stats_error("selected inference route has no MLA stages");
  return stages;
}

} // namespace

std::vector<MlaLayer>
internal::collect_mla_layer_stats(const pipeline_internal::sima::MpkContract& contract,
                                  std::span<const std::size_t> selected_plugins) {
  std::vector<MlaLayer> result;
  for (const auto& stage : resolve_mla_layer_stats_stages(contract, selected_plugins)) {
    auto layers =
        simaai::scheduler::client::measure_mla_layer_stats(stage.executable, stage.stats_yaml);
    if (const auto* error = std::get_if<simaai::scheduler::client::Error>(&layers))
      layer_stats_error("MLA stage " + std::to_string(stage.stage_index) + " (" +
                        stage.executable.filename().string() + "): " + error->message);
    for (auto& layer : std::get<std::vector<simaai::scheduler::client::MlaLayerLatency>>(layers))
      result.push_back({stage.stage_index, std::move(layer.name), layer.latency_us});
  }
  return result;
}

} // namespace simaai::neat
