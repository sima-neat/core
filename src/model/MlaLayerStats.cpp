#include "model/internal/MlaLayerStats.h"
#include "model/internal/MlaRt.h"
#include "pipeline/internal/sima/static_contract/MpkDecoder.h"

#include <glib.h>
#include <yaml-cpp/yaml.h>

#include <charconv>
#include <cmath>
#include <cstdint>
#include <fstream>
#include <map>
#include <set>
#include <stdexcept>
#include <string>
#include <utility>

namespace simaai::neat {
namespace {

[[noreturn]] void layer_stats_error(const std::string& message) {
  throw std::runtime_error(message);
}

struct MlaLayerStatsRow {
  std::string name;
  std::uint64_t start = 0;
  std::uint64_t end = 0;
  double run_time = 0;
};

using StatsRows = std::map<std::uint64_t, MlaLayerStatsRow>;

struct MlaLayerStatsStage {
  std::size_t stage_index = 0;
  std::filesystem::path executable;
  std::filesystem::path stats_yaml;
  StatsRows expected_layers; ///< Validated before mla-rt runs.
};

std::string read_stats_file(const std::filesystem::path& path) {
  constexpr std::size_t maximum = 8U * 1024U * 1024U;
  std::ifstream input(path, std::ios::binary);
  if (!input)
    layer_stats_error("cannot read " + path.string());
  std::string text(maximum + 1, '\0');
  input.read(text.data(), static_cast<std::streamsize>(text.size()));
  text.resize(static_cast<std::size_t>(input.gcount()));
  if (input.bad())
    layer_stats_error("failed reading " + path.string());
  if (text.size() > maximum)
    layer_stats_error("stats YAML exceeds 8 MiB: " + path.string());
  return text;
}

std::uint64_t stats_integer(const std::string& text) {
  std::uint64_t value{};
  const auto result = std::from_chars(text.data(), text.data() + text.size(), value);
  if (result.ec != std::errc{} || result.ptr != text.data() + text.size())
    layer_stats_error("invalid layer ID or cycle counter: " + text);
  return value;
}

// mla-rt reports every layer except ID 0 and placeholders.
bool has_stats(std::uint64_t id, const MlaLayerStatsRow& row) {
  return id != 0 && row.name.find("placeholder") == std::string::npos;
}

StatsRows parse_stats_rows(const std::filesystem::path& path, bool measured) {
  try {
    const auto documents = YAML::LoadAll(read_stats_file(path));
    if (documents.size() > 1)
      layer_stats_error("stats YAML must contain exactly one document");
    if (documents.empty() || !documents.front().IsMap())
      layer_stats_error("stats YAML must be a mapping");
    StatsRows rows;
    for (const auto& layer : documents.front()) {
      const auto id = stats_integer(layer.first.as<std::string>());
      if (!layer.second.IsMap())
        layer_stats_error("layer must be a mapping");
      std::set<std::string> keys;
      for (const auto& field : layer.second) {
        if (!keys.insert(field.first.as<std::string>()).second)
          layer_stats_error("duplicate layer field: " + field.first.as<std::string>());
      }
      // Allow unrelated compiler metadata.
      const auto required = [&](const char* key) {
        const auto value = layer.second[key];
        if (!value || !value.IsScalar())
          layer_stats_error(std::string("missing or non-scalar ") + key + " on layer " +
                            std::to_string(id));
        return value.Scalar();
      };
      MlaLayerStatsRow row;
      row.name = required("name");
      if (row.name.empty())
        layer_stats_error("empty layer name");
      if (measured || has_stats(id, row)) {
        row.start = stats_integer(required("start_cycle"));
        row.end = stats_integer(required("end_cycle"));
      }
      if (measured) {
        const auto time = required("run_time");
        if (!time.ends_with("us"))
          layer_stats_error("run_time must have microsecond units: " + time);
        const auto result =
            std::from_chars(time.data(), time.data() + time.size() - 2, row.run_time);
        if (result.ec != std::errc{} || result.ptr != time.data() + time.size() - 2 ||
            !std::isfinite(row.run_time) || row.run_time < 0)
          layer_stats_error("invalid run_time: " + time);
      }
      if (!rows.emplace(id, std::move(row)).second)
        layer_stats_error("duplicate layer ID");
    }
    if (rows.empty())
      layer_stats_error("stats YAML contains no layers");
    return rows;
  } catch (const YAML::Exception& error) {
    layer_stats_error("invalid YAML in " + path.string() + ": " + error.what());
  }
}

StatsRows load_stats_input(const std::filesystem::path& path) {
  auto rows = parse_stats_rows(path, false);
  std::uint64_t expected = 0;
  std::size_t layers = 0;
  for (const auto& [id, row] : rows) {
    if (id != expected++)
      layer_stats_error("input layer IDs must be contiguous from zero");
    if (has_stats(id, row)) {
      ++layers;
      if (row.start > INT32_MAX || row.end > INT32_MAX)
        layer_stats_error("cycle range exceeds mla-rt's supported range");
    }
  }
  if (layers == 0)
    layer_stats_error("input stats YAML has no layers mla-rt can time");
  return rows;
}

// Stage indices count MLA and A65 plugins in execution order, as MLA stage stats does.
// Every stage is validated before mla-rt runs.
std::vector<MlaLayerStatsStage>
resolve_mla_layer_stats_stages(const pipeline_internal::sima::MpkContract& contract) {
  std::vector<MlaLayerStatsStage> stages;
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
      layer_stats_error("missing MLA ELF for " + plugin.name + ": " + executable.string());
    const auto stats_yaml = std::filesystem::path(contract.mpk_json_path).parent_path() /
                            (executable.stem().string() + "_stats.yaml");
    if (!std::filesystem::is_regular_file(stats_yaml))
      layer_stats_error("missing matching stats YAML: " + stats_yaml.string());
    stages.push_back({current_index, executable, stats_yaml, load_stats_input(stats_yaml)});
  }
  if (stages.empty())
    layer_stats_error("model has no MLA stages");
  return stages;
}

struct StatsDirectory {
  std::filesystem::path path;
  StatsDirectory() {
    GError* error = nullptr;
    gchar* directory = g_dir_make_tmp("neat-mla-layer-stats-XXXXXX", &error);
    if (!directory) {
      const std::string detail = error ? error->message : "unknown error";
      g_clear_error(&error);
      layer_stats_error("cannot create temporary directory: " + detail);
    }
    path = std::filesystem::absolute(directory);
    g_free(directory);
  }
  ~StatsDirectory() {
    std::error_code ignored;
    std::filesystem::remove_all(path, ignored);
  }
};

} // namespace

std::vector<MlaLayer>
internal::collect_mla_layer_stats(const pipeline_internal::sima::MpkContract& contract,
                                  const std::filesystem::path& mla_rt) {
  std::vector<MlaLayer> result;
  for (const auto& stage : resolve_mla_layer_stats_stages(contract)) {
    const auto& expected = stage.expected_layers;
    std::string diagnostics;
    try {
      StatsDirectory temporary;
      const auto input_yaml = temporary.path / stage.stats_yaml.filename();
      std::filesystem::copy_file(stage.stats_yaml, input_yaml);
      // mla-rt writes <input>_output.yaml next to the input YAML.
      diagnostics = run_mla_rt(mla_rt, {"--connect", "mip,ma", "-vs", "-N", "--per-layer-stats",
                                        input_yaml.string(), stage.executable.string()});
      const auto output_yaml = temporary.path / (input_yaml.stem().string() + "_output.yaml");
      const auto actual = parse_stats_rows(output_yaml, true);
      std::size_t timed = 0;
      bool has_nonzero_time = false;
      bool expects_work = false;
      for (const auto& [id, row] : expected) {
        // Match MLA-RT's placeholder filtering; retain real no-ops.
        if (!has_stats(id, row))
          continue;
        ++timed;
        const auto found = actual.find(id);
        if (found == actual.end() || found->second.name != row.name ||
            found->second.start != row.start || found->second.end != row.end)
          layer_stats_error("mla-rt output does not match input layer " + std::to_string(id));
        const auto time = found->second.run_time;
        if (row.end < row.start && time != 0)
          layer_stats_error("mla-rt returned nonzero time for a zero-cycle layer; update mla-rt");
        has_nonzero_time |= time > 0;
        expects_work |= row.end >= row.start;
        result.push_back({stage.stage_index, row.name, time});
      }
      if (actual.size() != timed)
        layer_stats_error("mla-rt output has unexpected layer records");
      if (expects_work && !has_nonzero_time)
        layer_stats_error("mla-rt returned no hardware timings");
    } catch (const std::exception& error) {
      throw std::runtime_error(
          std::string(error.what()) + "\nMLA stage " + std::to_string(stage.stage_index) + " (" +
          stage.executable.filename().string() + ")" +
          (diagnostics.empty() ? "" : "\nmla-rt diagnostics:\n" + diagnostics));
    }
  }
  return result;
}

} // namespace simaai::neat
