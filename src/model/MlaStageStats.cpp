#include "model/internal/MlaStageStats.h"
#include "pipeline/internal/sima/static_contract/MpkDecoder.h"

#include <glib.h>
#include <yaml-cpp/yaml.h>

#include <sys/wait.h>

#include <cmath>
#include <fstream>
#include <iterator>
#include <stdexcept>
#include <string>
#include <utility>

namespace simaai::neat {
namespace {

constexpr int kRuns = 200;

[[noreturn]] void stage_error(const std::string& message) {
  throw std::runtime_error(message);
}

struct StatsDirectory {
  std::filesystem::path path;
  StatsDirectory() {
    GError* error = nullptr;
    gchar* directory = g_dir_make_tmp("neat-mla-stage-stats-XXXXXX", &error);
    if (!directory) {
      const std::string detail = error ? error->message : "unknown error";
      g_clear_error(&error);
      stage_error("cannot create temporary directory: " + detail);
    }
    path = directory;
    g_free(directory);
  }
  ~StatsDirectory() {
    std::error_code ignored;
    std::filesystem::remove_all(path, ignored);
  }
};

// Stage indices count MLA and A65 plugins in execution order, as layer stats does.
std::vector<std::pair<std::size_t, std::filesystem::path>>
mla_stage_executables(const pipeline_internal::sima::MpkContract& contract) {
  std::vector<std::pair<std::size_t, std::filesystem::path>> stages;
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
      stage_error("missing MLA ELF for " + plugin.name + ": " + executable.string());
    stages.emplace_back(current_index, executable);
  }
  if (stages.empty())
    stage_error("model has no MLA stages");
  return stages;
}

// Runs mla-rt to completion; Core consumes its results and never stops it.
void run_mla_rt(const std::filesystem::path& mla_rt, const std::filesystem::path& stats_file,
                const std::filesystem::path& executable) {
  std::vector<std::string> args{mla_rt.string(),
                                "--connect",
                                "kernel",
                                "-N",
                                "-m",
                                std::to_string(kRuns),
                                "-vss",
                                "--stats=" + stats_file.string(),
                                executable.string()};
  std::vector<gchar*> argv;
  for (auto& arg : args)
    argv.push_back(arg.data());
  argv.push_back(nullptr);
  gchar* output = nullptr;
  gchar* errors = nullptr;
  gint status = 0;
  GError* error = nullptr;
  if (!g_spawn_sync(nullptr, argv.data(), nullptr, G_SPAWN_DEFAULT, nullptr, nullptr, &output,
                    &errors, &status, &error)) {
    const std::string detail = error ? error->message : "unknown error";
    g_clear_error(&error);
    stage_error("cannot start mla-rt: " + detail);
  }
  std::string diagnostics = std::string(output ? output : "") + (errors ? errors : "");
  g_free(output);
  g_free(errors);
  constexpr std::size_t max_diagnostics = 4U * 1024U;
  if (diagnostics.size() > max_diagnostics)
    diagnostics.erase(0, diagnostics.size() - max_diagnostics);
  if (!WIFEXITED(status) || WEXITSTATUS(status) != 0) {
    const std::string result = WIFEXITED(status) ? "exit " + std::to_string(WEXITSTATUS(status))
                                                 : "signal " + std::to_string(WTERMSIG(status));
    stage_error("mla-rt failed (" + result + ")\n" + diagnostics);
  }
}

// mla-rt writes one root entry per ELF; its units appear only in the header comment.
double mean_tile_latency_us(const std::filesystem::path& stats_file) {
  std::ifstream file(stats_file);
  if (!file)
    stage_error("mla-rt wrote no stats file");
  const std::string text{std::istreambuf_iterator<char>(file), std::istreambuf_iterator<char>()};
  if (text.find("dt:usec") == std::string::npos)
    stage_error("stats file is not in microseconds");
  try {
    const auto root = YAML::Load(text);
    if (!root.IsMap() || root.size() != 1)
      stage_error("stats file must describe exactly one ELF");
    const auto tile = root.begin()->second["tile.lat"];
    if (!tile || !tile["avg"] || !tile["runcount"])
      stage_error("stats file has no tile.lat average and run count");
    const auto runs = tile["runcount"].as<int>();
    if (runs != kRuns)
      stage_error("mla-rt measured " + std::to_string(runs) + " runs, expected " +
                  std::to_string(kRuns));
    const auto latency = tile["avg"].as<double>();
    if (!std::isfinite(latency) || latency <= 0)
      stage_error("tile.lat average must be positive");
    return latency;
  } catch (const YAML::Exception& error) {
    stage_error(std::string("invalid stats file: ") + error.what());
  }
}

} // namespace

std::vector<MlaStageStats>
internal::collect_mla_stage_stats(const pipeline_internal::sima::MpkContract& contract,
                                  const std::filesystem::path& mla_rt) {
  StatsDirectory temporary;
  std::vector<MlaStageStats> result;
  for (const auto& [stage_index, executable] : mla_stage_executables(contract)) {
    const auto stats_file = temporary.path / ("stage" + std::to_string(stage_index) + ".yaml");
    try {
      run_mla_rt(mla_rt, stats_file, executable);
      result.push_back({stage_index, mean_tile_latency_us(stats_file)});
    } catch (const std::exception& error) {
      throw std::runtime_error(std::string(error.what()) + "\nMLA stage " +
                               std::to_string(stage_index) + " (" + executable.filename().string() +
                               ")");
    }
  }
  return result;
}

} // namespace simaai::neat
