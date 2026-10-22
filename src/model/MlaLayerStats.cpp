#include "model/internal/MlaLayerStats.h"
#include "pipeline/internal/sima/static_contract/MpkDecoder.h"

#include <glib.h>
#include <yaml-cpp/yaml.h>

#include <sys/wait.h>

#include <charconv>
#include <cmath>
#include <fstream>
#include <set>
#include <stdexcept>

namespace simaai::neat {
namespace {

[[noreturn]] void profile_error(const std::string& message) {
  throw std::runtime_error(message);
}

std::string read_stats_file(const std::filesystem::path& path) {
  constexpr std::size_t maximum = 8U * 1024U * 1024U;
  std::ifstream input(path, std::ios::binary);
  if (!input)
    profile_error("cannot read " + path.string());
  std::string text(maximum + 1, '\0');
  input.read(text.data(), static_cast<std::streamsize>(text.size()));
  text.resize(static_cast<std::size_t>(input.gcount()));
  if (input.bad())
    profile_error("failed reading " + path.string());
  if (text.size() > maximum)
    profile_error("stats YAML exceeds 8 MiB: " + path.string());
  return text;
}

std::uint64_t stats_integer(const std::string& text) {
  std::uint64_t value{};
  const auto result = std::from_chars(text.data(), text.data() + text.size(), value);
  if (result.ec != std::errc{} || result.ptr != text.data() + text.size())
    profile_error("invalid layer ID or cycle counter: " + text);
  return value;
}

using ProfileRow = internal::LayerProfileRow;
using ProfileRows = std::map<std::uint64_t, ProfileRow>;

bool profileable(std::uint64_t id, const ProfileRow& row) {
  return id != 0 && row.name.find("placeholder") == std::string::npos;
}

ProfileRows parse_stats_rows(const std::filesystem::path& path, bool measured) {
  try {
    const auto documents = YAML::LoadAll(read_stats_file(path));
    if (documents.size() > 1)
      profile_error("stats YAML must contain exactly one document");
    if (documents.empty() || !documents.front().IsMap())
      profile_error("stats YAML must be a mapping");
    ProfileRows rows;
    for (const auto& layer : documents.front()) {
      const auto id = stats_integer(layer.first.as<std::string>());
      if (!layer.second.IsMap())
        profile_error("layer must be a mapping");
      std::set<std::string> keys;
      for (const auto& field : layer.second) {
        if (!keys.insert(field.first.as<std::string>()).second)
          profile_error("duplicate layer field: " + field.first.as<std::string>());
      }
      // Allow unrelated compiler metadata.
      const auto required = [&](const char* key) {
        const auto value = layer.second[key];
        if (!value || !value.IsScalar())
          profile_error(std::string("missing or non-scalar ") + key + " on layer " +
                        std::to_string(id));
        return value.Scalar();
      };
      ProfileRow row;
      row.name = required("name");
      if (row.name.empty())
        profile_error("empty layer name");
      if (measured || profileable(id, row)) {
        row.start = stats_integer(required("start_cycle"));
        row.end = stats_integer(required("end_cycle"));
      }
      if (measured) {
        const auto time = required("run_time");
        if (!time.ends_with("us"))
          profile_error("run_time must have microsecond units: " + time);
        const auto result =
            std::from_chars(time.data(), time.data() + time.size() - 2, row.run_time);
        if (result.ec != std::errc{} || result.ptr != time.data() + time.size() - 2 ||
            !std::isfinite(row.run_time) || row.run_time < 0)
          profile_error("invalid run_time: " + time);
      }
      if (!rows.emplace(id, std::move(row)).second)
        profile_error("duplicate layer ID");
    }
    if (rows.empty())
      profile_error("stats YAML contains no layers");
    return rows;
  } catch (const YAML::Exception& error) {
    profile_error("invalid YAML in " + path.string() + ": " + error.what());
  }
}

ProfileRows load_profile_input(const std::filesystem::path& path) {
  auto rows = parse_stats_rows(path, false);
  std::uint64_t expected = 0;
  std::size_t layers = 0;
  for (const auto& [id, row] : rows) {
    if (id != expected++)
      profile_error("input layer IDs must be contiguous from zero");
    if (profileable(id, row)) {
      ++layers;
      if (row.start > INT32_MAX || row.end > INT32_MAX)
        profile_error("cycle range exceeds mla-rt's supported range");
    }
  }
  if (layers == 0)
    profile_error("input stats YAML has no profileable layers");
  return rows;
}

struct ProfileDirectory {
  std::filesystem::path path;
  ProfileDirectory() {
    GError* error = nullptr;
    gchar* directory = g_dir_make_tmp("neat-layer-stats-XXXXXX", &error);
    if (!directory) {
      const std::string detail = error ? error->message : "unknown error";
      g_clear_error(&error);
      profile_error("cannot create temporary directory: " + detail);
    }
    path = std::filesystem::absolute(directory);
    g_free(directory);
  }
  ~ProfileDirectory() {
    std::error_code ignored;
    std::filesystem::remove_all(path, ignored);
  }
};

// Runs mla-rt to completion; Core consumes its results and never stops it.
std::string run_profile_process(const std::filesystem::path& mla_rt,
                                const std::filesystem::path& input_yaml,
                                const std::filesystem::path& executable) {
  std::vector<std::string> args{
      mla_rt.string(),     "--connect",         "mip,ma",           "-vs", "-N",
      "--per-layer-stats", input_yaml.string(), executable.string()};
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
    profile_error("cannot start mla-rt: " + detail);
  }
  std::string diagnostics = std::string(output ? output : "") + (errors ? errors : "");
  g_free(output);
  g_free(errors);
  constexpr std::size_t max_diagnostics = 64U * 1024U;
  if (diagnostics.size() > max_diagnostics)
    diagnostics.erase(0, diagnostics.size() - max_diagnostics);
  if (!WIFEXITED(status) || WEXITSTATUS(status) != 0) {
    const std::string result = WIFEXITED(status) ? "exit " + std::to_string(WEXITSTATUS(status))
                                                 : "signal " + std::to_string(WTERMSIG(status));
    profile_error("mla-rt failed (" + result + ")\n" + diagnostics);
  }
  return diagnostics;
}

} // namespace

std::vector<internal::LayerProfileStage>
internal::resolve_layer_profile_stages(const pipeline_internal::sima::MpkContract& contract) {
  std::vector<LayerProfileStage> stages;
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
    if (executable.extension() != ".elf" || !std::filesystem::is_regular_file(executable))
      profile_error("missing MLA ELF for " + plugin.name + ": " + executable.string());
    const auto stats_yaml = std::filesystem::path(contract.mpk_json_path).parent_path() /
                            (executable.stem().string() + "_stats.yaml");
    if (!std::filesystem::is_regular_file(stats_yaml))
      profile_error("missing matching stats YAML: " + stats_yaml.string());
    stages.push_back({current_index, executable, stats_yaml, load_profile_input(stats_yaml)});
  }
  if (stages.empty())
    profile_error("model has no MLA stages to profile");
  return stages;
}

std::vector<MlaLayerStats>
internal::collect_mla_layer_stats(const std::vector<LayerProfileStage>& stages,
                                  const std::filesystem::path& mla_rt) {
  if (stages.empty())
    profile_error("model has no MLA stages to profile");
  std::vector<MlaLayerStats> result;
  for (const auto& stage : stages) {
    const auto& expected = stage.expected_layers;
    std::string diagnostics;
    try {
      ProfileDirectory temporary;
      const auto input_yaml = temporary.path / stage.stats_yaml.filename();
      std::filesystem::copy_file(stage.stats_yaml, input_yaml);
      diagnostics = run_profile_process(mla_rt, input_yaml, stage.executable);
      const auto output_yaml = temporary.path / (input_yaml.stem().string() + "_output.yaml");
      const auto actual = parse_stats_rows(output_yaml, true);
      std::size_t profiled = 0;
      bool has_nonzero_time = false;
      bool expects_work = false;
      for (const auto& [id, row] : expected) {
        // Match MLA-RT's placeholder filtering; retain real no-ops.
        if (!profileable(id, row))
          continue;
        ++profiled;
        const auto found = actual.find(id);
        if (found == actual.end() || found->second.name != row.name ||
            found->second.start != row.start || found->second.end != row.end)
          profile_error("mla-rt output does not match input layer " + std::to_string(id));
        const auto time = found->second.run_time;
        if (row.end < row.start && time != 0)
          profile_error("mla-rt returned nonzero time for a zero-cycle layer; update mla-rt");
        has_nonzero_time |= time > 0;
        expects_work |= row.end >= row.start;
        result.push_back({row.name, time, stage.stage_index});
      }
      if (actual.size() != profiled)
        profile_error("mla-rt output has unexpected layer records");
      if (expects_work && !has_nonzero_time)
        profile_error("mla-rt returned no hardware timings");
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
