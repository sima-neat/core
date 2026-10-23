#include "model/internal/MlaLayerStats.h"
#include "pipeline/internal/sima/static_contract/MpkDecoder.h"

#include <simaai/scheduler/client/capabilities.h>
#include <simaai/scheduler/client/mla_layer_profile.h>
#include <yaml-cpp/yaml.h>

#include <charconv>
#include <cmath>
#include <fstream>
#include <iterator>
#include <set>
#include <stdexcept>

namespace simaai::neat {
namespace {

[[noreturn]] void profile_error(const std::string& message) {
  throw std::runtime_error("Model::benchmark layer stats: " + message);
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

ProfileRows parse_stats_rows(const std::string& text, bool measured) {
  constexpr std::size_t maximum = 8U * 1024U * 1024U;
  if (text.size() > maximum)
    profile_error("stats YAML exceeds 8 MiB");
  try {
    const auto documents = YAML::LoadAll(text);
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
    profile_error("invalid YAML: " + std::string(error.what()));
  }
}

ProfileRows load_profile_input(const std::filesystem::path& path) {
  auto rows = parse_stats_rows(read_stats_file(path), false);
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

} // namespace

std::vector<internal::LayerProfileStage>
internal::resolve_layer_profile_stages(const pipeline_internal::sima::MpkContract& contract) {
  std::vector<std::size_t> selected;
  for (std::size_t i = 0; i < contract.plugins.size(); ++i)
    if (g_ascii_strcasecmp(contract.plugins[i].processor.c_str(), "MLA") == 0)
      selected.push_back(i);
  return resolve_layer_profile_stages(contract, selected);
}

std::vector<internal::LayerProfileStage>
internal::resolve_layer_profile_stages(const pipeline_internal::sima::MpkContract& contract,
                                       std::span<const std::size_t> selected_plugins) {
  std::set<std::size_t> selected(selected_plugins.begin(), selected_plugins.end());
  if (selected.size() != selected_plugins.size())
    profile_error("duplicate selected MLA stage identity");
  for (const auto index : selected)
    if (index >= contract.plugins.size() ||
        g_ascii_strcasecmp(contract.plugins[index].processor.c_str(), "MLA") != 0)
      profile_error("invalid selected MLA stage identity");
  std::vector<LayerProfileStage> stages;
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
    if (executable.extension() != ".elf" || !std::filesystem::is_regular_file(executable))
      profile_error("missing MLA ELF for " + plugin.name + ": " + executable.string());
    const auto stats_yaml = std::filesystem::path(contract.mpk_json_path).parent_path() /
                            (executable.stem().string() + "_stats.yaml");
    if (!std::filesystem::is_regular_file(stats_yaml))
      profile_error("missing matching stats YAML: " + stats_yaml.string());
    stages.push_back({current_index, executable, stats_yaml, load_profile_input(stats_yaml)});
  }
  if (!selected.empty())
    profile_error("selected MLA stage is absent from execution order");
  if (stages.empty())
    profile_error("selected inference route has no MLA stages to profile");
  return stages;
}

namespace {
namespace scheduler = simaai::scheduler::client;

bool safe_profile_failure(const scheduler::OfflineMlaProfileError& error) {
  using Code = scheduler::OfflineMlaProfileErrorCode;
  switch (error.code()) {
  case Code::Unavailable:
  case Code::InvalidInput:
  case Code::ProcessFailure:
  case Code::DataInvalid:
    break;
  case Code::Busy:
  case Code::RecoveryRequired:
  default:
    return false;
  }
  using Completion = scheduler::OfflineMlaProfileCompletion;
  switch (error.completion()) {
  case Completion::NotLaunched:
  case Completion::Completed:
    return true;
  case Completion::Unknown:
  default:
    return false;
  }
}

} // namespace

void internal::require_owned_mla_layer_profiling() {
  const auto capability = scheduler::mla_layer_statistics_capability();
  using Availability = scheduler::OfflineMlaProfileAvailability;
  using Code = scheduler::OfflineMlaProfileErrorCode;
  using Completion = scheduler::OfflineMlaProfileCompletion;
  switch (capability.availability) {
  case Availability::Available:
    return;
  case Availability::Unavailable:
    throw scheduler::OfflineMlaProfileError(Completion::NotLaunched, Code::Unavailable,
                                            capability.reason);
  case Availability::Busy:
    throw scheduler::OfflineMlaProfileError(Completion::NotLaunched, Code::Busy, capability.reason);
  case Availability::RecoveryRequired:
    throw scheduler::OfflineMlaProfileError(Completion::Unknown, Code::RecoveryRequired,
                                            capability.reason);
  default:
    throw scheduler::OfflineMlaProfileError(Completion::Unknown, Code::RecoveryRequired,
                                            "unrecognized MLA profiling capability; keep MLA idle");
  }
}

std::string internal::prepare_mla_layer_stats(const pipeline_internal::sima::MpkContract& contract,
                                              std::span<const std::size_t> selected_plugins,
                                              std::vector<LayerProfileStage>& stages) {
  stages.clear();
  try {
    require_owned_mla_layer_profiling();
  } catch (const scheduler::OfflineMlaProfileError& error) {
    if (!safe_profile_failure(error))
      throw;
    return error.what();
  }
  try {
    stages = resolve_layer_profile_stages(contract, selected_plugins);
    return {};
  } catch (const std::runtime_error& error) {
    // Only local metadata/file validation has run: no diagnostic was launched.
    return error.what();
  }
}

void internal::collect_mla_layer_stats_for_report(BenchmarkReport& report,
                                                  const std::vector<LayerProfileStage>& stages) {
  try {
    auto rows = collect_mla_layer_stats(stages);
    report.layer_stats = std::move(rows);
    report.layer_stats_status = "ok";
  } catch (const scheduler::OfflineMlaProfileError& error) {
    if (!safe_profile_failure(error))
      throw;
    report.layer_stats.clear();
    report.layer_stats_status = error.what();
  }
}

std::vector<LayerStats>
internal::read_mla_layer_statistics(const LayerProfileStage& stage,
                                    const std::filesystem::path& output_yaml) {
  return parse_mla_layer_statistics(stage, read_stats_file(output_yaml));
}

std::vector<LayerStats> internal::parse_mla_layer_statistics(const LayerProfileStage& stage,
                                                             const std::string& output_yaml) {
  const auto actual = parse_stats_rows(output_yaml, true);
  std::vector<LayerStats> result;
  std::size_t profiled = 0;
  bool has_nonzero_time = false;
  bool expects_work = false;
  for (const auto& [id, row] : stage.expected_layers) {
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
  return result;
}

std::vector<LayerStats>
internal::collect_mla_layer_stats(const std::vector<LayerProfileStage>& stages,
                                  std::chrono::milliseconds timeout) {
  if (stages.empty())
    profile_error("model has no MLA stages to profile");
  if (timeout <= std::chrono::milliseconds::zero())
    profile_error("timeout must be positive");
  std::vector<LayerStats> result;
  for (const auto& stage : stages) {
    std::string diagnostics;
    try {
      const auto measured = simaai::scheduler::client::profile_mla_layers_offline(
          {stage.executable, stage.stats_yaml, timeout});
      diagnostics = measured.diagnostics;
      auto records = parse_mla_layer_statistics(stage, measured.output_yaml);
      result.insert(result.end(), std::make_move_iterator(records.begin()),
                    std::make_move_iterator(records.end()));
    } catch (const scheduler::OfflineMlaProfileError& error) {
      throw scheduler::OfflineMlaProfileError(error.completion(), error.code(),
                                              std::string(error.what()) + "\nMLA stage " +
                                                  std::to_string(stage.stage_index) + " (" +
                                                  stage.executable.filename().string() + ")");
    } catch (const std::exception& error) {
      // A provider return is not a new hardware completion proof. Keep parser failures fatal.
      throw std::runtime_error(
          std::string(error.what()) + "\nMLA stage " + std::to_string(stage.stage_index) + " (" +
          stage.executable.filename().string() + ")" +
          (diagnostics.empty() ? "" : "\nMLA profiling diagnostics:\n" + diagnostics) +
          "\nHardware completion is not proven; keep MLA idle until recovery is verified.");
    }
  }
  return result;
}

} // namespace simaai::neat
