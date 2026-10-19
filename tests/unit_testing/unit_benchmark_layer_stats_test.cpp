#ifndef SIMA_NEAT_INTERNAL
#define SIMA_NEAT_INTERNAL 1
#endif
#include "model/internal/MlaLayerStats.h"
#include "test_main.h"

#include <glib.h>

#include <chrono>
#include <fstream>
#include <functional>
#include <sstream>

namespace {
using namespace simaai::neat;
using namespace simaai::neat::internal;

struct Fixture {
  std::filesystem::path root;
  Fixture() {
    gchar* directory = g_dir_make_tmp("neat-profile-test-XXXXXX", nullptr);
    require(directory != nullptr, "create fixture");
    root = std::filesystem::path(directory) / "model with spaces";
    g_free(directory);
    std::filesystem::create_directories(root / "etc");
    std::filesystem::create_directories(root / "share");
  }
  ~Fixture() {
    std::error_code ignored;
    std::filesystem::remove_all(root.parent_path(), ignored);
  }
  std::filesystem::path write(const std::string& name, const std::string& value) {
    const auto path = root / name;
    std::ofstream(path) << value;
    return path;
  }
};
std::string read(const std::filesystem::path& path) {
  std::ifstream file(path);
  return {std::istreambuf_iterator<char>(file), std::istreambuf_iterator<char>()};
}
void fails(const std::function<void()>& operation, const std::string& expected) {
  try {
    operation();
  } catch (const std::exception& error) {
    require(std::string(error.what()).find(expected) != std::string::npos, error.what());
    return;
  }
  require(false, "expected failure: " + expected);
}
const std::string input =
    "0:\n  name: placeholder\n1:\n  name: MLA_0/noop\n  start_cycle: 5\n  end_cycle: 4\n"
    "2:\n  name: 'MLA_0/conv: quoted'\n  start_cycle: 5\n  end_cycle: 20\n";
const std::string output =
    "1:\n  name: MLA_0/noop\n  start_cycle: 5\n  end_cycle: 4\n  run_time: 0.00us\n"
    "2:\n  name: 'MLA_0/conv: quoted'\n  start_cycle: 5\n  end_cycle: 20\n  run_time: 161.95us\n";
LayerProfileStage stage(Fixture& fixture, const std::string& source = input) {
  fixture.write("share/demo_stage1_mla.elf", "elf fixture");
  fixture.write("etc/demo_stage1_mla_stats.yaml", source);
  pipeline_internal::sima::MpkContract contract;
  contract.mpk_json_path = (fixture.root / "etc/demo_mpk.json").string();
  pipeline_internal::sima::MpkPluginIoContract plugin;
  plugin.processor = "MLA";
  plugin.executable = "demo_stage1_mla.elf";
  contract.plugins.push_back(plugin);
  auto selected = resolve_layer_profile_stages(contract).front();
  selected.stage_index = 2;
  return selected;
}
void pairing_and_stage_order() {
  Fixture fixture;
  stage(fixture);
  const auto second = fixture.write("share/demo_stage2_mla.elf", "elf fixture");
  fixture.write("etc/demo_stage2_mla_stats.yaml", input);
  pipeline_internal::sima::MpkContract contract;
  contract.mpk_json_path = (fixture.root / "etc/demo_mpk.json").string();
  auto plugin = [](std::string processor, int sequence, std::string executable) {
    pipeline_internal::sima::MpkPluginIoContract result;
    result.processor = std::move(processor);
    result.sequence = sequence;
    result.executable = std::move(executable);
    return result;
  };
  contract.plugins = {plugin("MLA", 4, second.string()), plugin("EV74", 1, "kernel"),
                      plugin("MLA", 2, "demo_stage1_mla.elf"), plugin("A65", 3, "host.so")};
  const auto stages = resolve_layer_profile_stages(contract);
  require(stages.size() == 2 && stages[0].stage_index == 0 && stages[1].stage_index == 2,
          "MLA/A65 execution order must determine stage indices");
  require(stages[0].stats_yaml.filename() == "demo_stage1_mla_stats.yaml", "exact ELF pairing");
  for (const auto& [mla, a65] : std::vector<std::pair<std::string, std::string>>{
           {"mla", "A65"}, {"MLA", "a65"}, {"mla", "a65"}, {"mLa", "a65"}}) {
    contract.plugins[0].processor = mla;
    contract.plugins[2].processor = mla;
    contract.plugins[3].processor = a65;
    const auto normalized = resolve_layer_profile_stages(contract);
    require(normalized.size() == stages.size(), "processor case must not change MLA selection");
    for (std::size_t i = 0; i < stages.size(); ++i) {
      require(normalized[i].stage_index == stages[i].stage_index &&
                  normalized[i].executable == stages[i].executable &&
                  normalized[i].stats_yaml == stages[i].stats_yaml,
              "processor case must not change stage indices or ELF/YAML pairing");
    }
  }
  std::filesystem::remove(stages[0].stats_yaml);
  fails([&] { resolve_layer_profile_stages(contract); }, "missing matching stats YAML");
}
void completed_report_preserves_original() {
  Fixture fixture;
  const auto selected = stage(fixture);
  const auto report = fixture.write("measured.yaml", output);
  auto second = selected;
  second.stage_index = 0;
  auto records = read_mla_layer_statistics(selected, report);
  const auto next = read_mla_layer_statistics(second, report);
  records.insert(records.end(), next.begin(), next.end());
  require(records.size() == 4 && records[0].name == "MLA_0/noop" &&
              records[0].total_execution_time_us == 0 &&
              records[1].total_execution_time_us == 161.95 && records[0].stage_index == 2 &&
              records[2].stage_index == 0,
          "retain no-ops, compiler names, supplied times, and stage identity");
  require(read(selected.stats_yaml) == input && read(report) == output,
          "report validation must not modify either source");
}
void unrelated_metadata_is_ignored() {
  Fixture fixture;
  const auto source = input + "  compiler_metadata:\n    tags: [conv, quantized]\n";
  const auto selected = stage(fixture, source);
  const auto report = fixture.write("measured.yaml", output + "  debug_metadata: {version: 1}\n");
  const auto records = read_mla_layer_statistics(selected, report);
  require(records.size() == 2 && records.back().total_execution_time_us == 161.95,
          "unrelated metadata must not change layer timings");
  require(read(selected.stats_yaml) == source, "compiler metadata remains unchanged");
}

void invalid_inputs_are_rejected_before_profiling() {
  Fixture fixture;
  for (const auto& [text, error] : std::vector<std::pair<std::string, std::string>>{
           {input + input, "duplicate layer ID"},
           {input + "  name: duplicate\n", "duplicate layer field"},
           {input + "---\n{}\n", "exactly one document"},
           {"0: {name: placeholder}\n2: {name: conv, start_cycle: 0, end_cycle: 20}\n",
            "contiguous from zero"}})
    fails([&] { stage(fixture, text); }, error);
}

void text_report_preserves_independent_validation() {
  Fixture fixture;
  const auto selected = stage(fixture);
  const auto report = fixture.write("measured.yaml", output);
  const auto file_records = read_mla_layer_statistics(selected, report);
  const auto text_records = parse_mla_layer_statistics(selected, output);
  require(text_records.size() == file_records.size(), "text and file reports have the same rows");
  for (std::size_t i = 0; i < text_records.size(); ++i) {
    require(text_records[i].name == file_records[i].name &&
                text_records[i].stage_index == file_records[i].stage_index &&
                text_records[i].total_execution_time_us == file_records[i].total_execution_time_us,
            "Scheduler-returned text must use the independent Core report validator");
  }
  fails([&] { parse_mla_layer_statistics(selected, std::string(8U * 1024U * 1024U + 1, ' ')); },
        "exceeds 8 MiB");
  require(read(selected.stats_yaml) == input, "text validation does not alter compiler YAML");
}

void invalid_collection_is_rejected_before_scheduler() {
  fails([] { collect_mla_layer_stats({}); }, "no MLA stages");
  Fixture fixture;
  const auto selected = stage(fixture);
  fails([&] { collect_mla_layer_stats({selected}, std::chrono::milliseconds::zero()); },
        "timeout must be positive");
  fails([&] { collect_mla_layer_stats({selected}, std::chrono::milliseconds(-1)); },
        "timeout must be positive");
}

void invalid_outputs_are_rejected() {
  Fixture fixture;
  const auto selected = stage(fixture);
  for (const auto& [text, error] : std::vector<std::pair<std::string, std::string>>{
           {"", "must be a mapping"},
           {"[invalid", "invalid YAML"},
           {"2:\n  name: wrong\n  start_cycle: 5\n  end_cycle: 20\n  run_time: 1us\n",
            "does not match"},
           {output + output, "duplicate layer ID"}}) {
    const auto report = fixture.write("measured.yaml", text);
    fails([&] { read_mla_layer_statistics(selected, report); }, error);
    fails([&] { parse_mla_layer_statistics(selected, text); }, error);
  }
  for (const auto& [time, error] :
       std::vector<std::pair<std::string, std::string>>{{"nanus", "invalid run_time"},
                                                        {"-1us", "invalid run_time"},
                                                        {"1ms", "microsecond units"},
                                                        {"0us", "no hardware timings"}}) {
    auto text = output;
    text.replace(text.find("161.95us"), 8, time);
    const auto report = fixture.write("measured.yaml", text);
    fails([&] { read_mla_layer_statistics(selected, report); }, error);
    fails([&] { parse_mla_layer_statistics(selected, text); }, error);
  }
  auto text = output;
  text.replace(text.find("0.00us"), 6, "100us");
  const auto report = fixture.write("measured.yaml", text);
  fails([&] { read_mla_layer_statistics(selected, report); }, "zero-cycle layer");
  require(read(selected.stats_yaml) == input, "invalid output must not modify source YAML");
}
} // namespace

RUN_TEST("unit_benchmark_layer_stats_test", ([] {
           pairing_and_stage_order();
           completed_report_preserves_original();
           unrelated_metadata_is_ignored();
           invalid_inputs_are_rejected_before_profiling();
           text_report_preserves_independent_validation();
           invalid_collection_is_rejected_before_scheduler();
           invalid_outputs_are_rejected();
         }));
