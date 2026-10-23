#ifndef SIMA_NEAT_INTERNAL
#define SIMA_NEAT_INTERNAL 1
#endif
#include "model/internal/MlaLayerStats.h"
#include "model/internal/ModelPack.h"
#include "test_main.h"

#include <glib.h>

#include <fstream>
#include <functional>

namespace {
using namespace simaai::neat;
using namespace simaai::neat::internal;

struct Fixture {
  std::filesystem::path root;
  Fixture() {
    gchar* directory = g_dir_make_tmp("neat-mla-layer-stats-test-XXXXXX", nullptr);
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
  // Places a fake mla-rt running `body` alone in PATH, where Scheduler's client finds it.
  void runtime(const std::string& body) {
    std::filesystem::create_directories(root / "bin");
    const auto file = write("bin/mla-rt", "#!/bin/sh\nPATH=/usr/bin:/bin\nset -eu\n" + body);
    std::filesystem::permissions(file, std::filesystem::perms::owner_all);
    require(g_setenv("PATH", (root / "bin").c_str(), TRUE), "set PATH");
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
pipeline_internal::sima::MpkPluginIoContract plugin(std::string processor, int sequence,
                                                    std::string executable) {
  pipeline_internal::sima::MpkPluginIoContract result;
  result.name = processor + std::to_string(sequence);
  result.processor = std::move(processor);
  result.sequence = sequence;
  result.executable = std::move(executable);
  return result;
}
// Two MLA stages separated by an A65 stage, listed out of execution order.
pipeline_internal::sima::MpkContract contract(Fixture& fixture, const std::string& first = input,
                                              const std::string& second = input) {
  fixture.write("share/demo_stage1_mla.elf", "elf fixture");
  const auto second_elf = fixture.write("share/demo_stage2_mla.elf", "elf fixture");
  fixture.write("etc/demo_stage1_mla_stats.yaml", first);
  fixture.write("etc/demo_stage2_mla_stats.yaml", second);
  pipeline_internal::sima::MpkContract result;
  result.mpk_json_path = (fixture.root / "etc/demo_mpk.json").string();
  result.plugins = {plugin("mla", 4, second_elf.string()), plugin("EV74", 1, "kernel"),
                    plugin("MLA", 2, "demo_stage1_mla.elf"), plugin("a65", 3, "host.so")};
  return result;
}
// Every MLA plugin index, the selection of a route with no terminal policy.
std::vector<std::size_t> all_mla(const pipeline_internal::sima::MpkContract& model) {
  std::vector<std::size_t> selected;
  for (std::size_t i = 0; i < model.plugins.size(); ++i)
    if (g_ascii_strcasecmp(model.plugins[i].processor.c_str(), "MLA") == 0)
      selected.push_back(i);
  return selected;
}
std::vector<MlaLayer> collect(const pipeline_internal::sima::MpkContract& model) {
  return collect_mla_layer_stats(model, all_mla(model));
}
std::string emit(const std::string& text) {
  return "cat > \"${6%.yaml}_output.yaml\" <<'YAML'\n" + text + "YAML\n";
}

// Scheduler and Runner own the mla-rt command and stats parsing; these tests cover Core's stages.
void stages_run_in_execution_order() {
  Fixture fixture;
  const auto log = fixture.root / "commands";
  fixture.runtime("echo \"$(basename \"$6\") $(basename \"$7\")\" >> '" + log.string() + "'\n" +
                  emit(output));
  auto model = contract(fixture);
  for (const auto& [mla, a65] : std::vector<std::pair<std::string, std::string>>{
           {"mla", "a65"}, {"MLA", "A65"}, {"mLa", "a65"}}) {
    model.plugins[0].processor = mla;
    model.plugins[2].processor = mla;
    model.plugins[3].processor = a65;
    std::filesystem::remove(log);
    const auto records = collect(model);
    require(records.size() == 4 && records[0].stage_index == 0 && records[2].stage_index == 2,
            "MLA/A65 execution order must determine stage indices in any processor case");
    require(records[0].name == "MLA_0/noop" && records[0].latency_us == 0 &&
                records[1].name == "MLA_0/conv: quoted" && records[1].latency_us == 161.95,
            "retain no-ops, compiler names, and hardware times");
    require(read(log) == "demo_stage1_mla_stats.yaml demo_stage1_mla.elf\n"
                         "demo_stage2_mla_stats.yaml demo_stage2_mla.elf\n",
            "each MLA ELF must run once with its matching stats YAML, in execution order");
  }
}

void failures_name_the_stage() {
  Fixture fixture;
  fixture.runtime("echo 'runtime diagnostic' >&2\nexit 7\n");
  fails([&] { collect(contract(fixture)); },
        "MLA stage 0 (demo_stage1_mla.elf): mla-rt failed (exit 7)\nruntime diagnostic");
  fails([&] { collect(contract(fixture, "0: {name: placeholder}\n")); },
        "MLA stage 0 (demo_stage1_mla.elf): input stats YAML has no layers mla-rt can time");
}

// The selection holds MPK plugin indices; reported stage indices still count the full model.
void selected_route_keeps_stage_indices() {
  Fixture fixture;
  const auto log = fixture.root / "commands";
  fixture.runtime("echo \"$(basename \"$7\")\" >> '" + log.string() + "'\n" + emit(output));
  auto model = contract(fixture);
  const std::vector<std::size_t> later{0};
  // An excluded stage's files are never opened.
  std::filesystem::remove(fixture.root / "etc/demo_stage1_mla_stats.yaml");
  const auto records = collect_mla_layer_stats(model, later);
  require(records.size() == 2 && records[0].stage_index == 2 &&
              read(log) == "demo_stage2_mla.elf\n",
          "selected later MLA keeps its full execution-order index");
  for (const auto& invalid : std::vector<std::vector<std::size_t>>{{0, 0}, {1}, {99}})
    fails([&] { collect_mla_layer_stats(model, invalid); }, "stage identity");
  fails([&] { collect_mla_layer_stats(model, {}); }, "selected inference route has no MLA stages");
}

void terminal_route_preserves_mpk_identity() {
  std::vector<ExecutionStage> stages(4);
  stages[0].kind = ExecutionStageKind::Mla;
  stages[0].mpk_plugin_index = 8;
  stages[0].stage_name = "first";
  stages[0].processor = "MLA";
  stages[1].kind = ExecutionStageKind::Unknown;
  stages[1].stage_name = "physical-cvu-cohort";
  stages[1].processor = "A65";
  stages[1].plugin_id = "processcvu";
  stages[2].kind = ExecutionStageKind::Mla;
  stages[2].mpk_plugin_index = 2;
  stages[2].stage_name = "last";
  stages[2].processor = "MLA";
  stages[3].stage_name = "tail";
  const std::vector<std::size_t> first{8}, both{8, 2};
  require(select_mla_profile_plugins(stages, {}) == both, "default route includes both MLA stages");
  for (int selector = 0; selector < 4; ++selector) {
    InferenceTerminalPolicy policy;
    if (selector == 0)
      policy.last_stage_index = 1;
    if (selector == 1)
      policy.last_stage_name = "physical-cvu-cohort";
    if (selector == 2)
      policy.last_plugin_id = "PROCESSCVU";
    if (selector == 3)
      policy.last_processor = "a65";
    require(select_mla_profile_plugins(stages, policy) == first,
            "same physical terminal selector excludes later MLA and preserves MPK index");
  }
  InferenceTerminalPolicy policy;
  policy.mla_only = true;
  require(select_mla_profile_plugins(stages, policy) == both, "mla_only ends at last MLA");
  policy.last_stage_index = 9;
  fails([&] { select_mla_profile_plugins(stages, policy); }, "index out of range");
  stages[2].mpk_plugin_index.reset();
  fails([&] { select_mla_profile_plugins(stages, {}); }, "exact MPK stage identity");
  policy.last_stage_index = 0;
  require(select_mla_profile_plugins(stages, policy) == first,
          "unselected later stage does not require a profiling identity");
}

void model_problems_are_reported() {
  Fixture fixture;
  auto model = contract(fixture);
  const auto marker = fixture.root / "mla-rt-ran";
  fixture.runtime("touch '" + marker.string() + "'\n" + emit(output));
  // Only the second stage is missing files, so the first must not run either.
  std::filesystem::remove(fixture.root / "etc/demo_stage2_mla_stats.yaml");
  fails([&] { collect(model); }, "missing matching stats YAML");
  std::filesystem::remove(fixture.root / "share/demo_stage2_mla.elf");
  fails([&] { collect(model); }, "missing MLA ELF");
  require(!std::filesystem::exists(marker), "every stage must be checked before mla-rt runs");
  model.plugins = {plugin("A65", 1, "host.so")};
  fails([&] { collect(model); }, "no MLA stages");
}
} // namespace

RUN_TEST("unit_benchmark_mla_layer_stats_test", ([] {
           stages_run_in_execution_order();
           failures_name_the_stage();
           selected_route_keeps_stage_indices();
           terminal_route_preserves_mpk_identity();
           model_problems_are_reported();
         }));
