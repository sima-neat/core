#ifndef SIMA_NEAT_INTERNAL
#define SIMA_NEAT_INTERNAL 1
#endif
#include "model/internal/BenchmarkInternal.h"
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
  std::filesystem::path runtime(const std::string& body) {
    const auto file = write("fake mla-rt", "#!/bin/sh\nset -eu\n" + body);
    std::filesystem::permissions(file, std::filesystem::perms::owner_all);
    return file;
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
LayerProfileStage stage(Fixture& fixture) {
  return {2, fixture.write("share/demo_stage1_mla.elf", "elf fixture"),
          fixture.write("etc/demo_stage1_mla_stats.yaml", input)};
}
std::string capture(const Fixture& fixture) {
  // Fixture paths have spaces; quote every shell argument in the fake runtime too.
  return "printf '%s' \"$6\" > '" + (fixture.root / "copied-path").string() + "'\n";
}
std::string emit(const std::string& text) {
  return "cat > \"${6%.yaml}_output.yaml\" <<'YAML'\n" + text + "YAML\n";
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
  std::filesystem::remove(stages[0].stats_yaml);
  fails([&] { resolve_layer_profile_stages(contract); }, "missing matching stats YAML");
}
void successful_profile_preserves_original_and_cleans_files() {
  Fixture fixture;
  const auto selected = stage(fixture);
  const auto command =
      fixture.runtime("[ \"$1\" = --connect ] && [ \"$2\" = mip,ma ] && [ \"$3\" = -vs ] && "
                      "[ \"$4\" = -N ] && [ \"$5\" = --per-layer-stats ]\n" +
                      capture(fixture) + emit(output));
  const auto records =
      collect_mla_layer_stats({selected, {0, selected.executable, selected.stats_yaml}}, command);
  require(records.size() == 4 && records[0].name == "MLA_0/noop" &&
              records[0].total_execution_time_us == 0 &&
              records[1].total_execution_time_us == 161.95 && records[0].stage_index == 2 &&
              records[2].stage_index == 0,
          "retain no-ops, compiler names, hardware times, and stage identity");
  require(read(selected.stats_yaml) == input, "compiler YAML must remain unchanged");
  require(!std::filesystem::exists(read(fixture.root / "copied-path")), "temporary YAML cleaned");
}
void process_failures_and_timeout_clean_up() {
  Fixture fixture;
  const auto selected = stage(fixture);
  auto command = fixture.runtime(capture(fixture) + "echo 'runtime diagnostic' >&2\nexit 7\n");
  fails([&] { collect_mla_layer_stats({selected}, command); }, "runtime diagnostic");
  require(!std::filesystem::exists(read(fixture.root / "copied-path")), "failed run cleaned");
  command = fixture.runtime(capture(fixture) + "sleep 30 &\nexit 0\n");
  const auto begin = std::chrono::steady_clock::now();
  fails([&] { collect_mla_layer_stats({selected}, command, std::chrono::milliseconds(80)); },
        "timed out");
  require(std::chrono::steady_clock::now() - begin < std::chrono::seconds(3), "timeout is bounded");
  require(!std::filesystem::exists(read(fixture.root / "copied-path")), "timeout cleaned");
  fails([&] { collect_mla_layer_stats({selected}, fixture.root / "missing-runtime"); },
        "cannot start mla-rt");
  command = fixture.runtime(capture(fixture) + "echo 'no stats generated' >&2\n");
  fails([&] { collect_mla_layer_stats({selected}, command); }, "no stats generated");
  require(!std::filesystem::exists(read(fixture.root / "copied-path")), "missing output cleaned");
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
    const auto command = fixture.runtime(capture(fixture) + emit(text));
    fails([&] { collect_mla_layer_stats({selected}, command); }, error);
    require(!std::filesystem::exists(read(fixture.root / "copied-path")), "parse failure cleaned");
  }
  for (const auto& [time, error] :
       std::vector<std::pair<std::string, std::string>>{{"nanus", "invalid run_time"},
                                                        {"-1us", "invalid run_time"},
                                                        {"1ms", "microsecond units"},
                                                        {"0us", "no hardware timings"}}) {
    auto text = output;
    text.replace(text.find("161.95us"), 8, time);
    const auto command = fixture.runtime(emit(text));
    fails([&] { collect_mla_layer_stats({selected}, command); }, error);
  }
  auto text = output;
  text.replace(text.find("0.00us"), 6, "100us");
  const auto command = fixture.runtime(emit(text));
  fails([&] { collect_mla_layer_stats({selected}, command); }, "zero-cycle layer");
  require(read(selected.stats_yaml) == input, "invalid output must not modify source YAML");
}
} // namespace

RUN_TEST("unit_benchmark_layer_stats_test", ([] {
           pairing_and_stage_order();
           successful_profile_preserves_original_and_cleans_files();
           process_failures_and_timeout_clean_up();
           invalid_outputs_are_rejected();
         }));
