#ifndef SIMA_NEAT_INTERNAL
#define SIMA_NEAT_INTERNAL 1
#endif
#include "model/internal/MlaStageStats.h"
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
    gchar* directory = g_dir_make_tmp("neat-stage-stats-test-XXXXXX", nullptr);
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
pipeline_internal::sima::MpkContract contract(Fixture& fixture) {
  fixture.write("share/demo_stage1_mla.elf", "elf fixture");
  fixture.write("share/demo_stage2_mla.elf", "elf fixture");
  pipeline_internal::sima::MpkContract result;
  result.mpk_json_path = (fixture.root / "etc/demo_mpk.json").string();
  result.plugins = {plugin("mla", 4, "demo_stage2_mla.elf"), plugin("EV74", 1, "kernel"),
                    plugin("MLA", 2, "demo_stage1_mla.elf"), plugin("A65", 3, "host.so")};
  return result;
}
// Mirrors mla-rt -vss: a microsecond header and one root entry keyed by the ELF path as given.
std::string stats(const std::string& tile) {
  return "# precision:dt:usec precision:0\n\"$8\":\n  filesize: 11\n  tile.lat:\n" + tile;
}
std::string emit(const std::string& text) {
  return "cat > \"${7#--stats=}\" <<YAML\n" + text + "YAML\n";
}
const std::string average =
    "    avg: 655\n    runcount: 200\n    min: 650\n    max: 700\n    std: 3\n";

void exact_command_and_stage_order() {
  Fixture fixture;
  const auto log = fixture.root / "commands";
  const auto command = fixture.runtime(
      "[ $# -eq 8 ] && [ \"$1\" = --connect ] && [ \"$2\" = kernel ] && [ \"$3\" = -N ] && "
      "[ \"$4\" = -m ] && [ \"$5\" = 200 ] && [ \"$6\" = -vss ]\n"
      "case \"$7\" in --stats=*) ;; *) exit 9 ;; esac\n"
      "basename \"$8\" >> '" +
      log.string() + "'\n" + emit(stats(average)));
  const auto records = collect_mla_stage_stats(contract(fixture), command);
  require(records.size() == 2 && records[0].stage_index == 0 && records[1].stage_index == 2,
          "MLA/A65 execution order must determine stage indices");
  require(records[0].latency_us == 655 && records[1].latency_us == 655,
          "latency must be the tile.lat average");
  require(read(log) == "demo_stage1_mla.elf\ndemo_stage2_mla.elf\n",
          "each MLA ELF must run once, in execution order");
}

void process_failures_are_reported() {
  Fixture fixture;
  const auto model = contract(fixture);
  fails([&] { collect_mla_stage_stats(model, fixture.root / "missing-runtime"); },
        "cannot start mla-rt");
  fails(
      [&] { collect_mla_stage_stats(model, fixture.runtime("echo 'device busy' >&2\nexit 7\n")); },
      "device busy");
  fails([&] { collect_mla_stage_stats(model, fixture.runtime("exit 0\n")); },
        "wrote no stats file\nMLA stage 0 (demo_stage1_mla.elf)");
}

void unusable_stats_are_rejected() {
  Fixture fixture;
  const auto model = contract(fixture);
  for (const auto& [text, error] : std::vector<std::pair<std::string, std::string>>{
           {"\"$8\":\n  tile.lat:\n" + average, "not in microseconds"},
           {stats(average) + "other.elf:\n  filesize: 1\n", "exactly one ELF"},
           {"# precision:dt:usec precision:0\n\"$8\":\n  filesize: 11\n", "no tile.lat average"},
           {stats("    avg: 655\n    runcount: 199\n"), "measured 199 runs, expected 200"},
           {stats("    avg: 0\n    runcount: 200\n"), "must be positive"},
           {stats("    avg: [1\n"), "invalid stats file"}})
    fails([&] { collect_mla_stage_stats(model, fixture.runtime(emit(text))); }, error);
}

void model_problems_are_reported() {
  Fixture fixture;
  auto model = contract(fixture);
  const auto command = fixture.runtime(emit(stats(average)));
  std::filesystem::remove(fixture.root / "share/demo_stage2_mla.elf");
  fails([&] { collect_mla_stage_stats(model, command); }, "missing MLA ELF");
  model.plugins = {plugin("A65", 1, "host.so")};
  fails([&] { collect_mla_stage_stats(model, command); }, "no MLA stages");
}
} // namespace

RUN_TEST("unit_benchmark_stage_stats_test", ([] {
           exact_command_and_stage_order();
           process_failures_are_reported();
           unusable_stats_are_rejected();
           model_problems_are_reported();
         }));
