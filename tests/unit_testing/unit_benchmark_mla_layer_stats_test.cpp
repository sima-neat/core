#ifndef SIMA_NEAT_INTERNAL
#define SIMA_NEAT_INTERNAL 1
#endif
#include "model/internal/MlaLayerStats.h"
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
std::string capture(const Fixture& fixture) {
  // Quote fixture paths containing spaces.
  return "printf '%s' \"$6\" > '" + (fixture.root / "copied-path").string() + "'\n";
}
std::string emit(const std::string& text) {
  return "cat > \"${6%.yaml}_output.yaml\" <<'YAML'\n" + text + "YAML\n";
}

void exact_command_and_stage_order() {
  Fixture fixture;
  const auto log = fixture.root / "commands";
  const auto command = fixture.runtime(
      "[ $# -eq 7 ] && [ \"$1\" = --connect ] && [ \"$2\" = mip,ma ] && [ \"$3\" = -vs ] && "
      "[ \"$4\" = -N ] && [ \"$5\" = --per-layer-stats ]\n"
      "echo \"$(basename \"$6\") $(basename \"$7\")\" >> '" +
      log.string() + "'\n" + emit(output));
  auto model = contract(fixture);
  for (const auto& [mla, a65] : std::vector<std::pair<std::string, std::string>>{
           {"mla", "a65"}, {"MLA", "A65"}, {"mLa", "a65"}}) {
    model.plugins[0].processor = mla;
    model.plugins[2].processor = mla;
    model.plugins[3].processor = a65;
    std::filesystem::remove(log);
    const auto records = collect_mla_layer_stats(model, command);
    require(records.size() == 4 && records[0].stage_index == 0 && records[2].stage_index == 2,
            "MLA/A65 execution order must determine stage indices in any processor case");
    require(read(log) == "demo_stage1_mla_stats.yaml demo_stage1_mla.elf\n"
                         "demo_stage2_mla_stats.yaml demo_stage2_mla.elf\n",
            "each MLA ELF must run once with its matching stats YAML, in execution order");
  }
}

void successful_profile_preserves_original_and_cleans_files() {
  Fixture fixture;
  const auto command = fixture.runtime(capture(fixture) + emit(output));
  const auto records = collect_mla_layer_stats(contract(fixture), command);
  require(records.size() == 4 && records[0].name == "MLA_0/noop" && records[0].latency_us == 0 &&
              records[1].name == "MLA_0/conv: quoted" && records[1].latency_us == 161.95,
          "retain no-ops, compiler names, and hardware times");
  require(read(fixture.root / "etc/demo_stage1_mla_stats.yaml") == input,
          "compiler YAML must remain unchanged");
  require(!std::filesystem::exists(read(fixture.root / "copied-path")), "temporary YAML cleaned");
}

void unrelated_metadata_is_ignored() {
  Fixture fixture;
  const auto source = input + "  compiler_metadata:\n    tags: [conv, quantized]\n";
  const auto command = fixture.runtime(emit(output + "  debug_metadata: {version: 1}\n"));
  const auto records = collect_mla_layer_stats(contract(fixture, source, source), command);
  require(records.size() == 4 && records[1].latency_us == 161.95,
          "unrelated nested metadata must not change layer timing collection");
  require(read(fixture.root / "etc/demo_stage1_mla_stats.yaml") == source,
          "compiler metadata must be preserved unchanged");
}

void invalid_inputs_are_rejected_before_profiling() {
  Fixture fixture;
  const auto marker = fixture.root / "mla-rt-ran";
  const auto command = fixture.runtime("touch '" + marker.string() + "'\n" + emit(output));
  for (const auto& [text, error] : std::vector<std::pair<std::string, std::string>>{
           {input + input, "duplicate layer ID"},
           {input + "  name: duplicate\n", "duplicate layer field"},
           {input + "---\n{}\n", "exactly one document"},
           {"0: {name: placeholder}\n2: {name: conv, start_cycle: 0, end_cycle: 20}\n",
            "contiguous from zero"},
           {"0: {name: placeholder}\n", "no layers mla-rt can time"}}) {
    // Only the second stage is invalid, so the first must not run either.
    fails([&] { collect_mla_layer_stats(contract(fixture, input, text), command); }, error);
    require(!std::filesystem::exists(marker), "every stage must be validated before mla-rt runs");
  }
}

void model_problems_are_reported() {
  Fixture fixture;
  auto model = contract(fixture);
  const auto command = fixture.runtime(emit(output));
  std::filesystem::remove(fixture.root / "etc/demo_stage2_mla_stats.yaml");
  fails([&] { collect_mla_layer_stats(model, command); }, "missing matching stats YAML");
  std::filesystem::remove(fixture.root / "share/demo_stage2_mla.elf");
  fails([&] { collect_mla_layer_stats(model, command); }, "missing MLA ELF");
  model.plugins = {plugin("A65", 1, "host.so")};
  fails([&] { collect_mla_layer_stats(model, command); }, "no MLA stages");
}

void process_failures_clean_up() {
  Fixture fixture;
  const auto model = contract(fixture);
  auto command = fixture.runtime(capture(fixture) + "echo 'runtime diagnostic' >&2\nexit 7\n");
  fails([&] { collect_mla_layer_stats(model, command); }, "runtime diagnostic");
  require(!std::filesystem::exists(read(fixture.root / "copied-path")), "failed run cleaned");
  fails([&] { collect_mla_layer_stats(model, fixture.root / "missing-runtime"); },
        "cannot start mla-rt");
  command = fixture.runtime(capture(fixture) + "echo 'no stats generated' >&2\n");
  fails([&] { collect_mla_layer_stats(model, command); },
        "MLA stage 0 (demo_stage1_mla.elf)\nmla-rt diagnostics:\nno stats generated");
  require(!std::filesystem::exists(read(fixture.root / "copied-path")), "missing output cleaned");
}

void invalid_outputs_are_rejected() {
  Fixture fixture;
  const auto model = contract(fixture);
  for (const auto& [text, error] : std::vector<std::pair<std::string, std::string>>{
           {"", "must be a mapping"},
           {"[invalid", "invalid YAML"},
           {"2:\n  name: wrong\n  start_cycle: 5\n  end_cycle: 20\n  run_time: 1us\n",
            "does not match"},
           {output + output, "duplicate layer ID"}}) {
    const auto command = fixture.runtime(capture(fixture) + emit(text));
    fails([&] { collect_mla_layer_stats(model, command); }, error);
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
    fails([&] { collect_mla_layer_stats(model, command); }, error);
  }
  auto text = output;
  text.replace(text.find("0.00us"), 6, "100us");
  const auto command = fixture.runtime(emit(text));
  fails([&] { collect_mla_layer_stats(model, command); }, "zero-cycle layer");
  require(read(fixture.root / "etc/demo_stage1_mla_stats.yaml") == input,
          "invalid output must not modify source YAML");
}
} // namespace

RUN_TEST("unit_benchmark_mla_layer_stats_test", ([] {
           exact_command_and_stage_order();
           successful_profile_preserves_original_and_cleans_files();
           unrelated_metadata_is_ignored();
           invalid_inputs_are_rejected_before_profiling();
           model_problems_are_reported();
           process_failures_clean_up();
           invalid_outputs_are_rejected();
         }));
