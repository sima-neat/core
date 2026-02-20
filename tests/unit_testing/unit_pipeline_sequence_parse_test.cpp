#include "mpk/MpKLoader.h"
#include "mpk/PipelineSequence.h"
#include "mpk_test_utils.h"
#include "test_main.h"
#include "test_utils.h"

#include <filesystem>
#include <fstream>
#include <functional>
#include <string>
#include <vector>

namespace {

std::string extract_etc_dir(const std::filesystem::path& fixture_path, const std::string& tag) {
  const std::string out_root = sima_test::make_temp_dir(tag);
  const auto extracted = simaai::neat::mpk::MpKLoader::extract(fixture_path.string(), out_root);
  return extracted.etc_dir;
}

bool throws_with(const std::function<void()>& fn, const std::string& needle) {
  try {
    fn();
  } catch (const std::exception& e) {
    if (needle.empty())
      return true;
    return std::string(e.what()).find(needle) != std::string::npos;
  }
  return false;
}

std::vector<std::string> names_only(const std::vector<simaai::neat::mpk::SequenceEntry>& seq) {
  std::vector<std::string> names;
  names.reserve(seq.size());
  for (const auto& e : seq) {
    names.push_back(e.name);
  }
  return names;
}

void write_text(const std::filesystem::path& p, const std::string& text) {
  std::error_code ec;
  std::filesystem::create_directories(p.parent_path(), ec);
  std::ofstream out(p);
  if (!out.is_open()) {
    throw std::runtime_error("failed to write test file: " + p.string());
  }
  out << text;
}

} // namespace

RUN_TEST(
    "unit_pipeline_sequence_parse_test", ([] {
      namespace fs = std::filesystem;
      using simaai::neat::mpk::load_pipeline_sequence;

      const fs::path basic = sima_test::mpk_fixture_path("valid/basic_valid.mpk");
      require(
          fs::exists(basic),
          "missing MPK fixtures; expected tests/assets/mpk (run tests/tools/make_mpk_fixtures.py)");

      const std::string basic_etc = extract_etc_dir(basic, "pipeline_sequence_parse_valid");
      const auto seq0 = load_pipeline_sequence(basic_etc);
      require(seq0.size() == 2, "basic pipeline sequence should contain two stages");
      require(seq0[0].name == "preproc_0", "sequence[0] should be preproc_0");
      require(seq0[1].name == "mla_0", "sequence[1] should be mla_0");

      const auto seq1 = load_pipeline_sequence(basic_etc);
      require(names_only(seq0) == names_only(seq1),
              "load_pipeline_sequence should be deterministic across repeated reads");

      {
        const std::string tmp = sima_test::make_temp_dir("pipeline_sequence_duplicate");
        const fs::path json_path = fs::path(tmp) / "pipeline_sequence.json";
        write_text(json_path,
                   R"json({
  "pipelines": [{
    "sequence": [
      {
        "sequence_id": 1,
        "name": "dup_stage",
        "pluginId": "processcvu",
        "configPath": "0_preproc.json",
        "processor": "CVU",
        "kernel": "preproc",
        "input": "decoder"
      },
      {
        "sequence_id": 2,
        "name": "dup_stage",
        "pluginId": "processmla",
        "configPath": "0_process_mla.json",
        "processor": "MLA",
        "kernel": "infer",
        "input": "dup_stage"
      }
    ]
  }]
})json");
        require(throws_with([&]() { (void)load_pipeline_sequence(tmp); }, "schema_error"),
                "duplicate stage names should fail with schema_error");
      }

      {
        const std::string tmp = sima_test::make_temp_dir("pipeline_sequence_invalid_dependency");
        const fs::path json_path = fs::path(tmp) / "pipeline_sequence.json";
        write_text(json_path,
                   R"json({
  "pipelines": [{
    "sequence": [
      {
        "sequence_id": 1,
        "name": "preproc_0",
        "pluginId": "processcvu",
        "configPath": "0_preproc.json",
        "processor": "CVU",
        "kernel": "preproc",
        "input": "decoder"
      },
      {
        "sequence_id": 2,
        "name": "infer_0",
        "pluginId": "processmla",
        "configPath": "0_process_mla.json",
        "processor": "MLA",
        "kernel": "infer",
        "input": "missing_stage"
      }
    ]
  }]
})json");
        require(throws_with([&]() { (void)load_pipeline_sequence(tmp); }, "schema_error"),
                "invalid dependency should fail with schema_error");
      }

      {
        const std::string tmp = sima_test::make_temp_dir("pipeline_sequence_unknown_kernel");
        const fs::path json_path = fs::path(tmp) / "pipeline_sequence.json";
        write_text(json_path,
                   R"json({
  "pipelines": [{
    "sequence": [{
      "sequence_id": 1,
      "name": "mystage",
      "pluginId": "processmla",
      "configPath": "0_process_mla.json",
      "processor": "MLA",
      "kernel": "unknown_kernel",
      "input": "decoder"
    }]
  }]
})json");
        require(throws_with([&]() { (void)load_pipeline_sequence(tmp); }, "schema_error"),
                "unknown kernel should fail with schema_error");
      }

      {
        const std::string tmp = sima_test::make_temp_dir("pipeline_sequence_invalid_types");
        const fs::path json_path = fs::path(tmp) / "pipeline_sequence.json";
        write_text(json_path,
                   R"json({
  "pipelines": [{
    "sequence": [{
      "sequence_id": 9223372036854775807,
      "name": "stage0",
      "pluginId": "processmla",
      "configPath": "0_process_mla.json",
      "processor": "MLA",
      "kernel": "infer",
      "input": ["decoder", 7]
    }]
  }]
})json");

        require(throws_with([&]() { (void)load_pipeline_sequence(tmp); }, "schema_error"),
                "overflow/typed-input violations should fail with schema_error");
      }

      {
        const std::string tmp = sima_test::make_temp_dir("pipeline_sequence_bad_path");
        const fs::path json_path = fs::path(tmp) / "pipeline_sequence.json";
        write_text(json_path,
                   R"json({
  "pipelines": [{
    "sequence": [{
      "sequence_id": 1,
      "name": "stage0",
      "pluginId": "processmla",
      "configPath": "../escape.json",
      "processor": "MLA",
      "kernel": "infer",
      "input": "decoder"
    }]
  }]
})json");

        require(throws_with([&]() { (void)load_pipeline_sequence(tmp); }, "schema_error"),
                "configPath traversal should fail with schema_error");
      }
    }));
