#include "model/internal/ModelPack.h"
#include "mpk_test_utils.h"
#include "test_main.h"
#include "test_utils.h"

#include <filesystem>
#include <functional>
#include <string>
#include <vector>

namespace {

using simaai::neat::internal::ModelPack;

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

std::vector<std::string> stage_names(const simaai::neat::mpk::SequenceSplit& split) {
  std::vector<std::string> names;
  for (const auto& entry : split.pre) {
    names.push_back(entry.name);
  }
  for (const auto& entry : split.infer) {
    names.push_back(entry.name);
  }
  for (const auto& entry : split.post) {
    names.push_back(entry.name);
  }
  return names;
}

} // namespace

RUN_TEST(
    "unit_modelpack_parse_test", ([] {
      namespace fs = std::filesystem;

      const fs::path basic = sima_test::mpk_fixture_path("valid/basic_valid.mpk");
      require(
          fs::exists(basic),
          "missing MPK fixtures; expected tests/assets/mpk (run tests/tools/make_mpk_fixtures.py)");

      const std::string extract_root = sima_test::make_temp_dir("modelpack_parse_extract");
      sima_test::ScopedEnvVar env_root("SIMA_MPK_EXTRACT_ROOT", extract_root);

      ModelPack pack_basic(basic.string());
      require(!pack_basic.etc_dir().empty(), "ModelPack etc_dir should not be empty");
      require(fs::exists(pack_basic.etc_dir()), "ModelPack etc_dir should exist on disk");

      const auto split0 = pack_basic.split_sequence();
      require(split0.pre.size() == 1, "basic fixture should have one pre stage");
      require(split0.infer.size() == 1, "basic fixture should have one infer stage");
      require(split0.post.empty(), "basic fixture should not have post stages");

      const std::string mla_cfg = pack_basic.find_config_path_by_plugin("processmla");
      require(!mla_cfg.empty(), "find_config_path_by_plugin(processmla) should resolve config");
      require(fs::exists(mla_cfg), "resolved MLA config path should exist");

      const std::string processor_cfg = pack_basic.find_config_path_by_processor("MLA");
      require(!processor_cfg.empty(), "find_config_path_by_processor(MLA) should resolve config");
      require(fs::exists(processor_cfg), "processor config path should exist");

      const auto split1 = pack_basic.split_sequence();
      require(stage_names(split0) == stage_names(split1),
              "split_sequence ordering should be deterministic across repeated calls");

      const fs::path multi = sima_test::mpk_fixture_path("valid/multi_stage_valid.mpk");
      ModelPack pack_multi(multi.string());
      const auto multi_split = pack_multi.split_sequence();
      require(multi_split.infer.size() == 2,
              "multi_stage_valid fixture should expose two infer stages");
      require(multi_split.infer.front().name == "mla_stage_a",
              "first infer stage should be mla_stage_a");
      require(multi_split.infer.back().name == "mla_stage_b",
              "second infer stage should be mla_stage_b");

      require(
          throws_with(
              [&]() {
                ModelPack bad(
                    sima_test::mpk_fixture_path("invalid/missing_pipeline_sequence.mpk").string());
                (void)bad.etc_dir();
              },
              "schema_error"),
          "missing pipeline_sequence fixture should fail with schema_error");

      require(throws_with(
                  [&]() {
                    ModelPack bad(
                        sima_test::mpk_fixture_path("invalid/unsupported_version.mpk").string());
                    (void)bad.etc_dir();
                  },
                  "unsupported_version"),
              "unsupported version fixture should fail with unsupported_version");

      require(throws_with(
                  [&]() {
                    ModelPack bad(
                        sima_test::mpk_fixture_path("invalid/path_traversal.mpk").string());
                    (void)bad.etc_dir();
                  },
                  "path_traversal"),
              "path traversal fixture should fail with path_traversal");
    }));
