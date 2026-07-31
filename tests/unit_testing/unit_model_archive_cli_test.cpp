#include "model_archive_fixture_utils.h"
#include "model_archive_test_utils.h"
#include "model/Model.h"
#include "test_main.h"
#include "test_utils.h"

#include <sys/wait.h>

#include <cstdlib>
#include <filesystem>
#include <string>
#include <vector>

namespace {

namespace fs = std::filesystem;

int run_cli(const std::vector<std::string>& args) {
  const fs::path built_cli(NEAT_MODEL_ARCHIVE_BIN);
  std::string command = sima_test::model_archive_shell_quote(
      fs::exists(built_cli) ? built_cli.string() : "neat-model-archive");
  for (const auto& arg : args) {
    command += " " + sima_test::model_archive_shell_quote(arg);
  }
  command += " >/dev/null 2>&1";
  const int status = std::system(command.c_str());
  require(WIFEXITED(status), "neat-model-archive did not exit normally");
  return WEXITSTATUS(status);
}

bool has_staging_directory(const fs::path& parent) {
  for (const auto& entry : fs::directory_iterator(parent)) {
    if (entry.path().filename().string().starts_with(".neat-model-archive.")) {
      return true;
    }
  }
  return false;
}

} // namespace

RUN_TEST(
    "unit_model_archive_cli_test", ([] {
      const fs::path valid = sima_test::model_archive_fixture_path("valid/basic_valid.tar.gz");
      require(fs::exists(valid), "missing basic_valid fixture");
      require(run_cli({"validate", valid.string()}) == 0,
              "validate must accept a model package with a valid MPK contract");

      const fs::path canonical =
          sima_test::model_archive_fixture_path("valid/canonical_mpk_name.tar.gz");
      require(run_cli({"validate", canonical.string()}) == 0,
              "validate must accept the canonical mpk.json name");

      const fs::path auxiliary =
          sima_test::model_archive_fixture_path("valid/auxiliary_json_ignored.tar.gz");
      require(run_cli({"validate", auxiliary.string()}) == 0,
              "validate must ignore auxiliary JSON contents");
      simaai::neat::Model auxiliary_model(auxiliary.string());

      const fs::path work = sima_test::make_temp_dir("model_archive_cli_contract");
      const fs::path output = work / "valid-package";
      require(run_cli({"extract", valid.string(), "--output", output.string()}) == 0,
              "extract must publish a package with a valid MPK contract");
      simaai::neat::Model model(output.string());
      require(fs::path(model.info().mpk_json_path).parent_path().parent_path() == output,
              "Model must load the CLI-extracted package in place");

      const fs::path missing_mpk =
          sima_test::model_archive_fixture_path("invalid/missing_mpk_contract.tar.gz");
      require(fs::exists(missing_mpk), "missing missing_mpk_contract fixture");
      require(run_cli({"validate", missing_mpk.string()}) != 0,
              "validate must reject a model package without *_mpk.json");
      const fs::path missing_output = work / "missing-mpk";
      require(run_cli({"extract", missing_mpk.string(), "--output", missing_output.string()}) != 0,
              "extract must reject a model package without *_mpk.json");
      require(!fs::exists(missing_output), "failed MPK validation must not publish output");
      require(!has_staging_directory(work), "failed MPK validation must remove staging output");

      const fs::path malformed_mpk =
          sima_test::model_archive_fixture_path("invalid/malformed_mpk_contract.tar.gz");
      require(fs::exists(malformed_mpk), "missing malformed_mpk_contract fixture");
      require(run_cli({"validate", malformed_mpk.string()}) != 0,
              "validate must reject a malformed MPK contract");
      const fs::path malformed_output = work / "malformed-mpk";
      require(run_cli({"extract", malformed_mpk.string(), "--output", malformed_output.string()}) !=
                  0,
              "extract must reject a malformed MPK contract");
      require(!fs::exists(malformed_output), "malformed MPK must not publish output");
      require(!has_staging_directory(work), "malformed MPK must not leave staging output");

      std::error_code ec;
      fs::remove_all(work, ec);
    }));
