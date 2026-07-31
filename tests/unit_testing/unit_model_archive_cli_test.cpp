#include "model/internal/ModelArchiveLoader.h"
#include "model_archive_fixture_utils.h"
#include "model_archive_test_utils.h"
#include "test_main.h"
#include "test_utils.h"

#include <sys/wait.h>

#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <string>
#include <vector>

namespace {

namespace fs = sima_test::fs;
using simaai::neat::internal::ModelArchiveError;
using simaai::neat::internal::ModelArchiveLoader;
using simaai::neat::internal::ModelArchiveLoaderOptions;

/// Must stay identical to runtime_parity_options() in tools/model_archive_cli/main.cpp, or the
/// parity check below compares configurations rather than acceptance.
ModelArchiveLoaderOptions runtime_parity_options() {
  ModelArchiveLoaderOptions opt;
  opt.reject_unsupported_file_types = false;
  opt.require_pipeline_sequence = false;
  return opt;
}

/// The build-tree path is compiled in, but CI installs the package and runs this test from the
/// extras tarball on another machine, where that path does not exist. Fall back to the command the
/// Core package installs, so the packaged run exercises the shipped helper.
const std::string& helper_path() {
  static const std::string resolved = [] {
    const char* override_path = std::getenv("NEAT_MODEL_ARCHIVE_BIN");
    if (override_path && *override_path) {
      return std::string(override_path);
    }
    if (fs::exists(NEAT_MODEL_ARCHIVE_BIN)) {
      return std::string(NEAT_MODEL_ARCHIVE_BIN);
    }
    const std::string installed = "/usr/bin/neat-model-archive";
    require(fs::exists(installed), "neat-model-archive found at neither " +
                                       std::string(NEAT_MODEL_ARCHIVE_BIN) + " nor " + installed);
    return installed;
  }();
  return resolved;
}

int run_cli(const std::vector<std::string>& args) {
  std::string cmd = sima_test::model_archive_shell_quote(helper_path());
  for (const auto& arg : args) {
    cmd += " " + sima_test::model_archive_shell_quote(arg);
  }
  cmd += " >/dev/null 2>&1";
  const int status = std::system(cmd.c_str());
  require(WIFEXITED(status), "neat-model-archive did not exit normally");
  return WEXITSTATUS(status);
}

bool loader_accepts(const fs::path& archive) {
  try {
    ModelArchiveLoader::inspect(archive.string(), runtime_parity_options());
    return true;
  } catch (const ModelArchiveError&) {
    return false;
  }
}

std::vector<fs::path> archive_fixtures() {
  std::vector<fs::path> archives;
  for (const char* group : {"valid", "invalid"}) {
    const fs::path dir = sima_test::model_archive_fixture_path(group);
    std::error_code ec;
    if (!fs::is_directory(dir, ec)) {
      continue;
    }
    for (const auto& entry : fs::directory_iterator(dir)) {
      if (entry.is_regular_file() && entry.path().extension() == ".gz") {
        archives.push_back(entry.path());
      }
    }
  }
  return archives;
}

bool has_staging_residue(const fs::path& parent) {
  for (const auto& entry : fs::directory_iterator(parent)) {
    if (entry.path().filename().string().rfind(".neat-model-archive.", 0) == 0) {
      return true;
    }
  }
  return false;
}

std::string read_file(const fs::path& path) {
  std::ifstream in(path, std::ios::binary);
  return std::string((std::istreambuf_iterator<char>(in)), std::istreambuf_iterator<char>());
}

} // namespace

RUN_TEST(
    "unit_model_archive_cli_test", ([] {
      const std::vector<fs::path> fixtures = archive_fixtures();
      require(!fixtures.empty(),
              "missing model archive fixtures; run tests/tools/make_model_archive_fixtures.py");

      for (const auto& archive : fixtures) {
        const bool accepted = loader_accepts(archive);
        const int rc = run_cli({"validate", archive.string()});
        require(accepted == (rc == 0), "validate disagrees with the runtime loader on " +
                                           archive.filename().string() +
                                           " (loader accepted=" + (accepted ? "yes" : "no") +
                                           ", exit=" + std::to_string(rc) + ")");
      }

      const auto fixture = sima_test::make_model_archive_fixture(
          "cli_extract",
          {
              {"etc/0_process_mla.json",
               R"({"node_name":"mla_0","simaai__params":{"model_path":"model.elf"}})"},
          });
      const fs::path work(sima_test::make_temp_dir("cli_extract_out"));
      const fs::path output = work / "package";

      require(run_cli({"extract", fixture.tar_path, "--output", output.string()}) == 0,
              "extract should publish a package at the requested output");
      for (const char* dir : {"etc", "lib", "share"}) {
        require(fs::is_directory(output / dir), std::string("extracted package is missing ") + dir);
      }
      require(!has_staging_residue(work), "extract left a staging directory behind");

      // The rewrite runs while the package still lives in staging, so this is what proves the
      // published JSON names the path the caller asked for.
      const std::string rewritten = read_file(output / "etc" / "0_process_mla.json");
      require(rewritten.find((output / "share" / "model.elf").string()) != std::string::npos,
              "extracted config should point at the published package root, got: " + rewritten);

      require(run_cli({"extract", fixture.tar_path, "--output", output.string()}) != 0,
              "extract should refuse an existing output");
      require(read_file(output / "etc" / "0_process_mla.json") == rewritten,
              "a refused extract must leave the existing output untouched");

      const auto malformed = sima_test::make_malformed_model_archive_fixture("cli_malformed");
      const fs::path failed_output = work / "failed";
      require(run_cli({"extract", malformed.tar_path, "--output", failed_output.string()}) != 0,
              "extract should fail on a malformed archive");
      require(!fs::exists(failed_output), "a failed extract must not create the output");
      require(!has_staging_residue(work), "a failed extract left a staging directory behind");
    }));
