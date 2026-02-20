#include "mpk/MpKLoader.h"
#include "mpk_test_utils.h"
#include "test_main.h"
#include "test_utils.h"

#include <filesystem>
#include <string>

namespace {

void require_mpk_error(const std::function<void()>& fn, simaai::neat::mpk::ErrorClass expected_code,
                       const std::string& context) {
  try {
    fn();
  } catch (const simaai::neat::mpk::MpKError& e) {
    require(e.code() == expected_code,
            context + ": unexpected error class " + simaai::neat::mpk::error_class_name(e.code()));
    return;
  }
  throw std::runtime_error(context + ": expected MpKError was not thrown");
}

} // namespace

RUN_TEST(
    "unit_mpk_loader_test", ([] {
      namespace fs = std::filesystem;
      using namespace simaai::neat::mpk;

      const fs::path valid = sima_test::mpk_fixture_path("valid/basic_valid.mpk");
      require(
          fs::exists(valid),
          "missing MPK fixtures; expected tests/assets/mpk (run tests/tools/make_mpk_fixtures.py)");

      const MpKManifest manifest = MpKLoader::inspect(valid.string());
      require(manifest.has_pipeline_sequence,
              "MpKLoader::inspect should detect pipeline_sequence.json");
      require(manifest.has_model_binary, "MpKLoader::inspect should detect model binary payload");
      require(!manifest.entries.empty(), "MpKLoader::inspect should expose archive entries");

      const std::string out_root = sima_test::make_temp_dir("mpk_loader_extract");
      const MpKExtractResult first = MpKLoader::extract(valid.string(), out_root);

      require(fs::exists(first.package_root), "MpKLoader::extract package_root missing");
      require(fs::exists(first.etc_dir), "MpKLoader::extract etc_dir missing");
      require(fs::exists(first.lib_dir), "MpKLoader::extract lib_dir missing");
      require(fs::exists(first.share_dir), "MpKLoader::extract share_dir missing");
      require(fs::exists(fs::path(first.etc_dir) / "pipeline_sequence.json"),
              "MpKLoader::extract should materialize pipeline_sequence.json");

      // Idempotent extraction to same root/path.
      const MpKExtractResult second = MpKLoader::extract(valid.string(), out_root);
      require(second.package_root == first.package_root,
              "MpKLoader::extract should be deterministic for same archive/root");

      require_mpk_error(
          [&]() {
            (void)MpKLoader::inspect(
                sima_test::mpk_fixture_path("invalid/missing_pipeline_sequence.mpk").string());
          },
          ErrorClass::SchemaError, "missing pipeline sequence should be schema_error");

      require_mpk_error(
          [&]() {
            (void)MpKLoader::inspect(
                sima_test::mpk_fixture_path("invalid/unsupported_version.mpk").string());
          },
          ErrorClass::UnsupportedVersion,
          "unsupported version fixture should fail with unsupported_version");

      MpKLoaderOptions tiny;
      tiny.max_archive_bytes = 1024ULL * 1024ULL;
      tiny.max_entry_bytes = 512ULL;
      tiny.max_total_json_bytes = 1024ULL;
      tiny.max_entries = 1024;
      require_mpk_error(
          [&]() {
            (void)MpKLoader::inspect(
                sima_test::mpk_fixture_path("invalid/oversized_entry.mpk").string(), tiny);
          },
          ErrorClass::SizeLimitExceeded, "oversized fixture should fail with size_limit_exceeded");
    }));
