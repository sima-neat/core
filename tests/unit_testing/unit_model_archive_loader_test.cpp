#include "model/internal/ModelArchiveLoader.h"
#include "model_archive_test_utils.h"
#include "test_main.h"
#include "test_utils.h"

#include <filesystem>
#include <fstream>
#include <functional>
#include <cstdint>
#include <limits>
#include <string>

namespace {

using simaai::neat::internal::model_archive_error_class_name;
using simaai::neat::internal::ModelArchiveError;
using simaai::neat::internal::ModelArchiveErrorClass;
using simaai::neat::internal::ModelArchiveExtractResult;
using simaai::neat::internal::ModelArchiveLoader;
using simaai::neat::internal::ModelArchiveLoaderOptions;
using simaai::neat::internal::ModelArchiveManifest;

void require_model_archive_error(const std::function<void()>& fn,
                                 ModelArchiveErrorClass expected_code, const std::string& context) {
  try {
    fn();
  } catch (const ModelArchiveError& e) {
    require(e.code() == expected_code,
            context + ": unexpected error class " + model_archive_error_class_name(e.code()));
    return;
  }
  throw std::runtime_error(context + ": expected ModelArchiveError was not thrown");
}

void write_empty_file(const std::filesystem::path& path) {
  std::error_code ec;
  std::filesystem::create_directories(path.parent_path(), ec);
  std::ofstream out(path, std::ios::binary);
  out << "not an archive";
}

} // namespace

RUN_TEST(
    "unit_model_archive_loader_test", ([] {
      namespace fs = std::filesystem;

      const fs::path valid = sima_test::model_archive_fixture_path("valid/basic_valid.tar.gz");
      require(fs::exists(valid),
              "missing model archive fixtures; run tests/tools/make_model_archive_fixtures.py");

      const ModelArchiveManifest manifest = ModelArchiveLoader::inspect(valid.string());
      require(manifest.has_pipeline_sequence,
              "ModelArchiveLoader::inspect should detect pipeline_sequence.json");
      require(manifest.has_model_binary,
              "ModelArchiveLoader::inspect should detect model binary payload");
      require(!manifest.entries.empty(),
              "ModelArchiveLoader::inspect should expose archive entries");

      const std::string out_root = sima_test::make_temp_dir("model_archive_loader_extract");
      const ModelArchiveExtractResult first = ModelArchiveLoader::extract(valid.string(), out_root);

      require(fs::exists(first.package_root), "ModelArchiveLoader::extract package_root missing");
      require(fs::exists(first.etc_dir), "ModelArchiveLoader::extract etc_dir missing");
      require(fs::exists(first.lib_dir), "ModelArchiveLoader::extract lib_dir missing");
      require(fs::exists(first.share_dir), "ModelArchiveLoader::extract share_dir missing");
      require(fs::exists(fs::path(first.etc_dir) / "pipeline_sequence.json"),
              "ModelArchiveLoader::extract should materialize pipeline_sequence.json");

      const ModelArchiveExtractResult second =
          ModelArchiveLoader::extract(valid.string(), out_root);
      require(second.package_root == first.package_root,
              "ModelArchiveLoader::extract should be deterministic for same archive/root");

      const std::string low_space_root = sima_test::make_temp_dir("model_archive_loader_low_space");
      std::error_code space_ec;
      const auto space = fs::space(low_space_root, space_ec);
      if (!space_ec && space.available < static_cast<std::uintmax_t>(
                                             std::numeric_limits<std::uint64_t>::max() - 1024ULL)) {
        ModelArchiveLoaderOptions low_space;
        low_space.min_output_free_bytes = static_cast<std::uint64_t>(space.available) + 1024ULL;
        require_model_archive_error(
            [&]() { (void)ModelArchiveLoader::extract(valid.string(), low_space_root, low_space); },
            ModelArchiveErrorClass::OutputStorageUnavailable,
            "insufficient extraction space should fail with output_storage_unavailable");
      }

      require_model_archive_error(
          [&]() {
            (void)ModelArchiveLoader::inspect(
                sima_test::model_archive_fixture_path("invalid/missing_pipeline_sequence.tar.gz")
                    .string());
          },
          ModelArchiveErrorClass::SchemaError, "missing pipeline sequence should be schema_error");

      require_model_archive_error(
          [&]() {
            (void)ModelArchiveLoader::inspect(
                sima_test::model_archive_fixture_path("invalid/unsupported_version.tar.gz")
                    .string());
          },
          ModelArchiveErrorClass::UnsupportedVersion,
          "unsupported version fixture should fail with unsupported_version");

      ModelArchiveLoaderOptions tiny;
      tiny.max_archive_bytes = 1024ULL * 1024ULL;
      tiny.max_entry_bytes = 512ULL;
      tiny.max_total_json_bytes = 1024ULL;
      tiny.max_entries = 1024;
      require_model_archive_error(
          [&]() {
            (void)ModelArchiveLoader::inspect(
                sima_test::model_archive_fixture_path("invalid/oversized_entry.tar.gz").string(),
                tiny);
          },
          ModelArchiveErrorClass::SizeLimitExceeded,
          "oversized fixture should fail with size_limit_exceeded");

      const fs::path ext_root = fs::path(sima_test::make_temp_dir("model_archive_loader_ext"));
      for (const char* ext : {".mpk", ".tgz", ".tar", ".gz"}) {
        const fs::path bad_path = ext_root / (std::string("bad") + ext);
        write_empty_file(bad_path);
        require_model_archive_error([&]() { (void)ModelArchiveLoader::inspect(bad_path.string()); },
                                    ModelArchiveErrorClass::UnsupportedExtension,
                                    std::string("unsupported archive extension ") + ext);
      }

      const fs::path collision =
          sima_test::model_archive_fixture_path("valid/destination_collision.tar.gz");
      require(fs::exists(collision),
              "missing destination_collision fixture; run "
              "tests/tools/make_model_archive_fixtures.py");

      // Default (warn-only): the colliding archive is accepted and extracts; the later
      // entry overwrites the earlier at the shared destination (etc/collide.json).
      const std::string warn_root = sima_test::make_temp_dir("model_archive_collision_warn");
      const ModelArchiveExtractResult warned =
          ModelArchiveLoader::extract(collision.string(), warn_root);
      require(fs::exists(fs::path(warned.etc_dir) / "collide.json"),
              "warn-mode collision extract should still materialize the shared destination");

      // Opt-in hard reject: the same archive fails with invalid_archive before extraction.
      ModelArchiveLoaderOptions strict;
      strict.reject_destination_collisions = true;
      require_model_archive_error(
          [&]() { (void)ModelArchiveLoader::inspect(collision.string(), strict); },
          ModelArchiveErrorClass::InvalidArchive,
          "destination collision should fail with invalid_archive when reject flag is set");
    }));
