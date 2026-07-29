#include "model/internal/ModelArchiveLoader.h"
#include "model_archive_test_utils.h"
#include "test_main.h"
#include "test_utils.h"

#include <algorithm>
#include <filesystem>
#include <fstream>
#include <functional>
#include <cstdint>
#include <limits>
#include <string>
#include <vector>

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

std::vector<std::string> extracted_file_set(const std::filesystem::path& package_root) {
  namespace fs = std::filesystem;
  std::vector<std::string> files;
  for (const auto& it : fs::recursive_directory_iterator(package_root)) {
    if (!it.is_regular_file())
      continue;
    files.push_back(fs::relative(it.path(), package_root).generic_string());
  }
  std::sort(files.begin(), files.end());
  return files;
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
      ModelArchiveLoaderOptions low_space;
      low_space.min_output_free_bytes = std::numeric_limits<std::uint64_t>::max() / 2ULL;
      require_model_archive_error(
          [&]() { (void)ModelArchiveLoader::extract(valid.string(), low_space_root, low_space); },
          ModelArchiveErrorClass::OutputStorageUnavailable,
          "insufficient extraction space should fail with output_storage_unavailable");

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
      require(fs::exists(collision), "missing destination_collision fixture; run "
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

      // Exactly the entries validation classified as extractable, and nothing else.
      require(extracted_file_set(fs::path(first.package_root)) ==
                  std::vector<std::string>{"etc/0_preproc.json", "etc/0_process_mla.json",
                                           "etc/pipeline_sequence.json", "lib/model.so",
                                           "share/model.elf"},
              "extracted file set should be exactly the archive's classified entries");

      // Baked into the JSON configs, so absolute even when the caller names the root relatively.
      {
        const fs::path cwd = fs::current_path();
        const std::string rel_root = sima_test::make_temp_dir("model_archive_loader_relative");
        fs::current_path(rel_root);
        ModelArchiveExtractResult relative;
        try {
          relative = ModelArchiveLoader::extract(valid.string(), "./nested/root");
        } catch (...) {
          fs::current_path(cwd);
          throw;
        }
        fs::current_path(cwd);
        require(fs::path(relative.package_root).is_absolute(),
                "extract should return an absolute package_root for a relative output root");
        require(fs::exists(fs::path(relative.etc_dir) / "pipeline_sequence.json"),
                "relative-root extraction should still materialize etc contents");
      }

      // The callback overload hands the manifest over so the caller can size the root.
      {
        const std::string callback_root = sima_test::make_temp_dir("model_archive_loader_callback");
        bool root_chosen = false;
        const ModelArchiveExtractResult chosen = ModelArchiveLoader::extract(
            valid.string(),
            [&](const ModelArchiveManifest& m) {
              require(!m.entries.empty(), "root-selection callback should receive the manifest");
              root_chosen = true;
              return callback_root;
            },
            ModelArchiveLoaderOptions{});
        require(root_chosen, "root-selection callback should be invoked");
        require(fs::path(chosen.package_root).parent_path() == fs::path(callback_root),
                "extract should honor the root returned by the callback");
      }

      // A private TMPDIR makes the staging copy observable: none survives any of the three
      // outcomes below.
      {
        const std::string private_tmp = sima_test::make_temp_dir("model_archive_loader_tmpdir");
        const std::string staging_root = sima_test::make_temp_dir("model_archive_loader_staging");
        {
          sima_test::ScopedEnvVar tmpdir("TMPDIR", private_tmp);
          (void)ModelArchiveLoader::extract(valid.string(), staging_root);

          ModelArchiveLoaderOptions bomb;
          bomb.max_inflated_archive_bytes = 64ULL;
          require_model_archive_error(
              [&]() { (void)ModelArchiveLoader::extract(valid.string(), staging_root, bomb); },
              ModelArchiveErrorClass::SizeLimitExceeded,
              "inflated archive over max_inflated_archive_bytes should fail with "
              "size_limit_exceeded");

          bool propagated = false;
          try {
            (void)ModelArchiveLoader::extract(
                valid.string(),
                [](const ModelArchiveManifest&) -> std::string {
                  throw std::runtime_error("no usable extraction root");
                },
                ModelArchiveLoaderOptions{});
          } catch (const std::runtime_error&) {
            propagated = true;
          }
          require(propagated, "a throwing root-selection callback should propagate");
        }
        require(fs::is_empty(private_tmp),
                "loader should leave no archive staging directories behind");
      }
    }));
