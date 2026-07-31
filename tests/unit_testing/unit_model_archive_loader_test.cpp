#include "model/internal/ModelArchiveLoader.h"
#include "model_archive_test_utils.h"
#include "test_main.h"
#include "test_utils.h"

#include <algorithm>
#include <atomic>
#include <filesystem>
#include <fstream>
#include <functional>
#include <cstdint>
#include <limits>
#include <string>
#include <thread>
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

std::string read_file_bytes(const std::filesystem::path& path) {
  std::ifstream in(path, std::ios::binary);
  return std::string((std::istreambuf_iterator<char>(in)), std::istreambuf_iterator<char>());
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

      // A decoder that stops at the first end-of-stream marker truncates this silently rather
      // than failing, so assert equality with the single-member extraction, not just success.
      const fs::path multi_member =
          sima_test::model_archive_fixture_path("valid/multi_member_valid.tar.gz");
      require(fs::exists(multi_member), "missing multi_member_valid fixture; run "
                                        "tests/tools/make_model_archive_fixtures.py");
      const std::string multi_root = sima_test::make_temp_dir("model_archive_loader_multi_member");
      const ModelArchiveExtractResult multi =
          ModelArchiveLoader::extract(multi_member.string(), multi_root);
      const std::vector<std::string> multi_files = extracted_file_set(multi.package_root);
      require(multi_files == extracted_file_set(first.package_root),
              "concatenated-member archive should extract the same file set as basic_valid");
      for (const auto& rel : multi_files) {
        require(read_file_bytes(fs::path(multi.package_root) / rel) ==
                    read_file_bytes(fs::path(first.package_root) / rel),
                "concatenated-member archive content differs for " + rel);
      }

      // An all-zero tail is block padding, which gzip 1.12 decompresses with exit 0. Rejecting it
      // would drop archives the subprocess decoder loaded, so assert it extracts, not merely that
      // it is accepted.
      const fs::path zero_padded =
          sima_test::model_archive_fixture_path("valid/zero_padded_valid.tar.gz");
      require(fs::exists(zero_padded), "missing zero_padded_valid fixture; run "
                                       "tests/tools/make_model_archive_fixtures.py");
      const std::string padded_root = sima_test::make_temp_dir("model_archive_loader_zero_padded");
      const ModelArchiveExtractResult padded =
          ModelArchiveLoader::extract(zero_padded.string(), padded_root);
      const std::vector<std::string> padded_files = extracted_file_set(padded.package_root);
      require(padded_files == extracted_file_set(first.package_root),
              "zero-padded archive should extract the same file set as basic_valid");
      for (const auto& rel : padded_files) {
        require(read_file_bytes(fs::path(padded.package_root) / rel) ==
                    read_file_bytes(fs::path(first.package_root) / rel),
                "zero-padded archive content differs for " + rel);
      }

      // Decoder parity with the `gzip -dc` subprocess this replaced, measured on GNU gzip 1.12:
      // trailing data exits 2 and an empty file exits 1, so all have always been rejected.
      for (const char* rejected :
           {"invalid/trailing_garbage.tar.gz", "invalid/zero_between_members.tar.gz",
            "invalid/empty_archive.tar.gz"}) {
        const fs::path fixture = sima_test::model_archive_fixture_path(rejected);
        require(fs::exists(fixture),
                std::string("missing ") + rejected +
                    " fixture; run tests/tools/make_model_archive_fixtures.py");
        require_model_archive_error([&]() { (void)ModelArchiveLoader::inspect(fixture.string()); },
                                    ModelArchiveErrorClass::InvalidArchive,
                                    std::string(rejected) + " should fail with invalid_archive");
      }

      // One inflation per load is the #653 guarantee. Timing cannot prove it, so assert the count
      // directly.
      {
        const std::uint64_t before = ModelArchiveLoader::inflation_count();
        (void)ModelArchiveLoader::extract(valid.string(),
                                          sima_test::make_temp_dir("model_archive_inflation"));
        require(ModelArchiveLoader::inflation_count() == before + 1,
                "one archive extract should inflate exactly once");
      }

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

      // The reserve must be refused before the write, not noticed after it. The margin has to
      // stay below the 10 KiB the fixture inflates to, so only a per-chunk budget can reject it.
      {
        const std::string private_tmp = sima_test::make_temp_dir("model_archive_loader_reserve");
        const std::string out = sima_test::make_temp_dir("model_archive_loader_reserve_out");
        constexpr std::uint64_t kWritableMargin = 9216ULL;
        const auto available = static_cast<std::uint64_t>(fs::space(private_tmp).available);
        require(available > kWritableMargin, "test needs a temp filesystem with free space");

        ModelArchiveLoaderOptions reserved;
        reserved.min_output_free_bytes = available - kWritableMargin;
        sima_test::ScopedEnvVar tmpdir("TMPDIR", private_tmp);

        // Matching the message pins this to the inflation budget: the extraction-root check
        // reports "extracting" and would otherwise satisfy the assertion.
        std::string reported;
        try {
          (void)ModelArchiveLoader::extract(valid.string(), out, reserved);
        } catch (const ModelArchiveError& e) {
          reported = e.what();
        }
        require_contains(reported, "insufficient free space inflating model archive",
                         "inflating past the free-space reserve should be refused before the "
                         "write, by the per-chunk budget");
        require(fs::is_empty(private_tmp), "a refused inflation should leave no staging copy");
      }

      // A relative TMPDIR still has to survive a callback that changes the working directory:
      // a cwd-relative staging path would break the member reads and strand the staging copy.
      {
        const fs::path tmp_parent = fs::path(sima_test::make_temp_dir("model_archive_loader_rel"));
        const fs::path relative_tmp = tmp_parent / "reltmp";
        const std::string chdir_target = sima_test::make_temp_dir("model_archive_loader_chdir");
        const std::string out = sima_test::make_temp_dir("model_archive_loader_relative_tmp_out");
        fs::create_directories(relative_tmp);

        const fs::path cwd = fs::current_path();
        try {
          fs::current_path(tmp_parent);
          sima_test::ScopedEnvVar tmpdir("TMPDIR", "reltmp");
          const ModelArchiveExtractResult moved = ModelArchiveLoader::extract(
              valid.string(),
              [&](const ModelArchiveManifest&) {
                fs::current_path(chdir_target);
                return out;
              },
              ModelArchiveLoaderOptions{});
          require(fs::exists(fs::path(moved.etc_dir) / "pipeline_sequence.json"),
                  "extraction should survive a callback that changes the working directory");
        } catch (...) {
          fs::current_path(cwd);
          throw;
        }
        fs::current_path(cwd);
        require(fs::is_empty(relative_tmp), "a relative TMPDIR should not strand the staging copy");
      }

      // An unusable TMPDIR must fail the load, not silently stage on the rootfs: the operator
      // chose that filesystem, and /tmp may not have room for the inflated archive.
      {
        const fs::path missing_tmp =
            fs::path(sima_test::make_temp_dir("model_archive_loader_missing")) / "absent";
        const std::string out = sima_test::make_temp_dir("model_archive_loader_missing_out");
        sima_test::ScopedEnvVar tmpdir("TMPDIR", missing_tmp.string());
        require_model_archive_error(
            [&]() { (void)ModelArchiveLoader::extract(valid.string(), out, {}); },
            ModelArchiveErrorClass::OutputStorageUnavailable,
            "a TMPDIR that does not exist should fail the load");
      }

      // Concatenated empty gzip members grow the compressed archive without changing what it
      // inflates to, so a check that treats the compressed size as a lower bound rejects an
      // archive that fits. The reserve leaves far more room than the inflated tar needs.
      {
        const std::string private_tmp = sima_test::make_temp_dir("model_archive_loader_padded");
        const std::string out = sima_test::make_temp_dir("model_archive_loader_padded_out");
        constexpr std::uint64_t kWindowBytes = 4ULL * 1024ULL * 1024ULL;
        static constexpr unsigned char kEmptyGzipMember[] = {
            0x1f, 0x8b, 0x08, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x03,
            0x03, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00};

        const fs::path padded = fs::path(private_tmp) / "padded_valid.tar.gz";
        {
          std::ifstream src(valid, std::ios::binary);
          std::ofstream dst(padded, std::ios::binary);
          dst << src.rdbuf();
          for (int i = 0; i < 250000; ++i) {
            dst.write(reinterpret_cast<const char*>(kEmptyGzipMember), sizeof(kEmptyGzipMember));
          }
          require(dst.good(), "writing the padded archive should succeed");
        }
        const auto padded_bytes = static_cast<std::uint64_t>(fs::file_size(padded));
        require(padded_bytes > kWindowBytes,
                "padding should push the compressed size past the writable window");

        const auto available = static_cast<std::uint64_t>(fs::space(private_tmp).available);
        require(available > kWindowBytes, "test needs a temp filesystem with free space");
        ModelArchiveLoaderOptions reserved;
        reserved.min_output_free_bytes = available - kWindowBytes;

        sima_test::ScopedEnvVar tmpdir("TMPDIR", private_tmp);
        const ModelArchiveExtractResult padded_result =
            ModelArchiveLoader::extract(padded.string(), out, reserved);
        require(fs::exists(fs::path(padded_result.etc_dir) / "pipeline_sequence.json"),
                "an archive whose compressed size exceeds the writable window should still "
                "extract when its inflated size fits");
      }

      // Concurrent loads must not see each other's staging copies. Distinct archives catch
      // contents bleeding between loads; the same archive four ways catches a staging path
      // derived from the archive rather than from mkdtemp.
      {
        const fs::path multi =
            sima_test::model_archive_fixture_path("valid/multi_stage_valid.tar.gz");
        require(fs::exists(multi), "multi_stage_valid fixture should exist");

        auto file_set_of = [](const fs::path& archive, const std::string& root) {
          return extracted_file_set(
              fs::path(ModelArchiveLoader::extract(archive.string(), root, {}).package_root));
        };
        const std::vector<std::string> valid_files =
            file_set_of(valid, sima_test::make_temp_dir("model_archive_loader_ref_valid"));
        const std::vector<std::string> multi_files =
            file_set_of(multi, sima_test::make_temp_dir("model_archive_loader_ref_multi"));
        require(valid_files != multi_files,
                "the two fixtures must differ for this test to detect content bleeding");

        // Roots come from the default TMPDIR, before the staging override below: make_temp_dir
        // honours TMPDIR, so allocating them later would put them under the directory this
        // asserts is free of staging copies.
        constexpr std::size_t kLoads = 4;
        std::vector<std::string> roots;
        roots.reserve(kLoads);
        for (std::size_t i = 0; i < kLoads; ++i) {
          roots.push_back(sima_test::make_temp_dir("model_archive_loader_parallel_out"));
        }

        const std::string private_tmp = sima_test::make_temp_dir("model_archive_loader_parallel");
        sima_test::ScopedEnvVar tmpdir("TMPDIR", private_tmp);

        auto run_concurrently = [&](const std::vector<fs::path>& archives,
                                    const std::string& context) {
          std::atomic<std::size_t> ready{0};
          std::vector<std::vector<std::string>> observed(kLoads);
          std::vector<std::string> failures(kLoads);

          std::vector<std::thread> threads;
          threads.reserve(kLoads);
          for (std::size_t i = 0; i < kLoads; ++i) {
            threads.emplace_back([&, i] {
              ready.fetch_add(1);
              while (ready.load() < kLoads) {
                std::this_thread::yield();
              }
              try {
                observed[i] = extracted_file_set(fs::path(
                    ModelArchiveLoader::extract(archives[i].string(), roots[i], {}).package_root));
              } catch (const std::exception& e) {
                failures[i] = e.what();
              }
            });
          }
          for (auto& t : threads) {
            t.join();
          }

          for (std::size_t i = 0; i < kLoads; ++i) {
            require(failures[i].empty(),
                    context + ": load " + std::to_string(i) + " failed: " + failures[i]);
            const std::vector<std::string>& expected =
                archives[i] == valid ? valid_files : multi_files;
            require(observed[i] == expected,
                    context + ": load " + std::to_string(i) +
                        " extracted contents that do not match its own archive");
          }
        };

        run_concurrently({valid, multi, valid, multi}, "distinct archives");
        run_concurrently({valid, valid, valid, valid}, "identical archives");
        require(fs::is_empty(private_tmp),
                "concurrent loads should leave no archive staging directories behind");
      }
    }));
