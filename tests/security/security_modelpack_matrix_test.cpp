#include "security/model_archive_security_test_utils.h"
#include "test_main.h"

#include <filesystem>
#include <fstream>
#include <string>
#include <vector>

namespace {

using simaai::neat::internal::ModelArchiveErrorClass;
using simaai::neat::internal::ModelArchiveLoader;
using simaai::neat::internal::ModelArchiveLoaderOptions;

void run_security_case(const std::string& id, const std::string& rel_fixture_path,
                       ModelArchiveErrorClass expected,
                       const ModelArchiveLoaderOptions* opt = nullptr) {
  try {
    if (opt != nullptr) {
      sima_test::model_archive_security::require_fixture_inspect_and_extract_error(rel_fixture_path,
                                                                                   expected, *opt);
    } else {
      sima_test::model_archive_security::require_fixture_inspect_and_extract_error(rel_fixture_path,
                                                                                   expected);
    }
  } catch (const std::exception& e) {
    throw std::runtime_error("SEC_CASE=" + id + " " + e.what());
  }
}

void write_empty_file(const std::filesystem::path& path) {
  std::error_code ec;
  std::filesystem::create_directories(path.parent_path(), ec);
  std::ofstream out(path, std::ios::binary);
  out << "not an archive";
}

} // namespace

RUN_TEST("security_modelpack_matrix_test", ([] {
           namespace fs = std::filesystem;
           sima_test::model_archive_security::require_fixture_root_exists();

           struct Case {
             const char* id;
             const char* rel;
             ModelArchiveErrorClass code;
           };

           const std::vector<Case> matrix = {
               {"archive_path_traversal", "invalid/path_traversal.tar.gz",
                ModelArchiveErrorClass::PathTraversal},
               {"archive_absolute_path", "invalid/absolute_path.tar.gz",
                ModelArchiveErrorClass::PathTraversal},
               {"archive_mixed_separator", "invalid/mixed_separator_traversal.tar.gz",
                ModelArchiveErrorClass::PathTraversal},
               {"archive_symlink_escape", "invalid/symlink_escape.tar.gz",
                ModelArchiveErrorClass::PathTraversal},
               {"archive_hardlink_escape", "invalid/hardlink_escape.tar.gz",
                ModelArchiveErrorClass::PathTraversal},
               {"archive_truncated", "invalid/truncated_archive.tar.gz",
                ModelArchiveErrorClass::InvalidArchive},
               {"archive_missing_sequence", "invalid/missing_pipeline_sequence.tar.gz",
                ModelArchiveErrorClass::SchemaError},
               {"archive_unsupported_version", "invalid/unsupported_version.tar.gz",
                ModelArchiveErrorClass::UnsupportedVersion},
               {"json_schema_error_sequence", "invalid/schema_error_sequence.tar.gz",
                ModelArchiveErrorClass::SchemaError},
               {"json_duplicate_stage_names", "invalid/duplicate_stage_names.tar.gz",
                ModelArchiveErrorClass::SchemaError},
               {"json_unknown_kernel", "invalid/unknown_kernel.tar.gz",
                ModelArchiveErrorClass::SchemaError},
               {"json_huge_numeric_sequence_id", "invalid/json_huge_numeric_sequence_id.tar.gz",
                ModelArchiveErrorClass::SchemaError},
               {"json_deep_nesting_256", "invalid/json_deep_nesting_256.tar.gz",
                ModelArchiveErrorClass::SchemaError},
               {"json_duplicate_keys_top", "invalid/json_duplicate_keys_top.tar.gz",
                ModelArchiveErrorClass::SchemaError},
               {"json_duplicate_keys_nested", "invalid/json_duplicate_keys_nested.tar.gz",
                ModelArchiveErrorClass::SchemaError},
               {"tar_duplicate_header_same_path", "invalid/duplicate_header_same_path.tar.gz",
                ModelArchiveErrorClass::InvalidArchive},
               {"tar_duplicate_header_safe_then_traversal",
                "invalid/duplicate_header_safe_then_traversal.tar.gz",
                ModelArchiveErrorClass::InvalidArchive},
               {"tar_checksum_corrupt_header", "invalid/checksum_corrupt_header.tar.gz",
                ModelArchiveErrorClass::InvalidArchive},
               {"tar_checksum_corrupt_body", "invalid/checksum_corrupt_body.tar.gz",
                ModelArchiveErrorClass::InvalidArchive},
               {"unicode_fullwidth_slash", "invalid/unicode_traversal_fullwidth_slash.tar.gz",
                ModelArchiveErrorClass::PathTraversal},
               {"unicode_division_slash", "invalid/unicode_traversal_division_slash.tar.gz",
                ModelArchiveErrorClass::PathTraversal},
               {"unicode_fullwidth_backslash",
                "invalid/unicode_traversal_fullwidth_backslash.tar.gz",
                ModelArchiveErrorClass::PathTraversal},
               {"unicode_confusable_dotdot", "invalid/unicode_confusable_dotdot.tar.gz",
                ModelArchiveErrorClass::PathTraversal},
               {"unicode_non_utf8_path_bytes", "invalid/non_utf8_path_bytes.tar.gz",
                ModelArchiveErrorClass::InvalidArchive},
           };

           for (const auto& tc : matrix) {
             run_security_case(tc.id, tc.rel, tc.code);
           }

           ModelArchiveLoaderOptions tiny;
           tiny.max_archive_bytes = 8ULL * 1024ULL * 1024ULL;
           tiny.max_entry_bytes = 128ULL;
           tiny.max_total_json_bytes = 4ULL * 1024ULL;
           tiny.max_entries = 2048;
           run_security_case("archive_oversized_entry_tiny_limits",
                             "invalid/oversized_entry.tar.gz",
                             ModelArchiveErrorClass::SizeLimitExceeded, &tiny);

           // JSON size ceiling check for inspect path on valid archive.
           {
             ModelArchiveLoaderOptions json_ceiling;
             json_ceiling.max_archive_bytes = 16ULL * 1024ULL * 1024ULL;
             json_ceiling.max_entry_bytes = 8ULL * 1024ULL * 1024ULL;
             json_ceiling.max_total_json_bytes = 256ULL;
             json_ceiling.max_entries = 2048ULL;

             try {
               sima_test::model_archive_security::require_model_archive_error(
                   [&]() {
                     (void)ModelArchiveLoader::inspect(
                         sima_test::model_archive_fixture_path("valid/basic_valid.tar.gz").string(),
                         json_ceiling);
                   },
                   ModelArchiveErrorClass::SizeLimitExceeded,
                   "SEC_CASE=json_size_ceiling_valid_archive_inspect");
             } catch (const std::exception& e) {
               throw std::runtime_error(
                   std::string("SEC_CASE=json_size_ceiling_valid_archive_inspect ") + e.what());
             }
           }

           const fs::path ext_root = fs::path(sima_test::make_temp_dir("model_archive_bad_ext"));
           for (const char* ext : {".mpk", ".tgz", ".tar", ".gz"}) {
             const fs::path bad_path = ext_root / (std::string("bad") + ext);
             write_empty_file(bad_path);
             sima_test::model_archive_security::require_model_archive_error(
                 [&]() { (void)ModelArchiveLoader::inspect(bad_path.string()); },
                 ModelArchiveErrorClass::UnsupportedExtension,
                 std::string("unsupported archive extension ") + ext);
           }
         }));
