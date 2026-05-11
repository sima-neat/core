#include "security/mpk_security_test_utils.h"
#include "test_main.h"

#include <string>
#include <vector>

namespace {

void run_security_case(const std::string& id, const std::string& rel_fixture_path,
                       simaai::neat::mpk::ErrorClass expected,
                       const simaai::neat::mpk::MpKLoaderOptions* opt = nullptr) {
  try {
    if (opt != nullptr) {
      sima_test::mpk_security::require_fixture_inspect_and_extract_error(rel_fixture_path, expected,
                                                                         *opt);
    } else {
      sima_test::mpk_security::require_fixture_inspect_and_extract_error(rel_fixture_path, expected);
    }
  } catch (const std::exception& e) {
    throw std::runtime_error("SEC_CASE=" + id + " " + e.what());
  }
}

} // namespace

RUN_TEST("security_modelpack_matrix_test", ([] {
           using namespace simaai::neat::mpk;

           sima_test::mpk_security::require_fixture_root_exists();

           struct Case {
             const char* id;
             const char* rel;
             ErrorClass code;
           };

           const std::vector<Case> matrix = {
               {"archive_path_traversal", "invalid/path_traversal.mpk", ErrorClass::PathTraversal},
               {"archive_absolute_path", "invalid/absolute_path.mpk", ErrorClass::PathTraversal},
               {"archive_mixed_separator", "invalid/mixed_separator_traversal.mpk",
                ErrorClass::PathTraversal},
               {"archive_symlink_escape", "invalid/symlink_escape.mpk", ErrorClass::PathTraversal},
               {"archive_hardlink_escape", "invalid/hardlink_escape.mpk", ErrorClass::PathTraversal},
               {"archive_truncated", "invalid/truncated_archive.mpk", ErrorClass::InvalidArchive},
               {"archive_missing_sequence", "invalid/missing_pipeline_sequence.mpk",
                ErrorClass::SchemaError},
               {"archive_unsupported_version", "invalid/unsupported_version.mpk",
                ErrorClass::UnsupportedVersion},
               {"json_schema_error_sequence", "invalid/schema_error_sequence.mpk",
                ErrorClass::SchemaError},
               {"json_duplicate_stage_names", "invalid/duplicate_stage_names.mpk",
                ErrorClass::SchemaError},
               {"json_unknown_kernel", "invalid/unknown_kernel.mpk", ErrorClass::SchemaError},
               {"json_huge_numeric_sequence_id", "invalid/json_huge_numeric_sequence_id.mpk",
                ErrorClass::SchemaError},
               {"json_deep_nesting_256", "invalid/json_deep_nesting_256.mpk",
                ErrorClass::SchemaError},
               {"json_duplicate_keys_top", "invalid/json_duplicate_keys_top.mpk",
                ErrorClass::SchemaError},
               {"json_duplicate_keys_nested", "invalid/json_duplicate_keys_nested.mpk",
                ErrorClass::SchemaError},
               {"tar_duplicate_header_same_path", "invalid/duplicate_header_same_path.mpk",
                ErrorClass::InvalidArchive},
               {"tar_duplicate_header_safe_then_traversal",
                "invalid/duplicate_header_safe_then_traversal.mpk", ErrorClass::InvalidArchive},
               {"tar_checksum_corrupt_header", "invalid/checksum_corrupt_header.mpk",
                ErrorClass::InvalidArchive},
               {"tar_checksum_corrupt_body", "invalid/checksum_corrupt_body.mpk",
                ErrorClass::InvalidArchive},
               {"unicode_fullwidth_slash", "invalid/unicode_traversal_fullwidth_slash.mpk",
                ErrorClass::PathTraversal},
               {"unicode_division_slash", "invalid/unicode_traversal_division_slash.mpk",
                ErrorClass::PathTraversal},
               {"unicode_fullwidth_backslash", "invalid/unicode_traversal_fullwidth_backslash.mpk",
                ErrorClass::PathTraversal},
               {"unicode_confusable_dotdot", "invalid/unicode_confusable_dotdot.mpk",
                ErrorClass::PathTraversal},
               {"unicode_non_utf8_path_bytes", "invalid/non_utf8_path_bytes.mpk",
                ErrorClass::InvalidArchive},
           };

           for (const auto& tc : matrix) {
             run_security_case(tc.id, tc.rel, tc.code);
           }

           MpKLoaderOptions tiny;
           tiny.max_archive_bytes = 8ULL * 1024ULL * 1024ULL;
           tiny.max_entry_bytes = 128ULL;
           tiny.max_total_json_bytes = 4ULL * 1024ULL;
           tiny.max_entries = 2048;
           run_security_case("archive_oversized_entry_tiny_limits", "invalid/oversized_entry.mpk",
                             ErrorClass::SizeLimitExceeded, &tiny);

           // JSON size ceiling check for inspect path on valid archive.
           {
             MpKLoaderOptions json_ceiling;
             json_ceiling.max_archive_bytes = 16ULL * 1024ULL * 1024ULL;
             json_ceiling.max_entry_bytes = 8ULL * 1024ULL * 1024ULL;
             json_ceiling.max_total_json_bytes = 256ULL;
             json_ceiling.max_entries = 2048ULL;

             try {
               sima_test::mpk_security::require_mpk_error(
                   [&]() {
                     (void)MpKLoader::inspect(
                         sima_test::mpk_fixture_path("valid/basic_valid.mpk").string(), json_ceiling);
                   },
                   ErrorClass::SizeLimitExceeded,
                   "SEC_CASE=json_size_ceiling_valid_archive_inspect");
             } catch (const std::exception& e) {
               throw std::runtime_error(
                   std::string("SEC_CASE=json_size_ceiling_valid_archive_inspect ") + e.what());
             }
           }
         }));
