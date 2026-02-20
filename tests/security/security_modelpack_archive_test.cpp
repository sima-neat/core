#include "security/mpk_security_test_utils.h"
#include "test_main.h"

RUN_TEST("security_modelpack_archive_test", ([] {
           using namespace simaai::neat::mpk;

           sima_test::mpk_security::require_fixture_root_exists();

           struct Case {
             const char* rel;
             ErrorClass code;
           };

           const Case matrix[] = {
               {"invalid/path_traversal.mpk", ErrorClass::PathTraversal},
               {"invalid/absolute_path.mpk", ErrorClass::PathTraversal},
               {"invalid/mixed_separator_traversal.mpk", ErrorClass::PathTraversal},
               {"invalid/symlink_escape.mpk", ErrorClass::PathTraversal},
               {"invalid/hardlink_escape.mpk", ErrorClass::PathTraversal},
               {"invalid/truncated_archive.mpk", ErrorClass::InvalidArchive},
               {"invalid/missing_pipeline_sequence.mpk", ErrorClass::SchemaError},
               {"invalid/unsupported_version.mpk", ErrorClass::UnsupportedVersion},
           };

           for (const auto& tc : matrix) {
             sima_test::mpk_security::require_fixture_inspect_and_extract_error(tc.rel, tc.code);
           }

           MpKLoaderOptions tiny;
           tiny.max_archive_bytes = 8ULL * 1024ULL * 1024ULL;
           tiny.max_entry_bytes = 128ULL;
           tiny.max_total_json_bytes = 4ULL * 1024ULL;
           tiny.max_entries = 2048;

           sima_test::mpk_security::require_fixture_inspect_and_extract_error(
               "invalid/oversized_entry.mpk", ErrorClass::SizeLimitExceeded, tiny);
         }));
