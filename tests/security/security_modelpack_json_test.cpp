#include "security/mpk_security_test_utils.h"
#include "test_main.h"

RUN_TEST("security_modelpack_json_test", ([] {
           using namespace simaai::neat::mpk;

           sima_test::mpk_security::require_fixture_root_exists();

           const char* schema_matrix[] = {
               "invalid/schema_error_sequence.mpk",
               "invalid/duplicate_stage_names.mpk",
               "invalid/unknown_kernel.mpk",
               "invalid/invalid_dependency.mpk",
               "invalid/json_huge_numeric_sequence_id.mpk",
           };

           for (const auto* rel : schema_matrix) {
             sima_test::mpk_security::require_fixture_inspect_and_extract_error(
                 rel, ErrorClass::SchemaError);
           }

           {
             MpKLoaderOptions opt;
             opt.max_archive_bytes = 16ULL * 1024ULL * 1024ULL;
             opt.max_entry_bytes = 8ULL * 1024ULL * 1024ULL;
             opt.max_total_json_bytes = 256ULL;
             opt.max_entries = 2048ULL;

             sima_test::mpk_security::require_mpk_error(
                 [&]() {
                   (void)MpKLoader::inspect(
                       sima_test::mpk_fixture_path("valid/basic_valid.mpk").string(), opt);
                 },
                 ErrorClass::SizeLimitExceeded,
                 "JSON size ceiling should produce size_limit_exceeded");
           }
         }));
