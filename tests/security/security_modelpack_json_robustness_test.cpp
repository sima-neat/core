#include "security/mpk_security_test_utils.h"
#include "test_main.h"

RUN_TEST("security_modelpack_json_robustness_test", ([] {
           using namespace simaai::neat::mpk;

           sima_test::mpk_security::require_fixture_root_exists();

           const char* schema_matrix[] = {
               "invalid/json_deep_nesting_256.mpk",
               "invalid/json_duplicate_keys_top.mpk",
               "invalid/json_duplicate_keys_nested.mpk",
               "invalid/json_huge_numeric_sequence_id.mpk",
           };

           for (const auto* rel : schema_matrix) {
             sima_test::mpk_security::require_fixture_inspect_and_extract_error(
                 rel, ErrorClass::SchemaError);
           }
         }));
