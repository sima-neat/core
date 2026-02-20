#include "security/mpk_security_test_utils.h"
#include "test_main.h"

RUN_TEST("security_modelpack_unicode_path_test", ([] {
           using namespace simaai::neat::mpk;

           sima_test::mpk_security::require_fixture_root_exists();

           const char* traversal_matrix[] = {
               "invalid/unicode_traversal_fullwidth_slash.mpk",
               "invalid/unicode_traversal_division_slash.mpk",
               "invalid/unicode_traversal_fullwidth_backslash.mpk",
               "invalid/unicode_confusable_dotdot.mpk",
           };

           for (const auto* rel : traversal_matrix) {
             sima_test::mpk_security::require_fixture_inspect_and_extract_error(
                 rel, ErrorClass::PathTraversal);
           }

           sima_test::mpk_security::require_fixture_inspect_and_extract_error(
               "invalid/non_utf8_path_bytes.mpk", ErrorClass::InvalidArchive);
         }));
