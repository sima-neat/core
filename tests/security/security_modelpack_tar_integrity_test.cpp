#include "security/mpk_security_test_utils.h"
#include "test_main.h"

RUN_TEST("security_modelpack_tar_integrity_test", ([] {
           using namespace simaai::neat::mpk;

           sima_test::mpk_security::require_fixture_root_exists();

           const char* invalid_archive_matrix[] = {
               "invalid/duplicate_header_same_path.mpk",
               "invalid/duplicate_header_safe_then_traversal.mpk",
               "invalid/checksum_corrupt_header.mpk",
               "invalid/checksum_corrupt_body.mpk",
           };

           for (const auto* rel : invalid_archive_matrix) {
             sima_test::mpk_security::require_fixture_inspect_and_extract_error(
                 rel, ErrorClass::InvalidArchive);
           }
         }));
