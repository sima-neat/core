#include "nodes/common/MultipartJpegDemux.h"
#include "nodes/common/internal/MultipartHeaderCapture.h"
#include "test_main.h"
#include <stdexcept>
#include <string>
#include <vector>
// Public Core options and validation stay in Core. Parser byte-stream tests live in Runtime.
static_assert(simaai::neat::kMultipartHeaderCaptureMaxHeaders ==
              simaai::neat::multipart_internal::kMultipartHeaderCaptureMaxHeaders);
static_assert(simaai::neat::kMultipartHeaderCaptureMaxNameBytes ==
              simaai::neat::multipart_internal::kMultipartHeaderCaptureMaxNameBytes);
static_assert(simaai::neat::kMultipartHeaderCaptureMaxLineBytes ==
              simaai::neat::multipart_internal::kMultipartHeaderCaptureMaxLineBytes);
static_assert(simaai::neat::kMultipartHeaderCaptureMaxBlockBytes ==
              simaai::neat::multipart_internal::kMultipartHeaderCaptureMaxBlockBytes);
namespace {
void test_allowlist_normalization() {
  using simaai::neat::MultipartHeaderCaptureOptions;
  using simaai::neat::normalize_multipart_header_capture;

  MultipartHeaderCaptureOptions opt;
  opt.headers = {"Image-Index", "image-index", "  Image-Time  ", "IMAGE-TIME"};
  const std::vector<std::string> names = normalize_multipart_header_capture(opt);
  require(names.size() == 2U, "duplicates must collapse after normalization");
  require(names[0] == "image-index", "normalized names must be lowercase and sorted");
  require(names[1] == "image-time", "normalized names must be lowercase and sorted");

  require(normalize_multipart_header_capture(MultipartHeaderCaptureOptions{}).empty(),
          "empty allowlist must stay empty");

  bool threw = false;
  try {
    MultipartHeaderCaptureOptions bad;
    bad.headers = {"Bad Name"};
    (void)normalize_multipart_header_capture(bad);
  } catch (const std::invalid_argument&) {
    threw = true;
  }
  require(threw, "non-token configured name must throw");

  threw = false;
  try {
    MultipartHeaderCaptureOptions bad;
    bad.headers = {std::string(simaai::neat::kMultipartHeaderCaptureMaxNameBytes + 1U, 'a')};
    (void)normalize_multipart_header_capture(bad);
  } catch (const std::invalid_argument&) {
    threw = true;
  }
  require(threw, "over-long configured name must throw");

  threw = false;
  try {
    MultipartHeaderCaptureOptions bad;
    for (std::size_t i = 0; i <= simaai::neat::kMultipartHeaderCaptureMaxHeaders; ++i) {
      bad.headers.push_back("h" + std::to_string(i));
    }
    (void)normalize_multipart_header_capture(bad);
  } catch (const std::invalid_argument&) {
    threw = true;
  }
  require(threw, "too many configured names must throw");
}

} // namespace
RUN_TEST("unit_multipart_header_capture", [] { test_allowlist_normalization(); })
