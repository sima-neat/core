#include "inputstream_test_utils.h"
#include "test_main.h"

RUN_TEST("inputstream_nv12_format_change_test", [] {
  sima_test::InputstreamFormatChangeSpec spec;
  spec.width = 1280;
  spec.height = 720;
  spec.max_input_bytes = static_cast<size_t>(1280 * 720 * 3);
  spec.expect_throw = false;
  sima_test::run_inputstream_format_change_test(spec);
});
