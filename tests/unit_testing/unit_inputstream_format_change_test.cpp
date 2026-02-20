#include "inputstream_test_utils.h"
#include "test_main.h"

RUN_TEST("unit_inputstream_format_change_test", [] {
  sima_test::InputstreamFormatChangeSpec spec;
  spec.width = 16;
  spec.height = 16;
  spec.rgb_fill = 0x11;
  spec.expect_throw = false;
  sima_test::run_inputstream_format_change_test(spec);
});
