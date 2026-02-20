#include "gst_test_utils.h"
#include "test_main.h"

RUN_TEST("appsrc_videoscale_fakesink_test",
         [] { sima_test::run_appsrc_fakesink_test("videoscale", 320, 240, 640, 480); });
