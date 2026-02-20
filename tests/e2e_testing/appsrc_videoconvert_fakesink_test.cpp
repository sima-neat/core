#include "gst_test_utils.h"
#include "test_main.h"

RUN_TEST("appsrc_videoconvert_fakesink_test",
         [] { sima_test::run_appsrc_fakesink_test("videoconvert", 640, 360, 1280, 720); });
