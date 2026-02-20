#include "gst/GstHelpers.h"
#include "gst/GstInit.h"
#include "pipeline/internal/GstDiagnosticsUtil.h"

#include "test_utils.h"

#include <gst/app/gstappsink.h>
#include <gst/gst.h>

#include <iostream>
#include <memory>
#include <stdexcept>

int main() {
  try {
    simaai::neat::gst_init_once();
    simaai::neat::gst_init_once();

    require(simaai::neat::element_exists("identity"), "identity element missing");
    simaai::neat::require_element("identity", "unit_gst_helpers_test");

    const char* desc =
        "videotestsrc num-buffers=1 ! video/x-raw,format=NV12,width=16,height=16,framerate=30/1 ! "
        "appsink name=mysink emit-signals=false sync=false max-buffers=1 drop=true";
    GError* err = nullptr;
    GstElement* pipeline = gst_parse_launch(desc, &err);
    require(pipeline != nullptr, "gst_parse_launch failed");
    if (err)
      g_error_free(err);

    GstElement* sink = gst_bin_get_by_name(GST_BIN(pipeline), "mysink");
    require(sink != nullptr, "appsink not found");

    gst_element_set_state(pipeline, GST_STATE_PLAYING);

    GstSample* sample = gst_app_sink_try_pull_sample(GST_APP_SINK(sink), 2 * GST_SECOND);
    require(sample != nullptr, "failed to pull sample");

    simaai::neat::pipeline_internal::SampleHolder holder(sample);
    std::string map_err;
    require(simaai::neat::pipeline_internal::map_video_frame_read(holder, map_err),
            "map_video_frame_read failed: " + map_err);

    GstCaps* caps = gst_sample_get_caps(sample);
    auto caps_str = simaai::neat::pipeline_internal::gst_caps_to_string_safe(caps);
    require(!caps_str.empty(), "caps string empty");

    gst_sample_unref(sample);
    gst_object_unref(sink);
    gst_element_set_state(pipeline, GST_STATE_NULL);
    gst_object_unref(pipeline);

    std::cout << "[OK] unit_gst_helpers_test passed\n";
    return 0;
  } catch (const std::exception& e) {
    std::cerr << "[FAIL] " << e.what() << "\n";
    return 1;
  }
}
