#ifndef SIMA_NEAT_INTERNAL
#define SIMA_NEAT_INTERNAL 1
#endif

// Regression for core #881: CameraInput must fail when frames arrive at another size.

#include "gst/GstInit.h"
#include "nodes/io/CameraInput.h"
#include "pipeline/ErrorCodes.h"
#include "pipeline/graph/internal/GraphBuildInternal.h"
#include "pipeline/internal/CameraFrameSizeCheck.h"
#include "pipeline/internal/GstErrorNormalizer.h"
#include "test_main.h"
#include "test_utils.h"

#include <gst/app/gstappsrc.h>
#include <gst/gst.h>
#include <gst/video/video.h>

#include <atomic>
#include <memory>
#include <optional>
#include <string>
#include <vector>

namespace {

using namespace simaai::neat;
using pipeline_internal::camera_frame_size_mismatch_reason;
using pipeline_internal::kCameraFrameSizeMismatchDiagnosticId;

constexpr int kCapsWidth = 2048;
constexpr int kCapsHeight = 1080;

// appsrc stands in for libcamerasrc; the capsfilter has CameraInput's element_names()[1].
GstElement* make_camera_like_pipeline() {
  const std::string launch = "appsrc name=n0_camera_src format=time "
                             "caps=video/x-raw,format=NV12,width=2048,height=1080,framerate=30/1 "
                             "! capsfilter name=n0_camera_caps "
                             "caps=video/x-raw,format=NV12,width=2048,height=1080,framerate=30/1 "
                             "! fakesink name=n0_sink sync=false";
  GError* error = nullptr;
  GstElement* pipeline = gst_parse_launch(launch.c_str(), &error);
  if (error) {
    const std::string message = error->message ? error->message : "unknown";
    g_error_free(error);
    throw std::runtime_error("gst_parse_launch failed: " + message);
  }
  return pipeline;
}

GstPadProbeReturn count_buffers_cb(GstPad*, GstPadProbeInfo*, gpointer user_data) {
  static_cast<std::atomic<int>*>(user_data)->fetch_add(1);
  return GST_PAD_PROBE_OK;
}

struct PushResult {
  std::optional<pipeline_internal::NormalizedDiagnostic> error;
  int buffers_reaching_sink = 0;
};

PushResult push_frames(GstElement* pipeline, int count, std::optional<int> meta_width,
                       std::optional<int> meta_height) {
  std::atomic<int> sink_buffers{0};
  GstElement* sink = gst_bin_get_by_name(GST_BIN(pipeline), "n0_sink");
  GstPad* sink_pad = gst_element_get_static_pad(sink, "sink");
  gst_pad_add_probe(sink_pad, GST_PAD_PROBE_TYPE_BUFFER, count_buffers_cb, &sink_buffers, nullptr);
  gst_object_unref(sink_pad);
  gst_object_unref(sink);

  require(gst_element_set_state(pipeline, GST_STATE_PLAYING) != GST_STATE_CHANGE_FAILURE,
          "pipeline must reach PLAYING");

  GstElement* src = gst_bin_get_by_name(GST_BIN(pipeline), "n0_camera_src");
  const gsize size = static_cast<gsize>(kCapsWidth) * kCapsHeight * 3 / 2;
  for (int i = 0; i < count; ++i) {
    GstBuffer* buffer = gst_buffer_new_allocate(nullptr, size, nullptr);
    GST_BUFFER_PTS(buffer) = static_cast<GstClockTime>(i) * GST_SECOND / 30;
    if (meta_width && meta_height) {
      gst_buffer_add_video_meta(buffer, GST_VIDEO_FRAME_FLAG_NONE, GST_VIDEO_FORMAT_NV12,
                                static_cast<guint>(*meta_width), static_cast<guint>(*meta_height));
    }
    gst_app_src_push_buffer(GST_APP_SRC(src), buffer);
  }
  gst_app_src_end_of_stream(GST_APP_SRC(src));
  gst_object_unref(src);

  PushResult result;
  GstBus* bus = gst_element_get_bus(pipeline);
  GstMessage* message = gst_bus_timed_pop_filtered(
      bus, 5 * GST_SECOND, static_cast<GstMessageType>(GST_MESSAGE_ERROR | GST_MESSAGE_EOS));
  require(message != nullptr, "pipeline must post ERROR or EOS within 5 s");
  if (GST_MESSAGE_TYPE(message) == GST_MESSAGE_ERROR) {
    result.error =
        pipeline_internal::classify_gst_error(pipeline_internal::parse_gst_error_message(message));
  }
  gst_message_unref(message);
  gst_object_unref(bus);

  gst_element_set_state(pipeline, GST_STATE_NULL);
  result.buffers_reaching_sink = sink_buffers.load();
  return result;
}

PushResult run_case(const std::shared_ptr<Node>& node, int count, std::optional<int> meta_width,
                    std::optional<int> meta_height) {
  GstElement* pipeline = make_camera_like_pipeline();
  session_build_attach_camera_frame_size_checks(pipeline, {node}, NameTransform{});
  PushResult result = push_frames(pipeline, count, meta_width, meta_height);
  gst_object_unref(pipeline);
  return result;
}

void test_reason_formatting() {
  require(!camera_frame_size_mismatch_reason(2048, 1080, 2048, 1080).has_value(),
          "matching sizes must not report a mismatch");
  require(camera_frame_size_mismatch_reason(1920, 1080, 1080, 1080).has_value(),
          "a height-only mismatch must report a reason");
  const std::optional<std::string> reason =
      camera_frame_size_mismatch_reason(2048, 1080, 1920, 1080);
  require(reason && reason->find("1920 by 1080") != std::string::npos &&
              reason->find("2048 by 1080") != std::string::npos,
          "reason must name the delivered and requested sizes");
  // The normalizer drops reasons containing "0x" or '=' or longer than 240 characters.
  require(reason->find("0x") == std::string::npos && reason->find('=') == std::string::npos &&
              reason->size() <= 240,
          "reason must survive the diagnostic normalizer: " + *reason);
}

void test_mismatch_fails() {
  CameraInputOptions opt;
  opt.width = kCapsWidth;
  opt.height = kCapsHeight;
  // The reporter's case (2048x1080 requested, 1920x1080 delivered), for both CameraInput kinds.
  for (const auto& node : {nodes::CameraInput(opt), nodes::CameraInputWithCaptureBuffers(opt, 8)}) {
    const PushResult result = run_case(node, 3, 1920, 1080);
    require(result.error && result.error->diagnostic_id == kCameraFrameSizeMismatchDiagnosticId,
            "a size mismatch must post the frame size diagnostic");
    require(result.error->error_code == error_codes::kRuntimeElementFailed,
            "unexpected error code: " + result.error->error_code);
    bool reason_fact = false;
    for (const auto& fact : result.error->facts)
      reason_fact |= fact.label == "Reason" && fact.value.find("1920 by 1080") != std::string::npos;
    require(reason_fact, "the rendered diagnostic must carry the size reason");
    require(result.buffers_reaching_sink == 0, "mismatched frames must not reach downstream");
  }
}

void test_frames_pass_without_mismatch() {
  CameraInputOptions opt;
  opt.width = kCapsWidth;
  opt.height = kCapsHeight;
  // Matching GstVideoMeta, and no GstVideoMeta at all (nothing to compare).
  for (const std::optional<int> meta : {std::optional<int>(kCapsWidth), std::optional<int>()}) {
    const std::optional<int> meta_height = meta ? std::optional<int>(kCapsHeight) : std::nullopt;
    const PushResult result = run_case(nodes::CameraInput(opt), 3, meta, meta_height);
    require(!result.error, "frames without a size mismatch must not post an error");
    require(result.buffers_reaching_sink == 3, "frames without a size mismatch must all pass");
  }
}

} // namespace

RUN_TEST("unit_camera_frame_size_check_test", [] {
  simaai::neat::gst_init_once();
  test_reason_formatting();
  test_mismatch_fails();
  test_frames_pass_without_mismatch();
})
