#ifndef SIMA_NEAT_INTERNAL
#define SIMA_NEAT_INTERNAL 1
#endif

// Regression coverage for core #881: libcamera can accept a size the sensor has no mode
// for and deliver frames of the sensor's current size, with only GstVideoMeta telling
// the truth. CameraInput graphs must fail loudly instead of passing those frames on.

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

// Stand-in for a CameraInput fragment: appsrc plays libcamerasrc, and the capsfilter
// carries the name the attach helper looks up (element_names()[1]).
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

// Pushes `count` NV12 buffers (with a GstVideoMeta of meta_width x meta_height when
// requested), then EOS, and returns the first bus error, if any.
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

void attach_to_pipeline(GstElement* pipeline, const std::shared_ptr<Node>& node) {
  const std::vector<std::shared_ptr<Node>> nodes{node};
  session_build_attach_camera_frame_size_checks(pipeline, nodes, NameTransform{});
}

bool has_fact(const pipeline_internal::NormalizedDiagnostic& diagnostic, const std::string& label,
              const std::string& needle) {
  for (const auto& fact : diagnostic.facts) {
    if (fact.label == label && fact.value.find(needle) != std::string::npos)
      return true;
  }
  return false;
}

void test_reason_formatting() {
  require(!camera_frame_size_mismatch_reason(2048, 1080, 2048, 1080).has_value(),
          "matching sizes must not report a mismatch");

  const std::optional<std::string> reason =
      camera_frame_size_mismatch_reason(2048, 1080, 1920, 1080);
  require(reason.has_value(), "a width mismatch must report a reason");
  require(reason->find("1920 by 1080") != std::string::npos &&
              reason->find("2048 by 1080") != std::string::npos,
          "reason must name the delivered and requested sizes: " + *reason);
  // The normalizer drops reasons containing "0x" or '=' or longer than 240 characters.
  require(reason->find("0x") == std::string::npos && reason->find('=') == std::string::npos &&
              reason->size() <= 240,
          "reason must survive the diagnostic normalizer: " + *reason);

  require(camera_frame_size_mismatch_reason(1920, 1080, 1080, 1080).has_value(),
          "a height-only mismatch must report a reason");
}

void test_mismatch_fails_with_structured_diagnostic() {
  CameraInputOptions opt;
  opt.width = kCapsWidth;
  opt.height = kCapsHeight;
  GstElement* pipeline = make_camera_like_pipeline();
  attach_to_pipeline(pipeline, nodes::CameraInput(opt));

  // The reporter's case: 2048x1080 requested, 1920x1080 delivered.
  const PushResult result = push_frames(pipeline, 3, 1920, 1080);
  gst_object_unref(pipeline);

  require(result.error.has_value(), "a size mismatch must post a pipeline error");
  const auto& diagnostic = *result.error;
  require(diagnostic.diagnostic_id == kCameraFrameSizeMismatchDiagnosticId,
          "unexpected diagnostic id: " + diagnostic.diagnostic_id);
  require(diagnostic.error_code == error_codes::kRuntimeElementFailed,
          "unexpected error code: " + diagnostic.error_code);
  require(has_fact(diagnostic, "Reason", "1920 by 1080"),
          "the rendered diagnostic must carry the size reason");
  require(result.buffers_reaching_sink == 0, "mismatched frames must not reach downstream");
}

void test_capture_buffer_wrapper_is_checked() {
  CameraInputOptions opt;
  opt.width = kCapsWidth;
  opt.height = kCapsHeight;
  GstElement* pipeline = make_camera_like_pipeline();
  attach_to_pipeline(pipeline, nodes::CameraInputWithCaptureBuffers(opt, 8));

  const PushResult result = push_frames(pipeline, 1, 1920, 1080);
  gst_object_unref(pipeline);

  require(result.error.has_value() &&
              result.error->diagnostic_id == kCameraFrameSizeMismatchDiagnosticId,
          "CameraInputWithCaptureBuffers graphs must get the frame size check too");
}

void test_matching_frames_pass() {
  CameraInputOptions opt;
  opt.width = kCapsWidth;
  opt.height = kCapsHeight;
  GstElement* pipeline = make_camera_like_pipeline();
  attach_to_pipeline(pipeline, nodes::CameraInput(opt));

  const PushResult result = push_frames(pipeline, 3, kCapsWidth, kCapsHeight);
  gst_object_unref(pipeline);

  require(!result.error.has_value(), "matching frames must not post an error");
  require(result.buffers_reaching_sink == 3, "matching frames must all reach downstream");
}

void test_frames_without_video_meta_pass() {
  CameraInputOptions opt;
  opt.width = kCapsWidth;
  opt.height = kCapsHeight;
  GstElement* pipeline = make_camera_like_pipeline();
  attach_to_pipeline(pipeline, nodes::CameraInput(opt));

  const PushResult result = push_frames(pipeline, 2, std::nullopt, std::nullopt);
  gst_object_unref(pipeline);

  require(!result.error.has_value(), "frames without GstVideoMeta have nothing to compare");
  require(result.buffers_reaching_sink == 2, "frames without GstVideoMeta must pass through");
}

} // namespace

RUN_TEST("unit_camera_frame_size_check_test", [] {
  simaai::neat::gst_init_once();
  test_reason_formatting();
  test_mismatch_fails_with_structured_diagnostic();
  test_capture_buffer_wrapper_is_checked();
  test_matching_frames_pass();
  test_frames_without_video_meta_pass();
})
