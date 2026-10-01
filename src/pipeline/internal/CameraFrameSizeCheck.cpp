#include "pipeline/internal/CameraFrameSizeCheck.h"

#include <gst/video/video.h>

#include <atomic>
#include <sstream>

namespace simaai::neat::pipeline_internal {
namespace {

struct CameraFrameSizeCheckCtx {
  std::atomic<bool> reported{false};
};

void post_frame_size_mismatch(GstElement* element, const std::string& reason) {
  GError* error = g_error_new_literal(GST_STREAM_ERROR, GST_STREAM_ERROR_FAILED,
                                      "Camera frame size differs from the requested size.");
  GstStructure* details =
      gst_structure_new("simaai-neat-error", "neat-schema-version", G_TYPE_UINT, 1U,
                        "neat-diagnostic-id", G_TYPE_STRING, kCameraFrameSizeMismatchDiagnosticId,
                        "neat-reason", G_TYPE_STRING, reason.c_str(), nullptr);
  GstMessage* message =
      gst_message_new_error_with_details(GST_OBJECT(element), error, reason.c_str(), details);
  g_error_free(error);
  gst_element_post_message(element, message);
}

GstPadProbeReturn camera_frame_size_probe_cb(GstPad* pad, GstPadProbeInfo* info,
                                             gpointer user_data) {
  auto* ctx = static_cast<CameraFrameSizeCheckCtx*>(user_data);
  if (ctx->reported.load(std::memory_order_acquire))
    return GST_PAD_PROBE_DROP;

  GstBuffer* buffer = GST_PAD_PROBE_INFO_BUFFER(info);
  if (!buffer)
    return GST_PAD_PROBE_OK;
  // Without GstVideoMeta the frame size comes from the caps: nothing to compare.
  const GstVideoMeta* meta = gst_buffer_get_video_meta(buffer);
  if (!meta)
    return GST_PAD_PROBE_REMOVE;

  GstCaps* caps = gst_pad_get_current_caps(pad);
  GstVideoInfo video_info;
  const bool have_info = caps && gst_video_info_from_caps(&video_info, caps);
  if (caps)
    gst_caps_unref(caps);
  if (!have_info)
    return GST_PAD_PROBE_REMOVE;

  const std::optional<std::string> reason = camera_frame_size_mismatch_reason(
      GST_VIDEO_INFO_WIDTH(&video_info), GST_VIDEO_INFO_HEIGHT(&video_info),
      static_cast<int>(meta->width), static_cast<int>(meta->height));
  if (!reason)
    return GST_PAD_PROBE_REMOVE;

  if (!ctx->reported.exchange(true, std::memory_order_acq_rel)) {
    GstElement* element = gst_pad_get_parent_element(pad);
    if (element) {
      post_frame_size_mismatch(element, *reason);
      gst_object_unref(element);
    }
  }
  return GST_PAD_PROBE_DROP;
}

} // namespace

std::optional<std::string> camera_frame_size_mismatch_reason(int caps_width, int caps_height,
                                                             int frame_width, int frame_height) {
  if (caps_width == frame_width && caps_height == frame_height)
    return std::nullopt;
  std::ostringstream reason;
  reason << "Camera frames arrived at " << frame_width << " by " << frame_height << " but "
         << caps_width << " by " << caps_height
         << " was requested. The sensor likely has no mode for this size; request a size "
            "that both the sensor and the ISP support.";
  return reason.str();
}

void attach_camera_frame_size_check(GstElement* element) {
  if (!element)
    return;
  GstPad* src = gst_element_get_static_pad(element, "src");
  if (!src)
    return;
  gst_pad_add_probe(
      src, GST_PAD_PROBE_TYPE_BUFFER, camera_frame_size_probe_cb, new CameraFrameSizeCheckCtx(),
      +[](gpointer p) { delete static_cast<CameraFrameSizeCheckCtx*>(p); });
  gst_object_unref(src);
}

} // namespace simaai::neat::pipeline_internal
