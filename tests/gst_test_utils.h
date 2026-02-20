#pragma once

#include "gst/GstHelpers.h"
#include "gst/GstInit.h"
#include "test_utils.h"

#include <gst/gst.h>
#include <gst/app/gstappsrc.h>
#include <gst/video/video.h>

#include <algorithm>
#include <chrono>
#include <cstring>
#include <string>
#include <thread>

namespace sima_test {

struct ProbeState {
  int caps_events = 0;
  int expected_w = 0;
  int expected_h = 0;
  std::size_t expected_bytes = 0;
  bool saw_buffer = false;
  std::string error;
};

inline GstBuffer* make_nv12_buffer(int w, int h, uint8_t value) {
  const std::size_t y_size = static_cast<std::size_t>(w * h);
  const std::size_t uv_size = static_cast<std::size_t>(w * h / 2);
  const std::size_t total = y_size + uv_size;
  GstBuffer* buf = gst_buffer_new_allocate(nullptr, total, nullptr);
  if (!buf)
    return nullptr;

  GstMapInfo map{};
  if (gst_buffer_map(buf, &map, GST_MAP_WRITE)) {
    std::memset(map.data, value, map.size);
    gst_buffer_unmap(buf, &map);
  }

  gsize offsets[GST_VIDEO_MAX_PLANES] = {0};
  gint strides[GST_VIDEO_MAX_PLANES] = {0};
  offsets[0] = 0;
  offsets[1] = y_size;
  strides[0] = w;
  strides[1] = w;

  GstVideoMeta* meta = gst_buffer_add_video_meta_full(buf, GST_VIDEO_FRAME_FLAG_NONE,
                                                      GST_VIDEO_FORMAT_NV12, static_cast<guint>(w),
                                                      static_cast<guint>(h), 2, offsets, strides);
  if (!meta) {
    gst_buffer_unref(buf);
    return nullptr;
  }

  return buf;
}

inline GstPadProbeReturn probe_cb(GstPad* /*pad*/, GstPadProbeInfo* info, gpointer user_data) {
  auto* st = static_cast<ProbeState*>(user_data);
  if (!st)
    return GST_PAD_PROBE_OK;

  if (GST_PAD_PROBE_INFO_TYPE(info) & GST_PAD_PROBE_TYPE_EVENT_DOWNSTREAM) {
    GstEvent* ev = gst_pad_probe_info_get_event(info);
    if (ev && GST_EVENT_TYPE(ev) == GST_EVENT_CAPS) {
      st->caps_events += 1;
    }
  }

  if (GST_PAD_PROBE_INFO_TYPE(info) & GST_PAD_PROBE_TYPE_BUFFER) {
    GstBuffer* buf = gst_pad_probe_info_get_buffer(info);
    if (!buf)
      return GST_PAD_PROBE_OK;
    st->saw_buffer = true;
    GstVideoMeta* meta = gst_buffer_get_video_meta(buf);
    if (!meta) {
      st->error = "missing GstVideoMeta";
      return GST_PAD_PROBE_OK;
    }
    if (meta->format != GST_VIDEO_FORMAT_NV12) {
      st->error = "unexpected video format";
      return GST_PAD_PROBE_OK;
    }
    if (meta->width != static_cast<guint>(st->expected_w) ||
        meta->height != static_cast<guint>(st->expected_h)) {
      st->error = "unexpected video dimensions";
      return GST_PAD_PROBE_OK;
    }
    if (meta->n_planes != 2) {
      st->error = "unexpected plane count";
      return GST_PAD_PROBE_OK;
    }
    const std::size_t y_size = static_cast<std::size_t>(st->expected_w * st->expected_h);
    if (meta->offset[0] != 0 || meta->stride[0] != st->expected_w || meta->offset[1] != y_size ||
        meta->stride[1] != st->expected_w) {
      st->error = "unexpected plane layout";
      return GST_PAD_PROBE_OK;
    }
    if (gst_buffer_get_size(buf) != st->expected_bytes) {
      st->error = "unexpected buffer size";
      return GST_PAD_PROBE_OK;
    }
  }

  return GST_PAD_PROBE_OK;
}

inline bool wait_for_bus_eos_or_error(GstElement* pipeline, int timeout_ms, std::string& err) {
  GstBus* bus = gst_element_get_bus(pipeline);
  if (!bus)
    return false;
  GstMessage* msg =
      gst_bus_timed_pop_filtered(bus, static_cast<GstClockTime>(timeout_ms) * GST_MSECOND,
                                 static_cast<GstMessageType>(GST_MESSAGE_ERROR | GST_MESSAGE_EOS));
  bool ok = false;
  if (msg) {
    if (GST_MESSAGE_TYPE(msg) == GST_MESSAGE_ERROR) {
      GError* gerr = nullptr;
      gchar* debug = nullptr;
      gst_message_parse_error(msg, &gerr, &debug);
      if (gerr) {
        err = gerr->message ? gerr->message : "unknown gst error";
        g_error_free(gerr);
      } else {
        err = "unknown gst error";
      }
      if (debug)
        g_free(debug);
      ok = false;
    } else {
      ok = true;
    }
    gst_message_unref(msg);
  }
  gst_object_unref(bus);
  return ok;
}

inline void run_appsrc_fakesink_test(const char* element, int w1, int h1, int w2, int h2) {
  simaai::neat::gst_init_once();
  require(simaai::neat::element_exists("appsrc"), "GStreamer appsrc element not available");

  const std::string pipeline_desc =
      std::string("appsrc name=mysrc is-live=true format=time do-timestamp=true ! ") + element +
      " ! fakesink";

  GError* err = nullptr;
  GstElement* pipeline = gst_parse_launch(pipeline_desc.c_str(), &err);
  if (!pipeline) {
    std::string msg = "pipeline create failed";
    if (err && err->message)
      msg = err->message;
    if (err)
      g_error_free(err);
    throw std::runtime_error(msg);
  }

  GstElement* appsrc = gst_bin_get_by_name(GST_BIN(pipeline), "mysrc");
  require(appsrc != nullptr, "missing appsrc element");

  GstElement* sink = gst_bin_get_by_name(GST_BIN(pipeline), "fakesink0");
  require(sink != nullptr, "missing fakesink element");

  ProbeState probe;
  GstPad* sinkpad = gst_element_get_static_pad(sink, "sink");
  require(sinkpad != nullptr, "missing fakesink sink pad");
  gst_pad_add_probe(
      sinkpad,
      static_cast<GstPadProbeType>(GST_PAD_PROBE_TYPE_EVENT_DOWNSTREAM | GST_PAD_PROBE_TYPE_BUFFER),
      probe_cb, &probe, nullptr);
  gst_object_unref(sinkpad);

  gst_element_set_state(pipeline, GST_STATE_PLAYING);

  const std::size_t bytes1 = static_cast<std::size_t>(w1 * h1 + (w1 * h1 / 2));
  probe.expected_w = w1;
  probe.expected_h = h1;
  probe.expected_bytes = bytes1;

  GstCaps* caps1 =
      gst_caps_new_simple("video/x-raw", "format", G_TYPE_STRING, "NV12", "width", G_TYPE_INT, w1,
                          "height", G_TYPE_INT, h1, "framerate", GST_TYPE_FRACTION, 30, 1, nullptr);
  gst_app_src_set_caps(GST_APP_SRC(appsrc), caps1);
  gst_caps_unref(caps1);

  GstBuffer* buf1 = make_nv12_buffer(w1, h1, 0x2a);
  require(buf1 != nullptr, "failed to allocate buffer #1");
  require(gst_app_src_push_buffer(GST_APP_SRC(appsrc), buf1) == GST_FLOW_OK, "push buffer failed");

  const std::size_t bytes2 = static_cast<std::size_t>(w2 * h2 + (w2 * h2 / 2));
  GstBuffer* pending = make_nv12_buffer(w2, h2, 0x3b);
  require(pending != nullptr, "failed to allocate buffer #2");

  GstBuffer* pending2 = make_nv12_buffer(w2, h2, 0x3b);
  require(pending2 != nullptr, "failed to allocate buffer #3");

  GstCaps* caps2 =
      gst_caps_new_simple("video/x-raw", "format", G_TYPE_STRING, "NV12", "width", G_TYPE_INT, w2,
                          "height", G_TYPE_INT, h2, "framerate", GST_TYPE_FRACTION, 30, 1, nullptr);
  gst_app_src_set_caps(GST_APP_SRC(appsrc), caps2);
  gst_caps_unref(caps2);

  probe.expected_w = w2;
  probe.expected_h = h2;
  probe.expected_bytes = bytes2;

  gst_app_src_push_buffer(GST_APP_SRC(appsrc), pending);

  require(gst_app_src_push_buffer(GST_APP_SRC(appsrc), pending2) == GST_FLOW_OK,
          "push buffer after caps failed");

  gst_app_src_end_of_stream(GST_APP_SRC(appsrc));

  std::string bus_err;
  require(wait_for_bus_eos_or_error(pipeline, 2000, bus_err), "pipeline error: " + bus_err);
  require(probe.error.empty(), probe.error);
  require(probe.saw_buffer, "never observed a buffer");
  require(probe.caps_events >= 2, "expected at least 2 caps events");

  gst_element_set_state(pipeline, GST_STATE_NULL);
  gst_object_unref(sink);
  gst_object_unref(appsrc);
  gst_object_unref(pipeline);
}

} // namespace sima_test
