#include "gst/GstBusWatch.h"

#include <gst/gst.h>

#include <sstream>
#include <string>

namespace simaai::neat {
namespace {

const char* state_name(GstState s) {
  switch (s) {
  case GST_STATE_VOID_PENDING:
    return "VOID_PENDING";
  case GST_STATE_NULL:
    return "NULL";
  case GST_STATE_READY:
    return "READY";
  case GST_STATE_PAUSED:
    return "PAUSED";
  case GST_STATE_PLAYING:
    return "PLAYING";
  default:
    return "UNKNOWN";
  }
}

std::string gst_structure_to_string_safe(const GstStructure* st) {
  if (!st)
    return "<null structure>";
  gchar* s = gst_structure_to_string(st);
  if (!s)
    return "<structure_to_string failed>";
  std::string out = s;
  g_free(s);
  return out;
}

} // namespace

std::string gst_message_to_string(GstMessage* msg) {
  if (!msg)
    return "<null message>";
  std::ostringstream ss;
  const GstMessageType t = GST_MESSAGE_TYPE(msg);
  ss << gst_message_type_get_name(t);

  if (t == GST_MESSAGE_ERROR) {
    GError* e = nullptr;
    gchar* dbg = nullptr;
    gst_message_parse_error(msg, &e, &dbg);
    ss << ": " << (e ? e->message : "unknown");
    if (dbg && *dbg)
      ss << " | " << dbg;
    if (e)
      g_error_free(e);
    if (dbg)
      g_free(dbg);
    return ss.str();
  }
  if (t == GST_MESSAGE_WARNING) {
    GError* e = nullptr;
    gchar* dbg = nullptr;
    gst_message_parse_warning(msg, &e, &dbg);
    ss << ": " << (e ? e->message : "unknown");
    if (dbg && *dbg)
      ss << " | " << dbg;
    if (e)
      g_error_free(e);
    if (dbg)
      g_free(dbg);
    return ss.str();
  }
  if (t == GST_MESSAGE_INFO) {
    GError* e = nullptr;
    gchar* dbg = nullptr;
    gst_message_parse_info(msg, &e, &dbg);
    ss << ": " << (e ? e->message : "unknown");
    if (dbg && *dbg)
      ss << " | " << dbg;
    if (e)
      g_error_free(e);
    if (dbg)
      g_free(dbg);
    return ss.str();
  }
  if (t == GST_MESSAGE_STATE_CHANGED) {
    if (GST_MESSAGE_SRC(msg) && GST_IS_OBJECT(GST_MESSAGE_SRC(msg))) {
      ss << " src=" << GST_OBJECT_NAME(GST_MESSAGE_SRC(msg));
    }
    GstState old_s, new_s, pend_s;
    gst_message_parse_state_changed(msg, &old_s, &new_s, &pend_s);
    ss << " " << state_name(old_s) << " -> " << state_name(new_s) << " (pending "
       << state_name(pend_s) << ")";
    return ss.str();
  }
  if (t == GST_MESSAGE_EOS) {
    ss << " (EOS)";
    return ss.str();
  }
  if (t == GST_MESSAGE_ASYNC_DONE) {
    ss << " (ASYNC_DONE)";
    return ss.str();
  }
  if (t == GST_MESSAGE_STREAM_START) {
    ss << " (STREAM_START)";
    return ss.str();
  }

  const GstStructure* st = gst_message_get_structure(msg);
  if (st)
    ss << " " << gst_structure_to_string_safe(st);
  return ss.str();
}

void drain_bus(GstElement* pipeline, BusMessageFn on_message, void* user_data) {
  if (!pipeline)
    return;
  GstBus* bus = gst_element_get_bus(pipeline);
  if (!bus)
    return;

  while (GstMessage* msg = gst_bus_pop(bus)) {
    const GstMessageType t = GST_MESSAGE_TYPE(msg);
    const char* src = (GST_MESSAGE_SRC(msg) && GST_IS_OBJECT(GST_MESSAGE_SRC(msg)))
                          ? GST_OBJECT_NAME(GST_MESSAGE_SRC(msg))
                          : "<unknown>";
    std::string line = gst_message_to_string(msg);
    if (on_message)
      on_message(gst_message_type_get_name(t), src ? src : "<unknown>", line, user_data);
    gst_message_unref(msg);
  }
  gst_object_unref(bus);
}

void throw_if_bus_error(GstElement* pipeline, BusMessageFn on_message, void* user_data,
                        BusErrorFn on_error, void* error_user_data) {
  if (!pipeline)
    return;

  GstBus* bus = gst_element_get_bus(pipeline);
  if (!bus)
    return;

  while (GstMessage* msg = gst_bus_pop(bus)) {
    const GstMessageType t = GST_MESSAGE_TYPE(msg);
    const char* src = (GST_MESSAGE_SRC(msg) && GST_IS_OBJECT(GST_MESSAGE_SRC(msg)))
                          ? GST_OBJECT_NAME(GST_MESSAGE_SRC(msg))
                          : "<unknown>";
    std::string line = gst_message_to_string(msg);
    if (on_message)
      on_message(gst_message_type_get_name(t), src ? src : "<unknown>", line, user_data);

    if (t == GST_MESSAGE_ERROR) {
      gst_message_unref(msg);
      gst_object_unref(bus);
      if (on_error)
        on_error(line, error_user_data);
      return;
    }

    gst_message_unref(msg);
  }
  gst_object_unref(bus);
}

} // namespace simaai::neat
