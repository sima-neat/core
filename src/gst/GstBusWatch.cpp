#include "gst/GstBusWatch.h"
#include "gst/internal/HttpDiagnostics.h"

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

void append_http_fields(std::ostringstream& out, const char* uri, const GstStructure* details) {
  if (const auto origin = gst::http_origin(uri); !origin.empty())
    out << " origin=" << origin;
  guint status = 0;
  gint signed_status = 0;
  if (details &&
      (gst_structure_get_uint(details, "http-status-code", &status) ||
       (gst_structure_get_int(details, "http-status-code", &signed_status) && signed_status > 0))) {
    const auto code = signed_status > 0 ? static_cast<guint>(signed_status) : status;
    if (code >= 100 && code <= 599)
      out << " status=" << code;
  }
}

} // namespace

std::string gst::http_origin(const char* location) {
  g_autoptr(GUri) uri = location ? g_uri_parse(location, G_URI_FLAGS_NONE, nullptr) : nullptr;
  const char* host = uri ? g_uri_get_host(uri) : nullptr;
  const char* scheme = uri ? g_uri_get_scheme(uri) : nullptr;
  if (!host || !*host || !scheme ||
      (g_ascii_strcasecmp(scheme, "http") && g_ascii_strcasecmp(scheme, "https")))
    return {};
  g_autofree gchar* origin = g_uri_join(G_URI_FLAGS_NONE, scheme, nullptr, host,
                                        g_uri_get_port(uri), "", nullptr, nullptr);
  return origin;
}

std::string gst_message_to_string(GstMessage* msg) {
  if (!msg)
    return "<null message>";
  std::ostringstream ss;
  const GstMessageType t = GST_MESSAGE_TYPE(msg);
  ss << gst_message_type_get_name(t);

  const GstStructure* st = gst_message_get_structure(msg);
  if (t == GST_MESSAGE_ELEMENT && st && gst_structure_has_name(st, "http-headers")) {
    ss << " http-headers";
    append_http_fields(ss, gst_structure_get_string(st, "uri"), st);
    return ss.str();
  }

  if (t == GST_MESSAGE_ERROR || t == GST_MESSAGE_WARNING || t == GST_MESSAGE_INFO) {
    g_autoptr(GError) error = nullptr;
    g_autofree gchar* debug = nullptr;
    const auto parse = t == GST_MESSAGE_ERROR     ? gst_message_parse_error
                       : t == GST_MESSAGE_WARNING ? gst_message_parse_warning
                                                  : gst_message_parse_info;
    auto* source = GST_MESSAGE_SRC(msg);
    auto* factory = GST_IS_ELEMENT(source) ? gst_element_get_factory(GST_ELEMENT(source)) : nullptr;
    if (factory && g_strcmp0(GST_OBJECT_NAME(factory), "souphttpsrc") == 0) {
      parse(msg, &error, nullptr);
      if (error)
        ss << " " << g_quark_to_string(error->domain) << "/" << error->code;
      const GstStructure* details = nullptr;
      if (t == GST_MESSAGE_ERROR)
        gst_message_parse_error_details(msg, &details);
      g_autofree gchar* location = nullptr;
      g_object_get(source, "location", &location, nullptr);
      append_http_fields(ss, location, details);
    } else {
      parse(msg, &error, &debug);
      ss << ": " << (error ? error->message : "unknown");
      if (debug && *debug)
        ss << " | " << debug;
    }
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
