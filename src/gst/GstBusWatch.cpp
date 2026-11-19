#include "gst/GstBusWatch.h"

#include "pipeline/internal/GstDiagnosticsUtil.h"

namespace simaai::neat {

std::string gst_message_to_string(GstMessage* msg) {
  return pipeline_internal::gst_message_to_string(msg);
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
