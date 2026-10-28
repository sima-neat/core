// Toggles whether the GStreamer registry has `neatjpegudpsink`, so a test covers both VideoSender
// MJPEG compositions whichever Scheduler is installed. "Present" registers a stand-in: a fakesink
// with the element's properties, which parses and accepts JPEG frames but sends nothing.
#pragma once

#include "gst/GstHelpers.h"
#include "gst/GstInit.h"
#include "nodes/io/internal/JpegRtpUdpSink.h"
#include "test_utils.h"

#include <gst/gst.h>

namespace jpeg_rtp_udp_sink_fixture {

inline void ignore_property(GObject*, guint, const GValue*, GParamSpec*) {}
inline void default_property(GObject*, guint, GValue*, GParamSpec*) {}

inline void stand_in_class_init(gpointer klass, gpointer) {
  auto* object_class = G_OBJECT_CLASS(klass);
  object_class->set_property = ignore_property;
  object_class->get_property = default_property;
  constexpr auto flags = static_cast<GParamFlags>(G_PARAM_READWRITE | G_PARAM_STATIC_STRINGS);
  g_object_class_install_property(object_class, 1,
                                  g_param_spec_string("host", "", "", "127.0.0.1", flags));
  g_object_class_install_property(object_class, 2,
                                  g_param_spec_int("port", "", "", 0, 65535, 5000, flags));
  g_object_class_install_property(object_class, 3,
                                  g_param_spec_int("pt", "", "", 0, 127, 26, flags));
  g_object_class_install_property(object_class, 4,
                                  g_param_spec_int("mtu", "", "", 28, G_MAXINT, 1400, flags));
  g_object_class_install_property(
      object_class, 5, g_param_spec_uint("timestamp-offset", "", "", 0, G_MAXUINT, 0, flags));
}

inline GType stand_in_type() {
  static const GType type = [] {
    GstElementFactory* fakesink = gst_element_factory_find("fakesink");
    require(fakesink != nullptr, "fakesink is required for the neatjpegudpsink stand-in");
    GstPluginFeature* loaded = gst_plugin_feature_load(GST_PLUGIN_FEATURE(fakesink));
    require(loaded != nullptr, "failed to load fakesink");
    const GType parent = gst_element_factory_get_element_type(GST_ELEMENT_FACTORY(loaded));
    gst_object_unref(loaded);
    gst_object_unref(fakesink);
    GTypeQuery query{};
    g_type_query(parent, &query);
    return g_type_register_static_simple(parent, "CoreTestNeatJpegUdpSink", query.class_size,
                                         stand_in_class_init, query.instance_size, nullptr,
                                         static_cast<GTypeFlags>(0));
  }();
  return type;
}

inline void set_factory_present(bool present) {
  const char* factory = simaai::neat::internal::kJpegRtpUdpSinkFactory;
  simaai::neat::gst_init_once();
  GstRegistry* registry = gst_registry_get();
  if (GstPluginFeature* existing = gst_registry_lookup_feature(registry, factory)) {
    gst_registry_remove_feature(registry, existing);
    gst_object_unref(existing);
  }
  if (present) {
    require(gst_element_register(nullptr, factory, GST_RANK_NONE, stand_in_type()),
            "failed to register the neatjpegudpsink stand-in");
  }
  require(simaai::neat::element_exists(factory) == present,
          "neatjpegudpsink registry state did not follow the fixture");
}

} // namespace jpeg_rtp_udp_sink_fixture
