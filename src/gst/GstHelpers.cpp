// src/gst/GstHelpers.cpp
#include "gst/GstHelpers.h"

#include "gst/GstInit.h"

#include <gst/gst.h>

#include <sstream>
#include <stdexcept>

namespace simaai::neat {

bool element_exists(const char* factory) {
  gst_init_once();
  GstElementFactory* f = gst_element_factory_find(factory);
  if (f) {
    gst_object_unref(f);
    return true;
  }
  return false;
}

bool element_property_exists(const char* factory, const char* property_name) {
  if (!factory || !*factory || !property_name || !*property_name) {
    return false;
  }
  gst_init_once();
  GstElementFactory* f = gst_element_factory_find(factory);
  if (!f) {
    return false;
  }
  const GType element_type = gst_element_factory_get_element_type(f);
  bool found = false;
  if (element_type != G_TYPE_INVALID) {
    if (auto* klass = G_OBJECT_CLASS(g_type_class_ref(element_type)); klass) {
      found = g_object_class_find_property(klass, property_name) != nullptr;
      g_type_class_unref(klass);
    }
  }
  gst_object_unref(f);
  return found;
}

std::string factory_plugin_path(const char* factory) {
  gst_init_once();
  GstElementFactory* f = gst_element_factory_find(factory);
  if (!f) {
    return {};
  }
  std::string out;
  GstPlugin* plugin = gst_plugin_feature_get_plugin(GST_PLUGIN_FEATURE(f));
  if (plugin) {
    const gchar* filename = gst_plugin_get_filename(plugin);
    if (filename && *filename) {
      out = filename;
    }
    gst_object_unref(plugin);
  }
  gst_object_unref(f);
  return out;
}

void require_element(const char* factory, const char* context) {
  if (!element_exists(factory)) {
    std::ostringstream ss;
    ss << (context ? context : "<unknown>")
       << ": required GStreamer element not found: " << factory;
    throw std::runtime_error(ss.str());
  }
}

void require_tensordecoder(const char* context) {
  if (!element_exists("neatdecoder")) {
    std::ostringstream ss;
    ss << (context ? context : "<unknown>")
       << ": required GStreamer element not found: neatdecoder. "
       << "Run scripts/sync_tensordecoder.sh and source scripts/use_tensordecoder.sh";
    throw std::runtime_error(ss.str());
  }
}

} // namespace simaai::neat
