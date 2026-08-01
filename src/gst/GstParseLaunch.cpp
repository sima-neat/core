#include "gst/internal/GstParseLaunch.h"

#include <glib-object.h>

#include <utility>

namespace simaai::neat::gst {
namespace {

std::string object_path(GstObject* object) {
  if (!object) {
    return {};
  }
  gchar* raw = gst_object_get_path_string(object);
  std::string out = raw ? raw : "";
  g_free(raw);
  return out;
}

} // namespace

ParsedLaunch::~ParsedLaunch() {
  if (root_) {
    gst_object_unref(root_);
  }
  if (error_) {
    g_error_free(error_);
  }
}

ParsedLaunch::ParsedLaunch(ParsedLaunch&& other) noexcept
    : root_(std::exchange(other.root_, nullptr)), error_(std::exchange(other.error_, nullptr)) {}

ParsedLaunch& ParsedLaunch::operator=(ParsedLaunch&& other) noexcept {
  if (this != &other) {
    if (root_) {
      gst_object_unref(root_);
    }
    if (error_) {
      g_error_free(error_);
    }
    root_ = std::exchange(other.root_, nullptr);
    error_ = std::exchange(other.error_, nullptr);
  }
  return *this;
}

GstElement* ParsedLaunch::release() noexcept {
  return std::exchange(root_, nullptr);
}

ParsedLaunch parse_launch(std::string_view launch, GstParseFlags flags) {
  ParsedLaunch result;
  const std::string launch_string(launch);
  result.root_ = gst_parse_launch_full(launch_string.c_str(), nullptr, flags, &result.error_);
  return result;
}

std::vector<ElementObjectInfo> inventory_elements(GstElement* root) {
  std::vector<ElementObjectInfo> out;
  if (!root) {
    return out;
  }

  auto append = [&](GstElement* element) {
    if (!element) {
      return;
    }
    ElementObjectInfo info;
    const gchar* name = GST_ELEMENT_NAME(element);
    info.short_name = name ? name : "";
    info.object_path = object_path(GST_OBJECT(element));
    GstObject* parent = gst_object_get_parent(GST_OBJECT(element));
    info.parent_path = object_path(parent);
    if (parent) {
      gst_object_unref(parent);
    }
    info.type_name = G_OBJECT_TYPE_NAME(element) ? G_OBJECT_TYPE_NAME(element) : "";
    if (GstElementFactory* factory = gst_element_get_factory(element)) {
      const gchar* factory_name = gst_plugin_feature_get_name(GST_PLUGIN_FEATURE(factory));
      info.factory_name = factory_name ? factory_name : "";
    }
    out.push_back(std::move(info));
  };

  if (!GST_IS_BIN(root)) {
    append(root);
    return out;
  }

  append(root);

  GstIterator* iterator = gst_bin_iterate_recurse(GST_BIN(root));
  if (!iterator) {
    return out;
  }
  GValue value = G_VALUE_INIT;
  bool done = false;
  while (!done) {
    switch (gst_iterator_next(iterator, &value)) {
    case GST_ITERATOR_OK:
      append(GST_ELEMENT(g_value_get_object(&value)));
      g_value_reset(&value);
      break;
    case GST_ITERATOR_RESYNC:
      gst_iterator_resync(iterator);
      break;
    case GST_ITERATOR_DONE:
    case GST_ITERATOR_ERROR:
      done = true;
      break;
    }
  }
  g_value_unset(&value);
  gst_iterator_free(iterator);
  return out;
}

} // namespace simaai::neat::gst
