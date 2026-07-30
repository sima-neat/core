#include "gst/internal/ElementCapability.h"

#include "gst/GstInit.h"

#include <gst/gst.h>

namespace simaai::neat::internal {

std::optional<bool> element_boolean_capability(const char* factory, const char* property) {
  if (!factory || !*factory || !property || !*property) {
    return std::nullopt;
  }

  gst_init_once();
  GstElement* element = gst_element_factory_make(factory, nullptr);
  if (!element) {
    return std::nullopt;
  }

  GParamSpec* spec = g_object_class_find_property(G_OBJECT_GET_CLASS(element), property);
  if (!spec || G_PARAM_SPEC_VALUE_TYPE(spec) != G_TYPE_BOOLEAN ||
      (spec->flags & G_PARAM_READABLE) == 0) {
    gst_object_unref(element);
    return std::nullopt;
  }

  gboolean value = FALSE;
  g_object_get(G_OBJECT(element), property, &value, nullptr);
  gst_object_unref(element);
  return value != FALSE;
}

} // namespace simaai::neat::internal
