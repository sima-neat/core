#pragma once

// What a process holds of GStreamer: whether GStreamer was initialised (gst_is_initialized()) and
// every GLib type of GStreamer's that exists in the process. GStreamer registers its types with the
// GLib type system, each the first time anything makes or names one (gst_init registers its core
// types; a GstAllocator, a GstSample, a GstFraction value or a GstVideoFormat enum made without it
// registers its own), so a process in which no GStreamer object was ever made has none. The walk
// visits every fundamental type and every type derived from one; a GStreamer type is one whose
// name starts with "Gst". Nothing here makes or registers a type: gst_is_initialized reads a flag.
//
// A process on the Dataflow backend holds nothing of GStreamer (DECISIONS.md 31): its footprint is
// empty at any point, the end included.

#include <gst/gst.h>

#include <string>
#include <vector>

namespace sima_test {

struct GStreamerFootprint {
  bool initialized = false;
  std::vector<std::string> types;

  bool empty() const {
    return !initialized && types.empty();
  }

  // "untouched", or what was found: "initialised; types GstObject, GstAllocator, ..."
  std::string text() const {
    if (empty()) {
      return "untouched";
    }
    std::string out = initialized ? "initialised" : "not initialised";
    if (!types.empty()) {
      out += "; " + std::to_string(types.size()) + " GStreamer types:";
      for (const std::string& name : types) {
        out += " " + name;
      }
    }
    return out;
  }
};

namespace gstreamer_footprint_detail {

inline void visit(GType type, std::vector<std::string>& out) {
  const char* name = g_type_name(type);
  if (name && name[0] == 'G' && name[1] == 's' && name[2] == 't') {
    out.emplace_back(name);
  }
  guint count = 0;
  GType* children = g_type_children(type, &count);
  for (guint i = 0; i < count; ++i) {
    visit(children[i], out);
  }
  g_free(children);
}

} // namespace gstreamer_footprint_detail

inline GStreamerFootprint gstreamer_footprint() {
  GStreamerFootprint out;
  out.initialized = gst_is_initialized() != FALSE;
  constexpr GType kLast = G_TYPE_FUNDAMENTAL_MAX >> G_TYPE_FUNDAMENTAL_SHIFT;
  for (GType id = 1; id <= kLast; ++id) {
    const GType fundamental = G_TYPE_MAKE_FUNDAMENTAL(id);
    if (g_type_name(fundamental)) {
      gstreamer_footprint_detail::visit(fundamental, out.types);
    }
  }
  return out;
}

} // namespace sima_test
