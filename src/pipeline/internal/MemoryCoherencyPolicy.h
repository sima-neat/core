#pragma once

#include <simaai/runtime/gstreamer/memory_coherency.h>
#include <simaai/runtime/memory/coherency.h>
#include <gst/gst.h>
#include <string_view>

namespace simaai::neat::pipeline_internal {
using MemoryCoherencyMode = simaai::runtime::gstreamer::MemoryCoherencyMode;

// Qualification-only build switch. No environment or public graph API silently
// opts existing callers into a stronger ownership contract.
inline constexpr bool demand_coherency_build_enabled() noexcept {
#if defined(SIMA_CORE_DEMAND_COHERENCY_V1) && SIMA_CORE_DEMAND_COHERENCY_V1
  return true;
#else
  return false;
#endif
}
inline bool demand_coherency_available() noexcept {
  return demand_coherency_build_enabled() &&
         simaai::neat::internal::dmabuf::initializeLocalCoherencyV1();
}
inline GQuark memory_coherency_mode_quark() {
  return g_quark_from_static_string("sima-core-admitted-memory-coherency-v1");
}
inline MemoryCoherencyMode graph_memory_coherency_mode(GstElement* pipeline) {
  return pipeline && g_object_get_qdata(G_OBJECT(pipeline), memory_coherency_mode_quark())
             ? MemoryCoherencyMode::DemandLocalV1
             : MemoryCoherencyMode::Legacy;
}
// These transports retain normal GstMemory ownership and never access DMA
// behind the allocator's guarded map API. Unknown writers remain Legacy.
inline bool participates_in_demand_coherency(std::string_view factory) noexcept {
  return factory == "neatkernelrunner" || factory == "neatprocessmla" || factory == "appsrc" ||
         factory == "appsink" || factory == "queue" || factory == "queue2" ||
         factory == "capsfilter" || factory == "identity" || factory == "tee" ||
         factory == "funnel";
}
// Cold transport check shared by graph admission and regression tests. This
// does not grant platform or allocation capability by itself.
inline bool graph_participates_in_demand_coherency(GstElement* pipeline) {
  if (!pipeline || !GST_IS_BIN(pipeline))
    return false;
  auto* iterator = gst_bin_iterate_recurse(GST_BIN(pipeline));
  GValue value = G_VALUE_INIT;
  bool closed = true;
  for (;;) {
    const auto step = gst_iterator_next(iterator, &value);
    if (step == GST_ITERATOR_DONE)
      break;
    if (step != GST_ITERATOR_OK) {
      closed = false;
      break;
    }
    auto* element = GST_ELEMENT(g_value_get_object(&value));
    auto* factory = gst_element_get_factory(element);
    const std::string_view name =
        factory ? gst_plugin_feature_get_name(GST_PLUGIN_FEATURE(factory)) : "";
    closed = closed && participates_in_demand_coherency(name);
    if (name == "identity") {
      gboolean callbacks = FALSE;
      g_object_get(element, "signal-handoffs", &callbacks, nullptr);
      closed = closed && !callbacks;
    }
    g_value_reset(&value);
  }
  if (G_IS_VALUE(&value))
    g_value_unset(&value);
  gst_iterator_free(iterator);
  return closed;
}
} // namespace simaai::neat::pipeline_internal
