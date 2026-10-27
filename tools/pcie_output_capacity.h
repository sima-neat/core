#pragma once

#include <simaai/scheduler/client/stage.h>

#include <cstdint>
#include <limits>
#include <memory>
#include <stdexcept>

namespace simaai::neat::pcie_builder {

inline std::uint64_t output_capacity(const scheduler::client::FrameLayout& layout) {
  std::uint64_t bytes = 0;
  for (const auto& allocation : layout.allocations) {
    if (!allocation.size_bytes ||
        allocation.size_bytes > std::numeric_limits<std::uint64_t>::max() - bytes)
      throw std::runtime_error("invalid compiled PCIe output allocation");
    bytes += allocation.size_bytes;
  }
  if (!bytes)
    throw std::runtime_error("compiled PCIe output has no receive-capacity bound");
  return bytes;
}

inline std::uint64_t pipeline_output_capacity(GstElement* pipeline) {
  // Follow the actual terminal edge, not MPK stage order or a second planner.
  std::unique_ptr<GstIterator, decltype(&gst_iterator_free)> sinks(
      gst_bin_iterate_sinks(GST_BIN(pipeline)), gst_iterator_free);
  GValue value = G_VALUE_INIT;
  GstElement* sink = nullptr;
  while (gst_iterator_next(sinks.get(), &value) == GST_ITERATOR_OK) {
    auto* element = GST_ELEMENT(g_value_get_object(&value));
    auto* factory = gst_element_get_factory(element);
    if (factory &&
        g_strcmp0(gst_plugin_feature_get_name(GST_PLUGIN_FEATURE(factory)), "neatpciesink") == 0)
      sink = GST_ELEMENT(gst_object_ref(element));
    g_value_unset(&value);
    if (sink)
      break;
  }
  while (sink) {
    GstPad* pad = gst_element_get_static_pad(sink, "sink_0");
    if (!pad)
      pad = gst_element_get_static_pad(sink, "sink");
    GstPad* peer = pad ? gst_pad_get_peer(pad) : nullptr;
    GstElement* upstream = peer ? gst_pad_get_parent_element(peer) : nullptr;
    if (peer)
      gst_object_unref(peer);
    if (pad)
      gst_object_unref(pad);
    gst_object_unref(sink);
    sink = upstream;
    if (!sink)
      break;
    if (const auto layout = scheduler::client::stage_frame_layout(sink)) {
      gst_object_unref(sink);
      return output_capacity(*layout);
    }
    auto* factory = gst_element_get_factory(sink);
    const char* name = factory ? gst_plugin_feature_get_name(GST_PLUGIN_FEATURE(factory)) : "";
    if (g_strcmp0(name, "queue") && g_strcmp0(name, "identity") && g_strcmp0(name, "capsfilter")) {
      gst_object_unref(sink);
      break;
    }
  }
  throw std::runtime_error("cannot resolve the compiled PCIe terminal output producer");
}

} // namespace simaai::neat::pcie_builder
