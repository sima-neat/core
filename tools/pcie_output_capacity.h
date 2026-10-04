#pragma once

#include <gst/SimaPluginStaticManifestAbi.h>

#include <algorithm>
#include <cstdint>
#include <limits>
#include <memory>
#include <stdexcept>
#include <string_view>

namespace simaai::neat::pcie_builder {

inline std::uint64_t output_capacity(const SimaPluginStageSpec& stage) {
  if (stage.payload_kind == SIMA_PLUGIN_STAGE_PAYLOAD_BOXDECODE) {
    const auto& decode = stage.payload.boxdecode;
    const std::string_view type = decode.decode_type ? decode.decode_type : "";
    if (decode.topk <= 0 || type.empty() || type.find("superpoint") != type.npos)
      throw std::runtime_error("unsupported PCIe detection output capacity");
    // BBOX wire ABI: uint32 count, then 24-byte records, optionally followed
    // by 160x160 mask bytes or 17 three-float keypoints per detection.
    std::uint64_t record = 24;
    if (type.find("seg") != type.npos || type.find("mask") != type.npos)
      record += 160U * 160U;
    if (type.find("pose") != type.npos || type.find("keypoint") != type.npos)
      record += 17U * 3U * sizeof(float);
    return sizeof(std::uint32_t) + static_cast<std::uint64_t>(decode.topk) * record;
  }
  // The compiled arena is a conservative bound, including physical gaps. For
  // materializers with their own pool, cover both packed and offset outputs.
  std::uint64_t bytes = stage.frame_arena_size_bytes;
  std::uint64_t packed = 0;
  for (guint i = 0; i < stage.physical_outputs_len; ++i) {
    const auto& output = stage.physical_outputs[i];
    if (output.source_byte_offset < 0 || output.size_bytes == 0 ||
        output.size_bytes > std::numeric_limits<std::uint64_t>::max() - packed ||
        output.size_bytes > std::numeric_limits<std::uint64_t>::max() -
                                static_cast<std::uint64_t>(output.source_byte_offset)) {
      throw std::runtime_error("invalid compiled PCIe output span");
    }
    packed += output.size_bytes;
    bytes =
        std::max(bytes, static_cast<std::uint64_t>(output.source_byte_offset) + output.size_bytes);
  }
  bytes = std::max(bytes, packed);
  if (!bytes)
    throw std::runtime_error("compiled PCIe output has no receive-capacity bound");
  return bytes;
}

inline std::uint64_t pipeline_output_capacity(GstElement* pipeline) {
  std::unique_ptr<GstContext, decltype(&gst_context_unref)> context(
      gst_element_get_context(pipeline, SIMA_PLUGIN_STATIC_MANIFEST_CONTEXT_TYPE),
      gst_context_unref);
  const auto* manifest = sima_plugin_manifest_context_accessor_checked(context.get(), nullptr);
  if (!manifest)
    throw std::runtime_error("PCIe pipeline has no compiled output manifest");

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
    const auto* stage =
        sima_plugin_manifest_stage_by_element_name(manifest, GST_ELEMENT_NAME(sink));
    if (stage) {
      gst_object_unref(sink);
      return output_capacity(*stage);
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
