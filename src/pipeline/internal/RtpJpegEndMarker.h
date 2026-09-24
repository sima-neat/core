#pragma once

#include <gst/gst.h>

namespace simaai::neat::pipeline_internal {

inline GstPadProbeReturn repair_rtp_jpeg_end_marker(GstPad*, GstPadProbeInfo* info, gpointer) {
  GstBuffer* buffer = GST_PAD_PROBE_INFO_BUFFER(info);
  const gsize size = buffer ? gst_buffer_get_size(buffer) : 0;
  if (size < 2) {
    return GST_PAD_PROBE_OK;
  }
  guint8 end[2];
  if (gst_buffer_extract(buffer, size - 2, end, sizeof(end)) != sizeof(end) ||
      (end[0] == 0xff && end[1] == 0xd9)) {
    return GST_PAD_PROBE_OK;
  }

  // Older rtpjpegdepay versions can omit EOI when only one trailing byte matches.
  const guint8 eoi[] = {0xff, 0xd9};
  GstBuffer* marker = gst_buffer_new_allocate(nullptr, sizeof(eoi), nullptr);
  gst_buffer_fill(marker, 0, eoi, sizeof(eoi));
  GST_PAD_PROBE_INFO_DATA(info) = gst_buffer_append(buffer, marker);
  return GST_PAD_PROBE_OK;
}

inline void attach_rtp_jpeg_end_marker_probes(GstElement* pipeline) {
  GstIterator* iterator = gst_bin_iterate_recurse(GST_BIN(pipeline));
  GValue item = G_VALUE_INIT;
  while (gst_iterator_next(iterator, &item) == GST_ITERATOR_OK) {
    auto* element = GST_ELEMENT(g_value_get_object(&item));
    GstElementFactory* factory = gst_element_get_factory(element);
    if (factory &&
        g_strcmp0(gst_plugin_feature_get_name(GST_PLUGIN_FEATURE(factory)), "rtpjpegdepay") == 0) {
      GstPad* pad = gst_element_get_static_pad(element, "src");
      gst_pad_add_probe(pad, GST_PAD_PROBE_TYPE_BUFFER, repair_rtp_jpeg_end_marker, nullptr,
                        nullptr);
      gst_object_unref(pad);
    }
    g_value_reset(&item);
  }
  g_value_unset(&item);
  gst_iterator_free(iterator);
}

} // namespace simaai::neat::pipeline_internal
