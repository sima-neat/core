#include "gst/GstNeatMultipartJpegDemux.h"

#include "gst/GstSampleAttributes.h"
#include "nodes/common/internal/MultipartHeaderCapture.h"

#include <gst/gst.h>

#include <cstdint>
#include <cstring>
#include <map>
#include <mutex>
#include <string>
#include <utility>
#include <vector>

namespace {

constexpr const char* kFactoryName = "neatmultipartjpegdemux";

GST_DEBUG_CATEGORY_STATIC(gst_neat_multipart_jpeg_demux_debug_category);
#define GST_CAT_DEFAULT gst_neat_multipart_jpeg_demux_debug_category

using simaai::neat::multipart_internal::MultipartParser;

struct _GstNeatMultipartJpegDemux {
  GstElement element;

  GstPad* sinkpad;
  GstPad* srcpad;

  GMutex lock;
  gchar* boundary;        ///< Configured boundary override; NULL means auto-detect.
  gchar* capture_headers; ///< Comma-separated normalized allowlist.
  gboolean single_stream; ///< Keep the first negotiated JPEG caps for a stable stream.

  MultipartParser* parser;
  gboolean caps_pushed;
  gboolean caps_have_dimensions;
  gint caps_width;
  gint caps_height;
  /* Timing of the input chunk currently being parsed, carried onto the parts it completes. */
  GstClockTime chunk_pts;
  GstClockTime chunk_dts;
  GstClockTime chunk_duration;
  guint64 parts_emitted;
  /* Caps are only known once a part has been framed, but sticky events must go out as
     stream-start -> caps -> segment. Hold the segment until caps have been pushed. */
  GstEvent* pending_segment;
};

struct _GstNeatMultipartJpegDemuxClass {
  GstElementClass parent_class;
};

using GstNeatMultipartJpegDemux = struct _GstNeatMultipartJpegDemux;
using GstNeatMultipartJpegDemuxClass = struct _GstNeatMultipartJpegDemuxClass;

#define GST_TYPE_NEAT_MULTIPART_JPEG_DEMUX (gst_neat_multipart_jpeg_demux_get_type())
#define GST_NEAT_MULTIPART_JPEG_DEMUX(obj)                                                         \
  (G_TYPE_CHECK_INSTANCE_CAST((obj), GST_TYPE_NEAT_MULTIPART_JPEG_DEMUX, GstNeatMultipartJpegDemux))

G_DEFINE_TYPE(GstNeatMultipartJpegDemux, gst_neat_multipart_jpeg_demux, GST_TYPE_ELEMENT)

enum {
  PROP_0,
  PROP_BOUNDARY,
  PROP_CAPTURE_HEADERS,
  PROP_SINGLE_STREAM,
};

struct PendingPart {
  std::vector<uint8_t> body;
  std::map<std::string, std::string> attributes;
  GstClockTime pts = GST_CLOCK_TIME_NONE;
  GstClockTime dts = GST_CLOCK_TIME_NONE;
  GstClockTime duration = GST_CLOCK_TIME_NONE;
};

GstStaticPadTemplate sink_template = GST_STATIC_PAD_TEMPLATE(
    "sink", GST_PAD_SINK, GST_PAD_ALWAYS,
    GST_STATIC_CAPS("multipart/x-mixed-replace; application/x-multipart; ANY"));

GstStaticPadTemplate src_template =
    GST_STATIC_PAD_TEMPLATE("src", GST_PAD_SRC, GST_PAD_ALWAYS, GST_STATIC_CAPS("image/jpeg"));

/// Minimal JPEG SOF scan: enough to advertise output caps without `jpegparse`.
///
/// Returns false when no SOF marker is found; the caller then leaves the dimensions
/// unspecified rather than guessing.
bool scan_jpeg_dimensions(const uint8_t* data, std::size_t size, gint* width, gint* height) {
  if (!data || size < 4U || data[0] != 0xFFU || data[1] != 0xD8U) {
    return false;
  }
  std::size_t pos = 2U;
  while (pos + 3U < size) {
    if (data[pos] != 0xFFU) {
      ++pos;
      continue;
    }
    const uint8_t marker = data[pos + 1U];
    if (marker == 0xFFU) {
      ++pos;
      continue;
    }
    // Standalone markers carry no length payload.
    if (marker == 0xD8U || marker == 0x01U || (marker >= 0xD0U && marker <= 0xD7U)) {
      pos += 2U;
      continue;
    }
    if (marker == 0xD9U || marker == 0xDAU) {
      return false; // end of image or start of scan: no SOF found
    }
    if (pos + 3U >= size) {
      return false;
    }
    const std::size_t seg_len =
        (static_cast<std::size_t>(data[pos + 2U]) << 8) | static_cast<std::size_t>(data[pos + 3U]);
    if (seg_len < 2U || pos + 2U + seg_len > size) {
      return false;
    }
    // SOF0..SOF15 except the non-frame markers DHT (C4), JPG (C8) and DAC (CC).
    const bool is_sof = (marker >= 0xC0U && marker <= 0xCFU) && marker != 0xC4U &&
                        marker != 0xC8U && marker != 0xCCU;
    if (is_sof) {
      if (seg_len < 7U) {
        return false;
      }
      const std::size_t p = pos + 4U; // skip marker + length
      *height = static_cast<gint>((static_cast<gint>(data[p + 1U]) << 8) | data[p + 2U]);
      *width = static_cast<gint>((static_cast<gint>(data[p + 3U]) << 8) | data[p + 4U]);
      return (*width > 0 && *height > 0);
    }
    pos += 2U + seg_len;
  }
  return false;
}

/// Pull a boundary out of an upstream `multipart/x-mixed-replace` caps string.
std::string boundary_from_caps(GstCaps* caps) {
  if (!caps || gst_caps_get_size(caps) == 0U) {
    return {};
  }
  const GstStructure* s = gst_caps_get_structure(caps, 0U);
  if (!s) {
    return {};
  }
  const gchar* value = gst_structure_get_string(s, "boundary");
  return value ? std::string(value) : std::string();
}

void reset_parser_locked(GstNeatMultipartJpegDemux* self) {
  std::vector<std::string> capture;
  if (self->capture_headers && *self->capture_headers) {
    capture = simaai::neat::multipart_internal::split_capture_names(self->capture_headers);
  }
  const std::string boundary = self->boundary ? self->boundary : "";
  delete self->parser;
  self->parser = new MultipartParser(boundary, capture);
  self->caps_pushed = FALSE;
  self->caps_have_dimensions = FALSE;
  self->caps_width = 0;
  self->caps_height = 0;
}

void queue_part_locked(GstNeatMultipartJpegDemux* self, std::vector<PendingPart>* pending,
                       const uint8_t* body, std::size_t size,
                       std::map<std::string, std::string>&& attributes) {
  PendingPart part;
  part.body.assign(body, body + size);
  part.attributes = std::move(attributes);
  part.pts = self->chunk_pts;
  part.dts = self->chunk_dts;
  part.duration = self->chunk_duration;
  pending->emplace_back(std::move(part));
}

gboolean push_src_caps_if_needed(GstNeatMultipartJpegDemux* self, const uint8_t* body,
                                 std::size_t size) {
  gint width = 0;
  gint height = 0;
  const bool have_dims = scan_jpeg_dimensions(body, size, &width, &height);

  GstEvent* pending_segment = nullptr;
  g_mutex_lock(&self->lock);
  if (self->caps_pushed &&
      (self->single_stream || !have_dims ||
       (self->caps_have_dimensions && self->caps_width == width && self->caps_height == height))) {
    g_mutex_unlock(&self->lock);
    return TRUE;
  }

  const gboolean old_caps_pushed = self->caps_pushed;
  const gboolean old_caps_have_dimensions = self->caps_have_dimensions;
  const gint old_caps_width = self->caps_width;
  const gint old_caps_height = self->caps_height;
  GstCaps* caps = gst_caps_new_simple("image/jpeg", "parsed", G_TYPE_BOOLEAN, TRUE, nullptr);
  if (have_dims) {
    gst_caps_set_simple(caps, "width", G_TYPE_INT, width, "height", G_TYPE_INT, height, nullptr);
  }
  self->caps_pushed = TRUE;
  self->caps_have_dimensions = have_dims ? TRUE : FALSE;
  self->caps_width = have_dims ? width : 0;
  self->caps_height = have_dims ? height : 0;
  if (!old_caps_pushed) {
    pending_segment = self->pending_segment;
    self->pending_segment = nullptr;
  }
  g_mutex_unlock(&self->lock);

  // Never call downstream while holding the parser lock. Downstream event handlers and probes
  // are allowed to re-enter this element and update properties or state.
  const gboolean ok = gst_pad_set_caps(self->srcpad, caps);
  gst_caps_unref(caps);
  if (!ok) {
    if (pending_segment != nullptr) {
      gst_event_unref(pending_segment);
    }
    g_mutex_lock(&self->lock);
    self->caps_pushed = old_caps_pushed;
    self->caps_have_dimensions = old_caps_have_dimensions;
    self->caps_width = old_caps_width;
    self->caps_height = old_caps_height;
    g_mutex_unlock(&self->lock);
    return FALSE;
  }
  if (pending_segment != nullptr) {
    gst_pad_push_event(self->srcpad, pending_segment);
  }
  return TRUE;
}

/// Emit a part that was fully copied out of the parser while its lock was held.
///
/// Parsing and downstream delivery are deliberately separate phases. This function must be
/// called without the element lock so downstream callbacks may safely re-enter the element.
GstFlowReturn emit_part(GstNeatMultipartJpegDemux* self, PendingPart&& part) {
  if (!push_src_caps_if_needed(self, part.body.data(), part.body.size())) {
    GST_ERROR_OBJECT(self, "failed to negotiate image/jpeg caps downstream");
    return GST_FLOW_NOT_NEGOTIATED;
  }

  GstBuffer* buffer = gst_buffer_new_allocate(nullptr, part.body.size(), nullptr);
  if (!buffer) {
    return GST_FLOW_ERROR;
  }
  GstMapInfo map;
  if (!gst_buffer_map(buffer, &map, GST_MAP_WRITE)) {
    gst_buffer_unref(buffer);
    return GST_FLOW_ERROR;
  }
  std::memcpy(map.data, part.body.data(), part.body.size());
  gst_buffer_unmap(buffer, &map);

  // The part's own headers are attached to the very buffer carrying its bytes; there is no
  // side channel that could drift relative to the payload.
  if (!part.attributes.empty() &&
      !simaai::neat::gst_internal::write_attributes(buffer, part.attributes)) {
    GST_WARNING_OBJECT(self, "failed to attach captured multipart headers");
  }

  GST_BUFFER_PTS(buffer) = part.pts;
  GST_BUFFER_DTS(buffer) = part.dts;
  GST_BUFFER_DURATION(buffer) = part.duration;

  g_mutex_lock(&self->lock);
  GST_BUFFER_OFFSET(buffer) = static_cast<guint64>(self->parts_emitted);
  ++self->parts_emitted;
  g_mutex_unlock(&self->lock);

  return gst_pad_push(self->srcpad, buffer);
}

GstFlowReturn gst_neat_multipart_jpeg_demux_chain(GstPad* pad, GstObject* parent,
                                                  GstBuffer* buffer) {
  auto* self = GST_NEAT_MULTIPART_JPEG_DEMUX(parent);
  (void)pad;
  if (!buffer) {
    return GST_FLOW_OK;
  }

  GstMapInfo map;
  if (!gst_buffer_map(buffer, &map, GST_MAP_READ)) {
    gst_buffer_unref(buffer);
    return GST_FLOW_ERROR;
  }

  GstFlowReturn flow = GST_FLOW_OK;
  std::string err;
  std::vector<PendingPart> pending;
  g_mutex_lock(&self->lock);
  self->chunk_pts = GST_BUFFER_PTS(buffer);
  self->chunk_dts = GST_BUFFER_DTS(buffer);
  self->chunk_duration = GST_BUFFER_DURATION(buffer);
  if (!self->parser) {
    reset_parser_locked(self);
  }
  auto sink = [&](const uint8_t* body, std::size_t size,
                  std::map<std::string, std::string>&& attributes) {
    queue_part_locked(self, &pending, body, size, std::move(attributes));
    return true;
  };
  const bool ok = self->parser->feed(map.data, map.size, sink, &err);
  g_mutex_unlock(&self->lock);

  gst_buffer_unmap(buffer, &map);
  gst_buffer_unref(buffer);

  for (PendingPart& part : pending) {
    flow = emit_part(self, std::move(part));
    if (flow != GST_FLOW_OK) {
      break;
    }
  }

  if (!ok) {
    if (flow != GST_FLOW_OK) {
      // Downstream refused the part; propagate its flow result unchanged.
      return flow;
    }
    GST_ELEMENT_ERROR(self, STREAM, DEMUX, ("Malformed multipart stream"), ("%s", err.c_str()));
    return GST_FLOW_ERROR;
  }
  return flow;
}

gboolean gst_neat_multipart_jpeg_demux_sink_event(GstPad* pad, GstObject* parent, GstEvent* event) {
  auto* self = GST_NEAT_MULTIPART_JPEG_DEMUX(parent);

  switch (GST_EVENT_TYPE(event)) {
  case GST_EVENT_CAPS: {
    GstCaps* caps = nullptr;
    gst_event_parse_caps(event, &caps);
    const std::string detected = boundary_from_caps(caps);
    g_mutex_lock(&self->lock);
    if (!detected.empty() && (!self->boundary || !*self->boundary)) {
      g_free(self->boundary);
      self->boundary = g_strdup(detected.c_str());
      reset_parser_locked(self);
    }
    g_mutex_unlock(&self->lock);
    gst_event_unref(event);
    // The source pad negotiates image/jpeg itself once the first part is framed.
    return TRUE;
  }
  case GST_EVENT_SEGMENT: {
    /* Defer until caps are known so downstream never sees segment-before-caps. */
    g_mutex_lock(&self->lock);
    if (!self->caps_pushed) {
      gst_event_replace(&self->pending_segment, event);
      g_mutex_unlock(&self->lock);
      gst_event_unref(event);
      return TRUE;
    }
    g_mutex_unlock(&self->lock);
    return gst_pad_push_event(self->srcpad, event);
  }
  case GST_EVENT_EOS: {
    GstFlowReturn flow = GST_FLOW_OK;
    std::string err;
    std::vector<PendingPart> pending;
    bool finished = true;
    g_mutex_lock(&self->lock);
    if (self->parser) {
      auto sink = [&](const uint8_t* body, std::size_t size,
                      std::map<std::string, std::string>&& attributes) {
        queue_part_locked(self, &pending, body, size, std::move(attributes));
        return true;
      };
      finished = self->parser->finish(sink, &err);
    }
    g_mutex_unlock(&self->lock);

    for (PendingPart& part : pending) {
      flow = emit_part(self, std::move(part));
      if (flow != GST_FLOW_OK) {
        break;
      }
    }
    if (!finished && flow == GST_FLOW_OK) {
      GST_WARNING_OBJECT(self, "multipart stream ended mid-part: %s", err.c_str());
    }
    return gst_pad_push_event(self->srcpad, event);
  }
  case GST_EVENT_FLUSH_STOP: {
    g_mutex_lock(&self->lock);
    // Buffered bytes and any half-assembled part's headers belong to the old segment.
    reset_parser_locked(self);
    gst_event_replace(&self->pending_segment, nullptr);
    g_mutex_unlock(&self->lock);
    return gst_pad_push_event(self->srcpad, event);
  }
  default:
    return gst_pad_event_default(pad, parent, event);
  }
}

void gst_neat_multipart_jpeg_demux_set_property(GObject* object, guint prop_id, const GValue* value,
                                                GParamSpec* pspec) {
  auto* self = GST_NEAT_MULTIPART_JPEG_DEMUX(object);
  g_mutex_lock(&self->lock);
  switch (prop_id) {
  case PROP_BOUNDARY:
    g_free(self->boundary);
    self->boundary = g_value_dup_string(value);
    reset_parser_locked(self);
    break;
  case PROP_CAPTURE_HEADERS:
    g_free(self->capture_headers);
    self->capture_headers = g_value_dup_string(value);
    reset_parser_locked(self);
    break;
  case PROP_SINGLE_STREAM:
    self->single_stream = g_value_get_boolean(value);
    reset_parser_locked(self);
    break;
  default:
    g_mutex_unlock(&self->lock);
    G_OBJECT_WARN_INVALID_PROPERTY_ID(object, prop_id, pspec);
    return;
  }
  g_mutex_unlock(&self->lock);
}

void gst_neat_multipart_jpeg_demux_get_property(GObject* object, guint prop_id, GValue* value,
                                                GParamSpec* pspec) {
  auto* self = GST_NEAT_MULTIPART_JPEG_DEMUX(object);
  g_mutex_lock(&self->lock);
  switch (prop_id) {
  case PROP_BOUNDARY:
    g_value_set_string(value, self->boundary ? self->boundary : "");
    break;
  case PROP_CAPTURE_HEADERS:
    g_value_set_string(value, self->capture_headers ? self->capture_headers : "");
    break;
  case PROP_SINGLE_STREAM:
    g_value_set_boolean(value, self->single_stream);
    break;
  default:
    g_mutex_unlock(&self->lock);
    G_OBJECT_WARN_INVALID_PROPERTY_ID(object, prop_id, pspec);
    return;
  }
  g_mutex_unlock(&self->lock);
}

void gst_neat_multipart_jpeg_demux_finalize(GObject* object) {
  auto* self = GST_NEAT_MULTIPART_JPEG_DEMUX(object);
  delete self->parser;
  self->parser = nullptr;
  gst_event_replace(&self->pending_segment, nullptr);
  g_free(self->boundary);
  g_free(self->capture_headers);
  g_mutex_clear(&self->lock);
  G_OBJECT_CLASS(gst_neat_multipart_jpeg_demux_parent_class)->finalize(object);
}

GstStateChangeReturn gst_neat_multipart_jpeg_demux_change_state(GstElement* element,
                                                                GstStateChange transition) {
  auto* self = GST_NEAT_MULTIPART_JPEG_DEMUX(element);
  if (transition == GST_STATE_CHANGE_READY_TO_PAUSED ||
      transition == GST_STATE_CHANGE_PAUSED_TO_READY) {
    g_mutex_lock(&self->lock);
    reset_parser_locked(self);
    self->parts_emitted = 0U;
    gst_event_replace(&self->pending_segment, nullptr);
    g_mutex_unlock(&self->lock);
  }
  return GST_ELEMENT_CLASS(gst_neat_multipart_jpeg_demux_parent_class)
      ->change_state(element, transition);
}

void gst_neat_multipart_jpeg_demux_class_init(GstNeatMultipartJpegDemuxClass* klass) {
  auto* gobject_class = G_OBJECT_CLASS(klass);
  auto* element_class = GST_ELEMENT_CLASS(klass);

  gobject_class->set_property = gst_neat_multipart_jpeg_demux_set_property;
  gobject_class->get_property = gst_neat_multipart_jpeg_demux_get_property;
  gobject_class->finalize = gst_neat_multipart_jpeg_demux_finalize;
  element_class->change_state = gst_neat_multipart_jpeg_demux_change_state;

  g_object_class_install_property(
      gobject_class, PROP_BOUNDARY,
      g_param_spec_string("boundary", "Boundary",
                          "Multipart boundary without the leading '--'; empty auto-detects", "",
                          static_cast<GParamFlags>(G_PARAM_READWRITE | G_PARAM_STATIC_STRINGS)));
  g_object_class_install_property(
      gobject_class, PROP_CAPTURE_HEADERS,
      g_param_spec_string(
          "capture-headers", "Capture headers",
          "Comma-separated, already-normalized part header names to capture as attributes", "",
          static_cast<GParamFlags>(G_PARAM_READWRITE | G_PARAM_STATIC_STRINGS)));
  g_object_class_install_property(
      gobject_class, PROP_SINGLE_STREAM,
      g_param_spec_boolean(
          "single-stream", "Single stream",
          "Keep the first negotiated JPEG caps instead of renegotiating dimension changes", FALSE,
          static_cast<GParamFlags>(G_PARAM_READWRITE | G_PARAM_STATIC_STRINGS)));

  gst_element_class_add_static_pad_template(element_class, &sink_template);
  gst_element_class_add_static_pad_template(element_class, &src_template);
  gst_element_class_set_static_metadata(
      element_class, "Neat multipart JPEG demuxer", "Codec/Demuxer",
      "Frames multipart JPEG parts and attaches selected part headers to the same buffer",
      "SiMa.ai");
}

void gst_neat_multipart_jpeg_demux_init(GstNeatMultipartJpegDemux* self) {
  g_mutex_init(&self->lock);
  self->boundary = g_strdup("");
  self->capture_headers = g_strdup("");
  self->single_stream = FALSE;
  self->parser = nullptr;
  self->caps_pushed = FALSE;
  self->caps_have_dimensions = FALSE;
  self->caps_width = 0;
  self->caps_height = 0;
  self->parts_emitted = 0U;
  self->pending_segment = nullptr;
  self->chunk_pts = GST_CLOCK_TIME_NONE;
  self->chunk_dts = GST_CLOCK_TIME_NONE;
  self->chunk_duration = GST_CLOCK_TIME_NONE;

  self->sinkpad = gst_pad_new_from_static_template(&sink_template, "sink");
  gst_pad_set_chain_function(self->sinkpad, gst_neat_multipart_jpeg_demux_chain);
  gst_pad_set_event_function(self->sinkpad, gst_neat_multipart_jpeg_demux_sink_event);
  GST_PAD_SET_PROXY_ALLOCATION(self->sinkpad);
  gst_element_add_pad(GST_ELEMENT(self), self->sinkpad);

  self->srcpad = gst_pad_new_from_static_template(&src_template, "src");
  gst_element_add_pad(GST_ELEMENT(self), self->srcpad);
}

} // namespace

namespace simaai::neat {

bool register_neat_multipart_jpeg_demux() {
  static std::once_flag once;
  static bool registered = false;
  std::call_once(once, []() {
    GST_DEBUG_CATEGORY_INIT(gst_neat_multipart_jpeg_demux_debug_category, kFactoryName, 0,
                            "Neat multipart JPEG demuxer");
    registered = gst_element_register(nullptr, kFactoryName, GST_RANK_NONE,
                                      GST_TYPE_NEAT_MULTIPART_JPEG_DEMUX) == TRUE;
  });
  return registered;
}

} // namespace simaai::neat
