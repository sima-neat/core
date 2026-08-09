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
  gchar* configured_boundary; ///< Property value; empty means auto-detect per connection.
  gchar* active_boundary;     ///< Boundary learned from the current upstream caps.
  gchar* capture_headers;     ///< Comma-separated normalized allowlist.
  gboolean single_stream;     ///< Keep the first negotiated JPEG caps for a stable stream.

  MultipartParser* parser;
  gboolean caps_pushed;
  gboolean caps_have_dimensions;
  gint caps_width;
  gint caps_height;
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

bool fail_jpeg(std::string* err, const char* message) {
  if (err) {
    *err = message;
  }
  return false;
}

/// Validate exactly one complete JPEG image and return its frame dimensions.
bool validate_jpeg_frame(const uint8_t* data, std::size_t size, gint* width, gint* height,
                         std::string* err) {
  if (!data || size < 4U || data[0] != 0xFFU || data[1] != 0xD8U) {
    return fail_jpeg(err, "JPEG part does not start with SOI");
  }
  std::size_t pos = 2U;
  bool in_scan = false;
  bool saw_sof = false;
  bool saw_sos = false;
  bool needs_dnl = false;
  while (pos < size) {
    if (data[pos] != 0xFFU) {
      if (in_scan) {
        ++pos;
        continue;
      }
      return fail_jpeg(err, "JPEG data outside a marker segment");
    }

    while (pos < size && data[pos] == 0xFFU) {
      ++pos;
    }
    if (pos >= size) {
      return fail_jpeg(err, "JPEG ends inside a marker");
    }
    const uint8_t marker = data[pos++];
    if (marker == 0x00U) {
      if (!in_scan) {
        return fail_jpeg(err, "JPEG contains stuffed data outside a scan");
      }
      continue;
    }
    if (marker == 0xD9U) {
      if (!saw_sof || !saw_sos || needs_dnl) {
        return fail_jpeg(err, "JPEG reaches EOI before a complete frame header");
      }
      if (pos != size) {
        return fail_jpeg(err, "multipart part contains data after JPEG EOI");
      }
      return true;
    }
    if (marker == 0xD8U) {
      return fail_jpeg(err, "multipart part contains more than one JPEG SOI");
    }
    if (marker == 0x01U || (marker >= 0xD0U && marker <= 0xD7U)) {
      if (marker != 0x01U && !in_scan) {
        return fail_jpeg(err, "JPEG restart marker appears outside a scan");
      }
      continue;
    }
    if (pos + 2U > size) {
      return fail_jpeg(err, "JPEG segment length is truncated");
    }
    const std::size_t seg_len =
        (static_cast<std::size_t>(data[pos]) << 8) | static_cast<std::size_t>(data[pos + 1U]);
    if (seg_len < 2U || pos + seg_len > size) {
      return fail_jpeg(err, "JPEG segment exceeds the MIME part");
    }
    // SOF0..SOF15 except the non-frame markers DHT (C4), JPG (C8) and DAC (CC).
    const bool is_sof = (marker >= 0xC0U && marker <= 0xCFU) && marker != 0xC4U &&
                        marker != 0xC8U && marker != 0xCCU;
    if (is_sof) {
      if (seg_len < 11U) {
        return fail_jpeg(err, "JPEG SOF segment is too short");
      }
      const std::size_t p = pos + 2U; // skip segment length
      const std::size_t component_count = data[p + 5U];
      if (component_count == 0U || seg_len != 8U + 3U * component_count) {
        return fail_jpeg(err, "JPEG SOF component table is malformed");
      }
      *height = static_cast<gint>((static_cast<gint>(data[p + 1U]) << 8) | data[p + 2U]);
      *width = static_cast<gint>((static_cast<gint>(data[p + 3U]) << 8) | data[p + 4U]);
      if (*width <= 0) {
        return fail_jpeg(err, "JPEG SOF dimensions are invalid");
      }
      needs_dnl = *height == 0;
      saw_sof = true;
    }
    if (marker == 0xDCU) {
      if (!saw_sof || !saw_sos || !needs_dnl || seg_len != 4U) {
        return fail_jpeg(err, "JPEG DNL segment is invalid");
      }
      *height = static_cast<gint>((static_cast<gint>(data[pos + 2U]) << 8) | data[pos + 3U]);
      if (*height <= 0) {
        return fail_jpeg(err, "JPEG DNL height is invalid");
      }
      needs_dnl = false;
    }
    if (marker == 0xDAU) {
      if (!saw_sof) {
        return fail_jpeg(err, "JPEG scan appears before a frame header");
      }
      if (seg_len < 8U) {
        return fail_jpeg(err, "JPEG SOS segment is too short");
      }
      const std::size_t scan_component_count = data[pos + 2U];
      if (scan_component_count == 0U || seg_len != 6U + 2U * scan_component_count) {
        return fail_jpeg(err, "JPEG SOS component table is malformed");
      }
      saw_sos = true;
      in_scan = true;
    } else {
      in_scan = false;
    }
    pos += seg_len;
  }
  return fail_jpeg(err, "JPEG part is missing EOI");
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
  const bool configured = self->configured_boundary && *self->configured_boundary;
  const std::string boundary =
      configured ? self->configured_boundary : (self->active_boundary ? self->active_boundary : "");
  delete self->parser;
  self->parser = new MultipartParser(boundary, capture);
  self->caps_pushed = FALSE;
  self->caps_have_dimensions = FALSE;
  self->caps_width = 0;
  self->caps_height = 0;
}

void queue_part_locked(std::vector<PendingPart>* pending, const uint8_t* body, std::size_t size,
                       std::map<std::string, std::string>&& attributes,
                       MultipartParser::PartTiming timing) {
  PendingPart part;
  part.body.assign(body, body + size);
  part.attributes = std::move(attributes);
  part.pts = timing.pts;
  part.dts = timing.dts;
  part.duration = timing.duration;
  pending->emplace_back(std::move(part));
}

gboolean push_src_caps_if_needed(GstNeatMultipartJpegDemux* self, gint width, gint height) {
  GstEvent* pending_segment = nullptr;
  g_mutex_lock(&self->lock);
  if (self->caps_pushed &&
      (self->single_stream ||
       (self->caps_have_dimensions && self->caps_width == width && self->caps_height == height))) {
    g_mutex_unlock(&self->lock);
    return TRUE;
  }

  const gboolean old_caps_pushed = self->caps_pushed;
  const gboolean old_caps_have_dimensions = self->caps_have_dimensions;
  const gint old_caps_width = self->caps_width;
  const gint old_caps_height = self->caps_height;
  GstCaps* caps = gst_caps_new_simple("image/jpeg", "parsed", G_TYPE_BOOLEAN, TRUE, nullptr);
  gst_caps_set_simple(caps, "width", G_TYPE_INT, width, "height", G_TYPE_INT, height, nullptr);
  self->caps_pushed = TRUE;
  self->caps_have_dimensions = TRUE;
  self->caps_width = width;
  self->caps_height = height;
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
  gint width = 0;
  gint height = 0;
  std::string jpeg_error;
  if (!validate_jpeg_frame(part.body.data(), part.body.size(), &width, &height, &jpeg_error)) {
    GST_ELEMENT_ERROR(self, STREAM, DECODE, ("Invalid JPEG MIME part"), ("%s", jpeg_error.c_str()));
    return GST_FLOW_ERROR;
  }
  if (!push_src_caps_if_needed(self, width, height)) {
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
    GST_ELEMENT_ERROR(self, STREAM, FAILED, ("Failed to attach multipart frame attributes"),
                      ("GstSimaMeta attribute write failed"));
    gst_buffer_unref(buffer);
    return GST_FLOW_ERROR;
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
  if (!self->parser) {
    reset_parser_locked(self);
  }
  auto sink = [&](const uint8_t* body, std::size_t size,
                  std::map<std::string, std::string>&& attributes,
                  MultipartParser::PartTiming timing) {
    queue_part_locked(&pending, body, size, std::move(attributes), timing);
    return true;
  };
  const MultipartParser::PartTiming timing{GST_BUFFER_PTS(buffer), GST_BUFFER_DTS(buffer),
                                           GST_BUFFER_DURATION(buffer)};
  const bool ok = self->parser->feed(map.data, map.size, sink, &err, timing);
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
  case GST_EVENT_STREAM_START: {
    g_mutex_lock(&self->lock);
    if (!self->configured_boundary || !*self->configured_boundary) {
      g_clear_pointer(&self->active_boundary, g_free);
    }
    reset_parser_locked(self);
    gst_event_replace(&self->pending_segment, nullptr);
    g_mutex_unlock(&self->lock);
    return gst_pad_push_event(self->srcpad, event);
  }
  case GST_EVENT_CAPS: {
    GstCaps* caps = nullptr;
    gst_event_parse_caps(event, &caps);
    const std::string detected = boundary_from_caps(caps);
    g_mutex_lock(&self->lock);
    if ((!self->configured_boundary || !*self->configured_boundary) &&
        g_strcmp0(self->active_boundary, detected.c_str()) != 0) {
      g_free(self->active_boundary);
      self->active_boundary = g_strdup(detected.c_str());
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
                      std::map<std::string, std::string>&& attributes,
                      MultipartParser::PartTiming timing) {
        queue_part_locked(&pending, body, size, std::move(attributes), timing);
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
    if (!finished || flow != GST_FLOW_OK) {
      if (!finished && flow == GST_FLOW_OK) {
        GST_ELEMENT_ERROR(self, STREAM, DEMUX, ("Malformed multipart stream at EOS"),
                          ("%s", err.c_str()));
      }
      gst_event_unref(event);
      return FALSE;
    }
    return gst_pad_push_event(self->srcpad, event);
  }
  case GST_EVENT_FLUSH_STOP: {
    g_mutex_lock(&self->lock);
    // Buffered bytes and any half-assembled part's headers belong to the old segment.
    if (!self->configured_boundary || !*self->configured_boundary) {
      g_clear_pointer(&self->active_boundary, g_free);
    }
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
    g_free(self->configured_boundary);
    self->configured_boundary = g_value_dup_string(value);
    g_clear_pointer(&self->active_boundary, g_free);
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
    g_value_set_string(value, self->configured_boundary ? self->configured_boundary : "");
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
  g_free(self->configured_boundary);
  g_free(self->active_boundary);
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
  self->configured_boundary = g_strdup("");
  self->active_boundary = g_strdup("");
  self->capture_headers = g_strdup("");
  self->single_stream = FALSE;
  self->parser = nullptr;
  self->caps_pushed = FALSE;
  self->caps_have_dimensions = FALSE;
  self->caps_width = 0;
  self->caps_height = 0;
  self->parts_emitted = 0U;
  self->pending_segment = nullptr;

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
