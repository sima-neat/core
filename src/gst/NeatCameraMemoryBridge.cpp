#include "gst/NeatCameraMemoryBridge.h"

#include "pipeline/internal/SimaaiGstCompat.h"

#include <gst/gst.h>

#include <algorithm>
#include <cstdint>
#include <cstring>
#include <mutex>
#include <string>

namespace {

constexpr guint kDefaultNumBuffers = 4;
constexpr guint kMaxNumBuffers = 128;
constexpr const char* kDefaultBufferName = "camera";

GST_DEBUG_CATEGORY_STATIC(gst_neat_camera_memory_bridge_debug_category);
#define GST_CAT_DEFAULT gst_neat_camera_memory_bridge_debug_category

using GstNeatCameraMemoryBridge = struct _GstNeatCameraMemoryBridge;
using GstNeatCameraMemoryBridgeClass = struct _GstNeatCameraMemoryBridgeClass;

#define GST_TYPE_NEAT_CAMERA_MEMORY_BRIDGE (gst_neat_camera_memory_bridge_get_type())
#define GST_NEAT_CAMERA_MEMORY_BRIDGE(obj)                                                         \
  (G_TYPE_CHECK_INSTANCE_CAST((obj), GST_TYPE_NEAT_CAMERA_MEMORY_BRIDGE, GstNeatCameraMemoryBridge))

struct _GstNeatCameraMemoryBridge {
  GstElement parent;

  GstPad* sinkpad = nullptr;
  GstPad* srcpad = nullptr;

  GString* buffer_name = nullptr;
  guint num_buffers = kDefaultNumBuffers;
  guint64 configured_buffer_size = 0;
  gboolean copy_allowed = TRUE;
  gboolean silent = TRUE;

  GstBufferPool* pool = nullptr;
  gsize pool_size = 0;
  gint64 sequence_id = 0;
  guint64 passthrough_count = 0;
  guint64 copy_count = 0;
};

struct _GstNeatCameraMemoryBridgeClass {
  GstElementClass parent_class;
};

GType gst_neat_camera_memory_bridge_get_type();

G_DEFINE_TYPE_WITH_CODE(GstNeatCameraMemoryBridge, gst_neat_camera_memory_bridge, GST_TYPE_ELEMENT,
                        GST_DEBUG_CATEGORY_INIT(gst_neat_camera_memory_bridge_debug_category,
                                                "neatcamerabridge", 0,
                                                "Neat private adaptive camera memory bridge"));

static GstStaticPadTemplate sink_template =
    GST_STATIC_PAD_TEMPLATE("sink", GST_PAD_SINK, GST_PAD_ALWAYS, GST_STATIC_CAPS_ANY);
static GstStaticPadTemplate src_template =
    GST_STATIC_PAD_TEMPLATE("src", GST_PAD_SRC, GST_PAD_ALWAYS, GST_STATIC_CAPS_ANY);

enum {
  PROP_0,
  PROP_BUFFER_NAME,
  PROP_NODE_NAME,
  PROP_BUFFER_SIZE,
  PROP_NUM_BUFFERS,
  PROP_COPY_ALLOWED,
  PROP_SILENT,
};

bool debug_enabled(const GstNeatCameraMemoryBridge* self) {
  if (self && !self->silent)
    return true;
  const gchar* env = g_getenv("SIMA_CAMERA_BRIDGE_DEBUG");
  return env && *env && g_strcmp0(env, "0") != 0 && g_ascii_strcasecmp(env, "false") != 0;
}

const gchar* bridge_buffer_name(const GstNeatCameraMemoryBridge* self) {
  if (!self || !self->buffer_name || !self->buffer_name->str || !*self->buffer_name->str)
    return kDefaultBufferName;
  return self->buffer_name->str;
}

void release_pool(GstNeatCameraMemoryBridge* self) {
  if (!self || !self->pool)
    return;
  gst_simaai_free_buffer_pool(self->pool);
  self->pool = nullptr;
  self->pool_size = 0;
}

bool ensure_pool(GstNeatCameraMemoryBridge* self, gsize required_size) {
#if SIMA_HAS_SIMAAI_POOL
  if (!self || required_size == 0)
    return false;
  if (self->pool && self->pool_size >= required_size)
    return true;

  release_pool(self);
  gst_simaai_segment_memory_init_once();

  GstAllocator* allocator = gst_simaai_memory_get_segment_allocator();
  if (!allocator) {
    GST_ELEMENT_ERROR(self, RESOURCE, FAILED, ("SiMaAI segment allocator unavailable"),
                      ("gst_simaai_memory_get_segment_allocator returned NULL"));
    return false;
  }

  const GstMemoryFlags flags =
      static_cast<GstMemoryFlags>(GST_SIMAAI_MEMORY_TARGET_EV74 | GST_SIMAAI_MEMORY_FLAG_CACHED);
  const gsize segment_size = required_size;
  const gchar* segment_name = bridge_buffer_name(self);
  const guint buffers = std::max<guint>(1, self->num_buffers);
  GstBufferPool* pool = gst_simaai_allocate_buffer_pool2(
      GST_OBJECT(self), allocator, buffers, buffers, flags, 1, &segment_size, &segment_name);
  if (!pool) {
    GST_ELEMENT_ERROR(
        self, RESOURCE, FAILED, ("Failed to allocate EV74 SiMaAI camera pool"),
        ("size=%" G_GSIZE_FORMAT " num-buffers=%u name=%s", required_size, buffers, segment_name));
    return false;
  }

  self->pool = pool;
  self->pool_size = required_size;
  if (debug_enabled(self)) {
    GST_INFO_OBJECT(self,
                    "created EV74 camera bridge pool size=%" G_GSIZE_FORMAT " buffers=%u name=%s",
                    required_size, buffers, segment_name);
  }
  return true;
#else
  (void)self;
  (void)required_size;
  return false;
#endif
}

guintptr memory_phys(const GstMemory* memory) {
#if SIMA_HAS_SIMAAI_POOL
  return gst_simaai_segment_memory_get_phys_addr(memory);
#else
  (void)memory;
  return 0;
#endif
}

bool memory_is_ev74_simaai(const GstMemory* memory) {
  if (!memory)
    return false;
  if (!GST_MEMORY_FLAG_IS_SET(memory, GST_SIMAAI_MEMORY_TARGET_EV74))
    return false;
  return memory_phys(memory) != 0;
}

bool buffer_is_ev74_simaai(GstBuffer* buffer) {
  if (!buffer)
    return false;
  const guint n = gst_buffer_n_memory(buffer);
  if (n == 0)
    return false;
  for (guint i = 0; i < n; ++i) {
    GstMemory* mem = gst_buffer_peek_memory(buffer, i);
    if (!memory_is_ev74_simaai(mem))
      return false;
  }
  return true;
}

guint64 first_buffer_phys(GstBuffer* buffer) {
  if (!buffer)
    return 0;
  const guint n = gst_buffer_n_memory(buffer);
  for (guint i = 0; i < n; ++i) {
    GstMemory* mem = gst_buffer_peek_memory(buffer, i);
    const guint64 phys = static_cast<guint64>(memory_phys(mem));
    if (phys != 0)
      return phys;
  }
  return 0;
}

struct ExistingSimaMeta {
  bool has_buffer_id = false;
  gint64 buffer_id = 0;
  bool has_buffer_name = false;
  std::string buffer_name;
  bool has_buffer_offset = false;
  gint64 buffer_offset = 0;
  bool has_stream_id = false;
  std::string stream_id;
  bool has_frame_id = false;
  gint64 frame_id = 0;
  bool has_pcie_buffer_id = false;
  gint64 pcie_buffer_id = 0;
};

ExistingSimaMeta read_existing_sima_meta(GstBuffer* buffer) {
  ExistingSimaMeta out;
  if (!buffer)
    return out;
  GstCustomMeta* meta = gst_buffer_get_custom_meta(buffer, "GstSimaMeta");
  GstStructure* s = meta ? gst_custom_meta_get_structure(meta) : nullptr;
  if (!s)
    return out;
  gint64 v = 0;
  if (gst_structure_get_int64(s, "buffer-id", &v) && v > 0) {
    out.has_buffer_id = true;
    out.buffer_id = v;
  }
  const char* name = gst_structure_get_string(s, "buffer-name");
  if (name && *name) {
    out.has_buffer_name = true;
    out.buffer_name = name;
  }
  if (gst_structure_get_int64(s, "buffer-offset", &v)) {
    out.has_buffer_offset = true;
    out.buffer_offset = v;
  }
  const char* stream = gst_structure_get_string(s, "stream-id");
  if (stream && *stream) {
    out.has_stream_id = true;
    out.stream_id = stream;
  }
  if (gst_structure_get_int64(s, "frame-id", &v)) {
    out.has_frame_id = true;
    out.frame_id = v;
  }
  if (gst_structure_get_int64(s, "pcie-buffer-id", &v)) {
    out.has_pcie_buffer_id = true;
    out.pcie_buffer_id = v;
  }
  return out;
}

bool stamp_sima_meta(GstNeatCameraMemoryBridge* self, GstBuffer* buffer, guint64 phys_addr,
                     bool passthrough) {
  if (!self || !buffer)
    return false;

  const ExistingSimaMeta old = read_existing_sima_meta(buffer);
  if (GstCustomMeta* existing = gst_buffer_get_custom_meta(buffer, "GstSimaMeta")) {
    gst_buffer_remove_meta(buffer, &existing->meta);
  }

  GstCustomMeta* meta = gst_buffer_add_custom_meta(buffer, "GstSimaMeta");
  if (!meta) {
    GST_ELEMENT_ERROR(self, STREAM, FAILED, ("Unable to add GstSimaMeta"),
                      ("buffer=%p", static_cast<void*>(buffer)));
    return false;
  }
  GstStructure* s = gst_custom_meta_get_structure(meta);
  if (!s) {
    GST_ELEMENT_ERROR(self, STREAM, FAILED, ("GstSimaMeta has no structure"), ("no GstStructure"));
    return false;
  }

  const gint64 frame_id = old.has_frame_id ? old.frame_id : self->sequence_id++;
  // Pass-through EV74 camera buffers may already carry the camera plugin's
  // exact segment identity. Preserve it instead of replacing it with the first
  // memory's physical address; multi-plane/DMABUF sources can have a more
  // precise `buffer-id`/offset/name contract than the bridge can infer.
  const gint64 buffer_id =
      (passthrough && old.has_buffer_id) ? old.buffer_id : static_cast<gint64>(phys_addr);
  const std::string buffer_name =
      (passthrough && old.has_buffer_name) ? old.buffer_name : bridge_buffer_name(self);
  const gint64 buffer_offset = (passthrough && old.has_buffer_offset) ? old.buffer_offset : 0;
  const std::string stream_id = old.has_stream_id ? old.stream_id : std::string("0");
  const guint64 timestamp = GST_CLOCK_TIME_IS_VALID(GST_BUFFER_PTS(buffer))
                                ? static_cast<guint64>(GST_BUFFER_PTS(buffer))
                                : static_cast<guint64>(0);
  gst_structure_set(s, "buffer-id", G_TYPE_INT64, buffer_id, "buffer-name", G_TYPE_STRING,
                    buffer_name.c_str(), "buffer-offset", G_TYPE_INT64, buffer_offset, "stream-id",
                    G_TYPE_STRING, stream_id.c_str(), "frame-id", G_TYPE_INT64, frame_id,
                    "orig-input-seq", G_TYPE_INT64, frame_id, "timestamp", G_TYPE_UINT64, timestamp,
                    "origin_stage_id", G_TYPE_STRING, buffer_name.c_str(), "origin_output_slot",
                    G_TYPE_INT, 0, nullptr);
  if (old.has_pcie_buffer_id) {
    gst_structure_set(s, "pcie-buffer-id", G_TYPE_INT64, old.pcie_buffer_id, nullptr);
  }

  if (debug_enabled(self)) {
    GST_INFO_OBJECT(
        self, "%s GstSimaMeta buffer-id=%" G_GUINT64_FORMAT " name=%s frame=%" G_GINT64_FORMAT,
        passthrough ? "passthrough" : "copied", static_cast<guint64>(buffer_id),
        buffer_name.c_str(), frame_id);
  }
  return true;
}

GstBuffer* stamp_passthrough_buffer(GstNeatCameraMemoryBridge* self, GstBuffer* input) {
  GstBuffer* out = input;
  if (!gst_buffer_is_writable(out)) {
    out = gst_buffer_make_writable(out);
  }
  if (!out)
    return nullptr;
  const guint64 phys = first_buffer_phys(out);
  if (phys == 0) {
    GST_ELEMENT_ERROR(self, STREAM, FAILED,
                      ("EV74 pass-through buffer has no SiMaAI physical address"), ("phys=0"));
    gst_buffer_unref(out);
    return nullptr;
  }
  if (!stamp_sima_meta(self, out, phys, true)) {
    gst_buffer_unref(out);
    return nullptr;
  }
  ++self->passthrough_count;
  return out;
}

gsize required_copy_size(GstNeatCameraMemoryBridge* self, GstBuffer* input) {
  if (!self || !input)
    return 0;
  if (self->configured_buffer_size > 0)
    return static_cast<gsize>(self->configured_buffer_size);
  return gst_buffer_get_size(input);
}

GstBuffer* copy_to_ev74_buffer(GstNeatCameraMemoryBridge* self, GstBuffer* input) {
  const gsize required = required_copy_size(self, input);
  if (required == 0) {
    GST_ELEMENT_ERROR(self, STREAM, FORMAT, ("Camera bridge cannot infer input buffer size"),
                      ("set buffer-size on %s", GST_ELEMENT_NAME(self)));
    return nullptr;
  }
  if (!ensure_pool(self, required))
    return nullptr;

  GstBuffer* out = nullptr;
  const GstFlowReturn acquire = gst_buffer_pool_acquire_buffer(self->pool, &out, nullptr);
  if (acquire != GST_FLOW_OK || !out) {
    GST_ELEMENT_ERROR(self, RESOURCE, FAILED, ("Failed to acquire EV74 camera buffer"),
                      ("flow=%s", gst_flow_get_name(acquire)));
    return nullptr;
  }

  gst_buffer_copy_into(out, input,
                       static_cast<GstBufferCopyFlags>(GST_BUFFER_COPY_FLAGS |
                                                       GST_BUFFER_COPY_TIMESTAMPS |
                                                       GST_BUFFER_COPY_META),
                       0, -1);

  GstMapInfo inmap{};
  GstMapInfo outmap{};
  if (!gst_buffer_map(input, &inmap, GST_MAP_READ)) {
    GST_ELEMENT_ERROR(self, STREAM, FAILED, ("Failed to map camera input buffer for read"),
                      ("gst_buffer_map READ failed"));
    gst_buffer_unref(out);
    return nullptr;
  }
  if (inmap.size < required) {
    GST_ELEMENT_ERROR(
        self, STREAM, FORMAT, ("Camera input buffer smaller than required bridge size"),
        ("input=%" G_GSIZE_FORMAT " required=%" G_GSIZE_FORMAT, inmap.size, required));
    gst_buffer_unmap(input, &inmap);
    gst_buffer_unref(out);
    return nullptr;
  }
  if (!gst_buffer_map(out, &outmap, GST_MAP_WRITE)) {
    GST_ELEMENT_ERROR(self, STREAM, FAILED, ("Failed to map EV74 camera output buffer for write"),
                      ("gst_buffer_map WRITE failed"));
    gst_buffer_unmap(input, &inmap);
    gst_buffer_unref(out);
    return nullptr;
  }
  if (outmap.size < required) {
    GST_ELEMENT_ERROR(
        self, STREAM, FORMAT, ("EV74 camera output buffer is too small"),
        ("output=%" G_GSIZE_FORMAT " required=%" G_GSIZE_FORMAT, outmap.size, required));
    gst_buffer_unmap(out, &outmap);
    gst_buffer_unmap(input, &inmap);
    gst_buffer_unref(out);
    return nullptr;
  }

  std::memcpy(outmap.data, inmap.data, required);
  gst_buffer_unmap(out, &outmap);
  gst_buffer_unmap(input, &inmap);

  gst_buffer_resize(out, 0, required);
  const guint64 phys = first_buffer_phys(out);
  if (phys == 0) {
    GST_ELEMENT_ERROR(self, STREAM, FAILED, ("Copied EV74 buffer has no SiMaAI physical address"),
                      ("phys=0"));
    gst_buffer_unref(out);
    return nullptr;
  }
  if (!stamp_sima_meta(self, out, phys, false)) {
    gst_buffer_unref(out);
    return nullptr;
  }
  ++self->copy_count;
  return out;
}

GstFlowReturn bridge_chain(GstPad* /*pad*/, GstObject* parent, GstBuffer* input) {
  auto* self = GST_NEAT_CAMERA_MEMORY_BRIDGE(parent);
  if (!input)
    return GST_FLOW_ERROR;

  GstBuffer* out = nullptr;
  if (buffer_is_ev74_simaai(input)) {
    out = stamp_passthrough_buffer(self, input);
    if (!out)
      return GST_FLOW_ERROR;
    return gst_pad_push(self->srcpad, out);
  }

  if (!self->copy_allowed) {
    GST_ELEMENT_ERROR(self, STREAM, FAILED,
                      ("Camera buffer is not EV74 SiMaAI memory and copying is disabled"),
                      ("copy-allowed=false"));
    gst_buffer_unref(input);
    return GST_FLOW_ERROR;
  }

  out = copy_to_ev74_buffer(self, input);
  gst_buffer_unref(input);
  if (!out)
    return GST_FLOW_ERROR;
  return gst_pad_push(self->srcpad, out);
}

gboolean bridge_sink_event(GstPad* pad, GstObject* parent, GstEvent* event) {
  auto* self = GST_NEAT_CAMERA_MEMORY_BRIDGE(parent);
  switch (GST_EVENT_TYPE(event)) {
  case GST_EVENT_CAPS:
  case GST_EVENT_SEGMENT:
  case GST_EVENT_STREAM_START:
  case GST_EVENT_TAG:
  case GST_EVENT_EOS:
  case GST_EVENT_FLUSH_START:
  case GST_EVENT_FLUSH_STOP:
    return gst_pad_push_event(self->srcpad, event);
  default:
    return gst_pad_event_default(pad, parent, event);
  }
}

gboolean bridge_src_query(GstPad* pad, GstObject* parent, GstQuery* query) {
  auto* self = GST_NEAT_CAMERA_MEMORY_BRIDGE(parent);
  if (GST_QUERY_TYPE(query) == GST_QUERY_CAPS) {
    GstCaps* filter = nullptr;
    gst_query_parse_caps(query, &filter);
    GstCaps* caps = gst_pad_peer_query_caps(self->sinkpad, filter);
    if (!caps)
      caps = gst_caps_new_any();
    gst_query_set_caps_result(query, caps);
    gst_caps_unref(caps);
    return TRUE;
  }
  return gst_pad_query_default(pad, parent, query);
}

void bridge_set_property(GObject* object, guint property_id, const GValue* value,
                         GParamSpec* pspec) {
  auto* self = GST_NEAT_CAMERA_MEMORY_BRIDGE(object);
  switch (property_id) {
  case PROP_BUFFER_NAME:
  case PROP_NODE_NAME: {
    const gchar* v = g_value_get_string(value);
    g_string_assign(self->buffer_name, (v && *v) ? v : kDefaultBufferName);
    break;
  }
  case PROP_BUFFER_SIZE:
    self->configured_buffer_size = g_value_get_uint64(value);
    break;
  case PROP_NUM_BUFFERS:
    self->num_buffers = g_value_get_uint(value);
    break;
  case PROP_COPY_ALLOWED:
    self->copy_allowed = g_value_get_boolean(value);
    break;
  case PROP_SILENT:
    self->silent = g_value_get_boolean(value);
    break;
  default:
    G_OBJECT_WARN_INVALID_PROPERTY_ID(object, property_id, pspec);
    break;
  }
}

void bridge_get_property(GObject* object, guint property_id, GValue* value, GParamSpec* pspec) {
  auto* self = GST_NEAT_CAMERA_MEMORY_BRIDGE(object);
  switch (property_id) {
  case PROP_BUFFER_NAME:
  case PROP_NODE_NAME:
    g_value_set_string(value, bridge_buffer_name(self));
    break;
  case PROP_BUFFER_SIZE:
    g_value_set_uint64(value, self->configured_buffer_size);
    break;
  case PROP_NUM_BUFFERS:
    g_value_set_uint(value, self->num_buffers);
    break;
  case PROP_COPY_ALLOWED:
    g_value_set_boolean(value, self->copy_allowed);
    break;
  case PROP_SILENT:
    g_value_set_boolean(value, self->silent);
    break;
  default:
    G_OBJECT_WARN_INVALID_PROPERTY_ID(object, property_id, pspec);
    break;
  }
}

void bridge_finalize(GObject* object) {
  auto* self = GST_NEAT_CAMERA_MEMORY_BRIDGE(object);
  release_pool(self);
  if (self->buffer_name) {
    g_string_free(self->buffer_name, TRUE);
    self->buffer_name = nullptr;
  }
  G_OBJECT_CLASS(gst_neat_camera_memory_bridge_parent_class)->finalize(object);
}

GstStateChangeReturn bridge_change_state(GstElement* element, GstStateChange transition) {
  auto* self = GST_NEAT_CAMERA_MEMORY_BRIDGE(element);
  if (transition == GST_STATE_CHANGE_READY_TO_NULL ||
      transition == GST_STATE_CHANGE_PAUSED_TO_READY) {
    release_pool(self);
  }
  return GST_ELEMENT_CLASS(gst_neat_camera_memory_bridge_parent_class)
      ->change_state(element, transition);
}

void gst_neat_camera_memory_bridge_class_init(GstNeatCameraMemoryBridgeClass* klass) {
  auto* gobject_class = G_OBJECT_CLASS(klass);
  auto* element_class = GST_ELEMENT_CLASS(klass);

  gobject_class->set_property = bridge_set_property;
  gobject_class->get_property = bridge_get_property;
  gobject_class->finalize = bridge_finalize;
  element_class->change_state = bridge_change_state;

  gst_element_class_add_static_pad_template(element_class, &sink_template);
  gst_element_class_add_static_pad_template(element_class, &src_template);
  gst_element_class_set_static_metadata(element_class, "Neat private camera memory bridge",
                                        "Filter/Converter/Video",
                                        "Adaptively passes through EV74 SiMaAI camera buffers or "
                                        "copies OS camera buffers into EV74 SiMaAI memory",
                                        "SiMa.ai");

  g_object_class_install_property(
      gobject_class, PROP_BUFFER_NAME,
      g_param_spec_string("buffer-name", "Buffer name",
                          "GstSimaMeta buffer-name and SiMaAI segment name", kDefaultBufferName,
                          static_cast<GParamFlags>(G_PARAM_READWRITE | G_PARAM_STATIC_STRINGS)));
  g_object_class_install_property(
      gobject_class, PROP_NODE_NAME,
      g_param_spec_string("node-name", "Node name",
                          "Alias for buffer-name for compatibility with OsToSima naming",
                          kDefaultBufferName,
                          static_cast<GParamFlags>(G_PARAM_READWRITE | G_PARAM_STATIC_STRINGS)));
  g_object_class_install_property(
      gobject_class, PROP_BUFFER_SIZE,
      g_param_spec_uint64("buffer-size", "Buffer size",
                          "Output EV74 allocation size; 0 derives from each input buffer", 0,
                          G_MAXUINT64, 0,
                          static_cast<GParamFlags>(G_PARAM_READWRITE | G_PARAM_STATIC_STRINGS)));
  g_object_class_install_property(
      gobject_class, PROP_NUM_BUFFERS,
      g_param_spec_uint("num-buffers", "Number of EV74 buffers",
                        "Number of buffers in the private EV74 copy pool", 1, kMaxNumBuffers,
                        kDefaultNumBuffers,
                        static_cast<GParamFlags>(G_PARAM_READWRITE | G_PARAM_STATIC_STRINGS)));
  g_object_class_install_property(
      gobject_class, PROP_COPY_ALLOWED,
      g_param_spec_boolean("copy-allowed", "Copy allowed",
                           "Allow CPU copy fallback when upstream is not EV74 SiMaAI memory", TRUE,
                           static_cast<GParamFlags>(G_PARAM_READWRITE | G_PARAM_STATIC_STRINGS)));
  g_object_class_install_property(
      gobject_class, PROP_SILENT,
      g_param_spec_boolean("silent", "Silent", "Suppress bridge info logging", TRUE,
                           static_cast<GParamFlags>(G_PARAM_READWRITE | G_PARAM_STATIC_STRINGS)));
}

void gst_neat_camera_memory_bridge_init(GstNeatCameraMemoryBridge* self) {
  self->buffer_name = g_string_new(kDefaultBufferName);
  self->num_buffers = kDefaultNumBuffers;
  self->configured_buffer_size = 0;
  self->copy_allowed = TRUE;
  self->silent = TRUE;
  self->pool = nullptr;
  self->pool_size = 0;
  self->sequence_id = 0;
  self->passthrough_count = 0;
  self->copy_count = 0;

  self->sinkpad = gst_pad_new_from_static_template(&sink_template, "sink");
  gst_pad_set_chain_function(self->sinkpad, GST_DEBUG_FUNCPTR(bridge_chain));
  gst_pad_set_event_function(self->sinkpad, GST_DEBUG_FUNCPTR(bridge_sink_event));
  gst_element_add_pad(GST_ELEMENT(self), self->sinkpad);

  self->srcpad = gst_pad_new_from_static_template(&src_template, "src");
  gst_pad_set_query_function(self->srcpad, GST_DEBUG_FUNCPTR(bridge_src_query));
  gst_element_add_pad(GST_ELEMENT(self), self->srcpad);
}

} // namespace

namespace simaai::neat {

bool register_neat_camera_memory_bridge() {
  static std::once_flag once;
  static bool registered = false;
  std::call_once(once, []() {
    registered = gst_element_register(nullptr, kNeatCameraMemoryBridgeFactory, GST_RANK_NONE,
                                      GST_TYPE_NEAT_CAMERA_MEMORY_BRIDGE) == TRUE;
  });
  return registered;
}

} // namespace simaai::neat
