#include "gst/NeatV4L2CopySource.h"
#include "gst/V4L2CopyCapture.h"
#include "gst/SimaTensorSetMetaAbi.h"
#include "gstsimaaitensorbuffer.h"

#include <gst/base/gstpushsrc.h>
#include <atomic>
#include <memory>
#include <mutex>
#include <stdexcept>
#include <string>

namespace {
namespace capture = simaai::neat::camera_copy;
struct SourceState {
  std::string device, fourcc = "GREY", name = "camera";
  guint width = 1, height = 1, count = 8, pool_count = 8, timeout_ms = 2000;
  std::shared_ptr<capture::Backend> backend;
  std::unique_ptr<capture::Capture> capture;
  GstBufferPool* pool = nullptr;
  std::mutex caps_mutex;
  GstCaps* negotiated_caps = nullptr;
  std::atomic<bool> interrupted{false};
  bool first_frame = true;
};
struct GstNeatV4L2CopySource {
  GstPushSrc parent;
  SourceState* state;
};
struct GstNeatV4L2CopySourceClass {
  GstPushSrcClass parent_class;
};
G_DEFINE_TYPE(GstNeatV4L2CopySource, gst_neat_v4l2_copy_source, GST_TYPE_PUSH_SRC)
enum {
  PROP_ZERO,
  PROP_DEVICE,
  PROP_FOURCC,
  PROP_NAME,
  PROP_WIDTH,
  PROP_HEIGHT,
  PROP_COUNT,
  PROP_POOL,
  PROP_TIMEOUT
};

void set_property(GObject* object, guint id, const GValue* value, GParamSpec* spec) {
  if (GST_STATE(object) > GST_STATE_READY) {
    GST_WARNING_OBJECT(object, "Raw camera properties are immutable while streaming");
    return;
  }
  auto& state = *reinterpret_cast<GstNeatV4L2CopySource*>(object)->state;
  switch (id) {
  case PROP_DEVICE:
    state.device = g_value_get_string(value) ? g_value_get_string(value) : "";
    break;
  case PROP_FOURCC:
    state.fourcc = g_value_get_string(value) ? g_value_get_string(value) : "";
    break;
  case PROP_NAME:
    state.name = g_value_get_string(value) ? g_value_get_string(value) : "";
    break;
  case PROP_WIDTH:
    state.width = g_value_get_uint(value);
    break;
  case PROP_HEIGHT:
    state.height = g_value_get_uint(value);
    break;
  case PROP_COUNT:
    state.count = g_value_get_uint(value);
    break;
  case PROP_POOL:
    state.pool_count = g_value_get_uint(value);
    break;
  case PROP_TIMEOUT:
    state.timeout_ms = g_value_get_uint(value);
    break;
  default:
    G_OBJECT_WARN_INVALID_PROPERTY_ID(object, id, spec);
  }
}
void get_property(GObject* object, guint id, GValue* value, GParamSpec* spec) {
  const auto& state = *reinterpret_cast<GstNeatV4L2CopySource*>(object)->state;
  switch (id) {
  case PROP_DEVICE:
    g_value_set_string(value, state.device.c_str());
    break;
  case PROP_FOURCC:
    g_value_set_string(value, state.fourcc.c_str());
    break;
  case PROP_NAME:
    g_value_set_string(value, state.name.c_str());
    break;
  case PROP_WIDTH:
    g_value_set_uint(value, state.width);
    break;
  case PROP_HEIGHT:
    g_value_set_uint(value, state.height);
    break;
  case PROP_COUNT:
    g_value_set_uint(value, state.count);
    break;
  case PROP_POOL:
    g_value_set_uint(value, state.pool_count);
    break;
  case PROP_TIMEOUT:
    g_value_set_uint(value, state.timeout_ms);
    break;
  default:
    G_OBJECT_WARN_INVALID_PROPERTY_ID(object, id, spec);
  }
}

GstCaps* get_caps(GstBaseSrc* base, GstCaps* filter) {
  auto& state = *reinterpret_cast<GstNeatV4L2CopySource*>(base)->state;
  GstCaps* caps = nullptr;
  {
    std::lock_guard lock(state.caps_mutex);
    caps = state.negotiated_caps ? gst_caps_ref(state.negotiated_caps)
                                 : gst_pad_get_pad_template_caps(GST_BASE_SRC_PAD(base));
  }
  if (filter) {
    GstCaps* intersection = gst_caps_intersect_full(filter, caps, GST_CAPS_INTERSECT_FIRST);
    gst_caps_unref(caps);
    return intersection;
  }
  return caps;
}
bool retire(SourceState& state) {
  {
    std::lock_guard lock(state.caps_mutex);
    gst_clear_caps(&state.negotiated_caps);
  }
  if (state.pool) {
    gst_buffer_pool_set_flushing(state.pool, TRUE);
    gst_buffer_pool_set_active(state.pool, FALSE);
    gst_object_unref(state.pool);
    state.pool = nullptr;
  }
  const bool safe = !state.capture || state.capture->stop();
  // On failure Capture deliberately retains its kernel mappings/fd; the caller
  // reports the failure instead of offering an automatic reopen/restart.
  if (safe)
    state.capture.reset();
  return safe;
}
gboolean start(GstBaseSrc* base) {
  auto* self = reinterpret_cast<GstNeatV4L2CopySource*>(base);
  auto& state = *self->state;
  try {
    if (state.capture)
      throw std::runtime_error(
          "CameraInput has a parked capture session; device recovery required");
    if (state.device.empty() || state.name.empty())
      throw std::runtime_error("CameraInput V4L2 requires device and buffer-name");
    state.interrupted.store(false);
    state.first_frame = true;
    state.capture = std::make_unique<capture::Capture>(state.backend ? state.backend
                                                                     : capture::linux_backend());
    state.capture->start(state.device,
                         {state.width, state.height, capture::parse_fourcc(state.fourcc), 0, 0},
                         state.count);
    const auto& format = state.capture->format();
    GstCaps* caps = gst_caps_new_simple(
        "application/vnd.simaai.tensor", "representation", G_TYPE_STRING, "tensor-set", "storage",
        G_TYPE_STRING, "tensorbuffer", "capture-format", G_TYPE_STRING, "V4L2_BYTES", "fourcc",
        G_TYPE_STRING, state.fourcc.c_str(), "dtype", G_TYPE_STRING, "UInt8", "width", G_TYPE_INT,
        static_cast<gint>(format.width), "height", G_TYPE_INT, static_cast<gint>(format.height),
        "row-stride", G_TYPE_UINT, format.stride, "sizeimage", G_TYPE_UINT, format.size, nullptr);
    // BaseSrc negotiates again after start(). Return the actual wire contract
    // from get_caps(), otherwise the broad pad template erases stride/sizeimage.
    {
      std::lock_guard lock(state.caps_mutex);
      state.negotiated_caps = gst_caps_ref(caps);
    }
    state.pool = gst_buffer_pool_new();
    GstStructure* config = gst_buffer_pool_get_config(state.pool);
    // Bounded system-memory pool. Pool exhaustion is cancellable backpressure,
    // never permission to overwrite a buffer retained by an application.
    gst_buffer_pool_config_set_params(config, caps, format.size, 0, state.pool_count);
    const bool configured = gst_buffer_pool_set_config(state.pool, config);
    const bool accepted = configured && gst_base_src_set_caps(base, caps);
    gst_caps_unref(caps);
    if (!accepted || !gst_buffer_pool_set_active(state.pool, TRUE))
      throw std::runtime_error("CameraInput copy pool/caps negotiation failed");
    return TRUE;
  } catch (const std::exception& error) {
    const bool safe = retire(state);
    GST_ELEMENT_ERROR(
        self, RESOURCE, FAILED, ("%s", error.what()),
        ("V4L2 retirement %s", safe ? "complete" : "UNKNOWN; device recovery required"));
    return FALSE;
  }
}
gboolean stop(GstBaseSrc* base) {
  auto* self = reinterpret_cast<GstNeatV4L2CopySource*>(base);
  self->state->interrupted.store(true);
  const bool safe = retire(*self->state);
  if (!safe)
    GST_ELEMENT_ERROR(self, RESOURCE, FAILED, ("CameraInput V4L2 STREAMOFF failed"),
                      ("Device recovery required; kernel mappings/fd retained"));
  return safe;
}
gboolean unlock(GstBaseSrc* base) {
  auto& state = *reinterpret_cast<GstNeatV4L2CopySource*>(base)->state;
  state.interrupted.store(true);
  if (state.pool)
    gst_buffer_pool_set_flushing(state.pool, TRUE);
  return TRUE;
}
gboolean unlock_stop(GstBaseSrc* base) {
  auto& state = *reinterpret_cast<GstNeatV4L2CopySource*>(base)->state;
  state.interrupted.store(false);
  if (state.pool)
    gst_buffer_pool_set_flushing(state.pool, FALSE);
  return TRUE;
}
GstFlowReturn create(GstPushSrc* source, GstBuffer** result) {
  auto* self = reinterpret_cast<GstNeatV4L2CopySource*>(source);
  auto& state = *self->state;
  GstBuffer* out = nullptr;
  GstMapInfo map = GST_MAP_INFO_INIT;
  bool mapped = false;
  try {
    const auto flow = gst_buffer_pool_acquire_buffer(state.pool, &out, nullptr);
    if (flow != GST_FLOW_OK)
      return flow;
    mapped = gst_buffer_map(out, &map, GST_MAP_WRITE);
    if (!mapped)
      throw std::runtime_error("CameraInput cannot map its owned copy buffer");
    const auto frame =
        state.capture->next({map.data, map.size}, state.interrupted, state.timeout_ms);
    gst_buffer_unmap(out, &map);
    mapped = false;
    if (!frame.bytes) {
      gst_buffer_unref(out);
      return GST_FLOW_FLUSHING;
    }
    gst_buffer_resize(out, 0, frame.bytes);
    GST_BUFFER_OFFSET(out) = frame.sequence;
    // Flat bytes preserve stride padding and any valid device trailer. Do not
    // invent an image tensor shape that silently discards proprietary payload.
    SimaTensorDescriptorV2 descriptor{};
    descriptor.size_bytes = frame.bytes;
    descriptor.dtype = SIMA_TENSOR_SET_DTYPE_UINT8_V1;
    descriptor.layout = SIMA_TENSOR_SET_LAYOUT_UNKNOWN_V1;
    descriptor.rank = 1;
    descriptor.shape[0] = frame.bytes;
    descriptor.stride_bytes[0] = 1;
    const gchar* names[] = {state.name.c_str(), nullptr};
    char* error = nullptr;
    const bool attached = simaai::gst::sima_tensor_buffer_attach_meta_flat(
        out, state.name.c_str(), &descriptor, 1, names, nullptr, 0, nullptr, 0, &error);
    const std::string reason = error ? error : "CameraInput tensor descriptor attachment failed";
    g_free(error);
    if (!attached)
      throw std::runtime_error(reason);
    if (state.first_frame)
      GST_BUFFER_FLAG_SET(out, GST_BUFFER_FLAG_DISCONT);
    state.first_frame = false;
    *result = out;
    return GST_FLOW_OK;
  } catch (const std::exception& error) {
    if (mapped)
      gst_buffer_unmap(out, &map);
    if (out)
      gst_buffer_unref(out);
    GST_ELEMENT_ERROR(self, RESOURCE, FAILED, ("%s", error.what()), ("V4L2 copy capture stopped"));
    return GST_FLOW_ERROR;
  }
}
void finalize(GObject* object) {
  auto* self = reinterpret_cast<GstNeatV4L2CopySource*>(object);
  retire(*self->state);
  delete self->state;
  G_OBJECT_CLASS(gst_neat_v4l2_copy_source_parent_class)->finalize(object);
}
void gst_neat_v4l2_copy_source_class_init(GstNeatV4L2CopySourceClass* klass) {
  auto* object = G_OBJECT_CLASS(klass);
  object->set_property = set_property;
  object->get_property = get_property;
  object->finalize = finalize;
  constexpr auto flags = static_cast<GParamFlags>(G_PARAM_READWRITE | G_PARAM_STATIC_STRINGS |
                                                  GST_PARAM_MUTABLE_READY);
  g_object_class_install_property(
      object, PROP_DEVICE,
      g_param_spec_string("device", "Device", "Explicit V4L2 device", "", flags));
  g_object_class_install_property(
      object, PROP_FOURCC,
      g_param_spec_string("fourcc", "Fourcc", "Exact eight-bit wire format", "GREY", flags));
  g_object_class_install_property(object, PROP_NAME,
                                  g_param_spec_string("buffer-name", "Tensor name",
                                                      "Logical capture tensor name", "camera",
                                                      flags));
  g_object_class_install_property(
      object, PROP_WIDTH, g_param_spec_uint("width", "Width", "Wire width", 1, G_MAXINT, 1, flags));
  g_object_class_install_property(
      object, PROP_HEIGHT,
      g_param_spec_uint("height", "Height", "Wire height", 1, G_MAXINT, 1, flags));
  g_object_class_install_property(object, PROP_COUNT,
                                  g_param_spec_uint("capture-buffer-count", "Capture count",
                                                    "MMAP queue depth", 4, 128, 8, flags));
  g_object_class_install_property(object, PROP_POOL,
                                  g_param_spec_uint("output-buffer-count", "Output count",
                                                    "Maximum owned buffers before backpressure", 2,
                                                    128, 8, flags));
  g_object_class_install_property(object, PROP_TIMEOUT,
                                  g_param_spec_uint("frame-timeout-ms", "Frame timeout",
                                                    "Timeout waiting for a camera frame", 1, 60000,
                                                    2000, flags));
  auto* element = GST_ELEMENT_CLASS(klass);
  gst_element_class_set_static_metadata(element, "Neat private V4L2 copy source", "Source/Video",
                                        "Owned raw camera byte tensors", "SiMa.ai");
  GstCaps* caps =
      gst_caps_from_string("application/"
                           "vnd.simaai.tensor,representation=(string)tensor-set,storage=(string)"
                           "tensorbuffer,capture-format=(string)V4L2_BYTES");
  gst_element_class_add_pad_template(
      element, gst_pad_template_new("src", GST_PAD_SRC, GST_PAD_ALWAYS, caps));
  gst_caps_unref(caps);
  auto* source = GST_BASE_SRC_CLASS(klass);
  source->get_caps = get_caps;
  source->start = start;
  source->stop = stop;
  source->unlock = unlock;
  source->unlock_stop = unlock_stop;
  GST_PUSH_SRC_CLASS(klass)->create = create;
}
void gst_neat_v4l2_copy_source_init(GstNeatV4L2CopySource* self) {
  self->state = new SourceState;
  gst_base_src_set_live(GST_BASE_SRC(self), TRUE);
  gst_base_src_set_format(GST_BASE_SRC(self), GST_FORMAT_TIME);
  // Timestamp delivery in pipeline running time; no assumption about the
  // driver's timestamp clock, and no timestamp bytes in the application tensor.
  gst_base_src_set_do_timestamp(GST_BASE_SRC(self), TRUE);
}
} // namespace
namespace simaai::neat {
bool register_neat_v4l2_copy_source() {
  static std::once_flag once;
  static bool result = false;
  std::call_once(once, [] {
    result = gst_element_register(nullptr, "neatv4l2copysrc", GST_RANK_NONE,
                                  gst_neat_v4l2_copy_source_get_type());
  });
  return result;
}
// Instance-scoped injection for internal tests. Not an installed/public API.
GstElement* make_v4l2_copy_source_for_test(std::shared_ptr<camera_copy::Backend> backend) {
  auto* element = GST_ELEMENT(g_object_new(gst_neat_v4l2_copy_source_get_type(), nullptr));
  reinterpret_cast<GstNeatV4L2CopySource*>(element)->state->backend = std::move(backend);
  return element;
}
} // namespace simaai::neat
