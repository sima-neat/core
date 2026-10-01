#include "gst/NeatRawCameraSource.h"
#include "gst/RawCameraCapture.h"
#include "gst/RawCameraMemoryContract.h"
#include "gst/SimaTensorSetMetaAbi.h"
#include "gstsimaaitensorbuffer.h"

#include <gst/base/gstpushsrc.h>
#include <atomic>
#include <cerrno>
#include <dlfcn.h>
#include <fcntl.h>
#include <map>
#include <linux/videodev2.h>
#include <memory>
#include <mutex>
#include <poll.h>
#include <stdexcept>
#include <string>
#include <sys/ioctl.h>
#include <sys/file.h>
#include <time.h>
#include <unistd.h>
#include <utility>

namespace {
namespace raw = simaai::neat::raw_camera;

struct MemoryOwner {
  GstMemory* memory = nullptr;
  int fd = -1;
  ~MemoryOwner() {
    if (fd >= 0)
      ::close(fd);
    if (memory)
      gst_memory_unref(memory);
  }
};

class LinuxBackend final : public raw::Backend {
public:
  LinuxBackend() {
    using Getter = const GstNeatCameraMemoryApiV1* (*)(guint32);
    const auto get =
        reinterpret_cast<Getter>(dlsym(RTLD_DEFAULT, "gst_neat_camera_memory_api_get"));
    api = get ? get(GST_NEAT_CAMERA_MEMORY_API_VERSION_1) : nullptr;
    if (!raw::supports_raw_camera_memory(api))
      throw std::runtime_error(
          "raw camera requires matching Internals camera-memory ABI v1 with DMA-BUF export "
          "and shared-owner retention; update the allocator runtime");
    phys_ = reinterpret_cast<guintptr (*)(const GstMemory*)>(
        dlsym(RTLD_DEFAULT, "gst_neat_segment_memory_get_phys_addr"));
    segment_ = reinterpret_cast<decltype(segment_)>(
        dlsym(RTLD_DEFAULT, "gst_simaai_memory_get_segment"));
    bus_ = reinterpret_cast<decltype(bus_)>(dlsym(RTLD_DEFAULT, "simaai_memory_get_bus"));
    if (!phys_ || !segment_ || !bus_)
      throw std::runtime_error(
          "raw camera requires SiMa allocation physical identity and EV bus-address accessors");
    api->init_once();
    allocator_ = api->get_allocator();
    if (!allocator_)
      throw std::runtime_error("raw camera SiMa allocator unavailable");
  }
  ~LinuxBackend() override {
    if (allocator_)
      gst_object_unref(allocator_);
  }
  int open_device(const std::string& device) override {
    const int fd = ::open(device.c_str(), O_RDWR | O_NONBLOCK | O_CLOEXEC);
    if (fd >= 0 && ::flock(fd, LOCK_EX | LOCK_NB) < 0) {
      ::close(fd);
      return -1;
    }
    return fd;
  }
  int io(int fd, unsigned long request, void* arg) override {
    int result;
    do {
      result = ::ioctl(fd, request, arg);
    } while (result < 0 && errno == EINTR);
    if (result == 0 && request == VIDIOC_QUERYCAP) {
      const auto* caps = static_cast<const v4l2_capability*>(arg);
      // No user bypass: leaking references cannot protect against process exit
      // closing the fd and reclaiming an unqualified driver's queued buffers.
      raw::require_qualified_capture_driver(
          std::string_view(reinterpret_cast<const char*>(caps->bus_info), sizeof(caps->bus_info)));
    }
    return result;
  }
  int wait(int fd, int timeout_ms) override {
    pollfd value{fd, POLLIN, 0};
    const int result = ::poll(&value, 1, timeout_ms);
    if (result > 0 && (value.revents & (POLLERR | POLLHUP | POLLNVAL))) {
      errno = EIO;
      return -1;
    }
    return result > 0 && !(value.revents & POLLIN) ? 0 : result;
  }
  void close_device(int fd) override {
    ::close(fd);
  }
  raw::Allocation allocate(std::uint32_t size, const std::string& name) override {
    GstSimaaiAllocationParams params;
    api->allocation_params_init(&params);
    params.parent.flags =
        static_cast<GstMemoryFlags>(GST_SIMAAI_MEMORY_TARGET_EV74 | GST_SIMAAI_MEMORY_FLAG_CACHED);
    if (!api->allocation_params_add_segment(&params, size, name.c_str()))
      throw std::runtime_error("raw camera cannot declare its packed capture allocation");
    auto owner = std::make_shared<MemoryOwner>();
    owner->memory = gst_allocator_alloc(allocator_, size, &params.parent);
    if (!owner->memory || !api->has_packed_segments(owner->memory))
      throw std::runtime_error("raw camera packed allocation failed");
    owner->fd = api->export_dmabuf_fd(owner->memory, name.c_str(), O_CLOEXEC);
    if (owner->fd < 0)
      throw std::runtime_error("raw camera DMA-BUF export failed");
    auto* segment = static_cast<simaai_memory_t*>(segment_(owner->memory, name.c_str()));
    if (!segment)
      throw std::runtime_error("raw camera cannot resolve its allocated capture segment");
    // Modalix physical DDR addresses can exceed 32 bits. Only the memory
    // manager's EV/STU bus address belongs in the EV addressability check.
    return {owner->fd, phys_(owner->memory), bus_(segment), size, owner};
  }
  const GstNeatCameraMemoryApiV1* api = nullptr;

private:
  GstAllocator* allocator_ = nullptr;
  guintptr (*phys_)(const GstMemory*) = nullptr;
  decltype(&gst_simaai_memory_get_segment) segment_ = nullptr;
  decltype(&simaai_memory_get_bus) bus_ = nullptr;
};

// Never destruct unknown-DMA sessions at static exit: freeing their camera
// allocations is unsafe. Recovery is an explicit device/kernel operation.
std::mutex& parked_mutex() {
  static auto* value = new std::mutex;
  return *value;
}
auto& parked_sessions() {
  static auto* value = new std::map<std::string, std::shared_ptr<raw::Capture>>;
  return *value;
}

struct SourceState {
  std::string device;
  std::string fourcc = "BA81";
  std::string name = "raw_src";
  guint width = 0, height = 0, count = 8;
  std::shared_ptr<LinuxBackend> backend;
  std::shared_ptr<raw::Capture> capture;
  std::atomic<bool> interrupted{false};
};

struct GstNeatRawCameraSource {
  GstPushSrc parent;
  SourceState* state;
};
struct GstNeatRawCameraSourceClass {
  GstPushSrcClass parent_class;
};
G_DEFINE_TYPE(GstNeatRawCameraSource, gst_neat_raw_camera_source, GST_TYPE_PUSH_SRC)

enum { PROP_ZERO, PROP_DEVICE, PROP_FOURCC, PROP_NAME, PROP_WIDTH, PROP_HEIGHT, PROP_COUNT };

void set_property(GObject* object, guint id, const GValue* value, GParamSpec* spec) {
  if (GST_STATE(object) > GST_STATE_READY) {
    GST_WARNING_OBJECT(object, "Raw camera properties are immutable while streaming");
    return;
  }
  auto& state = *reinterpret_cast<GstNeatRawCameraSource*>(object)->state;
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
  default:
    G_OBJECT_WARN_INVALID_PROPERTY_ID(object, id, spec);
  }
}
void get_property(GObject* object, guint id, GValue* value, GParamSpec* spec) {
  const auto& state = *reinterpret_cast<GstNeatRawCameraSource*>(object)->state;
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
  default:
    G_OBJECT_WARN_INVALID_PROPERTY_ID(object, id, spec);
  }
}

bool retire(SourceState& state) {
  if (!state.capture)
    return true;
  const bool safe = state.capture->stop();
  {
    std::lock_guard lock(parked_mutex());
    auto& sessions = parked_sessions();
    const auto found = sessions.find(state.device);
    if (safe && found != sessions.end() && found->second == state.capture)
      sessions.erase(found);
    else if (!safe)
      sessions.emplace(state.device, state.capture);
  }
  state.capture.reset();
  state.backend.reset();
  return safe;
}

gboolean start(GstBaseSrc* base) {
  auto* self = reinterpret_cast<GstNeatRawCameraSource*>(base);
  auto& state = *self->state;
  try {
    {
      std::lock_guard lock(parked_mutex());
      if (parked_sessions().count(state.device))
        throw std::runtime_error(
            "raw camera device is already owned or has parked DMA; do not automatically restart");
    }
    if (state.device.empty() || state.name.empty())
      throw std::runtime_error("raw camera requires a device and buffer-name");
    state.interrupted.store(false);
    state.backend = std::make_shared<LinuxBackend>();
    state.capture = std::make_shared<raw::Capture>(state.backend);
    {
      std::lock_guard lock(parked_mutex());
      if (!parked_sessions().emplace(state.device, state.capture).second)
        throw std::runtime_error("raw camera device is already owned by another source");
    }
    state.capture->start(state.device, state.name,
                         {state.width, state.height, raw::parse_fourcc(state.fourcc), 0, 0},
                         state.count);
    const auto& format = state.capture->format();
    GstCaps* caps = gst_caps_new_simple(
        "application/vnd.simaai.tensor", "representation", G_TYPE_STRING, "tensor-set", "storage",
        G_TYPE_STRING, "tensorbuffer", "capture-format", G_TYPE_STRING, "RAW_CAMERA_U8", "fourcc",
        G_TYPE_STRING, state.fourcc.c_str(), "dtype", G_TYPE_STRING, "UInt8", "layout",
        G_TYPE_STRING, "HW", "width", G_TYPE_INT, static_cast<gint>(format.width), "height",
        G_TYPE_INT, static_cast<gint>(format.height), "row-stride", G_TYPE_UINT, format.stride,
        nullptr);
    const bool accepted = gst_base_src_set_caps(base, caps);
    gst_caps_unref(caps);
    if (!accepted)
      throw std::runtime_error("raw camera tensor caps negotiation failed");
    return TRUE;
  } catch (const std::exception& error) {
    const bool safe = retire(state);
    GST_ELEMENT_ERROR(
        self, RESOURCE, FAILED, ("%s", error.what()),
        ("capture DMA retirement %s", safe ? "complete" : "UNKNOWN; allocations parked"));
    return FALSE;
  }
}
gboolean stop(GstBaseSrc* base) {
  auto* self = reinterpret_cast<GstNeatRawCameraSource*>(base);
  self->state->interrupted.store(true);
  const bool safe = retire(*self->state);
  if (!safe)
    GST_ELEMENT_ERROR(
        self, RESOURCE, FAILED, ("Raw camera DMA retirement is unknown"),
        ("Capture allocations and device fd are parked; do not automatically restart"));
  return safe;
}
gboolean unlock(GstBaseSrc* base) {
  reinterpret_cast<GstNeatRawCameraSource*>(base)->state->interrupted.store(true);
  return TRUE;
}
gboolean unlock_stop(GstBaseSrc* base) {
  reinterpret_cast<GstNeatRawCameraSource*>(base)->state->interrupted.store(false);
  return TRUE;
}

GstFlowReturn create(GstPushSrc* base, GstBuffer** result) {
  auto* self = reinterpret_cast<GstNeatRawCameraSource*>(base);
  auto& state = *self->state;
  GstBuffer* out = nullptr;
  try {
    const auto lease = state.capture->next(state.interrupted);
    if (!lease)
      return GST_FLOW_FLUSHING;
    const auto& allocation = state.capture->allocation(lease->frame.index);
    const auto owner = std::static_pointer_cast<MemoryOwner>(allocation.owner);
    state.backend->api->mark_device_written(owner->memory);
    GstMemory* view = state.backend->api->share_packed(owner->memory, state.name.c_str());
    if (!view)
      throw std::runtime_error("raw camera cannot create a retained SiMa memory view");
    // The lease is on memory, not only GstBuffer: downstream memory shares must
    // retain the camera queue slot even when the original buffer is destroyed.
    static const GQuark lease_key = g_quark_from_static_string("neat-raw-camera-capture-lease");
    gst_mini_object_set_qdata(
        GST_MINI_OBJECT(view), lease_key, new std::shared_ptr<raw::Lease>(lease),
        [](gpointer p) { delete static_cast<std::shared_ptr<raw::Lease>*>(p); });
    gst_memory_resize(view, 0, lease->frame.bytes);
    out = gst_buffer_new();
    gst_buffer_append_memory(out, view);
    timespec mono{};
    if (::clock_gettime(CLOCK_MONOTONIC, &mono) != 0)
      throw std::runtime_error("raw camera cannot read the capture clock");
    const auto now_mono = static_cast<guint64>(mono.tv_sec) * GST_SECOND + mono.tv_nsec;
    GstClock* clock = gst_element_get_clock(GST_ELEMENT(self));
    if (!clock)
      throw std::runtime_error("raw camera pipeline clock unavailable");
    const auto now = gst_clock_get_time(clock);
    gst_object_unref(clock);
    const auto base_time = gst_element_get_base_time(GST_ELEMENT(self));
    // Convert acquisition time into pipeline running time; preserve the raw
    // monotonic capture time separately rather than using uptime as media PTS.
    const auto age =
        now_mono >= lease->frame.timestamp_ns ? now_mono - lease->frame.timestamp_ns : 0;
    GST_BUFFER_PTS(out) = now >= base_time && now - base_time >= age ? now - base_time - age : 0;
    GST_BUFFER_OFFSET(out) = lease->frame.sequence;
    auto* meta = gst_buffer_add_custom_meta(out, "GstSimaMeta");
    if (!meta)
      throw std::runtime_error("raw camera GstSimaMeta unavailable");
    gst_structure_set(gst_custom_meta_get_structure(meta), "buffer-id", G_TYPE_INT64,
                      static_cast<gint64>(allocation.physical), "buffer-name", G_TYPE_STRING,
                      state.name.c_str(), "buffer-offset", G_TYPE_INT64, gint64{0}, "frame-id",
                      G_TYPE_INT64, static_cast<gint64>(lease->frame.sequence), "orig-input-seq",
                      G_TYPE_INT64, static_cast<gint64>(lease->frame.sequence), "timestamp",
                      G_TYPE_UINT64, GST_BUFFER_PTS(out), "capture-monotonic-ns", G_TYPE_UINT64,
                      lease->frame.timestamp_ns, "raw-capture-generation", G_TYPE_UINT64,
                      lease->frame.capture_generation, "origin_stage_id", G_TYPE_STRING,
                      state.name.c_str(), "origin_output_slot", G_TYPE_INT, 0, nullptr);
    SimaTensorDescriptorV2 descriptor{};
    descriptor.logical_index = descriptor.physical_index = descriptor.backend_output_index = 0;
    descriptor.route_slot = descriptor.memory_index = 0;
    descriptor.logical_name_id = descriptor.backend_name_id = descriptor.segment_name_id = 0;
    descriptor.byte_offset = 0;
    descriptor.size_bytes = lease->frame.bytes;
    descriptor.dtype = SIMA_TENSOR_SET_DTYPE_UINT8_V1;
    descriptor.layout = SIMA_TENSOR_SET_LAYOUT_HW_V1;
    descriptor.rank = 2;
    descriptor.shape[0] = state.height;
    descriptor.shape[1] = state.width;
    descriptor.stride_bytes[0] = state.capture->format().stride;
    descriptor.stride_bytes[1] = 1;
    const gchar* names[] = {state.name.c_str(), nullptr};
    char* error = nullptr;
    const bool attached = simaai::gst::sima_tensor_buffer_attach_meta_flat(
        out, state.name.c_str(), &descriptor, 1, names, nullptr, 0, nullptr, 0, &error);
    const std::string reason = error ? error : "raw camera tensor descriptor attachment failed";
    g_free(error);
    if (!attached)
      throw std::runtime_error(reason);
    if (lease->frame.discontinuity) {
      // A new capture session is a calibration boundary, not merely a frame
      // drop. The serialized EV worker applies this marker before dispatch.
      GST_BUFFER_FLAG_SET(out, GST_BUFFER_FLAG_DISCONT);
    }
    *result = out;
    return GST_FLOW_OK;
  } catch (const std::exception& error) {
    if (out)
      gst_buffer_unref(out);
    GST_ELEMENT_ERROR(self, RESOURCE, FAILED, ("%s", error.what()), ("raw capture stopped"));
    return GST_FLOW_ERROR;
  }
}
void finalize(GObject* object) {
  auto* self = reinterpret_cast<GstNeatRawCameraSource*>(object);
  retire(*self->state);
  delete self->state;
  G_OBJECT_CLASS(gst_neat_raw_camera_source_parent_class)->finalize(object);
}
void gst_neat_raw_camera_source_class_init(GstNeatRawCameraSourceClass* klass) {
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
      g_param_spec_string("fourcc", "Fourcc", "Exact raw UInt8 capture fourcc", "BA81", flags));
  g_object_class_install_property(object, PROP_NAME,
                                  g_param_spec_string("buffer-name", "Tensor name",
                                                      "Logical capture tensor name", "raw_src",
                                                      flags));
  g_object_class_install_property(
      object, PROP_WIDTH, g_param_spec_uint("width", "Width", "Wire width", 1, G_MAXINT, 1, flags));
  g_object_class_install_property(
      object, PROP_HEIGHT,
      g_param_spec_uint("height", "Height", "Wire height", 1, G_MAXINT, 1, flags));
  g_object_class_install_property(object, PROP_COUNT,
                                  g_param_spec_uint("capture-buffer-count", "Capture count",
                                                    "DMA capture queue depth", 4, 128, 8, flags));
  auto* element = GST_ELEMENT_CLASS(klass);
  gst_element_class_set_static_metadata(element, "Neat private raw camera source", "Source/Video",
                                        "Strict raw V4L2 DMA-BUF tensor capture", "SiMa.ai");
  GstCaps* caps =
      gst_caps_from_string("application/"
                           "vnd.simaai.tensor,representation=(string)tensor-set,storage=(string)"
                           "tensorbuffer,capture-format=(string)RAW_CAMERA_U8");
  gst_element_class_add_pad_template(
      element, gst_pad_template_new("src", GST_PAD_SRC, GST_PAD_ALWAYS, caps));
  gst_caps_unref(caps);
  auto* source = GST_BASE_SRC_CLASS(klass);
  source->start = start;
  source->stop = stop;
  source->unlock = unlock;
  source->unlock_stop = unlock_stop;
  GST_PUSH_SRC_CLASS(klass)->create = create;
}
void gst_neat_raw_camera_source_init(GstNeatRawCameraSource* self) {
  self->state = new SourceState;
  gst_base_src_set_live(GST_BASE_SRC(self), TRUE);
  gst_base_src_set_format(GST_BASE_SRC(self), GST_FORMAT_TIME);
}
} // namespace
namespace simaai::neat {
bool register_neat_raw_camera_source() {
  static std::once_flag once;
  static bool result = false;
  std::call_once(once, [] {
    result = gst_element_register(nullptr, "neatrawcamerasrc", GST_RANK_NONE,
                                  gst_neat_raw_camera_source_get_type());
  });
  return result;
}
} // namespace simaai::neat
