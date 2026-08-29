#include "gst/GstInit.h"
#include "gst/GstHelpers.h"
#include "nodes/common/Caps.h"
#include "nodes/common/Output.h"
#include "nodes/groups/VideoSender.h"
#include "nodes/io/Input.h"
#include "nodes/sima/SimaDecode.h"
#include "pipeline/EncodedSampleUtil.h"
#include "pipeline/Graph.h"
#include "pipeline/TensorAdapters.h"
#include "simaai/neat/internal/dmabuf/DmaBufPool.h"
#include "asset_utils.h"
#include "test_utils.h"

#include <gst/app/gstappsink.h>
#include <gst/app/gstappsrc.h>
#include <gst/gst.h>
#include <gst/video/video.h>

#include <algorithm>
#include <array>
#include <atomic>
#include <cerrno>
#include <chrono>
#include <cmath>
#include <condition_variable>
#include <cstdint>
#include <cstring>
#include <exception>
#include <filesystem>
#include <initializer_list>
#include <iostream>
#include <iterator>
#include <limits>
#include <memory>
#include <mutex>
#include <optional>
#include <set>
#include <sstream>
#include <stdexcept>
#include <string>
#include <string_view>
#include <thread>
#include <utility>
#include <vector>

#include <arpa/inet.h>
#include <netinet/in.h>
#include <sys/socket.h>
#include <sys/wait.h>
#include <unistd.h>

namespace {

using simaai::neat::FormatTag;
using simaai::neat::ImageSpec;
using simaai::neat::InputMemoryPolicy;
using simaai::neat::Sample;
using simaai::neat::Tensor;

struct TestGeometry {
  int width = 640;
  int height = 360;
};

TestGeometry g_geometry;
std::string g_encoder_profile = "baseline";
std::string g_encoder_level = "4.0";
std::string g_encoder_bitrate_mode;

int g_fps = 30;
int g_frames_per_scenario = 6;
int g_concurrency_channels = 1;
int g_concurrency_port_base = 0;
constexpr int kDefaultMinimumDecodedFrames = 3;
constexpr int kReceiverTimeoutMs = 15000;
constexpr int kScenarioSkippedExitCode = 77;
constexpr double kMinimumPsnrDb = 18.0;
constexpr std::uint8_t kPaddingSentinel = 0xD3;
constexpr std::string_view kDirectIngressKind = "VideoSenderRawIngress[direct_nv12]";
constexpr std::string_view kMaterializeIngressKind =
    "VideoSenderRawIngress[materialize_nv12]";
constexpr std::string_view kFallbackIngressKind = "VideoSenderRawIngress[convert_to_nv12]";
constexpr std::array<const char*, 21> kAggregateScenarioNames{{
    "system_nv12_tight",
    "system_nv12_padded",
    "auto_nv12",
    "ev74_nv12",
    "public_h265_nv12",
    "system_i420_tight",
    "system_i420_padded",
    "system_rgb",
    "system_bgr",
    "system_gray8",
    "plugin_padded_nv12_system",
    "plugin_padded_nv12_sima",
    "plugin_padded_nv12_sima_h265",
    "encoder_controls_h264",
    "encoder_controls_h265",
    "plugin_padded_i420_system",
    "plugin_padded_i420_sima",
    "encoded_h264_passthrough",
    "encoded_h265_passthrough",
    "native_h264_to_h264",
    "native_h265_to_h264",
}};

struct GstElementUnref {
  void operator()(GstElement* element) const {
    if (element) {
      gst_element_set_state(element, GST_STATE_NULL);
      gst_object_unref(element);
    }
  }
};

struct GstSampleUnref {
  void operator()(GstSample* sample) const {
    if (sample) {
      gst_sample_unref(sample);
    }
  }
};

struct GstObjectUnref {
  template <typename T> void operator()(T* object) const {
    if (object) {
      gst_object_unref(object);
    }
  }
};

using GstElementPtr = std::unique_ptr<GstElement, GstElementUnref>;
using GstSamplePtr = std::unique_ptr<GstSample, GstSampleUnref>;
template <typename T> using GstObjectPtr = std::unique_ptr<T, GstObjectUnref>;

class ScopedFileRemoval {
public:
  explicit ScopedFileRemoval(std::filesystem::path path) : path_(std::move(path)) {}
  ~ScopedFileRemoval() {
    std::error_code ignored;
    std::filesystem::remove(path_, ignored);
  }

  ScopedFileRemoval(const ScopedFileRemoval&) = delete;
  ScopedFileRemoval& operator=(const ScopedFileRemoval&) = delete;

private:
  std::filesystem::path path_;
};

struct RawFrame {
  FormatTag format = FormatTag::Auto;
  int width = 0;
  int height = 0;
  std::vector<std::uint8_t> bytes;
};

enum class Topology {
  Linear,
  Connected,
  Fanout,
};

struct RawScenario {
  const char* name;
  FormatTag format;
  InputMemoryPolicy memory;
  int row_padding;
  std::string_view expected_ingress;
  Topology topology = Topology::Linear;
  bool save_load = false;
  bool h265 = false;
};

struct MappedPlaneLayout {
  std::array<std::size_t, GST_VIDEO_MAX_PLANES> offsets{};
  std::array<int, GST_VIDEO_MAX_PLANES> strides{};
  std::array<int, GST_VIDEO_MAX_PLANES> row_bytes{};
  std::array<int, GST_VIDEO_MAX_PLANES> rows{};
  guint plane_count = 0;
  std::size_t total_bytes = 0;
};

std::string gst_format(FormatTag format) {
  switch (format) {
  case FormatTag::RGB:
    return "RGB";
  case FormatTag::BGR:
    return "BGR";
  case FormatTag::GRAY8:
    return "GRAY8";
  case FormatTag::NV12:
    return "NV12";
  case FormatTag::I420:
    return "I420";
  default:
    throw std::invalid_argument("unsupported raw test format");
  }
}

ImageSpec::PixelFormat pixel_format(FormatTag format) {
  switch (format) {
  case FormatTag::RGB:
    return ImageSpec::PixelFormat::RGB;
  case FormatTag::BGR:
    return ImageSpec::PixelFormat::BGR;
  case FormatTag::GRAY8:
    return ImageSpec::PixelFormat::GRAY8;
  case FormatTag::NV12:
    return ImageSpec::PixelFormat::NV12;
  case FormatTag::I420:
    return ImageSpec::PixelFormat::I420;
  default:
    return ImageSpec::PixelFormat::UNKNOWN;
  }
}

GstVideoFormat video_format(FormatTag format) {
  switch (format) {
  case FormatTag::RGB:
    return GST_VIDEO_FORMAT_RGB;
  case FormatTag::BGR:
    return GST_VIDEO_FORMAT_BGR;
  case FormatTag::GRAY8:
    return GST_VIDEO_FORMAT_GRAY8;
  case FormatTag::NV12:
    return GST_VIDEO_FORMAT_NV12;
  case FormatTag::I420:
    return GST_VIDEO_FORMAT_I420;
  default:
    return GST_VIDEO_FORMAT_UNKNOWN;
  }
}

std::string raw_caps(FormatTag format, int width = -1, int height = -1) {
  if (width <= 0) {
    width = g_geometry.width;
  }
  if (height <= 0) {
    height = g_geometry.height;
  }
  std::ostringstream caps;
  caps << "video/x-raw,format=(string)" << gst_format(format) << ",width=(int)" << width
       << ",height=(int)" << height << ",framerate=(fraction)" << g_fps << "/1";
  return caps.str();
}

bool encoder_supports_layout_aware_input() {
  GstObjectPtr<GstElementFactory> factory(gst_element_factory_find("neatencoder"));
  require(factory != nullptr, "required GStreamer factory is unavailable: neatencoder");
  GstObjectPtr<GstPluginFeature> loaded(gst_plugin_feature_load(GST_PLUGIN_FEATURE(factory.get())));
  require(loaded != nullptr, "failed to load the neatencoder plugin");

  GstElementPtr encoder(gst_element_factory_create(GST_ELEMENT_FACTORY(loaded.get()), nullptr));
  require(encoder != nullptr, "failed to instantiate neatencoder");
  GParamSpec* capability =
      g_object_class_find_property(G_OBJECT_GET_CLASS(encoder.get()), "input-layout-aware");

  std::string plugin_path = "<unknown>";
  if (GstObjectPtr<GstPlugin> plugin(gst_plugin_feature_get_plugin(loaded.get())); plugin) {
    if (const gchar* filename = gst_plugin_get_filename(plugin.get())) {
      plugin_path = filename;
    }
  }

  // Core must remain compatible with the pre-fix plugin: absence of the
  // capability selects the converter fallback rather than failing the run.
  if (capability == nullptr) {
    std::cout << "[INFO] neatencoder=" << plugin_path
              << " input-layout-aware=<absent>; testing compatibility fallback\n";
    return false;
  }
  require(G_PARAM_SPEC_VALUE_TYPE(capability) == G_TYPE_BOOLEAN &&
              (capability->flags & G_PARAM_READABLE) != 0,
          "neatencoder input-layout-aware capability is not a readable boolean");
  gboolean layout_aware = FALSE;
  g_object_get(encoder.get(), "input-layout-aware", &layout_aware, nullptr);
  std::cout << "[INFO] neatencoder=" << plugin_path
            << " input-layout-aware=" << (layout_aware ? "true" : "false") << "\n";
  return layout_aware;
}

GstElementPtr parse_pipeline(const std::string& launch, const std::string& context) {
  GError* error = nullptr;
  GstElement* pipeline = gst_parse_launch(launch.c_str(), &error);
  if (!pipeline) {
    const std::string detail = error && error->message ? error->message : "unknown parse error";
    if (error) {
      g_error_free(error);
    }
    throw std::runtime_error(context + ": failed to create pipeline: " + detail);
  }
  if (error) {
    g_error_free(error);
  }
  return GstElementPtr(pipeline);
}

GstElement* required_element(GstElement* pipeline, const char* name, const std::string& context) {
  GstElement* element = gst_bin_get_by_name(GST_BIN(pipeline), name);
  if (!element) {
    throw std::runtime_error(context + ": missing element " + name);
  }
  return element;
}

void start_pipeline(GstElement* pipeline, const std::string& context) {
  require(gst_element_set_state(pipeline, GST_STATE_PLAYING) != GST_STATE_CHANGE_FAILURE,
          context + ": failed to enter PLAYING");
  GstState state = GST_STATE_NULL;
  GstState pending = GST_STATE_NULL;
  const GstStateChangeReturn result =
      gst_element_get_state(pipeline, &state, &pending, 3 * GST_SECOND);
  require(result != GST_STATE_CHANGE_FAILURE, context + ": asynchronous state change failed");
}

std::string bus_error(GstElement* pipeline) {
  GstBus* bus = gst_element_get_bus(pipeline);
  if (!bus) {
    return {};
  }
  GstMessage* message = gst_bus_pop_filtered(bus, GST_MESSAGE_ERROR);
  gst_object_unref(bus);
  if (!message) {
    return {};
  }

  GError* error = nullptr;
  gchar* debug = nullptr;
  gst_message_parse_error(message, &error, &debug);
  std::string text = error && error->message ? error->message : "unknown GStreamer error";
  if (debug && *debug) {
    text += " (" + std::string(debug) + ")";
  }
  if (error) {
    g_error_free(error);
  }
  g_free(debug);
  gst_message_unref(message);
  return text;
}

MappedPlaneLayout tight_layout(FormatTag format, int width, int height, int row_padding = 0) {
  MappedPlaneLayout layout;
  switch (format) {
  case FormatTag::RGB:
  case FormatTag::BGR:
    layout.plane_count = 1;
    layout.row_bytes[0] = width * 3;
    layout.rows[0] = height;
    break;
  case FormatTag::GRAY8:
    layout.plane_count = 1;
    layout.row_bytes[0] = width;
    layout.rows[0] = height;
    break;
  case FormatTag::NV12:
    layout.plane_count = 2;
    layout.row_bytes[0] = width;
    layout.row_bytes[1] = width;
    layout.rows[0] = height;
    layout.rows[1] = height / 2;
    break;
  case FormatTag::I420:
    layout.plane_count = 3;
    layout.row_bytes[0] = width;
    layout.row_bytes[1] = width / 2;
    layout.row_bytes[2] = width / 2;
    layout.rows[0] = height;
    layout.rows[1] = height / 2;
    layout.rows[2] = height / 2;
    break;
  default:
    throw std::invalid_argument("unsupported raw test layout");
  }

  std::size_t cursor = 0;
  for (guint plane = 0; plane < layout.plane_count; ++plane) {
    const int plane_padding =
        (format == FormatTag::I420 && plane > 0) ? row_padding / 2 : row_padding;
    layout.offsets[plane] = cursor;
    layout.strides[plane] = layout.row_bytes[plane] + plane_padding;
    cursor += static_cast<std::size_t>(layout.strides[plane]) *
              static_cast<std::size_t>(layout.rows[plane]);
  }
  layout.total_bytes = cursor;
  return layout;
}

void add_video_meta(GstBuffer* buffer, FormatTag format, int width, int height,
                    const MappedPlaneLayout& layout, const std::string& context) {
  gsize offsets[GST_VIDEO_MAX_PLANES] = {};
  gint strides[GST_VIDEO_MAX_PLANES] = {};
  for (guint plane = 0; plane < layout.plane_count; ++plane) {
    offsets[plane] = layout.offsets[plane];
    strides[plane] = layout.strides[plane];
  }
  require(gst_buffer_add_video_meta_full(buffer, GST_VIDEO_FRAME_FLAG_NONE, video_format(format),
                                         static_cast<guint>(width), static_cast<guint>(height),
                                         layout.plane_count, offsets, strides) != nullptr,
          context + ": failed to attach GstVideoMeta");
}

RawFrame tight_frame_from_sample(GstSample* sample, FormatTag expected,
                                 const std::string& context) {
  require(sample != nullptr, context + ": missing sample");
  GstCaps* caps = gst_sample_get_caps(sample);
  GstBuffer* buffer = gst_sample_get_buffer(sample);
  require(caps != nullptr && buffer != nullptr, context + ": incomplete sample");

  GstVideoInfo info{};
  require(gst_video_info_from_caps(&info, caps), context + ": invalid video caps");
  require(GST_VIDEO_INFO_FORMAT(&info) == video_format(expected),
          context + ": unexpected output format");

  GstVideoFrame frame{};
  require(gst_video_frame_map(&frame, &info, buffer, GST_MAP_READ),
          context + ": failed to map video frame");

  RawFrame out;
  out.format = expected;
  out.width = GST_VIDEO_INFO_WIDTH(&info);
  out.height = GST_VIDEO_INFO_HEIGHT(&info);
  const MappedPlaneLayout layout = tight_layout(expected, out.width, out.height);
  out.bytes.resize(layout.total_bytes);
  for (guint plane = 0; plane < layout.plane_count; ++plane) {
    const auto* source =
        static_cast<const std::uint8_t*>(GST_VIDEO_FRAME_PLANE_DATA(&frame, plane));
    const int source_stride = GST_VIDEO_FRAME_PLANE_STRIDE(&frame, plane);
    for (int row = 0; row < layout.rows[plane]; ++row) {
      std::memcpy(out.bytes.data() + layout.offsets[plane] +
                      static_cast<std::size_t>(row * layout.strides[plane]),
                  source + static_cast<std::size_t>(row * source_stride),
                  static_cast<std::size_t>(layout.row_bytes[plane]));
    }
  }
  gst_video_frame_unmap(&frame);
  return out;
}

RawFrame load_real_image(const std::string& path) {
  const std::string context = "real-image fixture";
  auto pipeline = parse_pipeline("filesrc name=source ! jpegdec ! videoconvert ! videoscale ! "
                                 "video/x-raw,format=RGB,width=" +
                                     std::to_string(g_geometry.width) +
                                     ",height=" + std::to_string(g_geometry.height) +
                                     ",pixel-aspect-ratio=1/1 ! "
                                     "appsink name=sink sync=false max-buffers=1 drop=false",
                                 context);
  GstElement* source = required_element(pipeline.get(), "source", context);
  GstElement* sink = required_element(pipeline.get(), "sink", context);
  g_object_set(source, "location", path.c_str(), nullptr);
  gst_object_unref(source);

  start_pipeline(pipeline.get(), context);
  GstSamplePtr sample(
      gst_app_sink_try_pull_sample(GST_APP_SINK(sink), static_cast<GstClockTime>(5 * GST_SECOND)));
  gst_object_unref(sink);
  if (!sample) {
    const std::string error = bus_error(pipeline.get());
    throw std::runtime_error(context + ": no decoded frame" +
                             (error.empty() ? std::string{} : ": " + error));
  }
  return tight_frame_from_sample(sample.get(), FormatTag::RGB, context);
}

RawFrame convert_frame(const RawFrame& input, FormatTag output_format) {
  const std::string context =
      "convert " + gst_format(input.format) + " to " + gst_format(output_format);
  auto pipeline = parse_pipeline(
      "appsrc name=source is-live=false format=time block=true caps=\"" +
          raw_caps(input.format, input.width, input.height) +
          "\" ! videoconvert ! video/x-raw,format=" + gst_format(output_format) +
          ",width=" + std::to_string(input.width) + ",height=" + std::to_string(input.height) +
          " ! appsink name=sink sync=false max-buffers=1 drop=false",
      context);
  GstElement* source = required_element(pipeline.get(), "source", context);
  GstElement* sink = required_element(pipeline.get(), "sink", context);
  start_pipeline(pipeline.get(), context);

  const MappedPlaneLayout input_layout = tight_layout(input.format, input.width, input.height);
  require(input.bytes.size() == input_layout.total_bytes,
          context + ": input does not match its tight video layout");
  GstBuffer* buffer = gst_buffer_new_allocate(nullptr, input.bytes.size(), nullptr);
  require(buffer != nullptr, context + ": failed to allocate input buffer");
  gst_buffer_fill(buffer, 0, input.bytes.data(), input.bytes.size());
  add_video_meta(buffer, input.format, input.width, input.height, input_layout, context);
  GST_BUFFER_PTS(buffer) = 0;
  GST_BUFFER_DURATION(buffer) = GST_SECOND / g_fps;
  require(gst_app_src_push_buffer(GST_APP_SRC(source), buffer) == GST_FLOW_OK,
          context + ": appsrc rejected input");
  require(gst_app_src_end_of_stream(GST_APP_SRC(source)) == GST_FLOW_OK,
          context + ": failed to send EOS");

  GstSamplePtr sample(
      gst_app_sink_try_pull_sample(GST_APP_SINK(sink), static_cast<GstClockTime>(5 * GST_SECOND)));
  gst_object_unref(source);
  gst_object_unref(sink);
  if (!sample) {
    const std::string error = bus_error(pipeline.get());
    throw std::runtime_error(context + ": no converted frame" +
                             (error.empty() ? std::string{} : ": " + error));
  }
  return tight_frame_from_sample(sample.get(), output_format, context);
}

Tensor tensor_from_frame(const RawFrame& frame, int row_padding,
                         GstAllocator* allocator = nullptr) {
  const MappedPlaneLayout source_layout = tight_layout(frame.format, frame.width, frame.height);
  const MappedPlaneLayout destination_layout =
      tight_layout(frame.format, frame.width, frame.height, row_padding);

  GstBuffer* buffer = gst_buffer_new_allocate(
      allocator, static_cast<gsize>(destination_layout.total_bytes), nullptr);
  require(buffer != nullptr, "failed to allocate real-frame GstBuffer");
  GstMapInfo map{};
  require(gst_buffer_map(buffer, &map, GST_MAP_WRITE), "failed to map real-frame GstBuffer");
  std::memset(map.data, kPaddingSentinel, map.size);
  for (guint plane = 0; plane < destination_layout.plane_count; ++plane) {
    for (int row = 0; row < destination_layout.rows[plane]; ++row) {
      std::memcpy(map.data + destination_layout.offsets[plane] +
                      static_cast<std::size_t>(row * destination_layout.strides[plane]),
                  frame.bytes.data() + source_layout.offsets[plane] +
                      static_cast<std::size_t>(row * source_layout.strides[plane]),
                  static_cast<std::size_t>(destination_layout.row_bytes[plane]));
    }
  }
  gst_buffer_unmap(buffer, &map);

  add_video_meta(buffer, frame.format, frame.width, frame.height, destination_layout,
                 "real-frame buffer");

  GstCaps* caps = gst_caps_from_string(raw_caps(frame.format, frame.width, frame.height).c_str());
  require(caps != nullptr, "failed to create real-frame caps");
  GstSample* sample = gst_sample_new(buffer, caps, nullptr, nullptr);
  gst_caps_unref(caps);
  gst_buffer_unref(buffer);
  require(sample != nullptr, "failed to create real-frame GstSample");
  Tensor tensor = simaai::neat::from_gst_sample(sample);
  gst_sample_unref(sample);
  require(tensor.semantic.image.has_value() &&
              tensor.semantic.image->format == pixel_format(frame.format),
          "GstSample adapter lost the real-frame pixel format");
  return tensor;
}

Tensor place_frame_for_input_policy(Tensor tensor, InputMemoryPolicy policy) {
  if (policy == InputMemoryPolicy::Ev74 || policy == InputMemoryPolicy::Auto) {
    // The DMA-BUF backend owns the one explicit ingress materialization.  The
    // encoder itself accepts only the resulting ordinary CMA GstDmaBufMemory;
    // it never exports legacy /dev/simaai-mem storage or stages the frame.
    // Auto with the default SiMa pool is an authoritative SimaAI-memory input
    // contract, so this externally supplied test frame must honor the same
    // contract as an explicit EV74 input instead of masquerading as system
    // memory behind a direct ingress.
    return tensor.cvu();
  }
  return tensor;
}

std::uint64_t fnv1a(const Tensor& tensor) {
  require(tensor.storage != nullptr && tensor.storage->holder != nullptr,
          "cannot hash tensor without GstSample storage");
  auto* sample = static_cast<GstSample*>(tensor.storage->holder.get());
  GstBuffer* buffer = sample ? gst_sample_get_buffer(sample) : nullptr;
  require(buffer != nullptr, "cannot hash tensor without GstBuffer");
  GstMapInfo map{};
  require(gst_buffer_map(buffer, &map, GST_MAP_READ), "failed to map tensor for hashing");
  std::uint64_t hash = 14695981039346656037ULL;
  for (gsize index = 0; index < map.size; ++index) {
    hash ^= map.data[index];
    hash *= 1099511628211ULL;
  }
  gst_buffer_unmap(buffer, &map);
  return hash;
}

int choose_udp_port() {
  const int fd = ::socket(AF_INET, SOCK_DGRAM, 0);
  require(fd >= 0, "failed to create UDP port-selection socket");
  sockaddr_in address{};
  address.sin_family = AF_INET;
  address.sin_addr.s_addr = htonl(INADDR_LOOPBACK);
  address.sin_port = 0;
  if (::bind(fd, reinterpret_cast<const sockaddr*>(&address), sizeof(address)) != 0) {
    ::close(fd);
    throw std::runtime_error("failed to reserve UDP port");
  }
  socklen_t size = sizeof(address);
  if (::getsockname(fd, reinterpret_cast<sockaddr*>(&address), &size) != 0) {
    ::close(fd);
    throw std::runtime_error("failed to inspect UDP port");
  }
  const int port = ntohs(address.sin_port);
  ::close(fd);
  return port;
}

struct EncodedCodec {
  const char* name;
  const char* media_type;
  const char* rtp_encoding_name;
  const char* parser;
  const char* depayloader;
  FormatTag format;
  simaai::neat::nodes::groups::RtspCodec sender_codec;
  simaai::neat::SimaDecodeType decoder_type;
  int payload_type;
};

constexpr EncodedCodec kH264{
    .name = "H.264",
    .media_type = "video/x-h264",
    .rtp_encoding_name = "H264",
    .parser = "h264parse",
    .depayloader = "rtph264depay",
    .format = FormatTag::H264,
    .sender_codec = simaai::neat::nodes::groups::RtspCodec::H264,
    .decoder_type = simaai::neat::SimaDecodeType::H264,
    .payload_type = 96,
};

constexpr EncodedCodec kH265{
    .name = "H.265",
    .media_type = "video/x-h265",
    .rtp_encoding_name = "H265",
    .parser = "h265parse",
    .depayloader = "rtph265depay",
    .format = FormatTag::H265,
    .sender_codec = simaai::neat::nodes::groups::RtspCodec::H265,
    .decoder_type = simaai::neat::SimaDecodeType::H265,
    .payload_type = 98,
};

std::string first_available_element(std::initializer_list<const char*> candidates) {
  for (const char* candidate : candidates) {
    if (simaai::neat::element_exists(candidate)) {
      return candidate;
    }
  }
  return {};
}

std::string software_decoder(const EncodedCodec& codec) {
  if (codec.format == FormatTag::H264) {
    return first_available_element({"avdec_h264", "openh264dec"});
  }
  return first_available_element({"avdec_h265", "libde265dec"});
}

std::string software_encoder(const EncodedCodec& codec) {
  if (codec.format == FormatTag::H264) {
    if (simaai::neat::element_exists("x264enc")) {
      // All-IDR baseline keeps the native decoder's DPB empty enough for this
      // short fixture to produce output without an EOS-only flush.
      return "x264enc tune=zerolatency speed-preset=ultrafast key-int-max=1 byte-stream=true";
    }
    return first_available_element({"openh264enc", "avenc_h264"});
  }
  if (simaai::neat::element_exists("x265enc")) {
    return "x265enc tune=zerolatency speed-preset=ultrafast";
  }
  return first_available_element({"avenc_h265"});
}

std::string encoded_caps(const EncodedCodec& codec) {
  return std::string(codec.media_type) +
         ",parsed=(boolean)true,stream-format=(string)byte-stream,alignment=(string)au";
}

class RtpReceiver {
public:
  RtpReceiver(int port, const EncodedCodec& codec)
      : context_(codec.name + std::string(" RTP receiver")),
        decoder_may_reorder_(codec.format == FormatTag::H265) {
    const std::string decoder = software_decoder(codec);
    require(!decoder.empty(), context_ + ": no software decoder is installed");
    pipeline_ = parse_pipeline(
        // This receiver is the qualification oracle for up to 54 simultaneous
        // encoder sessions, not a 25 ms network-latency benchmark.  With the
        // old 25 ms jitter window, all RTP datagrams reached loopback but the
        // software decoder pipelines classified most synchronized 720p
        // packets as late once 32 channels were active.  A one-second bounded
        // window preserves every channel's exact RTP ordering while keeping
        // the encoder load and product VideoSender path unchanged.  The large
        // socket buffer is bounded by the board's configured rmem_max and
        // prevents the test oracle, rather than the codec, from becoming the
        // qualification bottleneck.
        "udpsrc name=source buffer-size=33554432 port=" + std::to_string(port) +
            " caps=\"application/x-rtp,media=(string)video,encoding-name=(string)" +
            codec.rtp_encoding_name + ",payload=(int)" + std::to_string(codec.payload_type) +
            ",clock-rate=(int)90000\" ! rtpjitterbuffer latency=1000 drop-on-latency=false ! " +
            codec.depayloader + " ! " +
            codec.parser + " ! " + decoder +
            " ! videoconvert ! "
            "video/x-raw,format=RGB,width=" +
            std::to_string(g_geometry.width) + ",height=" + std::to_string(g_geometry.height) +
            " ! appsink name=sink sync=false max-buffers=32 drop=false",
        context_);
    source_.reset(required_element(pipeline_.get(), "source", context_));
    sink_.reset(required_element(pipeline_.get(), "sink", context_));
    GstObjectPtr<GstPad> source_pad(
        gst_element_get_static_pad(source_.get(), "src"));
    require(source_pad != nullptr, context_ + ": UDP source has no src pad");
    require(gst_pad_add_probe(source_pad.get(), GST_PAD_PROBE_TYPE_BUFFER,
                              &RtpReceiver::count_rtp_marker, this, nullptr) != 0,
            context_ + ": failed to install RTP access-unit counter");
    start_pipeline(pipeline_.get(), context_);
    std::this_thread::sleep_for(std::chrono::milliseconds(100));
  }

  std::vector<RawFrame> pull_frames(int maximum, bool require_all = false) {
    std::vector<RawFrame> frames;
    std::optional<GstClockTime> previous_pts;
    const auto deadline =
        std::chrono::steady_clock::now() + std::chrono::milliseconds(kReceiverTimeoutMs);
    while (static_cast<int>(frames.size()) < maximum &&
           std::chrono::steady_clock::now() < deadline) {
      GstSamplePtr sample(
          gst_app_sink_try_pull_sample(GST_APP_SINK(sink_.get()), 250 * GST_MSECOND));
      if (!sample) {
        const std::string error = bus_error(pipeline_.get());
        if (!error.empty()) {
          throw std::runtime_error(context_ + ": " + error);
        }
        if (require_all && decoder_may_reorder_ &&
            completed_access_units() >= maximum &&
            frames.size() >= static_cast<std::size_t>(
                                 std::min(maximum, kDefaultMinimumDecodedFrames)))
          break;
        continue;
      }
      GstBuffer* buffer = gst_sample_get_buffer(sample.get());
      require(buffer != nullptr, context_ + ": sample has no buffer");
      const GstClockTime pts = GST_BUFFER_PTS(buffer);
      require(pts != GST_CLOCK_TIME_NONE, context_ + ": decoded frame has no PTS");
      require(!previous_pts.has_value() || pts > *previous_pts,
              context_ + ": decoded frame PTS is not strictly increasing");
      previous_pts = pts;
      frames.push_back(tight_frame_from_sample(sample.get(), FormatTag::RGB, context_));
      // An RTP marker proves a complete encoded access unit crossed the
      // product VideoSender path. H.265 software decoders retain their final
      // reorder pictures indefinitely because UDP has no EOS event, so exact
      // concurrency accounting is the marker count while decoded RGB remains
      // the independent quality oracle. H.264 remains fully decoded below.
      if (require_all && decoder_may_reorder_ &&
          completed_access_units() >= maximum &&
          frames.size() >= static_cast<std::size_t>(
                               std::min(maximum, kDefaultMinimumDecodedFrames)))
        break;
      if (!require_all &&
          frames.size() >= static_cast<std::size_t>(
                               std::min(maximum, std::min(kDefaultMinimumDecodedFrames,
                                                         g_frames_per_scenario)))) {
        break;
      }
    }
    return frames;
  }

  int completed_access_units() const {
    return completed_access_units_.load(std::memory_order_relaxed);
  }

private:
  static GstPadProbeReturn count_rtp_marker(GstPad*, GstPadProbeInfo* info,
                                            gpointer user_data) {
    auto* self = static_cast<RtpReceiver*>(user_data);
    GstBuffer* buffer = GST_PAD_PROBE_INFO_BUFFER(info);
    if (!self || !buffer)
      return GST_PAD_PROBE_OK;
    GstMapInfo map{};
    if (gst_buffer_map(buffer, &map, GST_MAP_READ)) {
      // RFC 3550: RTP version is the high two bits of byte 0; the marker is
      // the high bit of byte 1. Both Allegro payloader routes set it exactly
      // on the final packet of each H.264/H.265 access unit.
      if (map.size >= 2 && (map.data[0] >> 6U) == 2U &&
          (map.data[1] & 0x80U) != 0)
        self->completed_access_units_.fetch_add(1,
                                                std::memory_order_relaxed);
      gst_buffer_unmap(buffer, &map);
    }
    return GST_PAD_PROBE_OK;
  }

  std::string context_;
  bool decoder_may_reorder_ = false;
  std::atomic<int> completed_access_units_{0};
  GstElementPtr pipeline_;
  GstObjectPtr<GstElement> source_;
  GstObjectPtr<GstElement> sink_;
};

double psnr(const RawFrame& expected, const RawFrame& actual) {
  require(expected.format == FormatTag::RGB && actual.format == FormatTag::RGB,
          "PSNR requires RGB frames");
  require(expected.width == actual.width && expected.height == actual.height &&
              expected.bytes.size() == actual.bytes.size(),
          "PSNR frame shape mismatch");
  double squared_error = 0.0;
  for (std::size_t index = 0; index < expected.bytes.size(); ++index) {
    const int difference =
        static_cast<int>(expected.bytes[index]) - static_cast<int>(actual.bytes[index]);
    squared_error += static_cast<double>(difference * difference);
  }
  const double mse = squared_error / static_cast<double>(expected.bytes.size());
  return mse <= 1e-12 ? 100.0 : 10.0 * std::log10((255.0 * 255.0) / mse);
}

double require_frame_quality(const std::string& name, const RawFrame& expected,
                             const std::vector<RawFrame>& decoded) {
  double minimum_quality = 100.0;
  for (std::size_t frame_index = 0; frame_index < decoded.size(); ++frame_index) {
    const double quality = psnr(expected, decoded[frame_index]);
    require(quality >= kMinimumPsnrDb, name + ": decoded frame " + std::to_string(frame_index) +
                                           " PSNR too low: " + std::to_string(quality) + " dB");
    minimum_quality = std::min(minimum_quality, quality);
  }
  return minimum_quality;
}

std::vector<std::uint8_t> copy_buffer_bytes(GstBuffer* buffer, const std::string& context) {
  require(buffer != nullptr, context + ": encoded sample has no buffer");
  GstMapInfo map{};
  require(gst_buffer_map(buffer, &map, GST_MAP_READ), context + ": failed to map encoded buffer");
  std::vector<std::uint8_t> bytes(map.data, map.data + map.size);
  gst_buffer_unmap(buffer, &map);
  require(!bytes.empty(), context + ": encoder produced an empty access unit");
  return bytes;
}

std::vector<Sample> generate_access_units(const EncodedCodec& codec, const RawFrame& rgb) {
  const std::string context = std::string(codec.name) + " software fixture encoder";
  const std::string encoder = software_encoder(codec);
  require(!encoder.empty(), context + ": no supported software encoder is installed");
  require(simaai::neat::element_exists(codec.parser), context + ": parser is not installed");

  auto pipeline = parse_pipeline(
      "appsrc name=source is-live=false format=time block=true caps=\"" +
          raw_caps(FormatTag::RGB, rgb.width, rgb.height) +
          "\" ! videoconvert ! video/x-raw,format=I420 ! " + encoder + " ! " + codec.parser +
          " disable-passthrough=true config-interval=-1 ! capsfilter caps=\"" +
          encoded_caps(codec) + "\" ! appsink name=sink sync=false max-buffers=32 drop=false",
      context);
  GstElement* source = required_element(pipeline.get(), "source", context);
  GstElement* sink = required_element(pipeline.get(), "sink", context);
  start_pipeline(pipeline.get(), context);

  const MappedPlaneLayout rgb_layout = tight_layout(FormatTag::RGB, rgb.width, rgb.height);
  require(rgb.bytes.size() == rgb_layout.total_bytes,
          context + ": input does not match its tight video layout");
  for (int frame_index = 0; frame_index < g_frames_per_scenario; ++frame_index) {
    GstBuffer* buffer = gst_buffer_new_allocate(nullptr, rgb.bytes.size(), nullptr);
    require(buffer != nullptr, context + ": failed to allocate raw input");
    gst_buffer_fill(buffer, 0, rgb.bytes.data(), rgb.bytes.size());
    add_video_meta(buffer, FormatTag::RGB, rgb.width, rgb.height, rgb_layout, context);
    GST_BUFFER_PTS(buffer) = static_cast<GstClockTime>(frame_index) * GST_SECOND / g_fps;
    GST_BUFFER_DTS(buffer) = GST_BUFFER_PTS(buffer);
    GST_BUFFER_DURATION(buffer) = GST_SECOND / g_fps;
    require(gst_app_src_push_buffer(GST_APP_SRC(source), buffer) == GST_FLOW_OK,
            context + ": appsrc rejected the real-image frame");
  }
  require(gst_app_src_end_of_stream(GST_APP_SRC(source)) == GST_FLOW_OK,
          context + ": failed to send EOS");

  std::vector<Sample> access_units;
  access_units.reserve(g_frames_per_scenario);
  const auto deadline =
      std::chrono::steady_clock::now() + std::chrono::milliseconds(kReceiverTimeoutMs);
  while (access_units.size() < static_cast<std::size_t>(g_frames_per_scenario) &&
         std::chrono::steady_clock::now() < deadline) {
    GstSamplePtr encoded(gst_app_sink_try_pull_sample(GST_APP_SINK(sink), 250 * GST_MSECOND));
    if (!encoded) {
      const std::string error = bus_error(pipeline.get());
      if (!error.empty()) {
        gst_object_unref(source);
        gst_object_unref(sink);
        throw std::runtime_error(context + ": " + error);
      }
      if (gst_app_sink_is_eos(GST_APP_SINK(sink))) {
        break;
      }
      continue;
    }

    const std::int64_t frame_index = static_cast<std::int64_t>(access_units.size());
    const std::int64_t duration_ns = 1000000000LL / g_fps;
    Sample sample = simaai::neat::make_encoded_sample(
        copy_buffer_bytes(gst_sample_get_buffer(encoded.get()), context), encoded_caps(codec),
        frame_index * duration_ns, frame_index * duration_ns, duration_ns);
    sample.frame_id = frame_index;
    sample.stream_id = std::string("software-") + codec.name;
    access_units.push_back(std::move(sample));
  }
  gst_object_unref(source);
  gst_object_unref(sink);

  require(access_units.size() == static_cast<std::size_t>(g_frames_per_scenario),
          context + ": expected " + std::to_string(g_frames_per_scenario) +
              " access units, received " + std::to_string(access_units.size()));
  return access_units;
}

void run_encoded_sender_scenario(const std::string& name, const EncodedCodec& codec,
                                 const std::vector<Sample>& access_units,
                                 const RawFrame& expected_rgb, bool native_decode) {
  require(!access_units.empty(), name + ": no encoded access units");
  const int port =
      g_concurrency_port_base > 0 ? g_concurrency_port_base : choose_udp_port();
  RtpReceiver receiver(port, native_decode ? kH264 : codec);

  simaai::neat::InputOptions input;
  input.payload_type = simaai::neat::PayloadType::Encoded;
  input.format = codec.format;
  input.caps_override = encoded_caps(codec);
  input.block = true;
  input.pool_max_buffers = 12;
  input.memory_policy = InputMemoryPolicy::SystemMemory;

  simaai::neat::Graph graph(name);
  graph.add(simaai::neat::nodes::Input("encoded", std::move(input)));
  if (native_decode) {
    simaai::neat::SimaDecodeOptions decode;
    decode.type = codec.decoder_type;
    decode.out_format = FormatTag::NV12;
    decode.raw_output = true;
    decode.dec_width = g_geometry.width;
    decode.dec_height = g_geometry.height;
    decode.dec_fps = g_fps;
    graph.add(simaai::neat::nodes::SimaDecode(std::move(decode)));
    // Native decode contracts are deliberately hints. Pin the observed native
    // NV12 boundary so this E2E exercises the adaptive direct ingress too.
    graph.add(simaai::neat::nodes::CapsRaw("NV12", g_geometry.width, g_geometry.height, g_fps,
                                           simaai::neat::CapsMemory::Any));

    auto sender = simaai::neat::nodes::groups::VideoSenderOptions::H264RtpUdpFromRaw(
        g_geometry.width, g_geometry.height, g_fps);
    sender.encoder.bitrate_kbps = 4000;
    sender.encoder.profile = g_encoder_profile;
    sender.encoder.level = g_encoder_level;
    sender.host = "127.0.0.1";
    sender.video_port_base = port;
    graph.add(simaai::neat::nodes::groups::VideoSender(sender));
  } else {
    auto sender = simaai::neat::nodes::groups::VideoSenderOptions::Passthrough(codec.sender_codec);
    sender.host = "127.0.0.1";
    sender.video_port_base = port;
    graph.add(simaai::neat::nodes::groups::VideoSender(sender));
  }

  const std::string backend = graph.describe_backend(false);
  require((backend.find("neatdecoder") != std::string::npos) == native_decode,
          name + ": native decoder presence does not match the requested path");
  require((backend.find("neatencoder") != std::string::npos) == native_decode,
          name + ": native encoder presence does not match the requested path");
  if (!native_decode) {
    require_contains(backend, codec.parser, name + ": public encoded sender parser is missing");
  }

  simaai::neat::RunOptions options;
  options.queue_depth = 16;
  options.overflow_policy = simaai::neat::OverflowPolicy::Block;
  options.startup_preflight = false;
  simaai::neat::Run run = graph.build(Sample{access_units.front()}, options);
  // The build seed establishes the boundary contract; Run transports only
  // samples explicitly submitted through push().
  for (const Sample& access_unit : access_units) {
    require(run.push(Sample{access_unit}), name + ": push failed");
  }
  run.close_input();

  const std::vector<RawFrame> decoded = receiver.pull_frames(g_frames_per_scenario);
  require(decoded.size() >= static_cast<std::size_t>(std::min(kDefaultMinimumDecodedFrames, g_frames_per_scenario)),
          name + ": expected at least " + std::to_string(std::min(kDefaultMinimumDecodedFrames, g_frames_per_scenario)) +
              " decoded frames, received " + std::to_string(decoded.size()));
  const double quality = require_frame_quality(name, expected_rgb, decoded);
  run.stop();

  std::cout << "[PASS] " << name << " decoded=" << decoded.size() << " min_psnr_db=" << quality
            << "\n";
}

std::uint64_t fnv1a(GstBuffer* buffer) {
  require(buffer != nullptr, "cannot hash a null GstBuffer");
  GstMapInfo map{};
  require(gst_buffer_map(buffer, &map, GST_MAP_READ), "failed to map GstBuffer for hashing");
  std::uint64_t hash = 14695981039346656037ULL;
  for (gsize index = 0; index < map.size; ++index) {
    hash ^= map.data[index];
    hash *= 1099511628211ULL;
  }
  gst_buffer_unmap(buffer, &map);
  return hash;
}

int checked_align_up(int value, int alignment, const char* field) {
  require(value > 0 && alignment > 0, std::string(field) + " must be positive");
  const int remainder = value % alignment;
  if (remainder == 0) {
    return value;
  }
  const int increment = alignment - remainder;
  require(value <= std::numeric_limits<int>::max() - increment,
          std::string(field) + " alignment overflow");
  return value + increment;
}

std::size_t plane_end(const MappedPlaneLayout& layout, guint plane) {
  require(plane < layout.plane_count, "plane index exceeds the mapped layout");
  require(layout.rows[plane] > 0 && layout.row_bytes[plane] > 0 &&
              layout.strides[plane] >= layout.row_bytes[plane],
          "mapped plane has invalid rows, row bytes, or stride");

  const std::size_t row_bytes = static_cast<std::size_t>(layout.row_bytes[plane]);
  require(layout.offsets[plane] <= std::numeric_limits<std::size_t>::max() - row_bytes,
          "mapped plane offset overflows its address range");
  const std::size_t first_row_end = layout.offsets[plane] + row_bytes;
  const std::size_t remaining_rows = static_cast<std::size_t>(layout.rows[plane] - 1);
  const std::size_t stride = static_cast<std::size_t>(layout.strides[plane]);
  require(remaining_rows <= (std::numeric_limits<std::size_t>::max() - first_row_end) / stride,
          "mapped plane extent overflows its address range");
  return first_row_end + remaining_rows * stride;
}

MappedPlaneLayout decoder_padded_layout(FormatTag format, int width, int height) {
  // Preserve the original 640x360 fixture layout (768x384) while allowing the
  // visible geometry to vary independently from decoder-style storage.
  constexpr int kLumaStrideAlignment = 256;
  constexpr int kStorageHeightAlignment = 32;
  const int luma_stride = checked_align_up(width, kLumaStrideAlignment, "luma stride");
  const int storage_height =
      checked_align_up(height, kStorageHeightAlignment, "luma storage height");

  MappedPlaneLayout layout = tight_layout(format, width, height);
  layout.strides[0] = luma_stride;
  layout.offsets[0] = 0;
  if (format == FormatTag::NV12) {
    layout.strides[1] = luma_stride;
    layout.offsets[1] = static_cast<std::size_t>(luma_stride) * storage_height;
    layout.total_bytes =
        layout.offsets[1] + static_cast<std::size_t>(luma_stride) * (storage_height / 2);
    return layout;
  }
  if (format == FormatTag::I420) {
    const int chroma_stride = luma_stride / 2;
    const int chroma_storage_height = storage_height / 2;
    layout.strides[1] = chroma_stride;
    layout.strides[2] = chroma_stride;
    layout.offsets[1] = static_cast<std::size_t>(luma_stride) * storage_height;
    layout.offsets[2] =
        layout.offsets[1] + static_cast<std::size_t>(chroma_stride) * chroma_storage_height;
    layout.total_bytes =
        layout.offsets[2] + static_cast<std::size_t>(chroma_stride) * chroma_storage_height;
    return layout;
  }
  throw std::invalid_argument("decoder padding is only defined for NV12 and I420");
}

GstBuffer* make_decoder_padded_buffer(const RawFrame& frame, bool standard_dmabuf) {
  const MappedPlaneLayout source = tight_layout(frame.format, frame.width, frame.height);
  const MappedPlaneLayout destination =
      decoder_padded_layout(frame.format, frame.width, frame.height);
  require(destination.plane_count == source.plane_count,
          "decoder-style source and destination plane counts differ");
  for (guint plane = 0; plane < destination.plane_count; ++plane) {
    require(destination.rows[plane] == source.rows[plane] &&
                destination.row_bytes[plane] == source.row_bytes[plane],
            "decoder-style source and destination visible planes differ");
    require(plane_end(source, plane) <= frame.bytes.size(),
            "decoder-style source plane exceeds its input frame");
    require(plane_end(destination, plane) <= destination.total_bytes,
            "decoder-style destination plane exceeds its allocation");
  }
  GstBuffer* buffer = nullptr;
  if (standard_dmabuf) {
    simaai::neat::internal::dmabuf::Error error;
    buffer = simaai::neat::internal::dmabuf::allocateDmaBufBuffer(
        simaai::neat::internal::dmabuf::HeapKind::Cma, destination.total_bytes, {}, &error);
    require(buffer != nullptr,
            "failed to allocate standard CMA DMA-BUF padded input: " + error.message());
  } else {
    buffer = gst_buffer_new_allocate(nullptr, static_cast<gsize>(destination.total_bytes), nullptr);
  }
  require(buffer != nullptr, "failed to allocate decoder-style padded input");

  GstMapInfo map{};
  require(gst_buffer_map(buffer, &map, GST_MAP_WRITE), "failed to map decoder-style padded input");
  std::memset(map.data, kPaddingSentinel, map.size);
  for (guint plane = 0; plane < destination.plane_count; ++plane) {
    for (int row = 0; row < destination.rows[plane]; ++row) {
      std::memcpy(map.data + destination.offsets[plane] +
                      static_cast<std::size_t>(row * destination.strides[plane]),
                  frame.bytes.data() + source.offsets[plane] +
                      static_cast<std::size_t>(row * source.strides[plane]),
                  static_cast<std::size_t>(destination.row_bytes[plane]));
    }
  }
  gst_buffer_unmap(buffer, &map);

  gsize offsets[GST_VIDEO_MAX_PLANES] = {};
  gint strides[GST_VIDEO_MAX_PLANES] = {};
  for (guint plane = 0; plane < destination.plane_count; ++plane) {
    offsets[plane] = destination.offsets[plane];
    strides[plane] = destination.strides[plane];
  }
  require(gst_buffer_add_video_meta_full(
              buffer, GST_VIDEO_FRAME_FLAG_NONE, video_format(frame.format),
              static_cast<guint>(frame.width), static_cast<guint>(frame.height),
              destination.plane_count, offsets, strides) != nullptr,
          "failed to attach decoder-style GstVideoMeta");
  return buffer;
}

void run_plugin_padded_scenario(const std::string& name, const RawFrame& input,
                                const RawFrame& expected_rgb, bool standard_dmabuf,
                                const EncodedCodec& codec = kH264) {
  const int port = choose_udp_port();
  RtpReceiver receiver(port, codec);
  const bool explicit_i420_conversion = input.format == FormatTag::I420;
  const std::string conversion = explicit_i420_conversion
                                     ? " ! videoconvert ! video/x-raw,format=NV12,width=" +
                                           std::to_string(g_geometry.width) + ",height=" +
                                           std::to_string(g_geometry.height) + ",framerate=" +
                                           std::to_string(g_fps) + "/1"
                                     : std::string{};
  const std::string context = name + " plugin pipeline";
  const std::string encoder_profile = codec.format == FormatTag::H265 ? "main" : g_encoder_profile;
  const std::string encoder_type = codec.format == FormatTag::H265 ? "h265" : "h264";
  const std::string encoded_tail =
      codec.format == FormatTag::H265
          ? " ! h265parse config-interval=1 ! rtph265pay pt=98 config-interval=1 "
            "timestamp-offset=0"
          : " ! h264parse config-interval=1 ! rtph264pay pt=96 config-interval=1 "
            "timestamp-offset=0";
  auto pipeline = parse_pipeline(
      "appsrc name=source is-live=false format=time block=true caps=\"" + raw_caps(input.format) +
          "\"" + conversion + " ! neatencoder enc-width=" +
          std::to_string(g_geometry.width) + " enc-height=" +
          std::to_string(g_geometry.height) + " enc-frame-rate=" + std::to_string(g_fps) +
          " enc-bitrate=4000 enc-fmt=NV12 enc-type=" + encoder_type +
          " enc-profile=" + encoder_profile + " enc-level=" + g_encoder_level +
          " enc-ip-mode=async" + encoded_tail + " ! udpsink host=127.0.0.1 port=" +
          std::to_string(port) + " sync=false async=false",
      context);
  GstElement* source = required_element(pipeline.get(), "source", context);
  start_pipeline(pipeline.get(), context);

  if (!standard_dmabuf && !explicit_i420_conversion) {
    // Strict neatencoder is intentionally not a materializer. SystemMemory is
    // accepted only through Core's explicit materialize_nv12/convert_to_nv12
    // ingress; a direct plugin connection must fail closed.
    GstBuffer* buffer = make_decoder_padded_buffer(input, false);
    require(gst_app_src_push_buffer(GST_APP_SRC(source), buffer) == GST_FLOW_OK,
            name + ": appsrc rejected the negative-test input before neatencoder");
    (void)gst_app_src_end_of_stream(GST_APP_SRC(source));
    GstBus* bus = gst_element_get_bus(pipeline.get());
    GstMessage* message = gst_bus_timed_pop_filtered(
        bus, 10 * GST_SECOND, static_cast<GstMessageType>(GST_MESSAGE_ERROR | GST_MESSAGE_EOS));
    require(message != nullptr, name + ": strict SystemMemory rejection timed out");
    require(GST_MESSAGE_TYPE(message) == GST_MESSAGE_ERROR,
            name + ": neatencoder silently accepted direct SystemMemory");
    gst_message_unref(message);
    gst_object_unref(bus);
    gst_object_unref(source);
    std::cout << "[PASS] " << name
              << " rejected direct SystemMemory (explicit Core materializer required)\n";
    return;
  }

  std::vector<GstBuffer*> retained;
  std::vector<std::uint64_t> hashes;
  retained.reserve(g_frames_per_scenario);
  hashes.reserve(g_frames_per_scenario);
  for (int frame_index = 0; frame_index < g_frames_per_scenario; ++frame_index) {
    GstBuffer* buffer = make_decoder_padded_buffer(input, true);
    GST_BUFFER_PTS(buffer) = static_cast<GstClockTime>(frame_index) * GST_SECOND / g_fps;
    GST_BUFFER_DTS(buffer) = GST_BUFFER_PTS(buffer);
    GST_BUFFER_DURATION(buffer) = GST_SECOND / g_fps;
    retained.push_back(gst_buffer_ref(buffer));
    hashes.push_back(fnv1a(buffer));
    require(gst_app_src_push_buffer(GST_APP_SRC(source), buffer) == GST_FLOW_OK,
            name + ": appsrc rejected padded input");
  }
  require(gst_app_src_end_of_stream(GST_APP_SRC(source)) == GST_FLOW_OK,
          name + ": failed to send EOS");

  const std::vector<RawFrame> decoded = receiver.pull_frames(g_frames_per_scenario);
  require(decoded.size() >= static_cast<std::size_t>(std::min(kDefaultMinimumDecodedFrames, g_frames_per_scenario)),
          name + ": expected at least " + std::to_string(std::min(kDefaultMinimumDecodedFrames, g_frames_per_scenario)) +
              " decoded frames, received " + std::to_string(decoded.size()));
  const double quality = require_frame_quality(name, expected_rgb, decoded);

  GstBus* bus = gst_element_get_bus(pipeline.get());
  GstMessage* message = gst_bus_timed_pop_filtered(
      bus, 30 * GST_SECOND, static_cast<GstMessageType>(GST_MESSAGE_ERROR | GST_MESSAGE_EOS));
  require(message != nullptr, name + ": encoder pipeline timed out");
  if (GST_MESSAGE_TYPE(message) == GST_MESSAGE_ERROR) {
    GError* error = nullptr;
    gchar* debug = nullptr;
    gst_message_parse_error(message, &error, &debug);
    const std::string detail = error && error->message ? error->message : "unknown error";
    if (error) {
      g_error_free(error);
    }
    g_free(debug);
    gst_message_unref(message);
    gst_object_unref(bus);
    throw std::runtime_error(name + ": encoder pipeline failed: " + detail);
  }
  gst_message_unref(message);
  gst_object_unref(bus);

  for (std::size_t index = 0; index < retained.size(); ++index) {
    require(fnv1a(retained[index]) == hashes[index], name + ": encoder modified a source buffer");
    gst_buffer_unref(retained[index]);
  }
  gst_object_unref(source);
  std::cout << "[PASS] " << name << " decoded=" << decoded.size() << " min_psnr_db=" << quality
            << "\n";
}

double percentile_ms(std::vector<double> values, double quantile) {
  require(!values.empty(), "cannot compute a percentile of an empty sample");
  std::sort(values.begin(), values.end());
  const double position = static_cast<double>(values.size() - 1) * quantile;
  const std::size_t lower = static_cast<std::size_t>(std::floor(position));
  const std::size_t upper = static_cast<std::size_t>(std::ceil(position));
  if (lower == upper)
    return values[lower];
  return values[lower] * (static_cast<double>(upper) - position) +
         values[upper] * (position - static_cast<double>(lower));
}

void run_encoder_performance_scenario(const std::string& name,
                                      const RawFrame& nv12,
                                      const EncodedCodec& codec) {
  constexpr int kWarmupFrames = 100;
  constexpr int kMeasuredFrames = 1000;
  constexpr int kTotalFrames = kWarmupFrames + kMeasuredFrames;
  // The 32 stable input identities cover the configured 30-frame async queue.
  // Together with the 20 output slots and prepared codec buffers, persistent
  // imports remain well below the channel's bounded 256-entry table.
  constexpr unsigned int kPoolDepth = 32;
  const bool h265 = codec.format == FormatTag::H265;
  // The retained Phase-1 throughput baselines were both captured with the
  // codec's main profile.  Keep this qualification stimulus independent of
  // g_encoder_profile, which belongs to the route/control scenarios.
  constexpr const char* kBenchmarkProfile = "main";
  const double throughput_floor = h265 ? 1178.9100707839887 : 1219.0331478006328;
  const double p95_ceiling = (h265 ? 19.029588999999998 : 18.350578999999996) * 1.02;
  const double p99_ceiling = (h265 ? 19.104392009999998 : 18.3765418) * 1.02;
  const std::string context = name + " direct-CMA performance pipeline";
  // Preserve the exact Phase-1 benchmark stimulus.  The retained baseline
  // used a tight, constant NV12 frame (Y=96, UV=128); comparing a real-image
  // fixture with that result would mix codec-content cost into the transport
  // migration gate.  Route/quality scenarios above retain the real image.
  const std::size_t luma_bytes =
      static_cast<std::size_t>(g_geometry.width) * g_geometry.height;
  const std::size_t phase1_frame_bytes = luma_bytes + luma_bytes / 2;
  require(nv12.format == FormatTag::NV12 &&
              nv12.width == g_geometry.width &&
              nv12.height == g_geometry.height &&
              nv12.bytes.size() == phase1_frame_bytes,
          context + ": Phase-1 stimulus requires tight NV12 geometry");
  std::vector<std::uint8_t> phase1_payload(phase1_frame_bytes, 128);
  std::fill_n(phase1_payload.begin(), luma_bytes, 96);
  auto pipeline = parse_pipeline(
      "appsrc name=source is-live=false format=time block=true caps=\"" +
          raw_caps(FormatTag::NV12) +
          "\" ! neatencoder enc-width=" + std::to_string(g_geometry.width) +
          " enc-height=" + std::to_string(g_geometry.height) +
          " enc-frame-rate=" + std::to_string(g_fps) +
          " enc-bitrate=4000 enc-fmt=NV12 enc-type=" +
          (h265 ? "h265" : "h264") + " enc-profile=" +
          kBenchmarkProfile + " enc-level=" +
          g_encoder_level +
          " enc-ip-mode=async ip-queue-max-buffers=30 "
          "ip-queue-min-buffers=15 ! appsink name=sink sync=false "
          "emit-signals=false max-buffers=0 drop=false",
      context);
  GstElement* source = required_element(pipeline.get(), "source", context);
  GstElement* sink = required_element(pipeline.get(), "sink", context);

  simaai::neat::internal::dmabuf::Error pool_error;
  GstBufferPool* pool =
      simaai::neat::internal::dmabuf::createDmaBufPool(
          simaai::neat::internal::dmabuf::HeapKind::Cma,
          phase1_payload.size(), kPoolDepth, kPoolDepth, {}, &pool_error);
  require(pool != nullptr,
          context + ": failed to create fixed standard-CMA input pool: " +
              pool_error.message());

  std::array<gsize, GST_VIDEO_MAX_PLANES> offsets{};
  std::array<gint, GST_VIDEO_MAX_PLANES> strides{};
  offsets[0] = 0;
  offsets[1] = static_cast<gsize>(g_geometry.width) * g_geometry.height;
  strides[0] = g_geometry.width;
  strides[1] = g_geometry.width;

  // Prepare every stable slot before the timed interval. This models the
  // encoder's device-to-device contract: a camera/decoder/ISP producer owns
  // and completes its DMA write before handoff. Re-mapping and CPU-flushing a
  // 345 KiB CMA surface for every synthetic frame would benchmark the test
  // producer's cache maintenance rather than zero-copy encoder throughput.
  std::vector<GstBuffer *> prepared_slots;
  prepared_slots.reserve(kPoolDepth);
  for (unsigned int slot = 0; slot < kPoolDepth; ++slot) {
    GstBuffer *buffer = nullptr;
    require(gst_buffer_pool_acquire_buffer(pool, &buffer, nullptr) ==
                    GST_FLOW_OK &&
                buffer != nullptr,
            context + ": prepared input pool acquisition failed");
    GstMapInfo map{};
    require(gst_buffer_map(buffer, &map, GST_MAP_WRITE),
            context + ": prepared standard-CMA input map failed");
    require(map.size >= phase1_payload.size(),
            context + ": prepared pool slot is smaller than the NV12 frame");
    std::memcpy(map.data, phase1_payload.data(), phase1_payload.size());
    gst_buffer_unmap(buffer, &map);
    prepared_slots.push_back(buffer);
  }
  for (GstBuffer *slot : prepared_slots)
    gst_buffer_unref(slot);
  prepared_slots.clear();

  std::vector<std::chrono::steady_clock::time_point> pushes(kTotalFrames);
  std::vector<std::chrono::steady_clock::time_point> arrivals;
  arrivals.reserve(kTotalFrames);
  std::exception_ptr producer_error;

  start_pipeline(pipeline.get(), context);
  const auto startup_begin = std::chrono::steady_clock::now();
  std::thread producer([&] {
    try {
      for (int index = 0; index < kTotalFrames; ++index) {
        GstBuffer* buffer = nullptr;
        require(gst_buffer_pool_acquire_buffer(pool, &buffer, nullptr) ==
                    GST_FLOW_OK &&
                    buffer != nullptr,
                context + ": fixed input pool acquisition failed");
        if (!gst_buffer_get_video_meta(buffer)) {
          require(gst_buffer_add_video_meta_full(
                      buffer, GST_VIDEO_FRAME_FLAG_NONE, GST_VIDEO_FORMAT_NV12,
                      static_cast<guint>(g_geometry.width),
                      static_cast<guint>(g_geometry.height), 2, offsets.data(),
                      strides.data()) != nullptr,
                  context + ": failed to attach exact NV12 layout");
        }
        GST_BUFFER_PTS(buffer) =
            static_cast<GstClockTime>(index) * GST_SECOND / g_fps;
        GST_BUFFER_DTS(buffer) = GST_BUFFER_PTS(buffer);
        GST_BUFFER_DURATION(buffer) = GST_SECOND / g_fps;
        pushes[index] = std::chrono::steady_clock::now();
        require(gst_app_src_push_buffer(GST_APP_SRC(source), buffer) ==
                    GST_FLOW_OK,
                context + ": appsrc rejected a pooled standard-CMA frame");
      }
      require(gst_app_src_end_of_stream(GST_APP_SRC(source)) == GST_FLOW_OK,
              context + ": failed to send EOS");
    } catch (...) {
      producer_error = std::current_exception();
      (void)gst_app_src_end_of_stream(GST_APP_SRC(source));
    }
  });

  auto first_output = std::chrono::steady_clock::time_point{};
  const auto deadline = std::chrono::steady_clock::now() +
                        std::chrono::seconds(120);
  while (arrivals.size() < kTotalFrames &&
         std::chrono::steady_clock::now() < deadline) {
    GstSamplePtr sample(
        gst_app_sink_try_pull_sample(GST_APP_SINK(sink), GST_SECOND));
    if (sample) {
      const auto now = std::chrono::steady_clock::now();
      if (arrivals.empty())
        first_output = now;
      arrivals.push_back(now);
      continue;
    }
    const std::string error = bus_error(pipeline.get());
    require(error.empty(), context + ": " + error);
    if (gst_app_sink_is_eos(GST_APP_SINK(sink)))
      break;
  }
  producer.join();
  if (producer_error)
    std::rethrow_exception(producer_error);
  require(arrivals.size() == kTotalFrames,
          context + ": expected " + std::to_string(kTotalFrames) +
              " outputs, got " + std::to_string(arrivals.size()));
  GstBus* bus = gst_element_get_bus(pipeline.get());
  GstMessage* terminal = gst_bus_timed_pop_filtered(
      bus, 30 * GST_SECOND,
      static_cast<GstMessageType>(GST_MESSAGE_ERROR | GST_MESSAGE_EOS));
  require(terminal != nullptr && GST_MESSAGE_TYPE(terminal) == GST_MESSAGE_EOS,
          context + ": pipeline did not terminate with clean EOS");
  gst_message_unref(terminal);
  gst_object_unref(bus);

  std::vector<double> latency_ms;
  latency_ms.reserve(kMeasuredFrames);
  for (int index = kWarmupFrames; index < kTotalFrames; ++index) {
    latency_ms.push_back(
        std::chrono::duration<double, std::milli>(arrivals[index] -
                                                  pushes[index])
            .count());
  }
  const double elapsed_seconds =
      std::chrono::duration<double>(arrivals.back() - pushes[kWarmupFrames])
          .count();
  const double throughput = kMeasuredFrames / elapsed_seconds;
  const double p50 = percentile_ms(latency_ms, 0.50);
  const double p95 = percentile_ms(latency_ms, 0.95);
  const double p99 = percentile_ms(latency_ms, 0.99);
  const double maximum = *std::max_element(latency_ms.begin(), latency_ms.end());
  const double startup_ms =
      std::chrono::duration<double, std::milli>(first_output - startup_begin)
          .count();

  // Emit the measurements before enforcing the historical gate.  A failed
  // qualification must retain enough evidence to distinguish hardware time,
  // queue depth and tail-latency regressions instead of reporting only the
  // first violated threshold.
  std::cout << "[METRIC] " << name << " codec=" << codec.name
            << " profile=" << kBenchmarkProfile
            << " input_memory=standard-cma-prepared pool_depth=" << kPoolDepth
            << " stimulus=phase1-constant-y96-uv128"
            << " warmup=" << kWarmupFrames
            << " measured=" << kMeasuredFrames
            << " throughput_fps=" << throughput << " p50_ms=" << p50
            << " p95_ms=" << p95 << " p99_ms=" << p99
            << " max_ms=" << maximum << " startup_ms=" << startup_ms
            << "\n";

  require(throughput >= throughput_floor,
          context + ": throughput regression " + std::to_string(throughput) +
              " < " + std::to_string(throughput_floor));
  require(p95 <= p95_ceiling && p99 <= p99_ceiling,
          context + ": tail-latency regression p95=" + std::to_string(p95) +
              " p99=" + std::to_string(p99));
  gst_element_set_state(pipeline.get(), GST_STATE_NULL);
  require(gst_buffer_pool_set_active(pool, FALSE),
          context + ": failed to deactivate input pool");
  gst_object_unref(pool);
  gst_object_unref(source);
  gst_object_unref(sink);
  std::cout << "[PASS] " << name << " codec=" << codec.name
            << " input_memory=standard-cma-prepared pool_depth=" << kPoolDepth
            << " stimulus=phase1-constant-y96-uv128"
            << " warmup=" << kWarmupFrames
            << " measured=" << kMeasuredFrames
            << " throughput_fps=" << throughput << " p50_ms=" << p50
            << " p95_ms=" << p95 << " p99_ms=" << p99
            << " max_ms=" << maximum << " startup_ms=" << startup_ms
            << "\n";
}

struct AccessUnitFacts {
  GstClockTime pts = GST_CLOCK_TIME_NONE;
  std::vector<int> nal_types;
  bool contains_sei_payload = false;
  bool parser_key_unit = false;
};

struct EncoderFrameFacts {
  GstClockTime pts = GST_CLOCK_TIME_NONE;
  bool metadata_present = false;
  bool key_unit = false;
};

struct EncoderFrameProbe {
  std::mutex mutex;
  std::condition_variable changed;
  std::vector<EncoderFrameFacts> frames;
};

GstPadProbeReturn collect_encoder_frame_facts(GstPad*, GstPadProbeInfo* info,
                                              gpointer user_data) {
  auto* probe = static_cast<EncoderFrameProbe*>(user_data);
  GstBuffer* buffer = GST_PAD_PROBE_INFO_BUFFER(info);
  if (!probe || !buffer)
    return GST_PAD_PROBE_OK;
  EncoderFrameFacts facts;
  facts.pts = GST_BUFFER_PTS(buffer);
  if (GstCustomMeta* meta =
          gst_buffer_get_custom_meta(buffer, "GstSimaMeta")) {
    const GstStructure* structure = gst_custom_meta_get_structure(meta);
    const gchar* frame_type =
        structure ? gst_structure_get_string(structure, "frame-type")
                  : nullptr;
    facts.metadata_present = frame_type != nullptr;
    facts.key_unit =
        frame_type != nullptr && std::string_view(frame_type) == "I-frame";
  }
  {
    std::lock_guard<std::mutex> lock(probe->mutex);
    probe->frames.push_back(facts);
  }
  probe->changed.notify_all();
  return GST_PAD_PROBE_OK;
}

std::vector<std::vector<std::uint8_t>> annex_b_nals(
    const std::vector<std::uint8_t>& bytes) {
  std::vector<std::vector<std::uint8_t>> nals;
  const auto start_code = [&bytes](std::size_t offset) -> std::size_t {
    if (offset + 3 <= bytes.size() && bytes[offset] == 0 && bytes[offset + 1] == 0 &&
        bytes[offset + 2] == 1) {
      return 3;
    }
    if (offset + 4 <= bytes.size() && bytes[offset] == 0 && bytes[offset + 1] == 0 &&
        bytes[offset + 2] == 0 && bytes[offset + 3] == 1) {
      return 4;
    }
    return 0;
  };
  std::size_t cursor = 0;
  while (cursor < bytes.size()) {
    while (cursor < bytes.size() && start_code(cursor) == 0) {
      ++cursor;
    }
    const std::size_t prefix = start_code(cursor);
    if (prefix == 0) {
      break;
    }
    const std::size_t begin = cursor + prefix;
    std::size_t end = begin;
    while (end < bytes.size() && start_code(end) == 0) {
      ++end;
    }
    if (end > begin) {
      nals.emplace_back(bytes.begin() + static_cast<std::ptrdiff_t>(begin),
                        bytes.begin() + static_cast<std::ptrdiff_t>(end));
    }
    cursor = end;
  }
  return nals;
}

std::vector<std::uint8_t> unescape_rbsp(const std::vector<std::uint8_t>& nal,
                                        std::size_t header_bytes) {
  require(nal.size() >= header_bytes, "encoded NAL is shorter than its header");
  std::vector<std::uint8_t> rbsp;
  rbsp.reserve(nal.size() - header_bytes);
  int zero_count = 0;
  for (std::size_t index = header_bytes; index < nal.size(); ++index) {
    const std::uint8_t byte = nal[index];
    if (zero_count >= 2 && byte == 0x03) {
      zero_count = 0;
      continue;
    }
    rbsp.push_back(byte);
    zero_count = byte == 0 ? zero_count + 1 : 0;
  }
  return rbsp;
}

bool contains_bytes(const std::vector<std::uint8_t>& haystack,
                    const std::vector<std::uint8_t>& needle) {
  return !needle.empty() &&
         std::search(haystack.begin(), haystack.end(), needle.begin(), needle.end()) !=
             haystack.end();
}

void run_encoder_control_scenario(const std::string& name, const RawFrame& input,
                                  const EncodedCodec& codec) {
  constexpr int kControlFrames = 12;
  constexpr int kGopLength = 4;
  constexpr int kForcedFrame = 6;
  const std::vector<std::uint8_t> sei_payload{
      0x53, 0x69, 0x4d, 0x61, 0x2d, 0x50, 0x68, 0x61, 0x73, 0x65, 0x35};
  const std::string sei_hex = "53694d612d506861736535";
  const bool h265 = codec.format == FormatTag::H265;
  const std::string profile = h265 ? "main" : g_encoder_profile;
  const std::string bitrate_mode =
      g_encoder_bitrate_mode.empty() ? (h265 ? "vbr" : "cbr")
                                     : g_encoder_bitrate_mode;
  const std::string context = name + " direct encoder control pipeline";

  auto pipeline = parse_pipeline(
      "appsrc name=source is-live=false format=time block=true caps=\"" + raw_caps(input.format) +
          "\" ! neatencoder name=encoder enc-width=" + std::to_string(g_geometry.width) +
          " enc-height=" + std::to_string(g_geometry.height) + " enc-frame-rate=" +
          std::to_string(g_fps) + " enc-bitrate=4000 enc-bitrate-mode=" + bitrate_mode +
          " enc-gop-length=" + std::to_string(kGopLength) + " enc-sei-prefix=true "
          "enc-sei-payload-type=5 enc-sei-payload-hex=" + sei_hex + " enc-fmt=NV12 enc-type=" +
          (h265 ? "h265" : "h264") + " enc-profile=" + profile + " enc-level=" +
          g_encoder_level + " enc-ip-mode=async ! " + codec.parser +
          " config-interval=0 ! appsink name=sink sync=false max-buffers=32 drop=false",
      context);
  GstElement* source = required_element(pipeline.get(), "source", context);
  GstElement* encoder = required_element(pipeline.get(), "encoder", context);
  GstElement* sink = required_element(pipeline.get(), "sink", context);
  GstPad* encoder_src_pad = gst_element_get_static_pad(encoder, "src");
  require(encoder_src_pad != nullptr, context + ": encoder has no src pad");
  EncoderFrameProbe encoder_probe;
  const gulong encoder_probe_id = gst_pad_add_probe(
      encoder_src_pad, GST_PAD_PROBE_TYPE_BUFFER,
      &collect_encoder_frame_facts, &encoder_probe, nullptr);
  require(encoder_probe_id != 0, context + ": failed to attach encoder output probe");

  gchar* configured_mode = nullptr;
  gint configured_gop = 0;
  gchar* configured_sei = nullptr;
  g_object_get(encoder, "enc-bitrate-mode", &configured_mode, "enc-gop-length",
               &configured_gop, "enc-sei-payload-hex", &configured_sei, nullptr);
  require(configured_mode != nullptr && bitrate_mode == configured_mode,
          context + ": bitrate mode property did not round-trip");
  require(configured_gop == kGopLength, context + ": GOP property did not round-trip");
  require(configured_sei != nullptr && sei_hex == configured_sei,
          context + ": SEI payload property did not round-trip");
  g_free(configured_mode);
  g_free(configured_sei);

  start_pipeline(pipeline.get(), context);
  for (int frame_index = 0; frame_index < kControlFrames; ++frame_index) {
    if (frame_index == kForcedFrame) {
      std::unique_lock<std::mutex> lock(encoder_probe.mutex);
      require(encoder_probe.changed.wait_for(
                  lock, std::chrono::seconds(10), [&encoder_probe] {
                    return encoder_probe.frames.size() >= kForcedFrame;
                  }),
              context +
                  ": prior frames did not drain before the serialized force-key-unit event");
      lock.unlock();
      GstPad* source_pad = gst_element_get_static_pad(source, "src");
      require(source_pad != nullptr, context + ": appsrc has no src pad");
      GstEvent* force = gst_video_event_new_downstream_force_key_unit(
          static_cast<GstClockTime>(frame_index) * GST_SECOND / g_fps,
          GST_CLOCK_TIME_NONE, GST_CLOCK_TIME_NONE, TRUE, 1);
      require(gst_pad_push_event(source_pad, force),
              context + ": force-key-unit event was rejected");
      gst_object_unref(source_pad);
    }
    GstBuffer* buffer = make_decoder_padded_buffer(input, true);
    GST_BUFFER_PTS(buffer) = static_cast<GstClockTime>(frame_index) * GST_SECOND / g_fps;
    GST_BUFFER_DTS(buffer) = GST_BUFFER_PTS(buffer);
    GST_BUFFER_DURATION(buffer) = GST_SECOND / g_fps;
    require(gst_app_src_push_buffer(GST_APP_SRC(source), buffer) == GST_FLOW_OK,
            context + ": appsrc rejected control-test input");
  }
  require(gst_app_src_end_of_stream(GST_APP_SRC(source)) == GST_FLOW_OK,
          context + ": failed to send EOS");

  std::vector<AccessUnitFacts> facts;
  std::string caps_profile;
  std::string caps_level;
  const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(30);
  while (facts.size() < kControlFrames && std::chrono::steady_clock::now() < deadline) {
    GstSamplePtr sample(
        gst_app_sink_try_pull_sample(GST_APP_SINK(sink), 250 * GST_MSECOND));
    if (!sample) {
      const std::string error = bus_error(pipeline.get());
      require(error.empty(), context + ": " + error);
      if (gst_app_sink_is_eos(GST_APP_SINK(sink))) {
        break;
      }
      continue;
    }
    if (caps_profile.empty()) {
      GstCaps* caps = gst_sample_get_caps(sample.get());
      require(caps != nullptr && gst_caps_get_size(caps) == 1,
              context + ": parser did not publish exact output caps");
      const GstStructure* structure = gst_caps_get_structure(caps, 0);
      const gchar* parsed_profile = gst_structure_get_string(structure, "profile");
      const gchar* parsed_level = gst_structure_get_string(structure, "level");
      require(parsed_profile != nullptr && parsed_level != nullptr,
              context + ": parser caps omit profile/level");
      caps_profile = parsed_profile;
      caps_level = parsed_level;
    }
    GstBuffer* buffer = gst_sample_get_buffer(sample.get());
    AccessUnitFacts current;
    current.pts = GST_BUFFER_PTS(buffer);
    current.parser_key_unit =
        !GST_BUFFER_FLAG_IS_SET(buffer, GST_BUFFER_FLAG_DELTA_UNIT);
    const auto bytes = copy_buffer_bytes(buffer, context);
    for (const auto& nal : annex_b_nals(bytes)) {
      require(!nal.empty(), context + ": empty Annex-B NAL");
      const int type = h265 ? ((nal.front() >> 1) & 0x3f) : (nal.front() & 0x1f);
      current.nal_types.push_back(type);
      const bool is_sei = h265 ? (type == 39 || type == 40) : type == 6;
      if (is_sei && contains_bytes(unescape_rbsp(nal, h265 ? 2 : 1), sei_payload)) {
        current.contains_sei_payload = true;
      }
    }
    require(!current.nal_types.empty(), context + ": parser output has no Annex-B NAL");
    facts.push_back(std::move(current));
  }
  require(facts.size() == kControlFrames,
          context + ": expected " + std::to_string(kControlFrames) + " access units, got " +
              std::to_string(facts.size()));
  std::vector<EncoderFrameFacts> encoder_facts;
  {
    std::lock_guard<std::mutex> lock(encoder_probe.mutex);
    encoder_facts = encoder_probe.frames;
  }
  require(encoder_facts.size() == kControlFrames,
          context + ": encoder probe expected " +
              std::to_string(kControlFrames) + " frames, got " +
              std::to_string(encoder_facts.size()));
  require(std::all_of(encoder_facts.begin(), encoder_facts.end(),
                      [](const EncoderFrameFacts& frame) {
                        return frame.metadata_present;
                      }),
          context + ": encoder output omitted authoritative frame type metadata");

  const std::string expected_profile = h265 ? "main" : profile;
  require(caps_profile == expected_profile,
          context + ": parser profile mismatch: " + caps_profile + " != " + expected_profile);
  const std::string expected_level = g_encoder_level.size() > 2 &&
                                             g_encoder_level.substr(g_encoder_level.size() - 2) == ".0"
                                         ? g_encoder_level.substr(0, g_encoder_level.size() - 2)
                                         : g_encoder_level;
  require(caps_level == expected_level || caps_level == g_encoder_level,
          context + ": parser level mismatch: " + caps_level + " != " + g_encoder_level);

  const int sps = h265 ? 33 : 7;
  const int pps = h265 ? 34 : 8;
  const int vps = 32;
  const auto has_type = [&facts](int type) {
    return std::any_of(facts.begin(), facts.end(), [type](const AccessUnitFacts& fact) {
      return std::find(fact.nal_types.begin(), fact.nal_types.end(), type) != fact.nal_types.end();
    });
  };
  require(has_type(sps) && has_type(pps) && (!h265 || has_type(vps)),
          context + ": required VPS/SPS/PPS are absent");
  const auto is_idr_or_irap = [h265](const AccessUnitFacts& fact) {
    return std::any_of(fact.nal_types.begin(), fact.nal_types.end(), [h265](int type) {
      return h265 ? (type >= 16 && type <= 21) : type == 5;
    });
  };
  require(is_idr_or_irap(facts.front()), context + ": first access unit is not IDR/IRAP");
  const GstClockTime periodic_pts = static_cast<GstClockTime>(kGopLength) * GST_SECOND / g_fps;
  require(std::any_of(encoder_facts.begin(), encoder_facts.end(),
                      [&](const EncoderFrameFacts& fact) {
            return fact.pts == periodic_pts && fact.key_unit;
          }),
          context + ": configured GOP length did not produce the encoder-authored key unit");
  const GstClockTime forced_pts = static_cast<GstClockTime>(kForcedFrame) * GST_SECOND / g_fps;
  require(std::any_of(encoder_facts.begin(), encoder_facts.end(),
                      [&](const EncoderFrameFacts& fact) {
            return fact.pts == forced_pts && fact.key_unit;
          }),
          context + ": ordered force-key-unit did not produce an encoder-authored I-frame");
  require(std::all_of(facts.begin(), facts.end(), [](const AccessUnitFacts& fact) {
            return fact.contains_sei_payload;
          }),
          context + ": exact SEI payload is not present in every access unit");

  gst_object_unref(source);
  gst_pad_remove_probe(encoder_src_pad, encoder_probe_id);
  gst_object_unref(encoder_src_pad);
  gst_object_unref(encoder);
  gst_object_unref(sink);
  std::cout << "[PASS] " << name << " codec=" << codec.name << " profile=" << caps_profile
            << " level=" << caps_level << " bitrate_mode=" << bitrate_mode
            << " gop=" << kGopLength << " force_key_frame=" << kForcedFrame
            << " sei_bytes=" << sei_payload.size() << "\n";
}

simaai::neat::Graph input_graph(const RawScenario& scenario) {
  simaai::neat::InputOptions input;
  input.payload_type = simaai::neat::PayloadType::Image;
  input.format = scenario.format;
  input.width = g_geometry.width;
  input.height = g_geometry.height;
  input.fps_n = g_fps;
  input.fps_d = 1;
  input.block = true;
  input.pool_min_buffers = 2;
  input.pool_max_buffers = 12;
  input.memory_policy = scenario.memory;

  simaai::neat::Graph source(std::string(scenario.name) + "_source");
  source.add(simaai::neat::nodes::Input("frames", std::move(input)));
  return source;
}

simaai::neat::Graph sender_graph(const RawScenario& scenario, int port) {
  auto options = scenario.h265
                     ? simaai::neat::nodes::groups::VideoSenderOptions::
                           H265RtpUdpFromRaw(g_geometry.width,
                                            g_geometry.height, g_fps)
                     : simaai::neat::nodes::groups::VideoSenderOptions::
                           H264RtpUdpFromRaw(g_geometry.width,
                                            g_geometry.height, g_fps);
  options.host = "127.0.0.1";
  options.video_port_base = port;
  options.encoder.bitrate_kbps = 4000;
  options.encoder.profile = scenario.h265 ? "main" : g_encoder_profile;
  options.encoder.level = g_encoder_level;
  simaai::neat::Graph sender(std::string(scenario.name) + "_sender");
  sender.add(simaai::neat::nodes::groups::VideoSender(options));
  return sender;
}

simaai::neat::Graph scenario_graph(const RawScenario& scenario, int port) {
  auto source = input_graph(scenario);
  auto sender = sender_graph(scenario, port);
  if (scenario.topology == Topology::Linear) {
    simaai::neat::Graph graph(std::string(scenario.name) + "_linear");
    graph.add(std::move(source));
    graph.add(std::move(sender));
    return graph;
  }

  simaai::neat::Graph graph(std::string(scenario.name) + "_connected");
  graph.connect(source, sender);
  if (scenario.topology == Topology::Fanout) {
    simaai::neat::Graph preview(std::string(scenario.name) + "_preview");
    preview.add(simaai::neat::nodes::Output("preview", simaai::neat::OutputOptions::Latest()));
    graph.connect(source, preview);
  }
  return graph;
}

void require_selected_path(const simaai::neat::Graph& graph, const RawScenario& scenario) {
  // Linear Graph::describe_backend() intentionally renders the conservative,
  // unspecialized fragment. Connected graphs expose their edge contract during
  // description, so use those cases to assert the adaptive materialization;
  // the linear data path is still exercised end to end below.
  if (scenario.topology == Topology::Linear) {
    return;
  }
  const std::string backend = graph.describe_backend(false);
  const bool selected_direct = backend.find(kDirectIngressKind) != std::string::npos;
  const bool selected_materialize = backend.find(kMaterializeIngressKind) != std::string::npos;
  const bool selected_fallback = backend.find(kFallbackIngressKind) != std::string::npos;
  // Assert the semantic specialization, not an incidental element in its backend fragment.
  require(static_cast<int>(selected_direct) + static_cast<int>(selected_materialize) +
                  static_cast<int>(selected_fallback) ==
              1,
          std::string(scenario.name) + ": expected exactly one adaptive ingress variant\n" +
              backend);
  const std::string_view selected = selected_direct
                                        ? kDirectIngressKind
                                        : selected_materialize ? kMaterializeIngressKind
                                                               : kFallbackIngressKind;
  require(selected == scenario.expected_ingress,
          std::string(scenario.name) + ": selected the wrong adaptive ingress variant\n" + backend);
}

void run_raw_scenario(const RawScenario& scenario, const RawFrame& input,
                      const RawFrame& expected_rgb) {
  const int port = choose_udp_port();
  RtpReceiver receiver(port, scenario.h265 ? kH265 : kH264);
  simaai::neat::Graph graph = scenario_graph(scenario, port);
  require_selected_path(graph, scenario);

  if (scenario.save_load) {
    const std::filesystem::path saved_path =
        std::filesystem::temp_directory_path() /
        (std::string("video_sender_e2e_") + scenario.name + "_" +
         std::to_string(static_cast<long long>(::getpid())) + ".json");
    ScopedFileRemoval cleanup(saved_path);
    std::error_code error;
    std::filesystem::remove(saved_path, error);
    graph.save(saved_path.string());
    graph = simaai::neat::Graph::load(saved_path.string());
    require_selected_path(graph, scenario);
  }

  Tensor seed = place_frame_for_input_policy(
      tensor_from_frame(input, scenario.row_padding), scenario.memory);
  const std::uint64_t seed_hash = fnv1a(seed);
  Sample seed_sample = simaai::neat::sample_from_tensors(simaai::neat::TensorList{seed});
  seed_sample.frame_id = 0;
  seed_sample.stream_id = scenario.name;
  seed_sample.pts_ns = 0;
  seed_sample.dts_ns = 0;
  seed_sample.duration_ns = 1000000000LL / g_fps;
  simaai::neat::RunOptions run_options;
  run_options.queue_depth = 16;
  run_options.overflow_policy = simaai::neat::OverflowPolicy::Block;
  run_options.startup_preflight = false;
  simaai::neat::Run run = graph.build(seed_sample, run_options);

  std::vector<Tensor> inputs;
  std::vector<std::uint64_t> input_hashes;
  inputs.reserve(g_frames_per_scenario);
  input_hashes.reserve(g_frames_per_scenario);
  // Graph::build uses the seed for contract discovery; submit every frame
  // explicitly through the public streaming API.
  for (int frame_index = 0; frame_index < g_frames_per_scenario; ++frame_index) {
    inputs.push_back(place_frame_for_input_policy(
        tensor_from_frame(input, scenario.row_padding), scenario.memory));
    input_hashes.push_back(fnv1a(inputs.back()));
    const std::int64_t pts_ns = static_cast<std::int64_t>(frame_index) * 1000000000LL / g_fps;
    // sample_from_tensors keeps a GstSample-backed tensor as the transport
    // holder, including its GstVideoMeta, while still using the normal public
    // Run::push path supported by connected and fan-out Graphs.
    Sample sample = simaai::neat::sample_from_tensors(simaai::neat::TensorList{inputs.back()});
    sample.frame_id = frame_index;
    sample.stream_id = scenario.name;
    sample.pts_ns = pts_ns;
    sample.dts_ns = sample.pts_ns;
    sample.duration_ns = 1000000000LL / g_fps;
    require(run.push(std::move(sample)), std::string(scenario.name) + ": push failed");
    // Admission and fairness are expressed at the declared stream rate. Do
    // not create an artificial multi-client burst that exceeds the 30-fps
    // reservation each client actually obtained.
    std::this_thread::sleep_for(std::chrono::milliseconds(1000 / g_fps));
  }
  run.close_input();

  const std::vector<RawFrame> decoded = receiver.pull_frames(g_frames_per_scenario);
  require(decoded.size() >= static_cast<std::size_t>(std::min(kDefaultMinimumDecodedFrames, g_frames_per_scenario)),
          std::string(scenario.name) + ": expected at least " +
              std::to_string(std::min(kDefaultMinimumDecodedFrames, g_frames_per_scenario)) + " decoded frames, received " +
              std::to_string(decoded.size()));
  const double quality = require_frame_quality(scenario.name, expected_rgb, decoded);

  if (scenario.topology == Topology::Fanout) {
    const std::optional<Sample> preview = run.pull(1000);
    require(preview.has_value(), std::string(scenario.name) + ": fanout preview was not produced");
  }
  run.stop();

  // Zero-copy paths may still own the buffers until stop() joins all workers.
  require(fnv1a(seed) == seed_hash, std::string(scenario.name) + ": seed buffer was modified");
  for (std::size_t index = 0; index < inputs.size(); ++index) {
    const Tensor& tensor = inputs[index];
    require(tensor.semantic.image.has_value(),
            std::string(scenario.name) + ": pushed tensor lost image semantics");
    require(fnv1a(tensor) == input_hashes[index],
            std::string(scenario.name) + ": pushed source buffer was modified");
  }

  std::cout << "[PASS] " << scenario.name << " decoded=" << decoded.size()
            << " min_psnr_db=" << quality << "\n";
}

RawFrame channel_identity_frame(const RawFrame& rgb, int channel_index) {
  require(rgb.format == FormatTag::RGB, "channel identity requires an RGB frame");
  require(channel_index >= 0 && channel_index < 64,
          "channel identity supports exactly the Phase-5 1..54 range");

  RawFrame marked = rgb;
  constexpr int kIdentityBits = 6;
  for (int y = 0; y < marked.height; ++y) {
    for (int x = 0; x < marked.width; ++x) {
      const int bit = std::min(kIdentityBits - 1, x * kIdentityBits / marked.width);
      const std::uint8_t target =
          (channel_index & (1 << bit)) != 0 ? std::uint8_t{255} : std::uint8_t{0};
      const std::size_t pixel =
          (static_cast<std::size_t>(y) * marked.width + x) * 3U;
      for (std::size_t component = 0; component < 3U; ++component) {
        // Preserve real-image structure while making every channel in the
        // 1..54 qualification set strongly distinguishable after lossy 4:2:0
        // encoding.  Any two channels differ over at least one full-height
        // stripe, so a crossed RTP/session route cannot satisfy the identity
        // margin below.
        marked.bytes[pixel + component] = static_cast<std::uint8_t>(
            (2U * marked.bytes[pixel + component] + 3U * target) / 5U);
      }
    }
  }
  return marked;
}

struct ChannelIdentity {
  int index = 0;
  double minimum_threshold_margin = 0.0;
};

ChannelIdentity decode_channel_identity(const RawFrame& rgb) {
  require(rgb.format == FormatTag::RGB, "decoded channel identity requires RGB");
  constexpr int kIdentityBits = 6;
  std::array<std::uint64_t, kIdentityBits> totals{};
  std::array<std::uint64_t, kIdentityBits> samples{};
  for (int y = 0; y < rgb.height; ++y) {
    for (int x = 0; x < rgb.width; ++x) {
      const int bit = std::min(kIdentityBits - 1, x * kIdentityBits / rgb.width);
      const std::size_t pixel =
          (static_cast<std::size_t>(y) * rgb.width + x) * 3U;
      totals[static_cast<std::size_t>(bit)] += rgb.bytes[pixel] +
                                               rgb.bytes[pixel + 1U] +
                                               rgb.bytes[pixel + 2U];
      samples[static_cast<std::size_t>(bit)] += 3U;
    }
  }

  ChannelIdentity identity;
  identity.minimum_threshold_margin = std::numeric_limits<double>::infinity();
  for (int bit = 0; bit < kIdentityBits; ++bit) {
    require(samples[static_cast<std::size_t>(bit)] != 0,
            "decoded channel identity stripe is empty");
    const double mean =
        static_cast<double>(totals[static_cast<std::size_t>(bit)]) /
        static_cast<double>(samples[static_cast<std::size_t>(bit)]);
    if (mean >= 127.5)
      identity.index |= 1 << bit;
    identity.minimum_threshold_margin =
        std::min(identity.minimum_threshold_margin, std::abs(mean - 127.5));
  }
  return identity;
}

void run_encoder_concurrency_scenario(const std::string& name,
                                      const RawFrame& base_rgb,
                                      const EncodedCodec& codec) {
  require(g_concurrency_channels >= 1 && g_concurrency_channels <= 54,
          name + ": --channels must be in the qualified range 1..54");
  const std::string context =
      name + " channels=" + std::to_string(g_concurrency_channels);

  const std::size_t channel_count =
      static_cast<std::size_t>(g_concurrency_channels);
  std::vector<std::string> channel_names(channel_count);
  std::vector<RawFrame> expected_rgb(channel_count);
  std::vector<RawFrame> input_nv12(channel_count);
  std::vector<std::unique_ptr<RtpReceiver>> receivers(channel_count);
  std::vector<std::vector<Tensor>> immutable_inputs(channel_count);
  std::vector<std::vector<std::uint64_t>> immutable_input_hashes(channel_count);
  std::vector<simaai::neat::Run> runs(channel_count);
  std::vector<std::exception_ptr> setup_errors(channel_count);
  std::vector<std::thread> setup_workers;
  std::set<int> selected_ports;
  std::mutex selected_ports_mutex;
  setup_workers.reserve(channel_count);
  for (int channel = 0; channel < g_concurrency_channels; ++channel)
    channel_names[static_cast<std::size_t>(channel)] =
        name + "_channel_" + std::to_string(channel);

  simaai::neat::RunOptions run_options;
  run_options.queue_depth = 16;
  run_options.overflow_policy = simaai::neat::OverflowPolicy::Block;
  run_options.startup_preflight = false;

  const auto setup_begin = std::chrono::steady_clock::now();
  for (int channel = 0; channel < g_concurrency_channels; ++channel) {
    setup_workers.emplace_back([&, channel] {
      const std::size_t index = static_cast<std::size_t>(channel);
      try {
        expected_rgb[index] = channel_identity_frame(base_rgb, channel);
        input_nv12[index] = convert_frame(expected_rgb[index], FormatTag::NV12);

        int port = g_concurrency_port_base > 0
                       ? g_concurrency_port_base + channel
                       : 0;
        while (port == 0) {
          const int candidate = choose_udp_port();
          std::lock_guard<std::mutex> lock(selected_ports_mutex);
          if (selected_ports.insert(candidate).second)
            port = candidate;
        }
        receivers[index] = std::make_unique<RtpReceiver>(port, codec);
        RawScenario scenario{
            channel_names[index].c_str(), FormatTag::NV12,
            InputMemoryPolicy::Auto, 0, kDirectIngressKind,
            Topology::Connected, false, codec.format == FormatTag::H265};
        simaai::neat::Graph graph = scenario_graph(scenario, port);
        require_selected_path(graph, scenario);

        // A producer may reuse a DMA-BUF only after the encoder returns its
        // ownership.  At high channel counts one frame can still be in flight
        // at the next 10-fps tick, so a single immutable buffer per channel is
        // not a valid zero-copy producer contract.  Model the bounded camera /
        // decoder pool exactly: one stable CMA slot for every simultaneously
        // submitted fixture frame, with no allocation or materialization in
        // the timed stream loop.
        immutable_inputs[index].reserve(
            static_cast<std::size_t>(g_frames_per_scenario));
        immutable_input_hashes[index].reserve(
            static_cast<std::size_t>(g_frames_per_scenario));
        for (int frame = 0; frame < g_frames_per_scenario; ++frame) {
          Tensor input = place_frame_for_input_policy(
              tensor_from_frame(input_nv12[index], frame),
              InputMemoryPolicy::Auto);
          immutable_input_hashes[index].push_back(fnv1a(input));
          immutable_inputs[index].push_back(std::move(input));
        }
        Sample seed = simaai::neat::sample_from_tensors(
            simaai::neat::TensorList{immutable_inputs[index].front()});
        seed.frame_id = 0;
        seed.stream_id = channel_names[index];
        seed.pts_ns = 0;
        seed.dts_ns = 0;
        seed.duration_ns = 1000000000LL / g_fps;
        runs[index] = graph.build(seed, run_options);
      } catch (...) {
        setup_errors[index] = std::current_exception();
      }
    });
  }
  for (std::thread& worker : setup_workers)
    worker.join();
  for (std::size_t index = 0; index < channel_count; ++index) {
    if (!setup_errors[index])
      continue;
    try {
      std::rethrow_exception(setup_errors[index]);
    } catch (const std::exception& error) {
      throw std::runtime_error(channel_names[index] +
                               ": concurrent setup failed: " + error.what());
    }
  }
  const auto setup_end = std::chrono::steady_clock::now();

  // Submit one frame to every channel at each declared stream tick.  This is
  // simultaneous multi-session load, not N serial single-channel tests.  A
  // stable immutable DMA-BUF identity per channel also exercises persistent
  // import reuse without allocating N*frames CMA surfaces.
  const auto stream_begin = std::chrono::steady_clock::now();
  for (int frame = 0; frame < g_frames_per_scenario; ++frame) {
    for (int channel = 0; channel < g_concurrency_channels; ++channel) {
      Sample sample = simaai::neat::sample_from_tensors(
          simaai::neat::TensorList{
              immutable_inputs[static_cast<std::size_t>(channel)]
                              [static_cast<std::size_t>(frame)]});
      sample.frame_id = frame;
      sample.stream_id = channel_names[static_cast<std::size_t>(channel)];
      sample.pts_ns = static_cast<std::int64_t>(frame) * 1000000000LL / g_fps;
      sample.dts_ns = sample.pts_ns;
      sample.duration_ns = 1000000000LL / g_fps;
      require(runs[static_cast<std::size_t>(channel)].push(std::move(sample)),
              channel_names[static_cast<std::size_t>(channel)] + ": push failed");
    }
    std::this_thread::sleep_for(std::chrono::milliseconds(1000 / g_fps));
  }
  for (simaai::neat::Run& run : runs)
    run.close_input();

  std::vector<double> channel_quality(static_cast<std::size_t>(g_concurrency_channels));
  double minimum_quality = 100.0;
  double minimum_identity_margin = std::numeric_limits<double>::infinity();
  std::size_t total_access_units = 0;
  std::size_t total_decoded = 0;
  for (int channel = 0; channel < g_concurrency_channels; ++channel) {
    const std::vector<RawFrame> decoded =
        receivers[static_cast<std::size_t>(channel)]->pull_frames(g_frames_per_scenario, true);
    const int access_units =
        receivers[static_cast<std::size_t>(channel)]->completed_access_units();
    require(access_units == g_frames_per_scenario,
            channel_names[static_cast<std::size_t>(channel)] +
                ": expected exactly " + std::to_string(g_frames_per_scenario) +
                " complete encoded RTP access units, received " +
                std::to_string(access_units));
    total_access_units += static_cast<std::size_t>(access_units);
    const std::size_t required_frames = static_cast<std::size_t>(
        codec.format == FormatTag::H265
            ? std::min(g_frames_per_scenario, kDefaultMinimumDecodedFrames)
            : g_frames_per_scenario);
    require(decoded.size() >= required_frames,
            channel_names[static_cast<std::size_t>(channel)] + ": expected at least " +
                std::to_string(required_frames) + " decoded frames, received " +
                std::to_string(decoded.size()));
    total_decoded += decoded.size();
    channel_quality[static_cast<std::size_t>(channel)] = require_frame_quality(
        channel_names[static_cast<std::size_t>(channel)],
        expected_rgb[static_cast<std::size_t>(channel)], decoded);
    minimum_quality =
        std::min(minimum_quality, channel_quality[static_cast<std::size_t>(channel)]);

    const ChannelIdentity identity = decode_channel_identity(decoded.front());
    require(identity.index == channel,
            channel_names[static_cast<std::size_t>(channel)] +
                ": decoded payload belongs to channel " +
                std::to_string(identity.index));
    require(identity.minimum_threshold_margin >= 12.0,
            channel_names[static_cast<std::size_t>(channel)] +
                ": decoded channel marker is not robustly separated from its threshold; margin=" +
                std::to_string(identity.minimum_threshold_margin));
    minimum_identity_margin =
        std::min(minimum_identity_margin, identity.minimum_threshold_margin);
  }
  const auto stream_end = std::chrono::steady_clock::now();

  for (simaai::neat::Run& run : runs)
    run.stop();
  for (int channel = 0; channel < g_concurrency_channels; ++channel) {
    const std::size_t index = static_cast<std::size_t>(channel);
    for (int frame = 0; frame < g_frames_per_scenario; ++frame) {
      const std::size_t frame_index = static_cast<std::size_t>(frame);
      require(fnv1a(immutable_inputs[index][frame_index]) ==
                  immutable_input_hashes[index][frame_index],
              channel_names[index] + ": encoder modified immutable DMA-BUF pool slot " +
                  std::to_string(frame));
    }
  }

  const double setup_seconds =
      std::chrono::duration<double>(setup_end - setup_begin).count();
  const double stream_seconds =
      std::chrono::duration<double>(stream_end - stream_begin).count();
  const double submitted_fps =
      static_cast<double>(g_concurrency_channels * g_frames_per_scenario) /
      std::max(stream_seconds, 1e-9);
  std::cout << "[PASS] " << name
            << " channels=" << g_concurrency_channels
            << " submitted=" << g_concurrency_channels * g_frames_per_scenario
            << " encoded_access_units=" << total_access_units
            << " decoded=" << total_decoded
            << " setup_s=" << setup_seconds
            << " stream_s=" << stream_seconds
            << " aggregate_submit_fps=" << submitted_fps
            << " min_psnr_db=" << minimum_quality;
  std::cout << " min_identity_threshold_margin=" << minimum_identity_margin;
  std::cout << "\n";
}

std::string image_path_from_args(int argc, char** argv) {
  for (int index = 1; index + 1 < argc; ++index) {
    if (std::string(argv[index]) == "--image") {
      return argv[index + 1];
    }
  }
  if (const char* value = std::getenv("SIMA_VIDEO_SENDER_E2E_IMAGE")) {
    if (*value) {
      return value;
    }
  }
  return sima_test::test_shared_asset_path("tests/assets/preproc_dynamic/ilena_488.jpg").string();
}

std::string scenario_filter_from_args(int argc, char** argv) {
  for (int index = 1; index + 1 < argc; ++index) {
    if (std::string(argv[index]) == "--scenario") {
      return argv[index + 1];
    }
  }
  return {};
}

int positive_int_arg(int argc, char** argv, const char* flag, int default_value) {
  for (int index = 1; index + 1 < argc; ++index) {
    if (std::string(argv[index]) != flag) {
      continue;
    }
    std::size_t consumed = 0;
    const std::string value_text = argv[index + 1];
    const long value = std::stol(value_text, &consumed, 10);
    if (consumed != value_text.size() || value <= 0 || value > std::numeric_limits<int>::max()) {
      throw std::invalid_argument(std::string(flag) + " must be a positive integer");
    }
    return static_cast<int>(value);
  }
  return default_value;
}

TestGeometry geometry_from_args(int argc, char** argv) {
  TestGeometry geometry;
  geometry.width = positive_int_arg(argc, argv, "--width", geometry.width);
  geometry.height = positive_int_arg(argc, argv, "--height", geometry.height);
  if ((geometry.width & 1) != 0 || (geometry.height & 1) != 0) {
    throw std::invalid_argument("accepted raw 4:2:0 E2E geometry must have even width and height");
  }
  return geometry;
}

std::string encoder_profile_from_args(int argc, char** argv) {
  for (int index = 1; index + 1 < argc; ++index) {
    if (std::string(argv[index]) != "--encoder-profile") {
      continue;
    }
    const std::string profile = argv[index + 1];
    if (profile != "baseline" && profile != "main" && profile != "high") {
      throw std::invalid_argument(
          "--encoder-profile must be baseline, main, or high");
    }
    return profile;
  }
  return "baseline";
}

std::string encoder_level_from_args(int argc, char** argv) {
  for (int index = 1; index + 1 < argc; ++index) {
    if (std::string(argv[index]) != "--encoder-level") {
      continue;
    }
    const std::string level = argv[index + 1];
    constexpr std::array<std::string_view, 17> kLevels{{
        "1.0", "1.1", "1.2", "1.3", "2.0", "2.1", "2.2", "3.0", "3.1",
        "3.2", "4.0", "4.1", "4.2", "5.0", "5.1", "5.2", "6.0"}};
    if (std::find(kLevels.begin(), kLevels.end(), level) == kLevels.end()) {
      throw std::invalid_argument("--encoder-level is not a registered H.264 level");
    }
    return level;
  }
  return "4.0";
}

std::string encoder_bitrate_mode_from_args(int argc, char** argv) {
  for (int index = 1; index + 1 < argc; ++index) {
    if (std::string(argv[index]) != "--encoder-bitrate-mode") {
      continue;
    }
    const std::string mode = argv[index + 1];
    if (mode != "cbr" && mode != "vbr") {
      throw std::invalid_argument("--encoder-bitrate-mode must be cbr or vbr");
    }
    return mode;
  }
  return {};
}

int run_aggregate_scenarios(int argc, char** argv) {
  int failures = 0;
  int skipped = 0;
  for (std::size_t index = 0; index < kAggregateScenarioNames.size(); ++index) {
    const char* scenario = kAggregateScenarioNames[index];
    std::cout << "[RUN] isolated scenario " << scenario << "\n" << std::flush;

    const pid_t child = ::fork();
    if (child < 0) {
      std::cerr << "[FAIL] " << scenario << ": fork failed\n";
      ++failures;
      continue;
    }
    if (child == 0) {
      std::vector<std::string> arguments;
      arguments.reserve(static_cast<std::size_t>(argc) + 2U);
      for (int argument = 0; argument < argc; ++argument) {
        arguments.emplace_back(argv[argument]);
      }
      arguments.emplace_back("--scenario");
      arguments.emplace_back(scenario);

      std::vector<char*> child_argv;
      child_argv.reserve(arguments.size() + 1U);
      for (std::string& argument : arguments) {
        child_argv.push_back(argument.data());
      }
      child_argv.push_back(nullptr);
      ::execvp(child_argv.front(), child_argv.data());
      ::dprintf(STDERR_FILENO, "[FAIL] %s: execvp failed\n", scenario);
      ::_exit(127);
    }

    int status = 0;
    pid_t waited = -1;
    do {
      waited = ::waitpid(child, &status, 0);
    } while (waited < 0 && errno == EINTR);
    if (waited < 0) {
      ++failures;
      std::cerr << "[FAIL] " << scenario << ": waitpid failed\n";
    } else if (WIFEXITED(status) && WEXITSTATUS(status) == kScenarioSkippedExitCode) {
      ++skipped;
    } else if (!WIFEXITED(status) || WEXITSTATUS(status) != 0) {
      ++failures;
      if (WIFSIGNALED(status)) {
        std::cerr << "[FAIL] " << scenario << ": child terminated by signal " << WTERMSIG(status)
                  << "\n";
      } else {
        std::cerr << "[FAIL] " << scenario << ": child exit status " << WEXITSTATUS(status) << "\n";
      }
    }

    // The hardware codec IPC/CMA lifecycle is not process-reentrant. Give the
    // driver a short quiescence window after each isolated child tears down.
    if (index + 1U < kAggregateScenarioNames.size()) {
      std::this_thread::sleep_for(std::chrono::seconds(1));
    }
  }

  if (failures != 0) {
    std::cerr << "[FAIL] video_sender_adaptive_ingress_e2e_test: " << failures << "/"
              << kAggregateScenarioNames.size() << " isolated scenarios failed\n";
    return 1;
  }
  std::cout << "[OK] video_sender_adaptive_ingress_e2e_test passed "
            << (kAggregateScenarioNames.size() - static_cast<std::size_t>(skipped)) << "/"
            << kAggregateScenarioNames.size() << " isolated real-image scenarios";
  if (skipped != 0) {
    std::cout << "; skipped " << skipped
              << " companion-Internals scenarios because the installed encoder predates the "
                 "layout capability";
  }
  std::cout << "\n";
  return 0;
}

} // namespace

int main(int argc, char** argv) {
  // Runtime work happens on background threads. Preserve the original
  // exception text instead of reducing an unexpected escape to "terminate".
  std::set_terminate([] {
    try {
      if (std::exception_ptr pending = std::current_exception()) {
        std::rethrow_exception(pending);
      }
    } catch (const std::exception& error) {
      std::cerr << "[FATAL] uncaught background exception: " << error.what() << "\n";
    } catch (...) {
      std::cerr << "[FATAL] uncaught non-standard background exception\n";
    }
    std::abort();
  });
  try {
    g_geometry = geometry_from_args(argc, argv);
    g_fps = positive_int_arg(argc, argv, "--fps", g_fps);
    g_frames_per_scenario =
        positive_int_arg(argc, argv, "--frames", g_frames_per_scenario);
    g_concurrency_channels =
        positive_int_arg(argc, argv, "--channels", g_concurrency_channels);
    g_concurrency_port_base =
        positive_int_arg(argc, argv, "--port-base", g_concurrency_port_base);
    if (g_fps > 120)
      throw std::invalid_argument("--fps must not exceed 120");
    if (g_frames_per_scenario > 10000)
      throw std::invalid_argument("--frames must not exceed 10000");
    if (g_concurrency_channels > 54)
      throw std::invalid_argument("--channels must not exceed the qualified 54-channel range");
    if (g_concurrency_port_base != 0 &&
        (g_concurrency_port_base < 1024 ||
         g_concurrency_port_base + g_concurrency_channels - 1 > 65535))
      throw std::invalid_argument(
          "--port-base must reserve a complete non-privileged UDP range for all channels");
    g_encoder_profile = encoder_profile_from_args(argc, argv);
    g_encoder_level = encoder_level_from_args(argc, argv);
    g_encoder_bitrate_mode = encoder_bitrate_mode_from_args(argc, argv);
  } catch (const std::exception& error) {
    return fail_test(error.what());
  }
  if (scenario_filter_from_args(argc, argv).empty()) {
    return run_aggregate_scenarios(argc, argv);
  }
  try {
    simaai::neat::gst_init_once();
    const bool layout_aware = encoder_supports_layout_aware_input();
    const std::string image_path = image_path_from_args(argc, argv);
    const std::string scenario_filter = scenario_filter_from_args(argc, argv);
    std::cout << "[INFO] test_geometry=" << g_geometry.width << "x" << g_geometry.height
              << " fps=" << g_fps
              << " frames=" << g_frames_per_scenario
              << " channels=" << g_concurrency_channels
              << " port_base=" << g_concurrency_port_base
              << " encoder_profile=" << g_encoder_profile
              << " encoder_level=" << g_encoder_level << "\n";
    require(std::filesystem::is_regular_file(image_path),
            "missing real-image fixture: " + image_path);

    const RawFrame rgb = load_real_image(image_path);
    const std::array<RawFrame, 5> inputs{{
        rgb,
        convert_frame(rgb, FormatTag::BGR),
        convert_frame(rgb, FormatTag::GRAY8),
        convert_frame(rgb, FormatTag::NV12),
        convert_frame(rgb, FormatTag::I420),
    }};
    const auto frame_for = [&inputs](FormatTag format) -> const RawFrame& {
      for (const RawFrame& frame : inputs) {
        if (frame.format == format) {
          return frame;
        }
      }
      throw std::runtime_error("missing converted real-image frame");
    };

    const std::array<RawScenario, 10> scenarios{{
        {"system_nv12_tight", FormatTag::NV12, InputMemoryPolicy::SystemMemory, 0,
         kMaterializeIngressKind,
         Topology::Connected, true},
        {"system_nv12_padded", FormatTag::NV12, InputMemoryPolicy::SystemMemory, 128,
         kMaterializeIngressKind,
         Topology::Connected, false},
        {"auto_nv12", FormatTag::NV12, InputMemoryPolicy::Auto, 0, kDirectIngressKind,
         Topology::Fanout, false},
        {"ev74_nv12", FormatTag::NV12, InputMemoryPolicy::Ev74, 0, kDirectIngressKind,
         Topology::Linear, false},
        {"public_h265_nv12", FormatTag::NV12,
         InputMemoryPolicy::SystemMemory, 0, kMaterializeIngressKind,
         Topology::Connected, false, true},
        {"system_i420_tight", FormatTag::I420, InputMemoryPolicy::SystemMemory, 0,
         kFallbackIngressKind,
         Topology::Connected, false},
        {"system_i420_padded", FormatTag::I420, InputMemoryPolicy::SystemMemory, 128,
         kFallbackIngressKind,
         Topology::Connected, false},
        {"system_rgb", FormatTag::RGB, InputMemoryPolicy::SystemMemory, 0, kFallbackIngressKind,
         Topology::Linear,
         false},
        {"system_bgr", FormatTag::BGR, InputMemoryPolicy::SystemMemory, 0, kFallbackIngressKind,
         Topology::Linear,
         false},
        {"system_gray8", FormatTag::GRAY8, InputMemoryPolicy::SystemMemory, 0,
         kFallbackIngressKind,
         Topology::Linear, false},
    }};

    int failures = 0;
    std::size_t executed = 0;
    struct EncoderConcurrencyScenario {
      const char* name;
      const EncodedCodec* codec;
    };
    const std::array<EncoderConcurrencyScenario, 2> concurrency_scenarios{{
        {"encoder_concurrency_h264", &kH264},
        {"encoder_concurrency_h265", &kH265},
    }};
    for (const EncoderConcurrencyScenario& scenario : concurrency_scenarios) {
      if (scenario_filter != scenario.name)
        continue;
      ++executed;
      try {
        require(layout_aware,
                std::string(scenario.name) +
                    ": installed neatencoder predates strict direct DMA-BUF input");
        run_encoder_concurrency_scenario(scenario.name, rgb, *scenario.codec);
      } catch (const std::exception& error) {
        ++failures;
        std::cerr << "[FAIL] " << scenario.name << ": " << error.what() << "\n";
      }
    }

    for (const RawScenario& scenario : scenarios) {
      if (!scenario_filter.empty() && scenario_filter != scenario.name) {
        continue;
      }
      ++executed;
      // The legacy encoder ignores GstVideoMeta. Its fallback still covers
      // tight NV12 and converted formats, but padded NV12 needs the companion
      // Internals layout fix to avoid interpreting padding as pixels.
      if (!layout_aware && scenario.format == FormatTag::NV12 && scenario.row_padding != 0) {
        std::cout << "[SKIP] " << scenario.name
                  << ": installed neatencoder predates input-layout-aware\n";
        return kScenarioSkippedExitCode;
      }
      try {
        const RawFrame& input = frame_for(scenario.format);
        const RawFrame expected =
            scenario.format == FormatTag::RGB ? rgb : convert_frame(input, FormatTag::RGB);
        RawScenario effective = scenario;
        if (!layout_aware) {
          effective.expected_ingress = kFallbackIngressKind;
        }
        run_raw_scenario(effective, input, expected);
      } catch (const std::exception& error) {
        ++failures;
        std::cerr << "[FAIL] " << scenario.name << ": " << error.what() << "\n";
      }
    }

    struct PluginScenario {
      const char* name;
      FormatTag format;
      bool use_sima_memory;
      const EncodedCodec* codec;
    };
    const std::array<PluginScenario, 5> plugin_scenarios{{
        {"plugin_padded_nv12_system", FormatTag::NV12, false, &kH264},
        {"plugin_padded_nv12_sima", FormatTag::NV12, true, &kH264},
        {"plugin_padded_nv12_sima_h265", FormatTag::NV12, true, &kH265},
        {"plugin_padded_i420_system", FormatTag::I420, false, &kH264},
        {"plugin_padded_i420_sima", FormatTag::I420, true, &kH264},
    }};
    for (const PluginScenario& scenario : plugin_scenarios) {
      if (!scenario_filter.empty() && scenario_filter != scenario.name) {
        continue;
      }
      ++executed;
      if (!layout_aware) {
        std::cout << "[SKIP] " << scenario.name
                  << ": installed neatencoder predates input-layout-aware\n";
        return kScenarioSkippedExitCode;
      }
      try {
        const RawFrame& input = frame_for(scenario.format);
        const RawFrame expected = convert_frame(input, FormatTag::RGB);
        run_plugin_padded_scenario(scenario.name, input, expected, scenario.use_sima_memory,
                                   *scenario.codec);
      } catch (const std::exception& error) {
        ++failures;
        std::cerr << "[FAIL] " << scenario.name << ": " << error.what() << "\n";
      }
    }

    struct EncoderControlScenario {
      const char* name;
      const EncodedCodec* codec;
    };
    const std::array<EncoderControlScenario, 2> encoder_control_scenarios{{
        {"encoder_controls_h264", &kH264},
        {"encoder_controls_h265", &kH265},
    }};
    for (const EncoderControlScenario& scenario : encoder_control_scenarios) {
      if (!scenario_filter.empty() && scenario_filter != scenario.name) {
        continue;
      }
      ++executed;
      if (!layout_aware) {
        std::cout << "[SKIP] " << scenario.name
                  << ": installed neatencoder predates direct control support\n";
        return kScenarioSkippedExitCode;
      }
      try {
        run_encoder_control_scenario(scenario.name, frame_for(FormatTag::NV12),
                                     *scenario.codec);
      } catch (const std::exception& error) {
        ++failures;
        std::cerr << "[FAIL] " << scenario.name << ": " << error.what() << "\n";
      }
    }

    const std::array<EncoderControlScenario, 2> encoder_performance_scenarios{{
        {"encoder_performance_h264", &kH264},
        {"encoder_performance_h265", &kH265},
    }};
    for (const EncoderControlScenario& scenario :
         encoder_performance_scenarios) {
      if (scenario_filter != scenario.name)
        continue;
      ++executed;
      require(layout_aware,
              std::string(scenario.name) +
                  ": installed neatencoder predates strict direct DMA-BUF input");
      try {
        run_encoder_performance_scenario(
            scenario.name, frame_for(FormatTag::NV12), *scenario.codec);
      } catch (const std::exception& error) {
        ++failures;
        std::cerr << "[FAIL] " << scenario.name << ": " << error.what()
                  << "\n";
      }
    }

    struct EncodedScenario {
      const char* name;
      const EncodedCodec* codec;
      bool native_decode;
    };
    const std::array<EncodedScenario, 4> encoded_scenarios{{
        {"encoded_h264_passthrough", &kH264, false},
        {"encoded_h265_passthrough", &kH265, false},
        {"native_h264_to_h264", &kH264, true},
        {"native_h265_to_h264", &kH265, true},
    }};
    std::optional<std::vector<Sample>> h264_access_units;
    std::optional<std::vector<Sample>> h265_access_units;
    const auto access_units_for = [&](const EncodedCodec& codec) -> const std::vector<Sample>& {
      auto& cached = codec.format == FormatTag::H264 ? h264_access_units : h265_access_units;
      if (!cached.has_value()) {
        cached = generate_access_units(codec, rgb);
      }
      return *cached;
    };

    for (const EncodedScenario& scenario : encoded_scenarios) {
      if (!scenario_filter.empty() && scenario_filter != scenario.name) {
        continue;
      }
      ++executed;
      try {
        require(!scenario.native_decode || simaai::neat::element_exists("neatdecoder"),
                std::string(scenario.name) +
                    ": required release E2E dependency is unavailable: neatdecoder");
        run_encoded_sender_scenario(scenario.name, *scenario.codec,
                                    access_units_for(*scenario.codec), rgb, scenario.native_decode);
      } catch (const std::exception& error) {
        ++failures;
        std::cerr << "[FAIL] " << scenario.name << ": " << error.what() << "\n";
      }
    }

    require(executed != 0U, "requested scenario does not exist: " + scenario_filter);
    if (failures != 0) {
      std::cerr << "[FAIL] video_sender_adaptive_ingress_e2e_test: " << failures << "/" << executed
                << " scenarios failed\n";
      return 1;
    }
    std::cout << "[OK] video_sender_adaptive_ingress_e2e_test passed all " << executed
              << " real-image scenarios\n";
    return 0;
  } catch (const std::exception& error) {
    return fail_test(error.what());
  }
}
