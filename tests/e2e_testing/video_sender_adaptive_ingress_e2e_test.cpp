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
#include "test_utils.h"

#include <gst/app/gstappsink.h>
#include <gst/app/gstappsrc.h>
#include <gst/gst.h>
#include <gst/video/video.h>

#include <algorithm>
#include <array>
#include <cerrno>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <exception>
#include <filesystem>
#include <initializer_list>
#include <iostream>
#include <iterator>
#include <memory>
#include <optional>
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

constexpr int kWidth = 640;
constexpr int kHeight = 360;
constexpr int kFps = 30;
constexpr int kFramesPerScenario = 6;
constexpr int kMinimumDecodedFrames = 3;
constexpr int kReceiverTimeoutMs = 15000;
constexpr double kMinimumPsnrDb = 18.0;
constexpr std::uint8_t kPaddingSentinel = 0xD3;
constexpr std::string_view kDirectIngressKind = "VideoSenderRawIngress[direct_nv12]";
constexpr std::string_view kFallbackIngressKind = "VideoSenderRawIngress[convert_to_nv12]";
constexpr std::array<const char*, 17> kAggregateScenarioNames{{
    "system_nv12_tight",
    "system_nv12_padded",
    "auto_nv12",
    "ev74_nv12",
    "system_i420_tight",
    "system_i420_padded",
    "system_rgb",
    "system_bgr",
    "system_gray8",
    "plugin_padded_nv12_system",
    "plugin_padded_nv12_sima",
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
  bool expect_direct;
  Topology topology = Topology::Linear;
  bool save_load = false;
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

std::string raw_caps(FormatTag format, int width = kWidth, int height = kHeight) {
  std::ostringstream caps;
  caps << "video/x-raw,format=(string)" << gst_format(format) << ",width=(int)" << width
       << ",height=(int)" << height << ",framerate=(fraction)" << kFps << "/1";
  return caps.str();
}

void require_layout_aware_encoder() {
  GstObjectPtr<GstElementFactory> factory(gst_element_factory_find("neatencoder"));
  require(factory != nullptr, "required GStreamer factory is unavailable: neatencoder");
  GstObjectPtr<GstPluginFeature> loaded(gst_plugin_feature_load(GST_PLUGIN_FEATURE(factory.get())));
  require(loaded != nullptr, "failed to load the neatencoder plugin");

  GstElementPtr encoder(gst_element_factory_create(GST_ELEMENT_FACTORY(loaded.get()), nullptr));
  require(encoder != nullptr, "failed to instantiate neatencoder");
  GParamSpec* capability =
      g_object_class_find_property(G_OBJECT_GET_CLASS(encoder.get()), "input-layout-aware");
  require(capability != nullptr,
          "neatencoder is too old: missing the input-layout-aware capability");
  require(G_PARAM_SPEC_VALUE_TYPE(capability) == G_TYPE_BOOLEAN &&
              (capability->flags & G_PARAM_READABLE) != 0,
          "neatencoder input-layout-aware capability is not a readable boolean");
  gboolean layout_aware = FALSE;
  g_object_get(encoder.get(), "input-layout-aware", &layout_aware, nullptr);
  require(layout_aware, "neatencoder does not support layout-aware raw input");

  std::string plugin_path = "<unknown>";
  if (GstObjectPtr<GstPlugin> plugin(gst_plugin_feature_get_plugin(loaded.get())); plugin) {
    if (const gchar* filename = gst_plugin_get_filename(plugin.get())) {
      plugin_path = filename;
    }
  }
  std::cout << "[INFO] neatencoder=" << plugin_path << " input-layout-aware=true\n";
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
                                     std::to_string(kWidth) + ",height=" + std::to_string(kHeight) +
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

  GstBuffer* buffer = gst_buffer_new_allocate(nullptr, input.bytes.size(), nullptr);
  require(buffer != nullptr, context + ": failed to allocate input buffer");
  gst_buffer_fill(buffer, 0, input.bytes.data(), input.bytes.size());
  GST_BUFFER_PTS(buffer) = 0;
  GST_BUFFER_DURATION(buffer) = GST_SECOND / kFps;
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

  gsize offsets[GST_VIDEO_MAX_PLANES] = {};
  gint strides[GST_VIDEO_MAX_PLANES] = {};
  for (guint plane = 0; plane < destination_layout.plane_count; ++plane) {
    offsets[plane] = destination_layout.offsets[plane];
    strides[plane] = destination_layout.strides[plane];
  }
  require(gst_buffer_add_video_meta_full(
              buffer, GST_VIDEO_FRAME_FLAG_NONE, video_format(frame.format),
              static_cast<guint>(frame.width), static_cast<guint>(frame.height),
              destination_layout.plane_count, offsets, strides) != nullptr,
          "failed to attach real-frame GstVideoMeta");

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
      : context_(codec.name + std::string(" RTP receiver")) {
    const std::string decoder = software_decoder(codec);
    require(!decoder.empty(), context_ + ": no software decoder is installed");
    pipeline_ = parse_pipeline(
        "udpsrc port=" + std::to_string(port) +
            " caps=\"application/x-rtp,media=(string)video,encoding-name=(string)" +
            codec.rtp_encoding_name + ",payload=(int)" + std::to_string(codec.payload_type) +
            ",clock-rate=(int)90000\" ! rtpjitterbuffer latency=25 ! " + codec.depayloader + " ! " +
            codec.parser + " ! " + decoder +
            " ! videoconvert ! "
            "video/x-raw,format=RGB,width=" +
            std::to_string(kWidth) + ",height=" + std::to_string(kHeight) +
            " ! appsink name=sink sync=false max-buffers=32 drop=false",
        context_);
    sink_.reset(required_element(pipeline_.get(), "sink", context_));
    start_pipeline(pipeline_.get(), context_);
    std::this_thread::sleep_for(std::chrono::milliseconds(100));
  }

  std::vector<RawFrame> pull_frames(int maximum) {
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
      if (frames.size() >= static_cast<std::size_t>(std::min(maximum, kMinimumDecodedFrames))) {
        break;
      }
    }
    return frames;
  }

private:
  std::string context_;
  GstElementPtr pipeline_;
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

  for (int frame_index = 0; frame_index < kFramesPerScenario; ++frame_index) {
    GstBuffer* buffer = gst_buffer_new_allocate(nullptr, rgb.bytes.size(), nullptr);
    require(buffer != nullptr, context + ": failed to allocate raw input");
    gst_buffer_fill(buffer, 0, rgb.bytes.data(), rgb.bytes.size());
    GST_BUFFER_PTS(buffer) = static_cast<GstClockTime>(frame_index) * GST_SECOND / kFps;
    GST_BUFFER_DTS(buffer) = GST_BUFFER_PTS(buffer);
    GST_BUFFER_DURATION(buffer) = GST_SECOND / kFps;
    require(gst_app_src_push_buffer(GST_APP_SRC(source), buffer) == GST_FLOW_OK,
            context + ": appsrc rejected the real-image frame");
  }
  require(gst_app_src_end_of_stream(GST_APP_SRC(source)) == GST_FLOW_OK,
          context + ": failed to send EOS");

  std::vector<Sample> access_units;
  access_units.reserve(kFramesPerScenario);
  const auto deadline =
      std::chrono::steady_clock::now() + std::chrono::milliseconds(kReceiverTimeoutMs);
  while (access_units.size() < static_cast<std::size_t>(kFramesPerScenario) &&
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
    const std::int64_t duration_ns = 1000000000LL / kFps;
    Sample sample = simaai::neat::make_encoded_sample(
        copy_buffer_bytes(gst_sample_get_buffer(encoded.get()), context), encoded_caps(codec),
        frame_index * duration_ns, frame_index * duration_ns, duration_ns);
    sample.frame_id = frame_index;
    sample.stream_id = std::string("software-") + codec.name;
    access_units.push_back(std::move(sample));
  }
  gst_object_unref(source);
  gst_object_unref(sink);

  require(access_units.size() == static_cast<std::size_t>(kFramesPerScenario),
          context + ": expected " + std::to_string(kFramesPerScenario) +
              " access units, received " + std::to_string(access_units.size()));
  return access_units;
}

void run_encoded_sender_scenario(const std::string& name, const EncodedCodec& codec,
                                 const std::vector<Sample>& access_units,
                                 const RawFrame& expected_rgb, bool native_decode) {
  require(!access_units.empty(), name + ": no encoded access units");
  const int port = choose_udp_port();
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
    decode.dec_width = kWidth;
    decode.dec_height = kHeight;
    decode.dec_fps = kFps;
    graph.add(simaai::neat::nodes::SimaDecode(std::move(decode)));
    // Native decode contracts are deliberately hints. Pin the observed native
    // NV12 boundary so this E2E exercises the adaptive direct ingress too.
    graph.add(
        simaai::neat::nodes::CapsRaw("NV12", kWidth, kHeight, kFps, simaai::neat::CapsMemory::Any));

    auto sender =
        simaai::neat::nodes::groups::VideoSenderOptions::H264RtpUdpFromRaw(kWidth, kHeight, kFps);
    sender.encoder.bitrate_kbps = 4000;
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

  const std::vector<RawFrame> decoded = receiver.pull_frames(kFramesPerScenario);
  require(decoded.size() >= static_cast<std::size_t>(kMinimumDecodedFrames),
          name + ": expected at least " + std::to_string(kMinimumDecodedFrames) +
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

MappedPlaneLayout decoder_padded_layout(FormatTag format) {
  constexpr int kLumaStride = 768;
  constexpr int kStorageHeight = 384;
  MappedPlaneLayout layout = tight_layout(format, kWidth, kHeight);
  layout.strides[0] = kLumaStride;
  layout.offsets[0] = 0;
  if (format == FormatTag::NV12) {
    layout.strides[1] = kLumaStride;
    layout.offsets[1] = static_cast<std::size_t>(kLumaStride) * kStorageHeight;
    layout.total_bytes =
        layout.offsets[1] + static_cast<std::size_t>(kLumaStride) * (kStorageHeight / 2);
    return layout;
  }
  if (format == FormatTag::I420) {
    const int chroma_stride = kLumaStride / 2;
    const int chroma_storage_height = kStorageHeight / 2;
    layout.strides[1] = chroma_stride;
    layout.strides[2] = chroma_stride;
    layout.offsets[1] = static_cast<std::size_t>(kLumaStride) * kStorageHeight;
    layout.offsets[2] =
        layout.offsets[1] + static_cast<std::size_t>(chroma_stride) * chroma_storage_height;
    layout.total_bytes =
        layout.offsets[2] + static_cast<std::size_t>(chroma_stride) * chroma_storage_height;
    return layout;
  }
  throw std::invalid_argument("decoder padding is only defined for NV12 and I420");
}

GstBuffer* make_decoder_padded_buffer(const RawFrame& frame, GstAllocator* allocator) {
  const MappedPlaneLayout source = tight_layout(frame.format, frame.width, frame.height);
  const MappedPlaneLayout destination = decoder_padded_layout(frame.format);
  GstBuffer* buffer =
      gst_buffer_new_allocate(allocator, static_cast<gsize>(destination.total_bytes), nullptr);
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
                                const RawFrame& expected_rgb, bool use_sima_memory) {
  const int port = choose_udp_port();
  RtpReceiver receiver(port, kH264);
  const std::string encoder_format = input.format == FormatTag::NV12 ? "NV12" : "YUV420P";
  const std::string context = name + " plugin pipeline";
  auto pipeline = parse_pipeline(
      "appsrc name=source is-live=false format=time block=true caps=\"" + raw_caps(input.format) +
          "\" ! neatencoder enc-width=" + std::to_string(kWidth) +
          " enc-height=" + std::to_string(kHeight) + " enc-frame-rate=" + std::to_string(kFps) +
          " enc-bitrate=4000 enc-fmt=" + encoder_format +
          " enc-ip-mode=async ! h264parse config-interval=1 ! "
          "rtph264pay pt=96 config-interval=1 timestamp-offset=0 ! "
          "udpsink host=127.0.0.1 port=" +
          std::to_string(port) + " sync=false async=false",
      context);
  GstElement* source = required_element(pipeline.get(), "source", context);
  GstAllocator* allocator = use_sima_memory ? gst_allocator_find("NeatSimaaiMemory") : nullptr;
  require(!use_sima_memory || allocator != nullptr, name + ": SiMa input allocator is unavailable");
  start_pipeline(pipeline.get(), context);

  std::vector<GstBuffer*> retained;
  std::vector<std::uint64_t> hashes;
  retained.reserve(kFramesPerScenario);
  hashes.reserve(kFramesPerScenario);
  for (int frame_index = 0; frame_index < kFramesPerScenario; ++frame_index) {
    GstBuffer* buffer = make_decoder_padded_buffer(input, allocator);
    GST_BUFFER_PTS(buffer) = static_cast<GstClockTime>(frame_index) * GST_SECOND / kFps;
    GST_BUFFER_DTS(buffer) = GST_BUFFER_PTS(buffer);
    GST_BUFFER_DURATION(buffer) = GST_SECOND / kFps;
    retained.push_back(gst_buffer_ref(buffer));
    hashes.push_back(fnv1a(buffer));
    require(gst_app_src_push_buffer(GST_APP_SRC(source), buffer) == GST_FLOW_OK,
            name + ": appsrc rejected padded input");
  }
  require(gst_app_src_end_of_stream(GST_APP_SRC(source)) == GST_FLOW_OK,
          name + ": failed to send EOS");

  const std::vector<RawFrame> decoded = receiver.pull_frames(kFramesPerScenario);
  require(decoded.size() >= static_cast<std::size_t>(kMinimumDecodedFrames),
          name + ": expected at least " + std::to_string(kMinimumDecodedFrames) +
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
  if (allocator) {
    gst_object_unref(allocator);
  }
  gst_object_unref(source);
  std::cout << "[PASS] " << name << " decoded=" << decoded.size() << " min_psnr_db=" << quality
            << "\n";
}

simaai::neat::Graph input_graph(const RawScenario& scenario) {
  simaai::neat::InputOptions input;
  input.payload_type = simaai::neat::PayloadType::Image;
  input.format = scenario.format;
  input.width = kWidth;
  input.height = kHeight;
  input.fps_n = kFps;
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
  auto options =
      simaai::neat::nodes::groups::VideoSenderOptions::H264RtpUdpFromRaw(kWidth, kHeight, kFps);
  options.host = "127.0.0.1";
  options.video_port_base = port;
  options.encoder.bitrate_kbps = 4000;
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
  const bool selected_fallback = backend.find(kFallbackIngressKind) != std::string::npos;
  // Assert the semantic specialization, not an incidental element in its backend fragment.
  require(selected_direct != selected_fallback,
          std::string(scenario.name) + ": expected exactly one adaptive ingress variant\n" +
              backend);
  require(selected_direct == scenario.expect_direct,
          std::string(scenario.name) + ": selected the wrong adaptive ingress variant\n" + backend);
}

void run_raw_scenario(const RawScenario& scenario, const RawFrame& input,
                      const RawFrame& expected_rgb) {
  const int port = choose_udp_port();
  RtpReceiver receiver(port, kH264);
  simaai::neat::Graph graph = scenario_graph(scenario, port);
  require_selected_path(graph, scenario);
  GstObjectPtr<GstAllocator> input_allocator;
  if (scenario.memory == InputMemoryPolicy::Ev74) {
    input_allocator.reset(gst_allocator_find("NeatSimaaiMemory"));
    require(input_allocator != nullptr,
            std::string(scenario.name) + ": EV74 input allocator is unavailable");
  }

  if (scenario.save_load) {
    const std::filesystem::path saved_path =
        std::filesystem::temp_directory_path() /
        (std::string("video_sender_e2e_") + scenario.name + ".json");
    ScopedFileRemoval cleanup(saved_path);
    std::error_code error;
    std::filesystem::remove(saved_path, error);
    graph.save(saved_path.string());
    graph = simaai::neat::Graph::load(saved_path.string());
    require_selected_path(graph, scenario);
  }

  Tensor seed = tensor_from_frame(input, scenario.row_padding, input_allocator.get());
  const std::uint64_t seed_hash = fnv1a(seed);
  Sample seed_sample = simaai::neat::sample_from_tensors(simaai::neat::TensorList{seed});
  seed_sample.frame_id = 0;
  seed_sample.stream_id = scenario.name;
  seed_sample.pts_ns = 0;
  seed_sample.dts_ns = 0;
  seed_sample.duration_ns = 1000000000LL / kFps;
  simaai::neat::RunOptions run_options;
  run_options.queue_depth = 16;
  run_options.overflow_policy = simaai::neat::OverflowPolicy::Block;
  run_options.startup_preflight = false;
  simaai::neat::Run run = graph.build(seed_sample, run_options);

  std::vector<Tensor> inputs;
  std::vector<std::uint64_t> input_hashes;
  inputs.reserve(kFramesPerScenario);
  input_hashes.reserve(kFramesPerScenario);
  // Graph::build uses the seed for contract discovery; submit every frame
  // explicitly through the public streaming API.
  for (int frame_index = 0; frame_index < kFramesPerScenario; ++frame_index) {
    inputs.push_back(tensor_from_frame(input, scenario.row_padding, input_allocator.get()));
    input_hashes.push_back(fnv1a(inputs.back()));
    const std::int64_t pts_ns = static_cast<std::int64_t>(frame_index) * 1000000000LL / kFps;
    // sample_from_tensors keeps a GstSample-backed tensor as the transport
    // holder, including its GstVideoMeta, while still using the normal public
    // Run::push path supported by connected and fan-out Graphs.
    Sample sample = simaai::neat::sample_from_tensors(simaai::neat::TensorList{inputs.back()});
    sample.frame_id = frame_index;
    sample.stream_id = scenario.name;
    sample.pts_ns = pts_ns;
    sample.dts_ns = sample.pts_ns;
    sample.duration_ns = 1000000000LL / kFps;
    require(run.push(std::move(sample)), std::string(scenario.name) + ": push failed");
    std::this_thread::sleep_for(std::chrono::milliseconds(4));
  }
  run.close_input();

  const std::vector<RawFrame> decoded = receiver.pull_frames(kFramesPerScenario);
  require(decoded.size() >= static_cast<std::size_t>(kMinimumDecodedFrames),
          std::string(scenario.name) + ": expected at least " +
              std::to_string(kMinimumDecodedFrames) + " decoded frames, received " +
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
  return "tests/assets/preproc_dynamic/ilena_488.jpg";
}

std::string scenario_filter_from_args(int argc, char** argv) {
  for (int index = 1; index + 1 < argc; ++index) {
    if (std::string(argv[index]) == "--scenario") {
      return argv[index + 1];
    }
  }
  return {};
}

int run_aggregate_scenarios(int argc, char** argv) {
  int failures = 0;
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
  std::cout << "[OK] video_sender_adaptive_ingress_e2e_test passed all "
            << kAggregateScenarioNames.size() << " isolated real-image scenarios\n";
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
  if (scenario_filter_from_args(argc, argv).empty()) {
    return run_aggregate_scenarios(argc, argv);
  }
  try {
    simaai::neat::gst_init_once();
    require_layout_aware_encoder();
    const std::string image_path = image_path_from_args(argc, argv);
    const std::string scenario_filter = scenario_filter_from_args(argc, argv);
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

    const std::array<RawScenario, 9> scenarios{{
        {"system_nv12_tight", FormatTag::NV12, InputMemoryPolicy::SystemMemory, 0, true,
         Topology::Connected, true},
        {"system_nv12_padded", FormatTag::NV12, InputMemoryPolicy::SystemMemory, 128, true,
         Topology::Connected, false},
        {"auto_nv12", FormatTag::NV12, InputMemoryPolicy::Auto, 0, true, Topology::Fanout, false},
        {"ev74_nv12", FormatTag::NV12, InputMemoryPolicy::Ev74, 0, true, Topology::Linear, false},
        {"system_i420_tight", FormatTag::I420, InputMemoryPolicy::SystemMemory, 0, false,
         Topology::Connected, false},
        {"system_i420_padded", FormatTag::I420, InputMemoryPolicy::SystemMemory, 128, false,
         Topology::Connected, false},
        {"system_rgb", FormatTag::RGB, InputMemoryPolicy::SystemMemory, 0, false, Topology::Linear,
         false},
        {"system_bgr", FormatTag::BGR, InputMemoryPolicy::SystemMemory, 0, false, Topology::Linear,
         false},
        {"system_gray8", FormatTag::GRAY8, InputMemoryPolicy::SystemMemory, 0, false,
         Topology::Linear, false},
    }};

    int failures = 0;
    std::size_t executed = 0;
    for (const RawScenario& scenario : scenarios) {
      if (!scenario_filter.empty() && scenario_filter != scenario.name) {
        continue;
      }
      ++executed;
      try {
        const RawFrame& input = frame_for(scenario.format);
        const RawFrame expected =
            scenario.format == FormatTag::RGB ? rgb : convert_frame(input, FormatTag::RGB);
        run_raw_scenario(scenario, input, expected);
      } catch (const std::exception& error) {
        ++failures;
        std::cerr << "[FAIL] " << scenario.name << ": " << error.what() << "\n";
      }
    }

    struct PluginScenario {
      const char* name;
      FormatTag format;
      bool use_sima_memory;
    };
    const std::array<PluginScenario, 4> plugin_scenarios{{
        {"plugin_padded_nv12_system", FormatTag::NV12, false},
        {"plugin_padded_nv12_sima", FormatTag::NV12, true},
        {"plugin_padded_i420_system", FormatTag::I420, false},
        {"plugin_padded_i420_sima", FormatTag::I420, true},
    }};
    for (const PluginScenario& scenario : plugin_scenarios) {
      if (!scenario_filter.empty() && scenario_filter != scenario.name) {
        continue;
      }
      ++executed;
      try {
        const RawFrame& input = frame_for(scenario.format);
        const RawFrame expected = convert_frame(input, FormatTag::RGB);
        run_plugin_padded_scenario(scenario.name, input, expected, scenario.use_sima_memory);
      } catch (const std::exception& error) {
        ++failures;
        std::cerr << "[FAIL] " << scenario.name << ": " << error.what() << "\n";
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
