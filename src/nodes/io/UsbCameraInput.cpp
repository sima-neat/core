#include "nodes/io/UsbCameraInput.h"

#include "builder/OutputSpec.h"
#include "nodes/io/internal/UvcCamera.h"

#include <gst/video/video.h>
#include <limits>
#include <memory>
#include <sstream>
#include <stdexcept>
#include <utility>

namespace simaai::neat {
namespace {

// Frames held below the camera at once besides the queue: the decoder's in-flight inputs
// (2 by default, 4 with a raised SimaDecode input_buffers) plus one each for the passthrough
// branch and the graph handoff. When the fan-out is fused into one tee pipeline
// (ExecutionGraphCompiler), the sender branch holds up to 2 slots instead: the leaky sink
// queue's frame and the one the UDP sink is still sending. neatcapturepool accepts at most
// kMaxCaptureBuffers.
constexpr std::uint32_t kHeldDownstream = 6;
constexpr std::uint32_t kMaxCaptureBuffers = 64;

std::string name(int node_index, const char* suffix) {
  return "n" + std::to_string(node_index) + "_camera_" + suffix;
}

// The camera's JPEG frames, captured into CMA DMA-BUFs that the hardware decoder reads in
// place, for VideoSender passthrough and SimaDecode on the same Graph.
class UvcCameraNode final : public Node, public OutputSpecProvider {
public:
  UvcCameraNode(UsbCameraInputOptions opt, internal::uvc::ResolvedCamera camera)
      : opt_(std::move(opt)), camera_(std::move(camera)) {}

  std::string kind() const override {
    return "UsbCameraInput";
  }
  std::string user_label() const override {
    return opt_.camera_name;
  }
  InputRole input_role() const override {
    return InputRole::Source;
  }
  bool is_live_source() const override {
    return true;
  }
  NodeCapsBehavior caps_behavior() const override {
    return NodeCapsBehavior::Static;
  }
  MemoryContract memory_contract() const override {
    return MemoryContract::AllowEitherButReport;
  }
  std::string buffer_name_hint(int /*node_index*/) const override {
    return opt_.buffer_name;
  }

  std::string backend_fragment(int node_index) const override {
    internal::uvc::require_still_connected(camera_);
    std::ostringstream ss;
    // dmabuf-import captures into neatcapturepool's CMA buffers, which the decoder imports
    // in place. UVC delivers a frame damaged in transfer (lost USB packets) with
    // V4L2_BUF_FLAG_ERROR, which v4l2src maps to CORRUPTED; identity drops it so passthrough
    // never forwards it.
    const std::uint32_t pool = (opt_.insert_queue ? opt_.queue_depth : 0U) + kHeldDownstream;
    ss << "v4l2src name=" << name(node_index, "src") << " device=" << camera_.device
       << " io-mode=dmabuf-import ! capsfilter name=" << name(node_index, "caps") << " caps=\""
       << (camera_.format == UsbCameraFormat::YUYV
               ? "video/x-raw,format=YUY2,interlace-mode=progressive"
               : "image/jpeg")
       << ",width=" << opt_.width << ",height=" << opt_.height
       << ",framerate=" << opt_.framerate_num << '/' << opt_.framerate_den;
    if (camera_.format == UsbCameraFormat::YUYV)
      ss << ",colorimetry=" << camera_.colorimetry << ",chroma-site=jpeg";
    ss << "\" ! neatcapturepool name=" << name(node_index, "pool") << " buffers=" << pool
       << " ! identity name=" << name(node_index, "drop") << " drop-buffer-flags=corrupted";
    if (opt_.insert_queue) {
      ss << " ! queue name=" << name(node_index, "queue");
      if (opt_.queue_depth > 0)
        ss << " max-size-buffers=" << opt_.queue_depth;
      ss << " max-size-bytes=0 max-size-time=0";
      if (opt_.leaky_queue)
        ss << " leaky=downstream";
    }
    return ss.str();
  }

  std::vector<std::string> element_names(int node_index) const override {
    std::vector<std::string> names{name(node_index, "src"), name(node_index, "caps"),
                                   name(node_index, "pool"), name(node_index, "drop")};
    if (opt_.insert_queue)
      names.push_back(name(node_index, "queue"));
    return names;
  }

  OutputSpec output_spec(const OutputSpec& /*input*/) const override {
    OutputSpec out;
    const bool raw = camera_.format == UsbCameraFormat::YUYV;
    out.payload_type = raw ? PayloadType::Image : PayloadType::Encoded;
    out.media_type = raw ? "video/x-raw" : "image/jpeg";
    out.format = raw ? "YUY2" : "JPEG";
    if (raw) {
      out.dtype = "UInt8";
      out.layout = "HWC";
      out.depth = 2; // Packed storage bytes per pixel, not semantic color channels.
      out.memory = "SimaAI";
      out.byte_size = camera_.storage_bytes;
    }
    out.width = static_cast<int>(opt_.width);
    out.height = static_cast<int>(opt_.height);
    out.fps_num = static_cast<int>(opt_.framerate_num); // exact, e.g. 30000/1001
    out.fps_den = static_cast<int>(opt_.framerate_den);
    out.certainty = SpecCertainty::Hint;
    out.note =
        raw ? "USB YUYV camera input, packed raw frames" : "USB MJPEG camera input, JPEG frames";
    return out;
  }

private:
  UsbCameraInputOptions opt_;
  internal::uvc::ResolvedCamera camera_;
};

void validate_options(const UsbCameraInputOptions& opt) {
  if (opt.camera_name.empty())
    throw std::invalid_argument(
        "UsbCameraInput requires camera_name: a USB camera ID from usb_camera_ids()");
  if (opt.width == 0 || opt.height == 0 || opt.framerate_num == 0 || opt.framerate_den == 0)
    throw std::invalid_argument("UsbCameraInput requires a nonzero width, height and frame rate");
  constexpr auto limit = std::uint32_t(std::numeric_limits<int>::max());
  if (opt.width > limit || opt.height > limit || opt.framerate_num > limit ||
      opt.framerate_den > limit)
    throw std::invalid_argument("UsbCameraInput dimensions and rate must fit signed caps integers");
  if (opt.buffer_name.empty())
    throw std::invalid_argument("UsbCameraInput requires a buffer_name");
  if (opt.insert_queue && opt.queue_depth > kMaxCaptureBuffers - kHeldDownstream)
    throw std::invalid_argument("UsbCameraInput queue_depth exceeds " +
                                std::to_string(kMaxCaptureBuffers - kHeldDownstream) +
                                ": the capture pool (queue_depth + " +
                                std::to_string(kHeldDownstream) + ") would exceed " +
                                std::to_string(kMaxCaptureBuffers) + " buffers");
}

} // namespace

namespace internal::uvc {

std::shared_ptr<Node> camera_node(const UsbCameraInputOptions& opt, const ResolvedCamera& camera) {
  return std::make_shared<UvcCameraNode>(opt, camera);
}

} // namespace internal::uvc

std::vector<std::string> usb_camera_ids() {
  return internal::uvc::camera_ids(internal::uvc::list_nodes());
}

UsbCameraInfo usb_camera_info(const UsbCameraInputOptions& opt) {
  validate_options(opt);
  const auto camera = internal::uvc::resolve_mjpeg_camera(opt.camera_name, opt.width, opt.height,
                                                          opt.framerate_num, opt.framerate_den);
  return {camera.device, camera.sampling, camera.index};
}

UsbCameraCaptureInfo usb_camera_info(const UsbCameraInputOptions& opt, UsbCameraFormat format) {
  validate_options(opt);
  if (format != UsbCameraFormat::MJPEG && format != UsbCameraFormat::YUYV)
    throw std::invalid_argument("UsbCameraInput: unsupported capture format");
  if (format == UsbCameraFormat::YUYV && (opt.width & 1U))
    throw std::invalid_argument("UsbCameraInput: YUYV requires an even width");
  const auto camera = internal::uvc::resolve_camera(opt, format);
  return {{camera.device, camera.sampling, camera.index},
          opt,
          format,
          camera.row_stride,
          camera.storage_bytes,
          camera.color,
          camera.colorimetry};
}

} // namespace simaai::neat

namespace simaai::neat::nodes {

std::shared_ptr<simaai::neat::Node> UsbCameraInput(simaai::neat::UsbCameraInputOptions opt) {
  validate_options(opt);
  const auto camera = internal::uvc::resolve_mjpeg_camera(opt.camera_name, opt.width, opt.height,
                                                          opt.framerate_num, opt.framerate_den);
  return internal::uvc::camera_node(opt, camera);
}

std::shared_ptr<simaai::neat::Node> UsbCameraInput(simaai::neat::UsbCameraInputOptions opt,
                                                   const simaai::neat::UsbCameraInfo& info) {
  validate_options(opt);
  return internal::uvc::camera_node(opt, internal::uvc::camera_from_info(opt.camera_name, info));
}

std::shared_ptr<Node> UsbCameraInput(UsbCameraInputOptions opt, UsbCameraFormat format) {
  const auto info = usb_camera_info(opt, format);
  return UsbCameraInput(std::move(opt), info);
}

std::shared_ptr<Node> UsbCameraInput(UsbCameraInputOptions opt, const UsbCameraCaptureInfo& info) {
  validate_options(opt);
  const auto& checked = info.options;
  if (info.format != UsbCameraFormat::MJPEG && info.format != UsbCameraFormat::YUYV)
    throw std::invalid_argument("UsbCameraInput: unsupported capture format");
  if (opt.camera_name != checked.camera_name || opt.width != checked.width ||
      opt.height != checked.height || opt.framerate_num != checked.framerate_num ||
      opt.framerate_den != checked.framerate_den)
    throw std::invalid_argument(
        "UsbCameraInput: capture info describes another mode; call usb_camera_info again");
  if (info.format == UsbCameraFormat::YUYV &&
      ((opt.width & 1U) || info.row_stride < std::uint64_t(opt.width) * 2 ||
       info.storage_bytes <
           std::uint64_t(opt.height - 1) * info.row_stride + std::uint64_t(opt.width) * 2 ||
       info.colorimetry.empty()))
    throw std::invalid_argument(
        "UsbCameraInput: invalid YUYV capture info; call usb_camera_info again");
  GstVideoColorimetry color{};
  if (info.format == UsbCameraFormat::YUYV &&
      ((info.color.matrix != ImageColorMatrix::Bt601 &&
        info.color.matrix != ImageColorMatrix::Bt709) ||
       (info.color.range != ImageYuvRange::Full && info.color.range != ImageYuvRange::Limited) ||
       info.color.input_siting !=
           std::array{ImageChromaSiting::Centered, ImageChromaSiting::Centered}))
    throw std::invalid_argument("UsbCameraInput: invalid YUYV color interpretation");
  if (info.format == UsbCameraFormat::YUYV &&
      (!gst_video_colorimetry_from_string(&color, info.colorimetry.c_str()) ||
       (color.matrix != GST_VIDEO_COLOR_MATRIX_BT601 &&
        color.matrix != GST_VIDEO_COLOR_MATRIX_BT709) ||
       (color.range != GST_VIDEO_COLOR_RANGE_0_255 &&
        color.range != GST_VIDEO_COLOR_RANGE_16_235) ||
       color.primaries == GST_VIDEO_COLOR_PRIMARIES_UNKNOWN ||
       color.transfer == GST_VIDEO_TRANSFER_UNKNOWN ||
       (color.matrix == GST_VIDEO_COLOR_MATRIX_BT601) !=
           (info.color.matrix == ImageColorMatrix::Bt601) ||
       (color.range == GST_VIDEO_COLOR_RANGE_0_255) != (info.color.range == ImageYuvRange::Full)))
    throw std::invalid_argument(
        "UsbCameraInput: invalid YUYV color metadata; call usb_camera_info again");
  auto camera = internal::uvc::camera_from_info(opt.camera_name, info.camera);
  camera.format = info.format;
  camera.row_stride = info.row_stride;
  camera.storage_bytes = info.storage_bytes;
  camera.color = info.color;
  if (info.format == UsbCameraFormat::YUYV) {
    gchar* canonical = gst_video_colorimetry_to_string(&color);
    camera.colorimetry = canonical;
    g_free(canonical);
  }
  return internal::uvc::camera_node(opt, camera);
}

} // namespace simaai::neat::nodes
