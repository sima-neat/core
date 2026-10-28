#include "nodes/io/UsbCameraInput.h"

#include "builder/OutputSpec.h"
#include "nodes/io/internal/UvcCamera.h"

#include <memory>
#include <sstream>
#include <stdexcept>
#include <utility>

namespace simaai::neat {
namespace {

// Frames held below the camera at once besides the queue: the decoder's in-flight inputs
// (2 by default, 4 with a raised SimaDecode input_buffers) plus one each for the passthrough
// branch and the graph handoff. neatcapturepool accepts at most kMaxCaptureBuffers.
constexpr std::uint32_t kHeldDownstream = 6;
constexpr std::uint32_t kMaxCaptureBuffers = 64;

std::string name(int node_index, const char* suffix) {
  return "n" + std::to_string(node_index) + "_camera_" + suffix;
}

// The camera's JPEG frames, captured into CMA DMA-BUFs that the hardware decoder reads in
// place, for VideoSender passthrough and SimaDecode on the same Graph.
class UvcMjpegCameraNode final : public Node, public OutputSpecProvider {
public:
  UvcMjpegCameraNode(UsbCameraInputOptions opt, internal::uvc::ResolvedCamera camera)
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
       << " io-mode=dmabuf-import ! capsfilter name=" << name(node_index, "caps")
       << " caps=\"image/jpeg,width=" << opt_.width << ",height=" << opt_.height
       << ",framerate=" << opt_.framerate_num << '/' << opt_.framerate_den
       << "\" ! neatcapturepool name=" << name(node_index, "pool") << " buffers=" << pool
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
    out.payload_type = PayloadType::Encoded;
    out.media_type = "image/jpeg";
    out.format = "JPEG";
    out.width = static_cast<int>(opt_.width);
    out.height = static_cast<int>(opt_.height);
    out.fps_num = static_cast<int>(opt_.framerate_num); // exact, e.g. 30000/1001
    out.fps_den = static_cast<int>(opt_.framerate_den);
    out.certainty = SpecCertainty::Hint;
    out.note = "USB MJPEG camera input, JPEG frames";
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
  return std::make_shared<UvcMjpegCameraNode>(opt, camera);
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

} // namespace simaai::neat::nodes
