#include "nodes/io/UsbCameraInput.h"

#include "builder/OutputSpec.h"
#include "gst/GstHelpers.h"
#include "nodes/io/internal/UvcCamera.h"
#include "nodes/sima/SimaDecode.h"

#include <memory>
#include <sstream>
#include <stdexcept>
#include <utility>

namespace simaai::neat {
namespace {

// neatdecoder's default compressed-input slots; each holds one capture buffer
// while the codec reads it in place.
constexpr std::uint32_t kDecoderInputSlots = 2;
constexpr std::uint32_t kMaxCaptureBuffers = 64; // neatdecoder input-pool-buffers limit

std::string name(int node_index, const char* suffix) {
  return "n" + std::to_string(node_index) + "_camera_" + suffix;
}

std::string quoted(const std::string& value) {
  std::string out = "\"";
  for (char ch : value) {
    if (ch == '\\' || ch == '"')
      out.push_back('\\');
    out.push_back(ch);
  }
  return out + '"';
}

// Frames: one node keeps capture and decode in one GStreamer segment, so v4l2src gets
// its CMA buffers from neatdecoder's answer to its ALLOCATION query.
class UvcMjpegCameraNode final : public Node, public OutputSpecProvider {
public:
  UvcMjpegCameraNode(UsbCameraInputOptions opt, internal::uvc::ResolvedCamera camera,
                     std::uint32_t pool_buffers)
      : opt_(std::move(opt)), camera_(std::move(camera)), pool_buffers_(pool_buffers) {}

  std::string kind() const override {
    return "UsbCameraInput";
  }
  std::string user_label() const override {
    return opt_.camera_name;
  }
  InputRole input_role() const override {
    return InputRole::Source;
  }
  NodeCapsBehavior caps_behavior() const override {
    return NodeCapsBehavior::Static;
  }
  MemoryContract memory_contract() const override {
    return jpeg() ? MemoryContract::AllowEitherButReport : MemoryContract::PreferDeviceZeroCopy;
  }
  std::string buffer_name_hint(int /*node_index*/) const override {
    return opt_.buffer_name;
  }

  std::string backend_fragment(int node_index) const override {
    internal::uvc::require_still_connected(camera_);
    std::ostringstream ss;
    ss << "v4l2src name=" << name(node_index, "src") << " device=" << camera_.device
       << (jpeg() ? " io-mode=dmabuf" : " io-mode=dmabuf-import")
       << " ! capsfilter name=" << name(node_index, "caps")
       << " caps=\"image/jpeg,width=" << opt_.width << ",height=" << opt_.height
       << ",framerate=" << opt_.framerate_num << '/' << opt_.framerate_den << '"';
    if (opt_.insert_queue) {
      ss << " ! queue name=" << name(node_index, "queue");
      if (opt_.queue_depth > 0)
        ss << " max-size-buffers=" << opt_.queue_depth;
      ss << " max-size-bytes=0 max-size-time=0";
      if (opt_.leaky_queue)
        ss << " leaky=downstream";
    }
    if (jpeg())
      return ss.str();
    // Capture always lands in the decoder's own buffers, so no frame ever needs a copy.
    ss << " ! " << decoder().backend_fragment(node_index)
       << " op-buff-name=" << quoted(opt_.buffer_name) << " input-pool-buffers=" << pool_buffers_
       << " copy-allowed=false";
    return ss.str();
  }

  std::vector<std::string> element_names(int node_index) const override {
    std::vector<std::string> names{name(node_index, "src"), name(node_index, "caps")};
    if (opt_.insert_queue)
      names.push_back(name(node_index, "queue"));
    if (jpeg())
      return names;
    const auto decoder_names = decoder().element_names(node_index);
    names.insert(names.end(), decoder_names.begin(), decoder_names.end());
    return names;
  }

  OutputSpec output_spec(const OutputSpec& input) const override {
    OutputSpec out;
    if (jpeg()) {
      out.payload_type = PayloadType::Encoded;
      out.media_type = "image/jpeg";
      out.format = "JPEG";
      out.width = static_cast<int>(opt_.width);
      out.height = static_cast<int>(opt_.height);
      out.certainty = SpecCertainty::Hint;
      out.note = "USB MJPEG camera input, JPEG frames";
    } else {
      out = decoder().output_spec(input);
      out.payload_type = PayloadType::Image;
      out.note = "USB MJPEG camera input, decoded in place";
    }
    out.fps_num = static_cast<int>(opt_.framerate_num); // exact, e.g. 30000/1001
    out.fps_den = static_cast<int>(opt_.framerate_den);
    return out;
  }

private:
  bool jpeg() const {
    return opt_.output == UsbCameraOutput::Jpeg;
  }

  SimaDecode decoder() const {
    SimaDecodeOptions decode;
    decode.type = SimaDecodeType::MJPEG;
    decode.out_format = camera_.sampling;
    decode.dec_width = static_cast<int>(opt_.width);
    decode.dec_height = static_cast<int>(opt_.height);
    decode.dec_fps =
        static_cast<int>((opt_.framerate_num + opt_.framerate_den / 2) / opt_.framerate_den);
    return SimaDecode(std::move(decode));
  }

  UsbCameraInputOptions opt_;
  internal::uvc::ResolvedCamera camera_;
  std::uint32_t pool_buffers_;
};

void validate_options(const UsbCameraInputOptions& opt, std::uint32_t capture_buffer_count) {
  if (opt.camera_name.empty())
    throw std::invalid_argument(
        "UsbCameraInput requires camera_name: a USB camera ID from usb_camera_ids()");
  if (opt.width == 0 || opt.height == 0 || opt.framerate_num == 0 || opt.framerate_den == 0)
    throw std::invalid_argument("UsbCameraInput requires a nonzero width, height and frame rate");
  if (opt.buffer_name.empty())
    throw std::invalid_argument("UsbCameraInput requires a buffer_name");
  if (opt.output == UsbCameraOutput::Jpeg) {
    // Jpeg mode has no decoder, so no capture pool to size.
    if (capture_buffer_count != 0)
      throw std::invalid_argument(
          "UsbCameraInput capture_buffer_count applies only to UsbCameraOutput::Frames");
    return;
  }
  if (capture_buffer_count > kMaxCaptureBuffers)
    throw std::invalid_argument("UsbCameraInput capture_buffer_count exceeds 64");
  // The default pool is the queue depth plus the decoder's input slots.
  if (capture_buffer_count == 0 && opt.insert_queue &&
      opt.queue_depth > kMaxCaptureBuffers - kDecoderInputSlots)
    throw std::invalid_argument("UsbCameraInput queue_depth exceeds 62: its default capture pool "
                                "(queue_depth + 2 decoder slots) would exceed the 64 buffer limit");
}

} // namespace

namespace internal::uvc {

std::shared_ptr<Node> camera_node(const UsbCameraInputOptions& opt, const ResolvedCamera& camera,
                                  std::uint32_t capture_buffer_count) {
  const std::uint32_t pool = capture_buffer_count
                                 ? capture_buffer_count
                                 : (opt.insert_queue ? opt.queue_depth : 0U) + kDecoderInputSlots;
  return std::make_shared<UvcMjpegCameraNode>(opt, camera, pool);
}

} // namespace internal::uvc

std::vector<std::string> usb_camera_ids() {
  return internal::uvc::camera_ids(internal::uvc::list_nodes());
}

UsbCameraInfo usb_camera_info(const UsbCameraInputOptions& opt) {
  validate_options(opt, 0);
  const auto camera = internal::uvc::resolve_mjpeg_camera(opt.camera_name, opt.width, opt.height,
                                                          opt.framerate_num, opt.framerate_den);
  return {camera.device, camera.sampling};
}

} // namespace simaai::neat

namespace simaai::neat::nodes {

std::shared_ptr<simaai::neat::Node> UsbCameraInput(simaai::neat::UsbCameraInputOptions opt,
                                                   std::uint32_t capture_buffer_count) {
  validate_options(opt, capture_buffer_count);
  // Frames capture lands in CMA buffers that neatdecoder offers; older decoders cannot offer them.
  if (opt.output == UsbCameraOutput::Frames &&
      !element_property_exists("neatdecoder", "input-pool-buffers"))
    throw std::runtime_error("UsbCameraInput requires a neatdecoder with the input-pool-buffers "
                             "property; update the simaai-scheduler package");
  const auto camera = internal::uvc::resolve_mjpeg_camera(opt.camera_name, opt.width, opt.height,
                                                          opt.framerate_num, opt.framerate_den);
  return internal::uvc::camera_node(opt, camera, capture_buffer_count);
}

} // namespace simaai::neat::nodes
