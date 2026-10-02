#include "nodes/io/CameraInput.h"

#include "gst/GstHelpers.h"
#include "nodes/io/CameraDiscovery.h"

#include <algorithm>
#include <cctype>
#include <cstdint>
#include <sstream>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace simaai::neat {
namespace {

std::string upper_copy(std::string v) {
  std::transform(v.begin(), v.end(), v.begin(),
                 [](unsigned char c) { return static_cast<char>(std::toupper(c)); });
  return v;
}

std::string gst_quote(const std::string& value) {
  std::string out;
  out.reserve(value.size() + 2);
  out.push_back('"');
  for (char ch : value) {
    if (ch == '\\' || ch == '"')
      out.push_back('\\');
    out.push_back(ch);
  }
  out.push_back('"');
  return out;
}

std::string camera_src_name(int node_index) {
  return "n" + std::to_string(node_index) + "_camera_src";
}

std::string camera_caps_name(int node_index) {
  return "n" + std::to_string(node_index) + "_camera_caps";
}

std::string camera_queue_name(int node_index) {
  return "n" + std::to_string(node_index) + "_camera_queue";
}

std::string camera_bridge_name(int node_index) {
  return "n" + std::to_string(node_index) + "_camera_bridge";
}

std::uint64_t camera_expected_frame_bytes(std::uint32_t width, std::uint32_t height,
                                          const std::string& format) {
  if (width == 0 || height == 0)
    return 0;
  const std::uint64_t pixels = static_cast<std::uint64_t>(width) * height;
  const std::string fmt = upper_copy(format);
  if (fmt == "NV12" || fmt == "I420")
    return pixels * 3U / 2U;
  if (fmt == "RGB" || fmt == "BGR")
    return pixels * 3U;
  if (fmt == "GRAY" || fmt == "GRAY8")
    return pixels;
  if (fmt == "YUYV" || fmt == "UYVY")
    return pixels * 2U;
  return 0;
}

bool libcamerasrc_property_exists(const char* property_name) {
  return element_property_exists("libcamerasrc", property_name);
}

void validate_capture_buffer_count(std::uint32_t capture_buffer_count) {
  if (capture_buffer_count > 128U)
    throw std::invalid_argument(
        "CameraInput capture_buffer_count exceeds Neat's 128-buffer provider limit");
}

std::string camera_caps_string(const CameraInputOptions& opt) {
  std::ostringstream caps;
  caps << "video/x-raw";
  caps << ",format=" << upper_copy(opt.format);
  if (opt.width > 0)
    caps << ",width=" << opt.width;
  if (opt.height > 0)
    caps << ",height=" << opt.height;
  if (opt.framerate_num > 0 && opt.framerate_den > 0)
    caps << ",framerate=" << opt.framerate_num << "/" << opt.framerate_den;
  return caps.str();
}

std::string camera_backend_fragment(const CameraInputOptions& opt, int node_index,
                                    std::uint32_t capture_buffer_count) {
  const std::string src_name = camera_src_name(node_index);
  const std::string caps_name = camera_caps_name(node_index);

  std::ostringstream ss;
  ss << "libcamerasrc name=" << src_name;
  const bool has_external_buffer_mode = libcamerasrc_property_exists("external-buffer-mode");
  if (has_external_buffer_mode) {
    ss << " external-buffer-mode=" << (opt.allow_cpu_fallback ? "preferred" : "required");
  }
  if (capture_buffer_count > 0) {
    if (!libcamerasrc_property_exists("buffer-count")) {
      throw std::runtime_error(
          "CameraInput capture_buffer_count requires a libcamerasrc with the buffer-count "
          "property");
    }
    ss << " buffer-count=" << capture_buffer_count;
  }
  if (!opt.allow_cpu_fallback && !has_external_buffer_mode) {
    throw std::runtime_error(
        "CameraInput strict zero-copy requires a libcamerasrc with the "
        "external-buffer-mode property; set allow_cpu_fallback=true to permit Neat's private "
        "EV74 camera memory bridge to copy when direct capture is unavailable.");
  }
  if (opt.camera_name.has_value() && !opt.camera_name->empty()) {
    ss << " camera-name=" << gst_quote(*opt.camera_name);
  }
  ss << " ! capsfilter name=" << caps_name << " caps=" << gst_quote(camera_caps_string(opt));

  // Keep the memory-policy element adjacent to libcamerasrc. In addition to
  // validating (or adapting) buffers, the bridge answers the source's
  // downstream ALLOCATION query with a standard DMA-BUF pool backed privately
  // by Neat's SiMaAI allocator. A queue must not sit between the producer and
  // the element that owns this negotiation.
  ss << " ! neatcamerabridge name=" << camera_bridge_name(node_index);
  ss << " buffer-name=" << gst_quote(opt.buffer_name);
  if (capture_buffer_count > 0)
    ss << " capture-min-buffers=" << capture_buffer_count;
  // Let the private bridge derive any fallback copy span from each
  // GstBuffer/GstVideoMeta. libcamera buffers may have padded strides or plane
  // offsets, so a tight width*height estimate would truncate later planes.
  ss << " copy-allowed=" << (opt.allow_cpu_fallback ? "true" : "false");

  if (opt.insert_queue) {
    ss << " ! queue name=" << camera_queue_name(node_index);
    if (opt.queue_depth > 0)
      ss << " max-size-buffers=" << opt.queue_depth;
    ss << " max-size-bytes=0 max-size-time=0";
    if (opt.leaky_queue)
      ss << " leaky=downstream";
  }

  return ss.str();
}

class V4L2CameraInputNode final : public Node, public OutputSpecProvider {
public:
  // Only the unified resolver constructs this private node, with validated options.
  explicit V4L2CameraInputNode(CameraInputOptions opt) : opt_(std::move(opt)) {}
  std::string kind() const override {
    return "CameraInput";
  }
  std::string user_label() const override {
    return opt_.device;
  }
  InputRole input_role() const override {
    return InputRole::Source;
  }
  NodeCapsBehavior caps_behavior() const override {
    return NodeCapsBehavior::Static;
  }
  MemoryContract memory_contract() const override {
    return MemoryContract::RequireSystemMemoryMappable;
  }
  std::string buffer_name_hint(int) const override {
    return opt_.buffer_name;
  }
  std::string backend_fragment(int index) const override {
    std::ostringstream out;
    out << "neatv4l2copysrc name=" << camera_src_name(index) << " device=" << gst_quote(opt_.device)
        << " fourcc=\"BA81\"" << " width=" << opt_.width << " height=" << opt_.height
        << " buffer-name=" << gst_quote(opt_.buffer_name)
        << " capture-buffer-count=" << opt_.capture_buffer_count
        << " output-buffer-count=" << opt_.output_buffer_count
        << " frame-timeout-ms=" << opt_.frame_timeout_ms;
    if (opt_.insert_queue) {
      out << " ! queue name=" << camera_queue_name(index)
          << " max-size-buffers=" << opt_.queue_depth << " max-size-bytes=0 max-size-time=0";
      if (opt_.leaky_queue)
        out << " leaky=downstream";
    }
    return out.str();
  }
  std::vector<std::string> element_names(int index) const override {
    std::vector<std::string> result{camera_src_name(index)};
    if (opt_.insert_queue)
      result.push_back(camera_queue_name(index));
    return result;
  }
  OutputSpec output_spec(const OutputSpec&) const override {
    OutputSpec out;
    out.payload_type = PayloadType::Tensor;
    out.media_type = "application/vnd.simaai.tensor";
    out.format = "V4L2_BYTES";
    out.width = static_cast<int>(opt_.width);
    out.height = static_cast<int>(opt_.height);
    out.depth = 1;
    out.memory = "SystemMemory";
    out.dtype = "UInt8";
    out.layout = "";
    out.certainty = SpecCertainty::Hint;
    out.note = "owned flat V4L2 bytes including padding/trailers; geometry is in negotiated caps";
    return out;
  }

private:
  CameraInputOptions opt_;
};

CameraInputOptions resolve_camera_options(CameraInputOptions opt) {
  if (opt.profile != CameraProfile::Default && opt.profile != CameraProfile::MetoakSimor)
    throw std::invalid_argument("CameraInput invalid camera profile");
  if (opt.profile == CameraProfile::Default) {
    if (!opt.device.empty())
      throw std::invalid_argument("CameraInput device requires the MetoakSimor profile; "
                                  "use camera_name with the default libcamera profile");
    validate_capture_buffer_count(opt.capture_buffer_count);
    if (opt.zero_copy.has_value()) {
      if (*opt.zero_copy && opt.allow_cpu_fallback)
        throw std::invalid_argument("CameraInput zero_copy conflicts with allow_cpu_fallback");
      opt.allow_cpu_fallback = !*opt.zero_copy;
    }
    return opt;
  }
#if !defined(__linux__)
  throw std::runtime_error("CameraInput V4L2 backend requires Linux");
#else
  if (opt.camera_name.has_value())
    throw std::invalid_argument("CameraInput cannot combine camera_name with a raw device/profile");
  // The profile owns the default delivery policy. An explicit zero-copy
  // request remains an error, never a silent fallback to owned-copy capture.
  if (opt.zero_copy.value_or(false))
    throw std::invalid_argument("CameraInput MetoakSimor does not support zero_copy=true");
  opt.zero_copy = false;
  if (opt.allow_cpu_fallback)
    throw std::invalid_argument("allow_cpu_fallback is libcamera-only; V4L2 uses zero_copy=false");
  // This profile is the qualified 1920x360 SIMOR wire mode, not every Metoak
  // product. Legacy image defaults are replaced, contradictory overrides fail.
  if (opt.width != 1920 || (opt.height != 1080 && opt.height != 360) ||
      (opt.format != "NV12" && opt.format != "RAW8"))
    throw std::invalid_argument("CameraInput MetoakSimor requires RAW8 BA81 1920x360 wire mode");
  opt.width = 1920;
  opt.height = 360;
  opt.format = "RAW8";
  if (!opt.capture_buffer_count)
    opt.capture_buffer_count = 8;
  if (opt.capture_buffer_count < 4 || opt.capture_buffer_count > 128)
    throw std::invalid_argument("CameraInput MetoakSimor capture_buffer_count must be in [4,128]");
  if (opt.output_buffer_count < 2 || opt.output_buffer_count > 128 || opt.frame_timeout_ms < 1 ||
      opt.frame_timeout_ms > 60000)
    throw std::invalid_argument(
        "CameraInput MetoakSimor output count must be in [2,128] and timeout in [1,60000] ms");
  if (opt.insert_queue && (opt.queue_depth == 0 || opt.queue_depth >= opt.output_buffer_count))
    throw std::invalid_argument("CameraInput MetoakSimor queue_depth must be positive and smaller "
                                "than output_buffer_count");
  if (opt.buffer_name.empty())
    throw std::invalid_argument("CameraInput MetoakSimor requires a nonempty tensor buffer_name");
  // Explicit profiles with a device are construction-only, including in an SDK.
  // Validate the complete capture contract before inspecting any media topology.
  if (opt.device.empty())
    opt.device = camera_discovery::find_simor_device({});
  return opt;
#endif
}

} // namespace

CameraInput::CameraInput(CameraInputOptions opt) : opt_(resolve_camera_options(std::move(opt))) {
  if (opt_.profile == CameraProfile::MetoakSimor) {
    raw_backend_ = std::make_shared<V4L2CameraInputNode>(opt_);
    return;
  }
  if (opt_.format.empty())
    opt_.format = "NV12";
  if (opt_.framerate_den == 0)
    opt_.framerate_den = 1;
  if (opt_.buffer_name.empty())
    opt_.buffer_name = "camera";
}

MemoryContract CameraInput::memory_contract() const {
  return raw_backend_ ? raw_backend_->memory_contract() : MemoryContract::PreferDeviceZeroCopy;
}

std::string CameraInput::user_label() const {
  if (raw_backend_)
    return raw_backend_->user_label();
  if (opt_.camera_name.has_value() && !opt_.camera_name->empty())
    return *opt_.camera_name;
  return opt_.buffer_name;
}

std::string CameraInput::caps_string() const {
  if (raw_backend_)
    return "application/vnd.simaai.tensor";
  return camera_caps_string(opt_);
}

std::string CameraInput::buffer_name_hint(int /*node_index*/) const {
  return opt_.buffer_name;
}

std::string CameraInput::backend_fragment(int node_index) const {
  if (raw_backend_)
    return raw_backend_->backend_fragment(node_index);
  return camera_backend_fragment(opt_, node_index, opt_.capture_buffer_count);
}

std::vector<std::string> CameraInput::element_names(int node_index) const {
  if (raw_backend_)
    return raw_backend_->element_names(node_index);
  std::vector<std::string> names{camera_src_name(node_index), camera_caps_name(node_index),
                                 camera_bridge_name(node_index)};
  if (opt_.insert_queue) {
    names.push_back(camera_queue_name(node_index));
  }
  return names;
}

OutputSpec CameraInput::output_spec(const OutputSpec& input) const {
  if (raw_backend_)
    return dynamic_cast<const OutputSpecProvider&>(*raw_backend_).output_spec(input);
  OutputSpec out;
  out.payload_type = PayloadType::Image;
  out.media_type = "video/x-raw";
  out.format = upper_copy(opt_.format);
  out.width = static_cast<int>(opt_.width);
  out.height = static_cast<int>(opt_.height);
  out.fps_num = static_cast<int>(opt_.framerate_num);
  out.fps_den = static_cast<int>(opt_.framerate_den);
  out.memory = "SimaAI";
  out.dtype = "UInt8";
  if (out.format == "RGB" || out.format == "BGR") {
    out.layout = "HWC";
    out.depth = 3;
  } else if (out.format == "GRAY" || out.format == "GRAY8") {
    out.layout = "HW";
    out.depth = 1;
  } else if (out.format == "NV12" || out.format == "I420") {
    out.layout = "Planar";
    out.depth = 3;
  }
  out.certainty = SpecCertainty::Hint;
  out.note = opt_.allow_cpu_fallback
                 ? "libcamerasrc camera input with negotiated Neat allocation and CPU fallback"
                 : "libcamerasrc camera input with negotiated strict Neat zero-copy";
  out.byte_size = expected_byte_size(out);
  if (out.byte_size == 0) {
    out.byte_size =
        static_cast<std::size_t>(camera_expected_frame_bytes(opt_.width, opt_.height, opt_.format));
  }
  return out;
}

} // namespace simaai::neat

namespace simaai::neat::nodes {

std::shared_ptr<simaai::neat::Node> CameraInput(simaai::neat::CameraInputOptions opt) {
  return std::make_shared<simaai::neat::CameraInput>(std::move(opt));
}

std::shared_ptr<simaai::neat::Node>
CameraInputWithCaptureBuffers(simaai::neat::CameraInputOptions opt,
                              std::uint32_t capture_buffer_count) {
  if (capture_buffer_count)
    opt.capture_buffer_count = capture_buffer_count;
  return CameraInput(std::move(opt));
}

} // namespace simaai::neat::nodes
