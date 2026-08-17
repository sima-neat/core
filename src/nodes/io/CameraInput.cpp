#include "nodes/io/CameraInput.h"

#include "gst/GstHelpers.h"

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

class CameraInputCaptureNode final : public Node, public OutputSpecProvider {
public:
  CameraInputCaptureNode(CameraInputOptions opt, std::uint32_t capture_buffer_count)
      : base_(std::move(opt)), capture_buffer_count_(capture_buffer_count) {
    validate_capture_buffer_count(capture_buffer_count_);
  }

  std::string kind() const override {
    return base_.kind();
  }
  std::string user_label() const override {
    return base_.user_label();
  }
  InputRole input_role() const override {
    return base_.input_role();
  }
  NodeCapsBehavior caps_behavior() const override {
    return base_.caps_behavior();
  }
  MemoryContract memory_contract() const override {
    return base_.memory_contract();
  }
  std::string buffer_name_hint(int node_index) const override {
    return base_.buffer_name_hint(node_index);
  }
  std::string backend_fragment(int node_index) const override {
    return camera_backend_fragment(base_.options(), node_index, capture_buffer_count_);
  }
  std::vector<std::string> element_names(int node_index) const override {
    return base_.element_names(node_index);
  }
  OutputSpec output_spec(const OutputSpec& input) const override {
    return base_.output_spec(input);
  }

private:
  CameraInput base_;
  std::uint32_t capture_buffer_count_;
};

} // namespace

CameraInput::CameraInput(CameraInputOptions opt) : opt_(std::move(opt)) {
  if (opt_.format.empty())
    opt_.format = "NV12";
  if (opt_.framerate_den == 0)
    opt_.framerate_den = 1;
  if (opt_.buffer_name.empty())
    opt_.buffer_name = "camera";
}

std::string CameraInput::user_label() const {
  if (opt_.camera_name.has_value() && !opt_.camera_name->empty())
    return *opt_.camera_name;
  return opt_.buffer_name;
}

std::string CameraInput::caps_string() const {
  return camera_caps_string(opt_);
}

std::string CameraInput::buffer_name_hint(int /*node_index*/) const {
  return opt_.buffer_name;
}

std::string CameraInput::backend_fragment(int node_index) const {
  return camera_backend_fragment(opt_, node_index, 0);
}

std::vector<std::string> CameraInput::element_names(int node_index) const {
  std::vector<std::string> names{camera_src_name(node_index), camera_caps_name(node_index),
                                 camera_bridge_name(node_index)};
  if (opt_.insert_queue) {
    names.push_back(camera_queue_name(node_index));
  }
  return names;
}

OutputSpec CameraInput::output_spec(const OutputSpec& /*input*/) const {
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
  return std::make_shared<CameraInputCaptureNode>(std::move(opt), capture_buffer_count);
}

} // namespace simaai::neat::nodes
