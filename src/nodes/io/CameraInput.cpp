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

} // namespace

CameraInput::CameraInput(CameraInputOptions opt) : opt_(std::move(opt)) {
  if (opt_.format.empty())
    opt_.format = "NV12";
  if (opt_.framerate_den == 0)
    opt_.framerate_den = 1;
  if (opt_.buffer_name.empty())
    opt_.buffer_name = "camera";
  if (opt_.capture_buffer_count > 32U)
    throw std::invalid_argument("CameraInput capture_buffer_count must be between 0 and 32");
}

std::string CameraInput::user_label() const {
  if (opt_.camera_name.has_value() && !opt_.camera_name->empty())
    return *opt_.camera_name;
  return opt_.buffer_name;
}

std::string CameraInput::caps_string() const {
  std::ostringstream caps;
  caps << "video/x-raw";
  const std::string fmt = upper_copy(opt_.format);
  caps << ",format=" << fmt;
  if (opt_.width > 0)
    caps << ",width=" << opt_.width;
  if (opt_.height > 0)
    caps << ",height=" << opt_.height;
  if (opt_.framerate_num > 0 && opt_.framerate_den > 0) {
    caps << ",framerate=" << opt_.framerate_num << "/" << opt_.framerate_den;
  }
  return caps.str();
}

std::string CameraInput::buffer_name_hint(int /*node_index*/) const {
  return opt_.buffer_name;
}

std::string CameraInput::backend_fragment(int node_index) const {
  const std::string src_name = camera_src_name(node_index);
  const std::string caps_name = camera_caps_name(node_index);

  std::ostringstream ss;
  ss << "libcamerasrc name=" << src_name;
  const bool has_zero_copy = libcamerasrc_property_exists("simaai-zero-copy");
  const bool has_zero_copy_required = libcamerasrc_property_exists("simaai-zero-copy-required");
  if (has_zero_copy) {
    ss << " simaai-zero-copy=true";
  }
  if (opt_.capture_buffer_count > 0) {
    if (!libcamerasrc_property_exists("buffer-count")) {
      throw std::runtime_error(
          "CameraInput capture_buffer_count requires a libcamerasrc with the buffer-count "
          "property");
    }
    ss << " buffer-count=" << opt_.capture_buffer_count;
  }
  if (!opt_.allow_cpu_fallback) {
    if (!has_zero_copy || !has_zero_copy_required) {
      throw std::runtime_error(
          "CameraInput strict zero-copy requires a libcamerasrc with simaai-zero-copy and "
          "simaai-zero-copy-required properties; set allow_cpu_fallback=true to use Neat's "
          "private EV74 camera memory bridge.");
    }
    ss << " simaai-zero-copy-required=true";
  }
  if (opt_.camera_name.has_value() && !opt_.camera_name->empty()) {
    ss << " camera-name=" << gst_quote(*opt_.camera_name);
  }
  ss << " ! capsfilter name=" << caps_name << " caps=" << gst_quote(caps_string());

  // Keep the memory-policy element adjacent to libcamerasrc.  In addition to
  // validating (or adapting) buffers, the bridge answers the source's
  // downstream ALLOCATION query with a standard DMA-BUF pool backed privately
  // by Neat's SiMaAI allocator. A queue must not sit between the producer and
  // the element that owns this negotiation.
  ss << " ! neatcamerabridge name=" << camera_bridge_name(node_index);
  ss << " buffer-name=" << gst_quote(opt_.buffer_name);
  const std::uint32_t bridgeBufferCount = opt_.capture_buffer_count > 0
                                              ? opt_.capture_buffer_count
                                              : std::max<std::uint32_t>(2U, opt_.queue_depth);
  ss << " num-buffers=" << bridgeBufferCount;
  // Let the private bridge derive any fallback copy span from each
  // GstBuffer/GstVideoMeta. libcamera buffers may have padded strides or plane
  // offsets, so a tight width*height estimate would truncate later planes.
  ss << " copy-allowed=" << (opt_.allow_cpu_fallback ? "true" : "false");

  if (opt_.insert_queue) {
    ss << " ! queue name=" << camera_queue_name(node_index);
    if (opt_.queue_depth > 0) {
      ss << " max-size-buffers=" << opt_.queue_depth;
    }
    ss << " max-size-bytes=0 max-size-time=0";
    if (opt_.leaky_queue) {
      ss << " leaky=downstream";
    }
  }

  return ss.str();
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

} // namespace simaai::neat::nodes
