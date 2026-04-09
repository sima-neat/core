#include "nodes/io/V4L2Input.h"

#include "gst/GstHelpers.h"

#include <cctype>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace simaai::neat {
namespace {

std::string upper_copy(std::string value) {
  for (char& c : value) {
    c = static_cast<char>(std::toupper(static_cast<unsigned char>(c)));
  }
  return value;
}

bool has_fixed_caps(const V4L2InputOptions& opt) {
  return !opt.media_type.empty() && opt.width > 0 && opt.height > 0;
}

bool has_partial_caps(const V4L2InputOptions& opt) {
  const int set_count =
      (!opt.media_type.empty() ? 1 : 0) + (opt.width > 0 ? 1 : 0) + (opt.height > 0 ? 1 : 0);
  return set_count > 0 && set_count < 3;
}

} // namespace

V4L2Input::V4L2Input(V4L2InputOptions opt) : opt_(std::move(opt)) {
  if (opt_.device.empty()) {
    throw std::invalid_argument("V4L2Input: device path must not be empty");
  }
  if (opt_.fps_n > 0 && opt_.fps_d <= 0) {
    opt_.fps_d = 1;
  }
  if (has_partial_caps(opt_)) {
    std::cerr << "[V4L2Input] warning: partial caps specified (need all of media_type, "
                 "width, height for capsfilter; currently some are missing — caps will be "
                 "unconstrained)\n";
  }
}

std::string V4L2Input::backend_fragment(int node_index) const {
  require_element("v4l2src", "V4L2Input::backend_fragment");

  std::ostringstream ss;
  ss << "v4l2src name=n" << node_index << "_v4l2src device=" << opt_.device;

  if (!opt_.io_mode.empty()) {
    ss << " io-mode=" << opt_.io_mode;
  }
  if (opt_.num_buffers >= 0) {
    ss << " num-buffers=" << opt_.num_buffers;
  }

  if (has_fixed_caps(opt_)) {
    ss << " ! capsfilter name=n" << node_index << "_v4l2src_caps"
       << " caps=\"" << opt_.media_type;
    if (!opt_.format.empty()) {
      ss << ",format=" << opt_.format;
    }
    ss << ",width=" << opt_.width << ",height=" << opt_.height;
    if (opt_.fps_n > 0 && opt_.fps_d > 0) {
      ss << ",framerate=" << opt_.fps_n << "/" << opt_.fps_d;
    }
    ss << "\"";
  }

  return ss.str();
}

std::vector<std::string> V4L2Input::element_names(int node_index) const {
  std::vector<std::string> names;
  names.push_back("n" + std::to_string(node_index) + "_v4l2src");
  if (has_fixed_caps(opt_)) {
    names.push_back("n" + std::to_string(node_index) + "_v4l2src_caps");
  }
  return names;
}

OutputSpec V4L2Input::output_spec(const OutputSpec& /*input*/) const {
  OutputSpec out;
  if (!has_fixed_caps(opt_)) {
    out.note = "V4L2Input output unknown until caps negotiation";
    return out;
  }

  out.media_type = opt_.media_type;
  out.format = opt_.format;
  out.width = opt_.width;
  out.height = opt_.height;
  out.fps_num = (opt_.fps_n > 0) ? opt_.fps_n : 0;
  out.fps_den = (opt_.fps_n > 0 && opt_.fps_d > 0) ? opt_.fps_d : 1;
  out.memory = "SystemMemory";
  out.certainty = SpecCertainty::Hint;
  out.note = "V4L2Input caps hint";

  if (out.media_type == "image/jpeg") {
    out.depth = -1;
    out.note = "V4L2Input compressed JPEG";
    return out;
  }

  if (out.media_type == "video/x-bayer") {
    out.layout = "HW";
    out.depth = 1;
    if (!opt_.format.empty()) {
      const std::string fmt = upper_copy(opt_.format);
      out.dtype = (fmt.find("8") != std::string::npos) ? "UInt8" : "UInt16";
    }
    out.byte_size = expected_byte_size(out);
    return out;
  }

  const std::string fmt = upper_copy(opt_.format);
  if (fmt == "RGB" || fmt == "BGR") {
    out.dtype = "UInt8";
    out.layout = "HWC";
    out.depth = 3;
  } else if (fmt == "GRAY" || fmt == "GRAY8") {
    out.dtype = "UInt8";
    out.layout = "HW";
    out.depth = 1;
  } else if (fmt == "NV12" || fmt == "I420") {
    out.dtype = "UInt8";
    out.layout = "Planar";
    out.depth = 3;
  } else if (!opt_.format.empty()) {
    out.note = "unrecognized format: " + opt_.format;
  }

  out.byte_size = expected_byte_size(out);
  return out;
}

} // namespace simaai::neat

namespace simaai::neat::nodes {

std::shared_ptr<simaai::neat::Node> V4L2Input(V4L2InputOptions opt) {
  return std::make_shared<simaai::neat::V4L2Input>(std::move(opt));
}

} // namespace simaai::neat::nodes
