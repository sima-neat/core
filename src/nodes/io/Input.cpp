#include "nodes/io/Input.h"

#include <sstream>
#include <string>
#include <utility>
#include <vector>

namespace simaai::neat {
namespace {

const char* stream_type_string(int stream_type) {
  switch (stream_type) {
  case 1:
    return "seekable";
  case 2:
    return "random-access";
  case 0:
  default:
    return "stream";
  }
}

std::string build_caps_string(const InputOptions& opt) {
  if (!opt.caps_override.empty()) {
    return opt.caps_override;
  }
  const bool has_fields = !opt.format.empty() || opt.width > 0 || opt.height > 0 || opt.depth > 0;
  if (!has_fields)
    return "";

  std::ostringstream caps;
  const std::string media = opt.media_type.empty() ? "video/x-raw" : opt.media_type;
  caps << media;

  if (!opt.format.empty()) {
    caps << ",format=" << opt.format;
  }
  if (opt.width > 0) {
    caps << ",width=" << opt.width;
  }
  if (opt.height > 0) {
    caps << ",height=" << opt.height;
  }
  if (opt.depth > 0) {
    caps << ",depth=" << opt.depth;
  }
  if (opt.fps_n > 0 && opt.fps_d > 0) {
    caps << ",framerate=" << opt.fps_n << "/" << opt.fps_d;
  }

  return caps.str();
}

} // namespace

Input::Input(InputOptions opt) : opt_(std::move(opt)) {}

std::string Input::caps_string() const {
  return build_caps_string(opt_);
}

std::string Input::buffer_name_hint(int /*node_index*/) const {
  return opt_.buffer_name;
}

std::string Input::backend_fragment(int /*node_index*/) const {
  std::ostringstream ss;
  ss << "appsrc name=mysrc";
  ss << " is-live=" << (opt_.is_live ? "true" : "false");
  ss << " format=time";
  ss << " do-timestamp=" << (opt_.do_timestamp ? "true" : "false");
  ss << " block=" << (opt_.block ? "true" : "false");
  ss << " stream-type=" << stream_type_string(opt_.stream_type);
  ss << " max-bytes=" << opt_.max_bytes;

  const std::string caps = caps_string();
  if (!caps.empty()) {
    ss << " caps=\"" << caps << "\"";
  }

  return ss.str();
}

std::vector<std::string> Input::element_names(int /*node_index*/) const {
  return {"mysrc"};
}

OutputSpec Input::output_spec(const OutputSpec& /*input*/) const {
  OutputSpec out;
  out.media_type = opt_.media_type.empty() ? "video/x-raw" : opt_.media_type;
  out.format = opt_.format;
  out.width = opt_.width;
  out.height = opt_.height;
  out.depth = opt_.depth;
  out.certainty = SpecCertainty::Derived;
  out.note = "Input options";

  if (out.media_type == "video/x-raw") {
    out.dtype = "UInt8";
    if (out.format == "RGB" || out.format == "BGR") {
      out.layout = "HWC";
      if (out.depth <= 0)
        out.depth = 3;
    } else if (out.format == "GRAY8") {
      out.layout = "HW";
      if (out.depth <= 0)
        out.depth = 1;
    } else if (out.format == "NV12" || out.format == "I420") {
      out.layout = "Planar";
    }
  } else if (out.media_type == "application/vnd.simaai.tensor") {
    if (out.format == "FP32")
      out.dtype = "Float32";
    if (out.depth > 0 && out.width > 0 && out.height > 0) {
      out.layout = "HWC";
    }
  }

  out.memory = opt_.use_simaai_pool ? "SimaAI" : "SystemMemory";
  out.byte_size = expected_byte_size(out);
  return out;
}

} // namespace simaai::neat

namespace simaai::neat::nodes {

std::shared_ptr<simaai::neat::Node> Input(InputOptions opt) {
  return std::make_shared<simaai::neat::Input>(std::move(opt));
}

} // namespace simaai::neat::nodes
