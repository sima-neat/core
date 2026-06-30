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

InputMemoryPolicy effective_memory_policy(const InputOptions& opt) {
  if (!opt.use_simaai_pool && opt.memory_policy == InputMemoryPolicy::Auto) {
    return InputMemoryPolicy::SystemMemory;
  }
  return opt.memory_policy;
}

std::string build_caps_string(const InputOptions& opt) {
  if (!opt.caps_override.empty()) {
    return opt.caps_override;
  }
  const bool has_fields = !opt.format.empty() || opt.width > 0 || opt.height > 0 || opt.depth > 0;
  if (!has_fields)
    return "";

  std::ostringstream caps;
  const std::string resolved_media = resolve_input_media_type(opt);
  const std::string media = resolved_media.empty() ? "video/x-raw" : resolved_media;
  caps << media;

  if (!opt.format.empty()) {
    const std::string format = normalize_caps_format_for_media(media, opt.format.str());
    caps << ",format=" << format;
  }
  if (media == "application/vnd.simaai.tensor") {
    if (opt.width > 0 && opt.height > 0 && opt.depth > 0) {
      caps << ",rank=3" << ",dim0=" << opt.height << ",dim1=" << opt.width << ",dim2=" << opt.depth;
    } else if (opt.width > 0 && opt.height > 0) {
      caps << ",rank=2" << ",dim0=" << opt.height << ",dim1=" << opt.width;
    } else if (opt.width > 0) {
      caps << ",rank=1" << ",dim0=" << opt.width;
    }
  } else {
    if (opt.width > 0) {
      caps << ",width=" << opt.width;
    }
    if (opt.height > 0) {
      caps << ",height=" << opt.height;
    }
    if (opt.depth > 0) {
      caps << ",depth=" << opt.depth;
    }
  }
  if (opt.fps_n > 0 && opt.fps_d > 0) {
    caps << ",framerate=" << opt.fps_n << "/" << opt.fps_d;
  }

  return caps.str();
}

} // namespace

Input::Input(InputOptions opt) : opt_(std::move(opt)) {}

Input::Input(std::string name) : endpoint_name_(std::move(name)) {}

Input::Input(std::string name, InputOptions opt)
    : opt_(std::move(opt)), endpoint_name_(std::move(name)) {}

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

OutputSpec Input::output_spec(const OutputSpec& input) const {
  OutputSpec out;
  const std::string resolved_media = resolve_input_media_type(opt_);
  out.payload_type = opt_.payload_type != PayloadType::Auto
                         ? opt_.payload_type
                         : (input.payload_type != PayloadType::Auto
                                ? input.payload_type
                                : payload_type_from_media_type(input.media_type));
  out.media_type = !resolved_media.empty()
                       ? resolved_media
                       : (input.media_type.empty() ? std::string("video/x-raw") : input.media_type);
  if (out.payload_type == PayloadType::Auto) {
    out.payload_type = payload_type_from_media_type(out.media_type);
  }
  out.format = !opt_.format.empty() ? opt_.format.str() : input.format;
  out.width = opt_.width > 0 ? opt_.width : input.width;
  out.height = opt_.height > 0 ? opt_.height : input.height;
  out.depth = opt_.depth > 0 ? opt_.depth : input.depth;
  if (opt_.fps_n > 0 && opt_.fps_d > 0) {
    out.fps_num = opt_.fps_n;
    out.fps_den = opt_.fps_d;
  } else {
    out.fps_num = input.fps_num;
    out.fps_den = input.fps_den;
  }
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

  switch (effective_memory_policy(opt_)) {
  case InputMemoryPolicy::Ev74:
  case InputMemoryPolicy::Dms0:
  case InputMemoryPolicy::Auto:
    out.memory = "SimaAI";
    break;
  case InputMemoryPolicy::SystemMemory:
    out.memory = "SystemMemory";
    break;
  }
  out.byte_size = expected_byte_size(out);
  return out;
}

} // namespace simaai::neat

namespace simaai::neat::nodes {

std::shared_ptr<simaai::neat::Node> Input(InputOptions opt) {
  return std::make_shared<simaai::neat::Input>(std::move(opt));
}

std::shared_ptr<simaai::neat::Node> Input(std::string name, InputOptions opt) {
  return std::make_shared<simaai::neat::Input>(std::move(name), std::move(opt));
}

} // namespace simaai::neat::nodes
