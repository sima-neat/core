#include "nodes/sima/SimaDecode.h"

#include "gst/GstHelpers.h"

#include <memory>
#include <sstream>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace simaai::neat {
namespace {

const char* decoder_type_name(SimaDecodeType type) {
  switch (type) {
  case SimaDecodeType::H264:
    return "h264";
  case SimaDecodeType::JPEG:
    return "jpeg";
  case SimaDecodeType::MJPEG:
    return "mjpeg";
  }
  throw std::invalid_argument("SimaDecode: unsupported decode type");
}

std::string public_output_format(const FormatSpec& format) {
  return format.empty() ? std::string("NV12") : format.str();
}

std::string native_decoder_output_format(const FormatSpec& format) {
  const std::string out = public_output_format(format);
  if (out == "NV12") {
    return "NV12";
  }
  if (out == "I420") {
    return "YUV420P";
  }
  return {};
}

void require_raw_output_format_supported(const SimaDecodeOptions& opt) {
  if (!opt.raw_output || !native_decoder_output_format(opt.out_format).empty()) {
    return;
  }
  throw std::invalid_argument("SimaDecode: raw_output supports only NV12 or I420");
}

void append_decoder_properties(std::ostringstream& ss, const SimaDecodeOptions& opt) {
  ss << " sima-allocator-type=" << opt.sima_allocator_type;
  ss << " dec-type=" << decoder_type_name(opt.type);
  if (!opt.decoder_name.empty()) {
    ss << " op-buff-name=" << opt.decoder_name;
  }
  const std::string dec_fmt = native_decoder_output_format(opt.out_format);
  if (!dec_fmt.empty()) {
    ss << " dec-fmt=" << dec_fmt;
  }
  if (!opt.next_element.empty()) {
    ss << " next-element=" << opt.next_element;
  }
  if (opt.dec_width > 0) {
    ss << " dec-width=" << opt.dec_width;
  }
  if (opt.dec_height > 0) {
    ss << " dec-height=" << opt.dec_height;
  }
  if (opt.dec_fps > 0) {
    ss << " dec-fps=" << opt.dec_fps;
  }
  if (opt.num_buffers > 0) {
    ss << " num-buffers=" << opt.num_buffers;
  }
}

} // namespace

SimaDecode::SimaDecode(SimaDecodeOptions opt) : opt_(std::move(opt)) {}

std::string SimaDecode::buffer_name_hint(int node_index) const {
  return opt_.decoder_name.empty() ? ("n" + std::to_string(node_index) + "_decoder")
                                   : opt_.decoder_name;
}

std::string SimaDecode::backend_fragment(int node_index) const {
  require_tensordecoder("SimaDecode::backend_fragment");
  require_raw_output_format_supported(opt_);
  const std::string dec = opt_.decoder_name.empty()
                              ? ("n" + std::to_string(node_index) + "_decoder")
                              : opt_.decoder_name;

  std::ostringstream ss;
  ss << "neatdecoder name=" << dec;
  append_decoder_properties(ss, opt_);
  if (opt_.raw_output) {
    return ss.str();
  }

  const std::string vc = "n" + std::to_string(node_index) + "_videoconvert";
  const std::string cap = "n" + std::to_string(node_index) + "_raw_caps";
  ss << " ! videoconvert name=" << vc << " ! capsfilter name=" << cap << " caps=\""
     << "video/x-raw(memory:SystemMemory),format=" << public_output_format(opt_.out_format) << "\"";
  return ss.str();
}

std::vector<std::string> SimaDecode::element_names(int node_index) const {
  const std::string dec = opt_.decoder_name.empty()
                              ? ("n" + std::to_string(node_index) + "_decoder")
                              : opt_.decoder_name;
  if (opt_.raw_output) {
    return {dec};
  }
  return {dec, "n" + std::to_string(node_index) + "_videoconvert",
          "n" + std::to_string(node_index) + "_raw_caps"};
}

OutputSpec SimaDecode::output_spec(const OutputSpec& input) const {
  OutputSpec out;
  out.media_type = "video/x-raw";
  out.format = public_output_format(opt_.out_format);
  if (opt_.dec_width > 0) {
    out.width = opt_.dec_width;
  } else if (input.width > 0) {
    out.width = input.width;
  }
  if (opt_.dec_height > 0) {
    out.height = opt_.dec_height;
  } else if (input.height > 0) {
    out.height = input.height;
  }
  if (opt_.dec_fps > 0) {
    out.fps_num = opt_.dec_fps;
    out.fps_den = 1;
  } else if (input.fps_num > 0) {
    out.fps_num = input.fps_num;
    out.fps_den = (input.fps_den > 0) ? input.fps_den : 1;
  }
  out.layout = (out.format == "NV12" || out.format == "I420") ? "Planar" : "HWC";
  out.dtype = "UInt8";
  out.memory = opt_.raw_output ? "SimaAI" : "SystemMemory";
  out.certainty = SpecCertainty::Hint;
  out.note = "SimaDecode output";
  out.byte_size = expected_byte_size(out);
  return out;
}

} // namespace simaai::neat

namespace simaai::neat::nodes {

std::shared_ptr<simaai::neat::Node> SimaDecode(SimaDecodeOptions opt) {
  return std::make_shared<simaai::neat::SimaDecode>(std::move(opt));
}

} // namespace simaai::neat::nodes
