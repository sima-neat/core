#include "nodes/sima/H264DecodeSima.h"

#include "gst/GstHelpers.h"
#include <memory>
#include <sstream>
#include <string>
#include <utility>
#include <vector>

namespace simaai::neat {

H264Decode::H264Decode(int sima_allocator_type, std::string out_format, std::string decoder_name,
                       bool raw_output, std::string next_element, int dec_width, int dec_height,
                       int dec_fps, int num_buffers)
    : sima_allocator_type_(sima_allocator_type), out_format_(std::move(out_format)),
      decoder_name_(std::move(decoder_name)), raw_output_(raw_output),
      next_element_(std::move(next_element)), dec_width_(dec_width), dec_height_(dec_height),
      dec_fps_(dec_fps), num_buffers_(num_buffers) {}

std::string H264Decode::backend_fragment(int node_index) const {
  const std::string dec =
      decoder_name_.empty() ? ("n" + std::to_string(node_index) + "_decoder") : decoder_name_;
  const std::string vc = "n" + std::to_string(node_index) + "_videoconvert";
  const std::string cap = "n" + std::to_string(node_index) + "_raw_caps";
  require_tensordecoder("H264Decode::backend_fragment");
  const char* element = "neatdecoder";

  if (raw_output_) {
    std::ostringstream ss;
    ss << element << " name=" << dec << " sima-allocator-type=" << sima_allocator_type_;
    if (!decoder_name_.empty()) {
      ss << " op-buff-name=" << decoder_name_;
    }
    if (!out_format_.empty()) {
      std::string dec_fmt = out_format_;
      if (dec_fmt == "I420")
        dec_fmt = "YUV420P";
      if (dec_fmt == "NV12" || dec_fmt == "YUV420P") {
        ss << " dec-fmt=" << dec_fmt;
      }
    }
    if (!next_element_.empty()) {
      ss << " next-element=" << next_element_;
    }
    if (dec_width_ > 0)
      ss << " dec-width=" << dec_width_;
    if (dec_height_ > 0)
      ss << " dec-height=" << dec_height_;
    if (dec_fps_ > 0)
      ss << " dec-fps=" << dec_fps_;
    if (num_buffers_ > 0)
      ss << " num-buffers=" << num_buffers_;
    return ss.str();
  }

  std::ostringstream caps;
  caps << "video/x-raw(memory:SystemMemory),format=" << out_format_;

  std::ostringstream ss;
  ss << element << " name=" << dec << " sima-allocator-type=" << sima_allocator_type_;
  if (!decoder_name_.empty()) {
    ss << " op-buff-name=" << decoder_name_;
  }
  if (!next_element_.empty()) {
    ss << " next-element=" << next_element_;
  }
  if (dec_width_ > 0)
    ss << " dec-width=" << dec_width_;
  if (dec_height_ > 0)
    ss << " dec-height=" << dec_height_;
  if (dec_fps_ > 0)
    ss << " dec-fps=" << dec_fps_;
  if (num_buffers_ > 0)
    ss << " num-buffers=" << num_buffers_;
  ss << " ! videoconvert name=" << vc << " ! capsfilter name=" << cap << " caps=\"" << caps.str()
     << "\"";
  return ss.str();
}

std::vector<std::string> H264Decode::element_names(int node_index) const {
  const std::string dec =
      decoder_name_.empty() ? ("n" + std::to_string(node_index) + "_decoder") : decoder_name_;
  if (raw_output_) {
    return {dec};
  }
  return {dec, "n" + std::to_string(node_index) + "_videoconvert",
          "n" + std::to_string(node_index) + "_raw_caps"};
}

OutputSpec H264Decode::output_spec(const OutputSpec& input) const {
  OutputSpec out;
  out.media_type = "video/x-raw";
  out.format = out_format_.empty() ? "NV12" : out_format_;
  if (input.width > 0)
    out.width = input.width;
  if (input.height > 0)
    out.height = input.height;
  if (input.fps_num > 0) {
    out.fps_num = input.fps_num;
    out.fps_den = (input.fps_den > 0) ? input.fps_den : 1;
  }
  out.layout = (out.format == "NV12" || out.format == "I420") ? "Planar" : "HWC";
  out.dtype = "UInt8";
  out.memory = raw_output_ ? "SimaAI" : "SystemMemory";
  out.certainty = SpecCertainty::Hint;
  out.note = "H264Decode output";
  out.byte_size = expected_byte_size(out);
  return out;
}

} // namespace simaai::neat

namespace simaai::neat::nodes {

std::shared_ptr<simaai::neat::Node> H264Decode(int sima_allocator_type, std::string out_format,
                                               std::string decoder_name, bool raw_output,
                                               std::string next_element, int dec_width,
                                               int dec_height, int dec_fps, int num_buffers) {
  return std::make_shared<simaai::neat::H264Decode>(
      sima_allocator_type, std::move(out_format), std::move(decoder_name), raw_output,
      std::move(next_element), dec_width, dec_height, dec_fps, num_buffers);
}

} // namespace simaai::neat::nodes
