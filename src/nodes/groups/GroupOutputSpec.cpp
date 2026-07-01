#include "nodes/groups/GroupOutputSpec.h"

#include "nodes/sima/SimaDecode.h"

namespace simaai::neat::nodes::groups {
namespace {

OutputSpec from_caps(const std::string& format, int width, int height, int fps,
                     simaai::neat::CapsMemory memory, const char* note, SpecCertainty certainty) {
  OutputSpec out;
  out.media_type = "video/x-raw";
  out.format = format;
  out.width = width;
  out.height = height;
  out.fps_num = fps;
  out.fps_den = (fps > 0) ? 1 : 1;
  out.memory = (memory == simaai::neat::CapsMemory::SystemMemory) ? "SystemMemory" : "Any";
  out.dtype = "UInt8";
  if (out.format == "RGB" || out.format == "BGR")
    out.layout = "HWC";
  if (out.format == "GRAY8")
    out.layout = "HW";
  if (out.format == "NV12" || out.format == "I420")
    out.layout = "Planar";
  out.certainty = certainty;
  out.note = note ? note : "";
  out.byte_size = expected_byte_size(out);
  return out;
}

} // namespace

OutputSpec HttpMjpegDecodedInputOutputSpec(const HttpMjpegDecodedInputOptions& opt) {
  const auto& c = opt.output_caps;
  if (c.enable) {
    return from_caps(c.format.empty() ? "NV12" : c.format, c.width, c.height, c.fps, c.memory,
                     "HttpMjpegDecodedInput output_caps", SpecCertainty::Derived);
  }

  simaai::neat::SimaDecodeOptions dec;
  dec.type = simaai::neat::SimaDecodeType::MJPEG;
  dec.sima_allocator_type = opt.sima_allocator_type;
  dec.out_format = opt.out_format;
  dec.decoder_name = opt.decoder_name;
  dec.raw_output = opt.decoder_raw_output;
  dec.next_element = opt.decoder_next_element;
  dec.dec_width = opt.dec_width;
  dec.dec_height = opt.dec_height;
  dec.dec_fps = opt.dec_fps;
  dec.num_buffers = opt.num_buffers;
  simaai::neat::SimaDecode decoder(dec);
  OutputSpec out = decoder.output_spec({});
  out.note = "HttpMjpegDecodedInput (hint)";
  return out;
}

OutputSpec ImageInputGroupOutputSpec(const ImageInputGroupOptions& opt) {
  const auto& c = opt.output_caps;
  const bool has_caps = c.enable || c.width > 0 || c.height > 0 || c.fps > 0;
  if (has_caps) {
    return from_caps(c.format.empty() ? "NV12" : c.format, c.width, c.height, c.fps, c.memory,
                     "ImageInputGroup output_caps", SpecCertainty::Derived);
  }

  return from_caps(c.format.empty() ? "NV12" : c.format, -1, -1, opt.fps, c.memory,
                   "ImageInputGroup (hint)", SpecCertainty::Hint);
}

OutputSpec RtspDecodedInputOutputSpec(const RtspDecodedInputOptions& opt) {
  const auto& c = opt.output_caps;
  const bool has_caps = c.enable || c.width > 0 || c.height > 0 || c.fps > 0;
  if (has_caps) {
    return from_caps(c.format.empty() ? "NV12" : c.format, c.width, c.height, c.fps, c.memory,
                     "RtspDecodedInput output_caps", SpecCertainty::Derived);
  }

  return from_caps(opt.out_format.empty() ? "NV12" : opt.out_format, -1, -1, -1, c.memory,
                   "RtspDecodedInput (hint)", SpecCertainty::Hint);
}

OutputSpec VideoInputGroupOutputSpec(const VideoInputGroupOptions& opt) {
  const auto& c = opt.output_caps;
  const bool has_caps = c.enable || c.width > 0 || c.height > 0 || c.fps > 0;
  if (has_caps) {
    return from_caps(c.format.empty() ? "NV12" : c.format, c.width, c.height, c.fps, c.memory,
                     "VideoInputGroup output_caps", SpecCertainty::Derived);
  }

  return from_caps(opt.out_format.empty() ? "NV12" : opt.out_format, -1, -1, -1, c.memory,
                   "VideoInputGroup (hint)", SpecCertainty::Hint);
}

} // namespace simaai::neat::nodes::groups
