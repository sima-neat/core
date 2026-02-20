#include "nodes/groups/GroupOutputSpec.h"

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
