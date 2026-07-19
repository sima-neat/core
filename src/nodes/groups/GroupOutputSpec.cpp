#include "nodes/groups/GroupOutputSpec.h"

#include "nodes/sima/SimaDecode.h"
#include "pipeline/PayloadType.h"

#include <stdexcept>

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

OutputSpec with_fps(OutputSpec out, int fps, const char* note) {
  out.fps_num = fps;
  out.fps_den = 1;
  if (note) {
    out.note = note;
  }
  return out;
}

OutputSpec apply_raw_caps(OutputSpec out, const simaai::neat::FormatSpec& format, int width,
                          int height, int fps, simaai::neat::CapsMemory memory, const char* note) {
  out.media_type = "video/x-raw";
  if (!format.empty())
    out.format = format;
  if (width > 0)
    out.width = width;
  if (height > 0)
    out.height = height;
  if (fps > 0) {
    out.fps_num = fps;
    out.fps_den = 1;
  }
  if (memory == simaai::neat::CapsMemory::SystemMemory)
    out.memory = "SystemMemory";
  out.dtype = "UInt8";
  if (out.format == "RGB" || out.format == "BGR")
    out.layout = "HWC";
  if (out.format == "GRAY8")
    out.layout = "HW";
  if (out.format == "NV12" || out.format == "I420")
    out.layout = "Planar";
  out.certainty = SpecCertainty::Derived;
  if (note)
    out.note = note;
  out.byte_size = expected_byte_size(out);
  return out;
}

RtspEncodedInputOptions encoded_options_from_decoded(const RtspDecodedInputOptions& opt,
                                                     int source_fps) {
  RtspEncodedInputOptions out;
  out.url = opt.url;
  out.codec = opt.codec;
  out.latency_ms = opt.latency_ms;
  out.tcp = opt.tcp;
  out.drop_on_latency = opt.drop_on_latency;
  out.buffer_mode = opt.buffer_mode;
  out.insert_queue = opt.insert_queue;
  out.sync_mode = opt.sync_mode;
  out.h264_payload_type = opt.payload_type;
  out.mjpeg_payload_type = opt.mjpeg_payload_type;
  out.h265_payload_type = opt.payload_type;
  out.h264_parse_config_interval = opt.h264_parse_config_interval;
  out.h264_fps = opt.h264_fps;
  out.h264_width = opt.h264_width;
  out.h264_height = opt.h264_height;
  out.auto_caps_from_stream = opt.auto_caps_from_stream;
  out.fallback_h264_fps = opt.fallback_h264_fps;
  out.fallback_h264_width = opt.fallback_h264_width;
  out.fallback_h264_height = opt.fallback_h264_height;
  out.source_fps = (opt.codec == RtspCodec::H264)
                       ? ((opt.source_fps > 0) ? opt.source_fps : opt.h264_fps)
                       : source_fps;
  return out;
}

SimaDecodeType sima_decode_type(RtspCodec type) {
  switch (type) {
  case RtspCodec::H264:
    return SimaDecodeType::H264;
  case RtspCodec::MJPEG:
    return SimaDecodeType::MJPEG;
  case RtspCodec::H265:
    return SimaDecodeType::H265;
  }
  throw std::invalid_argument("RtspDecodedInputOutputSpec: unsupported codec");
}

int h264_dec_width(const RtspDecodedInputOptions& opt) {
  return (opt.dec_width > 0) ? opt.dec_width
                             : ((opt.h264_width > 0) ? opt.h264_width : opt.fallback_h264_width);
}

int h264_dec_height(const RtspDecodedInputOptions& opt) {
  return (opt.dec_height > 0)
             ? opt.dec_height
             : ((opt.h264_height > 0) ? opt.h264_height : opt.fallback_h264_height);
}

int h264_dec_fps(const RtspDecodedInputOptions& opt) {
  if (opt.source_fps > 0)
    return opt.source_fps;
  if (opt.h264_fps > 0)
    return opt.h264_fps;
  if (opt.dec_fps > 0)
    return opt.dec_fps;
  return opt.fallback_h264_fps;
}

int mjpeg_dec_fps(const RtspDecodedInputOptions& opt) {
  if (opt.source_fps > 0)
    return opt.source_fps;
  if (opt.dec_fps > 0)
    return opt.dec_fps;
  return opt.output_caps.fps;
}

int h265_dec_fps(const RtspDecodedInputOptions& opt) {
  return (opt.source_fps > 0) ? opt.source_fps : opt.dec_fps;
}

int mjpeg_dec_width(const RtspDecodedInputOptions& opt) {
  if (opt.dec_width > 0 || opt.use_videoscale)
    return opt.dec_width;
  return (opt.output_caps.width > 0) ? opt.output_caps.width : opt.dec_width;
}

int mjpeg_dec_height(const RtspDecodedInputOptions& opt) {
  if (opt.dec_height > 0 || opt.use_videoscale)
    return opt.dec_height;
  return (opt.output_caps.height > 0) ? opt.output_caps.height : opt.dec_height;
}

int encoded_h264_fps(const RtspEncodedInputOptions& opt) {
  if (opt.source_fps > 0)
    return opt.source_fps;
  if (opt.h264_fps > 0)
    return opt.h264_fps;
  return opt.fallback_h264_fps;
}

int decoded_tail_fps(const RtspDecodedInputOptions& opt, int source_fps) {
  if (!opt.use_videorate)
    return opt.output_caps.fps;
  return (opt.video_rate_fps > 0) ? opt.video_rate_fps : source_fps;
}

int http_mjpeg_dec_fps(const HttpMjpegDecodedInputOptions& opt) {
  if (opt.source_fps > 0)
    return opt.source_fps;
  return opt.dec_fps;
}

int http_mjpeg_tail_fps(const HttpMjpegDecodedInputOptions& opt, int source_fps) {
  if (!opt.use_videorate)
    return opt.output_caps.fps;
  return (opt.video_rate_fps > 0) ? opt.video_rate_fps : source_fps;
}

OutputSpec http_mjpeg_decoder_spec(const HttpMjpegDecodedInputOptions& opt, int source_fps) {
  simaai::neat::SimaDecodeOptions dec;
  dec.type = simaai::neat::SimaDecodeType::MJPEG;
  dec.sima_allocator_type = opt.sima_allocator_type;
  dec.out_format = opt.out_format;
  dec.decoder_name = opt.decoder_name;
  dec.raw_output = opt.decoder_raw_output;
  dec.next_element = opt.decoder_next_element;
  dec.dec_width = opt.dec_width;
  dec.dec_height = opt.dec_height;
  dec.dec_fps = source_fps;
  dec.num_buffers = opt.num_buffers;
  simaai::neat::SimaDecode decoder(dec);
  OutputSpec out = decoder.output_spec({});
  out.note = "HttpMjpegDecodedInput (hint)";
  return out;
}

OutputSpec rtsp_decoder_spec(const RtspDecodedInputOptions& opt, int source_fps) {
  simaai::neat::SimaDecodeOptions dec;
  dec.type = sima_decode_type(opt.codec);
  dec.sima_allocator_type = opt.sima_allocator_type;
  dec.out_format = opt.out_format;
  dec.decoder_name = opt.decoder_name;
  dec.raw_output = opt.decoder_raw_output;
  dec.next_element = opt.decoder_next_element;
  dec.dec_width = (opt.codec == RtspCodec::H264)
                      ? h264_dec_width(opt)
                      : ((opt.codec == RtspCodec::H265) ? opt.dec_width : mjpeg_dec_width(opt));
  dec.dec_height = (opt.codec == RtspCodec::H264)
                       ? h264_dec_height(opt)
                       : ((opt.codec == RtspCodec::H265) ? opt.dec_height : mjpeg_dec_height(opt));
  dec.dec_fps = source_fps;
  dec.num_buffers = opt.num_buffers;
  simaai::neat::SimaDecode decoder(dec);
  OutputSpec out = decoder.output_spec(
      RtspEncodedInputOutputSpec(encoded_options_from_decoded(opt, source_fps)));
  out.note = "RtspDecodedInput (hint)";
  return out;
}

} // namespace

OutputSpec HttpMjpegDecodedInputOutputSpec(const HttpMjpegDecodedInputOptions& opt) {
  const auto& c = opt.output_caps;
  const int source_fps = http_mjpeg_dec_fps(opt);
  OutputSpec out = http_mjpeg_decoder_spec(opt, source_fps);
  if (c.enable) {
    const int fps = http_mjpeg_tail_fps(opt, source_fps);
    return apply_raw_caps(out, c.format, c.width, c.height, fps, c.memory,
                          "HttpMjpegDecodedInput output_caps");
  }

  if (opt.use_videorate) {
    out = with_fps(out, http_mjpeg_tail_fps(opt, source_fps),
                   "HttpMjpegDecodedInput videorate (hint)");
  }
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

OutputSpec RtspEncodedInputOutputSpec(const RtspEncodedInputOptions& opt) {
  OutputSpec out;
  out.payload_type = PayloadType::Encoded;
  switch (opt.codec) {
  case RtspCodec::H264:
    out.media_type = "video/x-h264";
    out.format = "H264";
    out.width = (opt.h264_width > 0) ? opt.h264_width : opt.fallback_h264_width;
    out.height = (opt.h264_height > 0) ? opt.h264_height : opt.fallback_h264_height;
    out.fps_num = encoded_h264_fps(opt);
    out.fps_den = 1;
    out.note = "RtspEncodedInput H264 (hint)";
    break;
  case RtspCodec::MJPEG:
    out.media_type = "image/jpeg";
    out.format = "JPEG";
    out.fps_num = opt.source_fps;
    out.fps_den = 1;
    out.note = "RtspEncodedInput MJPEG (hint)";
    break;
  case RtspCodec::H265:
    out.media_type = "video/x-h265";
    out.format = "H265";
    out.fps_num = opt.source_fps;
    out.fps_den = 1;
    out.note = "RtspEncodedInput H265 (hint)";
    break;
  }
  if (out.media_type.empty()) {
    throw std::invalid_argument("RtspEncodedInputOutputSpec: unsupported codec");
  }
  out.certainty = SpecCertainty::Hint;
  return out;
}

OutputSpec RtspDecodedInputOutputSpec(const RtspDecodedInputOptions& opt) {
  const auto& c = opt.output_caps;
  const int source_fps =
      (opt.codec == RtspCodec::H264)
          ? h264_dec_fps(opt)
          : ((opt.codec == RtspCodec::H265) ? h265_dec_fps(opt) : mjpeg_dec_fps(opt));
  OutputSpec out = rtsp_decoder_spec(opt, source_fps);
  if (c.enable) {
    const int fps = decoded_tail_fps(opt, source_fps);
    return apply_raw_caps(out, c.format, c.width, c.height, fps, c.memory,
                          "RtspDecodedInput output_caps");
  }

  if (opt.use_videorate) {
    out = with_fps(out, decoded_tail_fps(opt, source_fps), "RtspDecodedInput videorate (hint)");
  }
  return out;
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
