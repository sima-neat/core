#include "nodes/groups/RtspDecodedInput.h"

#include "nodes/common/Caps.h"
#include "nodes/common/VideoConvert.h"
#include "nodes/common/VideoRate.h"
#include "nodes/common/VideoScale.h"
#include "nodes/sima/SimaDecode.h"

#include <stdexcept>
#include <string>

namespace simaai::neat::nodes::groups {
namespace {

bool tail_caps_enabled(const RtspDecodedInputOptions::OutputCaps& c) {
  return c.enable;
}

bool use_h264_auto_caps(const RtspDecodedInputOptions& opt) {
  const bool has_source_fps = opt.source_fps > 0 || opt.h264_fps > 0;
  return opt.auto_caps_from_stream &&
         (!has_source_fps || opt.h264_width <= 0 || opt.h264_height <= 0);
}

void require_same_fps_if_set(const char* group, int source_fps, const char* option_name,
                             int option_fps) {
  if (source_fps > 0 && option_fps > 0 && option_fps != source_fps) {
    throw std::invalid_argument(std::string(group) + ": source_fps conflicts with " + option_name);
  }
}

int output_caps_fps_fallback(const RtspDecodedInputOptions& opt) {
  return opt.output_caps.fps;
}

int resolve_h264_source_fps(const RtspDecodedInputOptions& opt) {
  if (opt.source_fps > 0) {
    require_same_fps_if_set("RtspDecodedInput", opt.source_fps, "h264_fps", opt.h264_fps);
    require_same_fps_if_set("RtspDecodedInput", opt.source_fps, "dec_fps", opt.dec_fps);
    return opt.source_fps;
  }
  if (opt.h264_fps > 0) {
    require_same_fps_if_set("RtspDecodedInput", opt.h264_fps, "dec_fps", opt.dec_fps);
    return opt.h264_fps;
  }
  if (opt.dec_fps > 0) {
    require_same_fps_if_set("RtspDecodedInput", opt.dec_fps, "fallback_h264_fps",
                            opt.fallback_h264_fps);
    return opt.dec_fps;
  }
  return opt.fallback_h264_fps;
}

int resolve_mjpeg_source_fps(const RtspDecodedInputOptions& opt) {
  if (opt.source_fps > 0) {
    require_same_fps_if_set("RtspDecodedInput", opt.source_fps, "dec_fps", opt.dec_fps);
    return opt.source_fps;
  }
  if (opt.dec_fps > 0) {
    return opt.dec_fps;
  }
  return output_caps_fps_fallback(opt);
}

int resolve_source_fps(const RtspDecodedInputOptions& opt) {
  return (opt.codec == RtspCodec::H264) ? resolve_h264_source_fps(opt)
                                        : resolve_mjpeg_source_fps(opt);
}

int resolve_video_rate_fps(const RtspDecodedInputOptions& opt, int source_fps) {
  if (opt.video_rate_fps > 0 && !opt.use_videorate) {
    throw std::invalid_argument("RtspDecodedInput: video_rate_fps requires use_videorate=true");
  }
  if (!opt.use_videorate) {
    if (opt.source_fps > 0 && opt.output_caps.fps > 0 && opt.output_caps.fps != source_fps) {
      throw std::invalid_argument(
          "RtspDecodedInput: output_caps.fps conflicts with source_fps; use video_rate_fps for "
          "rate conversion");
    }
    return -1;
  }

  const int fps = (opt.video_rate_fps > 0) ? opt.video_rate_fps : source_fps;
  if (fps <= 0) {
    throw std::invalid_argument(
        "RtspDecodedInput: use_videorate requires video_rate_fps or source_fps");
  }
  if (tail_caps_enabled(opt.output_caps) && opt.output_caps.fps > 0 && opt.output_caps.fps != fps) {
    throw std::invalid_argument("RtspDecodedInput: output_caps.fps conflicts with video_rate_fps");
  }
  return fps;
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
  }
  throw std::invalid_argument("RtspDecodedInput: unsupported codec");
}

} // namespace

simaai::neat::Graph RtspDecodedInput(const RtspDecodedInputOptions& opt) {
  const int source_fps = resolve_source_fps(opt);
  const int video_rate_fps = resolve_video_rate_fps(opt, source_fps);
  const bool use_auto_caps = use_h264_auto_caps(opt);
  const int h264_dec_w = (opt.dec_width > 0)
                             ? opt.dec_width
                             : ((opt.h264_width > 0) ? opt.h264_width : opt.fallback_h264_width);
  const int h264_dec_h = (opt.dec_height > 0)
                             ? opt.dec_height
                             : ((opt.h264_height > 0) ? opt.h264_height : opt.fallback_h264_height);
  const int mjpeg_dec_w = mjpeg_dec_width(opt);
  const int mjpeg_dec_h = mjpeg_dec_height(opt);
  if (opt.codec == RtspCodec::H264 && opt.decoder_raw_output && !use_auto_caps &&
      (h264_dec_w <= 0 || h264_dec_h <= 0 || source_fps <= 0)) {
    throw std::runtime_error("RtspDecodedInput: decoder_raw_output requires h264 width/height/fps");
  }

  simaai::neat::SimaDecodeOptions dec;
  dec.type = sima_decode_type(opt.codec);
  dec.sima_allocator_type = opt.sima_allocator_type;
  dec.out_format = opt.out_format;
  dec.decoder_name = opt.decoder_name;
  dec.raw_output = opt.decoder_raw_output;
  dec.next_element = opt.decoder_next_element;
  dec.dec_width = (opt.codec == RtspCodec::H264) ? h264_dec_w : mjpeg_dec_w;
  dec.dec_height = (opt.codec == RtspCodec::H264) ? h264_dec_h : mjpeg_dec_h;
  dec.dec_fps = source_fps;
  dec.num_buffers = opt.num_buffers;
  dec.input_buffers = opt.decoder_input_buffers;
  dec.decoder_tuning = opt.decoder_tuning;
  dec.memory_opt = opt.decoder_memory_opt;

  simaai::neat::Graph graph;
  graph.add(RtspEncodedInput(encoded_options_from_decoded(opt, source_fps)));
  graph.add(nodes::SimaDecode(dec));

  if (opt.use_videoconvert)
    graph.add(nodes::VideoConvert());
  if (opt.use_videorate)
    graph.add(nodes::VideoRate());
  if (opt.use_videoscale)
    graph.add(nodes::VideoScale());

  const bool tail_caps = tail_caps_enabled(opt.output_caps);
  if (tail_caps || opt.use_videorate) {
    const auto& c = opt.output_caps;
    const auto memory = tail_caps ? c.memory : simaai::neat::CapsMemory::Any;
    const int fps = (video_rate_fps > 0) ? video_rate_fps : (tail_caps ? c.fps : -1);
    graph.add(nodes::CapsRaw(tail_caps ? c.format : simaai::neat::FormatSpec{},
                             tail_caps ? c.width : -1, tail_caps ? c.height : -1, fps, memory));
  }

  if (!opt.extra_fragment.empty()) {
    graph.add(nodes::Custom(opt.extra_fragment));
  }

  return graph;
}

} // namespace simaai::neat::nodes::groups
