#include "nodes/groups/RtspDecodedInput.h"

#include "nodes/common/Caps.h"
#include "nodes/common/EncodedCapsFixup.h"
#include "nodes/common/VideoConvert.h"
#include "nodes/common/VideoScale.h"
#include "nodes/sima/SimaDecode.h"

#include <stdexcept>

namespace simaai::neat::nodes::groups {
namespace {

bool caps_enabled(const RtspDecodedInputOptions::OutputCaps& c) {
  return c.enable || c.width > 0 || c.height > 0 || c.fps > 0;
}

bool use_h264_auto_caps(const RtspDecodedInputOptions& opt) {
  return opt.auto_caps_from_stream &&
         (opt.h264_fps <= 0 || opt.h264_width <= 0 || opt.h264_height <= 0);
}

RtspEncodedInputOptions encoded_options_from_decoded(const RtspDecodedInputOptions& opt) {
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
  const bool use_auto_caps = use_h264_auto_caps(opt);
  const int h264_dec_w = (opt.dec_width > 0)
                             ? opt.dec_width
                             : ((opt.h264_width > 0) ? opt.h264_width : opt.fallback_h264_width);
  const int h264_dec_h = (opt.dec_height > 0)
                             ? opt.dec_height
                             : ((opt.h264_height > 0) ? opt.h264_height : opt.fallback_h264_height);
  const int h264_dec_fps =
      (opt.dec_fps > 0) ? opt.dec_fps : ((opt.h264_fps > 0) ? opt.h264_fps : opt.fallback_h264_fps);
  if (opt.codec == RtspCodec::H264 && opt.decoder_raw_output && !use_auto_caps &&
      (h264_dec_w <= 0 || h264_dec_h <= 0 || h264_dec_fps <= 0)) {
    throw std::runtime_error("RtspDecodedInput: decoder_raw_output requires h264 width/height/fps");
  }

  simaai::neat::SimaDecodeOptions dec;
  dec.type = sima_decode_type(opt.codec);
  dec.sima_allocator_type = opt.sima_allocator_type;
  dec.out_format = opt.out_format;
  dec.decoder_name = opt.decoder_name;
  dec.raw_output = opt.decoder_raw_output;
  dec.next_element = opt.decoder_next_element;
  dec.dec_width = (opt.codec == RtspCodec::H264) ? h264_dec_w : opt.dec_width;
  dec.dec_height = (opt.codec == RtspCodec::H264) ? h264_dec_h : opt.dec_height;
  dec.dec_fps = (opt.codec == RtspCodec::H264) ? h264_dec_fps : opt.dec_fps;
  dec.num_buffers = opt.num_buffers;

  simaai::neat::Graph graph;
  graph.add(RtspEncodedInput(encoded_options_from_decoded(opt)));
  if (opt.codec == RtspCodec::MJPEG && (opt.dec_fps > 0 || opt.auto_caps_from_stream)) {
    EncodedCapsFixupOptions fixup{"image/jpeg", opt.dec_fps};
    fixup.use_rtsp_sdp_fps = opt.auto_caps_from_stream;
    graph.add(nodes::EncodedCapsFixup(fixup));
  }
  graph.add(nodes::SimaDecode(dec));

  if (opt.use_videoconvert)
    graph.add(nodes::VideoConvert());
  if (opt.use_videoscale)
    graph.add(nodes::VideoScale());

  if (caps_enabled(opt.output_caps)) {
    const auto& c = opt.output_caps;
    graph.add(nodes::CapsRaw(c.format, c.width, c.height, c.fps, c.memory));
  }

  if (!opt.extra_fragment.empty()) {
    graph.add(nodes::Custom(opt.extra_fragment));
  }

  return graph;
}

} // namespace simaai::neat::nodes::groups
