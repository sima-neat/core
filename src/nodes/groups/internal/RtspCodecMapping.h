/**
 * @file
 * @brief Codec mapping shared by the RTSP decoded-input builder and its OutputSpec.
 *
 * `RtspDecodedInput(...)` builds the graph and `RtspDecodedInputOutputSpec(...)`
 * declares what that graph will emit. Both must derive the encoded source
 * options, the decoder type, and the decoder dimensions from the same
 * `RtspDecodedInputOptions` in exactly the same way: the declared spec is a
 * promise about the built pipeline. Keeping one definition here upholds that
 * invariant by construction instead of by review.
 *
 * FPS resolution deliberately stays out of this header. The builder validates
 * conflicting FPS options and throws; the spec reports a non-validating hint.
 * Unifying them would change when that throw fires, which is a behavior change
 * rather than a deduplication.
 */
#pragma once
#ifndef SIMA_NEAT_INTERNAL
#error "Internal header. Not part of the public API."
#endif

#include "nodes/groups/RtspDecodedInput.h"
#include "nodes/groups/RtspEncodedInput.h"
#include "nodes/sima/SimaDecode.h"

#include <stdexcept>
#include <string>

namespace simaai::neat::nodes::groups {

/**
 * @brief Project decoded-input options onto the encoded source they imply.
 *
 * @param source_fps Resolved source cadence, used for codecs that carry no
 *        codec-specific FPS option. H.264 prefers its own `h264_fps` fallback
 *        and H.265 forwards only an explicitly declared `source_fps`, so
 *        neither reads this argument.
 */
inline RtspEncodedInputOptions encoded_options_from_decoded(const RtspDecodedInputOptions& opt,
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
                       : ((opt.codec == RtspCodec::H265) ? opt.source_fps : source_fps);
  return out;
}

/**
 * @brief Map an RTSP codec path to the decoder type that consumes it.
 *
 * @param group Caller name used in the diagnostic, so a failure names the
 *        public entry point the user called rather than this helper.
 */
inline SimaDecodeType sima_decode_type(RtspCodec type, const char* group) {
  switch (type) {
  case RtspCodec::H264:
    return SimaDecodeType::H264;
  case RtspCodec::MJPEG:
    return SimaDecodeType::MJPEG;
  case RtspCodec::H265:
    return SimaDecodeType::H265;
  }
  throw std::invalid_argument(std::string(group) + ": unsupported codec");
}

/**
 * @brief Decoder width for the H.264 path.
 *
 * An explicit decoder override wins, then the width declared for the parser
 * caps, then the auto-caps fallback. All three may be unset, in which case the
 * decoder negotiates from upstream caps.
 */
inline int h264_dec_width(const RtspDecodedInputOptions& opt) {
  return (opt.dec_width > 0) ? opt.dec_width
                             : ((opt.h264_width > 0) ? opt.h264_width : opt.fallback_h264_width);
}

/// Decoder height for the H.264 path; mirrors h264_dec_width().
inline int h264_dec_height(const RtspDecodedInputOptions& opt) {
  return (opt.dec_height > 0)
             ? opt.dec_height
             : ((opt.h264_height > 0) ? opt.h264_height : opt.fallback_h264_height);
}

/**
 * @brief Decoder width for the MJPEG path.
 *
 * An explicit `dec_width`, or a videoscale tail that will resize anyway, wins.
 * Otherwise the requested output width doubles as the decode width so the
 * decoder emits the final size directly.
 */
inline int mjpeg_dec_width(const RtspDecodedInputOptions& opt) {
  if (opt.dec_width > 0 || opt.use_videoscale)
    return opt.dec_width;
  return (opt.output_caps.width > 0) ? opt.output_caps.width : opt.dec_width;
}

/// Decoder height for the MJPEG path; mirrors mjpeg_dec_width().
inline int mjpeg_dec_height(const RtspDecodedInputOptions& opt) {
  if (opt.dec_height > 0 || opt.use_videoscale)
    return opt.dec_height;
  return (opt.output_caps.height > 0) ? opt.output_caps.height : opt.dec_height;
}

} // namespace simaai::neat::nodes::groups
