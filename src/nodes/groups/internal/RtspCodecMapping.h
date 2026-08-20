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

#include <cstdio>
#include <mutex>
#include <stdexcept>
#include <string>

namespace simaai::neat::nodes::groups {

/**
 * @brief RTP payload type a codec uses when the caller declares none.
 *
 * MJPEG carries the static RTP/AVP JPEG payload; the H.26x paths use the first
 * dynamic payload. This is the *input* side default only: the encoded video
 * sender picks its own transmit payload per codec and must not read this.
 */
inline int default_payload_type(RtspCodec codec) {
  return codec == RtspCodec::MJPEG ? 26 : 96;
}

inline void warn_deprecated_h264_payload_type_once() {
  static std::once_flag warned;
  std::call_once(warned, []() {
    std::fprintf(stderr, "[WARN] RtspEncodedInputOptions::h264_payload_type is deprecated. "
                         "Set RtspEncodedInputOptions::payload_type instead.\n");
  });
}

inline void warn_deprecated_mjpeg_payload_type_once() {
  static std::once_flag warned;
  std::call_once(warned, []() {
    // Names the field rather than a struct: the decoded options project onto the
    // encoded ones, so this fires for callers that only ever set the decoded field.
    std::fprintf(stderr, "[WARN] mjpeg_payload_type is deprecated. "
                         "Set payload_type instead.\n");
  });
}

/**
 * @brief Resolve the RTP payload type the encoded group filters on.
 *
 * Resolution order is load-bearing: a declared `payload_type` wins, then the
 * codec's deprecated per-codec field while one still exists, then the codec
 * default. Because only negative values fall through, `payload_type == 0`
 * reaches the depacketizers as 0, which is how a caller disables payload
 * filtering for a stream whose payload number is unknown.
 *
 * The deprecation warning fires only when a legacy field is both consulted and
 * carries a value the codec default would not have produced, so a default
 * configuration stays silent.
 */
inline int resolve_payload_type(const RtspEncodedInputOptions& opt) {
  if (opt.payload_type >= 0) {
    return opt.payload_type;
  }
  const int fallback = default_payload_type(opt.codec);
  if (opt.codec == RtspCodec::H264 && opt.h264_payload_type != fallback) {
    warn_deprecated_h264_payload_type_once();
    return opt.h264_payload_type;
  }
  if (opt.codec == RtspCodec::MJPEG && opt.mjpeg_payload_type != fallback) {
    warn_deprecated_mjpeg_payload_type_once();
    return opt.mjpeg_payload_type;
  }
  return fallback;
}

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
  out.payload_type = opt.payload_type;
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
