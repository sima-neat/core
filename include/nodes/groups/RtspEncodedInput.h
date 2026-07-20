/**
 * @file
 * @ingroup nodes_groups
 * @brief `RtspEncodedInput` - RTSP source plus codec-specific depacketize/parse.
 *
 * Source-owned input Graph for RTSP streams. The group emits parsed encoded
 * frames suitable for `SimaDecode`, video sender paths, or app-level encoded
 * sample handoff.
 */
#pragma once

#include "pipeline/Graph.h"

#include <string>

namespace simaai::neat::nodes::groups {

/// RTSP stream codec path to depayload and parse.
enum class RtspCodec {
  H264 = 0,    ///< RTSP RTP/H.264 path.
  MJPEG = 1,   ///< RTSP RTP/JPEG MJPEG path.
  H265 = 2,    ///< RTSP RTP/H.265 path.
  HEVC = H265, ///< Alias for H.265.
};

/**
 * @brief Configuration for `RtspEncodedInput`.
 *
 * Controls the RTSP source and codec-specific RTP depayloading/parsing before
 * encoded frames leave the group.
 *
 * @ingroup nodes_groups
 */
struct RtspEncodedInputOptions {
  std::string url;                   ///< `rtsp://` URL to consume.
  RtspCodec codec = RtspCodec::H264; ///< Encoded RTSP path to build.

  int latency_ms = 200;         ///< Jitter-buffer latency in milliseconds.
  bool tcp = true;              ///< If true, request the RTSP TCP transport.
  bool drop_on_latency = false; ///< If true, ask `rtspsrc` to drop late buffers.
  std::string buffer_mode;      ///< Optional `rtspsrc` buffer-mode value; empty = default.
  bool insert_queue = true;     ///< Insert queues around depacketize/parse.
  bool sync_mode = false;       ///< If true, sink elements run in sync (real-time) mode.
  int h264_payload_type = 96;   ///< RTP payload type for H.264 streams.
  int mjpeg_payload_type = 26;  ///< RTP payload type for MJPEG/RTP JPEG streams.
  int h264_parse_config_interval =
      -1;               ///< SPS/PPS reinjection interval for H.264 parser (-1 = default).
  int h264_fps = -1;    ///< Expected H.264 FPS injected into parser caps (-1 = unspecified).
  int h264_width = -1;  ///< Expected H.264 width injected into parser caps (-1 = unspecified).
  int h264_height = -1; ///< Expected H.264 height injected into parser caps (-1 = unspecified).
  bool auto_caps_from_stream = true; ///< Try to derive H.264 caps automatically from the stream.
  int fallback_h264_fps = -1;        ///< Fallback H.264 FPS used if auto-caps fails.
  int fallback_h264_width = -1;      ///< Fallback H.264 width used if auto-caps fails.
  int fallback_h264_height = -1;     ///< Fallback H.264 height used if auto-caps fails.
  int source_fps =
      -1; ///< Declared source stream FPS for codec caps repair (-1 = use legacy FPS fields).
  int h265_payload_type = 96; ///< RTP payload type for H.265 streams.
};

/**
 * @brief Build the live-RTSP encoded input Graph.
 *
 * H.264 chain: `rtspsrc` -> RTP H.264 depay/parse -> optional H.264 caps fixup.
 * H.265 chain: `rtspsrc` -> RTP H.265 depay/parse.
 * MJPEG chain: `rtspsrc` -> RTP JPEG depay -> `jpegparse`.
 *
 * @param opt Configuration for source, transport, queueing, and codec framing.
 * @return Configured `Graph` ready to be `add()`ed to another Graph.
 *
 * @see RtspDecodedInput
 * @see SimaDecode
 * @ingroup nodes_groups
 */
simaai::neat::Graph RtspEncodedInput(const RtspEncodedInputOptions& opt);

} // namespace simaai::neat::nodes::groups
