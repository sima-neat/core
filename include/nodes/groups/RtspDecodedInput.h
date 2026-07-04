/**
 * @file
 * @ingroup nodes_groups
 * @brief `RtspDecodedInput` - RTSP source plus depacketize/parse plus hardware decode.
 *
 * The "live camera" preset: pulls an encoded stream from an `rtsp://` URL,
 * depacketizes and parses it, and runs SiMa hardware decode to emit raw frames
 * for downstream Nodes. H.264 is the default path for source compatibility.
 *
 * @see RtspEncodedInput
 * @see VideoInputGroup
 * @see ImageInputGroup
 * @see SimaDecode
 */
#pragma once

#include "pipeline/Graph.h"
#include "contracts/ContractTypes.h"
#include "nodes/groups/RtspEncodedInput.h"
#include "pipeline/FormatSpec.h"

#include <string>

namespace simaai::neat::nodes::groups {

/**
 * @brief Configuration for `RtspDecodedInput`.
 *
 * Controls the RTSP source, codec-specific depacketize/parse stage, and SiMa
 * hardware-decoder output configuration. Defaults preserve the existing H.264
 * behavior.
 *
 * @ingroup nodes_groups
 */
struct RtspDecodedInputOptions {
  std::string url;       ///< `rtsp://` URL to consume.
  int latency_ms = 200;  ///< Jitter-buffer latency in milliseconds.
  bool tcp = true;       ///< If true, request the RTSP TCP transport.
  int payload_type = 96; ///< RTP payload type number for the H.264 stream.
  int h264_parse_config_interval =
      -1;                   ///< SPS/PPS reinjection interval for the H.264 parser (-1 = default).
  int h264_fps = -1;        ///< Expected FPS injected into the parser caps (-1 = unspecified).
  int h264_width = -1;      ///< Expected width injected into the parser caps (-1 = unspecified).
  int h264_height = -1;     ///< Expected height injected into the parser caps (-1 = unspecified).
  bool insert_queue = true; ///< Insert a queue between depayloader and parser.
  bool sync_mode = false;   ///< If true, sink elements run in sync (real-time) mode.
  bool auto_caps_from_stream =
      true; ///< Try to derive caps automatically from the live stream, including RTSP MJPEG FPS.
  int fallback_h264_fps = -1;    ///< Fallback FPS used if H.264 auto-caps fails.
  int fallback_h264_width = -1;  ///< Fallback width used if auto-caps fails.
  int fallback_h264_height = -1; ///< Fallback height used if auto-caps fails.

  int sima_allocator_type = 2;             ///< SiMa allocator type for decoder output buffers.
  FormatSpec out_format = FormatTag::NV12; ///< Pixel format produced by the decoder.
  std::string decoder_name;                ///< Optional element instance name for the decoder.
  bool decoder_raw_output = true;          ///< Request raw (non-encoded) output from the decoder.
  std::string
      decoder_next_element; ///< Optional next-element selector ("CVU" or "APU") for `neatdecoder`.

  bool use_videoconvert = false; ///< Insert `videoconvert` after decode for format adaptation.
  bool use_videoscale = false;   ///< Insert `videoscale` after decode for resolution adaptation.

  /// Optional explicit output caps applied at the group's tail.
  struct OutputCaps {
    bool enable = false;                 ///< If false, no caps filter is inserted.
    FormatSpec format = FormatTag::NV12; ///< Pixel format the group should advertise.
    int width = -1;                      ///< Output width (-1 = leave unspecified).
    int height = -1;                     ///< Output height (-1 = leave unspecified).
    int fps = -1;                        ///< Output frame rate (-1 = leave unspecified).
    simaai::neat::CapsMemory memory =
        simaai::neat::CapsMemory::SystemMemory; ///< Buffer memory domain.
  } output_caps; ///< Optional explicit output caps applied at the group's tail.

  /// Optional raw GStreamer fragment inserted into the group (advanced use).
  std::string extra_fragment;

  RtspCodec codec =
      RtspCodec::H264;          ///< RTSP codec path to build. Default preserves H.264 behavior.
  bool drop_on_latency = false; ///< If true, ask `rtspsrc` to drop late buffers.
  std::string buffer_mode;      ///< Optional `rtspsrc` buffer-mode value; empty = default.
  int mjpeg_payload_type = 26;  ///< RTP payload type number for the MJPEG/RTP JPEG stream.
  int dec_width = -1;           ///< Decoded frame width override; `-1` = upstream-defined.
  int dec_height = -1;          ///< Decoded frame height override; `-1` = upstream-defined.
  int dec_fps = -1; ///< Decoded frame rate override; for MJPEG also a missing-caps FPS fallback.
  int num_buffers = -1; ///< Decoder output buffer pool size override; `-1` = element default.
  int source_fps = -1;  ///< Declared source stream FPS; feeds source caps and decoder FPS when set.
  bool use_videorate = false; ///< Insert `videorate` after decode to enforce an output FPS.
  int video_rate_fps = -1; ///< FPS requested from `videorate`; `-1` = use the resolved source FPS.
};

/**
 * @brief Build the live-RTSP input Graph: source, depayload+parse, hardware decode.
 *
 * Typical H.264 chain: `RtspEncodedInput(H264)` -> `SimaDecode(H264)`.
 * Typical MJPEG chain: `RtspEncodedInput(MJPEG)` -> `SimaDecode(MJPEG)`.
 *
 * @param opt Configuration (URL, transport, parser fallback caps, decoder output).
 * @return The configured `Graph` ready to be `add()`ed to a Graph.
 *
 * @see RtspEncodedInput
 * @see SimaDecode
 * @ingroup nodes_groups
 */
simaai::neat::Graph RtspDecodedInput(const RtspDecodedInputOptions& opt);

} // namespace simaai::neat::nodes::groups
