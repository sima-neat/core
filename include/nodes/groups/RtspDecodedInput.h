/**
 * @file
 * @ingroup nodes_groups
 * @brief `RtspDecodedInput` — RTSP source plus H.264 depacketize/parse plus hardware decode.
 *
 * The "live camera" preset: pulls an H.264 stream from an `rtsp://` URL, depacketizes
 * and parses the bitstream, and runs SiMa hardware H.264 decode to emit raw frames
 * for downstream Nodes. Typical placement: very first NodeGroup in a Session that
 * consumes a live IP camera feed.
 *
 * @see VideoInputGroup
 * @see ImageInputGroup
 * @see H264Parse
 */
#pragma once

#include "builder/NodeGroup.h"
#include "contracts/ContractTypes.h"
#include "pipeline/FormatSpec.h"

#include <string>

namespace simaai::neat::nodes::groups {

/**
 * @brief Configuration for `RtspDecodedInput`.
 *
 * Controls the RTSP source (URL, transport, latency), the H.264 parse stage's
 * fallback caps when the live stream omits them, and the SiMa hardware-decoder
 * output configuration.
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
  bool auto_caps_from_stream = true; ///< Try to derive caps automatically from the live stream.
  int fallback_h264_fps = -1;        ///< Fallback FPS used if auto-caps fails.
  int fallback_h264_width = -1;      ///< Fallback width used if auto-caps fails.
  int fallback_h264_height = -1;     ///< Fallback height used if auto-caps fails.

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
};

/**
 * @brief Build the live-RTSP input NodeGroup: source, depayload+parse, hardware H.264 decode.
 *
 * Typical chain: `rtspsrc` -> RTP H.264 depayloader -> `H264Parse` -> SiMa hardware
 * H.264 decoder -> optional `videoconvert` / `videoscale`. Use this as the head of
 * a Session that runs detection or analytics on a live IP camera feed.
 *
 * @param opt Configuration (URL, transport, parser fallback caps, decoder output).
 * @return The configured `NodeGroup` ready to be `add()`ed to a Session.
 *
 * @see VideoInputGroup
 * @see H264Parse
 * @ingroup nodes_groups
 */
simaai::neat::NodeGroup RtspDecodedInput(const RtspDecodedInputOptions& opt);

} // namespace simaai::neat::nodes::groups
