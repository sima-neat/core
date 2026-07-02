/**
 * @file
 * @ingroup nodes_groups
 * @brief `HttpMjpegDecodedInput` - HTTP multipart MJPEG source plus native decode.
 *
 * Source-owned input Graph for HTTP/HTTPS multipart MJPEG streams. The group
 * reads from an HTTP source, frames JPEG parts, and decodes them through the
 * native SiMa decoder path.
 *
 * @see RtspDecodedInput
 * @see SimaDecode
 */
#pragma once

#include "contracts/ContractTypes.h"
#include "pipeline/FormatSpec.h"
#include "pipeline/Graph.h"

#include <string>

namespace simaai::neat::nodes::groups {

/**
 * @brief Configuration for `HttpMjpegDecodedInput`.
 *
 * Controls the HTTP source, multipart framing, JPEG parsing, native MJPEG
 * decoder output, and optional tail caps.
 *
 * @ingroup nodes_groups
 */
struct HttpMjpegDecodedInputOptions {
  std::string url;                ///< HTTP/HTTPS multipart MJPEG URL to consume.
  int timeout_seconds = 15;       ///< Blocking HTTP I/O timeout in seconds; `0` disables timeout.
  int retries = 3;                ///< Maximum HTTP retries before failing; `-1` means infinite.
  bool is_live = true;            ///< Mark the HTTP source as live.
  bool do_timestamp = true;       ///< Timestamp outgoing source buffers with stream time.
  std::string user_agent;         ///< Optional HTTP User-Agent override.
  std::string multipart_boundary; ///< Optional multipart boundary override; empty = auto-detect.
  bool multipart_single_stream = false; ///< If true, assume multipart content type is stable.
  bool insert_queue = true;             ///< Insert queues around source/framing and decode.
  bool sync_mode = false;               ///< If true, sink elements run in sync mode.

  int sima_allocator_type = 2;             ///< SiMa allocator type for decoder output buffers.
  FormatSpec out_format = FormatTag::NV12; ///< Pixel format produced by the decoder.
  std::string decoder_name;                ///< Optional element instance name for the decoder.
  bool decoder_raw_output = true;          ///< Request decoder-native raw output.
  std::string
      decoder_next_element; ///< Optional next-element selector ("CVU" or "APU") for `neatdecoder`.
  int dec_width = -1;       ///< Decoded frame width override; `-1` = upstream-defined.
  int dec_height = -1;      ///< Decoded frame height override; `-1` = upstream-defined.
  int dec_fps = -1;         ///< Decoded frame rate override; also fixes missing input caps.
  int num_buffers = -1;     ///< Decoder output buffer pool size override; `-1` = element default.

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

  bool ssl_strict = true; ///< If false, disable strict TLS certificate validation.
};

/**
 * @brief Build the HTTP MJPEG input Graph: HTTP source, multipart demux, JPEG parse, decode.
 *
 * Typical chain: `souphttpsrc` -> `multipartdemux` -> `jpegparse` -> SiMa
 * native MJPEG decoder -> optional `videoconvert` / `videoscale`.
 *
 * @param opt Configuration (URL, source/framing options, decoder output, caps).
 * @return The configured `Graph` ready to be `add()`ed to a Graph.
 *
 * @see RtspDecodedInput
 * @see SimaDecode
 * @ingroup nodes_groups
 */
simaai::neat::Graph HttpMjpegDecodedInput(const HttpMjpegDecodedInputOptions& opt);

} // namespace simaai::neat::nodes::groups
