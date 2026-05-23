/**
 * @file
 * @ingroup nodes_groups
 * @brief `VideoInputGroup` — file source plus demux plus H.264 parse plus hardware decode.
 *
 * The "feed me a video file" preset: opens a media file, demuxes it, picks the video
 * pad, runs the H.264 parser, and decodes the bitstream with the SiMa hardware
 * decoder. Typical placement: very first Graph in a Graph that should run
 * against a recorded `.mp4` (or similar) video file.
 *
 * @see RtspDecodedInput
 * @see ImageInputGroup
 * @see H264Parse
 */
#pragma once

#include "pipeline/Graph.h"
#include "contracts/ContractTypes.h"
#include "nodes/sima/H264Parse.h"
#include "pipeline/FormatSpec.h"

#include <string>

namespace simaai::neat::nodes::groups {

/**
 * @brief Configuration for `VideoInputGroup`.
 *
 * Controls the source path and demuxer pad selection, the H.264 parser tunables,
 * the SiMa decoder output, and any optional output caps the Graph fragment advertises.
 *
 * @ingroup nodes_groups
 */
struct VideoInputGroupOptions {
  std::string path;              ///< Filesystem path to the video file to read.
  int demux_video_pad_index = 0; ///< Index of the demuxed video pad to follow.
  bool insert_queue = true;      ///< Insert a queue between demux and the parser.
  bool sync_mode = false;        ///< If true, sink elements run in sync (real-time) mode.

  int parse_config_interval = 1; ///< H.264 parser SPS/PPS repeat interval (seconds).
  bool parse_enforce_au = true;  ///< Have the parser align output on access-unit boundaries.

  int sima_allocator_type = 2;             ///< SiMa allocator type for decoder output buffers.
  FormatSpec out_format = FormatTag::NV12; ///< Pixel format produced by the decoder.

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
 * @brief Build the video-file input Graph: file source, demux, H.264 parse, hardware decode.
 *
 * Typical chain: `FileInput` -> `decodebin`/demuxer -> `H264Parse` -> SiMa hardware
 * H.264 decoder -> optional `videoconvert` / `videoscale`. Use this as the head of
 * a Graph that runs against a recorded video file.
 *
 * @param opt Configuration (path, parser tunables, decoder output, caps).
 * @return The configured `Graph` ready to be `add()`ed to a Graph.
 *
 * @see RtspDecodedInput
 * @see H264Parse
 * @ingroup nodes_groups
 */
simaai::neat::Graph VideoInputGroup(const VideoInputGroupOptions& opt);

} // namespace simaai::neat::nodes::groups
