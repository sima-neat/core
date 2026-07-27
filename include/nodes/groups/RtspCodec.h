/**
 * @file
 * @ingroup nodes_groups
 * @brief `RtspCodec` - codec selector shared by RTSP input and encoded output Graphs.
 *
 * The selector lives in its own header so an encoded *output* group can name a
 * codec without pulling in an RTSP *input* header and the source options that
 * come with it.
 */
#pragma once

namespace simaai::neat::nodes::groups {

/// RTSP stream codec path to depayload and parse.
enum class RtspCodec {
  H264 = 0,    ///< RTSP RTP/H.264 path.
  AVC = H264,  ///< Alias for H.264.
  MJPEG = 1,   ///< RTSP RTP/JPEG MJPEG path.
  H265 = 2,    ///< RTSP RTP/H.265 path.
  HEVC = H265, ///< Alias for H.265.
};

} // namespace simaai::neat::nodes::groups
