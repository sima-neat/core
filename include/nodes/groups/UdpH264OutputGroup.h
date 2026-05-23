/**
 * @file
 * @ingroup nodes_groups
 * @brief `UdpH264OutputGroup` — H.264 encode plus RTP packetize plus UDP send.
 *
 * Bundles an H.264 encoder, an RTP H.264 payloader, and a `udpsink` so a Graph can
 * stream its rendered output to a remote receiver as RTP/H.264 over UDP. Typical
 * placement: tail end of a Graph that wants to broadcast detection results to a
 * VLC-style viewer or a downstream analytics host.
 *
 * @see UdpOutputGroupG
 * @see ImageToH264RtspGroup
 */
#pragma once

#include "pipeline/Graph.h"

#include <string>

namespace simaai::neat::nodes::groups {

/**
 * @brief Configuration for `UdpH264OutputGroup`.
 *
 * Specifies the H.264 elementary-stream caps, RTP packetization parameters, and the
 * UDP destination host/port the group sends to.
 *
 * @ingroup nodes_groups
 */
struct UdpH264OutputGroupOptions {
  std::string h264_caps;   ///< Optional caps string applied to the H.264 elementary stream.
  int payload_type = 96;   ///< RTP payload type number for H.264.
  int config_interval = 1; ///< SPS/PPS repeat interval (seconds).
  std::string udp_host = "127.0.0.1"; ///< Destination UDP host.
  int udp_port = 5000;                ///< Destination UDP port.
  bool udp_sync = false;              ///< Pass `sync` to the underlying `udpsink` element.
  bool udp_async = false;             ///< Pass `async` to the underlying `udpsink` element.
};

/**
 * @brief Build a Graph that H.264-encodes, RTP-packetizes, and sends frames over UDP.
 *
 * Typical chain: H.264 encoder -> RTP H.264 payloader -> `udpsink`. Use as the tail
 * of a Graph that should broadcast its rendered output as RTP/H.264 over UDP.
 *
 * @param opt Encoder, RTP, and UDP-sink configuration.
 * @return The configured `Graph` ready to be `add()`ed to a Graph.
 *
 * @see UdpOutputGroupG
 * @see ImageToH264RtspGroup
 * @ingroup nodes_groups
 */
simaai::neat::Graph UdpH264OutputGroup(const UdpH264OutputGroupOptions& opt);

} // namespace simaai::neat::nodes::groups
