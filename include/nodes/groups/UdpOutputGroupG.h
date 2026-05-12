/**
 * @file
 * @ingroup nodes_groups
 * @brief `UdpOutputGroupG` — generic render plus mux plus UDP send.
 *
 * A more generic UDP-output preset than `UdpH264OutputGroup`: bundles an optional
 * render stage (driven by `render_config`), an encode/mux step, and a UDP sink so
 * Sessions can broadcast rendered output without committing to a specific encoder
 * configuration up front. Typical placement: tail end of a Session that needs a
 * configurable render+stream-out path.
 *
 * @see UdpH264OutputGroup
 */
#pragma once

#include "builder/NodeGroup.h"

#include <string>

namespace simaai::neat::nodes::groups {

/**
 * @brief Configuration for `UdpOutputGroupG`.
 *
 * Carries render-stage config, encoder geometry/bitrate, and the UDP destination
 * host/port the group sends to.
 *
 * @ingroup nodes_groups
 */
struct UdpOutputGroupGOptions {
  std::string render_config;          ///< Render-stage configuration string (group-specific).
  int width = 0;                      ///< Encoder input width (0 = inherit from upstream caps).
  int height = 0;                     ///< Encoder input height (0 = inherit from upstream caps).
  int fps = 30;                       ///< Encoder input frame rate.
  int bitrate_kbps = 4000;            ///< Target encoder bitrate in kbps.
  int payload_type = 96;              ///< RTP payload type number.
  int config_interval = 1;            ///< SPS/PPS repeat interval (seconds).
  bool sync_mode = false;             ///< If true, sink elements run in sync (real-time) mode.
  std::string udp_host = "127.0.0.1"; ///< Destination UDP host.
  int udp_port = 5000;                ///< Destination UDP port.
  bool udp_sync = false;              ///< Pass `sync` to the underlying `udpsink` element.
  bool udp_async = false;             ///< Pass `async` to the underlying `udpsink` element.
};

/**
 * @brief Build a NodeGroup that renders, encodes, and sends frames over UDP.
 *
 * More flexible counterpart to `UdpH264OutputGroup`: accepts a render-stage
 * configuration string and explicit encoder geometry, then mux/packetize and send
 * to the configured UDP endpoint.
 *
 * @param opt Render, encoder, and UDP-sink configuration.
 * @return The configured `NodeGroup` ready to be `add()`ed to a Session.
 *
 * @see UdpH264OutputGroup
 * @ingroup nodes_groups
 */
simaai::neat::NodeGroup UdpOutputGroupG(const UdpOutputGroupGOptions& opt);

} // namespace simaai::neat::nodes::groups
