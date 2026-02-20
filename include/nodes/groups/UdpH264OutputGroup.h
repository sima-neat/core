/**
 * @file
 * @ingroup nodes_groups
 * @brief H264-to-UDP output group options and builder.
 */
#pragma once

#include "builder/NodeGroup.h"

#include <string>

namespace simaai::neat::nodes::groups {

struct UdpH264OutputGroupOptions {
  std::string h264_caps;
  int payload_type = 96;
  int config_interval = 1;
  std::string udp_host = "127.0.0.1";
  int udp_port = 5000;
  bool udp_sync = false;
  bool udp_async = false;
};

simaai::neat::NodeGroup UdpH264OutputGroup(const UdpH264OutputGroupOptions& opt);

} // namespace simaai::neat::nodes::groups
