/**
 * @file
 * @ingroup nodes_groups
 * @brief UDP sink output group options and builder.
 */
#pragma once

#include "builder/NodeGroup.h"

#include <string>

namespace simaai::neat::nodes::groups {

struct UdpOutputGroupGOptions {
  std::string render_config;
  int width = 0;
  int height = 0;
  int fps = 30;
  int bitrate_kbps = 4000;
  int payload_type = 96;
  int config_interval = 1;
  bool sync_mode = false;
  std::string udp_host = "127.0.0.1";
  int udp_port = 5000;
  bool udp_sync = false;
  bool udp_async = false;
};

simaai::neat::NodeGroup UdpOutputGroupG(const UdpOutputGroupGOptions& opt);

} // namespace simaai::neat::nodes::groups
