#include "nodes/groups/UdpOutputGroupG.h"

#include "nodes/io/UdpOutput.h"
#include "nodes/sima/H264EncodeSima.h"
#include "nodes/sima/H264Parse.h"
#include "nodes/sima/H264Packetize.h"
#include "nodes/common/Caps.h"
#include "nodes/common/Queue.h"

#include <memory>
#include <vector>

namespace simaai::neat::nodes::groups {

simaai::neat::Graph UdpOutputGroupG(const UdpOutputGroupGOptions& opt) {
  std::vector<std::shared_ptr<simaai::neat::Node>> nodes;

  std::string render_fragment = "simaai_sampledemux name=demux "
                                "demux.bbox ! queue ! render.sink_0 "
                                "demux.image ! queue ! render.sink_1 "
                                "neatrender name=render config=\"" +
                                opt.render_config + "\"";
  nodes.push_back(nodes::Custom(render_fragment));

  nodes.push_back(nodes::H264EncodeSima(opt.width, opt.height, opt.fps, opt.bitrate_kbps));
  nodes.push_back(nodes::H264Parse());
  nodes.push_back(nodes::H264Packetize(opt.payload_type, opt.config_interval));

  UdpOutputOptions udp_opt;
  udp_opt.host = opt.udp_host;
  udp_opt.port = opt.udp_port;
  udp_opt.sync = opt.udp_sync;
  udp_opt.async = opt.udp_async;
  nodes.push_back(nodes::UdpOutput(udp_opt));

  simaai::neat::Graph graph;
  for (auto& node : nodes) {
    graph.add(std::move(node));
  }
  return graph;
}

} // namespace simaai::neat::nodes::groups
