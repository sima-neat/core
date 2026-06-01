#include "nodes/groups/UdpH264OutputGroup.h"

#include "nodes/common/Caps.h"
#include "nodes/io/UdpOutput.h"
#include "nodes/sima/H264Parse.h"
#include "nodes/sima/H264Packetize.h"

#include <memory>
#include <string>
#include <vector>

namespace simaai::neat::nodes::groups {
namespace {

std::string gst_escape(const std::string& s) {
  std::string out;
  out.reserve(s.size());
  for (char c : s) {
    if (c == '"' || c == '\\')
      out.push_back('\\');
    out.push_back(c);
  }
  return out;
}

} // namespace

simaai::neat::Graph UdpH264OutputGroup(const UdpH264OutputGroupOptions& opt) {
  std::vector<std::shared_ptr<simaai::neat::Node>> nodes;
  nodes.reserve(4);

  nodes.push_back(nodes::H264Parse(opt.config_interval));
  if (!opt.h264_caps.empty()) {
    const std::string caps = std::string("capsfilter caps=\"") + gst_escape(opt.h264_caps) + "\"";
    nodes.push_back(nodes::Custom(caps));
  }
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
