#include "nodes/sima/H264Packetize.h"

#include <memory>
#include <sstream>
#include <string>
#include <utility>
#include <vector>

namespace simaai::neat {

H264Packetize::H264Packetize(PayloadType pt, ConfigInterval config_interval)
    : pt_(pt.value), config_interval_(config_interval.value) {}

std::string H264Packetize::backend_fragment(int /*node_index*/) const {
  std::ostringstream ss;
  ss << "rtph264pay name=pay0 pt=" << pt_ << " config-interval=" << config_interval_
     << " timestamp-offset=0";
  return ss.str();
}

std::vector<std::string> H264Packetize::element_names(int /*node_index*/) const {
  return {"pay0"};
}

OutputSpec H264Packetize::output_spec(const OutputSpec& /*input*/) const {
  OutputSpec out;
  out.media_type = "application/x-rtp";
  out.format = "H264";
  out.certainty = SpecCertainty::Hint;
  out.note = "RTP H264 payload";
  return out;
}

} // namespace simaai::neat

namespace simaai::neat::nodes {

std::shared_ptr<simaai::neat::Node>
H264Packetize(simaai::neat::H264Packetize::PayloadType pt,
              simaai::neat::H264Packetize::ConfigInterval config_interval) {
  return std::make_shared<simaai::neat::H264Packetize>(pt, config_interval);
}

} // namespace simaai::neat::nodes
