/**
 * @file
 * @ingroup nodes_sima
 * @brief SimaAI RTP H264 payloader node wrapper.
 */
#pragma once

#include "builder/Node.h"
#include "builder/OutputSpec.h"

#include <memory>
#include <vector>

namespace simaai::neat {

class H264Packetize final : public Node, public OutputSpecProvider {
public:
  struct PayloadType {
    int value;
    constexpr PayloadType(int v = 96) : value(v) {}
  };

  struct ConfigInterval {
    int value;
    constexpr ConfigInterval(int v = 1) : value(v) {}
  };

  H264Packetize(PayloadType pt = PayloadType{}, ConfigInterval config_interval = ConfigInterval{});
  std::string kind() const override {
    return "H264Packetize";
  }
  NodeCapsBehavior caps_behavior() const override {
    return NodeCapsBehavior::Dynamic;
  }

  std::string backend_fragment(int node_index) const override;
  std::vector<std::string> element_names(int node_index) const override;
  OutputSpec output_spec(const OutputSpec& input) const override;

  int pt() const {
    return pt_;
  }
  int config_interval() const {
    return config_interval_;
  }

private:
  int pt_ = 96;
  int config_interval_ = 1;
};

} // namespace simaai::neat

namespace simaai::neat::nodes {
std::shared_ptr<simaai::neat::Node>
H264Packetize(simaai::neat::H264Packetize::PayloadType pt = {},
              simaai::neat::H264Packetize::ConfigInterval config_interval = {});
} // namespace simaai::neat::nodes
