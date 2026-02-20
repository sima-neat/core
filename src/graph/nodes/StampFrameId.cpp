#include "graph/nodes/StampFrameId.h"

#include <utility>

namespace simaai::neat::graph::nodes {

void StampFrameId::set_ports(const StagePorts& ports) {
  const PortId only = ports.only_output();
  if (only != kInvalidPort) {
    out_port_ = only;
  }
}

void StampFrameId::on_input(StageMsg&& msg, std::vector<StageOutMsg>& out) {
  Sample sample = std::move(msg.sample);
  if (sample.stream_id.empty()) {
    sample.stream_id = "stream0";
  }
  if (sample.frame_id < 0) {
    auto& next = next_id_[sample.stream_id];
    sample.frame_id = next++;
  }

  const PortId out_port = (out_port_ == kInvalidPort) ? kInvalidPort : out_port_;
  out.push_back(StageOutMsg{.out_port = out_port, .sample = std::move(sample)});
}

std::shared_ptr<simaai::neat::graph::Node> StampFrameIdNode(std::string label) {
  StageNode::StageExecutorFactory factory = []() { return std::make_unique<StampFrameId>(); };
  std::vector<PortDesc> inputs = {PortDesc{.name = "in", .spec = OutputSpec{}}};
  std::vector<PortDesc> outputs = {PortDesc{.name = "out", .spec = OutputSpec{}}};
  return std::make_shared<StageNode>("StampFrameId", std::move(factory), std::move(inputs),
                                     std::move(outputs), std::move(label));
}

} // namespace simaai::neat::graph::nodes
