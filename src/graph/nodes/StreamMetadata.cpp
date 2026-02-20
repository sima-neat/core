#include "graph/nodes/StreamMetadata.h"

#include <utility>

namespace simaai::neat::graph::nodes {

StreamMetadata::StreamMetadata(StreamMetadataDefaults defaults) : defaults_(std::move(defaults)) {}

void StreamMetadata::set_ports(const StagePorts& ports) {
  const PortId only = ports.only_output();
  if (only != kInvalidPort) {
    out_port_ = only;
  }
}

void StreamMetadata::on_input(StageMsg&& msg, std::vector<StageOutMsg>& out) {
  Sample sample = std::move(msg.sample);
  ensure_stream_metadata(sample, defaults_, &next_seq_);
  const PortId out_port = (out_port_ == kInvalidPort) ? kInvalidPort : out_port_;
  out.push_back(StageOutMsg{.out_port = out_port, .sample = std::move(sample)});
}

std::shared_ptr<simaai::neat::graph::Node> StreamMetadataNode(StreamMetadataDefaults defaults,
                                                              std::string label) {
  StageNode::StageExecutorFactory factory = [defaults = std::move(defaults)]() mutable {
    return std::make_unique<StreamMetadata>(std::move(defaults));
  };
  std::vector<PortDesc> inputs = {PortDesc{.name = "in", .spec = OutputSpec{}}};
  std::vector<PortDesc> outputs = {PortDesc{.name = "out", .spec = OutputSpec{}}};
  return std::make_shared<StageNode>("StreamMetadata", std::move(factory), std::move(inputs),
                                     std::move(outputs), std::move(label));
}

} // namespace simaai::neat::graph::nodes
