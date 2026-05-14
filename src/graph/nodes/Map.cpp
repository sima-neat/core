#include "graph/nodes/Map.h"

#include "graph/nodes/LambdaStage.h"

#include <stdexcept>
#include <utility>

namespace simaai::neat::graph::nodes {

std::shared_ptr<simaai::neat::graph::Node>
Map(SampleMapFn fn, std::string label, StageNodeOptions options, StageNode::OutputSpecFn out_fn) {
  if (!fn) {
    throw std::invalid_argument("Map: function is required");
  }
  auto on_input = [fn = std::move(fn)](StageMsg&& msg, std::vector<StageOutMsg>& out,
                                       const StagePorts& ports) mutable {
    Sample sample = std::move(msg.sample);
    fn(sample);
    const PortId out_port = ports.only_output();
    out.push_back(StageOutMsg{.out_port = out_port, .sample = std::move(sample)});
  };

  return LambdaStageNode("Map", {"in"}, {"out"}, std::move(on_input), std::move(label),
                         std::move(options), std::move(out_fn));
}

std::shared_ptr<simaai::neat::graph::Node> TensorMap(TensorMapFn fn, std::string label,
                                                     StageNodeOptions options,
                                                     StageNode::OutputSpecFn out_fn) {
  if (!fn) {
    throw std::invalid_argument("TensorMap: function is required");
  }
  return Map(
      [fn = std::move(fn)](Sample& sample) mutable {
        if (!sample_has_tensor_list(sample))
          return;
        for (auto& tensor : sample.tensors) {
          fn(sample, tensor);
        }
      },
      std::move(label), std::move(options), std::move(out_fn));
}

std::shared_ptr<simaai::neat::graph::Node> Map(SampleMapTransformFn fn, std::string label,
                                               StageNodeOptions options,
                                               StageNode::OutputSpecFn out_fn) {
  if (!fn) {
    throw std::invalid_argument("Map: function is required");
  }
  auto on_input = [fn = std::move(fn)](StageMsg&& msg, std::vector<StageOutMsg>& out,
                                       const StagePorts& ports) mutable {
    Sample sample = std::move(msg.sample);
    Sample transformed = fn(std::move(sample));
    const PortId out_port = ports.only_output();
    out.push_back(StageOutMsg{.out_port = out_port, .sample = std::move(transformed)});
  };

  return LambdaStageNode("Map", {"in"}, {"out"}, std::move(on_input), std::move(label),
                         std::move(options), std::move(out_fn));
}

} // namespace simaai::neat::graph::nodes
