/**
 * @file
 * @ingroup graph_nodes
 * @brief Function-based stage executor wrapper.
 */
#pragma once

#include "graph/StageExecutor.h"
#include "graph/nodes/StageNode.h"

#include <functional>
#include <memory>
#include <string>
#include <utility>
#include <vector>

namespace simaai::neat::graph::nodes {

using LambdaOnInput = std::function<void(StageMsg&&, std::vector<StageOutMsg>&, const StagePorts&)>;
using LambdaOnTick =
    std::function<void(std::int64_t, std::vector<StageOutMsg>&, const StagePorts&)>;

class LambdaStage final : public simaai::neat::graph::StageExecutor {
public:
  LambdaStage(LambdaOnInput on_input, LambdaOnTick on_tick = {})
      : on_input_(std::move(on_input)), on_tick_(std::move(on_tick)) {
    if (!on_input_) {
      throw std::invalid_argument("LambdaStage: on_input function is required");
    }
  }

  void set_ports(const StagePorts& ports) override {
    ports_ = ports;
  }

  void on_input(StageMsg&& msg, std::vector<StageOutMsg>& out) override {
    on_input_(std::move(msg), out, ports_);
  }

  void on_tick(std::int64_t now_ns, std::vector<StageOutMsg>& out) override {
    if (on_tick_)
      on_tick_(now_ns, out, ports_);
  }

private:
  StagePorts ports_;
  LambdaOnInput on_input_;
  LambdaOnTick on_tick_;
};

// Convenience: wrap a LambdaStage in a StageNode.
inline std::shared_ptr<simaai::neat::graph::Node>
LambdaStageNode(std::string kind, std::vector<std::string> inputs, std::vector<std::string> outputs,
                LambdaOnInput on_input, std::string label = {}, StageNodeOptions options = {},
                StageNode::OutputSpecFn out_fn = {}, LambdaOnTick on_tick = {}) {
  LambdaOnInput fn = std::move(on_input);
  LambdaOnTick tick = std::move(on_tick);
  StageNode::StageExecutorFactory factory = [fn, tick]() mutable {
    return std::make_unique<LambdaStage>(fn, tick);
  };

  std::vector<PortDesc> in_ports;
  in_ports.reserve(inputs.size());
  for (auto& name : inputs) {
    in_ports.push_back(PortDesc{.name = std::move(name), .spec = OutputSpec{}});
  }

  std::vector<PortDesc> out_ports;
  out_ports.reserve(outputs.size());
  for (auto& name : outputs) {
    out_ports.push_back(PortDesc{.name = std::move(name), .spec = OutputSpec{}});
  }

  return std::make_shared<StageNode>(std::move(kind), std::move(factory), std::move(in_ports),
                                     std::move(out_ports), std::move(label), std::move(out_fn),
                                     std::move(options));
}

} // namespace simaai::neat::graph::nodes
