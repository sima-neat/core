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

/// Callback invoked by `LambdaStage` for every incoming `StageMsg`; receives the bound
/// `StagePorts`.
using LambdaOnInput = std::function<void(StageMsg&&, std::vector<StageOutMsg>&, const StagePorts&)>;
/// Callback invoked by `LambdaStage` on each periodic tick; gets the current time in nanoseconds
/// and the bound `StagePorts`.
using LambdaOnTick =
    std::function<void(std::int64_t, std::vector<StageOutMsg>&, const StagePorts&)>;

/**
 * @brief Generic stage executor whose behavior is supplied by user-provided callbacks.
 *
 * Wraps a `LambdaOnInput` (required) and an optional `LambdaOnTick` so callers can express
 * stage logic inline without subclassing `StageExecutor`. The bound `StagePorts` is forwarded
 * to each callback so it can resolve port ids by name.
 *
 * @see LambdaStageNode
 * @see StageExecutor
 * @ingroup graph
 */
class LambdaStage final : public simaai::neat::graph::StageExecutor {
public:
  /// Construct a LambdaStage. `on_input` is required; `on_tick` is optional.
  LambdaStage(LambdaOnInput on_input, LambdaOnTick on_tick = {})
      : on_input_(std::move(on_input)), on_tick_(std::move(on_tick)) {
    if (!on_input_) {
      throw std::invalid_argument("LambdaStage: on_input function is required");
    }
  }

  /// Capture the runtime ports for forwarding to user callbacks.
  void set_ports(const StagePorts& ports) override {
    ports_ = ports;
  }

  /// Forward the incoming message to the user-supplied input handler.
  void on_input(StageMsg&& msg, std::vector<StageOutMsg>& out) override {
    on_input_(std::move(msg), out, ports_);
  }

  /// Forward the tick to the user-supplied tick handler if one was provided.
  void on_tick(std::int64_t now_ns, std::vector<StageOutMsg>& out) override {
    if (on_tick_)
      on_tick_(now_ns, out, ports_);
  }

private:
  StagePorts ports_;
  LambdaOnInput on_input_;
  LambdaOnTick on_tick_;
};

/**
 * @brief Convenience helper that wraps a `LambdaStage` in a `StageNode`.
 *
 * @param kind     Node kind label used in diagnostics.
 * @param inputs   Names of the input ports to expose.
 * @param outputs  Names of the output ports to expose.
 * @param on_input Required input handler.
 * @param label    Optional human-readable label.
 * @param options  Optional `StageNode` options (instances, keying, inflight bound).
 * @param out_fn   Optional output-spec resolver.
 * @param on_tick  Optional tick handler.
 * @return Shared pointer to a `graph::Node` ready for insertion in a runtime graph.
 */
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
