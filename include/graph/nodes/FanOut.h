/**
 * @file
 * @ingroup graph_nodes
 * @brief Stage executor that fans out a sample to multiple output ports.
 */
#pragma once

#include "graph/StageExecutor.h"
#include "graph/nodes/StageNode.h"

#include <memory>
#include <string>
#include <utility>
#include <vector>

namespace simaai::neat::graph::nodes {

/**
 * @brief Configuration for a `FanOut` stage executor.
 *
 * Lists the names of the output ports the stage will replicate the incoming sample to.
 *
 * @ingroup graph
 */
struct FanOutOptions {
  std::vector<std::string> outputs; ///< Names of output ports to replicate the input sample onto.
};

/**
 * @brief Stage executor that duplicates each incoming sample to multiple output ports.
 *
 * Receives a sample on its single input port and emits a copy on every output port listed
 * in `FanOutOptions::outputs`. Used to fan a single stream out to multiple downstream
 * consumers in the runtime graph.
 *
 * @see FanOutNode
 * @see StageExecutor
 * @ingroup graph
 */
class FanOut final : public simaai::neat::graph::StageExecutor {
public:
  /// Construct a FanOut from the given options.
  explicit FanOut(FanOutOptions opt);

  /// Bind the executor to its runtime ports; resolves output PortIds.
  void set_ports(const StagePorts& ports) override;
  /// Replicate the incoming message onto each configured output port.
  void on_input(StageMsg&& msg, std::vector<StageOutMsg>& out) override;

private:
  void validate_outputs_(const StagePorts& ports);

  FanOutOptions opt_;
  std::vector<PortId> out_ports_;
  bool validated_ = false;
};

/**
 * @brief Convenience helper that wraps a `FanOut` executor in a `StageNode`.
 *
 * @param outputs Names of the output ports to replicate the input onto.
 * @param label   Optional human-readable label.
 * @param input   Name of the single input port (default `"in"`).
 * @return Shared pointer to a `graph::Node` ready for insertion in a runtime graph.
 */
std::shared_ptr<simaai::neat::graph::Node>
FanOutNode(std::vector<std::string> outputs, std::string label = {}, std::string input = "in");

} // namespace simaai::neat::graph::nodes
