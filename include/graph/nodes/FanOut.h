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

struct FanOutOptions {
  std::vector<std::string> outputs;
};

class FanOut final : public simaai::neat::graph::StageExecutor {
public:
  explicit FanOut(FanOutOptions opt);

  void set_ports(const StagePorts& ports) override;
  void on_input(StageMsg&& msg, std::vector<StageOutMsg>& out) override;

private:
  void validate_outputs_(const StagePorts& ports);

  FanOutOptions opt_;
  std::vector<PortId> out_ports_;
  bool validated_ = false;
};

// Convenience: wrap FanOut in a StageNode.
std::shared_ptr<simaai::neat::graph::Node>
FanOutNode(std::vector<std::string> outputs, std::string label = {}, std::string input = "in");

} // namespace simaai::neat::graph::nodes
