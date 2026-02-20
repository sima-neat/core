/**
 * @file
 * @ingroup graph_nodes
 * @brief Stage executor to stamp missing frame_id per stream.
 */
#pragma once

#include "graph/StageExecutor.h"
#include "graph/nodes/StageNode.h"

#include <cstdint>
#include <memory>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

namespace simaai::neat::graph::nodes {

class StampFrameId final : public simaai::neat::graph::StageExecutor {
public:
  void set_ports(const StagePorts& ports) override;
  void on_input(StageMsg&& msg, std::vector<StageOutMsg>& out) override;

private:
  std::unordered_map<std::string, std::int64_t> next_id_;
  PortId out_port_ = kInvalidPort;
};

// Convenience: wrap StampFrameId in a StageNode.
std::shared_ptr<simaai::neat::graph::Node> StampFrameIdNode(std::string label = {});

} // namespace simaai::neat::graph::nodes
