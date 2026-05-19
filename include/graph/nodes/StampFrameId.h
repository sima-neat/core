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

/**
 * @brief Stage executor that assigns a monotonically-increasing frame id per stream.
 *
 * For samples lacking a frame id the executor stamps the next id available for that
 * stream. Used for telemetry, debugging, and as a stable key for downstream joins.
 *
 * @see StampFrameIdNode
 * @see StageExecutor
 * @ingroup graph
 */
class StampFrameId final : public simaai::neat::graph::StageExecutor {
public:
  /// Bind the executor to its runtime ports.
  void set_ports(const StagePorts& ports) override;
  /// Stamp the next per-stream frame id onto the incoming sample if it doesn't already have one.
  void on_input(StageMsg&& msg, std::vector<StageOutMsg>& out) override;

private:
  std::unordered_map<std::string, std::int64_t> next_id_;
  PortId out_port_ = kInvalidPort;
};

/**
 * @brief Convenience helper that wraps a `StampFrameId` executor in a `StageNode`.
 *
 * @param label Optional human-readable label.
 * @return Shared pointer to a `graph::Node` ready for insertion in a runtime graph.
 */
std::shared_ptr<simaai::neat::graph::Node> StampFrameIdNode(std::string label = {});

} // namespace simaai::neat::graph::nodes
