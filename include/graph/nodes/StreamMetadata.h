/**
 * @file
 * @ingroup graph_nodes
 * @brief Stage executor to ensure stream metadata defaults.
 */
#pragma once

#include "graph/GraphMetadata.h"
#include "graph/StageExecutor.h"
#include "graph/nodes/StageNode.h"

#include <memory>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

namespace simaai::neat::graph::nodes {

/**
 * @brief Stage executor that fills in stream metadata defaults on passing samples.
 *
 * Applies `StreamMetadataDefaults` to incoming samples (stream id, caps, label, port name,
 * media type, format, payload tag, sequence numbers) and forwards them downstream. Used
 * to attach a stable identity to a stream of samples that originate without one.
 *
 * @see StreamMetadataNode
 * @see StreamMetadataDefaults
 * @see StageExecutor
 * @ingroup graph
 */
class StreamMetadata final : public simaai::neat::graph::StageExecutor {
public:
  /// Construct with a `StreamMetadataDefaults` payload describing what to fill in.
  explicit StreamMetadata(StreamMetadataDefaults defaults = {});
  /// Bind the executor to its runtime ports.
  void set_ports(const StagePorts& ports) override;
  /// Apply defaults to the incoming sample and forward downstream.
  void on_input(StageMsg&& msg, std::vector<StageOutMsg>& out) override;

private:
  StreamMetadataDefaults defaults_;
  std::unordered_map<std::string, int64_t> next_seq_;
  PortId out_port_ = kInvalidPort;
};

/**
 * @brief Convenience helper that wraps a `StreamMetadata` executor in a `StageNode`.
 *
 * @param defaults Defaults to apply to passing samples.
 * @param label    Optional human-readable label.
 * @return Shared pointer to a `graph::Node` ready for insertion in a runtime graph.
 */
std::shared_ptr<simaai::neat::graph::Node> StreamMetadataNode(StreamMetadataDefaults defaults = {},
                                                              std::string label = {});

} // namespace simaai::neat::graph::nodes
