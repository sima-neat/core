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

class StreamMetadata final : public simaai::neat::graph::StageExecutor {
public:
  explicit StreamMetadata(StreamMetadataDefaults defaults = {});
  void set_ports(const StagePorts& ports) override;
  void on_input(StageMsg&& msg, std::vector<StageOutMsg>& out) override;

private:
  StreamMetadataDefaults defaults_;
  std::unordered_map<std::string, int64_t> next_seq_;
  PortId out_port_ = kInvalidPort;
};

// Convenience: wrap StreamMetadata in a StageNode.
std::shared_ptr<simaai::neat::graph::Node> StreamMetadataNode(StreamMetadataDefaults defaults = {},
                                                              std::string label = {});

} // namespace simaai::neat::graph::nodes
