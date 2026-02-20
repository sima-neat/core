/**
 * @file
 * @ingroup graph
 * @brief Hybrid graph compiler: partitions pipeline segments and propagates OutputSpec.
 */
#pragma once

#include "builder/NodeGroup.h"
#include "builder/OutputSpec.h"
#include "graph/Graph.h"
#include "graph/nodes/StageNode.h"

#include <cstddef>
#include <memory>
#include <string>
#include <unordered_map>
#include <vector>

namespace simaai::neat::graph {

struct EdgeSpec {
  OutputSpec spec;
  bool complete = false;
};

struct CompiledPipelineSegment {
  int id = -1;
  std::vector<NodeId> node_ids;
  simaai::neat::NodeGroup group;
  std::vector<std::size_t> input_edges;
  std::vector<std::size_t> output_edges;
  bool source_like = false;
  OutputSpec input_spec;
  bool input_complete = false;
  OutputSpec output_spec;
  bool output_complete = false;
};

struct CompiledStageNode {
  NodeId node_id = kInvalidNode;
  std::shared_ptr<graph::nodes::StageNode> node;
};

struct CompiledGraph {
  std::vector<CompiledPipelineSegment> pipelines;
  std::vector<CompiledStageNode> stages;
  std::vector<Edge> edges;
  std::vector<EdgeSpec> edge_specs;
  std::vector<std::string> port_names;
};

class Compiler final {
public:
  CompiledGraph compile(const Graph& g) const;

private:
  static bool spec_complete_(const OutputSpec& spec);
};

} // namespace simaai::neat::graph
