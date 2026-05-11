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

/**
 * @brief Per-edge spec captured by the compiler — the propagated `OutputSpec` and a completeness flag.
 *
 * @ingroup graph
 */
struct EdgeSpec {
  OutputSpec spec;        ///< Propagated output spec on this edge.
  bool complete = false;  ///< True iff the spec is fully specified after propagation.
};

/**
 * @brief Compiled view of a contiguous pipeline-backend segment in the runtime graph.
 *
 * Groups the `NodeId`s and the merged `NodeGroup` belonging to a single pipeline run, plus
 * the input/output edges that bound the segment and the input/output specs flowing through.
 *
 * @ingroup graph
 */
struct CompiledPipelineSegment {
  int id = -1;                                  ///< Stable segment id.
  std::vector<NodeId> node_ids;                 ///< Runtime-graph node ids merged into this segment.
  simaai::neat::NodeGroup group;                ///< Builder `NodeGroup` representing the merged segment.
  std::vector<std::size_t> input_edges;         ///< Indices into `CompiledGraph::edges` feeding the segment.
  std::vector<std::size_t> output_edges;        ///< Indices into `CompiledGraph::edges` leaving the segment.
  bool source_like = false;                     ///< True iff the segment starts with a source-like node.
  OutputSpec input_spec;                        ///< Spec entering the segment.
  bool input_complete = false;                  ///< True iff `input_spec` is fully specified.
  OutputSpec output_spec;                       ///< Spec leaving the segment.
  bool output_complete = false;                 ///< True iff `output_spec` is fully specified.
};

/**
 * @brief Compiled record for a single stage-backend node in the runtime graph.
 *
 * @ingroup graph
 */
struct CompiledStageNode {
  NodeId node_id = kInvalidNode;                       ///< Runtime-graph node id.
  std::shared_ptr<graph::nodes::StageNode> node;       ///< The stage node itself.
};

/**
 * @brief Result produced by `Compiler::compile`: pipelines, stages, edges, specs, and port names.
 *
 * Contains everything the runtime needs to materialise a graph: pipeline segments to launch
 * as GStreamer pipelines, stage nodes to spawn as actors, the edges that connect them, and
 * the port-name table to resolve `PortId`s.
 *
 * @ingroup graph
 */
struct CompiledGraph {
  std::vector<CompiledPipelineSegment> pipelines; ///< Compiled pipeline-backend segments.
  std::vector<CompiledStageNode> stages;          ///< Compiled stage-backend nodes.
  std::vector<Edge> edges;                        ///< All edges as captured in the runtime graph.
  std::vector<EdgeSpec> edge_specs;               ///< Per-edge specs, parallel to `edges`.
  std::vector<std::string> port_names;            ///< Port-name table, indexable by `PortId`.
};

/**
 * @brief Runtime-graph compiler that partitions a `Graph` into pipeline segments and stages.
 *
 * Performs partitioning of contiguous pipeline-backend nodes into pipeline segments, leaves
 * stage-backend nodes intact, and propagates `OutputSpec` along edges so each edge carries
 * its negotiated tensor/format spec.
 *
 * @see CompiledGraph
 * @ingroup graph
 */
class Compiler final {
public:
  /// Compile a runtime `Graph` into a `CompiledGraph`.
  CompiledGraph compile(const Graph& g) const;

private:
  static bool spec_complete_(const OutputSpec& spec);
};

} // namespace simaai::neat::graph
