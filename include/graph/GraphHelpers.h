/**
 * @file
 * @ingroup graph
 * @brief Convenience helpers for building simple graphs.
 */
#pragma once

#include "graph/GraphSession.h"
#include "graph/nodes/PipelineNode.h"

#include <memory>
#include <string>
#include <utility>
#include <vector>

namespace simaai::neat::graph::helpers {

/// @brief Wrap a `NodeGroup` in a `PipelineNode` and add it to the graph.
inline NodeId add_pipeline(Graph& g, simaai::neat::NodeGroup group, std::string label = {}) {
  auto node = std::make_shared<simaai::neat::graph::nodes::PipelineNode>(std::move(group),
                                                                         std::move(label));
  return g.add(std::move(node));
}

/// @brief Wrap a builder-`Node` in a `PipelineNode` and add it to the graph.
inline NodeId add_pipeline(Graph& g, std::shared_ptr<simaai::neat::Node> node,
                           std::string label = {}) {
  auto wrapper =
      std::make_shared<simaai::neat::graph::nodes::PipelineNode>(std::move(node), std::move(label));
  return g.add(std::move(wrapper));
}

/// @brief Compile a `Graph` into a runnable `GraphRun` via a transient `GraphSession`.
inline GraphRun build(Graph g, const GraphRunOptions& opt = {}) {
  GraphSession session(std::move(g));
  return session.build(opt);
}

/// @brief Connect a sequence of nodes end-to-end (each `nodes[i-1]` -> `nodes[i]`).
inline void chain(Graph& g, const std::vector<NodeId>& nodes) {
  if (nodes.size() < 2)
    return;
  for (std::size_t i = 1; i < nodes.size(); ++i) {
    g.connect(nodes[i - 1], nodes[i]);
  }
}

/// @brief Initializer-list overload of `chain` for inline node sequences.
inline void chain(Graph& g, std::initializer_list<NodeId> nodes) {
  chain(g, std::vector<NodeId>(nodes));
}

} // namespace simaai::neat::graph::helpers
