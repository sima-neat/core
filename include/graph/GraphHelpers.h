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

inline NodeId add_pipeline(Graph& g, simaai::neat::NodeGroup group, std::string label = {}) {
  auto node = std::make_shared<simaai::neat::graph::nodes::PipelineNode>(std::move(group),
                                                                         std::move(label));
  return g.add(std::move(node));
}

inline NodeId add_pipeline(Graph& g, std::shared_ptr<simaai::neat::Node> node,
                           std::string label = {}) {
  auto wrapper =
      std::make_shared<simaai::neat::graph::nodes::PipelineNode>(std::move(node), std::move(label));
  return g.add(std::move(wrapper));
}

inline GraphRun build(Graph g, const GraphRunOptions& opt = {}) {
  GraphSession session(std::move(g));
  return session.build(opt);
}

inline void chain(Graph& g, const std::vector<NodeId>& nodes) {
  if (nodes.size() < 2)
    return;
  for (std::size_t i = 1; i < nodes.size(); ++i) {
    g.connect(nodes[i - 1], nodes[i]);
  }
}

inline void chain(Graph& g, std::initializer_list<NodeId> nodes) {
  chain(g, std::vector<NodeId>(nodes));
}

} // namespace simaai::neat::graph::helpers
