/**
 * @file
 * @ingroup graph_nodes
 * @brief Convenience helpers to build graph nodes.
 */
#pragma once

#include "graph/nodes/PipelineNode.h"
#include "graph/nodes/StageNode.h"

#include <memory>
#include <string>
#include <utility>
#include <vector>

namespace simaai::neat::graph::nodes {

/// @brief Wrap a `NodeGroup` (by const ref) in a `PipelineNode` graph node.
inline std::shared_ptr<simaai::neat::graph::Node> Pipeline(const simaai::neat::NodeGroup& group,
                                                           std::string label = {}) {
  return std::make_shared<PipelineNode>(group, std::move(label));
}

/// @brief Wrap a `NodeGroup` (by rvalue) in a `PipelineNode` graph node.
inline std::shared_ptr<simaai::neat::graph::Node> Pipeline(simaai::neat::NodeGroup&& group,
                                                           std::string label = {}) {
  return std::make_shared<PipelineNode>(std::move(group), std::move(label));
}

/// @brief Wrap a builder-`Node` shared pointer in a `PipelineNode` graph node.
inline std::shared_ptr<simaai::neat::graph::Node> Pipeline(std::shared_ptr<simaai::neat::Node> node,
                                                           std::string label = {}) {
  return std::make_shared<PipelineNode>(std::move(node), std::move(label));
}

/// @brief Construct a `StageNode` graph node from an executor factory and port descriptions.
inline std::shared_ptr<simaai::neat::graph::Node>
Stage(std::string kind, StageNode::StageExecutorFactory factory, std::vector<PortDesc> inputs,
      std::vector<PortDesc> outputs, std::string label = {}, StageNode::OutputSpecFn out_fn = {},
      StageNodeOptions options = {}) {
  return std::make_shared<StageNode>(std::move(kind), std::move(factory), std::move(inputs),
                                     std::move(outputs), std::move(label), std::move(out_fn),
                                     std::move(options));
}

} // namespace simaai::neat::graph::nodes
