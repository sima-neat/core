/**
 * @file
 * @ingroup builder
 * @brief Linear container of Nodes shared between builder and runtime.
 */
// include/builder/NodeGroup.h
#pragma once

#include <memory>
#include <utility>
#include <vector>

#include "builder/Node.h"

namespace simaai::neat {

/**
 * @brief Simple wrapper around a linear list of nodes.
 *
 * Builder/Graph are intentionally STL-only; NodeGroup is the shared container
 * between them and the pipeline runtime.
 */
class NodeGroup final {
public:
  using NodePtr = std::shared_ptr<Node>;

  NodeGroup() = default;

  explicit NodeGroup(std::vector<NodePtr>&& nodes) : nodes_(std::move(nodes)) {}

  explicit NodeGroup(const std::vector<NodePtr>& nodes) : nodes_(nodes) {}

  const std::vector<NodePtr>& nodes() const noexcept {
    return nodes_;
  }

  std::vector<NodePtr>& nodes_mut() noexcept {
    return nodes_;
  }

  NodeCapsBehavior caps_behavior() const {
    bool any_static = false;
    for (const auto& node : nodes_) {
      if (!node)
        continue;
      const auto behavior = node->caps_behavior();
      if (behavior == NodeCapsBehavior::Static)
        any_static = true;
    }
    return any_static ? NodeCapsBehavior::Static : NodeCapsBehavior::Dynamic;
  }

  bool is_static_group() const {
    return caps_behavior() == NodeCapsBehavior::Static;
  }

  bool empty() const noexcept {
    return nodes_.empty();
  }
  std::size_t size() const noexcept {
    return nodes_.size();
  }

private:
  std::vector<NodePtr> nodes_;
};

} // namespace simaai::neat
