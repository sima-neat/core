/**
 * @file
 * @ingroup builder
 * @brief Fluent builder for assembling linear NodeGroups.
 */
// include/builder/Builder.h
#pragma once

#include <cstddef>
#include <initializer_list>
#include <memory>
#include <utility>
#include <vector>

#include "builder/Node.h"
#include "builder/NodeGroup.h"

namespace simaai::neat {

/**
 * @brief Fluent builder for assembling a linear NodeGroup (the common case).
 *
 * Notes:
 * - This module is STL-only (no GStreamer/OpenCV includes).
 * - Node indices are implied by order-of-insertion; Session uses that
 *   ordering for deterministic element naming (n<idx>_...).
 * - For non-linear graphs, use builder/Graph directly (kept separate on purpose).
 */
class Builder final {
public:
  using NodePtr = std::shared_ptr<Node>;

  Builder() = default;

  /// Construct from an existing group (copies node pointers).
  explicit Builder(const NodeGroup& g) : nodes_(g.nodes()) {}

  /// Construct from an existing vector of nodes (moves the vector).
  explicit Builder(std::vector<NodePtr> nodes) : nodes_(std::move(nodes)) {}

  // -------- Core ops --------

  /// Append a node to the end of the chain.
  Builder& add(NodePtr node) {
    nodes_.push_back(std::move(node));
    return *this;
  }

  /// Fluent alias for add().
  Builder& then(NodePtr node) {
    return add(std::move(node));
  }

  /// Append all nodes from a group (preserves order).
  Builder& add(const NodeGroup& group) {
    const auto& gnodes = group.nodes();
    nodes_.insert(nodes_.end(), gnodes.begin(), gnodes.end());
    return *this;
  }

  /// Append nodes from an initializer_list.
  Builder& add(std::initializer_list<NodePtr> nodes) {
    nodes_.insert(nodes_.end(), nodes.begin(), nodes.end());
    return *this;
  }

  /// Append nodes from an iterator range.
  template <class It> Builder& add(It begin, It end) {
    nodes_.insert(nodes_.end(), begin, end);
    return *this;
  }

  /// Reserve capacity for N nodes.
  Builder& reserve(std::size_t n) {
    nodes_.reserve(n);
    return *this;
  }

  /// Clear the builder.
  void clear() noexcept {
    nodes_.clear();
  }

  // -------- Introspection --------

  std::size_t size() const noexcept {
    return nodes_.size();
  }
  bool empty() const noexcept {
    return nodes_.empty();
  }

  const std::vector<NodePtr>& nodes() const noexcept {
    return nodes_;
  }

  // -------- Materialization --------

  /// Return a copy of the assembled NodeGroup.
  NodeGroup build() const {
    return NodeGroup(nodes_);
  }

  /// Move out the nodes and reset the builder.
  std::vector<NodePtr> release_nodes() {
    std::vector<NodePtr> out;
    out.swap(nodes_);
    return out;
  }

  /// Move out as NodeGroup and reset the builder.
  NodeGroup release() {
    return NodeGroup(release_nodes());
  }

private:
  std::vector<NodePtr> nodes_;
};

} // namespace simaai::neat
