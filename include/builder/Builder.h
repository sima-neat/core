/**
 * @file
 * @ingroup builder
 * @brief Fluent builder for assembling linear NodeGroups.
 *
 * `Builder` is the lightweight, STL-only API used by app code to chain Nodes into a
 * linear pipeline before handing them off to a Session. It implies Node indices from
 * insertion order, which the Session then uses to mint deterministic element names.
 * For non-linear (multi-input/multi-output) topologies, use `Graph` directly instead.
 *
 * @see NodeGroup
 * @see Graph
 * @see Session
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
 * Use `add()` / `then()` to append Nodes in order, then call `build()` (copy) or
 * `release()` (move) to materialize a `NodeGroup`. The Builder itself stores nothing
 * but `shared_ptr<Node>` and is safe to discard once the NodeGroup is produced.
 *
 * Notes:
 * - This module is STL-only (no GStreamer/OpenCV includes).
 * - Node indices are implied by order-of-insertion; Session uses that
 *   ordering for deterministic element naming (n<idx>_...).
 * - For non-linear graphs, use builder/Graph directly (kept separate on purpose).
 *
 * @ingroup builder
 * @see NodeGroup
 * @see Graph
 */
class Builder final {
public:
  using NodePtr = std::shared_ptr<Node>;

  /// @brief Construct an empty builder.
  Builder() = default;

  /// @brief Construct from an existing group (copies node pointers).
  explicit Builder(const NodeGroup& g) : nodes_(g.nodes()) {}

  /// @brief Construct from an existing vector of nodes (moves the vector).
  explicit Builder(std::vector<NodePtr> nodes) : nodes_(std::move(nodes)) {}

  // -------- Core ops --------

  /// @brief Append a node to the end of the chain.
  Builder& add(NodePtr node) {
    nodes_.push_back(std::move(node));
    return *this;
  }

  /// @brief Fluent alias for `add()` — reads naturally as `b.add(a).then(b).then(c)`.
  Builder& then(NodePtr node) {
    return add(std::move(node));
  }

  /// @brief Append all nodes from a group (preserves order).
  Builder& add(const NodeGroup& group) {
    const auto& gnodes = group.nodes();
    nodes_.insert(nodes_.end(), gnodes.begin(), gnodes.end());
    return *this;
  }

  /// @brief Append nodes from an initializer_list.
  Builder& add(std::initializer_list<NodePtr> nodes) {
    nodes_.insert(nodes_.end(), nodes.begin(), nodes.end());
    return *this;
  }

  /// @brief Append nodes from an iterator range.
  template <class It> Builder& add(It begin, It end) {
    nodes_.insert(nodes_.end(), begin, end);
    return *this;
  }

  /// @brief Reserve capacity for N nodes (avoids reallocation when chain length is known).
  Builder& reserve(std::size_t n) {
    nodes_.reserve(n);
    return *this;
  }

  /// @brief Drop all accumulated nodes; the builder becomes empty.
  void clear() noexcept {
    nodes_.clear();
  }

  // -------- Introspection --------

  /// @brief Number of nodes accumulated so far.
  std::size_t size() const noexcept {
    return nodes_.size();
  }
  /// @brief True if no nodes have been added yet.
  bool empty() const noexcept {
    return nodes_.empty();
  }

  /// @brief Read-only access to the underlying node list.
  const std::vector<NodePtr>& nodes() const noexcept {
    return nodes_;
  }

  // -------- Materialization --------

  /// @brief Return a copy of the assembled NodeGroup. Builder remains usable.
  NodeGroup build() const {
    return NodeGroup(nodes_);
  }

  /// @brief Move out the nodes and reset the builder.
  std::vector<NodePtr> release_nodes() {
    std::vector<NodePtr> out;
    out.swap(nodes_);
    return out;
  }

  /// @brief Move out as NodeGroup and reset the builder.
  NodeGroup release() {
    return NodeGroup(release_nodes());
  }

private:
  std::vector<NodePtr> nodes_;
};

} // namespace simaai::neat
