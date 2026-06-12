/**
 * @file
 * @ingroup builder
 * @brief Fluent builder for assembling linear node lists.
 *
 * `Builder` is the lightweight, STL-only API used by app code to chain Nodes into a
 * linear pipeline before handing them off to a Graph. It implies Node indices from
 * insertion order, which the Graph then uses to mint deterministic element names.
 * For non-linear (multi-input/multi-output) topologies, use public `Graph` with named
 * `Input`/`Output` endpoints and `connect()`.
 *
 * @see Graph
 */
// include/builder/Builder.h
#pragma once

#include <cstddef>
#include <initializer_list>
#include <memory>
#include <utility>
#include <vector>

#include "builder/Node.h"

namespace simaai::neat {

/**
 * @brief Fluent builder for assembling a linear node list (the common case).
 *
 * Use `add()` / `then()` to append Nodes in order, then call `build()` (copy) or
 * `release_nodes()` (move) to materialize a node list. The Builder itself stores nothing
 * but `shared_ptr<Node>` and is safe to discard once the node list is produced.
 *
 * Notes:
 * - This module is STL-only (no GStreamer/OpenCV includes).
 * - Node indices are implied by order-of-insertion; Graph uses that
 *   ordering for deterministic element naming (n<idx>_...).
 * - For non-linear topologies, use public `Graph` fragments and `connect()`.
 *
 * @ingroup builder
 * @see Graph
 */
class Builder final {
public:
  using NodePtr = std::shared_ptr<Node>;

  /// @brief Construct an empty builder.
  Builder() = default;

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

  /// @brief Return a copy of the assembled node list. Builder remains usable.
  std::vector<NodePtr> build() const {
    return nodes_;
  }

  /// @brief Move out the nodes and reset the builder.
  std::vector<NodePtr> release_nodes() {
    std::vector<NodePtr> out;
    out.swap(nodes_);
    return out;
  }

private:
  std::vector<NodePtr> nodes_;
};

} // namespace simaai::neat
