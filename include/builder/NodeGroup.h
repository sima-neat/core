/**
 * @file
 * @ingroup builder
 * @brief `NodeGroup` — bundle of Nodes that travels as a unit between builder and runtime.
 *
 * A `NodeGroup` is the shared container that lets a tuned chain of Nodes (e.g., the four
 * Nodes that implement a model's preprocess) be composed, validated, and reused as one
 * thing. NodeGroups are the framework's mechanism for **composability** — packaging
 * common pipeline shapes (preprocess, postprocess, RTSP I/O) without inventing a new
 * Node subtype. They are also the unit that `Model` returns when it expands its plan
 * into Nodes.
 *
 * @see Node for the elements a NodeGroup holds
 * @see "NodeGroups" (§0.10 / §0.11 of the design deep dive)
 */
// include/builder/NodeGroup.h
#pragma once

#include <memory>
#include <utility>
#include <vector>

#include "builder/Node.h"

namespace simaai::neat {

/**
 * @brief Linear list of Nodes shared by builder and runtime.
 *
 * NodeGroups are how the framework packages tuned, reusable Node sequences (preprocess,
 * detection postprocess, RTSP source, etc.) into something that can be added to a Session
 * with a single `add()` call. Builder/Graph are intentionally STL-only — NodeGroup is the
 * one shared type they pass between layers.
 *
 * @code
 * sima::Session sess;
 * sess.add(sima::nodes::groups::FileMp4H264In("input.mp4"));   // a NodeGroup
 * sess.add(model.session());                                    // model expands to a NodeGroup
 * sess.add(sima::nodes::groups::Mp4FileOut("output.mp4"));     // a NodeGroup
 * sess.run();
 * @endcode
 *
 * @ingroup builder
 */
class NodeGroup final {
public:
  /// Convenience alias for the shared-pointer type held by the group.
  using NodePtr = std::shared_ptr<Node>;

  /// Construct an empty group.
  NodeGroup() = default;

  /// Construct from a moved-in vector of node pointers (preferred — avoids a copy).
  explicit NodeGroup(std::vector<NodePtr>&& nodes) : nodes_(std::move(nodes)) {}

  /// Construct from a const-ref vector of node pointers (copies the vector).
  explicit NodeGroup(const std::vector<NodePtr>& nodes) : nodes_(nodes) {}

  /// Const view of the underlying node list.
  const std::vector<NodePtr>& nodes() const noexcept {
    return nodes_;
  }

  /// Mutable view of the underlying node list (for builders that need to splice/modify).
  std::vector<NodePtr>& nodes_mut() noexcept {
    return nodes_;
  }

  /**
   * @brief Aggregate caps behavior of the contained Nodes.
   *
   * Returns `Static` if any node in the group is static (the conservative answer for
   * caps-negotiation purposes); otherwise `Dynamic`.
   */
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

  /// Convenience: true iff `caps_behavior() == Static`.
  bool is_static_group() const {
    return caps_behavior() == NodeCapsBehavior::Static;
  }

  /// True iff the group holds no nodes.
  bool empty() const noexcept {
    return nodes_.empty();
  }
  /// Number of nodes in the group.
  std::size_t size() const noexcept {
    return nodes_.size();
  }

private:
  std::vector<NodePtr> nodes_;
};

} // namespace simaai::neat
