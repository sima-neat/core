/**
 * @file
 * @ingroup nodes_common
 * @brief `Queue` Node — wraps GStreamer `queue` for buffer decoupling between stages.
 *
 * Inserts a `queue` element that decouples upstream and downstream threads. Useful
 * anywhere the natural producer rate doesn't match the consumer rate, or when crossing
 * a thread boundary that GStreamer's caps negotiation needs explicit help with.
 */
#pragma once

#include "builder/Node.h"

#include <memory>
#include <vector>

namespace simaai::neat {

/**
 * @brief Wraps GStreamer's `queue` element — inserts a thread/buffer boundary.
 *
 * Usage:
 * @code
 * sess.add(sima::nodes::Queue());
 * @endcode
 *
 * @ingroup nodes_common
 */
class Queue final : public Node {
public:
  /// Type label for this Node kind.
  std::string kind() const override {
    return "Queue";
  }
  /// Whether the Node negotiates static or dynamic caps.
  NodeCapsBehavior caps_behavior() const override {
    return NodeCapsBehavior::Dynamic;
  }
  /// GStreamer fragment this Node emits.
  std::string backend_fragment(int node_index) const override;
  /// Deterministic element names this Node will create.
  std::vector<std::string> element_names(int node_index) const override;
};

} // namespace simaai::neat

namespace simaai::neat::nodes {
/// Convenience factory for a `Queue` Node.
std::shared_ptr<simaai::neat::Node> Queue();
} // namespace simaai::neat::nodes
