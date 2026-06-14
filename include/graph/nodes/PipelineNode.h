/**
 * @file
 * @ingroup graph_nodes
 * @brief Pipeline-backed graph node wrapper.
 */
#pragma once

#include "builder/Node.h"
#include "graph/GraphTypes.h"
#include "graph/Node.h"

#include <memory>
#include <string>
#include <utility>
#include <vector>

namespace simaai::neat::graph::nodes {

/**
 * @brief Runtime-graph node that wraps a linear node list (or single `Node`) as a
 * pipeline-backend node.
 *
 * Lets a linear, GStreamer-pipeline-shaped fragment participate in the runtime actor graph.
 * The wrapped fragment's first node's `InputRole` is inspected to decide whether this node is
 * source-like (no input port) or push-style (single `"in"` port). Always exposes a single
 * `"out"` port.
 *
 * @see simaai::neat::graph::Node
 * @ingroup graph
 */
class PipelineNode final : public simaai::neat::graph::Node {
public:
  using NodePtr = std::shared_ptr<simaai::neat::Node>;

  /// Construct from a node vector by move.
  explicit PipelineNode(std::vector<NodePtr> nodes, std::string label = {},
                        int input_max_in_edges = 1)
      : nodes_(std::move(nodes)), label_(std::move(label)),
        input_max_in_edges_(input_max_in_edges) {
    init_();
  }

  /// Construct from a single builder `Node`.
  explicit PipelineNode(std::shared_ptr<simaai::neat::Node> node, std::string label = {},
                        int input_max_in_edges = 1)
      : nodes_(std::vector<NodePtr>{std::move(node)}), label_(std::move(label)),
        input_max_in_edges_(input_max_in_edges) {
    init_();
  }

  /// Always returns `Backend::Pipeline`.
  Backend backend() const override {
    return Backend::Pipeline;
  }

  /// Returns the wrapped node's kind, or `"PipelineFragment"` when the fragment has multiple nodes.
  std::string kind() const override {
    return kind_;
  }

  /// Returns the explicit label if set, else the wrapped single node's `user_label()`.
  std::string user_label() const override {
    if (!label_.empty())
      return label_;
    if (nodes_.size() == 1 && nodes_.front()) {
      return nodes_.front()->user_label();
    }
    return "";
  }

  /// Returns the input port (`"in"`) unless the group is source-like, in which case empty.
  std::vector<PortDesc> input_ports() const override {
    if (!requires_input_)
      return {};
    return {PortDesc{.name = "in", .spec = OutputSpec{}, .max_in_edges = input_max_in_edges_}};
  }

  /// Always exposes a single `"out"` port.
  std::vector<PortDesc> output_ports() const override {
    return {PortDesc{.name = "out", .spec = OutputSpec{}}};
  }

  /// Access the wrapped node list.
  const std::vector<NodePtr>& nodes() const {
    return nodes_;
  }

  /// True iff the wrapped fragment is source-like (has a Source role and no Push role).
  bool is_source_like() const {
    return is_source_like_;
  }
  /// True iff this node requires an upstream input port.
  bool requires_input() const {
    return requires_input_;
  }

private:
  void init_() {
    bool has_source = false;
    bool has_push = false;
    if (nodes_.size() == 1 && nodes_.front()) {
      kind_ = nodes_.front()->kind();
    } else {
      kind_ = "PipelineFragment";
    }

    for (const auto& n : nodes_) {
      if (!n)
        continue;
      const InputRole role = n->input_role();
      if (role == InputRole::Source)
        has_source = true;
      if (role == InputRole::Push)
        has_push = true;
    }

    is_source_like_ = has_source && !has_push;
    requires_input_ = !is_source_like_;
  }

  std::vector<NodePtr> nodes_;
  std::string label_;
  std::string kind_;
  int input_max_in_edges_ = 1;
  bool is_source_like_ = false;
  bool requires_input_ = true;
};

} // namespace simaai::neat::graph::nodes
