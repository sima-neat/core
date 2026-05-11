/**
 * @file
 * @ingroup graph_nodes
 * @brief Pipeline-backed graph node wrapper.
 */
#pragma once

#include "builder/Node.h"
#include "builder/NodeGroup.h"
#include "graph/GraphTypes.h"
#include "graph/Node.h"

#include <memory>
#include <string>
#include <utility>
#include <vector>

namespace simaai::neat::graph::nodes {

/**
 * @brief Runtime-graph node that wraps a builder-side `NodeGroup` (or single `Node`) as a pipeline-backend node.
 *
 * Lets a linear, GStreamer-pipeline-shaped fragment participate in the runtime actor graph.
 * The wrapped group's first node's `InputRole` is inspected to decide whether this node is
 * source-like (no input port) or push-style (single `"in"` port). Always exposes a single
 * `"out"` port.
 *
 * @see simaai::neat::NodeGroup
 * @see simaai::neat::graph::Node
 * @ingroup graph
 */
class PipelineNode final : public simaai::neat::graph::Node {
public:
  /// Construct from a `NodeGroup` by copy.
  explicit PipelineNode(const simaai::neat::NodeGroup& group, std::string label = {})
      : group_(group), label_(std::move(label)) {
    init_();
  }

  /// Construct from a `NodeGroup` by move.
  explicit PipelineNode(simaai::neat::NodeGroup&& group, std::string label = {})
      : group_(std::move(group)), label_(std::move(label)) {
    init_();
  }

  /// Construct from a single builder `Node`.
  explicit PipelineNode(std::shared_ptr<simaai::neat::Node> node, std::string label = {})
      : group_(std::vector<std::shared_ptr<simaai::neat::Node>>{std::move(node)}),
        label_(std::move(label)) {
    init_();
  }

  /// Always returns `Backend::Pipeline`.
  Backend backend() const override {
    return Backend::Pipeline;
  }

  /// Returns the wrapped node's kind, or `"PipelineGroup"` when the group has multiple nodes.
  std::string kind() const override {
    return kind_;
  }

  /// Returns the explicit label if set, else the wrapped single node's `user_label()`.
  std::string user_label() const override {
    if (!label_.empty())
      return label_;
    if (group_.size() == 1 && group_.nodes().front()) {
      return group_.nodes().front()->user_label();
    }
    return "";
  }

  /// Returns the input port (`"in"`) unless the group is source-like, in which case empty.
  std::vector<PortDesc> input_ports() const override {
    if (!requires_input_)
      return {};
    return {PortDesc{.name = "in", .spec = OutputSpec{}}};
  }

  /// Always exposes a single `"out"` port.
  std::vector<PortDesc> output_ports() const override {
    return {PortDesc{.name = "out", .spec = OutputSpec{}}};
  }

  /// Access the wrapped builder `NodeGroup`.
  const simaai::neat::NodeGroup& group() const {
    return group_;
  }

  /// True iff the wrapped group is source-like (has a Source role and no Push role).
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
    const auto& nodes = group_.nodes();
    if (nodes.size() == 1 && nodes.front()) {
      kind_ = nodes.front()->kind();
    } else {
      kind_ = "PipelineGroup";
    }

    for (const auto& n : nodes) {
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

  simaai::neat::NodeGroup group_;
  std::string label_;
  std::string kind_;
  bool is_source_like_ = false;
  bool requires_input_ = true;
};

} // namespace simaai::neat::graph::nodes
