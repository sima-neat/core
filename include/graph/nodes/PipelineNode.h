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

class PipelineNode final : public simaai::neat::graph::Node {
public:
  explicit PipelineNode(const simaai::neat::NodeGroup& group, std::string label = {})
      : group_(group), label_(std::move(label)) {
    init_();
  }

  explicit PipelineNode(simaai::neat::NodeGroup&& group, std::string label = {})
      : group_(std::move(group)), label_(std::move(label)) {
    init_();
  }

  explicit PipelineNode(std::shared_ptr<simaai::neat::Node> node, std::string label = {})
      : group_(std::vector<std::shared_ptr<simaai::neat::Node>>{std::move(node)}),
        label_(std::move(label)) {
    init_();
  }

  Backend backend() const override {
    return Backend::Pipeline;
  }

  std::string kind() const override {
    return kind_;
  }

  std::string user_label() const override {
    if (!label_.empty())
      return label_;
    if (group_.size() == 1 && group_.nodes().front()) {
      return group_.nodes().front()->user_label();
    }
    return "";
  }

  std::vector<PortDesc> input_ports() const override {
    if (!requires_input_)
      return {};
    return {PortDesc{.name = "in", .spec = OutputSpec{}}};
  }

  std::vector<PortDesc> output_ports() const override {
    return {PortDesc{.name = "out", .spec = OutputSpec{}}};
  }

  const simaai::neat::NodeGroup& group() const {
    return group_;
  }

  bool is_source_like() const {
    return is_source_like_;
  }
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
