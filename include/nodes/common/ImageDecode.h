/**
 * @file
 * @ingroup nodes_common
 * @brief Image decode helper node.
 */
#pragma once

#include "builder/Node.h"

#include <memory>
#include <vector>

namespace simaai::neat {

class ImageDecode final : public Node {
public:
  std::string kind() const override {
    return "ImageDecode";
  }
  NodeCapsBehavior caps_behavior() const override {
    return NodeCapsBehavior::Dynamic;
  }

  std::string backend_fragment(int node_index) const override;
  std::vector<std::string> element_names(int node_index) const override;
};

} // namespace simaai::neat

namespace simaai::neat::nodes {
std::shared_ptr<simaai::neat::Node> ImageDecode();
} // namespace simaai::neat::nodes
