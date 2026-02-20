/**
 * @file
 * @ingroup nodes_common
 * @brief Videoconvert node wrapper.
 */
#pragma once

#include "builder/Node.h"

#include <memory>
#include <vector>

namespace simaai::neat {

class VideoConvert final : public Node {
public:
  std::string kind() const override {
    return "VideoConvert";
  }
  NodeCapsBehavior caps_behavior() const override {
    return NodeCapsBehavior::Dynamic;
  }
  std::string backend_fragment(int node_index) const override;
  std::vector<std::string> element_names(int node_index) const override;
};

} // namespace simaai::neat

namespace simaai::neat::nodes {
std::shared_ptr<simaai::neat::Node> VideoConvert();
} // namespace simaai::neat::nodes
