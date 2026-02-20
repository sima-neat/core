/**
 * @file
 * @ingroup nodes_common
 * @brief File source node.
 */
#pragma once

#include "builder/Node.h"

#include <memory>
#include <string>
#include <vector>

namespace simaai::neat {

class FileInput final : public Node {
public:
  explicit FileInput(std::string path);

  std::string kind() const override {
    return "FileInput";
  }
  std::string user_label() const override {
    return path_;
  }
  InputRole input_role() const override {
    return InputRole::Source;
  }
  NodeCapsBehavior caps_behavior() const override {
    return NodeCapsBehavior::Dynamic;
  }

  std::string backend_fragment(int node_index) const override;
  std::vector<std::string> element_names(int node_index) const override;

  const std::string& path() const {
    return path_;
  }

private:
  std::string path_;
};

} // namespace simaai::neat

namespace simaai::neat::nodes {
std::shared_ptr<simaai::neat::Node> FileInput(std::string path);
} // namespace simaai::neat::nodes
