/**
 * @file
 * @ingroup nodes_common
 * @brief Output node for terminal pipeline output.
 */
#pragma once

#include "builder/Node.h"

#include <memory>
#include <utility>
#include <vector>

namespace simaai::neat {

struct OutputOptions {
  int max_buffers = 1;
  bool drop = false;
  bool sync = false;

  static OutputOptions Latest();
  static OutputOptions EveryFrame(int max_buffers = 30);
  static OutputOptions Clocked(int max_buffers = 1);
};

class Output final : public Node {
public:
  Output() = default;
  explicit Output(OutputOptions opt) : opt_(std::move(opt)) {}

  const OutputOptions& options() const {
    return opt_;
  }

  std::string kind() const override {
    return "Output";
  }
  NodeCapsBehavior caps_behavior() const override {
    return NodeCapsBehavior::Dynamic;
  }
  std::string backend_fragment(int node_index) const override;
  std::vector<std::string> element_names(int node_index) const override;

private:
  OutputOptions opt_;
};

} // namespace simaai::neat

namespace simaai::neat::nodes {
std::shared_ptr<simaai::neat::Node> Output(OutputOptions opt = {});
} // namespace simaai::neat::nodes
