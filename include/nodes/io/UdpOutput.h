/**
 * @file
 * @ingroup nodes_io
 * @brief UDP sink node wrapper.
 */
#pragma once

#include "builder/Node.h"

#include <memory>
#include <string>
#include <vector>

namespace simaai::neat {

struct UdpOutputOptions {
  std::string host = "127.0.0.1";
  int port = 5000;
  bool sync = false;
  bool async = false;
};

class UdpOutput final : public Node {
public:
  explicit UdpOutput(UdpOutputOptions opt) : opt_(std::move(opt)) {}

  std::string kind() const override {
    return "UdpOutput";
  }
  NodeCapsBehavior caps_behavior() const override {
    return NodeCapsBehavior::Dynamic;
  }
  std::string backend_fragment(int node_index) const override;
  std::vector<std::string> element_names(int node_index) const override;

private:
  UdpOutputOptions opt_;
};

} // namespace simaai::neat

namespace simaai::neat::nodes {
std::shared_ptr<simaai::neat::Node> UdpOutput(UdpOutputOptions opt = {});
} // namespace simaai::neat::nodes
