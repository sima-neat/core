/**
 * @file
 * @ingroup nodes_io
 * @brief RTSP input node wrapper.
 */
#pragma once
#include "builder/Node.h"
#include <memory>
#include <string>
#include <vector>

namespace simaai::neat {

class RTSPInput final : public Node {
public:
  RTSPInput(std::string url, int latency_ms = 200, bool tcp = true, bool drop_on_latency = false,
            std::string buffer_mode = "");

  std::string kind() const override {
    return "RTSPInput";
  }
  std::string user_label() const override {
    return url_;
  }
  InputRole input_role() const override {
    return InputRole::Source;
  }
  NodeCapsBehavior caps_behavior() const override {
    return NodeCapsBehavior::Dynamic;
  }

  std::string backend_fragment(int node_index) const override;
  std::vector<std::string> element_names(int node_index) const override;

  const std::string& url() const {
    return url_;
  }
  int latency_ms() const {
    return latency_ms_;
  }
  bool tcp() const {
    return tcp_;
  }
  bool drop_on_latency() const {
    return drop_on_latency_;
  }
  const std::string& buffer_mode() const {
    return buffer_mode_;
  }

private:
  std::string url_;
  int latency_ms_ = 200;
  bool tcp_ = true;
  bool drop_on_latency_ = false;
  std::string buffer_mode_;
};

} // namespace simaai::neat

namespace simaai::neat::nodes {
std::shared_ptr<simaai::neat::Node> RTSPInput(std::string url, int latency_ms = 200,
                                              bool tcp = true, bool drop_on_latency = false,
                                              std::string buffer_mode = "");
} // namespace simaai::neat::nodes
