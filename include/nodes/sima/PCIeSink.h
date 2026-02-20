/**
 * @file
 * @ingroup nodes_sima
 * @brief PCIe sink node for sending data to host via PCIe.
 */
#pragma once

#include "builder/Node.h"

#include <cstdint>
#include <memory>
#include <string>
#include <vector>

namespace simaai::neat {

struct PCIeSinkOptions {
  // Core properties (simaaipciesink)
  std::string config_file;
  std::string data_buf_name = "overlay";
  int data_buffer_size = 4194304;
  int num_buffers = 5;
  int queue = 0;

  // Multi-buffer mode
  std::string param_buf_name = "camera_params";
  int param_buffer_size = 48;
  bool use_multi_buffers = false;

  // Timing controls
  bool sync = true;
  bool async_state = true;
  int64_t max_lateness_ns = -1;
  uint64_t processing_deadline_ns = 20000000;

  // Optional diagnostics
  bool transmit_kpi = false;
  bool qos = false;
};

class PCIeSink final : public Node {
public:
  explicit PCIeSink(PCIeSinkOptions opt = {});

  std::string kind() const override {
    return "PCIeSink";
  }
  std::string user_label() const override {
    return "pciesink";
  }
  NodeCapsBehavior caps_behavior() const override {
    return NodeCapsBehavior::Dynamic;
  }

  std::string backend_fragment(int node_index) const override;
  std::vector<std::string> element_names(int node_index) const override;

  const PCIeSinkOptions& options() const {
    return opt_;
  }

private:
  PCIeSinkOptions opt_;
};

} // namespace simaai::neat

namespace simaai::neat::nodes {
std::shared_ptr<simaai::neat::Node> PCIeSink(PCIeSinkOptions opt = {});
} // namespace simaai::neat::nodes
