/**
 * @file
 * @ingroup nodes_sima
 * @brief `PCIeSink` Node — sends samples to a PCIe-connected host (Modalix as PCIe target).
 *
 * Wraps the `neatpciesink` GStreamer element, which delivers buffers across the PCIe
 * link to the host driver. Use as a terminal sink in pipelines where the Modalix board
 * is acting as a PCIe target and the host is the actual consumer of the output.
 */
#pragma once

#include "builder/Node.h"

#include <memory>
#include <string>
#include <vector>

namespace simaai::neat {

/**
 * @brief Construction options for a `PCIeSink` Node.
 *
 * @ingroup nodes_sima
 */
struct PCIeSinkOptions {
  std::string config_file;   ///< Optional neatpciesink config file path.
  int queue = 0;             ///< PCIe hardware queue number.
  bool transmit_kpi = false; ///< If true, transmit KPI/diagnostic packets alongside data.
};

/**
 * @brief Terminal sink Node that streams samples to a PCIe-connected host.
 *
 * @ingroup nodes_sima
 */
class PCIeSink final : public Node {
public:
  /// Construct with optional `PCIeSinkOptions`.
  explicit PCIeSink(PCIeSinkOptions opt = {});

  /// Type label for this Node kind.
  std::string kind() const override {
    return "PCIeSink";
  }
  /// User-facing label for this Node.
  std::string user_label() const override {
    return "pciesink";
  }
  /// Whether the Node negotiates static or dynamic caps.
  NodeCapsBehavior caps_behavior() const override {
    return NodeCapsBehavior::Dynamic;
  }

  /// GStreamer fragment this Node emits.
  std::string backend_fragment(int node_index) const override;
  /// Deterministic element names this Node will create.
  std::vector<std::string> element_names(int node_index) const override;

  /// Inspect the Node's options.
  const PCIeSinkOptions& options() const {
    return opt_;
  }

private:
  PCIeSinkOptions opt_;
};

} // namespace simaai::neat

namespace simaai::neat::nodes {
/// Convenience factory for a `PCIeSink` Node with optional `PCIeSinkOptions`.
std::shared_ptr<simaai::neat::Node> PCIeSink(PCIeSinkOptions opt = {});
} // namespace simaai::neat::nodes
