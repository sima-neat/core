/**
 * @file
 * @ingroup nodes_sima
 * @brief `PCIeSink` Node — sends samples to a PCIe-connected host (Modalix as PCIe target).
 *
 * Wraps the `simaaipciesink` GStreamer element, which delivers buffers across the PCIe
 * link to the host driver. Use as a terminal sink in pipelines where the Modalix board
 * is acting as a PCIe target and the host is the actual consumer of the output.
 */
#pragma once

#include "builder/Node.h"

#include <cstdint>
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
  // Core properties (simaaipciesink)
  std::string config_file;                  ///< Optional simaaipciesink config file path.
  std::string data_buf_name = "overlay";    ///< Name of the host-side buffer that receives data.
  int data_buffer_size = 4194304;           ///< Data buffer size in bytes (default 4 MiB).
  int num_buffers = 5;                      ///< Number of buffers in the pool.
  int queue = 0;                            ///< Queue depth for outgoing samples.

  // Multi-buffer mode
  std::string param_buf_name = "camera_params"; ///< Auxiliary parameter buffer name (multi-buffer mode).
  int param_buffer_size = 48;                   ///< Parameter buffer size in bytes (multi-buffer mode).
  bool use_multi_buffers = false;               ///< Enable the multi-buffer (data + params) protocol.

  // Timing controls
  bool sync = true;                  ///< Sync to pipeline clock when delivering buffers.
  bool async_state = true;           ///< Allow async state changes on this sink.
  int64_t max_lateness_ns = -1;      ///< Max lateness in ns; `-1` = unlimited (no late-frame drop).
  uint64_t processing_deadline_ns = 20000000; ///< Processing deadline in ns (default 20 ms).

  // Optional diagnostics
  bool transmit_kpi = false;         ///< If true, transmit KPI/diagnostic packets alongside data.
  bool qos = false;                  ///< Enable QoS reporting on this sink.
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
