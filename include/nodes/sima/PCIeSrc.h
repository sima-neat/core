/**
 * @file
 * @ingroup nodes_sima
 * @brief `PCIeSrc` Node — receives samples from a PCIe-connected host (Modalix as PCIe target).
 *
 * Wraps the `neatpciesrc` GStreamer element. The element receives per-stream
 * caps from the host and allocates its output from card-side SiMa memory pools.
 * Use as the source in pipelines where the host pushes samples into the Modalix
 * board over PCIe.
 */
#pragma once

#include "builder/Node.h"

#include <memory>
#include <string>
#include <vector>

namespace simaai::neat {

/**
 * @brief Construction options for a `PCIeSrc` Node.
 *
 * @ingroup nodes_sima
 */
struct PCIeSrcOptions {
  /// PCIe data queue to listen on.
  int queue = 0;

  /// Fallback output buffer size in bytes.
  /// Used as the fallback allocation size and encoded-buffer limit.
  /// Default 4 MB matches the `neatpciesrc` default.
  int buffer_size = 4194304;

  /// Per-stream output pool size. Zero keeps the plugin default.
  int pool_size = 0;
};

/**
 * @brief Source-role Node that receives samples from a PCIe-connected host.
 *
 * @ingroup nodes_sima
 */
class PCIeSrc final : public Node {
public:
  /// Construct with optional `PCIeSrcOptions`.
  explicit PCIeSrc(PCIeSrcOptions opt = {});

  /// Type label for this Node kind.
  std::string kind() const override {
    return "PCIeSrc";
  }
  /// User-facing label for this Node.
  std::string user_label() const override {
    return "pciesrc";
  }
  /// Role this Node plays as a stream source.
  InputRole input_role() const override {
    return InputRole::Source;
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
  const PCIeSrcOptions& options() const {
    return opt_;
  }

private:
  PCIeSrcOptions opt_;
};

} // namespace simaai::neat

namespace simaai::neat::nodes {
/// Convenience factory for a `PCIeSrc` Node with optional `PCIeSrcOptions`.
std::shared_ptr<simaai::neat::Node> PCIeSrc(PCIeSrcOptions opt = {});
} // namespace simaai::neat::nodes
