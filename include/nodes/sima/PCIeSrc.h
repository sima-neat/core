/**
 * @file
 * @ingroup nodes_sima
 * @brief `PCIeSrc` Node — receives samples from a PCIe-connected host (Modalix as PCIe target).
 *
 * Wraps the legacy `simaaipciesrc` GStreamer element which uses standard GStreamer
 * buffer allocation (no SiMa DMA allocator), keeping memory usage low and compatible
 * with all board configurations. Use as the source in pipelines where the host pushes
 * frames into the Modalix board over PCIe.
 */
#pragma once

#include "builder/Node.h"
#include "builder/OutputSpec.h"

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
  /// Size in bytes of each incoming frame buffer.
  /// Default 4 MB matches the simaaipciesrc default.
  int buffer_size = 4194304;

  /// Optional caps enforcement — when format, width, and height are all
  /// set, a capsfilter is appended to lock the negotiated format.
  std::string format;     ///< Pixel format (e.g. `"NV12"`); empty = no caps lock.
  int width = -1;         ///< Frame width in pixels; `-1` = no caps lock.
  int height = -1;        ///< Frame height in pixels; `-1` = no caps lock.
  int fps_n = 0;          ///< Framerate numerator (caps lock); `0` = unset.
  int fps_d = 1;          ///< Framerate denominator (caps lock).
};

/**
 * @brief Source-role Node that receives samples from a PCIe-connected host.
 *
 * @ingroup nodes_sima
 */
class PCIeSrc final : public Node, public OutputSpecProvider {
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
  /// Negotiated downstream caps produced by this Node.
  OutputSpec output_spec(const OutputSpec& input) const override;

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
