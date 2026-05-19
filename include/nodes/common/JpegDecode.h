/**
 * @file
 * @ingroup nodes_common
 * @brief `JpegDecode` Node — JPEG-specific decode stage.
 */
#pragma once

#include "builder/Node.h"

#include <memory>
#include <vector>

namespace simaai::neat {

/**
 * @brief Decodes JPEG-encoded buffers to raw frames. Use when input is known JPEG.
 *
 * For format-agnostic decoding, prefer `ImageDecode`.
 *
 * @ingroup nodes_common
 */
class JpegDecode final : public Node {
public:
  /// Type label for this Node kind.
  std::string kind() const override {
    return "JpegDecode";
  }
  /// Whether the Node negotiates static or dynamic caps.
  NodeCapsBehavior caps_behavior() const override {
    return NodeCapsBehavior::Dynamic;
  }

  /// GStreamer fragment this Node emits.
  std::string backend_fragment(int node_index) const override;
  /// Deterministic element names this Node will create.
  std::vector<std::string> element_names(int node_index) const override;
};

} // namespace simaai::neat

namespace simaai::neat::nodes {
/// Convenience factory for a `JpegDecode` Node.
std::shared_ptr<simaai::neat::Node> JpegDecode();
} // namespace simaai::neat::nodes
