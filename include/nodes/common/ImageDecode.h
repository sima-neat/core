/**
 * @file
 * @ingroup nodes_common
 * @brief `ImageDecode` Node — decodes still images (JPEG/PNG/etc.) to raw frames.
 */
#pragma once

#include "builder/Node.h"

#include <memory>
#include <vector>

namespace simaai::neat {

/**
 * @brief Auto-detects and decodes still-image formats (JPEG, PNG, BMP, …) to raw video.
 *
 * Use as the first decode stage when feeding still images into a pipeline that wants raw
 * pixels (e.g., before resize / preprocess). For known-JPEG inputs prefer `JpegDecode`,
 * which skips the format-detection step.
 *
 * @ingroup nodes_common
 */
class ImageDecode final : public Node {
public:
  /// Type label for this Node kind.
  std::string kind() const override {
    return "ImageDecode";
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
/// Convenience factory for an `ImageDecode` Node.
std::shared_ptr<simaai::neat::Node> ImageDecode();
} // namespace simaai::neat::nodes
