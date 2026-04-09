/**
 * @file
 * @ingroup nodes_io
 * @brief V4L2 camera source node wrapper.
 */
#pragma once

#include "builder/Node.h"
#include "builder/OutputSpec.h"

#include <memory>
#include <string>
#include <vector>

namespace simaai::neat {

struct V4L2InputOptions {
  /// V4L2 device path (must not be empty).
  std::string device = "/dev/video0";

  /// Optional caps enforcement. A capsfilter is appended only when media_type
  /// is non-empty and both width and height are positive (> 0). Setting only
  /// some of these fields triggers a diagnostic warning; the capsfilter will
  /// not be emitted unless all three are specified.
  std::string media_type; ///< e.g. "video/x-raw", "image/jpeg", "video/x-bayer"
  std::string format;     ///< e.g. "RGB", "NV12", "rggb12le" (omit for MJPEG)
  int width = -1;         ///< Capture width in pixels (-1 = unconstrained).
  int height = -1;        ///< Capture height in pixels (-1 = unconstrained).
  /// Framerate numerator. 0 or negative omits the framerate field from caps.
  /// Use 0 when the ISP advertises a rate different from what you expect.
  int fps_n = 0;
  /// Framerate denominator (default: 1). Clamped to 1 if fps_n > 0 and fps_d <= 0.
  int fps_d = 1;

  /// V4L2 I/O method. Valid values: "mmap", "userptr", "dmabuf",
  /// "dmabuf-import", "read". Empty string uses the GStreamer default.
  std::string io_mode;
  /// Number of buffers for v4l2src. -1 (default) omits the property, letting
  /// GStreamer use its built-in default (continuous capture).
  int num_buffers = -1;
};

class V4L2Input final : public Node, public OutputSpecProvider {
public:
  explicit V4L2Input(V4L2InputOptions opt = {});

  std::string kind() const override {
    return "V4L2Input";
  }
  std::string user_label() const override {
    return opt_.device;
  }
  InputRole input_role() const override {
    return InputRole::Source;
  }
  NodeCapsBehavior caps_behavior() const override {
    return NodeCapsBehavior::Dynamic;
  }

  std::string backend_fragment(int node_index) const override;
  std::vector<std::string> element_names(int node_index) const override;
  OutputSpec output_spec(const OutputSpec& input) const override;

  const V4L2InputOptions& options() const {
    return opt_;
  }

private:
  V4L2InputOptions opt_;
};

} // namespace simaai::neat

namespace simaai::neat::nodes {
std::shared_ptr<simaai::neat::Node> V4L2Input(V4L2InputOptions opt = {});
} // namespace simaai::neat::nodes
