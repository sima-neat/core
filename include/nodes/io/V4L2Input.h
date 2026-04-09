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
  /// V4L2 device path.
  std::string device = "/dev/video0";

  /// Optional caps enforcement. A capsfilter is appended only when media_type,
  /// width, and height are all set.
  std::string media_type;
  std::string format;
  int width = -1;
  int height = -1;
  int fps_n = 0;
  int fps_d = 1;

  /// Optional v4l2src tuning properties.
  std::string io_mode;
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
