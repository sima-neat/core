/**
 * @file
 * @ingroup nodes_io
 * @brief Input node for push pipelines.
 */
#pragma once

#include "builder/Node.h"
#include "builder/OutputSpec.h"

#include <memory>
#include <cstdint>
#include <string>
#include <vector>

namespace simaai::neat {

struct InputOptions {
  std::string media_type = "video/x-raw";
  std::string format;
  int width = -1;
  int height = -1;
  int depth = -1;
  // Optional dynamic-input limits used for validation/pool sizing.
  // These do not constrain negotiated caps and are only checked at push time.
  int max_width = -1;
  int max_height = -1;
  int max_depth = -1;
  // Optional fixed framerate for caps (0/1 means "unspecified").
  int fps_n = 0;
  int fps_d = 1;
  // Optional full caps string override (used for multi-tensor caps).
  std::string caps_override;

  bool is_live = true;
  bool do_timestamp = true;
  bool block = true;
  int stream_type = 0; // GST_APP_STREAM_TYPE_STREAM
  std::uint64_t max_bytes = 0;

  bool use_simaai_pool = true;
  int pool_min_buffers = 1;
  int pool_max_buffers = 2;

  // Optional GstSimaMeta buffer name override. Leave empty to avoid forcing a legacy default.
  std::string buffer_name;
};

class Input final : public Node, public OutputSpecProvider {
public:
  explicit Input(InputOptions opt);

  std::string kind() const override {
    return "Input";
  }
  std::string user_label() const override {
    return "mysrc";
  }
  InputRole input_role() const override {
    return InputRole::Push;
  }
  NodeCapsBehavior caps_behavior() const override {
    return NodeCapsBehavior::Static;
  }
  std::string buffer_name_hint(int node_index) const override;

  std::string backend_fragment(int node_index) const override;
  std::vector<std::string> element_names(int node_index) const override;
  OutputSpec output_spec(const OutputSpec& input) const override;

  const InputOptions& options() const {
    return opt_;
  }
  std::string caps_string() const;

private:
  InputOptions opt_;
};

} // namespace simaai::neat

namespace simaai::neat::nodes {
std::shared_ptr<simaai::neat::Node> Input(InputOptions opt = {});
} // namespace simaai::neat::nodes
