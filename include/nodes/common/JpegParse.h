/**
 * @file
 * @ingroup nodes_common
 * @brief `JpegParse` Node — parses JPEG frames without decoding them.
 */
#pragma once

#include "builder/Node.h"
#include "builder/OutputSpec.h"

#include <memory>
#include <vector>

namespace simaai::neat {

/// Construction options for `JpegParse`.
struct JpegParseOptions {
  bool disable_passthrough = true; ///< Force parser processing instead of passthrough.
};

/**
 * @brief JPEG stream parser Node backed by GStreamer's `jpegparse`.
 *
 * Use after multipart or RTP JPEG depayloading and before native JPEG/MJPEG decode.
 *
 * @ingroup nodes_common
 */
class JpegParse final : public Node, public OutputSpecProvider {
public:
  explicit JpegParse(JpegParseOptions opt = {});

  std::string kind() const override {
    return "JpegParse";
  }
  NodeCapsBehavior caps_behavior() const override {
    return NodeCapsBehavior::Dynamic;
  }

  std::string backend_fragment(int node_index) const override;
  std::vector<std::string> element_names(int node_index) const override;
  OutputSpec output_spec(const OutputSpec& input) const override;

  const JpegParseOptions& options() const {
    return opt_;
  }

private:
  JpegParseOptions opt_;
};

} // namespace simaai::neat

namespace simaai::neat::nodes {
/// Convenience factory for a `JpegParse` Node.
std::shared_ptr<simaai::neat::Node> JpegParse(JpegParseOptions opt = {});
} // namespace simaai::neat::nodes
