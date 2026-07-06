/**
 * @file
 * @ingroup nodes_common
 * @brief `MultipartJpegDemux` Node — extracts parts from multipart MJPEG streams.
 */
#pragma once

#include "builder/Node.h"

#include <memory>
#include <string>
#include <vector>

namespace simaai::neat {

/// Construction options for `MultipartJpegDemux`.
struct MultipartJpegDemuxOptions {
  std::string boundary;       ///< Optional multipart boundary override; empty = auto-detect.
  bool single_stream = false; ///< If true, assume the multipart content type is stable.
};

/**
 * @brief Demux multipart HTTP streams into per-part buffers.
 *
 * This wraps GStreamer's `multipartdemux`. For MJPEG streams, place `JpegParse`
 * after this node to normalize each part into parsed `image/jpeg` frames.
 *
 * @ingroup nodes_common
 */
class MultipartJpegDemux final : public Node {
public:
  explicit MultipartJpegDemux(MultipartJpegDemuxOptions opt = {});

  std::string kind() const override {
    return "MultipartJpegDemux";
  }
  NodeCapsBehavior caps_behavior() const override {
    return NodeCapsBehavior::Dynamic;
  }

  std::string backend_fragment(int node_index) const override;
  std::vector<std::string> element_names(int node_index) const override;

  const MultipartJpegDemuxOptions& options() const {
    return opt_;
  }

private:
  MultipartJpegDemuxOptions opt_;
};

} // namespace simaai::neat

namespace simaai::neat::nodes {
/// Convenience factory for a `MultipartJpegDemux` Node.
std::shared_ptr<simaai::neat::Node> MultipartJpegDemux(MultipartJpegDemuxOptions opt = {});
} // namespace simaai::neat::nodes
