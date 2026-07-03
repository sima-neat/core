/**
 * @file
 * @ingroup nodes_rtp
 * @brief `RTPJpegDepacketize` Node - extracts JPEG frames from RTP packets.
 *
 * Wraps `rtpjpegdepay`. Insert between an RTP source such as `RTSPInput` and
 * `JpegParse` when consuming RTSP MJPEG streams.
 */
#pragma once

#include "builder/Node.h"
#include "builder/OutputSpec.h"

#include <memory>
#include <vector>

namespace simaai::neat {

/**
 * @brief Depayloads RTP-encapsulated JPEG frames into `image/jpeg` buffers.
 *
 * @ingroup nodes_rtp
 */
class RTPJpegDepacketize final : public Node, public OutputSpecProvider {
public:
  /// Construct with RTP payload type. `payload_type <= 0` disables payload filtering.
  explicit RTPJpegDepacketize(int payload_type = 26);

  std::string kind() const override {
    return "RTPJpegDepacketize";
  }
  NodeCapsBehavior caps_behavior() const override {
    return NodeCapsBehavior::Dynamic;
  }

  std::string backend_fragment(int node_index) const override;
  std::vector<std::string> element_names(int node_index) const override;
  OutputSpec output_spec(const OutputSpec& input) const override;

  /// RTP payload type filter; values `<= 0` disable filtering.
  int payload_type() const {
    return payload_type_;
  }

private:
  int payload_type_ = 26;
};

} // namespace simaai::neat

namespace simaai::neat::nodes {
/// Convenience factory for an `RTPJpegDepacketize` Node.
std::shared_ptr<simaai::neat::Node> RTPJpegDepacketize(int payload_type = 26);
} // namespace simaai::neat::nodes
