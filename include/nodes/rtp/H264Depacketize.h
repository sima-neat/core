/**
 * @file
 * @ingroup nodes_rtp
 * @brief `H264Depacketize` Node — extracts H.264 NAL units from RTP packets.
 *
 * Wraps `rtph264depay`. Insert between an RTP source (e.g. `RTSPInput` or a
 * `udpsrc`) and an H.264 decoder. Optional caps enforcement injects framerate /
 * geometry into the depayloaded stream when the upstream source is ambiguous.
 */
#pragma once
#include "builder/Node.h"
#include "builder/OutputSpec.h"
#include <memory>
#include <vector>

namespace simaai::neat {

/**
 * @brief Depayloads RTP-encapsulated H.264 into a raw H.264 byte stream.
 *
 * Place between an RTP source and a decoder. When `enforce_h264_caps` is true,
 * the Node enforces `h264_fps`, `h264_width`, and `h264_height` only when all
 * three are provided. If none are provided, caps negotiate dynamically. Partial
 * explicit caps are rejected because they create ambiguous H.264 boundaries.
 *
 * @ingroup nodes_rtp
 */
class H264Depacketize final : public Node, public OutputSpecProvider {
public:
  /// Construct with RTP payload type plus optional H.264 parser / caps overrides.
  /// `payload_type <= 0` disables RTP payload filtering in caps.
  explicit H264Depacketize(int payload_type = 96, int h264_parse_config_interval = -1,
                           int h264_fps = -1, int h264_width = -1, int h264_height = -1,
                           bool enforce_h264_caps = true);
  /// Type label for this Node kind.
  std::string kind() const override {
    return "H264Depacketize";
  }
  /// Whether the Node negotiates static or dynamic caps.
  NodeCapsBehavior caps_behavior() const override {
    if (enforce_h264_caps_ && h264_fps_ > 0 && h264_width_ > 0 && h264_height_ > 0) {
      return NodeCapsBehavior::Static;
    }
    return NodeCapsBehavior::Dynamic;
  }
  /// GStreamer fragment this Node emits.
  std::string backend_fragment(int node_index) const override;
  /// Deterministic element names this Node will create.
  std::vector<std::string> element_names(int node_index) const override;
  /// Negotiated downstream caps produced by this Node.
  OutputSpec output_spec(const OutputSpec& input) const override;

  /// RTP payload type filter; values `<= 0` disable filtering.
  int payload_type() const {
    return payload_type_;
  }
  /// `h264parse` `config-interval` passthrough; `-1` leaves the element default.
  int h264_parse_config_interval() const {
    return h264_parse_config_interval_;
  }
  /// Framerate enforced on depayloaded caps, in fps; `-1` = unspecified.
  int h264_fps() const {
    return h264_fps_;
  }
  /// Width enforced on depayloaded caps, in pixels; `-1` = unspecified.
  int h264_width() const {
    return h264_width_;
  }
  /// Height enforced on depayloaded caps, in pixels; `-1` = unspecified.
  int h264_height() const {
    return h264_height_;
  }
  /// True if caps enforcement is enabled when geometry/fps overrides are set.
  bool enforce_h264_caps() const {
    return enforce_h264_caps_;
  }

private:
  int payload_type_ = 96;
  int h264_parse_config_interval_ = -1;
  int h264_fps_ = -1;
  int h264_width_ = -1;
  int h264_height_ = -1;
  bool enforce_h264_caps_ = true;
};

} // namespace simaai::neat

namespace simaai::neat::nodes {
/// Convenience factory for an `H264Depacketize` Node.
std::shared_ptr<simaai::neat::Node>
H264Depacketize(int payload_type = 96, int h264_parse_config_interval = -1, int h264_fps = -1,
                int h264_width = -1, int h264_height = -1, bool enforce_h264_caps = true);
} // namespace simaai::neat::nodes
