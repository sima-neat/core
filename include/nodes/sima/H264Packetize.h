/**
 * @file
 * @ingroup nodes_sima
 * @brief `H264Packetize` Node — wraps encoded H.264 into RTP packets for network streaming.
 *
 * Wraps GStreamer's `rtph264pay` element. Place after `H264EncodeSima` (or any H.264
 * encoder) when streaming over RTP/RTSP — it splits NAL units into RTP payloads and
 * periodically re-injects SPS/PPS so late-joining receivers can decode.
 */
#pragma once

#include "builder/Node.h"
#include "builder/OutputSpec.h"

#include <memory>
#include <vector>

namespace simaai::neat {

/**
 * @brief RTP H.264 payloader Node — wraps encoded H.264 into RTP packets.
 *
 * @ingroup nodes_sima
 */
class H264Packetize final : public Node, public OutputSpecProvider {
public:
  /// RTP dynamic payload type (default `96`, the conventional value for H.264 over RTP).
  struct PayloadType {
    int value;                                      ///< Underlying integer payload type.
    /// Construct from raw integer value (defaults to 96).
    constexpr PayloadType(int v = 96) : value(v) {}
  };

  /// SPS/PPS re-injection interval in seconds (default `1`).
  struct ConfigInterval {
    int value;                                          ///< Underlying integer interval, in seconds.
    /// Construct from raw integer value (defaults to 1).
    constexpr ConfigInterval(int v = 1) : value(v) {}
  };

  /// Construct with optional payload type and SPS/PPS re-injection interval.
  H264Packetize(PayloadType pt = PayloadType{}, ConfigInterval config_interval = ConfigInterval{});
  /// Type label for this Node kind.
  std::string kind() const override {
    return "H264Packetize";
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

  /// Configured RTP payload type.
  int pt() const {
    return pt_;
  }
  /// Configured SPS/PPS re-injection interval (seconds).
  int config_interval() const {
    return config_interval_;
  }

private:
  int pt_ = 96;
  int config_interval_ = 1;
};

} // namespace simaai::neat

namespace simaai::neat::nodes {
/// Convenience factory for an `H264Packetize` Node.
std::shared_ptr<simaai::neat::Node>
H264Packetize(simaai::neat::H264Packetize::PayloadType pt = {},
              simaai::neat::H264Packetize::ConfigInterval config_interval = {});
} // namespace simaai::neat::nodes
