// include/nodes/sima/H264Parse.h
/**
 * @file
 * @ingroup nodes_sima
 * @brief `H264Parse` Node — parses an H.264 stream into NAL units, optionally enforcing caps.
 *
 * Wraps GStreamer's `h264parse` element and an optional `capsfilter` to deterministically
 * lock the access-unit alignment and stream format. Use upstream of any H.264 decoder when
 * the source isn't already AU-aligned, and upstream of RTP packetizers where receivers need
 * periodic SPS/PPS injection.
 */
#pragma once

#include "builder/Node.h"
#include "builder/OutputSpec.h"

#include <memory>
#include <optional>
#include <string>
#include <vector>

namespace simaai::neat {

// GStreamer h264parse + (optional) caps enforcement.
//
// Why caps enforcement is inside this node:
// - AU/NAL alignment is often a CAPS negotiation detail (not always a property)
// - Keeping it here avoids users sprinkling raw capsfilter strings everywhere
// - Still prints a clean gst-launch repro string

/**
 * @brief Construction options for an `H264Parse` Node — config-interval and optional caps lock.
 *
 * @ingroup nodes_sima
 */
struct H264ParseOptions {
  /// Sends SPS/PPS periodically (seconds). Commonly required for RTP/RTSP robustness.
  /// Maps to the `h264parse` `config-interval` property.
  int config_interval = 1;

  // Optional: enforce downstream caps after h264parse via a capsfilter.
  // This is how we express AU alignment and stream-format deterministically.
  /// Access-unit alignment to enforce on the downstream caps.
  enum class Alignment { Auto, AU, NAL };
  /// Byte-stream vs. AVC framing to enforce on the downstream caps.
  enum class StreamFormat { Auto, AVC, ByteStream };

  Alignment alignment = Alignment::Auto;       ///< Alignment to enforce when `enforce_caps` is true.
  StreamFormat stream_format = StreamFormat::Auto; ///< Stream-format to enforce when `enforce_caps` is true.

  /// If true, append a named capsfilter after h264parse with the non-`Auto` fields applied.
  bool enforce_caps = false;
};

/**
 * @brief H.264 stream parser Node — parses NAL units, finds keyframes, optionally locks caps.
 *
 * @ingroup nodes_sima
 */
class H264Parse final : public Node, public OutputSpecProvider {
public:
  /// Construct with optional `H264ParseOptions`.
  explicit H264Parse(H264ParseOptions opt = {});
  /// Type label for this Node kind.
  std::string kind() const override {
    return "H264Parse";
  }
  /// Whether the Node negotiates static or dynamic caps.
  NodeCapsBehavior caps_behavior() const override {
    return opt_.enforce_caps ? NodeCapsBehavior::Static : NodeCapsBehavior::Dynamic;
  }

  /// GStreamer fragment this Node emits.
  std::string backend_fragment(int node_index) const override;
  /// Deterministic element names this Node will create.
  std::vector<std::string> element_names(int node_index) const override;
  /// Negotiated downstream caps produced by this Node.
  OutputSpec output_spec(const OutputSpec& input) const override;

  /// Inspect the Node's options.
  const H264ParseOptions& options() const {
    return opt_;
  }

private:
  H264ParseOptions opt_;
};

} // namespace simaai::neat

namespace simaai::neat::nodes {

/// Fully configurable `H264Parse` factory (power-user form).
std::shared_ptr<simaai::neat::Node> H264Parse(simaai::neat::H264ParseOptions opt);

/// Convenience: legacy behavior — only `config_interval`, no enforced caps.
inline std::shared_ptr<simaai::neat::Node> H264Parse(int config_interval = 1) {
  simaai::neat::H264ParseOptions opt;
  opt.config_interval = config_interval;
  opt.enforce_caps = false;
  return H264Parse(opt);
}

/// Convenience preset: AU-aligned access units. Default for demux→decode paths where AU boundaries matter.
inline std::shared_ptr<simaai::neat::Node> H264ParseAu(int config_interval = 1) {
  simaai::neat::H264ParseOptions opt;
  opt.config_interval = config_interval;
  opt.enforce_caps = true;
  opt.alignment = simaai::neat::H264ParseOptions::Alignment::AU;
  // stream_format left Auto unless you know you need ByteStream/AVC.
  return H264Parse(opt);
}

/// Convenience preset: streaming-safe defaults for RTP/RTSP publishing where late-joiners need SPS/PPS.
inline std::shared_ptr<simaai::neat::Node> H264ParseForRtp(int config_interval = 1) {
  simaai::neat::H264ParseOptions opt;
  opt.config_interval = config_interval; // usually 1
  opt.enforce_caps = true;
  opt.alignment = simaai::neat::H264ParseOptions::Alignment::AU;
  // stream_format left Auto to avoid forcing an incorrect mode for some MPKs.
  return H264Parse(opt);
}

} // namespace simaai::neat::nodes
