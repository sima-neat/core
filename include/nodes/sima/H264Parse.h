// include/nodes/sima/H264Parse.h
/**
 * @file
 * @ingroup nodes_sima
 * @brief SimaAI H264 parse node wrapper.
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
struct H264ParseOptions {
  // Sends SPS/PPS periodically. Commonly required for RTP/RTSP robustness.
  // (Maps to h264parse property: config-interval)
  int config_interval = 1;

  // Optional: enforce downstream caps after h264parse via a capsfilter.
  // This is how we express AU alignment and stream-format deterministically.
  enum class Alignment { Auto, AU, NAL };
  enum class StreamFormat { Auto, AVC, ByteStream };

  Alignment alignment = Alignment::Auto;
  StreamFormat stream_format = StreamFormat::Auto;

  // If true, we add a named capsfilter after h264parse.
  // It will include only the fields that are not Auto.
  bool enforce_caps = false;
};

class H264Parse final : public Node, public OutputSpecProvider {
public:
  explicit H264Parse(H264ParseOptions opt = {});
  std::string kind() const override {
    return "H264Parse";
  }
  NodeCapsBehavior caps_behavior() const override {
    return opt_.enforce_caps ? NodeCapsBehavior::Static : NodeCapsBehavior::Dynamic;
  }

  std::string backend_fragment(int node_index) const override;
  std::vector<std::string> element_names(int node_index) const override;
  OutputSpec output_spec(const OutputSpec& input) const override;

  const H264ParseOptions& options() const {
    return opt_;
  }

private:
  H264ParseOptions opt_;
};

} // namespace simaai::neat

namespace simaai::neat::nodes {

// Fully configurable (power user).
std::shared_ptr<simaai::neat::Node> H264Parse(simaai::neat::H264ParseOptions opt);

// Convenience: old behavior (only config-interval, no enforced caps).
inline std::shared_ptr<simaai::neat::Node> H264Parse(int config_interval = 1) {
  simaai::neat::H264ParseOptions opt;
  opt.config_interval = config_interval;
  opt.enforce_caps = false;
  return H264Parse(opt);
}

// Convenience preset: “AU-aligned access units” (replaces H264ParseAu node).
// This should be the default for demux->decode paths where AU boundaries matter.
inline std::shared_ptr<simaai::neat::Node> H264ParseAu(int config_interval = 1) {
  simaai::neat::H264ParseOptions opt;
  opt.config_interval = config_interval;
  opt.enforce_caps = true;
  opt.alignment = simaai::neat::H264ParseOptions::Alignment::AU;
  // stream_format left Auto unless you know you need ByteStream/AVC.
  return H264Parse(opt);
}

// Convenience preset: “Streaming-safe” defaults.
// Good for RTP/RTSP publishing where late-joiners need SPS/PPS.
inline std::shared_ptr<simaai::neat::Node> H264ParseForRtp(int config_interval = 1) {
  simaai::neat::H264ParseOptions opt;
  opt.config_interval = config_interval; // usually 1
  opt.enforce_caps = true;
  opt.alignment = simaai::neat::H264ParseOptions::Alignment::AU;
  // stream_format left Auto to avoid forcing an incorrect mode for some MPKs.
  return H264Parse(opt);
}

} // namespace simaai::neat::nodes
