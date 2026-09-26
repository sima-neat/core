/**
 * @file
 * @ingroup nodes_sima
 * @brief `H264EncodeSima` Node — hardware-accelerated H.264 encoder.
 *
 * Wraps the SiMa hardware H.264 encoder. Use after a video source (raw frames) when
 * running on a SiMa board. Pair with `H264Packetize` for RTP streaming or with a
 * file muxer for writing to disk.
 */
#pragma once

#include "builder/Node.h"
#include "builder/OutputSpec.h"
#include "nodes/sima/SimaEncode.h"

#include <memory>
#include <string>
#include <vector>

namespace simaai::neat {

/**
 * @brief Hardware-accelerated H.264 encoder Node.
 * @deprecated Use SimaEncode with SimaEncodeType::H264 for new code.
 *
 * @ingroup nodes_sima
 */
class H264EncodeSima final : public Node, public OutputSpecProvider {
public:
  /**
   * @brief Construct with stream parameters and encoder profile.
   *
   * @param w            Frame width in pixels.
   * @param h            Frame height in pixels.
   * @param fps          Target encode framerate.
   * @param bitrate_kbps Target output bitrate in kbps (default `4000`).
   * @param profile      H.264 profile string (`"baseline"`, `"main"`, `"high"`).
   * @param level        H.264 level string (e.g. `"4.0"`).
   */
  H264EncodeSima(int w, int h, int fps, int bitrate_kbps = 4000, std::string profile = "baseline",
                 std::string level = "4.0");

  /// Type label for this Node kind.
  std::string kind() const override {
    return "H264EncodeSima";
  }
  /// Whether the Node negotiates static or dynamic caps.
  NodeCapsBehavior caps_behavior() const override {
    return NodeCapsBehavior::Static;
  }

  /// GStreamer fragment this Node emits.
  std::string backend_fragment(int node_index) const override;
  /// Deterministic element names this Node will create.
  std::vector<std::string> element_names(int node_index) const override;
  /// Negotiated downstream caps produced by this Node.
  OutputSpec output_spec(const OutputSpec& input) const override;

  /// Configured frame width.
  int width() const {
    return options_.width;
  }
  /// Configured frame height.
  int height() const {
    return options_.height;
  }
  /// Configured target framerate.
  int fps() const {
    return options_.fps;
  }
  /// Configured target bitrate, kbps.
  int bitrate_kbps() const {
    return *options_.bitrate_kbps;
  }
  /// Configured H.264 profile string.
  const std::string& profile() const {
    return *options_.profile;
  }
  /// Configured H.264 level string.
  const std::string& level() const {
    return *options_.level;
  }

private:
  SimaEncodeOptions options_;
};

} // namespace simaai::neat

namespace simaai::neat::nodes {
/// Convenience factory for a hardware `H264EncodeSima` Node.
/// @deprecated Use SimaEncode with SimaEncodeType::H264.
std::shared_ptr<simaai::neat::Node> H264EncodeSima(int w, int h, int fps, int bitrate_kbps = 4000,
                                                   std::string profile = "baseline",
                                                   std::string level = "4.0");

/// Software H.264 encoder factory — picks `x264enc`/`openh264enc`/`avenc_h264` at runtime.
std::shared_ptr<simaai::neat::Node> H264EncodeSW(int bitrate_kbps = 4000);
} // namespace simaai::neat::nodes
