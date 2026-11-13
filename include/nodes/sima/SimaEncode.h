/**
 * @file
 * @ingroup nodes_sima
 * @brief Native hardware encoding of raw frames to H.264 or H.265 (MJPEG encode is rejected).
 */
#pragma once

#include "builder/Node.h"
#include "builder/OutputSpec.h"

#include <memory>
#include <optional>
#include <string>
#include <vector>

namespace simaai::neat {

/// Encoded output type produced by `SimaEncode`.
enum class SimaEncodeType {
  H264 = 0,
  AVC = H264,
  H265 = 1,
  HEVC = H265,
  MJPEG = 2, ///< Rejected by validation: simaaiencoder has no JPEG encode.
};

/// Codec-specific fields are optional; defaults are selected after `type`.
struct SimaEncodeOptions {
  SimaEncodeType type = SimaEncodeType::H264;
  int fps = 30;                            ///< Stream cadence, not a producer submission throttle.
  std::optional<int> bitrate_kbps;         ///< H.264/H.265: positive target, default 4000.
  std::optional<std::string> rate_control; ///< H.264/H.265: vbr (default) or cbr.
  std::optional<std::string> profile;      ///< H.264: baseline/main/high; H.265: main.
  std::optional<std::string> level;        ///< H.264/H.265: default 4.0, adjusted by hardware.
  std::optional<int> gop_length;           ///< H.264/H.265: 0..1000; 0 uses FPS, 1 is all-intra.
  std::optional<int> idr_interval;         ///< H.264/H.265: nonnegative; 0 uses the GOP length.
  std::optional<int> quality;              ///< MJPEG only; must stay unset (MJPEG is rejected).
  int num_buffers = -1; ///< Encoded outputs: -1 uses the element default (10), or 2..64.
};

namespace internal {
struct SimaEncodeAccess;
}

/**
 * @brief Hardware encoder with native 8-bit NV12 input preparation.
 *
 * Uses simaaiencoder. NV12 dma-bufs from the codec heap with a compatible layout
 * can be encoded without a copy; other NV12 input is copied by the encoder, and other
 * raw formats are converted to NV12 with videoconvert.
 * Resolution follows the initial input frames and stays fixed for a run.
 * This node does not resize frames.
 * Output consists of encoded frames, without RTP packetization.
 * @ingroup nodes_sima
 */
class SimaEncode final : public Node, public OutputSpecProvider {
public:
  explicit SimaEncode(SimaEncodeOptions options = {});
  std::string kind() const override {
    return "SimaEncode";
  }
  NodeCapsBehavior caps_behavior() const override {
    // Live size changes require draining and restarting the codec.
    return NodeCapsBehavior::Static;
  }
  MemoryContract memory_contract() const override;
  std::string backend_fragment(int node_index) const override;
  std::vector<std::string> element_names(int node_index) const override;
  OutputSpec output_spec(const OutputSpec& input) const override;
  const SimaEncodeOptions& options() const {
    return options_;
  }

private:
  friend struct internal::SimaEncodeAccess;
  SimaEncode(SimaEncodeOptions options, bool prepare_input);
  SimaEncodeOptions options_;
  std::shared_ptr<Node> input_adapter_;
};

} // namespace simaai::neat

namespace simaai::neat::nodes {
std::shared_ptr<simaai::neat::Node> SimaEncode(SimaEncodeOptions options = {});
} // namespace simaai::neat::nodes
