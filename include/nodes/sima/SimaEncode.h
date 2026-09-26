/**
 * @file
 * @ingroup nodes_sima
 * @brief Native hardware encoding of raw frames to H.264, H.265 or MJPEG.
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
  MJPEG = 2,
};

/// Codec-specific fields are optional; defaults are selected after `type`.
struct SimaEncodeOptions {
  SimaEncodeType type = SimaEncodeType::H264;
  int width = 0;                           ///< Positive even frame width.
  int height = 0;                          ///< Positive even frame height.
  int fps = 30;                            ///< Stream cadence, not a producer submission throttle.
  std::optional<int> bitrate_kbps;         ///< H.264/H.265: positive target, default 4000.
  std::optional<std::string> rate_control; ///< H.264/H.265: vbr (default) or cbr.
  std::optional<std::string> profile;      ///< H.264: baseline/main/high; H.265: main.
  std::optional<std::string> level;        ///< H.264/H.265: default 4.0, adjusted by hardware.
  std::optional<int> gop_length;           ///< H.264/H.265: 0..1000; 0 uses FPS, 1 is all-intra.
  std::optional<int> idr_interval;         ///< H.264/H.265: nonnegative; 0 uses three times FPS.
  std::optional<int> quality;              ///< MJPEG: 1..100, default 80.
  int num_buffers = -1; ///< Encoded outputs: -1 uses the native default, or 1..20.
};

namespace internal {
struct SimaEncodeAccess;
}

/**
 * @brief Hardware encoder with native 8-bit NV12 input preparation.
 *
 * Compatible SiMaAI NV12 storage is retained without copying. Other supported
 * raw CPU layouts are materialized by the existing encoder input adapter.
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
    return NodeCapsBehavior::Static;
  }
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
  bool prepare_input_;
};

} // namespace simaai::neat

namespace simaai::neat::nodes {
std::shared_ptr<simaai::neat::Node> SimaEncode(SimaEncodeOptions options = {});
} // namespace simaai::neat::nodes
