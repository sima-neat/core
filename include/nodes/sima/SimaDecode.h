/**
 * @file
 * @ingroup nodes_sima
 * @brief `SimaDecode` Node - native SiMa hardware decoder for encoded frames.
 */
#pragma once

#include "builder/Node.h"
#include "builder/OutputSpec.h"
#include "pipeline/FormatSpec.h"

#include <cstdint>
#include <memory>
#include <string>
#include <vector>

namespace simaai::neat {

/// Encoded media type decoded by `SimaDecode`.
enum class SimaDecodeType {
  H264 = 0, ///< H.264 elementary stream.
  JPEG,     ///< Single JPEG frame.
  MJPEG,    ///< JPEG-frame video stream after framing/depacketization.
};

/**
 * @brief Configuration for `SimaDecode`.
 *
 * The node wraps `neatdecoder`. It receives encoded frames and emits raw video,
 * normally NV12 in SiMaAI memory for downstream CVU/MLA stages.
 */
struct SimaDecodeOptions {
  SimaDecodeType type = SimaDecodeType::H264; ///< Encoded input type.
  int sima_allocator_type = 2;                ///< SiMa allocator type for decoder output buffers.
  FormatSpec out_format = FormatTag::NV12;    ///< Raw output pixel format.
  std::string decoder_name;                   ///< Decoder element name override; empty = auto-pick.
  bool raw_output = true;   ///< If true, emit decoder-native raw buffers directly.
  std::string next_element; ///< Optional next-element selector ("CVU" or "APU") for `neatdecoder`.
  int dec_width = -1;       ///< Decoded frame width override; `-1` = upstream-defined.
  int dec_height = -1;      ///< Decoded frame height override; `-1` = upstream-defined.
  int dec_fps = -1;         ///< Decoded frame rate override; `-1` = upstream-defined.
  int num_buffers = -1;     ///< Output buffer pool size override; `-1` = element default.
  int input_buffers = -1;   ///< Encoded input buffer pool size override; `-1` = element default.
  std::string decoder_tuning; ///< Optional decoder tuning profile; empty = element default.
  bool memory_opt = false;    ///< Enable decoder low-memory mode when supported by the platform.

  // Framework-owned graph admission lease.  These are normally populated by
  // RunCore when a graph contains multiple automatic hardware decoders, so app
  // code can keep using decoder_tuning:auto while the runtime applies a
  // graph-wide pool/tuning decision before any decoder starts allocating.
  bool admission_required = false;
  std::string admission_group_id;
  int admission_stream_index = -1;
  std::uint64_t admission_lease_token_hi = 0;
  std::uint64_t admission_lease_token_lo = 0;
};

/**
 * @brief Hardware-accelerated native decoder node.
 *
 * Use this for encoded streams that should be decoded by the SiMa native decoder.
 * Deprecated `H264Decode` remains available as the compatibility H.264 wrapper.
 *
 * @ingroup nodes_sima
 */
class SimaDecode final : public Node, public OutputSpecProvider {
public:
  explicit SimaDecode(SimaDecodeOptions opt = {});

  std::string kind() const override {
    return "SimaDecode";
  }
  NodeCapsBehavior caps_behavior() const override {
    return NodeCapsBehavior::Dynamic;
  }
  MemoryContract memory_contract() const override {
    return opt_.raw_output ? MemoryContract::PreferDeviceZeroCopy
                           : MemoryContract::AllowEitherButReport;
  }
  std::string buffer_name_hint(int node_index) const override;

  std::string backend_fragment(int node_index) const override;
  std::vector<std::string> element_names(int node_index) const override;
  OutputSpec output_spec(const OutputSpec& input) const override;

  const SimaDecodeOptions& options() const {
    return opt_;
  }

private:
  SimaDecodeOptions opt_;
};

} // namespace simaai::neat

namespace simaai::neat::nodes {
/// Convenience factory for a native `SimaDecode` node.
std::shared_ptr<simaai::neat::Node> SimaDecode(SimaDecodeOptions opt = {});
} // namespace simaai::neat::nodes
