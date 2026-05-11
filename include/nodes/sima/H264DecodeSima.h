/**
 * @file
 * @ingroup nodes_sima
 * @brief `H264Decode` Node — hardware-accelerated H.264 video decoder via SiMa's plugin.
 *
 * Wraps the SiMa hardware H.264 decoder element. Faster than the software fallback and
 * uses the SiMa DMA allocator so decoded frames flow into downstream MLA/CVU stages
 * without an extra copy.
 */
#pragma once

#include "builder/Node.h"
#include "builder/OutputSpec.h"

#include <memory>
#include <string>
#include <vector>

namespace simaai::neat {

/**
 * @brief Hardware-accelerated H.264 decoder Node.
 *
 * Place after an `H264Parse` (or any AU-aligned H.264 source) when running on a SiMa
 * board. The decoder produces raw video (NV12 by default) ready for downstream
 * preprocessing.
 *
 * @ingroup nodes_sima
 */
class H264Decode final : public Node, public OutputSpecProvider {
public:
  /**
   * @brief Construct with explicit decoder configuration.
   *
   * @param sima_allocator_type DMA allocator selector (board-specific; default `2`).
   * @param out_format          Raw output pixel format (e.g. `"NV12"`, `"I420"`).
   * @param decoder_name        Override decoder element name; empty = auto-pick.
   * @param raw_output          If true, emit raw frames without GStreamer metadata adapters.
   * @param next_element        Optional name of the downstream element used for wiring.
   * @param dec_width           Decoded frame width override; `-1` = let upstream decide.
   * @param dec_height          Decoded frame height override; `-1` = let upstream decide.
   * @param dec_fps             Decoded framerate override; `-1` = let upstream decide.
   * @param num_buffers         Output buffer pool size; `-1` = element default.
   */
  H264Decode(int sima_allocator_type = 2, std::string out_format = "NV12",
             std::string decoder_name = {}, bool raw_output = false, std::string next_element = {},
             int dec_width = -1, int dec_height = -1, int dec_fps = -1, int num_buffers = -1);
  /// Type label for this Node kind.
  std::string kind() const override {
    return "H264Decode";
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

  /// SiMa DMA allocator type passed to the decoder element.
  int sima_allocator_type() const {
    return sima_allocator_type_;
  }
  /// Configured raw output format.
  const std::string& out_format() const {
    return out_format_;
  }
  /// Decoder element name override (empty = auto-pick).
  const std::string& decoder_name() const {
    return decoder_name_;
  }
  /// Whether the decoder emits raw frames without metadata adapters.
  bool raw_output() const {
    return raw_output_;
  }
  /// Optional downstream element name used during graph wiring.
  const std::string& next_element() const {
    return next_element_;
  }
  /// Decoded frame width override (`-1` = upstream-defined).
  int dec_width() const {
    return dec_width_;
  }
  /// Decoded frame height override (`-1` = upstream-defined).
  int dec_height() const {
    return dec_height_;
  }
  /// Decoded framerate override (`-1` = upstream-defined).
  int dec_fps() const {
    return dec_fps_;
  }
  /// Output buffer pool size override (`-1` = element default).
  int num_buffers() const {
    return num_buffers_;
  }

private:
  int sima_allocator_type_ = 2;
  std::string out_format_ = "NV12";
  std::string decoder_name_;
  bool raw_output_ = false;
  std::string next_element_;
  int dec_width_ = -1;
  int dec_height_ = -1;
  int dec_fps_ = -1;
  int num_buffers_ = -1;
};

} // namespace simaai::neat

namespace simaai::neat::nodes {
/// Convenience factory for a hardware `H264Decode` Node — see the class docs for parameters.
std::shared_ptr<simaai::neat::Node>
H264Decode(int sima_allocator_type = 2, std::string out_format = "NV12",
           std::string decoder_name = {}, bool raw_output = false, std::string next_element = {},
           int dec_width = -1, int dec_height = -1, int dec_fps = -1, int num_buffers = -1);
} // namespace simaai::neat::nodes
