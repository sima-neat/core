/**
 * @file
 * @ingroup nodes_io
 * @brief `RTSPInput` Node — source that reads from an RTSP server.
 *
 * Wraps `rtspsrc`. Place at the head of a Graph that consumes a network camera
 * or other RTSP stream; downstream stages typically chain RTP depayload and an
 * H.264 decoder. Because it carries `InputRole::Source`, the Graph is driven
 * with `Run::run()` (no `push()`).
 */
#pragma once
#include "builder/Node.h"
#include <memory>
#include <string>
#include <vector>

namespace simaai::neat {

/**
 * @brief RTSP source Node — pulls a stream from an RTSP URL.
 *
 * Pair with `H264Depacketize` (and optionally `H264CapsFixup`) plus a decoder to
 * obtain raw frames. The Graph uses `Graph::run()` since this Node is a
 * source and no samples are pushed by the application.
 *
 * @ingroup nodes_io
 */
class RTSPInput final : public Node {
public:
  /// Construct with an RTSP URL plus optional jitter-buffer / transport tuning.
  RTSPInput(std::string url, int latency_ms = 200, bool tcp = true, bool drop_on_latency = false,
            std::string buffer_mode = "");

  /// Type label for this Node kind.
  std::string kind() const override {
    return "RTSPInput";
  }
  /// User-facing label for this Node.
  std::string user_label() const override {
    return url_;
  }
  /// Role this Node plays as a stream source.
  InputRole input_role() const override {
    return InputRole::Source;
  }
  /// Whether the Node negotiates static or dynamic caps.
  NodeCapsBehavior caps_behavior() const override {
    return NodeCapsBehavior::Dynamic;
  }

  /// GStreamer fragment this Node emits.
  std::string backend_fragment(int node_index) const override;
  /// Deterministic element names this Node will create.
  std::vector<std::string> element_names(int node_index) const override;

  /// The RTSP URL this Node was constructed with.
  const std::string& url() const {
    return url_;
  }
  /// Jitter-buffer latency, in milliseconds.
  int latency_ms() const {
    return latency_ms_;
  }
  /// True if the RTSP transport is forced to TCP (false = UDP).
  bool tcp() const {
    return tcp_;
  }
  /// True if late buffers should be dropped when the jitter-buffer overflows.
  bool drop_on_latency() const {
    return drop_on_latency_;
  }
  /// Buffer-mode label passed through to `rtspsrc` (empty = default).
  const std::string& buffer_mode() const {
    return buffer_mode_;
  }

private:
  std::string url_;
  int latency_ms_ = 200;
  bool tcp_ = true;
  bool drop_on_latency_ = false;
  std::string buffer_mode_;
};

} // namespace simaai::neat

namespace simaai::neat::nodes {
/// Convenience factory for an `RTSPInput` Node.
std::shared_ptr<simaai::neat::Node> RTSPInput(std::string url, int latency_ms = 200,
                                              bool tcp = true, bool drop_on_latency = false,
                                              std::string buffer_mode = "");
} // namespace simaai::neat::nodes
