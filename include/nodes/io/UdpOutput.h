/**
 * @file
 * @ingroup nodes_io
 * @brief `UdpOutput` Node — terminal sink that writes packets to a UDP destination.
 *
 * Wraps `udpsink`. Place at the tail of a Graph that streams over the network
 * (RTP video, telemetry forwarding, etc.). Pair with an upstream payloader for
 * RTP, or feed raw packets directly for custom protocols.
 */
#pragma once

#include "builder/Node.h"

#include <memory>
#include <string>
#include <vector>

namespace simaai::neat {

/**
 * @brief Destination + sync options for the `UdpOutput` Node.
 *
 * @ingroup nodes_io
 */
struct UdpOutputOptions {
  std::string host = "127.0.0.1"; ///< Destination host or IP for outgoing UDP packets.
  int port = 5000;                ///< Destination UDP port.
  bool sync = false;              ///< If true, sync to the pipeline clock before sending.
  bool async = false;             ///< If true, allow asynchronous state changes on the sink.
};

/**
 * @brief UDP sink Node — writes packets to a UDP destination.
 *
 * Use as the final stage of an RTP streaming pipeline or any Graph that
 * forwards data over UDP. For RTP video, place after an H.264 payloader.
 *
 * @ingroup nodes_io
 */
class UdpOutput final : public Node {
public:
  /// Construct with destination + sync options.
  explicit UdpOutput(UdpOutputOptions opt) : opt_(std::move(opt)) {}

  /// Type label for this Node kind.
  std::string kind() const override {
    return "UdpOutput";
  }
  /// Whether the Node negotiates static or dynamic caps.
  NodeCapsBehavior caps_behavior() const override {
    return NodeCapsBehavior::Dynamic;
  }
  /// GStreamer fragment this Node emits.
  std::string backend_fragment(int node_index) const override;
  /// Deterministic element names this Node will create.
  std::vector<std::string> element_names(int node_index) const override;

  /// Destination and sync options this Node was constructed with.
  const UdpOutputOptions& options() const {
    return opt_;
  }

private:
  UdpOutputOptions opt_;
};

} // namespace simaai::neat

namespace simaai::neat::nodes {
/// Convenience factory for a `UdpOutput` Node with optional `UdpOutputOptions`.
std::shared_ptr<simaai::neat::Node> UdpOutput(UdpOutputOptions opt = {});
} // namespace simaai::neat::nodes
