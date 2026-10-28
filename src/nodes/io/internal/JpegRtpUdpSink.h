#pragma once

#include "builder/Node.h"
#include "builder/OutputSpec.h"
#include "builder/internal/InputSpecSpecialization.h"

#include <memory>
#include <string>
#include <utility>
#include <vector>

namespace simaai::neat::internal {

/// Scheduler element that packetizes JPEG frames as RTP/JPEG and sends them over UDP.
inline constexpr const char* kJpegRtpUdpSinkFactory = "neatjpegudpsink";

struct JpegRtpUdpSinkOptions {
  std::string host = "127.0.0.1";
  int port = 5000;
  int payload_type = 26;
  int mtu = 1400;
  bool sync = false;
  bool async = false;
};

/// Terminal sink that replaces JpegPacketize + UdpOutput when the installed Scheduler has
/// `neatjpegudpsink`. The fusion compiler matches its kind() by name; keep it stable.
///
/// The sink reads each frame's JPEG headers itself, so it needs no parser, but its caps require
/// width and height. Until the compiler knows the input carries them, the fragment keeps a
/// `jpegparse` in front of the sink; an input with known dimensions (a USB camera, an encoder,
/// an Input with sized caps) specializes to the bare sink.
class JpegRtpUdpSink final : public Node, public InputSpecSpecializer {
public:
  enum class Input { ParseDimensions, SizedCaps };

  explicit JpegRtpUdpSink(JpegRtpUdpSinkOptions opt, Input input = Input::ParseDimensions)
      : opt_(std::move(opt)), input_(input) {}

  std::string kind() const override {
    return "JpegRtpUdpSink";
  }
  NodeCapsBehavior caps_behavior() const override {
    return NodeCapsBehavior::Dynamic;
  }
  std::string backend_fragment(int node_index) const override;
  std::vector<std::string> element_names(int node_index) const override;
  std::shared_ptr<Node>
  specialize_for_input(const OutputSpec& input,
                       const InputSpecSpecializationContext& context) const override;

  const JpegRtpUdpSinkOptions& options() const {
    return opt_;
  }
  Input input() const {
    return input_;
  }

private:
  JpegRtpUdpSinkOptions opt_;
  Input input_;
};

} // namespace simaai::neat::internal
