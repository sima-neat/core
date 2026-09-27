/**
 * @file
 * @ingroup nodes_groups
 * @brief Customer-facing video sender Graph fragment.
 */
#pragma once

#include "pipeline/Graph.h"
#include "nodes/groups/RtspCodec.h"
#include "nodes/sima/SimaEncode.h"

#include <string>

namespace simaai::neat::nodes::groups {

struct VideoSenderRtpOptions {
  int payload_type = 96;
  int config_interval = 1;
};

struct VideoSenderEncoderOptions {
  int bitrate_kbps = 4000;
  std::string profile = "baseline";
  std::string level = "4.0";
};

class VideoSenderOptions {
public:
  /// Encode raw frames, then packetize the selected codec as RTP over UDP.
  static VideoSenderOptions FromRaw(SimaEncodeOptions encode);
  /// @deprecated Use FromRaw with SimaEncodeType::H264.
  static VideoSenderOptions H264RtpUdpFromRaw(int width, int height, int fps);
  [[deprecated("use Passthrough(RtspCodec::H264)")]] static VideoSenderOptions
  H264RtpUdpFromEncoded();

  /// Forward already-encoded frames of `codec` as RTP over UDP without
  /// re-encoding. Throws `std::invalid_argument` for codecs the sender cannot
  /// packetize.
  static VideoSenderOptions Passthrough(RtspCodec codec);

  bool is_raw_input() const {
    return input_kind_ == InputKind::Raw;
  }
  bool is_encoded_input() const {
    return input_kind_ == InputKind::Encoded;
  }
  int width() const {
    return width_;
  }
  int height() const {
    return height_;
  }
  int fps() const {
    return fps_;
  }
  int video_port() const {
    return video_port_base + channel;
  }

  std::string host = "127.0.0.1";
  int channel = 0;
  int video_port_base = 9000;
  bool sync = false;  ///< Wait for packet timestamps against the pipeline clock.
  bool async = false; ///< Permit the sink to wait for its first preroll buffer.
  VideoSenderRtpOptions rtp{};
  VideoSenderEncoderOptions encoder{};

private:
  enum class InputKind { Raw, Encoded };

  VideoSenderOptions() = default;

  InputKind input_kind_ = InputKind::Encoded;
  /// Codec used consistently by the raw encoder and RTP packetization.
  RtspCodec codec_ = RtspCodec::H264;
  int width_ = 0;
  int height_ = 0;
  int fps_ = 0;

  std::optional<std::string> rate_control_;
  std::optional<int> gop_length_;
  std::optional<int> idr_interval_;
  std::optional<int> quality_;
  int num_buffers_ = -1;

  friend simaai::neat::Graph VideoSender(const VideoSenderOptions& opt);
};

simaai::neat::Graph VideoSender(const VideoSenderOptions& opt);

} // namespace simaai::neat::nodes::groups
