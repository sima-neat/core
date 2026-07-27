/**
 * @file
 * @ingroup nodes_groups
 * @brief Customer-facing video sender Graph fragment.
 */
#pragma once

#include "pipeline/Graph.h"

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
  static VideoSenderOptions H264RtpUdpFromRaw(int width, int height, int fps);
  static VideoSenderOptions H264RtpUdpFromEncoded();
  static VideoSenderOptions H265RtpUdpFromEncoded();

  bool is_raw_input() const {
    return input_kind_ == InputKind::Raw;
  }
  bool is_encoded_input() const {
    return input_kind_ == InputKind::EncodedH264 || input_kind_ == InputKind::EncodedH265;
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
  bool sync = false;
  bool async = false;
  VideoSenderRtpOptions rtp{};
  VideoSenderEncoderOptions encoder{};

private:
  enum class InputKind { Raw, EncodedH264, EncodedH265 };

  VideoSenderOptions() = default;

  InputKind input_kind_ = InputKind::EncodedH264;
  int width_ = 0;
  int height_ = 0;
  int fps_ = 0;

  friend simaai::neat::Graph VideoSender(const VideoSenderOptions& opt);
};

simaai::neat::Graph VideoSender(const VideoSenderOptions& opt);

} // namespace simaai::neat::nodes::groups
