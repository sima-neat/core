/**
 * @file
 * @ingroup nodes_rtp
 * @brief RTP H264 depay node wrapper.
 */
#pragma once
#include "builder/Node.h"
#include "builder/OutputSpec.h"
#include <memory>
#include <vector>

namespace simaai::neat {

class H264Depacketize final : public Node, public OutputSpecProvider {
public:
  // payload_type <= 0 disables RTP payload filtering in caps.
  explicit H264Depacketize(int payload_type = 96, int h264_parse_config_interval = -1,
                           int h264_fps = -1, int h264_width = -1, int h264_height = -1,
                           bool enforce_h264_caps = true);
  std::string kind() const override {
    return "H264Depacketize";
  }
  NodeCapsBehavior caps_behavior() const override {
    if (enforce_h264_caps_ && (h264_fps_ > 0 || h264_width_ > 0 || h264_height_ > 0)) {
      return NodeCapsBehavior::Static;
    }
    return NodeCapsBehavior::Dynamic;
  }
  std::string backend_fragment(int node_index) const override;
  std::vector<std::string> element_names(int node_index) const override;
  OutputSpec output_spec(const OutputSpec& input) const override;

  int payload_type() const {
    return payload_type_;
  }
  int h264_parse_config_interval() const {
    return h264_parse_config_interval_;
  }
  int h264_fps() const {
    return h264_fps_;
  }
  int h264_width() const {
    return h264_width_;
  }
  int h264_height() const {
    return h264_height_;
  }
  bool enforce_h264_caps() const {
    return enforce_h264_caps_;
  }

private:
  int payload_type_ = 96;
  int h264_parse_config_interval_ = -1;
  int h264_fps_ = -1;
  int h264_width_ = -1;
  int h264_height_ = -1;
  bool enforce_h264_caps_ = true;
};

} // namespace simaai::neat

namespace simaai::neat::nodes {
std::shared_ptr<simaai::neat::Node>
H264Depacketize(int payload_type = 96, int h264_parse_config_interval = -1, int h264_fps = -1,
                int h264_width = -1, int h264_height = -1, bool enforce_h264_caps = true);
} // namespace simaai::neat::nodes
