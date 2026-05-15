/**
 * @file
 * @ingroup nodes_groups
 * @brief Runtime helper for UDP H.264 video fan-out.
 */
#pragma once

#include "pipeline/Run.h"
#include "pipeline/SessionOptions.h"

#include <cstddef>
#include <memory>
#include <string>
#include <vector>

namespace simaai::neat::nodes::groups {

struct UdpOutputNodeGroupOptions {
  std::string h264_caps;
  int payload_type = 96;
  int config_interval = 1;
  bool enable_timings = false;
  std::string host = "127.0.0.1";
  int video_port_base = 9000;
  bool udp_sync = false;
  bool udp_async = false;
};

class UdpOutputNodeGroup {
public:
  bool init(const UdpOutputNodeGroupOptions& opt, size_t streams, std::string* err = nullptr);

  bool push_video(size_t idx, const simaai::neat::Sample& sample) const;
  bool try_push_video(size_t idx, const simaai::neat::Sample& sample) const;

  void stop();
  size_t size() const {
    return runs_.size();
  }

  const std::vector<std::shared_ptr<simaai::neat::Run>>& runs() const {
    return runs_;
  }

private:
  UdpOutputNodeGroupOptions opt_{};
  std::vector<std::shared_ptr<simaai::neat::Run>> runs_;
};

} // namespace simaai::neat::nodes::groups
