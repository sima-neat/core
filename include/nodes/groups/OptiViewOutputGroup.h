/**
 * @file
 * @ingroup nodes_groups
 * @brief Runtime helpers for OptiView UDP video + JSON fan-out.
 */
#pragma once

#include "nodes/io/OptiViewJsonOutput.h"
#include "pipeline/SessionOptions.h"
#include "pipeline/Run.h"

#include <cstddef>
#include <cstdint>
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

struct OptiViewOutputNodeGroupOptions {
  UdpOutputNodeGroupOptions udp{};
  bool send_json = true;
  int json_port_base = 9100;
  int frame_w = 0;
  int frame_h = 0;
  int topk = 100;
  bool parse_debug = false;
  int json_delay_ms = 0;
  int video_delay_ms = 0;
  std::vector<std::string> labels;
};

struct OptiViewJsonInput {
  size_t stream_idx = 0;
  std::string stream_id;
  int64_t frame_id = -1;
  int64_t capture_ms = -1;
  int64_t yolo_ms = -1;
  int output_frame_id = -1;
  const simaai::neat::Sample* yolo_sample = nullptr;
  const simaai::neat::Sample* decoded_sample = nullptr;
};

struct OptiViewJsonResult {
  bool ok = false;
  bool nonempty = false;
  int boxes = 0;
  std::string error;
};

class OptiViewOutputNodeGroup {
public:
  OptiViewOutputNodeGroup() = default;
  ~OptiViewOutputNodeGroup() = default;
  OptiViewOutputNodeGroup(const OptiViewOutputNodeGroup&) = delete;
  OptiViewOutputNodeGroup& operator=(const OptiViewOutputNodeGroup&) = delete;
  OptiViewOutputNodeGroup(OptiViewOutputNodeGroup&&) noexcept = default;
  OptiViewOutputNodeGroup& operator=(OptiViewOutputNodeGroup&&) noexcept = default;

  bool init(const OptiViewOutputNodeGroupOptions& opt, size_t streams, std::string* err = nullptr);

  bool push_video(size_t idx, const simaai::neat::Sample& sample) const;
  bool try_push_video(size_t idx, const simaai::neat::Sample& sample) const;
  bool emit_json(const OptiViewJsonInput& in, OptiViewJsonResult* out = nullptr) const;

  void stop();

  const std::vector<std::shared_ptr<simaai::neat::Run>>& video_runs() const {
    return udp_.runs();
  }

private:
  int64_t pick_timestamp_ms_(const OptiViewJsonInput& in) const;

  OptiViewOutputNodeGroupOptions opt_{};
  UdpOutputNodeGroup udp_;
  std::vector<std::unique_ptr<simaai::neat::OptiViewJsonOutput>> senders_;
};

} // namespace simaai::neat::nodes::groups
