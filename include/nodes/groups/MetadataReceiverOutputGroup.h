/**
 * @file
 * @ingroup nodes_groups
 * @brief Runtime helper for MetadataReceiver video + JSON metadata output.
 */
#pragma once

#include "nodes/groups/UdpOutputGroup.h"
#include "nodes/io/MetadataReceiverOutput.h"

#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>
#include <vector>

namespace simaai::neat::nodes::groups {

struct MetadataReceiverOutputNodeGroupOptions {
  UdpOutputNodeGroupOptions udp{};
  bool send_metadata = true;
  int metadata_port_base = 9100;
  int frame_w = 0;
  int frame_h = 0;
  int topk = 100;
  bool parse_debug = false;
  int metadata_delay_ms = 0;
  int video_delay_ms = 0;
  std::vector<std::string> labels;
};

struct MetadataReceiverObjectDetectionInput {
  size_t stream_idx = 0;
  std::string stream_id;
  int64_t frame_id = -1;
  int64_t capture_ms = -1;
  int64_t yolo_ms = -1;
  int output_frame_id = -1;
  const simaai::neat::Sample* yolo_sample = nullptr;
  const simaai::neat::Sample* decoded_sample = nullptr;
};

struct MetadataReceiverObjectDetectionResult {
  bool ok = false;
  bool nonempty = false;
  int boxes = 0;
  std::string error;
};

class MetadataReceiverOutputNodeGroup {
public:
  MetadataReceiverOutputNodeGroup() = default;
  ~MetadataReceiverOutputNodeGroup() = default;
  MetadataReceiverOutputNodeGroup(const MetadataReceiverOutputNodeGroup&) = delete;
  MetadataReceiverOutputNodeGroup& operator=(const MetadataReceiverOutputNodeGroup&) = delete;
  MetadataReceiverOutputNodeGroup(MetadataReceiverOutputNodeGroup&&) noexcept = default;
  MetadataReceiverOutputNodeGroup& operator=(MetadataReceiverOutputNodeGroup&&) noexcept = default;

  bool init(const MetadataReceiverOutputNodeGroupOptions& opt, size_t streams,
            std::string* err = nullptr);

  bool push_video(size_t idx, const simaai::neat::Sample& sample) const;
  bool try_push_video(size_t idx, const simaai::neat::Sample& sample) const;
  bool send_json(size_t stream_idx, const std::string& payload, std::string* err = nullptr) const;
  bool send_metadata(size_t stream_idx, const simaai::neat::MetadataReceiverPayload& payload,
                     std::string* err = nullptr) const;
  bool emit_object_detection(const MetadataReceiverObjectDetectionInput& in,
                             MetadataReceiverObjectDetectionResult* out = nullptr) const;

  void stop();

  const std::vector<std::shared_ptr<simaai::neat::Run>>& video_runs() const {
    return udp_.runs();
  }

private:
  int64_t pick_timestamp_ms_(const MetadataReceiverObjectDetectionInput& in) const;

  MetadataReceiverOutputNodeGroupOptions opt_{};
  UdpOutputNodeGroup udp_;
  std::vector<std::unique_ptr<simaai::neat::MetadataReceiverOutput>> senders_;
};

} // namespace simaai::neat::nodes::groups
