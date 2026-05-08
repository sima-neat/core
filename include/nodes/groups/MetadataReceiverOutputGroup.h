/**
 * @file
 * @ingroup nodes_groups
 * @brief Runtime helper for MetadataReceiver JSON metadata fan-out.
 */
#pragma once

#include "nodes/io/MetadataReceiverOutput.h"

#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>
#include <vector>

namespace simaai::neat::nodes::groups {

struct MetadataReceiverOutputGroupOptions {
  std::string host = "127.0.0.1";
  int metadata_port_base = 9100;
};

class MetadataReceiverOutputGroup {
public:
  MetadataReceiverOutputGroup() = default;
  ~MetadataReceiverOutputGroup() = default;
  MetadataReceiverOutputGroup(const MetadataReceiverOutputGroup&) = delete;
  MetadataReceiverOutputGroup& operator=(const MetadataReceiverOutputGroup&) = delete;
  MetadataReceiverOutputGroup(MetadataReceiverOutputGroup&&) noexcept = default;
  MetadataReceiverOutputGroup& operator=(MetadataReceiverOutputGroup&&) noexcept = default;

  bool init(const MetadataReceiverOutputGroupOptions& opt, size_t streams,
            std::string* err = nullptr);

  size_t size() const {
    return senders_.size();
  }
  int metadata_port(size_t stream_idx) const;

  bool send_raw_json(size_t stream_idx, const std::string& payload,
                     std::string* err = nullptr) const;
  bool send_metadata(size_t stream_idx, const std::string& type, const std::string& data_json,
                     int64_t timestamp_ms, const std::string& frame_id,
                     std::string* err = nullptr) const;
  void stop();

private:
  std::vector<std::unique_ptr<simaai::neat::MetadataReceiverOutput>> senders_;
};

} // namespace simaai::neat::nodes::groups
