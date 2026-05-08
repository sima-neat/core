#include "nodes/groups/MetadataReceiverOutputGroup.h"

#include <exception>
#include <memory>
#include <string>
#include <utility>

namespace simaai::neat::nodes::groups {

bool MetadataReceiverOutputGroup::init(const MetadataReceiverOutputGroupOptions& opt,
                                       size_t streams, std::string* err) {
  senders_.clear();

  if (streams == 0) {
    if (err)
      *err = "MetadataReceiverOutputGroup: streams must be > 0";
    return false;
  }

  try {
    senders_.reserve(streams);
    for (size_t i = 0; i < streams; ++i) {
      simaai::neat::MetadataReceiverChannelOptions sender_opt;
      sender_opt.host = opt.host;
      sender_opt.channel = static_cast<int>(i);
      sender_opt.metadata_port_base = opt.metadata_port_base;

      std::string sender_err;
      auto sender = std::make_unique<simaai::neat::MetadataReceiverOutput>(sender_opt, &sender_err);
      if (!sender->ok()) {
        if (err)
          *err = "MetadataReceiverOutputGroup: sender init failed: " + sender_err;
        stop();
        return false;
      }
      senders_.push_back(std::move(sender));
    }
  } catch (const std::exception& ex) {
    stop();
    if (err)
      *err = ex.what();
    return false;
  }

  return true;
}

int MetadataReceiverOutputGroup::metadata_port(size_t stream_idx) const {
  if (stream_idx >= senders_.size() || !senders_[stream_idx])
    return -1;
  return senders_[stream_idx]->metadata_port();
}

bool MetadataReceiverOutputGroup::send_raw_json(size_t stream_idx, const std::string& payload,
                                                std::string* err) const {
  if (stream_idx >= senders_.size() || !senders_[stream_idx]) {
    if (err)
      *err = "invalid stream index for metadata sender";
    return false;
  }
  return senders_[stream_idx]->send_raw_json(payload, err);
}

bool MetadataReceiverOutputGroup::send_metadata(size_t stream_idx, const std::string& type,
                                                const std::string& data_json, int64_t timestamp_ms,
                                                const std::string& frame_id,
                                                std::string* err) const {
  if (stream_idx >= senders_.size() || !senders_[stream_idx]) {
    if (err)
      *err = "invalid stream index for metadata sender";
    return false;
  }
  return senders_[stream_idx]->send_metadata(type, data_json, timestamp_ms, frame_id, err);
}

void MetadataReceiverOutputGroup::stop() {
  senders_.clear();
}

} // namespace simaai::neat::nodes::groups
