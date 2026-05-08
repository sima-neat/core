/**
 * @file
 * @ingroup nodes_io
 * @brief UDP sender for MetadataReceiver JSON payloads.
 */
#pragma once

#include <cstdint>
#include <memory>
#include <string>

namespace simaai::neat {

struct MetadataReceiverChannelOptions {
  std::string host = "127.0.0.1";
  int channel = 0;
  int metadata_port_base = 9100;
};

class MetadataReceiverOutput {
public:
  explicit MetadataReceiverOutput(const MetadataReceiverChannelOptions& opt,
                                  std::string* err = nullptr);
  ~MetadataReceiverOutput();
  MetadataReceiverOutput(const MetadataReceiverOutput&) = delete;
  MetadataReceiverOutput& operator=(const MetadataReceiverOutput&) = delete;
  MetadataReceiverOutput(MetadataReceiverOutput&&) noexcept;
  MetadataReceiverOutput& operator=(MetadataReceiverOutput&&) noexcept;

  bool ok() const;
  const std::string& host() const;
  int metadata_port() const;

  bool send_raw_json(const std::string& payload, std::string* err = nullptr) const;
  bool send_metadata(const std::string& type, const std::string& data_json, int64_t timestamp_ms,
                     const std::string& frame_id, std::string* err = nullptr) const;

private:
  struct Impl;
  std::unique_ptr<Impl> impl_;
};

} // namespace simaai::neat
