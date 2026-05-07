/**
 * @file
 * @ingroup nodes_io
 * @brief UDP sender for MetadataReceiver JSON payloads.
 */
#pragma once

#include <cstdint>
#include <memory>
#include <string>
#include <vector>

namespace simaai::neat {

struct MetadataReceiverObject {
  int x = 0;
  int y = 0;
  int w = 0;
  int h = 0;
  float score = 0.0f;
  int class_id = -1;
};

struct MetadataReceiverPayload {
  std::string type;
  std::string data_json = "{}";
  int64_t timestamp_ms = -1;
  std::string frame_id;
};

struct MetadataReceiverChannelOptions {
  std::string host = "127.0.0.1";
  int channel = 0;
  int metadata_port_base = 9100;
};

std::vector<std::string> MetadataReceiverDefaultLabels();

bool MetadataReceiverMakeJson(const MetadataReceiverPayload& payload, std::string* out_json,
                              std::string* err = nullptr);

std::string
MetadataReceiverMakeObjectDetectionJson(int64_t timestamp_ms, const std::string& frame_id,
                                        const std::vector<MetadataReceiverObject>& objects,
                                        const std::vector<std::string>& labels);

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

  bool send_json(const std::string& payload, std::string* err = nullptr) const;
  bool send_metadata(const MetadataReceiverPayload& payload, std::string* err = nullptr) const;
  bool send_object_detection(int64_t timestamp_ms, const std::string& frame_id,
                             const std::vector<MetadataReceiverObject>& objects,
                             const std::vector<std::string>& labels,
                             std::string* err = nullptr) const;

private:
  struct Impl;
  std::unique_ptr<Impl> impl_;
};

} // namespace simaai::neat
