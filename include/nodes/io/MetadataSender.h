/**
 * @file
 * @ingroup nodes_io
 * @brief UDP sender for JSON metadata payloads.
 */
#pragma once

#include <cstdint>
#include <memory>
#include <string>

namespace simaai::neat {

struct MetadataSenderOptions {
  std::string host = "127.0.0.1";
  int channel = 0;
  int metadata_port_base = 9100;
};

class MetadataSender {
public:
  explicit MetadataSender(const MetadataSenderOptions& opt, std::string* err = nullptr);
  ~MetadataSender();
  MetadataSender(const MetadataSender&) = delete;
  MetadataSender& operator=(const MetadataSender&) = delete;
  MetadataSender(MetadataSender&&) noexcept;
  MetadataSender& operator=(MetadataSender&&) noexcept;

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
