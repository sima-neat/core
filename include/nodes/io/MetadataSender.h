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

struct MetadataSenderSendOptions {
  /**
   * Send each datagram with `MSG_DONTWAIT`.
   *
   * The default preserves the historical blocking socket behavior. Enable
   * this for real-time paths where dropping metadata during local socket
   * congestion is preferable to stalling video or inference dispatch.
   */
  bool nonblocking = false;
};

/** @brief Point-in-time diagnostics for one metadata sender. */
struct MetadataSenderStats {
  uint64_t send_attempts = 0;         ///< Calls that reached `sendto`.
  uint64_t datagrams_sent = 0;        ///< Complete UDP datagrams accepted by the kernel.
  uint64_t send_failures = 0;         ///< All failed or partial sends.
  uint64_t would_block = 0;           ///< Failures with `EAGAIN`/`EWOULDBLOCK`.
  uint64_t no_buffer_space = 0;       ///< Failures with `ENOBUFS`.
  uint64_t last_send_duration_ns = 0; ///< Duration of the most recent `sendto` call.
  uint64_t max_send_duration_ns = 0;  ///< Longest observed `sendto` call.
  int last_errno = 0;                 ///< Last failed-send errno, or zero after success.
};

class MetadataSender {
public:
  explicit MetadataSender(const MetadataSenderOptions& opt, std::string* err = nullptr);
  MetadataSender(const MetadataSenderOptions& opt, const MetadataSenderSendOptions& send_opt,
                 std::string* err = nullptr);
  ~MetadataSender();
  MetadataSender(const MetadataSender&) = delete;
  MetadataSender& operator=(const MetadataSender&) = delete;
  MetadataSender(MetadataSender&&) noexcept;
  MetadataSender& operator=(MetadataSender&&) noexcept;

  bool ok() const;
  const std::string& host() const;
  int metadata_port() const;
  bool nonblocking() const;
  MetadataSenderStats stats() const;

  bool send_raw_json(const std::string& payload, std::string* err = nullptr) const;
  bool send_metadata(const std::string& type, const std::string& data_json, int64_t timestamp_ms,
                     const std::string& frame_id, std::string* err = nullptr) const;

private:
  struct Impl;
  std::unique_ptr<Impl> impl_;
};

} // namespace simaai::neat
