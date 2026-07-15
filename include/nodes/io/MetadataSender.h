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
   * This is enabled by default so local socket congestion cannot stall video
   * or inference dispatch. Set it to `false` only when the caller explicitly
   * prefers blocking delivery attempts over real-time progress.
   * `EAGAIN`/`EWOULDBLOCK` and `ENOBUFS` then return `false` without an error
   * string; inspect `MetadataSenderStats` for drop diagnostics.
   */
  bool nonblocking = true;
};

/**
 * @brief Point-in-time diagnostics for one metadata sender.
 *
 * `MetadataSender::stats()` may be called while other threads are sending. The
 * returned fields are individually coherent counters, but the structure is a
 * concurrent diagnostic snapshot rather than a transactional snapshot of one
 * instant.
 */
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
  /// Return a concurrent diagnostic snapshot; fields need not represent one transaction.
  MetadataSenderStats stats() const;

  bool send_raw_json(const std::string& payload, std::string* err = nullptr) const;
  bool send_metadata(const std::string& type, const std::string& data_json, int64_t timestamp_ms,
                     const std::string& frame_id, std::string* err = nullptr) const;

private:
  struct Impl;
  std::unique_ptr<Impl> impl_;
};

} // namespace simaai::neat
