#include "nodes/io/MetadataSender.h"

#include <nlohmann/json.hpp>

#include <atomic>
#include <cerrno>
#include <chrono>
#include <cstring>
#include <exception>
#include <memory>
#include <netdb.h>
#include <netinet/in.h>
#include <string>
#include <sys/socket.h>
#include <unistd.h>

namespace simaai::neat {
namespace {

using json = nlohmann::json;

void update_max(std::atomic<uint64_t>& maximum, uint64_t value) {
  uint64_t current = maximum.load(std::memory_order_relaxed);
  while (current < value &&
         !maximum.compare_exchange_weak(current, value, std::memory_order_relaxed,
                                        std::memory_order_relaxed)) {
  }
}

bool resolve_udp_addr(const std::string& host, int port, sockaddr_storage& out, socklen_t& out_len,
                      std::string& err) {
  addrinfo hints{};
  hints.ai_family = AF_UNSPEC;
  hints.ai_socktype = SOCK_DGRAM;
  hints.ai_protocol = IPPROTO_UDP;

  addrinfo* result = nullptr;
  const std::string port_str = std::to_string(port);
  const int rc = ::getaddrinfo(host.c_str(), port_str.c_str(), &hints, &result);
  if (rc != 0 || !result) {
    err = "getaddrinfo failed for " + host + ":" + port_str + " (" + gai_strerror(rc) + ")";
    return false;
  }

  bool ok = false;
  for (addrinfo* ai = result; ai != nullptr; ai = ai->ai_next) {
    if (!ai->ai_addr || ai->ai_addrlen == 0)
      continue;
    std::memset(&out, 0, sizeof(out));
    std::memcpy(&out, ai->ai_addr, ai->ai_addrlen);
    out_len = static_cast<socklen_t>(ai->ai_addrlen);
    ok = true;
    break;
  }
  ::freeaddrinfo(result);

  if (!ok) {
    err = "failed to resolve UDP address for " + host + ":" + port_str;
    return false;
  }
  return true;
}

bool make_metadata_json(const std::string& type, const std::string& data_json, int64_t timestamp_ms,
                        const std::string& frame_id, std::string* out_json, std::string* err) {
  if (!out_json) {
    if (err)
      *err = "MetadataSender send_metadata requires out_json";
    return false;
  }
  out_json->clear();

  if (type.empty()) {
    if (err)
      *err = "MetadataSender metadata type must not be empty";
    return false;
  }

  json data;
  try {
    data = json::parse(data_json);
  } catch (const std::exception& ex) {
    if (err)
      *err = std::string("MetadataSender data_json parse failed: ") + ex.what();
    return false;
  }

  if (!data.is_object()) {
    if (err)
      *err = "MetadataSender data_json must be a JSON object";
    return false;
  }

  json output;
  output["type"] = type;
  output["data"] = std::move(data);
  if (timestamp_ms >= 0) {
    output["timestamp"] = timestamp_ms;
  }
  if (!frame_id.empty()) {
    output["frame_id"] = frame_id;
  }

  *out_json = output.dump();
  return true;
}

} // namespace

struct MetadataSender::Impl {
  ~Impl() {
    if (fd >= 0) {
      ::close(fd);
    }
  }

  int fd = -1;
  sockaddr_storage addr{};
  socklen_t addr_len = 0;
  std::string host;
  int metadata_port = 0;
  bool nonblocking = false;
  bool ok = false;
  std::atomic<uint64_t> send_attempts{0};
  std::atomic<uint64_t> datagrams_sent{0};
  std::atomic<uint64_t> send_failures{0};
  std::atomic<uint64_t> would_block{0};
  std::atomic<uint64_t> no_buffer_space{0};
  std::atomic<uint64_t> last_send_duration_ns{0};
  std::atomic<uint64_t> max_send_duration_ns{0};
  std::atomic<int> last_errno{0};
};

MetadataSender::MetadataSender(const MetadataSenderOptions& opt, std::string* err)
    : MetadataSender(opt, MetadataSenderSendOptions{}, err) {}

MetadataSender::MetadataSender(const MetadataSenderOptions& opt,
                               const MetadataSenderSendOptions& send_opt, std::string* err) {
  impl_ = std::make_unique<Impl>();
  impl_->host = opt.host.empty() ? "127.0.0.1" : opt.host;
  impl_->metadata_port = opt.metadata_port_base + opt.channel;
  impl_->nonblocking = send_opt.nonblocking;

  std::string addr_err;
  if (!resolve_udp_addr(impl_->host, impl_->metadata_port, impl_->addr, impl_->addr_len,
                        addr_err)) {
    if (err)
      *err = addr_err;
    return;
  }

  const int fd = ::socket(impl_->addr.ss_family, SOCK_DGRAM, 0);
  if (fd < 0) {
    if (err)
      *err = std::string("socket failed: ") + std::strerror(errno);
    return;
  }

  impl_->fd = fd;
  impl_->ok = true;
}

MetadataSender::~MetadataSender() = default;
MetadataSender::MetadataSender(MetadataSender&&) noexcept = default;
MetadataSender& MetadataSender::operator=(MetadataSender&&) noexcept = default;

bool MetadataSender::ok() const {
  return impl_ && impl_->ok;
}

const std::string& MetadataSender::host() const {
  static const std::string empty;
  return impl_ ? impl_->host : empty;
}

int MetadataSender::metadata_port() const {
  return impl_ ? impl_->metadata_port : -1;
}

bool MetadataSender::nonblocking() const {
  return impl_ && impl_->nonblocking;
}

MetadataSenderStats MetadataSender::stats() const {
  MetadataSenderStats out;
  if (!impl_) {
    return out;
  }
  out.send_attempts = impl_->send_attempts.load(std::memory_order_relaxed);
  out.datagrams_sent = impl_->datagrams_sent.load(std::memory_order_relaxed);
  out.send_failures = impl_->send_failures.load(std::memory_order_relaxed);
  out.would_block = impl_->would_block.load(std::memory_order_relaxed);
  out.no_buffer_space = impl_->no_buffer_space.load(std::memory_order_relaxed);
  out.last_send_duration_ns = impl_->last_send_duration_ns.load(std::memory_order_relaxed);
  out.max_send_duration_ns = impl_->max_send_duration_ns.load(std::memory_order_relaxed);
  out.last_errno = impl_->last_errno.load(std::memory_order_relaxed);
  return out;
}

bool MetadataSender::send_raw_json(const std::string& payload, std::string* err) const {
  if (!ok()) {
    if (err)
      *err = "MetadataSender not initialized";
    return false;
  }
  impl_->send_attempts.fetch_add(1, std::memory_order_relaxed);
  const auto send_start = std::chrono::steady_clock::now();
  const int flags = impl_->nonblocking ? MSG_DONTWAIT : 0;
  const ssize_t sent = ::sendto(impl_->fd, payload.data(), payload.size(), flags,
                                reinterpret_cast<const sockaddr*>(&impl_->addr), impl_->addr_len);
  const int send_errno = sent < 0 ? errno : 0;
  const auto duration_ns =
      static_cast<uint64_t>(std::chrono::duration_cast<std::chrono::nanoseconds>(
                                std::chrono::steady_clock::now() - send_start)
                                .count());
  impl_->last_send_duration_ns.store(duration_ns, std::memory_order_relaxed);
  update_max(impl_->max_send_duration_ns, duration_ns);
  if (sent < 0) {
    impl_->send_failures.fetch_add(1, std::memory_order_relaxed);
    impl_->last_errno.store(send_errno, std::memory_order_relaxed);
    if (send_errno == EAGAIN || send_errno == EWOULDBLOCK) {
      impl_->would_block.fetch_add(1, std::memory_order_relaxed);
    } else if (send_errno == ENOBUFS) {
      impl_->no_buffer_space.fetch_add(1, std::memory_order_relaxed);
    }
    if (err)
      *err = std::string("sendto failed: ") + std::strerror(send_errno);
    return false;
  }
  if (static_cast<size_t>(sent) != payload.size()) {
    impl_->send_failures.fetch_add(1, std::memory_order_relaxed);
    impl_->last_errno.store(EIO, std::memory_order_relaxed);
    if (err)
      *err = "sendto sent a partial datagram";
    return false;
  }
  impl_->datagrams_sent.fetch_add(1, std::memory_order_relaxed);
  impl_->last_errno.store(0, std::memory_order_relaxed);
  return true;
}

bool MetadataSender::send_metadata(const std::string& type, const std::string& data_json,
                                   int64_t timestamp_ms, const std::string& frame_id,
                                   std::string* err) const {
  std::string payload_json;
  if (!make_metadata_json(type, data_json, timestamp_ms, frame_id, &payload_json, err))
    return false;
  return send_raw_json(payload_json, err);
}

} // namespace simaai::neat
