#include "nodes/io/MetadataSender.h"

#include <nlohmann/json.hpp>

#include <cerrno>
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
  bool ok = false;
};

MetadataSender::MetadataSender(const MetadataSenderOptions& opt, std::string* err) {
  impl_ = std::make_unique<Impl>();
  impl_->host = opt.host.empty() ? "127.0.0.1" : opt.host;
  impl_->metadata_port = opt.metadata_port_base + opt.channel;

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

bool MetadataSender::send_raw_json(const std::string& payload, std::string* err) const {
  if (!ok()) {
    if (err)
      *err = "MetadataSender not initialized";
    return false;
  }
  const ssize_t sent = ::sendto(impl_->fd, payload.data(), payload.size(), 0,
                                reinterpret_cast<const sockaddr*>(&impl_->addr), impl_->addr_len);
  if (sent < 0) {
    if (err)
      *err = std::string("sendto failed: ") + std::strerror(errno);
    return false;
  }
  if (static_cast<size_t>(sent) != payload.size()) {
    if (err)
      *err = "sendto sent a partial datagram";
    return false;
  }
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
