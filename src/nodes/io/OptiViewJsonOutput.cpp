#include "nodes/io/OptiViewJsonOutput.h"

#include <nlohmann/json.hpp>

#include <algorithm>
#include <cerrno>
#include <cstring>
#include <memory>
#include <netdb.h>
#include <netinet/in.h>
#include <string>
#include <sys/socket.h>
#include <unistd.h>
#include <vector>

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

} // namespace

std::vector<std::string> OptiViewDefaultLabels() {
  std::vector<std::string> labels;
  labels.reserve(80);
  for (int i = 0; i < 80; ++i) {
    labels.push_back("label_" + std::to_string(i));
  }
  return labels;
}

std::string OptiViewMakeJson(int64_t timestamp_ms, const std::string& frame_id,
                             const std::vector<OptiViewObject>& objects,
                             const std::vector<std::string>& labels) {
  json output;
  output["type"] = "object-detection";
  output["timestamp"] = timestamp_ms;
  output["frame_id"] = frame_id;
  output["data"]["objects"] = json::array();

  for (size_t i = 0; i < objects.size(); ++i) {
    const auto& obj = objects[i];
    const std::string label =
        (obj.class_id >= 0 && static_cast<size_t>(obj.class_id) < labels.size())
            ? labels[static_cast<size_t>(obj.class_id)]
            : "Unknown";

    json item;
    item["id"] = "obj_" + std::to_string(i + 1);
    item["label"] = label;
    item["confidence"] = obj.score;
    item["bbox"] = {obj.x, obj.y, obj.w, obj.h};
    output["data"]["objects"].push_back(std::move(item));
  }

  return output.dump();
}

struct OptiViewJsonOutput::Impl {
  int fd = -1;
  sockaddr_storage addr{};
  socklen_t addr_len = 0;
  std::string host;
  int json_port = 0;
  int video_port = 0;
  bool ok = false;
};

OptiViewJsonOutput::OptiViewJsonOutput(const OptiViewChannelOptions& opt, std::string* err) {
  impl_ = std::make_unique<Impl>();
  impl_->host = opt.host.empty() ? "127.0.0.1" : opt.host;
  impl_->json_port = opt.json_port_base + opt.channel;
  impl_->video_port = opt.video_port_base + opt.channel;

  std::string addr_err;
  if (!resolve_udp_addr(impl_->host, impl_->json_port, impl_->addr, impl_->addr_len, addr_err)) {
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

OptiViewJsonOutput::~OptiViewJsonOutput() {
  if (impl_ && impl_->fd >= 0) {
    ::close(impl_->fd);
  }
}

OptiViewJsonOutput::OptiViewJsonOutput(OptiViewJsonOutput&&) noexcept = default;
OptiViewJsonOutput& OptiViewJsonOutput::operator=(OptiViewJsonOutput&&) noexcept = default;

bool OptiViewJsonOutput::ok() const {
  return impl_ && impl_->ok;
}

const std::string& OptiViewJsonOutput::host() const {
  static const std::string empty;
  return impl_ ? impl_->host : empty;
}

int OptiViewJsonOutput::json_port() const {
  return impl_ ? impl_->json_port : -1;
}

int OptiViewJsonOutput::video_port() const {
  return impl_ ? impl_->video_port : -1;
}

bool OptiViewJsonOutput::send_json(const std::string& payload, std::string* err) const {
  if (!ok()) {
    if (err)
      *err = "OptiViewJsonOutput not initialized";
    return false;
  }
  const ssize_t sent = ::sendto(impl_->fd, payload.data(), payload.size(), 0,
                                reinterpret_cast<const sockaddr*>(&impl_->addr), impl_->addr_len);
  if (sent < 0) {
    if (err)
      *err = std::string("sendto failed: ") + std::strerror(errno);
    return false;
  }
  return true;
}

bool OptiViewJsonOutput::send_detection(int64_t timestamp_ms, const std::string& frame_id,
                                        const std::vector<OptiViewObject>& objects,
                                        const std::vector<std::string>& labels,
                                        std::string* err) const {
  return send_json(OptiViewMakeJson(timestamp_ms, frame_id, objects, labels), err);
}

} // namespace simaai::neat
