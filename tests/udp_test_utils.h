#pragma once

#include "test_utils.h"

#include <arpa/inet.h>
#include <netinet/in.h>
#include <sys/select.h>
#include <sys/socket.h>
#include <unistd.h>

#include <array>
#include <cerrno>
#include <cstring>
#include <stdexcept>
#include <string>
#include <vector>

namespace sima_test {

class UdpReceiver {
public:
  explicit UdpReceiver(int port = 0, const std::string& bind_host = "127.0.0.1") {
    fd_ = ::socket(AF_INET, SOCK_DGRAM, 0);
    if (fd_ < 0) {
      throw std::runtime_error("UdpReceiver socket() failed");
    }

    int reuse = 1;
    (void)::setsockopt(fd_, SOL_SOCKET, SO_REUSEADDR, &reuse, sizeof(reuse));

    sockaddr_in addr{};
    addr.sin_family = AF_INET;
    addr.sin_port = htons(static_cast<uint16_t>(port));

    if (::inet_pton(AF_INET, bind_host.c_str(), &addr.sin_addr) != 1) {
      ::close(fd_);
      throw std::runtime_error("UdpReceiver invalid bind_host: " + bind_host);
    }

    if (::bind(fd_, reinterpret_cast<const sockaddr*>(&addr), sizeof(addr)) < 0) {
      const std::string why = std::strerror(errno);
      ::close(fd_);
      throw std::runtime_error("UdpReceiver bind() failed: " + why);
    }

    socklen_t len = sizeof(addr);
    if (::getsockname(fd_, reinterpret_cast<sockaddr*>(&addr), &len) < 0) {
      const std::string why = std::strerror(errno);
      ::close(fd_);
      throw std::runtime_error("UdpReceiver getsockname() failed: " + why);
    }

    port_ = static_cast<int>(ntohs(addr.sin_port));
  }

  UdpReceiver(const UdpReceiver&) = delete;
  UdpReceiver& operator=(const UdpReceiver&) = delete;

  UdpReceiver(UdpReceiver&& other) noexcept {
    *this = std::move(other);
  }

  UdpReceiver& operator=(UdpReceiver&& other) noexcept {
    if (this != &other) {
      close_();
      fd_ = other.fd_;
      port_ = other.port_;
      other.fd_ = -1;
      other.port_ = -1;
    }
    return *this;
  }

  ~UdpReceiver() {
    close_();
  }

  int port() const {
    return port_;
  }

  bool recv_one(std::string* payload, int timeout_ms = 2000) const {
    if (!payload || fd_ < 0)
      return false;
    payload->clear();

    fd_set rfds;
    FD_ZERO(&rfds);
    FD_SET(fd_, &rfds);

    timeval tv{};
    tv.tv_sec = timeout_ms / 1000;
    tv.tv_usec = (timeout_ms % 1000) * 1000;

    const int ready = ::select(fd_ + 1, &rfds, nullptr, nullptr, &tv);
    if (ready <= 0)
      return false;

    std::array<char, 65536> buf{};
    const ssize_t n = ::recvfrom(fd_, buf.data(), buf.size(), 0, nullptr, nullptr);
    if (n <= 0)
      return false;

    payload->assign(buf.data(), static_cast<size_t>(n));
    return true;
  }

  int drain(std::vector<std::string>* payloads, int max_packets, int timeout_ms_each = 20) const {
    if (!payloads || max_packets <= 0)
      return 0;
    int received = 0;
    for (int i = 0; i < max_packets; ++i) {
      std::string payload;
      if (!recv_one(&payload, timeout_ms_each))
        break;
      payloads->push_back(std::move(payload));
      ++received;
    }
    return received;
  }

private:
  void close_() {
    if (fd_ >= 0) {
      ::close(fd_);
      fd_ = -1;
    }
  }

  int fd_ = -1;
  int port_ = -1;
};

inline bool likely_runtime_missing(const std::string& err) {
  return is_dispatcher_unavailable(err) || err.find("no element") != std::string::npos ||
         err.find("No such element") != std::string::npos ||
         err.find("not found") != std::string::npos || err.find("failed to") != std::string::npos;
}

} // namespace sima_test
