#pragma once

#include <cstdint>
#include <cerrno>
#include <netinet/in.h>
#include <sys/socket.h>
#include <unistd.h>

inline bool rtsp_port_is_available_tcp(int port) {
  if (port <= 0 || port > 65535)
    return false;
  int fd = ::socket(AF_INET, SOCK_STREAM, 0);
  if (fd < 0)
    return false;
  int opt = 1;
  (void)::setsockopt(fd, SOL_SOCKET, SO_REUSEADDR, &opt, sizeof(opt));
#ifdef SO_REUSEPORT
  (void)::setsockopt(fd, SOL_SOCKET, SO_REUSEPORT, &opt, sizeof(opt));
#endif
  sockaddr_in addr{};
  addr.sin_family = AF_INET;
  addr.sin_port = htons(static_cast<uint16_t>(port));
  addr.sin_addr.s_addr = htonl(INADDR_ANY);
  const bool ok = (::bind(fd, reinterpret_cast<sockaddr*>(&addr), sizeof(addr)) == 0);
  ::close(fd);
  return ok;
}

inline bool rtsp_port_is_available_udp(int port) {
  if (port <= 0 || port > 65535)
    return false;
  int fd = ::socket(AF_INET, SOCK_DGRAM, 0);
  if (fd < 0)
    return false;
  sockaddr_in addr{};
  addr.sin_family = AF_INET;
  addr.sin_port = htons(static_cast<uint16_t>(port));
  addr.sin_addr.s_addr = htonl(INADDR_ANY);
  const bool ok = (::bind(fd, reinterpret_cast<sockaddr*>(&addr), sizeof(addr)) == 0);
  ::close(fd);
  return ok;
}

inline bool rtsp_port_is_available(int port) {
  return rtsp_port_is_available_tcp(port) && rtsp_port_is_available_udp(port);
}

inline bool rtsp_port_range_available(int base_port, int ports_needed, int stride) {
  if (ports_needed <= 0 || stride <= 0)
    return false;
  const int max_port = base_port + (ports_needed - 1) * stride;
  if (base_port <= 0 || max_port > 65535)
    return false;
  for (int i = 0; i < ports_needed; ++i) {
    const int port = base_port + i * stride;
    if (!rtsp_port_is_available(port))
      return false;
  }
  return true;
}

inline bool rtsp_udp_port_range_available(int base_port, int ports_needed, int stride) {
  if (ports_needed <= 0 || stride <= 0)
    return false;
  const int max_port = base_port + (ports_needed - 1) * stride;
  if (base_port <= 0 || max_port > 65535)
    return false;
  for (int i = 0; i < ports_needed; ++i) {
    const int port = base_port + i * stride;
    if (!rtsp_port_is_available_udp(port))
      return false;
  }
  return true;
}

inline int rtsp_find_free_port_range(int base_port, int ports_needed, int stride, int max_tries) {
  if (max_tries <= 0)
    return -1;
  for (int i = 0; i < max_tries; ++i) {
    const int candidate = base_port + i;
    if (rtsp_port_range_available(candidate, ports_needed, stride)) {
      return candidate;
    }
  }
  return -1;
}

inline int rtsp_find_free_port_range_with_rtp(int base_port, int rtsp_ports_needed, int rtsp_stride,
                                              int max_tries, int rtp_port_offset,
                                              int rtp_ports_per_server, int rtp_port_stride) {
  if (max_tries <= 0)
    return -1;
  if (rtsp_ports_needed <= 0 || rtsp_stride <= 0)
    return -1;
  if (rtp_ports_per_server <= 0 || rtp_port_stride <= 0)
    return -1;
  for (int i = 0; i < max_tries; ++i) {
    const int candidate = base_port + i;
    if (!rtsp_port_range_available(candidate, rtsp_ports_needed, rtsp_stride)) {
      continue;
    }
    bool udp_ok = true;
    for (int s = 0; s < rtsp_ports_needed; ++s) {
      const int rtp_base = candidate + rtp_port_offset + s * rtp_port_stride;
      if (!rtsp_udp_port_range_available(rtp_base, rtp_ports_per_server, 1)) {
        udp_ok = false;
        break;
      }
    }
    if (udp_ok)
      return candidate;
  }
  return -1;
}
