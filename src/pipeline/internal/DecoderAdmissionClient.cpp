#include "pipeline/internal/DecoderAdmissionClient.h"

#include <algorithm>
#include <array>
#include <atomic>
#include <cerrno>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <iomanip>
#include <random>
#include <sstream>
#include <string>
#include <sys/socket.h>
#include <sys/stat.h>
#include <sys/time.h>
#include <sys/un.h>
#include <unistd.h>
#include <vector>

namespace simaai::neat::pipeline_internal {
namespace {

constexpr const char* kDecoderAdmissionEndpoint = "/tmp/dec-admission-v2.sock";
constexpr std::uint32_t kMagic = 0x32434544u; // "DEC2" little-endian.
constexpr std::uint16_t kVersion = 1;
constexpr std::uint32_t kMaxStreams = 128;
constexpr std::uint16_t kCmdAdmitGraph = 2;
constexpr std::uint16_t kCmdAdmitGraphResp = 3;
constexpr std::uint16_t kCmdReleaseGraph = 4;
constexpr std::uint16_t kCmdCheckGraph = 6;
constexpr std::uint32_t kStatusSuccess = 501;
constexpr int kSocketIoTimeoutMs = 5000;

struct DecAdmissionV2Header {
  std::uint32_t magic;
  std::uint16_t version;
  std::uint16_t command;
  std::uint32_t payload_size;
  std::uint32_t status;
  std::uint64_t request_id;
};

struct DecAdmissionV2Stream {
  std::uint32_t stream_index;
  std::uint32_t codec;
  std::uint32_t width;
  std::uint32_t height;
  std::uint32_t fps_num;
  std::uint32_t fps_den;
  std::uint32_t stream_mode;
  std::uint32_t requested_policy;
};

struct DecAdmissionV2AdmitGraph {
  std::uint8_t run_uuid[16];
  std::uint8_t public_graph_id[16];
  std::uint8_t admission_group_uuid[16];
  std::uint32_t graph_version;
  std::uint32_t policy;
  std::uint32_t stream_count;
  std::uint32_t reserved;
};

struct DecAdmissionV2Lease {
  std::uint32_t stream_index;
  std::uint32_t resolved_output_buffers;
  std::uint32_t resolved_input_buffers;
  std::uint32_t resolved_tuning;
  std::uint64_t lease_token_hi;
  std::uint64_t lease_token_lo;
  std::uint64_t estimated_reserved_bytes;
};

struct DecAdmissionV2AdmitGraphResp {
  std::uint32_t stream_count;
  std::uint32_t reserved;
  std::uint64_t estimated_reserved_bytes;
};

struct DecAdmissionV2ReleaseGraph {
  std::uint8_t admission_group_uuid[16];
  std::uint32_t flags;
  std::uint32_t reserved;
};

static_assert(sizeof(DecAdmissionV2Header) == 24, "decoder admission v2 header ABI size changed");
static_assert(sizeof(DecAdmissionV2Stream) == 32, "decoder admission v2 stream ABI size changed");
static_assert(sizeof(DecAdmissionV2AdmitGraph) == 64,
              "decoder admission v2 graph ABI size changed");
static_assert(sizeof(DecAdmissionV2AdmitGraphResp) == 16,
              "decoder admission v2 graph response ABI size changed");
static_assert(sizeof(DecAdmissionV2Lease) == 40, "decoder admission v2 lease ABI size changed");
static_assert(sizeof(DecAdmissionV2ReleaseGraph) == 24,
              "decoder admission v2 release ABI size changed");

std::atomic<std::uint64_t> g_request_id{1};

std::string errno_message(const char* what, int err) {
  std::ostringstream oss;
  oss << what << ": " << std::strerror(err) << " (errno=" << err << ")";
  return oss.str();
}

std::string timeout_message(const char* what) {
  std::ostringstream oss;
  oss << what << " timed out after " << kSocketIoTimeoutMs << " ms";
  return oss.str();
}

bool set_socket_io_timeouts(int fd, std::string* error) {
  timeval tv{};
  tv.tv_sec = kSocketIoTimeoutMs / 1000;
  tv.tv_usec = (kSocketIoTimeoutMs % 1000) * 1000;
  if (::setsockopt(fd, SOL_SOCKET, SO_SNDTIMEO, &tv, sizeof(tv)) != 0) {
    if (error) {
      *error = errno_message("set decoder admission send timeout", errno);
    }
    return false;
  }
  if (::setsockopt(fd, SOL_SOCKET, SO_RCVTIMEO, &tv, sizeof(tv)) != 0) {
    if (error) {
      *error = errno_message("set decoder admission receive timeout", errno);
    }
    return false;
  }
  return true;
}

std::array<std::uint8_t, 16> random_uuid() {
  std::array<std::uint8_t, 16> out{};
  std::random_device rd;
  for (auto& byte : out) {
    byte = static_cast<std::uint8_t>(rd() & 0xffU);
  }
  // Mark it as a UUID-like opaque id for easier diagnostics. The protocol only
  // requires 16 stable bytes; the exact version bits are not semantically used.
  out[6] = static_cast<std::uint8_t>((out[6] & 0x0fU) | 0x40U);
  out[8] = static_cast<std::uint8_t>((out[8] & 0x3fU) | 0x80U);
  return out;
}

int connect_socket(std::string* error, bool* endpoint_missing) {
  if (endpoint_missing) {
    *endpoint_missing = false;
  }

  int fd = ::socket(AF_UNIX, SOCK_SEQPACKET | SOCK_CLOEXEC, 0);
  if (fd < 0) {
    if (error) {
      *error = errno_message("socket", errno);
    }
    return -1;
  }

  sockaddr_un addr{};
  addr.sun_family = AF_UNIX;
  if (std::strlen(kDecoderAdmissionEndpoint) >= sizeof(addr.sun_path)) {
    ::close(fd);
    if (error) {
      *error = "decoder admission socket path is too long";
    }
    return -1;
  }
  std::strncpy(addr.sun_path, kDecoderAdmissionEndpoint, sizeof(addr.sun_path) - 1);

  if (::connect(fd, reinterpret_cast<sockaddr*>(&addr), sizeof(addr)) != 0) {
    const int err = errno;
    ::close(fd);
    if (endpoint_missing && (err == ENOENT || err == ECONNREFUSED)) {
      *endpoint_missing = true;
    }
    if (error) {
      *error = "connect " + std::string(kDecoderAdmissionEndpoint) + ": " + std::strerror(err) +
               " (errno=" + std::to_string(err) + ")";
    }
    return -1;
  }
  if (!set_socket_io_timeouts(fd, error)) {
    ::close(fd);
    return -1;
  }
  return fd;
}

std::string status_payload_message(const DecAdmissionV2Header& header,
                                   const std::vector<std::uint8_t>& payload) {
  std::ostringstream oss;
  oss << decoder_admission_status_name(header.status);
  if (!payload.empty()) {
    const char* text = reinterpret_cast<const char*>(payload.data());
    const auto len = strnlen(text, payload.size());
    if (len > 0) {
      oss << ": " << std::string(text, len);
    }
  }
  return oss.str();
}

bool send_packet(int fd, const std::vector<std::uint8_t>& packet, std::string* error) {
  const ssize_t n = ::send(fd, packet.data(), packet.size(), MSG_NOSIGNAL);
  if (n < 0) {
    if (error) {
      *error = (errno == EAGAIN || errno == EWOULDBLOCK)
                   ? timeout_message("send decoder admission request")
                   : errno_message("send decoder admission request", errno);
    }
    return false;
  }
  if (static_cast<std::size_t>(n) != packet.size()) {
    if (error) {
      *error = "send decoder admission request: short packet write";
    }
    return false;
  }
  return true;
}

bool recv_packet(int fd, std::vector<std::uint8_t>& packet, std::string* error) {
  packet.assign(1024U * 1024U, 0);
  const ssize_t n = ::recv(fd, packet.data(), packet.size(), 0);
  if (n < 0) {
    if (errno == EINTR) {
      return recv_packet(fd, packet, error);
    }
    if (error) {
      *error = (errno == EAGAIN || errno == EWOULDBLOCK)
                   ? timeout_message("read decoder admission response")
                   : errno_message("read decoder admission response", errno);
    }
    return false;
  }
  if (n == 0) {
    if (error) {
      *error = "read decoder admission response: connection closed before full response";
    }
    return false;
  }
  packet.resize(static_cast<std::size_t>(n));
  return true;
}

DecoderAdmissionResult send_graph_request(const std::vector<DecoderAdmissionStreamRequest>& streams,
                                          bool dry_run) {
  DecoderAdmissionResult result;
  result.group_uuid = random_uuid();

  if (streams.empty()) {
    result.error = "decoder admission requires at least one stream";
    return result;
  }
  if (streams.size() > kMaxStreams) {
    result.error = "decoder admission stream count exceeds protocol maximum";
    return result;
  }

  std::string err;
  bool endpoint_missing = false;
  const int fd = connect_socket(&err, &endpoint_missing);
  if (fd < 0) {
    result.endpoint_missing = endpoint_missing;
    result.error = err;
    return result;
  }

  DecAdmissionV2AdmitGraph graph{};
  const auto run_uuid = random_uuid();
  const auto public_graph_id = random_uuid();
  std::copy(run_uuid.begin(), run_uuid.end(), graph.run_uuid);
  std::copy(public_graph_id.begin(), public_graph_id.end(), graph.public_graph_id);
  std::copy(result.group_uuid.begin(), result.group_uuid.end(), graph.admission_group_uuid);
  graph.graph_version = 1;
  graph.policy = 0;
  graph.stream_count = static_cast<std::uint32_t>(streams.size());

  std::vector<DecAdmissionV2Stream> wire_streams;
  wire_streams.reserve(streams.size());
  for (const auto& stream : streams) {
    DecAdmissionV2Stream wire{};
    wire.stream_index = stream.stream_index;
    wire.codec = stream.codec;
    wire.width = stream.width;
    wire.height = stream.height;
    wire.fps_num = stream.fps_num;
    wire.fps_den = stream.fps_den == 0 ? 1 : stream.fps_den;
    wire.stream_mode = stream.stream_mode;
    wire.requested_policy = stream.requested_policy;
    wire_streams.push_back(wire);
  }

  const std::size_t payload_size =
      sizeof(graph) + wire_streams.size() * sizeof(DecAdmissionV2Stream);
  std::vector<std::uint8_t> payload(payload_size);
  std::memcpy(payload.data(), &graph, sizeof(graph));
  std::memcpy(payload.data() + sizeof(graph), wire_streams.data(),
              wire_streams.size() * sizeof(DecAdmissionV2Stream));

  DecAdmissionV2Header req{};
  req.magic = kMagic;
  req.version = kVersion;
  req.command = dry_run ? kCmdCheckGraph : kCmdAdmitGraph;
  req.payload_size = static_cast<std::uint32_t>(payload.size());
  req.status = 0;
  req.request_id = g_request_id.fetch_add(1, std::memory_order_relaxed);

  std::vector<std::uint8_t> packet(sizeof(req) + payload.size());
  std::memcpy(packet.data(), &req, sizeof(req));
  std::memcpy(packet.data() + sizeof(req), payload.data(), payload.size());
  if (!send_packet(fd, packet, &err)) {
    ::close(fd);
    result.error = err;
    return result;
  }

  std::vector<std::uint8_t> resp_packet;
  if (!recv_packet(fd, resp_packet, &err)) {
    ::close(fd);
    result.error = err;
    return result;
  }
  if (resp_packet.size() < sizeof(DecAdmissionV2Header)) {
    ::close(fd);
    result.error = "decoder admission response missing header";
    return result;
  }
  DecAdmissionV2Header resp{};
  std::memcpy(&resp, resp_packet.data(), sizeof(resp));
  if (resp.magic != kMagic || resp.version != kVersion || resp.command != kCmdAdmitGraphResp ||
      resp.request_id != req.request_id) {
    ::close(fd);
    result.error = "decoder admission response header mismatch";
    return result;
  }
  if (resp.payload_size > 1024U * 1024U) {
    ::close(fd);
    result.error = "decoder admission response payload is too large";
    return result;
  }
  if (resp.payload_size != resp_packet.size() - sizeof(resp)) {
    ::close(fd);
    result.error = "decoder admission response payload size mismatch";
    return result;
  }

  std::vector<std::uint8_t> resp_payload(resp_packet.begin() + sizeof(resp), resp_packet.end());
  ::close(fd);

  if (resp.status != kStatusSuccess) {
    result.error = status_payload_message(resp, resp_payload);
    return result;
  }
  if (resp_payload.size() < sizeof(DecAdmissionV2AdmitGraphResp)) {
    result.error = "decoder admission success response missing graph payload";
    return result;
  }

  DecAdmissionV2AdmitGraphResp graph_resp{};
  std::memcpy(&graph_resp, resp_payload.data(), sizeof(graph_resp));
  const std::size_t lease_bytes = resp_payload.size() - sizeof(graph_resp);
  if (lease_bytes !=
      static_cast<std::size_t>(graph_resp.stream_count) * sizeof(DecAdmissionV2Lease)) {
    result.error = "decoder admission response lease count does not match payload size";
    return result;
  }
  if (graph_resp.stream_count != streams.size()) {
    result.error = "decoder admission response stream count does not match request";
    return result;
  }

  result.leases.reserve(graph_resp.stream_count);
  result.estimated_reserved_bytes = graph_resp.estimated_reserved_bytes;
  const auto* lease_base = resp_payload.data() + sizeof(graph_resp);
  for (std::uint32_t i = 0; i < graph_resp.stream_count; ++i) {
    DecAdmissionV2Lease wire{};
    std::memcpy(&wire, lease_base + static_cast<std::size_t>(i) * sizeof(wire), sizeof(wire));
    DecoderAdmissionLease lease;
    lease.stream_index = wire.stream_index;
    lease.resolved_output_buffers = wire.resolved_output_buffers;
    lease.resolved_input_buffers = wire.resolved_input_buffers;
    lease.resolved_tuning = wire.resolved_tuning;
    lease.lease_token_hi = wire.lease_token_hi;
    lease.lease_token_lo = wire.lease_token_lo;
    lease.estimated_reserved_bytes = wire.estimated_reserved_bytes;
    result.leases.push_back(lease);
  }
  result.admitted = true;
  return result;
}

} // namespace

bool decoder_admission_endpoint_available() {
  struct stat st {};
  return ::stat(kDecoderAdmissionEndpoint, &st) == 0 && S_ISSOCK(st.st_mode);
}

DecoderAdmissionResult
admit_decoder_graph(const std::vector<DecoderAdmissionStreamRequest>& streams, bool dry_run) {
  return send_graph_request(streams, dry_run);
}

bool release_decoder_graph(const std::array<std::uint8_t, 16>& group_uuid, std::string* error) {
  std::string err;
  bool endpoint_missing = false;
  const int fd = connect_socket(&err, &endpoint_missing);
  if (fd < 0) {
    if (error) {
      *error = err;
    }
    return false;
  }

  DecAdmissionV2ReleaseGraph release{};
  std::copy(group_uuid.begin(), group_uuid.end(), release.admission_group_uuid);

  DecAdmissionV2Header req{};
  req.magic = kMagic;
  req.version = kVersion;
  req.command = kCmdReleaseGraph;
  req.payload_size = sizeof(release);
  req.status = 0;
  req.request_id = g_request_id.fetch_add(1, std::memory_order_relaxed);

  std::vector<std::uint8_t> packet(sizeof(req) + sizeof(release));
  std::memcpy(packet.data(), &req, sizeof(req));
  std::memcpy(packet.data() + sizeof(req), &release, sizeof(release));
  if (!send_packet(fd, packet, &err)) {
    ::close(fd);
    if (error) {
      *error = err;
    }
    return false;
  }

  std::vector<std::uint8_t> resp_packet;
  if (!recv_packet(fd, resp_packet, &err)) {
    ::close(fd);
    if (error) {
      *error = err;
    }
    return false;
  }
  if (resp_packet.size() < sizeof(DecAdmissionV2Header)) {
    ::close(fd);
    if (error) {
      *error = "decoder admission release response missing header";
    }
    return false;
  }
  DecAdmissionV2Header resp{};
  std::memcpy(&resp, resp_packet.data(), sizeof(resp));
  ::close(fd);

  if (resp.magic != kMagic || resp.version != kVersion || resp.command != kCmdReleaseGraph ||
      resp.request_id != req.request_id || resp.status != kStatusSuccess) {
    if (error) {
      std::ostringstream oss;
      oss << "decoder admission release failed: " << decoder_admission_status_name(resp.status);
      *error = oss.str();
    }
    return false;
  }
  return true;
}

std::string decoder_admission_uuid_to_string(const std::array<std::uint8_t, 16>& uuid) {
  std::ostringstream oss;
  oss << std::hex << std::setfill('0');
  for (std::size_t i = 0; i < uuid.size(); ++i) {
    if (i == 4 || i == 6 || i == 8 || i == 10) {
      oss << '-';
    }
    oss << std::setw(2) << static_cast<unsigned>(uuid[i]);
  }
  return oss.str();
}

const char* decoder_admission_tuning_name(std::uint32_t tuning) {
  switch (tuning) {
  case 0:
    return "default";
  case 1:
    return "low-memory";
  case 2:
    return "throughput-low-latency";
  case 3:
    return "auto";
  default:
    return "auto";
  }
}

const char* decoder_admission_status_name(std::uint32_t status) {
  switch (status) {
  case 500:
    return "invalid request";
  case 501:
    return "success";
  case 502:
    return "insufficient decoder memory";
  case 503:
    return "stream-count limit exceeded";
  case 504:
    return "slice-count limit exceeded";
  case 505:
    return "no decoder channel available";
  case 506:
    return "decoder processing capacity unavailable";
  case 507:
    return "unsupported or incomplete stream settings";
  case 508:
    return "decoder frame processing failed";
  case 509:
    return "decoder internal failure";
  default:
    return "unknown decoder admission status";
  }
}

} // namespace simaai::neat::pipeline_internal
