#include "peripherals/PeripheralCatalog.h"

#include "peripherals/internal/PeripheralClient.h"
#include "peripherals/internal/ProtocolContract.h"
#include "pipeline/ErrorCodes.h"
#include "pipeline/GraphReport.h"
#include "pipeline/NeatError.h"

#include <nlohmann/json.hpp>

#include <poll.h>
#include <sys/socket.h>
#include <sys/un.h>
#include <unistd.h>

#include <algorithm>
#include <cerrno>
#include <charconv>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <limits>
#include <optional>
#include <set>
#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>

namespace simaai::neat::peripherals_internal {
namespace {

using Clock = std::chrono::steady_clock;
using Deadline = Clock::time_point;

constexpr std::size_t kMaximumHeaderBytes = 8192;

class FileDescriptor {
public:
  explicit FileDescriptor(int value = -1) : value_(value) {}
  ~FileDescriptor() {
    if (value_ >= 0)
      ::close(value_);
  }

  FileDescriptor(const FileDescriptor&) = delete;
  FileDescriptor& operator=(const FileDescriptor&) = delete;
  FileDescriptor(FileDescriptor&& other) noexcept : value_(std::exchange(other.value_, -1)) {}
  FileDescriptor& operator=(FileDescriptor&&) = delete;

  int get() const noexcept {
    return value_;
  }

private:
  int value_;
};

[[noreturn]] void fail(const char* code, std::string reason) {
  GraphReport report;
  report.error_code = code;
  report.repro_note = std::move(reason);
  throw NeatError("[" + report.error_code + "] " + report.repro_note, std::move(report));
}

[[noreturn]] void fail_parse(std::string reason) {
  fail(error_codes::kIoParse, "Sentinel returned an invalid v1 peripheral catalog: " + reason +
                                  ". Restart simaai-sentinel.service; if the error persists, "
                                  "update Sentinel with `sima-cli neat install sentinel`.");
}

[[noreturn]] void fail_timeout() {
  fail(error_codes::kPeripheralDaemonTimeout,
       "Timed out waiting for the Sentinel peripheral catalog. Check "
       "simaai-sentinel.service and its journal, then try again.");
}

[[noreturn]] void fail_connect(int error) {
  if (error == EACCES || error == EPERM) {
    fail(error_codes::kPermissionDenied,
         "Permission was denied opening the Sentinel API socket. Check the permissions on "
         "/run/simaai-sentinel and /run/simaai-sentinel/api.sock.");
  }
  if (error == ENOENT || error == ECONNREFUSED) {
    fail(error_codes::kPeripheralDaemonUnavailable,
         "The peripheral catalog is unavailable because SiMa Sentinel is not running. Install "
         "Sentinel with `sima-cli neat install sentinel`, or start simaai-sentinel.service if "
         "it is already installed, then try again.");
  }
  fail(error_codes::kIoOpen,
       "Could not connect to the Sentinel API socket: " + std::string(std::strerror(error)) +
           ". Check simaai-sentinel.service and its journal.");
}

int remaining_milliseconds(Deadline deadline) {
  const auto now = Clock::now();
  if (now >= deadline)
    fail_timeout();
  const auto remaining = std::chrono::ceil<std::chrono::milliseconds>(deadline - now).count();
  return static_cast<int>(std::min<std::int64_t>(remaining, std::numeric_limits<int>::max()));
}

short wait_for(int fd, short events, Deadline deadline) {
  for (;;) {
    pollfd descriptor{fd, events, 0};
    const int result = ::poll(&descriptor, 1, remaining_milliseconds(deadline));
    if (result > 0)
      return descriptor.revents;
    if (result == 0)
      fail_timeout();
    if (errno != EINTR)
      fail(error_codes::kIoOpen,
           "Sentinel API socket polling failed: " + std::string(std::strerror(errno)) +
               ". Check simaai-sentinel.service and its journal.");
  }
}

FileDescriptor connect_socket(const std::string& path, Deadline deadline) {
  sockaddr_un address{};
  if (path.empty() || path.size() >= sizeof(address.sun_path))
    fail(error_codes::kIoOpen, "The Sentinel API socket path is invalid.");

  FileDescriptor socket(::socket(AF_UNIX, SOCK_STREAM | SOCK_CLOEXEC | SOCK_NONBLOCK, 0));
  if (socket.get() < 0)
    fail_connect(errno);

  address.sun_family = AF_UNIX;
  std::memcpy(address.sun_path, path.c_str(), path.size() + 1);
  if (::connect(socket.get(), reinterpret_cast<sockaddr*>(&address), sizeof(address)) == 0)
    return socket;
  if (errno != EINPROGRESS && errno != EAGAIN && errno != EWOULDBLOCK)
    fail_connect(errno);

  (void)wait_for(socket.get(), POLLOUT, deadline);
  int socket_error = 0;
  socklen_t length = sizeof(socket_error);
  if (::getsockopt(socket.get(), SOL_SOCKET, SO_ERROR, &socket_error, &length) < 0)
    fail_connect(errno);
  if (socket_error != 0)
    fail_connect(socket_error);
  return socket;
}

void send_all(int fd, std::string_view data, Deadline deadline, std::size_t maximum_send_bytes) {
  while (!data.empty()) {
    const short events = wait_for(fd, POLLOUT, deadline);
    if (!(events & POLLOUT))
      fail(error_codes::kPeripheralDaemonUnavailable,
           "Sentinel closed the connection before accepting the peripheral catalog request. "
           "Check simaai-sentinel.service and its journal.");
    const ssize_t sent =
        ::send(fd, data.data(), std::min(data.size(), maximum_send_bytes), MSG_NOSIGNAL);
    if (sent > 0) {
      data.remove_prefix(static_cast<std::size_t>(sent));
      continue;
    }
    if (sent < 0 && (errno == EINTR || errno == EAGAIN || errno == EWOULDBLOCK))
      continue;
    fail_connect(sent < 0 ? errno : ECONNRESET);
  }
}

std::string lowercase_ascii(std::string value) {
  std::transform(value.begin(), value.end(), value.begin(), [](unsigned char character) {
    if (character >= 'A' && character <= 'Z')
      return static_cast<char>(character - 'A' + 'a');
    return static_cast<char>(character);
  });
  return value;
}

std::string_view trim_ascii(std::string_view value) {
  while (!value.empty() && (value.front() == ' ' || value.front() == '\t'))
    value.remove_prefix(1);
  while (!value.empty() && (value.back() == ' ' || value.back() == '\t'))
    value.remove_suffix(1);
  return value;
}

std::size_t parse_content_length(std::string_view value) {
  value = trim_ascii(value);
  std::size_t result = 0;
  const auto parsed = std::from_chars(value.data(), value.data() + value.size(), result);
  if (value.empty() || parsed.ec != std::errc{} || parsed.ptr != value.data() + value.size())
    fail_parse("Content-Length is not a non-negative decimal integer");
  if (result > kMaximumResponseBytes) {
    fail(error_codes::kResponseTooLarge,
         "The Sentinel peripheral catalog response exceeded the 4 MiB v1 limit. Reduce the "
         "catalog size or update Sentinel and Core together.");
  }
  return result;
}

struct ResponseHead {
  int status = 0;
  std::size_t content_length = 0;
};

ResponseHead parse_response_head(std::string_view value) {
  const std::size_t status_end = value.find("\r\n");
  if (status_end == std::string_view::npos)
    fail_parse("the HTTP status line is incomplete");
  const std::string_view status_line = value.substr(0, status_end);
  constexpr std::string_view prefix = "HTTP/1.1 ";
  if (!status_line.starts_with(prefix) || status_line.size() < prefix.size() + 4 ||
      status_line[prefix.size() + 3] != ' ')
    fail_parse("the HTTP status line is invalid");
  int status = 0;
  const auto status_result = std::from_chars(status_line.data() + prefix.size(),
                                             status_line.data() + prefix.size() + 3, status);
  if (status_result.ec != std::errc{} ||
      status_result.ptr != status_line.data() + prefix.size() + 3)
    fail_parse("the HTTP status code is invalid");

  std::optional<std::size_t> content_length;
  std::size_t position = status_end + 2;
  while (position < value.size()) {
    const std::size_t line_end = value.find("\r\n", position);
    if (line_end == std::string_view::npos)
      fail_parse("the HTTP headers are incomplete");
    const std::string_view line = value.substr(position, line_end - position);
    if (line.empty())
      break;
    const std::size_t separator = line.find(':');
    if (separator == std::string_view::npos)
      fail_parse("an HTTP header has no separator");
    const std::string name = lowercase_ascii(std::string(trim_ascii(line.substr(0, separator))));
    const std::string_view header_value = trim_ascii(line.substr(separator + 1));
    if (name == "content-length") {
      if (content_length)
        fail_parse("the HTTP response contains duplicate Content-Length headers");
      content_length = parse_content_length(header_value);
    } else if (name == "transfer-encoding") {
      fail_parse("chunked HTTP responses are not part of the local v1 protocol");
    }
    position = line_end + 2;
  }
  if (!content_length)
    fail_parse("the HTTP response has no Content-Length header");
  return {status, *content_length};
}

std::pair<int, std::string> read_response(int fd, Deadline deadline) {
  std::string response;
  response.reserve(4096);
  std::optional<std::size_t> header_end;
  std::optional<ResponseHead> head;

  for (;;) {
    if (header_end && head) {
      const std::size_t body_size = response.size() - *header_end;
      if (body_size == head->content_length)
        return {head->status, response.substr(*header_end)};
      if (body_size > head->content_length)
        fail_parse("the HTTP body is longer than Content-Length");
    }

    const short events = wait_for(fd, POLLIN, deadline);
    if (!(events & (POLLIN | POLLHUP)))
      fail_parse("the socket failed before the catalog response completed");
    char buffer[8192];
    const ssize_t received = ::recv(fd, buffer, sizeof(buffer), 0);
    if (received > 0) {
      response.append(buffer, static_cast<std::size_t>(received));
      if (!header_end) {
        const std::size_t marker = response.find("\r\n\r\n");
        if (marker == std::string::npos) {
          if (response.size() > kMaximumHeaderBytes)
            fail_parse("the HTTP headers exceed 8 KiB");
        } else {
          if (marker + 4 > kMaximumHeaderBytes)
            fail_parse("the HTTP headers exceed 8 KiB");
          header_end = marker + 4;
          head = parse_response_head(std::string_view(response).substr(0, *header_end));
        }
      }
      continue;
    }
    if (received < 0 && (errno == EINTR || errno == EAGAIN || errno == EWOULDBLOCK))
      continue;
    if (received < 0)
      fail_parse("socket read failed: " + std::string(std::strerror(errno)));
    fail_parse("the socket closed before the catalog response completed");
  }
}

const nlohmann::json& require_field(const nlohmann::json& object, const char* name) {
  const auto found = object.find(name);
  if (found == object.end())
    fail_parse("missing required field '" + std::string(name) + "'");
  return *found;
}

const nlohmann::json& require_object(const nlohmann::json& value, std::string_view context) {
  if (!value.is_object())
    fail_parse(std::string(context) + " must be an object");
  return value;
}

std::string require_string(const nlohmann::json& object, const char* name,
                           bool allow_empty = false) {
  const auto& value = require_field(object, name);
  if (!value.is_string())
    fail_parse("field '" + std::string(name) + "' must be a string");
  std::string result = value.get<std::string>();
  if (!allow_empty && result.empty())
    fail_parse("field '" + std::string(name) + "' must not be empty");
  return result;
}

bool require_bool(const nlohmann::json& object, const char* name) {
  const auto& value = require_field(object, name);
  if (!value.is_boolean())
    fail_parse("field '" + std::string(name) + "' must be a boolean");
  return value.get<bool>();
}

std::uint64_t require_u64(const nlohmann::json& object, const char* name) {
  const auto& value = require_field(object, name);
  if (!value.is_number_unsigned())
    fail_parse("field '" + std::string(name) + "' must be an unsigned integer");
  return value.get<std::uint64_t>();
}

std::uint32_t require_u32(const nlohmann::json& object, const char* name, bool positive = true) {
  const std::uint64_t value = require_u64(object, name);
  if (value > std::numeric_limits<std::uint32_t>::max() || (positive && value == 0))
    fail_parse("field '" + std::string(name) + "' is outside its supported range");
  return static_cast<std::uint32_t>(value);
}

std::optional<std::string> require_optional_string(const nlohmann::json& object, const char* name,
                                                   bool allow_empty = false) {
  const auto& value = require_field(object, name);
  if (value.is_null())
    return std::nullopt;
  if (!value.is_string())
    fail_parse("field '" + std::string(name) + "' must be a string or null");
  std::string result = value.get<std::string>();
  if (!allow_empty && result.empty())
    fail_parse("field '" + std::string(name) + "' must not be empty");
  return result;
}

std::optional<std::string> optional_string(const nlohmann::json& object, const char* name,
                                           bool allow_empty = false) {
  const auto found = object.find(name);
  if (found == object.end() || found->is_null())
    return std::nullopt;
  if (!found->is_string())
    fail_parse("field '" + std::string(name) + "' must be a string when present");
  std::string result = found->get<std::string>();
  if (!allow_empty && result.empty())
    fail_parse("field '" + std::string(name) + "' must not be empty");
  return result;
}

peripherals::CatalogError parse_error(const nlohmann::json& value) {
  const auto& object = require_object(value, "catalog error");
  return {require_string(object, "code"), require_string(object, "reason")};
}

peripherals::ProviderIssue parse_issue(const nlohmann::json& value) {
  const auto& object = require_object(value, "provider issue");
  return {
      .provider = require_string(object, "provider"),
      .code = require_string(object, "code"),
      .reason = require_string(object, "reason"),
      .retained_last_good = require_bool(object, "retained_last_good"),
  };
}

peripherals::CameraSizeRange parse_size_range(const nlohmann::json& value) {
  const auto& object = require_object(value, "camera size_range");
  peripherals::CameraSizeRange range{
      .min_width = require_u32(object, "min_width"),
      .min_height = require_u32(object, "min_height"),
      .max_width = require_u32(object, "max_width"),
      .max_height = require_u32(object, "max_height"),
      .step_width = require_u32(object, "step_width"),
      .step_height = require_u32(object, "step_height"),
  };
  if (range.max_width < range.min_width || range.max_height < range.min_height)
    fail_parse("camera size_range maxima must not be smaller than minima");
  return range;
}

peripherals::CameraMode parse_mode(const nlohmann::json& value) {
  const auto& object = require_object(value, "camera mode");
  peripherals::CameraMode mode;
  mode.format = require_string(object, "format");
  mode.framerate_num = require_u32(object, "framerate_num");
  mode.framerate_den = require_u32(object, "framerate_den");
  mode.supported = require_bool(object, "supported");
  mode.reason = require_string(object, "reason", true);

  const bool has_width = object.contains("width");
  const bool has_height = object.contains("height");
  const bool has_range = object.contains("size_range");
  if (has_range == (has_width || has_height) || has_width != has_height)
    fail_parse("a camera mode must contain either width/height or size_range");
  if (has_range) {
    mode.size_range = parse_size_range(require_field(object, "size_range"));
  } else {
    mode.width = require_u32(object, "width");
    mode.height = require_u32(object, "height");
  }
  if (!mode.supported && mode.reason.empty())
    fail_parse("an unsupported camera mode must include a reason");
  return mode;
}

peripherals::CameraDetails parse_camera(const nlohmann::json& value) {
  const auto& object = require_object(value, "camera details");
  peripherals::CameraDetails camera;
  camera.camera_name = optional_string(object, "camera_name");
  camera.model = optional_string(object, "model");
  camera.backend = require_string(object, "backend");
  const auto& modes = require_field(object, "modes");
  if (!modes.is_array())
    fail_parse("camera field 'modes' must be an array");
  camera.modes.reserve(modes.size());
  for (const auto& mode : modes)
    camera.modes.push_back(parse_mode(mode));
  return camera;
}

peripherals::Peripheral parse_device(const nlohmann::json& value) {
  const auto& object = require_object(value, "peripheral record");
  peripherals::Peripheral device{
      .id = require_string(object, "id"),
      .type = require_string(object, "type"),
      .provider = require_string(object, "provider"),
      .camera = std::nullopt,
  };
  // Camera details are part of the typed v1 contract, so invalid ones fail the
  // whole read. Core cannot judge other types' details; a non-object value
  // leaves that one device with "{}" instead of hiding every other device.
  if (device.type == "camera")
    device.camera = parse_camera(require_field(object, "camera"));
  const auto details = object.find(device.type);
  if (details != object.end() && details->is_object())
    device.details_json = details->dump();
  return device;
}

peripherals::Catalog parse_catalog(const std::string& body) {
  nlohmann::json root;
  try {
    root = nlohmann::json::parse(body);
  } catch (const nlohmann::json::exception& error) {
    fail_parse("JSON parsing failed: " + std::string(error.what()));
  }
  const auto& object = require_object(root, "catalog response");
  const std::uint64_t schema_version = require_u64(object, "schema_version");
  if (schema_version != kPeripheralSchemaVersion) {
    fail(error_codes::kRuntimeAbiMismatch,
         "Peripheral catalog schema " + std::to_string(schema_version) +
             " is incompatible with this Core build, which requires schema " +
             std::to_string(kPeripheralSchemaVersion) +
             ". Install matching Core and Sentinel versions; update Sentinel with "
             "`sima-cli neat install sentinel`.");
  }

  peripherals::Catalog catalog;
  catalog.instance_id = require_string(object, "instance_id");
  catalog.state = require_string(object, "state");
  if (catalog.state != "starting" && catalog.state != "ready" && catalog.state != "degraded")
    fail_parse("field 'state' contains an unknown v1 value");
  const bool ready = require_bool(object, "ready");
  catalog.stale = require_bool(object, "stale");
  catalog.revision = require_u64(object, "revision");
  catalog.sequence = require_u64(object, "sequence");
  catalog.scan_sequence = require_u64(object, "scan_sequence");
  catalog.last_success_at = require_optional_string(object, "last_success_at");
  catalog.last_attempt_at = require_optional_string(object, "last_attempt_at");

  const auto& error = require_field(object, "error");
  if (!error.is_null())
    catalog.error = parse_error(error);

  const auto& issues = require_field(object, "issues");
  if (!issues.is_array())
    fail_parse("field 'issues' must be an array");
  std::set<std::string> issue_providers;
  catalog.issues.reserve(issues.size());
  for (const auto& value : issues) {
    auto issue = parse_issue(value);
    if (!issue_providers.insert(issue.provider).second)
      fail_parse("provider issues contain a duplicate provider");
    catalog.issues.push_back(std::move(issue));
  }

  const auto& devices = require_field(object, "devices");
  if (!devices.is_array())
    fail_parse("field 'devices' must be an array");
  std::set<std::string> device_ids;
  catalog.devices.reserve(devices.size());
  for (const auto& value : devices) {
    auto device = parse_device(value);
    if (!device_ids.insert(device.id).second)
      fail_parse("peripheral records contain a duplicate id");
    catalog.devices.push_back(std::move(device));
  }

  if (catalog.state == "ready" && !ready)
    fail_parse("state 'ready' contradicts ready=false");
  if (catalog.state == "starting" && ready)
    fail_parse("state 'starting' contradicts ready=true");
  if (!ready) {
    std::string reason = "Sentinel has not produced an initial peripheral catalog";
    if (catalog.error)
      reason += ": " + catalog.error->reason;
    else if (!catalog.issues.empty())
      reason += ": " + catalog.issues.front().provider + ": " + catalog.issues.front().reason;
    reason += ". Check simaai-sentinel.service and its journal, then retry.";
    fail(error_codes::kPeripheralDaemonNotReady, std::move(reason));
  }
  return catalog;
}

// Sentinel answers every failed request with `{"error": "<message>"}`.
// Returns that message, or an empty string when the body has no such field.
std::string error_from_body(const std::string& body) {
  try {
    const auto value = nlohmann::json::parse(body);
    if (value.is_object()) {
      const auto found = value.find("error");
      if (found != value.end() && found->is_string())
        return found->get<std::string>();
    }
  } catch (const nlohmann::json::exception&) {
  }
  return {};
}

} // namespace

peripherals::Catalog list_from_socket(const std::string& socket_path,
                                      std::chrono::milliseconds timeout,
                                      std::size_t maximum_send_bytes) {
  if (timeout.count() <= 0)
    fail_timeout();
  if (maximum_send_bytes == 0)
    fail(error_codes::kIoOpen, "The peripheral catalog request chunk limit is invalid.");
  const Deadline deadline = Clock::now() + timeout;
  FileDescriptor socket = connect_socket(socket_path, deadline);
  const std::string request = "GET " + std::string(kCatalogPath) +
                              " HTTP/1.1\r\nHost: localhost\r\nConnection: close\r\n\r\n";
  send_all(socket.get(), request, deadline, maximum_send_bytes);
  auto [status, body] = read_response(socket.get(), deadline);
  if (status != 200) {
    const std::string sentinel_error = error_from_body(body);
    if (status == 500 && sentinel_error == "response_too_large") {
      fail(error_codes::kResponseTooLarge,
           "The Sentinel peripheral catalog exceeded the 4 MiB v1 limit. Reduce the catalog "
           "size or update Sentinel and Core together.");
    }
    const std::string reported =
        sentinel_error.empty() ? std::string() : " Sentinel reported: " + sentinel_error + ".";
    if (status == 404) {
      fail(error_codes::kPeripheralDaemonUnavailable,
           "The installed SiMa Sentinel is too old to serve the peripheral catalog (HTTP 404 "
           "for " +
               std::string(kCatalogPath) + ")." + reported +
               " Update Sentinel with `sima-cli neat install sentinel`, then try again.");
    }
    const char* code =
        status == 503 ? error_codes::kPeripheralDaemonUnavailable : error_codes::kIoParse;
    fail(code, "Sentinel returned unexpected HTTP status " + std::to_string(status) +
                   " for the peripheral catalog." + reported +
                   " Update Sentinel with `sima-cli neat install sentinel` and check "
                   "simaai-sentinel.service.");
  }
  return parse_catalog(body);
}

} // namespace simaai::neat::peripherals_internal

namespace simaai::neat::peripherals {

Catalog list() {
  return peripherals_internal::list_from_socket(peripherals_internal::kPeripheralSocketPath,
                                                std::chrono::seconds(5));
}

} // namespace simaai::neat::peripherals
