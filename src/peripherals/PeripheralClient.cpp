#include "peripherals/PeripheralCatalog.h"

#include "nodes/io/CameraInput.h"
#include "peripherals/internal/PeripheralClient.h"
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
#include <string>
#include <string_view>
#include <utility>

namespace simaai::neat::peripherals_internal {
namespace {

using Clock = std::chrono::steady_clock;
using Deadline = Clock::time_point;

constexpr const char* kCatalogPath = "/v1/peripherals";
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
                                  ". Install matching Sentinel and Core versions; update "
                                  "Sentinel with `sima-cli neat install sentinel`.");
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
  // A nonblocking AF_UNIX connect completes at once or fails; EAGAIN means the
  // listen backlog is full and the socket is not connected, so retry.
  while (::connect(socket.get(), reinterpret_cast<sockaddr*>(&address), sizeof(address)) != 0) {
    if (errno != EAGAIN && errno != EINTR)
      fail_connect(errno);
    (void)::poll(nullptr, 0, std::min(remaining_milliseconds(deadline), 10));
  }
  return socket;
}

void send_all(int fd, std::string_view data, Deadline deadline) {
  while (!data.empty()) {
    const short events = wait_for(fd, POLLOUT, deadline);
    if (!(events & POLLOUT))
      fail(error_codes::kPeripheralDaemonUnavailable,
           "Sentinel closed the connection before accepting the peripheral catalog request. "
           "Check simaai-sentinel.service and its journal.");
    const ssize_t sent = ::send(fd, data.data(), data.size(), MSG_NOSIGNAL);
    if (sent > 0) {
      data.remove_prefix(static_cast<std::size_t>(sent));
      continue;
    }
    if (sent < 0 && (errno == EINTR || errno == EAGAIN || errno == EWOULDBLOCK))
      continue;
    fail(error_codes::kIoOpen, "Could not send the peripheral catalog request to Sentinel: " +
                                   std::string(std::strerror(sent < 0 ? errno : ECONNRESET)) +
                                   ". Check simaai-sentinel.service and its journal.");
  }
}

std::string_view trim_ascii(std::string_view value) {
  while (!value.empty() && (value.front() == ' ' || value.front() == '\t'))
    value.remove_prefix(1);
  while (!value.empty() && (value.back() == ' ' || value.back() == '\t'))
    value.remove_suffix(1);
  return value;
}

template <typename Integer> bool parse_decimal(std::string_view text, Integer& value) {
  const auto result = std::from_chars(text.data(), text.data() + text.size(), value);
  return !text.empty() && result.ec == std::errc{} && result.ptr == text.data() + text.size();
}

// ASCII-only, so the result does not depend on the process locale.
bool equals_ignoring_case(std::string_view value, std::string_view lowercase) {
  return std::equal(value.begin(), value.end(), lowercase.begin(), lowercase.end(),
                    [](char a, char b) { return (a >= 'A' && a <= 'Z' ? a - 'A' + 'a' : a) == b; });
}

struct ResponseHead {
  int status = 0;
  std::size_t header_bytes = 0;
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
  if (!parse_decimal(status_line.substr(prefix.size(), 3), status))
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
    const std::string_view name = trim_ascii(line.substr(0, separator));
    if (equals_ignoring_case(name, "content-length")) {
      if (content_length)
        fail_parse("the HTTP response contains duplicate Content-Length headers");
      std::size_t length = 0;
      if (!parse_decimal(trim_ascii(line.substr(separator + 1)), length))
        fail_parse("Content-Length is not a non-negative decimal integer");
      if (length > kMaximumResponseBytes)
        fail(error_codes::kResponseTooLarge,
             "The Sentinel peripheral catalog response exceeded the 4 MiB v1 limit. Reduce the "
             "catalog size or update Sentinel and Core together.");
      content_length = length;
    } else if (equals_ignoring_case(name, "transfer-encoding")) {
      fail_parse("chunked HTTP responses are not part of the local v1 protocol");
    }
    position = line_end + 2;
  }
  if (!content_length)
    fail_parse("the HTTP response has no Content-Length header");
  return {status, value.size(), *content_length};
}

std::pair<int, std::string> read_response(int fd, Deadline deadline) {
  std::string response;
  std::optional<ResponseHead> head;
  for (;;) {
    if (head) {
      const std::size_t body_size = response.size() - head->header_bytes;
      if (body_size == head->content_length)
        return {head->status, response.substr(head->header_bytes)};
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
      if (head)
        continue;
      const std::size_t marker = response.find("\r\n\r\n");
      if ((marker == std::string::npos ? response.size() : marker + 4) > kMaximumHeaderBytes)
        fail_parse("the HTTP headers exceed 8 KiB");
      if (marker != std::string::npos)
        head = parse_response_head(std::string_view(response).substr(0, marker + 4));
      continue;
    }
    if (received < 0 && (errno == EINTR || errno == EAGAIN || errno == EWOULDBLOCK))
      continue;
    if (received < 0)
      fail_parse("socket read failed: " + std::string(std::strerror(errno)));
    fail_parse("the socket closed before the catalog response completed");
  }
}

using JsonTypeCheck = bool (nlohmann::json::*)() const noexcept;

const nlohmann::json& require_field(const nlohmann::json& object, const char* name,
                                    JsonTypeCheck has_type = nullptr,
                                    const char* type_name = nullptr) {
  const auto found = object.find(name);
  if (found == object.end())
    fail_parse("missing required field '" + std::string(name) + "'");
  if (has_type && !((*found).*has_type)())
    fail_parse("field '" + std::string(name) + "' must be " + type_name);
  return *found;
}

const nlohmann::json& require_object(const nlohmann::json& value, std::string_view context) {
  if (!value.is_object())
    fail_parse(std::string(context) + " must be an object");
  return value;
}

std::string require_string(const nlohmann::json& object, const char* name) {
  auto value =
      require_field(object, name, &nlohmann::json::is_string, "a string").get<std::string>();
  if (value.empty())
    fail_parse("field '" + std::string(name) + "' must not be empty");
  return value;
}

std::uint64_t require_u64(const nlohmann::json& object, const char* name) {
  return require_field(object, name, &nlohmann::json::is_number_unsigned, "an unsigned integer")
      .get<std::uint64_t>();
}

std::uint32_t require_u32(const nlohmann::json& object, const char* name) {
  const std::uint64_t value = require_u64(object, name);
  if (value == 0 || value > std::numeric_limits<std::uint32_t>::max())
    fail_parse("field '" + std::string(name) + "' is outside its supported range");
  return static_cast<std::uint32_t>(value);
}

const nlohmann::json& require_array(const nlohmann::json& object, const char* name) {
  return require_field(object, name, &nlohmann::json::is_array, "an array");
}

// A nullable string that must be present when `required`; never empty when set.
std::optional<std::string> nullable_string(const nlohmann::json& object, const char* name,
                                           bool required) {
  const auto found = object.find(name);
  if (found == object.end() && required)
    fail_parse("missing required field '" + std::string(name) + "'");
  if (found == object.end() || found->is_null())
    return std::nullopt;
  if (!found->is_string())
    fail_parse("field '" + std::string(name) +
               (required ? "' must be a string or null" : "' must be a string when present"));
  if (found->get_ref<const std::string&>().empty())
    fail_parse("field '" + std::string(name) + "' must not be empty");
  return found->get<std::string>();
}

peripherals::CatalogError parse_error(const nlohmann::json& value) {
  const auto& object = require_object(value, "catalog error");
  return {require_string(object, "provider"), require_string(object, "code"),
          require_string(object, "reason")};
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

// A frame interval in seconds per frame.
struct Interval {
  std::uint64_t numerator = 0;
  std::uint64_t denominator = 1;
};

bool shorter(Interval a, Interval b) {
  return a.numerator * b.denominator < b.numerator * a.denominator;
}

Interval parse_fraction(const nlohmann::json& value) {
  const auto& object = require_object(value, "frame interval");
  return {require_u32(object, "numerator"), require_u32(object, "denominator")};
}

// The shortest and longest interval one V4L2 frame-interval entry covers, or
// nothing for an unknown type or an inverted range, which the caller skips.
std::optional<std::pair<Interval, Interval>> parse_interval(const nlohmann::json& value) {
  const auto& object = require_object(value, "frame interval");
  const std::string type = require_string(object, "type");
  if (type == "discrete") {
    const Interval interval = parse_fraction(object);
    return std::pair{interval, interval};
  }
  if (type != "stepwise" && type != "continuous")
    return std::nullopt;
  const Interval minimum = parse_fraction(require_field(object, "minimum"));
  const Interval maximum = parse_fraction(require_field(object, "maximum"));
  if (shorter(maximum, minimum))
    return std::nullopt;
  return std::pair{minimum, maximum};
}

// Why CameraInput's default libcamera profile rejects a mode, or "" when it
// accepts it. Rules are checked in order; the first failure is the reason.
std::string rejection(const std::string& backend, const std::string& format, bool isp_output,
                      bool lists_intervals, bool covers_default_rate) {
  const CameraInputOptions defaults;
  if (backend != "mipi")
    return "CameraInput's default libcamera profile accepts MIPI cameras only. These rules do "
           "not classify raw V4L2 profiles such as MetoakSimor (RAW8 1920x360, selected with "
           "CameraInputOptions.profile and device).";
  if (format != defaults.format)
    return "CameraInput's default libcamera profile supports " + defaults.format + " output only.";
  // A mode without intervals is not rejected: CameraInput sets the rate through caps.
  if (lists_intervals && !covers_default_rate)
    return "This mode does not advertise the " + std::to_string(defaults.framerate_num) + "/" +
           std::to_string(defaults.framerate_den) +
           " frame rate of CameraInput's default libcamera profile.";
  if (!isp_output)
    return "CameraInput's default libcamera profile requires an ISP output size; this resolution "
           "is not one on this board.";
  return {};
}

peripherals::CameraMode parse_mode(const nlohmann::json& value, const std::string& backend) {
  const auto& object = require_object(value, "camera mode");
  peripherals::CameraMode mode;
  mode.format = require_string(object, "format");

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

  const auto isp_output = object.find("isp_output");
  if (isp_output != object.end() && !isp_output->is_boolean())
    fail_parse("field 'isp_output' must be a boolean when present");

  const CameraInputOptions defaults;
  const Interval default_interval{defaults.framerate_den, defaults.framerate_num};
  std::optional<Interval> fastest;
  bool lists_intervals = false;
  bool covers_default_rate = false;
  if (object.contains("frame_intervals")) {
    for (const auto& size : require_array(object, "frame_intervals")) {
      for (const auto& entry :
           require_array(require_object(size, "frame interval size"), "intervals")) {
        lists_intervals = true;
        const auto interval = parse_interval(entry);
        if (!interval)
          continue;
        const auto [shortest, longest] = *interval;
        covers_default_rate = covers_default_rate || (!shorter(default_interval, shortest) &&
                                                      !shorter(longest, default_interval));
        if (!fastest || shorter(shortest, *fastest))
          fastest = shortest;
      }
    }
  }
  if (fastest) {
    mode.framerate_num = static_cast<std::uint32_t>(fastest->denominator);
    mode.framerate_den = static_cast<std::uint32_t>(fastest->numerator);
  }

  mode.reason =
      rejection(backend, mode.format, isp_output != object.end() && isp_output->get<bool>(),
                lists_intervals, covers_default_rate);
  mode.supported = mode.reason.empty();
  return mode;
}

peripherals::CameraDetails parse_camera(const nlohmann::json& object) {
  peripherals::CameraDetails camera;
  camera.camera_name = nullable_string(object, "camera_name", false);
  camera.model = nullable_string(object, "model", false);
  camera.backend = require_string(object, "backend");
  const auto& modes = require_array(object, "modes");
  camera.modes.reserve(modes.size());
  for (const auto& mode : modes)
    camera.modes.push_back(parse_mode(mode, camera.backend));
  return camera;
}

peripherals::Peripheral parse_device(const nlohmann::json& value) {
  const auto& object = require_object(value, "peripheral record");
  peripherals::Peripheral device{
      .id = require_string(object, "id"),
      .type = require_string(object, "type"),
      .camera = std::nullopt,
      .details_json = object.dump(),
  };
  // A camera record Core cannot read leaves `camera` unset for that device only;
  // its id, type and full record stay available.
  if (device.type == "camera") {
    try {
      device.camera = parse_camera(object);
    } catch (const NeatError&) {
    }
  }
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

  peripherals::Catalog catalog;
  catalog.revision = require_u64(object, "revision");
  catalog.observed_at = nullable_string(object, "observed_at", true);

  const auto& errors = require_array(object, "errors");
  catalog.errors.reserve(errors.size());
  for (const auto& value : errors)
    catalog.errors.push_back(parse_error(value));

  const auto& devices = require_array(object, "devices");
  std::set<std::string> device_ids;
  catalog.devices.reserve(devices.size());
  for (const auto& value : devices) {
    auto device = parse_device(value);
    if (!device_ids.insert(device.id).second)
      fail_parse("peripheral records contain a duplicate id");
    catalog.devices.push_back(std::move(device));
  }
  return catalog;
}

// Sentinel answers failed requests with `{"error": "<message>"}`.
std::string error_from_body(const std::string& body) {
  const auto value = nlohmann::json::parse(body, nullptr, false);
  if (value.is_object() && value.contains("error") && value["error"].is_string())
    return value["error"].get<std::string>();
  return {};
}

} // namespace

peripherals::Catalog list_from_socket(const std::string& socket_path,
                                      std::chrono::milliseconds timeout) {
  if (timeout.count() <= 0)
    fail_timeout();
  const Deadline deadline = Clock::now() + timeout;
  FileDescriptor socket = connect_socket(socket_path, deadline);
  const std::string request = "GET " + std::string(kCatalogPath) +
                              " HTTP/1.1\r\nHost: localhost\r\nConnection: close\r\n\r\n";
  send_all(socket.get(), request, deadline);
  auto [status, body] = read_response(socket.get(), deadline);
  if (status != 200) {
    const std::string sentinel_error = error_from_body(body);
    const std::string reported =
        sentinel_error.empty() ? std::string() : " Sentinel reported: " + sentinel_error + ".";
    if (status == 404) {
      fail(error_codes::kPeripheralDaemonUnavailable,
           "The installed SiMa Sentinel is too old to serve the peripheral catalog (HTTP 404 "
           "for " +
               std::string(kCatalogPath) + ")." + reported +
               " Update Sentinel with `sima-cli neat install sentinel`, then try again.");
    }
    if (status == 503) {
      fail(error_codes::kPeripheralDaemonUnavailable,
           "SiMa Sentinel's peripheral discovery is not running." + reported +
               " Check simaai-sentinel.service and its journal.");
    }
    fail(error_codes::kPeripheralDaemonUnavailable,
         "Sentinel returned unexpected HTTP status " + std::to_string(status) +
             " for the peripheral catalog." + reported +
             " Update Sentinel with `sima-cli neat install sentinel` and check "
             "simaai-sentinel.service.");
  }
  return parse_catalog(body);
}

} // namespace simaai::neat::peripherals_internal

namespace simaai::neat::peripherals {

Catalog list() {
  return peripherals_internal::list_from_socket("/run/simaai-sentinel/api.sock",
                                                std::chrono::seconds(5));
}

} // namespace simaai::neat::peripherals
