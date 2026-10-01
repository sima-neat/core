#include "PeripheralApiServer.h"
#include "PeripheralProtocol.h"

#include <nlohmann/json.hpp>

#include <fcntl.h>
#include <poll.h>
#include <sys/file.h>
#include <sys/socket.h>
#include <sys/stat.h>
#include <sys/un.h>
#include <unistd.h>

#include <atomic>
#include <cerrno>
#include <charconv>
#include <chrono>
#include <condition_variable>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <exception>
#include <filesystem>
#include <mutex>
#include <optional>
#include <stdexcept>
#include <string>
#include <string_view>
#include <thread>
#include <unordered_map>
#include <unordered_set>
#include <utility>

namespace simaai::neat::peripherals_internal {
namespace {

constexpr std::size_t kMaximumRequestBytes = 8192;
constexpr auto kClientIoTimeout = std::chrono::seconds(5);
constexpr auto kShutdownResponseGrace = std::chrono::milliseconds(250);

struct FileDescriptor {
  int value = -1;

  FileDescriptor() = default;
  explicit FileDescriptor(int fd) : value(fd) {}
  ~FileDescriptor() {
    reset();
  }

  FileDescriptor(const FileDescriptor&) = delete;
  FileDescriptor& operator=(const FileDescriptor&) = delete;

  FileDescriptor(FileDescriptor&& other) noexcept : value(std::exchange(other.value, -1)) {}

  FileDescriptor& operator=(FileDescriptor&& other) noexcept {
    if (this != &other) {
      reset();
      value = std::exchange(other.value, -1);
    }
    return *this;
  }

  void reset(int fd = -1) {
    if (value >= 0)
      ::close(value);
    value = fd;
  }
};

std::string status_text(int status) {
  switch (status) {
  case 200:
    return "OK";
  case 202:
    return "Accepted";
  case 400:
    return "Bad Request";
  case 404:
    return "Not Found";
  case 405:
    return "Method Not Allowed";
  case 413:
    return "Payload Too Large";
  case 503:
    return "Service Unavailable";
  default:
    return "Internal Server Error";
  }
}

void send_all(int fd, std::string_view value, std::chrono::milliseconds timeout_value) {
  const auto deadline = std::chrono::steady_clock::now() + timeout_value;
  while (!value.empty()) {
    const auto now = std::chrono::steady_clock::now();
    if (now >= deadline)
      return;
    const auto remaining = std::chrono::ceil<std::chrono::milliseconds>(deadline - now).count();
    pollfd descriptor{fd, POLLOUT, 0};
    const int poll_result = ::poll(&descriptor, 1, static_cast<int>(remaining));
    if (poll_result < 0 && errno == EINTR)
      continue;
    if (poll_result <= 0 || !(descriptor.revents & POLLOUT))
      return;
    const ssize_t sent = ::send(fd, value.data(), value.size(), MSG_NOSIGNAL);
    if (sent == 0)
      return;
    if (sent < 0) {
      if (errno == EINTR || errno == EAGAIN || errno == EWOULDBLOCK)
        continue;
      return;
    }
    value.remove_prefix(static_cast<std::size_t>(sent));
  }
}

void send_json(int fd, int status, const nlohmann::json& body,
               std::chrono::milliseconds timeout_value = kClientIoTimeout) {
  std::string payload = body.dump();
  if (payload.size() > kMaximumResponseBytes) {
    status = 500;
    payload =
        encode_error("response_too_large", "The peripheral catalog exceeded the v1 response limit.")
            .dump();
  }
  const std::string response =
      "HTTP/1.1 " + std::to_string(status) + " " + status_text(status) +
      "\r\nContent-Type: application/json\r\nContent-Length: " + std::to_string(payload.size()) +
      "\r\nConnection: close\r\n\r\n" + payload;
  send_all(fd, response, timeout_value);
}

std::optional<std::uint64_t> parse_u64(std::string_view value) {
  std::uint64_t parsed = 0;
  const auto result = std::from_chars(value.data(), value.data() + value.size(), parsed);
  if (result.ec != std::errc{} || result.ptr != value.data() + value.size())
    return std::nullopt;
  return parsed;
}

struct Request {
  std::string method;
  std::string path;
  std::unordered_map<std::string, std::string> query;
};

std::optional<Request> parse_request(const std::string& raw) {
  const std::size_t line_end = raw.find("\r\n");
  if (line_end == std::string::npos)
    return std::nullopt;
  const std::string_view line(raw.data(), line_end);
  const std::size_t first_space = line.find(' ');
  const std::size_t second_space =
      first_space == std::string_view::npos ? first_space : line.find(' ', first_space + 1);
  if (first_space == std::string_view::npos || second_space == std::string_view::npos)
    return std::nullopt;
  Request request;
  request.method = line.substr(0, first_space);
  const std::string_view target = line.substr(first_space + 1, second_space - first_space - 1);
  const std::size_t question = target.find('?');
  request.path = target.substr(0, question);
  if (question == std::string_view::npos)
    return request;

  std::string_view query = target.substr(question + 1);
  while (!query.empty()) {
    const std::size_t separator = query.find('&');
    const std::string_view item = query.substr(0, separator);
    const std::size_t equals = item.find('=');
    if (equals == std::string_view::npos)
      request.query.emplace(std::string(item), "");
    else
      request.query.emplace(std::string(item.substr(0, equals)),
                            std::string(item.substr(equals + 1)));
    if (separator == std::string_view::npos)
      break;
    query.remove_prefix(separator + 1);
  }
  return request;
}

std::optional<std::string>
read_request(int fd, std::chrono::milliseconds timeout_value = std::chrono::seconds(5)) {
  const auto deadline = std::chrono::steady_clock::now() + timeout_value;
  std::string request;
  request.reserve(1024);
  while (request.find("\r\n\r\n") == std::string::npos) {
    const auto now = std::chrono::steady_clock::now();
    if (now >= deadline)
      return std::nullopt;
    const auto remaining = std::chrono::ceil<std::chrono::milliseconds>(deadline - now).count();
    pollfd descriptor{fd, POLLIN, 0};
    const int poll_result = ::poll(&descriptor, 1, static_cast<int>(remaining));
    if (poll_result < 0 && errno == EINTR)
      continue;
    if (poll_result <= 0)
      return std::nullopt;
    if (!(descriptor.revents & POLLIN))
      return std::nullopt;
    char buffer[1024];
    const ssize_t received = ::recv(fd, buffer, sizeof(buffer), 0);
    if (received == 0)
      return std::nullopt;
    if (received < 0) {
      if (errno == EINTR || errno == EAGAIN || errno == EWOULDBLOCK)
        continue;
      return std::nullopt;
    }
    request.append(buffer, static_cast<std::size_t>(received));
    if (request.size() > kMaximumRequestBytes)
      return std::string{};
  }
  return request;
}

bool socket_is_active(const std::string& path) {
  FileDescriptor probe(::socket(AF_UNIX, SOCK_STREAM | SOCK_CLOEXEC | SOCK_NONBLOCK, 0));
  if (probe.value < 0)
    throw std::runtime_error("failed to create peripheral API socket probe: " +
                             std::string(std::strerror(errno)));
  sockaddr_un address{};
  address.sun_family = AF_UNIX;
  std::memcpy(address.sun_path, path.c_str(), path.size() + 1);
  if (::connect(probe.value, reinterpret_cast<sockaddr*>(&address), sizeof(address)) == 0)
    return true;
  if (errno == EAGAIN || errno == EWOULDBLOCK || errno == EINPROGRESS)
    return true;
  if (errno == ECONNREFUSED || errno == ENOENT)
    return false;
  throw std::runtime_error("failed to inspect existing peripheral API socket: " +
                           std::string(std::strerror(errno)));
}

} // namespace

struct PeripheralApiServer::Impl {
  Impl(std::string socket_path_value, PeripheralCatalog& catalog_value,
       RefreshRequest refresh_request_value, std::size_t max_clients_value)
      : socket_path(std::move(socket_path_value)), catalog(catalog_value),
        refresh_request(std::move(refresh_request_value)), max_clients(max_clients_value) {
    if (socket_path.empty())
      throw std::invalid_argument("peripheral API socket path must not be empty");
    if (!refresh_request)
      throw std::invalid_argument("peripheral API refresh callback must be set");
    if (max_clients == 0)
      throw std::invalid_argument("peripheral API max clients must be greater than zero");
  }

  void route(int fd, const Request& request) {
    if (request.path == kHealthPath) {
      if (request.method != "GET") {
        send_json(fd, 405, encode_error("method_not_allowed"));
        return;
      }
      send_json(fd, 200, catalog.health_json());
      return;
    }
    if (request.path == kCatalogPath) {
      if (request.method != "GET") {
        send_json(fd, 405, encode_error("method_not_allowed"));
        return;
      }
      send_json(fd, 200, catalog.catalog_json());
      return;
    }
    if (request.path == kEventsPath) {
      if (request.method != "GET") {
        send_json(fd, 405, encode_error("method_not_allowed"));
        return;
      }
      std::uint64_t after_sequence = 0;
      std::uint64_t wait_ms = 0;
      if (const auto item = request.query.find("after_sequence"); item != request.query.end()) {
        const auto value = parse_u64(item->second);
        if (!value) {
          send_json(fd, 400, encode_error("invalid_after_sequence"));
          return;
        }
        after_sequence = *value;
      }
      if (const auto item = request.query.find("wait_ms"); item != request.query.end()) {
        const auto value = parse_u64(item->second);
        if (!value || *value > kMaximumEventWaitMs) {
          nlohmann::json response = encode_error("invalid_wait_ms");
          response["maximum_wait_ms"] = kMaximumEventWaitMs;
          send_json(fd, 400, response);
          return;
        }
        wait_ms = *value;
      }
      std::optional<std::string> instance_id;
      if (const auto item = request.query.find("instance_id"); item != request.query.end())
        instance_id = item->second;
      send_json(
          fd, 200,
          catalog.events_json(after_sequence, std::chrono::milliseconds(wait_ms), instance_id));
      return;
    }
    if (request.path == kRefreshPath) {
      if (request.method != "POST") {
        send_json(fd, 405, encode_error("method_not_allowed"));
        return;
      }
      send_json(fd, 202, encode_refresh_accepted(refresh_request()));
      return;
    }
    send_json(fd, 404, encode_error("not_found"));
  }

  void serve(int fd) {
    const auto raw = read_request(fd);
    if (!raw)
      return;
    if (raw->empty()) {
      send_json(fd, 413, encode_error("request_too_large"));
      return;
    }
    const auto request = parse_request(*raw);
    if (!request) {
      send_json(fd, 400, encode_error("invalid_request"));
      return;
    }
    try {
      route(fd, *request);
    } catch (const std::exception& error) {
      send_json(fd, 500, encode_error("internal_error", error.what()));
    }
  }

  void accept_loop() {
    while (!stopping.load()) {
      pollfd descriptor{listener.value, POLLIN, 0};
      const int result = ::poll(&descriptor, 1, 250);
      if (result < 0) {
        if (errno == EINTR)
          continue;
        throw std::runtime_error("peripheral API listener poll failed: " +
                                 std::string(std::strerror(errno)));
      }
      if (result == 0)
        continue;
      if (!(descriptor.revents & POLLIN)) {
        if (stopping.load())
          return;
        throw std::runtime_error("peripheral API listener became unavailable");
      }
      const int client = ::accept4(listener.value, nullptr, nullptr, SOCK_CLOEXEC | SOCK_NONBLOCK);
      if (client < 0) {
        if (errno == EINTR || errno == EAGAIN || errno == EWOULDBLOCK || errno == ECONNABORTED)
          continue;
        if (stopping.load())
          return;
        throw std::runtime_error("failed to accept peripheral API client: " +
                                 std::string(std::strerror(errno)));
      }
      {
        std::lock_guard lock(clients_mutex);
        if (active_client_fds.size() >= max_clients) {
          FileDescriptor rejected(client);
          (void)read_request(rejected.value, std::chrono::milliseconds(100));
          send_json(rejected.value, 503, encode_error("too_many_clients"),
                    std::chrono::milliseconds(100));
          ::shutdown(rejected.value, SHUT_WR);
          continue;
        }
        active_client_fds.insert(client);
      }
      try {
        std::thread([this, client] {
          try {
            serve(client);
          } catch (...) {
          }
          std::lock_guard lock(clients_mutex);
          active_client_fds.erase(client);
          ::close(client);
          clients_changed.notify_all();
        }).detach();
      } catch (...) {
        std::lock_guard lock(clients_mutex);
        active_client_fds.erase(client);
        ::close(client);
        clients_changed.notify_all();
      }
    }
  }

  void start() {
    if (worker.joinable())
      throw std::logic_error("peripheral API server is already running");
    sockaddr_un address{};
    if (socket_path.size() >= sizeof(address.sun_path))
      throw std::invalid_argument("peripheral API socket path is too long");

    const std::filesystem::path parent = std::filesystem::path(socket_path).parent_path();
    if (!parent.empty())
      std::filesystem::create_directories(parent);

    const std::string lock_path = socket_path + ".lock";
    FileDescriptor lifetime_lock(
        ::open(lock_path.c_str(), O_CREAT | O_RDWR | O_CLOEXEC | O_NOFOLLOW, 0660));
    if (lifetime_lock.value < 0)
      throw std::runtime_error("failed to open peripheral API ownership lock: " +
                               std::string(std::strerror(errno)));
    if (::flock(lifetime_lock.value, LOCK_EX | LOCK_NB) != 0) {
      if (errno == EWOULDBLOCK || errno == EAGAIN)
        throw std::runtime_error("peripheral API socket is already starting or active: " +
                                 socket_path);
      throw std::runtime_error("failed to lock peripheral API ownership: " +
                               std::string(std::strerror(errno)));
    }
    if (::fchmod(lifetime_lock.value, 0660) != 0)
      throw std::runtime_error("failed to set peripheral API ownership lock permissions: " +
                               std::string(std::strerror(errno)));

    struct stat existing {};
    if (::lstat(socket_path.c_str(), &existing) == 0) {
      if (!S_ISSOCK(existing.st_mode))
        throw std::runtime_error("refusing to replace non-socket path: " + socket_path);
      if (socket_is_active(socket_path))
        throw std::runtime_error("peripheral API socket is already active: " + socket_path);
      if (::unlink(socket_path.c_str()) != 0)
        throw std::runtime_error("failed to remove stale peripheral socket: " +
                                 std::string(std::strerror(errno)));
    } else if (errno != ENOENT) {
      throw std::runtime_error("failed to inspect peripheral socket: " +
                               std::string(std::strerror(errno)));
    }

    listener.reset(::socket(AF_UNIX, SOCK_STREAM | SOCK_CLOEXEC | SOCK_NONBLOCK, 0));
    if (listener.value < 0)
      throw std::runtime_error("failed to create peripheral API socket: " +
                               std::string(std::strerror(errno)));
    bool bound_by_this_instance = false;
    try {
      address.sun_family = AF_UNIX;
      std::memcpy(address.sun_path, socket_path.c_str(), socket_path.size() + 1);
      if (::bind(listener.value, reinterpret_cast<sockaddr*>(&address), sizeof(address)) != 0)
        throw std::runtime_error("failed to bind peripheral API socket: " +
                                 std::string(std::strerror(errno)));
      bound_by_this_instance = true;
      if (::chmod(socket_path.c_str(), 0660) != 0)
        throw std::runtime_error("failed to set peripheral API socket permissions: " +
                                 std::string(std::strerror(errno)));
      if (::listen(listener.value, static_cast<int>(max_clients)) != 0)
        throw std::runtime_error("failed to listen on peripheral API socket: " +
                                 std::string(std::strerror(errno)));
      struct stat bound {};
      if (::lstat(socket_path.c_str(), &bound) != 0 || !S_ISSOCK(bound.st_mode))
        throw std::runtime_error("failed to identify bound peripheral API socket");
      socket_device = bound.st_dev;
      socket_inode = bound.st_ino;
      owns_socket_path = true;
    } catch (...) {
      listener.reset();
      if (bound_by_this_instance)
        ::unlink(socket_path.c_str());
      owns_socket_path = false;
      throw;
    }
    clear_failure();
    stopping = false;
    try {
      worker = std::thread([this] {
        try {
          accept_loop();
        } catch (...) {
          record_failure(std::current_exception());
        }
      });
    } catch (...) {
      listener.reset();
      unlink_owned_socket();
      throw;
    }
    ownership_lock = std::move(lifetime_lock);
  }

  void stop() {
    if (!worker.joinable()) {
      listener.reset();
      unlink_owned_socket();
      ownership_lock.reset();
      return;
    }
    stopping = true;
    catalog.shutdown();
    ::shutdown(listener.value, SHUT_RDWR);
    worker.join();
    listener.reset();
    {
      std::unique_lock lock(clients_mutex);
      if (!clients_changed.wait_for(lock, kShutdownResponseGrace,
                                    [this] { return active_client_fds.empty(); })) {
        for (const int fd : active_client_fds)
          ::shutdown(fd, SHUT_RDWR);
        clients_changed.wait(lock, [this] { return active_client_fds.empty(); });
      }
    }
    unlink_owned_socket();
    ownership_lock.reset();
  }

  std::string socket_path;
  PeripheralCatalog& catalog;
  RefreshRequest refresh_request;
  std::size_t max_clients;
  FileDescriptor listener;
  FileDescriptor ownership_lock;
  dev_t socket_device = 0;
  ino_t socket_inode = 0;
  bool owns_socket_path = false;
  std::thread worker;
  std::atomic<bool> stopping{false};
  std::mutex clients_mutex;
  std::condition_variable clients_changed;
  std::unordered_set<int> active_client_fds;
  mutable std::mutex failure_mutex;
  std::exception_ptr worker_failure;

  void unlink_owned_socket() noexcept {
    if (!owns_socket_path)
      return;
    struct stat current {};
    if (::lstat(socket_path.c_str(), &current) == 0 && S_ISSOCK(current.st_mode) &&
        current.st_dev == socket_device && current.st_ino == socket_inode) {
      ::unlink(socket_path.c_str());
    }
    owns_socket_path = false;
  }

  void clear_failure() {
    std::lock_guard lock(failure_mutex);
    worker_failure = nullptr;
  }

  void record_failure(std::exception_ptr failure) noexcept {
    std::lock_guard lock(failure_mutex);
    worker_failure = std::move(failure);
  }

  void throw_if_failed() const {
    std::exception_ptr failure;
    {
      std::lock_guard lock(failure_mutex);
      failure = worker_failure;
    }
    if (failure)
      std::rethrow_exception(failure);
  }
};

PeripheralApiServer::PeripheralApiServer(std::string socket_path, PeripheralCatalog& catalog,
                                         RefreshRequest refresh_request, std::size_t max_clients)
    : impl_(std::make_unique<Impl>(std::move(socket_path), catalog, std::move(refresh_request),
                                   max_clients)) {}

PeripheralApiServer::~PeripheralApiServer() {
  stop();
}

void PeripheralApiServer::start() {
  impl_->start();
}

void PeripheralApiServer::stop() {
  if (impl_)
    impl_->stop();
}

void PeripheralApiServer::throw_if_failed() const {
  impl_->throw_if_failed();
}

} // namespace simaai::neat::peripherals_internal
