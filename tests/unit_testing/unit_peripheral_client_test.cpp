#include "peripherals/PeripheralCatalog.h"

#include "nodes/io/CameraInput.h"
#include "peripherals/internal/PeripheralClient.h"
#include "peripherals/internal/ProtocolContract.h"
#include "pipeline/ErrorCodes.h"
#include "pipeline/NeatError.h"

#include <nlohmann/json.hpp>

#include <sys/socket.h>
#include <sys/stat.h>
#include <sys/un.h>
#include <unistd.h>

#include <atomic>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <exception>
#include <filesystem>
#include <fstream>
#include <functional>
#include <future>
#include <iostream>
#include <stdexcept>
#include <string>
#include <string_view>
#include <thread>
#include <utility>

namespace {

using namespace std::chrono_literals;
using simaai::neat::NeatError;
using simaai::neat::peripherals_internal::list_from_socket;

void require(bool condition, const std::string& message) {
  if (!condition)
    throw std::runtime_error(message);
}

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

  int get() const noexcept {
    return value_;
  }

private:
  int value_;
};

class TemporaryDirectory {
public:
  TemporaryDirectory() {
    std::string pattern = "/tmp/sima-neat-peripheral-client-XXXXXX";
    path_.resize(pattern.size());
    std::memcpy(path_.data(), pattern.data(), pattern.size());
    if (!::mkdtemp(path_.data()))
      throw std::runtime_error("mkdtemp failed");
  }
  ~TemporaryDirectory() {
    std::error_code ignored;
    std::filesystem::permissions(path_, std::filesystem::perms::owner_all,
                                 std::filesystem::perm_options::add, ignored);
    std::filesystem::remove_all(path_, ignored);
  }

  std::string socket_path(std::string_view name = "api.sock") const {
    return path_ + "/" + std::string(name);
  }
  const std::string& path() const noexcept {
    return path_;
  }

private:
  std::string path_;
};

std::string read_request_one_byte_at_a_time(int fd) {
  std::string request;
  while (request.find("\r\n\r\n") == std::string::npos) {
    char value = 0;
    const ssize_t received = ::recv(fd, &value, 1, 0);
    if (received <= 0)
      throw std::runtime_error("client closed before sending a complete request");
    request.push_back(value);
    if (request.size() > 8192)
      throw std::runtime_error("client request exceeded test limit");
  }
  return request;
}

void send_in_chunks(int fd, std::string_view value, std::size_t chunk_size = 1) {
  while (!value.empty()) {
    const std::size_t current = std::min(chunk_size, value.size());
    const ssize_t sent = ::send(fd, value.data(), current, MSG_NOSIGNAL);
    if (sent <= 0)
      return;
    value.remove_prefix(static_cast<std::size_t>(sent));
  }
}

class FakeServer {
public:
  using Handler = std::function<void(int)>;

  FakeServer(std::string path, Handler handler) : path_(std::move(path)) {
    FileDescriptor listener(::socket(AF_UNIX, SOCK_STREAM | SOCK_CLOEXEC, 0));
    if (listener.get() < 0)
      throw std::runtime_error("failed to create fake server socket");
    sockaddr_un address{};
    address.sun_family = AF_UNIX;
    require(path_.size() < sizeof(address.sun_path), "fake socket path is too long");
    std::memcpy(address.sun_path, path_.c_str(), path_.size() + 1);
    if (::bind(listener.get(), reinterpret_cast<sockaddr*>(&address), sizeof(address)) != 0)
      throw std::runtime_error("failed to bind fake server socket");
    if (::listen(listener.get(), 1) != 0)
      throw std::runtime_error("failed to listen on fake server socket");

    thread_ = std::thread([listener = std::move(listener), handler = std::move(handler), this]() {
      try {
        FileDescriptor client(::accept4(listener.get(), nullptr, nullptr, SOCK_CLOEXEC));
        if (client.get() < 0)
          throw std::runtime_error("fake server accept failed");
        handler(client.get());
      } catch (...) {
        error_ = std::current_exception();
      }
    });
  }

  ~FakeServer() {
    if (thread_.joinable())
      thread_.join();
    ::unlink(path_.c_str());
  }

  void finish() {
    if (thread_.joinable())
      thread_.join();
    if (error_)
      std::rethrow_exception(error_);
  }

private:
  std::string path_;
  std::thread thread_;
  std::exception_ptr error_;
};

nlohmann::json ready_catalog() {
  return {
      {"schema_version", 1},
      {"instance_id", "daemon-instance-a"},
      {"state", "ready"},
      {"ready", true},
      {"stale", false},
      {"revision", 7},
      {"sequence", 11},
      {"scan_sequence", 13},
      {"last_success_at", "2026-10-01T01:02:03.004Z"},
      {"last_attempt_at", "2026-10-01T01:02:03.004Z"},
      {"error", nullptr},
      {"issues", nlohmann::json::array()},
      {"devices", nlohmann::json::array({{{"id", "camera:imx477 5-001a"},
                                          {"type", "camera"},
                                          {"provider", "daemon.camera.libcamera"},
                                          {"future_device_field", 42},
                                          {"camera",
                                           {{"camera_name", "imx477 5-001a"},
                                            {"model", "imx477"},
                                            {"backend", "libcamera"},
                                            {"modes",
                                             {{{"format", "NV12"},
                                               {"width", 1920},
                                               {"height", 1080},
                                               {"framerate_num", 30},
                                               {"framerate_den", 1},
                                               {"supported", true},
                                               {"reason", ""},
                                               {"future_mode_field", true}},
                                              {{"format", "NV12"},
                                               {"size_range",
                                                {{"min_width", 640},
                                                 {"min_height", 480},
                                                 {"max_width", 1920},
                                                 {"max_height", 1080},
                                                 {"step_width", 16},
                                                 {"step_height", 8}}},
                                               {"framerate_num", 30},
                                               {"framerate_den", 1},
                                               {"supported", false},
                                               {"reason", "range is advisory"}}}}}}},
                                         {{"id", "lidar:future-1"},
                                          {"type", "lidar"},
                                          {"provider", "daemon.lidar.future"},
                                          {"lidar", {{"vendor_extension", true}}}}})},
      {"future_root_field", nlohmann::json::object({{"ignored", true}})},
  };
}

nlohmann::json canonical_catalog_fixture() {
  const std::string path = std::string(SIMAAI_PERIPHERAL_FIXTURE_DIR) + "/catalog.json";
  std::ifstream input(path);
  require(input.good(), "could not open canonical daemon fixture " + path);
  return nlohmann::json::parse(input);
}

std::string response_for(const std::string& body, int status = 200) {
  const std::string reason = status == 200   ? "OK"
                             : status == 404 ? "Not Found"
                                             : "Service Unavailable";
  return "HTTP/1.1 " + std::to_string(status) + " " + reason +
         "\r\nContent-Type: application/json\r\nContent-Length: " + std::to_string(body.size()) +
         "\r\nConnection: close\r\n\r\n" + body;
}

auto serve_response(const std::string& path, std::string response, std::size_t chunk_size = 1) {
  return FakeServer(path, [response = std::move(response), chunk_size](int client) {
    const std::string request = read_request_one_byte_at_a_time(client);
    require(request.starts_with("GET /v1/peripherals HTTP/1.1\r\n"),
            "client must perform exactly GET /v1/peripherals");
    send_in_chunks(client, response, chunk_size);
  });
}

template <typename Function>
std::string require_error(Function&& function, const char* code,
                          std::string_view message_fragment = {}) {
  try {
    function();
  } catch (const NeatError& error) {
    require(error.report().error_code == code,
            "expected error code " + std::string(code) + ", got " + error.report().error_code);
    if (!message_fragment.empty())
      require(std::string(error.what()).find(message_fragment) != std::string::npos,
              "error message lacks required actionable context");
    return error.what();
  }
  throw std::runtime_error("expected NeatError " + std::string(code));
}

void test_success_and_partial_io() {
  TemporaryDirectory directory;
  auto server = serve_response(directory.socket_path(), response_for(ready_catalog().dump()), 1);
  const auto catalog = list_from_socket(directory.socket_path(), 2s, 3);
  server.finish();

  require(catalog.instance_id == "daemon-instance-a" && catalog.revision == 7 &&
              catalog.sequence == 11 && catalog.scan_sequence == 13,
          "catalog metadata was not preserved");
  require(catalog.state == "ready" && !catalog.stale && !catalog.error && catalog.issues.empty(),
          "ready catalog freshness was not preserved");
  require(catalog.size() == 2 && !catalog.empty() && catalog.begin()->id == catalog[0].id,
          "catalog must be iterable over all devices");
  const auto& camera_device = catalog[0];
  require(camera_device.camera && camera_device.camera->camera_name == "imx477 5-001a" &&
              camera_device.camera->model == "imx477" &&
              camera_device.camera->backend == "libcamera",
          "camera identity/details were not preserved");
  require(camera_device.camera->modes.size() == 2 && camera_device.camera->modes[0].width == 1920 &&
              camera_device.camera->modes[0].height == 1080 &&
              camera_device.camera->modes[0].supported &&
              camera_device.camera->modes[1].is_range() &&
              camera_device.camera->modes[1].size_range->step_width == 16 &&
              !camera_device.camera->modes[1].supported &&
              camera_device.camera->modes[1].reason == "range is advisory",
          "camera modes/ranges were not mapped exactly");
  require(catalog[1].type == "lidar" && catalog[1].provider == "daemon.lidar.future" &&
              !catalog[1].camera,
          "unknown peripheral types must retain common identity without camera details");
}

void test_canonical_daemon_fixture() {
  TemporaryDirectory directory;
  const auto fixture = canonical_catalog_fixture();
  require(fixture.contains("changes") && fixture["changes"].is_array() &&
              !fixture["changes"].empty(),
          "canonical fixture must carry the optional Sentinel change log");
  require(fixture.contains("support") && fixture["support"]["state"] == "applied",
          "canonical fixture must carry Sentinel's support status");
  auto server = serve_response(directory.socket_path(), response_for(fixture.dump()), 2);
  const auto catalog = list_from_socket(directory.socket_path(), 2s, 5);
  server.finish();

  require(catalog.instance_id == "fixture-daemon-a" && catalog.revision == 4 &&
              catalog.sequence == 9 && catalog.scan_sequence == 12 && catalog.stale &&
              catalog.error && catalog.issues.size() == 1 && catalog.size() == 2,
          "canonical daemon metadata was not preserved");
  const auto& mipi = catalog[0];
  require(mipi.provider == "daemon.camera.mipi" && mipi.camera &&
              mipi.camera->camera_name == "imx477 5-001a" && mipi.camera->model == "imx477" &&
              mipi.camera->backend == "mipi" && mipi.camera->modes.size() == 9,
          "canonical MIPI camera details were not preserved");
  for (const auto& mode : mipi.camera->modes) {
    require(!mode.is_range() && mode.framerate_num == 30 && mode.framerate_den == 1 &&
                mode.supported == (mode.format == "NV12") &&
                mode.reason.empty() == (mode.format == "NV12"),
            "canonical MIPI modes must carry Sentinel's support verdicts");
  }
  require(catalog[1].camera && !catalog[1].camera->camera_name &&
              catalog[1].camera->backend == "v4l2" && !catalog[1].camera->modes[0].supported &&
              catalog[1].camera->modes[0].reason ==
                  "CameraInput currently accepts MIPI cameras only; direct V4L2 capture is not "
                  "supported.",
          "canonical USB camera details or unknown optional fields were mishandled");
  for (std::size_t index = 0; index < catalog.size(); ++index)
    require(nlohmann::json::parse(catalog[index].details_json) ==
                fixture["devices"][index]["camera"],
            "canonical camera details_json must preserve every daemon field");
}

void test_details_for_any_type() {
  TemporaryDirectory directory;
  const auto microphone = nlohmann::json::parse(
      R"({"channels": 2, "formats": ["S16_LE"], "nested": {"a": [1, 2]}, "gain_db": -3.5,
          "label": "USB \u00e9 mic", "big": 18446744073709551615})");
  auto body = ready_catalog();
  body["devices"].push_back({{"id", "microphone:usb-1"},
                             {"type", "microphone"},
                             {"provider", "daemon.audio.alsa"},
                             {"microphone", microphone}});
  body["devices"].push_back(
      {{"id", "imu:absent"}, {"type", "imu"}, {"provider", "daemon.imu.future"}});
  body["devices"].push_back({{"id", "radar:null"},
                             {"type", "radar"},
                             {"provider", "daemon.radar.future"},
                             {"radar", nullptr}});
  auto server = serve_response(directory.socket_path(), response_for(body.dump()), 7);
  const auto catalog = list_from_socket(directory.socket_path(), 2s);
  server.finish();

  require(catalog.size() == 5, "every peripheral type must remain in the catalog");
  require(catalog[0].camera && catalog[0].camera->backend == "libcamera" &&
              nlohmann::json::parse(catalog[0].details_json) == body["devices"][0]["camera"],
          "cameras must carry typed details and the exact details_json");
  require(nlohmann::json::parse(catalog[1].details_json) == body["devices"][1]["lidar"] &&
              !catalog[1].camera,
          "unknown lidar details must round-trip through details_json");
  require(catalog[2].type == "microphone" && !catalog[2].camera &&
              nlohmann::json::parse(catalog[2].details_json) == microphone,
          "unknown microphone details must round-trip exactly through details_json");
  require(catalog[3].type == "imu" && catalog[3].details_json == "{}",
          "an absent details key must yield an empty JSON object");
  require(catalog[4].type == "radar" && catalog[4].details_json == "{}",
          "a null details key must yield an empty JSON object");
}

void test_empty_and_stale_catalogs() {
  TemporaryDirectory directory;
  auto empty = ready_catalog();
  empty["devices"] = nlohmann::json::array();
  {
    auto server = serve_response(directory.socket_path("empty.sock"), response_for(empty.dump()));
    const auto catalog = list_from_socket(directory.socket_path("empty.sock"), 2s);
    server.finish();
    require(catalog.empty(), "a ready empty daemon catalog must succeed");
  }

  auto stale = ready_catalog();
  stale["state"] = "degraded";
  stale["stale"] = true;
  stale["error"] = {{"code", "peripherals.discovery_failed"}, {"reason", "camera provider failed"}};
  stale["issues"] = {{{"provider", "daemon.camera.libcamera"},
                      {"code", "io.permission_denied"},
                      {"reason", "permission denied"},
                      {"retained_last_good", true}}};
  {
    auto server = serve_response(directory.socket_path("stale.sock"), response_for(stale.dump()));
    const auto catalog = list_from_socket(directory.socket_path("stale.sock"), 2s);
    server.finish();
    require(catalog.stale && catalog.error &&
                catalog.error->code == "peripherals.discovery_failed" &&
                catalog.issues.size() == 1 && catalog.issues[0].retained_last_good &&
                catalog.size() == 2,
            "stale last-good data and diagnostics must be returned together");
  }
}

void test_connection_failures() {
  TemporaryDirectory directory;
  require_error([&] { (void)list_from_socket(directory.socket_path("missing.sock"), 100ms); },
                simaai::neat::error_codes::kPeripheralDaemonUnavailable,
                "sima-cli neat install sentinel");

  const std::string refused_path = directory.socket_path("refused.sock");
  {
    FileDescriptor stale(::socket(AF_UNIX, SOCK_STREAM | SOCK_CLOEXEC, 0));
    require(stale.get() >= 0, "failed to create refused socket fixture");
    sockaddr_un address{};
    address.sun_family = AF_UNIX;
    std::memcpy(address.sun_path, refused_path.c_str(), refused_path.size() + 1);
    require(::bind(stale.get(), reinterpret_cast<sockaddr*>(&address), sizeof(address)) == 0,
            "failed to bind refused socket fixture");
  }
  require_error([&] { (void)list_from_socket(refused_path, 100ms); },
                simaai::neat::error_codes::kPeripheralDaemonUnavailable, "simaai-sentinel.service");

  require(::chmod(directory.path().c_str(), 0000) == 0, "failed to protect permission fixture");
  require_error([&] { (void)list_from_socket(directory.socket_path("denied.sock"), 100ms); },
                simaai::neat::error_codes::kPermissionDenied, "/run/simaai-sentinel/api.sock");
  require(::chmod(directory.path().c_str(), 0700) == 0, "failed to restore permission fixture");
}

void test_timeout_and_incomplete_response() {
  TemporaryDirectory directory;
  {
    FakeServer server(directory.socket_path("timeout.sock"), [](int client) {
      (void)read_request_one_byte_at_a_time(client);
      std::this_thread::sleep_for(150ms);
    });
    const auto start = std::chrono::steady_clock::now();
    require_error([&] { (void)list_from_socket(directory.socket_path("timeout.sock"), 40ms); },
                  simaai::neat::error_codes::kPeripheralDaemonTimeout);
    require(std::chrono::steady_clock::now() - start < 120ms,
            "catalog request did not honor its absolute deadline");
    server.finish();
  }
  {
    auto server = serve_response(directory.socket_path("partial.sock"),
                                 "HTTP/1.1 200 OK\r\nContent-Length: 12\r\n\r\n{}", 1);
    require_error([&] { (void)list_from_socket(directory.socket_path("partial.sock"), 1s); },
                  simaai::neat::error_codes::kIoParse, "response completed");
    server.finish();
  }
}

void test_protocol_failures() {
  TemporaryDirectory directory;
  {
    auto server = serve_response(directory.socket_path("json.sock"), response_for("{"));
    require_error([&] { (void)list_from_socket(directory.socket_path("json.sock"), 1s); },
                  simaai::neat::error_codes::kIoParse, "JSON parsing failed");
    server.finish();
  }
  {
    const std::string oversized =
        "HTTP/1.1 200 OK\r\nContent-Length: " +
        std::to_string(simaai::neat::peripherals_internal::kMaximumResponseBytes + 1) + "\r\n\r\n";
    auto server = serve_response(directory.socket_path("oversized.sock"), oversized);
    require_error([&] { (void)list_from_socket(directory.socket_path("oversized.sock"), 1s); },
                  simaai::neat::error_codes::kResponseTooLarge, "4 MiB");
    server.finish();
  }
  {
    const std::string body = R"({"error":"response_too_large"})";
    auto server =
        serve_response(directory.socket_path("daemon-limit.sock"), response_for(body, 500));
    require_error([&] { (void)list_from_socket(directory.socket_path("daemon-limit.sock"), 1s); },
                  simaai::neat::error_codes::kResponseTooLarge, "4 MiB");
    server.finish();
  }
  {
    auto schema = ready_catalog();
    schema["schema_version"] = 2;
    auto server = serve_response(directory.socket_path("schema.sock"), response_for(schema.dump()));
    require_error([&] { (void)list_from_socket(directory.socket_path("schema.sock"), 1s); },
                  simaai::neat::error_codes::kRuntimeAbiMismatch, "matching Core");
    server.finish();
  }
  {
    auto malformed = ready_catalog();
    malformed["devices"][0].erase("provider");
    auto server =
        serve_response(directory.socket_path("fields.sock"), response_for(malformed.dump()));
    require_error([&] { (void)list_from_socket(directory.socket_path("fields.sock"), 1s); },
                  simaai::neat::error_codes::kIoParse, "provider");
    server.finish();
  }
  {
    auto malformed = ready_catalog();
    malformed["devices"][0]["camera"] = nlohmann::json::array({1, 2});
    auto server =
        serve_response(directory.socket_path("camera.sock"), response_for(malformed.dump()));
    require_error([&] { (void)list_from_socket(directory.socket_path("camera.sock"), 1s); },
                  simaai::neat::error_codes::kIoParse, "camera details must be an object");
    server.finish();
  }
  {
    const std::string body = R"({"error":"too_many_clients"})";
    auto server = serve_response(directory.socket_path("busy.sock"), response_for(body, 503));
    const auto message = require_error(
        [&] { (void)list_from_socket(directory.socket_path("busy.sock"), 1s); },
        simaai::neat::error_codes::kPeripheralDaemonUnavailable, "unexpected HTTP status 503");
    require(message.find("Sentinel reported: too_many_clients.") != std::string::npos,
            "a 503 must carry Sentinel's error string");
    server.finish();
  }
  {
    const std::string body = R"({"error":"peripheral catalog unavailable: /run/x: missing"})";
    auto server =
        serve_response(directory.socket_path("missing-catalog.sock"), response_for(body, 503));
    const auto message = require_error(
        [&] { (void)list_from_socket(directory.socket_path("missing-catalog.sock"), 1s); },
        simaai::neat::error_codes::kPeripheralDaemonUnavailable, "simaai-sentinel.service");
    require(message.find("peripheral catalog unavailable: /run/x: missing") != std::string::npos,
            "a 503 must carry Sentinel's error string");
    server.finish();
  }
  {
    const std::string body = R"({"error":"unknown Sentinel API endpoint"})";
    auto server = serve_response(directory.socket_path("old.sock"), response_for(body, 404));
    const auto message =
        require_error([&] { (void)list_from_socket(directory.socket_path("old.sock"), 1s); },
                      simaai::neat::error_codes::kPeripheralDaemonUnavailable, "too old");
    require(message.find("sima-cli neat install sentinel") != std::string::npos &&
                message.find("unknown Sentinel API endpoint") != std::string::npos,
            "a 404 must ask the user to update Sentinel");
    server.finish();
  }
}

void test_non_object_details_degrade_per_device() {
  TemporaryDirectory directory;
  auto body = ready_catalog();
  body["devices"][1]["lidar"] = nlohmann::json::array({1, 2});
  body["devices"].push_back({{"id", "imu:string"},
                             {"type", "imu"},
                             {"provider", "daemon.imu.future"},
                             {"imu", "not an object"}});
  auto server = serve_response(directory.socket_path(), response_for(body.dump()));
  const auto catalog = list_from_socket(directory.socket_path(), 1s);
  server.finish();

  require(catalog.size() == 3 && catalog[0].camera,
          "invalid details of one type must not hide other devices");
  require(catalog[1].type == "lidar" && catalog[1].details_json == "{}" && !catalog[1].camera,
          "non-object lidar details must yield an empty JSON object");
  require(catalog[2].type == "imu" && catalog[2].details_json == "{}",
          "non-object imu details must yield an empty JSON object");
}

// The installed rules are this template with only @SIMANEAT_VERSION@ replaced,
// inside the "source" string, so the template itself is valid JSON.
nlohmann::json support_rules_template() {
  const auto path = std::filesystem::path(SIMAAI_PERIPHERAL_FIXTURE_DIR) /
                    "../../../../src/peripherals/sentinel-support-rules.json.in";
  std::ifstream input(path);
  require(input.good(), "could not open Sentinel support rules template " + path.string());
  return nlohmann::json::parse(input);
}

void test_support_rules_match_camera_input_defaults() {
  const auto rules = support_rules_template();
  const simaai::neat::CameraInputOptions defaults;
  require(rules.at("format") == 1, "Sentinel support rules must use format 1");
  require(rules.at("source") == "neat-core @SIMANEAT_VERSION@",
          "Sentinel support rules must name their Core source");

  const auto& formats = rules.at("camera").at("formats").at("accept");
  bool format_accepted = false;
  for (const auto& format : formats)
    format_accepted = format_accepted || format == defaults.format;
  require(format_accepted,
          "support rules must accept CameraInput's default format " + defaults.format);

  // Sentinel compares frame rates as ratios, so 60/2 matches 30/1.
  const auto& framerates = rules.at("camera").at("framerates");
  const auto& accepted = framerates.at("accept");
  require(accepted.size() == 1, "support rules must accept exactly CameraInput's default rate");
  const auto num = accepted[0].at("num").get<std::uint64_t>();
  const auto den = accepted[0].at("den").get<std::uint64_t>();
  require(den != 0 && num * defaults.framerate_den == defaults.framerate_num * den,
          "support rules frame rate must match CameraInput's default rate");
  const std::string rate =
      std::to_string(defaults.framerate_num) + "/" + std::to_string(defaults.framerate_den);
  require(framerates.at("reason").get<std::string>().find(rate) != std::string::npos,
          "support rules frame-rate reason must name CameraInput's default rate " + rate);
}

void test_never_ready() {
  TemporaryDirectory directory;
  auto starting = ready_catalog();
  starting["state"] = "starting";
  starting["ready"] = false;
  starting["stale"] = false;
  starting["revision"] = 0;
  starting["last_success_at"] = nullptr;
  starting["issues"] = {{{"provider", "daemon.camera.libcamera"},
                         {"code", "io.backend_unavailable"},
                         {"reason", "camera backend unavailable"},
                         {"retained_last_good", false}}};
  starting["devices"] = nlohmann::json::array();
  auto server = serve_response(directory.socket_path(), response_for(starting.dump()));
  require_error([&] { (void)list_from_socket(directory.socket_path(), 1s); },
                simaai::neat::error_codes::kPeripheralDaemonNotReady, "camera backend unavailable");
  server.finish();
}

} // namespace

int main() {
  try {
    test_success_and_partial_io();
    test_canonical_daemon_fixture();
    test_details_for_any_type();
    test_empty_and_stale_catalogs();
    test_connection_failures();
    test_timeout_and_incomplete_response();
    test_protocol_failures();
    test_non_object_details_degrade_per_device();
    test_never_ready();
    test_support_rules_match_camera_input_defaults();
    std::cout << "unit_peripheral_client_test: PASS\n";
    return 0;
  } catch (const std::exception& error) {
    std::cerr << "unit_peripheral_client_test: FAIL: " << error.what() << '\n';
    return 1;
  }
}
