#include "peripherals/PeripheralCatalog.h"

#include "nodes/io/CameraInput.h"
#include "peripherals/internal/PeripheralClient.h"
#include "pipeline/ErrorCodes.h"
#include "pipeline/NeatError.h"

#include <nlohmann/json.hpp>

#include <sys/socket.h>
#include <sys/stat.h>
#include <sys/un.h>
#include <unistd.h>

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <exception>
#include <filesystem>
#include <iostream>
#include <stdexcept>
#include <string>
#include <string_view>
#include <thread>
#include <type_traits>
#include <utility>
#include <vector>

namespace {

using namespace std::chrono_literals;
using json = nlohmann::json;
using simaai::neat::peripherals::Catalog;
using simaai::neat::peripherals_internal::list_from_socket;
namespace codes = simaai::neat::error_codes;

void require(bool condition, const std::string& message) {
  if (!condition)
    throw std::runtime_error(message);
}

class TemporaryDirectory {
public:
  TemporaryDirectory() {
    char pattern[] = "/tmp/neat-peripherals-XXXXXX";
    require(::mkdtemp(pattern) != nullptr, "mkdtemp failed");
    path = pattern;
  }
  ~TemporaryDirectory() {
    std::error_code ignored;
    std::filesystem::permissions(path, std::filesystem::perms::owner_all,
                                 std::filesystem::perm_options::add, ignored);
    std::filesystem::remove_all(path, ignored);
  }
  std::string socket(const char* name = "api.sock") const {
    return path + "/" + name;
  }

  std::string path;
};

int bound_socket(const std::string& path) {
  const int fd = ::socket(AF_UNIX, SOCK_STREAM | SOCK_CLOEXEC, 0);
  sockaddr_un address{};
  address.sun_family = AF_UNIX;
  std::strncpy(address.sun_path, path.c_str(), sizeof(address.sun_path) - 1);
  require(fd >= 0 && ::bind(fd, reinterpret_cast<sockaddr*>(&address), sizeof(address)) == 0,
          "failed to bind " + path);
  return fd;
}

// Serves `response` to one GET /v1/peripherals, `chunk` bytes per write, and
// reads it through a client limited to 3-byte writes.
Catalog list_from(const std::string& response, std::size_t chunk = 1) {
  TemporaryDirectory directory;
  const int listener = bound_socket(directory.socket());
  require(::listen(listener, 1) == 0, "listen failed");
  std::exception_ptr server_error;
  std::thread server([&] {
    try {
      const int client = ::accept4(listener, nullptr, nullptr, SOCK_CLOEXEC);
      require(client >= 0, "accept failed");
      std::string request;
      char byte = 0;
      while (request.find("\r\n\r\n") == std::string::npos && ::recv(client, &byte, 1, 0) == 1)
        request.push_back(byte);
      require(request.starts_with("GET /v1/peripherals HTTP/1.1\r\n"), "unexpected request");
      for (std::size_t offset = 0; offset < response.size(); offset += chunk)
        (void)::send(client, response.data() + offset, std::min(chunk, response.size() - offset),
                     MSG_NOSIGNAL);
      ::close(client);
    } catch (...) {
      server_error = std::current_exception();
    }
  });
  std::exception_ptr client_error;
  Catalog catalog;
  try {
    catalog = list_from_socket(directory.socket(), 1s, 3);
  } catch (...) {
    client_error = std::current_exception();
  }
  server.join();
  ::close(listener);
  if (server_error)
    std::rethrow_exception(server_error);
  if (client_error)
    std::rethrow_exception(client_error);
  return catalog;
}

std::string http(const std::string& body, int status = 200) {
  return "HTTP/1.1 " + std::to_string(status) + " Status\r\nContent-Type: application/json\r\n" +
         "Content-Length: " + std::to_string(body.size()) + "\r\n\r\n" + body;
}

template <typename Function>
void require_error(Function&& function, const char* code, std::string_view fragment) {
  try {
    function();
  } catch (const simaai::neat::NeatError& error) {
    const std::string message = error.what();
    require(error.report().error_code == code && message.find(fragment) != std::string::npos,
            "expected " + std::string(code) + " with '" + std::string(fragment) + "', got " +
                message);
    return;
  }
  throw std::runtime_error("expected NeatError " + std::string(code));
}

// Sentinel's `changes`, `support`, and provider fields such as `connection`
// are not part of the typed v1 contract and must be tolerated.
json base_catalog() {
  return json::parse(R"({
    "schema_version": 1, "instance_id": "daemon-a", "state": "ready", "ready": true,
    "stale": false, "revision": 7, "sequence": 11, "scan_sequence": 13,
    "last_success_at": "2026-10-01T01:02:03Z", "last_attempt_at": "2026-10-01T01:02:04Z",
    "error": null, "issues": [], "changes": [], "support": {"state": "applied"},
    "devices": [
      {"id": "camera:imx477 5-001a", "type": "camera", "provider": "daemon.camera.mipi",
       "camera": {"camera_name": "imx477 5-001a", "model": "imx477", "backend": "mipi",
                  "connection": "mipi-csi2", "modes": [
         {"format": "NV12", "width": 1920, "height": 1080, "framerate_num": 30,
          "framerate_den": 1, "supported": true, "reason": "", "isp_output": true},
         {"format": "NV12", "size_range": {"min_width": 640, "min_height": 480,
          "max_width": 1920, "max_height": 1080, "step_width": 16, "step_height": 8},
          "framerate_num": 30, "framerate_den": 1, "supported": false,
          "reason": "range is advisory"}]}},
      {"id": "lidar:1", "type": "lidar", "provider": "daemon.lidar", "lidar": {"points": 42}}]
  })");
}

void test_success() {
  const auto body = base_catalog();
  const auto catalog = list_from(http(body.dump()));
  require(catalog.instance_id == "daemon-a" && catalog.state == "ready" && !catalog.stale &&
              catalog.revision == 7 && catalog.sequence == 11 && catalog.scan_sequence == 13 &&
              catalog.last_attempt_at == "2026-10-01T01:02:04Z" && !catalog.error &&
              catalog.issues.empty() && catalog.size() == 2 && catalog.begin()->id == catalog[0].id,
          "catalog metadata was not preserved");
  const auto& camera = *catalog[0].camera;
  const auto& range = *camera.modes[1].size_range;
  require(camera.camera_name == "imx477 5-001a" && camera.model == "imx477" &&
              camera.backend == "mipi" && camera.modes.size() == 2 &&
              camera.modes[0].width == 1920 && camera.modes[0].height == 1080 &&
              camera.modes[0].framerate_num == 30 && camera.modes[0].supported &&
              camera.modes[1].is_range() && range.min_width == 640 && range.max_height == 1080 &&
              range.step_width == 16 && range.step_height == 8 && !camera.modes[1].supported &&
              camera.modes[1].reason == "range is advisory",
          "camera details were not mapped exactly");
  require(json::parse(catalog[0].details_json) == body["devices"][0]["camera"],
          "camera details_json must preserve every daemon field");
}

// Catalog is a mutable and const container over `devices`.
void test_catalog_container_api() {
  using simaai::neat::peripherals::Peripheral;
  static_assert(std::is_same_v<Catalog::iterator, std::vector<Peripheral>::iterator>);
  static_assert(std::is_same_v<Catalog::const_iterator, std::vector<Peripheral>::const_iterator>);
  static_assert(std::is_same_v<decltype(std::declval<Catalog&>()[0]), Peripheral&>);
  static_assert(std::is_same_v<decltype(std::declval<const Catalog&>()[0]), const Peripheral&>);

  Catalog catalog;
  require(catalog.empty() && catalog.begin() == catalog.end(), "a new catalog must be empty");
  catalog.devices.resize(2);
  for (Peripheral& device : catalog)
    device.type = "camera";
  catalog[1].id = "b";
  Catalog::iterator first = catalog.begin();
  first->id = "a";
  const Catalog& view = catalog;
  std::string ids;
  for (Catalog::const_iterator it = catalog.cbegin(); it != catalog.cend(); ++it)
    ids += it->id + it->type;
  require(ids == "acamerabcamera" && view.begin() == catalog.cbegin() &&
              view.end() == catalog.cend() && view.size() == 2 && &view[1] == &catalog.devices[1],
          "catalog iterators and indexing must address devices");
}

void test_details_for_any_type() {
  auto body = base_catalog();
  const auto microphone = json::parse(R"({"channels": 2, "nested": {"a": [1, 2]},
                                          "gain_db": -3.5, "big": 18446744073709551615})");
  body["devices"][1]["lidar"] = json::array({1, 2});
  body["devices"].push_back(
      {{"id", "mic:1"}, {"type", "microphone"}, {"provider", "alsa"}, {"microphone", microphone}});
  body["devices"].push_back({{"id", "imu:1"}, {"type", "imu"}, {"provider", "imu"}});
  body["devices"].push_back(
      {{"id", "radar:1"}, {"type", "radar"}, {"provider", "radar"}, {"radar", nullptr}});
  const auto catalog = list_from(http(body.dump()), 64);

  // Non-object, absent, and null details degrade to "{}" for that device only.
  const json expected[] = {body["devices"][0]["camera"], json::object(), microphone, json::object(),
                           json::object()};
  require(catalog.size() == 5, "every peripheral type must remain in the catalog");
  for (std::size_t index = 0; index < catalog.size(); ++index)
    require(json::parse(catalog[index].details_json) == expected[index] &&
                catalog[index].camera.has_value() == (index == 0),
            "details_json mismatch for " + catalog[index].id);
}

void test_stale_and_empty_catalogs() {
  auto stale = base_catalog();
  stale["state"] = "degraded";
  stale["stale"] = true;
  stale["error"] = {{"code", "peripherals.discovery_failed"}, {"reason", "camera scan failed"}};
  stale["issues"] = json::array({{{"provider", "daemon.camera.mipi"},
                                  {"code", "io.permission_denied"},
                                  {"reason", "permission denied"},
                                  {"retained_last_good", true}}});
  const auto catalog = list_from(http(stale.dump()), 64);
  require(catalog.stale && catalog.error->code == "peripherals.discovery_failed" &&
              catalog.issues.size() == 1 && catalog.issues[0].retained_last_good &&
              catalog.size() == 2,
          "stale last-good data and diagnostics must be returned together");

  auto empty = base_catalog();
  empty["devices"] = json::array();
  require(list_from(http(empty.dump()), 64).empty(), "a ready empty catalog must succeed");
}

void test_connection_failures() {
  TemporaryDirectory directory;
  require_error([&] { (void)list_from_socket(directory.socket("missing.sock"), 100ms); },
                codes::kPeripheralDaemonUnavailable, "sima-cli neat install sentinel");

  ::close(bound_socket(directory.socket("refused.sock")));
  require_error([&] { (void)list_from_socket(directory.socket("refused.sock"), 100ms); },
                codes::kPeripheralDaemonUnavailable, "simaai-sentinel.service");

  // A daemon that accepts but never answers must hit the absolute deadline.
  const int silent = bound_socket(directory.socket("silent.sock"));
  require(::listen(silent, 1) == 0, "listen failed");
  const auto start = std::chrono::steady_clock::now();
  require_error([&] { (void)list_from_socket(directory.socket("silent.sock"), 40ms); },
                codes::kPeripheralDaemonTimeout, "Timed out");
  require(std::chrono::steady_clock::now() - start < 120ms, "the deadline was not honored");
  ::close(silent);

  require_error([&] { (void)list_from_socket(directory.socket("missing.sock"), 100ms, 0); },
                codes::kIoOpen, "chunk limit is invalid");

  require(::chmod(directory.path.c_str(), 0000) == 0, "chmod failed");
  require_error([&] { (void)list_from_socket(directory.socket("denied.sock"), 100ms); },
                codes::kPermissionDenied, "/run/simaai-sentinel/api.sock");
}

void test_protocol_failures() {
  const auto edited = [](auto edit) {
    auto body = base_catalog();
    edit(body);
    return http(body.dump());
  };
  const struct {
    std::string response;
    const char* code;
    const char* fragment;
  } cases[] = {
      {http("{"), codes::kIoParse, "JSON parsing failed"},
      {"HTTP/1.1 200 OK\r\nContent-Length: 12\r\n\r\n{}", codes::kIoParse, "response completed"},
      {"HTTP/1.1 200 OK\r\nContent-Length: " +
           std::to_string(simaai::neat::peripherals_internal::kMaximumResponseBytes + 1) +
           "\r\n\r\n",
       codes::kResponseTooLarge, "catalog response exceeded the 4 MiB"},
      {http(R"({"error":"response_too_large"})", 500), codes::kResponseTooLarge,
       "catalog exceeded the 4 MiB"},
      {"HTTP/1.1 200 OK\r\nContent-Length: 2\r\ncontent-length: 2\r\n\r\n{}", codes::kIoParse,
       "duplicate Content-Length headers"},
      {"HTTP/1.1 200 OK\r\nContent-Length: +2\r\n\r\n{}", codes::kIoParse,
       "Content-Length is not a non-negative decimal integer"},
      {"HTTP/1.1 200 OK\r\nContent-Length: 2\r\nTRANSFER-ENCODING: chunked\r\n\r\n{}",
       codes::kIoParse, "chunked HTTP responses"},
      {edited([](json& body) { body["last_attempt_at"] = 1; }), codes::kIoParse,
       "field 'last_attempt_at' must be a string or null"},
      {edited([](json& body) { body["last_success_at"] = ""; }), codes::kIoParse,
       "field 'last_success_at' must not be empty"},
      {edited([](json& body) { body.erase("last_success_at"); }), codes::kIoParse,
       "missing required field 'last_success_at'"},
      {edited([](json& body) { body["devices"][0]["camera"]["model"] = false; }), codes::kIoParse,
       "field 'model' must be a string when present"},
      {edited([](json& body) { body["devices"][0]["camera"]["modes"] = json::object(); }),
       codes::kIoParse, "camera field 'modes' must be an array"},
      {edited([](json& body) { body["schema_version"] = 2; }), codes::kRuntimeAbiMismatch,
       "matching Core"},
      {edited([](json& body) { body["devices"][0].erase("provider"); }), codes::kIoParse,
       "'provider'"},
      {edited([](json& body) { body["devices"][0]["camera"] = json::array(); }), codes::kIoParse,
       "camera details must be an object"},
      {edited([](json& body) {
         body["state"] = "starting";
         body["ready"] = false;
         body["error"] = {{"code", "io.backend_unavailable"}, {"reason", "no camera backend"}};
       }),
       codes::kPeripheralDaemonNotReady, "no camera backend"},
      {http(R"({"error":"too_many_clients"})", 503), codes::kPeripheralDaemonUnavailable,
       "Sentinel reported: too_many_clients."},
      {http(R"({"error":"unknown endpoint"})", 404), codes::kPeripheralDaemonUnavailable,
       "Sentinel reported: unknown endpoint. Update Sentinel with `sima-cli neat install "
       "sentinel`"},
  };
  for (const auto& test_case : cases)
    require_error([&] { (void)list_from(test_case.response, 64); }, test_case.code,
                  test_case.fragment);

  // Header names compare case-insensitively.
  const std::string body = base_catalog().dump();
  require(list_from("HTTP/1.1 200 OK\r\nCONTENT-length: " + std::to_string(body.size()) +
                    "\r\n\r\n" + body)
                  .size() == 2,
          "Content-Length must match case-insensitively");
}

// Sentinel classifies camera modes with the rules Core installs; keep them in
// step with CameraInput's defaults and scoped to its default profile.
void test_support_rules_match_camera_input_defaults() {
  const auto rules = json::parse(
#include "sentinel_support_rules.inc"
  );
  const auto& accepted_formats = rules.at("camera").at("formats").at("accept");
  const auto& framerates = rules.at("camera").at("framerates");
  const auto& rate = framerates.at("accept");
  const simaai::neat::CameraInputOptions defaults;
  const std::string default_rate =
      std::to_string(defaults.framerate_num) + "/" + std::to_string(defaults.framerate_den);
  require(rules.at("format") == 1 && rules.at("source") == "neat-core @SIMANEAT_VERSION@" &&
              std::find(accepted_formats.begin(), accepted_formats.end(), defaults.format) !=
                  accepted_formats.end() &&
              rate.size() == 1 &&
              rate[0].at("num").get<std::uint64_t>() * defaults.framerate_den ==
                  defaults.framerate_num * rate[0].at("den").get<std::uint64_t>() &&
              framerates.at("reason").get<std::string>().find(default_rate) != std::string::npos,
          "Sentinel support rules drifted from CameraInput's defaults");

  // The rules classify only the default libcamera profile. Format 1 cannot
  // express a second profile, so every reason names that scope and the backend
  // reason says the MetoakSimor raw V4L2 profile is not classified.
  const auto& camera = rules.at("camera");
  require(defaults.profile == simaai::neat::CameraProfile::Default &&
              camera.at("backends").at("accept") == json::array({"mipi"}) &&
              camera.at("formats").at("accept") == json::array({"NV12"}),
          "Sentinel support rules must describe only the default libcamera profile");
  for (const auto* rule : {"backends", "formats", "framerates", "isp_output"}) {
    require(camera.at(rule).at("reason").get<std::string>().find("default libcamera profile") !=
                std::string::npos,
            std::string("support rule reason must name the default profile: ") + rule);
  }
  const auto backend = camera.at("backends").at("reason").get<std::string>();
  require(backend.find("MetoakSimor") != std::string::npos &&
              backend.find("do not classify") != std::string::npos,
          "backend reason must say raw V4L2 profiles are not classified");
}

} // namespace

int main() {
  try {
    test_success();
    test_catalog_container_api();
    test_details_for_any_type();
    test_stale_and_empty_catalogs();
    test_connection_failures();
    test_protocol_failures();
    test_support_rules_match_camera_input_defaults();
    std::cout << "unit_peripheral_client_test: PASS\n";
    return 0;
  } catch (const std::exception& error) {
    std::cerr << "unit_peripheral_client_test: FAIL: " << error.what() << '\n';
    return 1;
  }
}
