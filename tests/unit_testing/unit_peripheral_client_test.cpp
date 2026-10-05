#include "peripherals/PeripheralCatalog.h"

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
#include <functional>
#include <iostream>
#include <optional>
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
using simaai::neat::peripherals::CameraMode;
using simaai::neat::peripherals::Catalog;
using simaai::neat::peripherals_internal::list_from_socket;
namespace codes = simaai::neat::error_codes;

// The reasons Core gives for CameraInput's default libcamera profile.
constexpr const char* kBackendReason =
    "CameraInput's default libcamera profile accepts MIPI cameras only. These rules do not "
    "classify raw V4L2 profiles such as MetoakSimor (RAW8 1920x360, selected with "
    "CameraInputOptions.profile and device).";
constexpr const char* kFormatReason =
    "CameraInput's default libcamera profile supports NV12 output only.";
constexpr const char* kFramerateReason =
    "This mode does not advertise the 30/1 frame rate of CameraInput's default libcamera profile.";
constexpr const char* kIspReason = "CameraInput's default libcamera profile requires an ISP output "
                                   "size; this resolution is not one on this board.";

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

// Sentinel's published contract fixture: docs/peripherals/catalog-example.json
// at sima-neat/sentinel commit 7efb980, a DevKit capture (IMX477 MIPI camera,
// Logitech C920 camera and microphone) with the USB camera trimmed to one mode
// per format. tests/assets/peripherals/catalog-example.json is a byte-for-byte
// copy.
json published_example() {
  return json::parse(
#include "sentinel_catalog_example.inc"
  );
}

std::optional<std::string> optional_string(const json& object, const char* name) {
  return object.contains(name) ? std::optional(object[name].get<std::string>()) : std::nullopt;
}

// Every field Core reads from the example reaches the typed catalog unchanged.
void test_example_fields_preserved() {
  const auto body = published_example();
  const auto catalog = list_from(http(body.dump()), 4096);
  require(catalog.revision == body["revision"].get<std::uint64_t>() &&
              catalog.observed_at == body["observed_at"].get<std::string>() &&
              catalog.errors.size() == body["errors"].size() &&
              catalog.size() == body["devices"].size(),
          "catalog fields were lost");
  for (std::size_t index = 0; index < catalog.size(); ++index) {
    const auto& record = body["devices"][index];
    const auto& device = catalog[index];
    require(device.id == record["id"] && device.type == record["type"] &&
                json::parse(device.details_json) == record &&
                device.camera.has_value() == (record["type"] == "camera"),
            "device fields were lost: " + device.id);
    if (!device.camera)
      continue;
    const auto& camera = *device.camera;
    require(camera.camera_name == optional_string(record, "camera_name") &&
                camera.model == optional_string(record, "model") &&
                camera.backend == record["backend"] &&
                camera.modes.size() == record["modes"].size(),
            "camera fields were lost: " + device.id);
    for (std::size_t m = 0; m < camera.modes.size(); ++m) {
      const auto& expected = record["modes"][m];
      const auto& mode = camera.modes[m];
      require(mode.format == expected["format"] && mode.width == expected.value("width", 0U) &&
                  mode.height == expected.value("height", 0U) &&
                  mode.is_range() == expected.contains("size_range"),
              "mode fields were lost: " + device.id + " " + mode.format);
    }
  }
}

// Core's classification and frame rates for the example's devices.
void test_published_example() {
  const auto catalog = list_from(http(published_example().dump()), 4096);
  require(catalog.errors.empty() && catalog.size() == 3, "the example has three devices");

  // MIPI: the NV12 ISP sizes are supported; the ISP lists no frame intervals.
  const auto& mipi = *catalog[0].camera;
  require(mipi.camera_name == "imx477 5-001a" && mipi.model == "imx477" && mipi.backend == "mipi" &&
              mipi.modes.size() == 9,
          "MIPI camera fields were not mapped");
  for (const auto& mode : mipi.modes) {
    const bool nv12 = mode.format == "NV12";
    require(mode.supported == nv12 && mode.reason == (nv12 ? "" : kFormatReason) &&
                mode.framerate_num == 0 && mode.framerate_den == 1 && !mode.is_range(),
            "MIPI mode " + mode.format + " was misclassified");
  }

  // USB: never supported by the default profile; the rate is the fastest interval.
  const auto& usb = *catalog[1].camera;
  require(!usb.camera_name && usb.model == "HD Pro Webcam C920" && usb.backend == "v4l2" &&
              usb.modes.size() == 2,
          "USB camera fields were not mapped");
  for (const auto& mode : usb.modes)
    require(!mode.supported && mode.reason == kBackendReason, "a USB mode was supported");
  const auto rate = [&](std::string_view format, std::uint32_t width, std::uint32_t height) {
    const auto mode = std::find_if(usb.modes.begin(), usb.modes.end(), [&](const CameraMode& m) {
      return m.format == format && m.width == width && m.height == height;
    });
    require(mode != usb.modes.end(), "missing USB mode");
    return std::to_string(mode->framerate_num) + "/" + std::to_string(mode->framerate_den);
  };
  require(rate("MJPG", 1920, 1080) == "30/1" && rate("YUYV", 2560, 1472) == "2/1",
          "USB frame rates must be the fastest advertised interval");

  require(catalog[2].type == "microphone" && !catalog[2].camera,
          "a microphone must have no camera details");
}

// One MIPI camera with one mode; `edit` changes the camera record.
CameraMode classify(const std::function<void(json&)>& edit) {
  json camera = {
      {"type", "camera"},
      {"id", "camera:imx477 5-001a"},
      {"backend", "mipi"},
      {"camera_name", "imx477 5-001a"},
      {"modes",
       json::array(
           {{{"format", "NV12"}, {"width", 1920}, {"height", 1080}, {"isp_output", true}}})}};
  edit(camera);
  const json body = {{"revision", 1},
                     {"observed_at", "2026-10-05T00:00:00Z"},
                     {"devices", json::array({camera})},
                     {"errors", json::array()}};
  return list_from(http(body.dump()), 4096)[0].camera->modes.at(0);
}

json intervals(json list) {
  return json::array({{{"width", 1920}, {"height", 1080}, {"intervals", std::move(list)}}});
}

json discrete(int numerator, int denominator) {
  return {{"type", "discrete"}, {"numerator", numerator}, {"denominator", denominator}};
}

json range(const char* type, std::pair<int, int> minimum, std::pair<int, int> maximum) {
  const auto fraction = [](std::pair<int, int> value) {
    return json{{"numerator", value.first}, {"denominator", value.second}};
  };
  return {{"type", type},
          {"minimum", fraction(minimum)},
          {"maximum", fraction(maximum)},
          {"step", fraction({1, 1000})}};
}

void test_classification_rules() {
  const struct {
    const char* name;
    std::function<void(json&)> edit;
    const char* reason;
    std::uint32_t num;
    std::uint32_t den;
  } cases[] = {
      {"no intervals", [](json&) {}, "", 0, 1},
      {"v4l2 backend", [](json& c) { c["backend"] = "v4l2"; }, kBackendReason, 0, 1},
      {"backend before format",
       [](json& c) {
         c["backend"] = "v4l2";
         c["modes"][0]["format"] = "YUYV";
       },
       kBackendReason, 0, 1},
      {"format", [](json& c) { c["modes"][0]["format"] = "AR24"; }, kFormatReason, 0, 1},
      {"discrete 30",
       [](json& c) { c["modes"][0]["frame_intervals"] = intervals({discrete(1, 30)}); }, "", 30, 1},
      {"discrete 2/60",
       [](json& c) { c["modes"][0]["frame_intervals"] = intervals({discrete(2, 60)}); }, "", 60, 2},
      {"discrete without 30",
       [](json& c) {
         c["modes"][0]["frame_intervals"] = intervals({discrete(1, 15), discrete(1, 60)});
       },
       kFramerateReason, 60, 1},
      {"stepwise containing 30",
       [](json& c) {
         c["modes"][0]["frame_intervals"] = intervals({range("stepwise", {1, 60}, {1, 15})});
       },
       "", 60, 1},
      {"continuous at its bound",
       [](json& c) {
         c["modes"][0]["frame_intervals"] = intervals({range("continuous", {1, 30}, {1, 5})});
       },
       "", 30, 1},
      {"continuous without 30",
       [](json& c) {
         c["modes"][0]["frame_intervals"] = intervals({range("continuous", {1, 120}, {1, 60})});
       },
       kFramerateReason, 120, 1},
      {"rate before ISP",
       [](json& c) {
         c["modes"][0]["frame_intervals"] = intervals({discrete(1, 15)});
         c["modes"][0].erase("isp_output");
       },
       kFramerateReason, 15, 1},
      // Unknown types and inverted ranges are skipped: they neither cover 30/1
      // nor set the rate, but the mode still lists intervals.
      {"unknown interval type skipped",
       [](json& c) {
         json unknown = discrete(1, 30);
         unknown["type"] = "future";
         c["modes"][0]["frame_intervals"] = intervals({unknown, discrete(1, 15)});
       },
       kFramerateReason, 15, 1},
      {"inverted range skipped",
       [](json& c) {
         c["modes"][0]["frame_intervals"] =
             intervals({range("stepwise", {1, 15}, {1, 60}), discrete(1, 10)});
       },
       kFramerateReason, 10, 1},
      {"only skipped intervals",
       [](json& c) {
         c["modes"][0]["frame_intervals"] = intervals({range("continuous", {1, 15}, {1, 60})});
       },
       kFramerateReason, 0, 1},
      {"no ISP output", [](json& c) { c["modes"][0].erase("isp_output"); }, kIspReason, 0, 1},
      {"ISP output false", [](json& c) { c["modes"][0]["isp_output"] = false; }, kIspReason, 0, 1},
  };
  for (const auto& test_case : cases) {
    const auto mode = classify(test_case.edit);
    require(mode.supported == (std::string_view(test_case.reason).empty()) &&
                mode.reason == test_case.reason && mode.framerate_num == test_case.num &&
                mode.framerate_den == test_case.den,
            std::string("classification case failed: ") + test_case.name + " (got '" + mode.reason +
                "', " + std::to_string(mode.framerate_num) + "/" +
                std::to_string(mode.framerate_den) + ")");
  }

  // Size ranges are parsed for any backend.
  const auto ranged = classify([](json& c) {
    c["backend"] = "v4l2";
    c["modes"][0].erase("width");
    c["modes"][0].erase("height");
    c["modes"][0]["size_range"] = {{"type", "stepwise"}, {"min_width", 640},   {"min_height", 480},
                                   {"max_width", 1920},  {"max_height", 1080}, {"step_width", 16},
                                   {"step_height", 8}};
  });
  require(ranged.is_range() && ranged.size_range->min_width == 640 &&
              ranged.size_range->max_height == 1080 && ranged.size_range->step_width == 16 &&
              ranged.size_range->step_height == 8 && ranged.reason == kBackendReason,
          "size ranges were not mapped");
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

void test_errors_and_first_scan() {
  auto failed = published_example();
  failed["errors"] = json::array({{{"provider", "camera.v4l2"},
                                   {"code", "io.permission_denied"},
                                   {"reason", "permission denied"}}});
  const auto catalog = list_from(http(failed.dump()), 4096);
  require(catalog.errors.size() == 1 && catalog.errors[0].provider == "camera.v4l2" &&
              catalog.errors[0].code == "io.permission_denied" &&
              catalog.errors[0].reason == "permission denied" && catalog.size() == 3,
          "provider errors and retained devices must be returned together");

  const auto first = list_from(
      http(R"({"revision": 1791164913635, "observed_at": null, "devices": [], "errors": []})"));
  require(first.empty() && !first.observed_at && first.revision == 1791164913635,
          "a catalog before the first scan must succeed with observed_at unset");
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
    auto body = published_example();
    edit(body);
    return http(body.dump());
  };
  const auto mode = [](json& body) -> json& { return body["devices"][1]["modes"][0]; };
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
      {"HTTP/1.1 200 OK\r\nContent-Length: 2\r\ncontent-length: 2\r\n\r\n{}", codes::kIoParse,
       "duplicate Content-Length headers"},
      {"HTTP/1.1 200 OK\r\nContent-Length: +2\r\n\r\n{}", codes::kIoParse,
       "Content-Length is not a non-negative decimal integer"},
      {"HTTP/1.1 200 OK\r\nContent-Length: 2\r\nTRANSFER-ENCODING: chunked\r\n\r\n{}",
       codes::kIoParse, "chunked HTTP responses"},
      {edited([](json& body) { body["observed_at"] = 1; }), codes::kIoParse,
       "field 'observed_at' must be a string or null"},
      {edited([](json& body) { body.erase("observed_at"); }), codes::kIoParse,
       "missing required field 'observed_at'"},
      {edited([](json& body) { body["errors"] = json::array({{{"code", "x"}, {"reason", "y"}}}); }),
       codes::kIoParse, "missing required field 'provider'"},
      {edited([](json& body) { body["devices"][1] = body["devices"][0]; }), codes::kIoParse,
       "duplicate id"},
      {edited([](json& body) { body["devices"][0]["model"] = false; }), codes::kIoParse,
       "field 'model' must be a string when present"},
      {edited([](json& body) { body["devices"][0]["modes"] = json::object(); }), codes::kIoParse,
       "field 'modes' must be an array"},
      {edited([&](json& body) { mode(body)["frame_intervals"] = json::object(); }), codes::kIoParse,
       "field 'frame_intervals' must be an array"},
      {edited([](json& body) { body["devices"][0]["modes"][0]["isp_output"] = 1; }),
       codes::kIoParse, "field 'isp_output' must be a boolean when present"},
      {http(R"({"error":"peripheral discovery is not running"})", 503),
       codes::kPeripheralDaemonUnavailable,
       "Sentinel reported: peripheral discovery is not running."},
      {http(R"({"error":"unknown Sentinel API endpoint"})", 404),
       codes::kPeripheralDaemonUnavailable,
       "Sentinel reported: unknown Sentinel API endpoint. Update Sentinel with `sima-cli neat "
       "install sentinel`"},
  };
  for (const auto& test_case : cases)
    require_error([&] { (void)list_from(test_case.response, 4096); }, test_case.code,
                  test_case.fragment);

  // A skipped interval keeps the rest of the device and catalog.
  auto skipped = published_example();
  mode(skipped)["frame_intervals"][0]["intervals"][0]["type"] = "future";
  mode(skipped)["frame_intervals"][0]["intervals"].push_back(range("stepwise", {1, 5}, {1, 30}));
  const auto kept = list_from(http(skipped.dump()), 4096);
  require(kept.size() == 3 && kept[1].camera->modes.size() == 2 &&
              kept[1].camera->modes[0].framerate_num == 24 &&
              json::parse(kept[1].details_json) == skipped["devices"][1],
          "a skipped frame interval must not drop the device or catalog");

  // Header names compare case-insensitively.
  const std::string body = published_example().dump();
  require(list_from("HTTP/1.1 200 OK\r\nCONTENT-length: " + std::to_string(body.size()) +
                        "\r\n\r\n" + body,
                    4096)
                  .size() == 3,
          "Content-Length must match case-insensitively");
}

} // namespace

int main() {
  try {
    test_example_fields_preserved();
    test_published_example();
    test_classification_rules();
    test_catalog_container_api();
    test_errors_and_first_scan();
    test_connection_failures();
    test_protocol_failures();
    std::cout << "unit_peripheral_client_test: PASS\n";
    return 0;
  } catch (const std::exception& error) {
    std::cerr << "unit_peripheral_client_test: FAIL: " << error.what() << '\n';
    return 1;
  }
}
