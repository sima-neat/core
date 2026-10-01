#include "PeripheralApiServer.h"
#include "PeripheralCatalog.h"
#include "PeripheralCatalogManager.h"
#include "PeripheralProtocol.h"
#include "providers/CameraProvider.h"
#include "test_main.h"
#include "test_utils.h"

#include <nlohmann/json.hpp>

#include <sys/socket.h>
#include <sys/un.h>
#include <unistd.h>

#include <atomic>
#include <cerrno>
#include <chrono>
#include <cstdint>
#include <cstring>
#include <future>
#include <stdexcept>
#include <string>
#include <thread>
#include <utility>
#include <vector>

namespace {

using simaai::neat::peripherals_internal::camera_records;
using simaai::neat::peripherals_internal::CatalogIssue;
using simaai::neat::peripherals_internal::encode_peripheral_record;
using simaai::neat::peripherals_internal::PeripheralApiServer;
using simaai::neat::peripherals_internal::PeripheralCatalog;
using simaai::neat::peripherals_internal::PeripheralCatalogManager;
using simaai::neat::peripherals_internal::PeripheralProvider;
using simaai::neat::peripherals_internal::PeripheralRecord;

PeripheralRecord camera(std::string name, std::uint32_t width = 1920) {
  PeripheralRecord record;
  record.id = "camera:" + name;
  record.type = "camera";
  record.provider = "test.camera";
  record.details = {
      {"camera_name", std::move(name)},
      {"model", "same-model"},
      {"modes", {{{"format", "NV12"}, {"width", width}, {"height", 1080}}}},
  };
  return record;
}

PeripheralRecord microphone(std::string name) {
  PeripheralRecord record;
  record.id = "microphone:" + name;
  record.type = "microphone";
  record.provider = "test.microphone";
  record.details = {{"name", std::move(name)}, {"capture", true}};
  return record;
}

struct HttpResponse {
  int status = 0;
  nlohmann::json body;
};

int connect_client(const std::string& socket_path) {
  const int fd = ::socket(AF_UNIX, SOCK_STREAM | SOCK_CLOEXEC, 0);
  require(fd >= 0, "failed to create test Unix socket");
  sockaddr_un address{};
  address.sun_family = AF_UNIX;
  require(socket_path.size() < sizeof(address.sun_path), "test socket path is too long");
  std::memcpy(address.sun_path, socket_path.c_str(), socket_path.size() + 1);
  require(::connect(fd, reinterpret_cast<sockaddr*>(&address), sizeof(address)) == 0,
          "failed to connect to peripheral API test socket");
  return fd;
}

HttpResponse raw_request(const std::string& socket_path, const std::string& raw) {
  const int fd = connect_client(socket_path);
  timeval timeout{5, 0};
  require(::setsockopt(fd, SOL_SOCKET, SO_RCVTIMEO, &timeout, sizeof(timeout)) == 0,
          "failed to bound peripheral API test response time");
  std::size_t sent = 0;
  while (sent < raw.size()) {
    const ssize_t count = ::send(fd, raw.data() + sent, raw.size() - sent, MSG_NOSIGNAL);
    if (count < 0 && errno == EINTR)
      continue;
    if (count < 0 && (errno == EPIPE || errno == ECONNRESET))
      break;
    require(count > 0, "failed to send peripheral API test request");
    sent += static_cast<std::size_t>(count);
  }

  std::string response;
  char buffer[1024];
  while (true) {
    const ssize_t received = ::recv(fd, buffer, sizeof(buffer), 0);
    if (received == 0)
      break;
    require(received > 0, "failed to read peripheral API test response");
    response.append(buffer, static_cast<std::size_t>(received));
  }
  ::close(fd);
  const std::size_t first_space = response.find(' ');
  require(first_space != std::string::npos, "HTTP response did not contain a status");
  HttpResponse parsed;
  parsed.status = std::stoi(response.substr(first_space + 1, 3));
  const std::size_t body = response.find("\r\n\r\n");
  require(body != std::string::npos, "HTTP response did not contain a body");
  parsed.body = nlohmann::json::parse(response.substr(body + 4));
  return parsed;
}

HttpResponse request(const std::string& socket_path, const std::string& method,
                     const std::string& target) {
  return raw_request(socket_path, method + " " + target +
                                      " HTTP/1.1\r\nHost: localhost\r\nConnection: close\r\n\r\n");
}

void wait_for_status(const std::string& socket_path, int expected_status) {
  const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(3);
  int last_status = 0;
  do {
    last_status = request(socket_path, "GET", "/v1/health").status;
    if (last_status == expected_status)
      return;
    std::this_thread::yield();
  } while (std::chrono::steady_clock::now() < deadline);
  throw std::runtime_error("timed out waiting for peripheral API status " +
                           std::to_string(expected_status) + " at " + socket_path +
                           "; last status was " + std::to_string(last_status));
}

void wait_for_scan_sequence(PeripheralCatalog& catalog, std::uint64_t expected) {
  const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(3);
  do {
    if (catalog.scan_sequence() >= expected)
      return;
    std::this_thread::yield();
  } while (std::chrono::steady_clock::now() < deadline);
  throw std::runtime_error("timed out waiting for peripheral scan sequence " +
                           std::to_string(expected));
}

} // namespace

RUN_TEST(
    "unit_peripheral_daemon_test", ([] {
      simaai::neat::peripherals_internal::CameraMode discrete_mode;
      discrete_mode.format = "NV12";
      discrete_mode.width = 1920;
      discrete_mode.height = 1080;
      discrete_mode.framerate_num = 60;
      discrete_mode.framerate_den = 2;
      discrete_mode.supported = true;
      simaai::neat::peripherals_internal::CameraMode range_mode;
      range_mode.format = "YUY2";
      range_mode.size_range = simaai::neat::peripherals_internal::CameraSizeRange{
          .min_width = 640,
          .min_height = 480,
          .max_width = 1920,
          .max_height = 1080,
          .step_width = 16,
          .step_height = 8,
      };
      range_mode.supported = false;
      range_mode.reason = "range is advisory";
      simaai::neat::peripherals_internal::CameraInfo first_camera{
          .name = "camera-path-a",
          .model = "same-model",
          .modes = {discrete_mode, range_mode},
      };
      simaai::neat::peripherals_internal::CameraInfo second_camera{
          .name = "camera-path-b",
          .model = "",
          .modes = {discrete_mode},
      };
      const auto adapted = camera_records({first_camera, second_camera});
      require(adapted.size() == 2 && adapted[0].id == "camera:camera-path-a" &&
                  adapted[1].id == "camera:camera-path-b",
              "camera adapter must preserve distinct exact camera identities");
      const auto first_json = encode_peripheral_record(adapted[0]);
      const auto second_json = encode_peripheral_record(adapted[1]);
      require(first_json["camera"]["backend"] == "libcamera" &&
                  first_json["camera"]["model"] == "same-model" &&
                  !second_json["camera"].contains("model"),
              "camera adapter must preserve backend and optional model fields");
      require(first_json["camera"]["modes"][0]["width"] == 1920 &&
                  first_json["camera"]["modes"][0]["height"] == 1080 &&
                  first_json["camera"]["modes"][0]["framerate_num"] == 60 &&
                  first_json["camera"]["modes"][0]["framerate_den"] == 2 &&
                  first_json["camera"]["modes"][0]["supported"],
              "camera adapter must preserve a supported discrete mode");
      require(first_json["camera"]["modes"][1]["size_range"]["step_width"] == 16 &&
                  first_json["camera"]["modes"][1]["size_range"]["step_height"] == 8 &&
                  !first_json["camera"]["modes"][1]["supported"] &&
                  first_json["camera"]["modes"][1]["reason"] == "range is advisory",
              "camera adapter must preserve ranges and rejection reasons");

      PeripheralCatalog catalog("instance-a", 8);
      catalog.apply_success({camera("same-serial-a"), camera("same-serial-b")});
      auto snapshot = catalog.catalog_json();
      require(snapshot["state"] == "ready", "initial catalog must become ready");
      require(snapshot["revision"] == 1, "initial catalog revision must be one");
      require(snapshot["scan_sequence"] == 1, "initial catalog scan sequence must be one");
      require(snapshot["devices"].size() == 2,
              "same-model devices with distinct identities must both remain");
      require(catalog.events_json(0, std::chrono::milliseconds(0), "instance-a")["events"].empty(),
              "initial discovery must not synthesize add events");

      auto without_model = camera("model-not-reported");
      without_model.details.erase("model");
      require(!encode_peripheral_record(without_model)["camera"].contains("model"),
              "missing optional fields must remain omitted");
      bool duplicate_rejected = false;
      try {
        catalog.apply_success({camera("duplicate"), camera("duplicate")});
      } catch (const std::invalid_argument&) {
        duplicate_rejected = true;
      }
      require(duplicate_rejected, "duplicate stable identities must be rejected");

      catalog.apply_success({camera("same-serial-a", 2048), camera("same-serial-c")});
      const auto changes = catalog.events_json(0, std::chrono::milliseconds(0), "instance-a");
      require(changes["revision"] == 2, "one changed scan must advance one revision");
      require(changes["events"].size() == 3,
              "changed, removed, and added devices must each emit an event");
      for (const auto& event : changes["events"])
        require(event["revision"] == 2, "one scan's events must share a revision");
      require(changes["events"][0]["kind"] == "changed", "changed event ordering mismatch");
      require(changes["events"][1]["kind"] == "removed", "removed event ordering mismatch");
      require(changes["events"][2]["kind"] == "added", "added event ordering mismatch");

      catalog.apply_scan_error("io.permission_denied", "permission denied");
      snapshot = catalog.catalog_json();
      require(snapshot["state"] == "degraded" && snapshot["stale"],
              "errors must mark the last successful snapshot stale");
      require(snapshot["scan_sequence"] == 3, "failed scans must advance the scan sequence");
      require(snapshot["devices"].size() == 2, "errors must preserve the last successful snapshot");
      const auto error_page = catalog.events_json(3, std::chrono::milliseconds(0), "instance-a");
      require(error_page["events"].size() == 1 && error_page["events"][0]["kind"] == "error",
              "discovery failures must emit structured error events");

      auto recovery_waiter = std::async(std::launch::async, [&] {
        return catalog.events_json(4, std::chrono::seconds(2), "instance-a");
      });
      catalog.apply_success({camera("same-serial-a", 2048), camera("same-serial-c")});
      snapshot = catalog.catalog_json();
      require(snapshot["state"] == "ready" && !snapshot["stale"],
              "a successful scan must clear degraded state");
      require(snapshot["revision"] == 2,
              "recovery without a device change must not advance the revision");
      require(snapshot["scan_sequence"] == 4, "successful recovery must advance the scan sequence");
      const auto recovery_page = recovery_waiter.get();
      require(recovery_page["events"].size() == 1 &&
                  recovery_page["events"][0]["kind"] == "recovered",
              "recovery must wake event clients even when devices are unchanged");
      require(
          catalog.events_json(5, std::chrono::milliseconds(0), "old-instance")["resync_required"],
          "daemon instance changes must require a snapshot resync");
      require(
          catalog.events_json(99, std::chrono::milliseconds(0), "instance-a")["resync_required"],
          "future cursors must require a snapshot resync");

      PeripheralCatalog startup_recovery("startup-recovery", 8);
      startup_recovery.apply_scan_error("io.backend_unavailable", "backend unavailable");
      startup_recovery.apply_success({camera("recovered-camera")});
      const auto startup_recovery_events =
          startup_recovery.events_json(1, std::chrono::milliseconds(0), "startup-recovery");
      require(startup_recovery.catalog_json()["revision"] == 1 &&
                  startup_recovery_events["events"].size() == 1 &&
                  startup_recovery_events["events"][0]["kind"] == "recovered",
              "startup recovery must become ready and publish a recovery event");

      PeripheralCatalog monitor_failure("monitor-failure", 8);
      monitor_failure.apply_error("peripherals.monitor_failed", "udev unavailable");
      const auto monitor_failure_health = monitor_failure.health_json();
      require(monitor_failure_health["scan_sequence"] == 0 &&
                  monitor_failure_health["last_attempt_at"].is_null(),
              "monitor failures must not satisfy a discovery completion token");

      PeripheralCatalog partial_after_error("partial-after-error", 8);
      partial_after_error.apply_scan_error("peripherals.discovery_failed", "all providers failed");
      partial_after_error.apply_success(
          {camera("partially-recovered-camera")},
          {CatalogIssue{"test.microphone", "peripherals.discovery_failed", "audio unavailable",
                        false}});
      const auto partial_after_error_events =
          partial_after_error.events_json(1, std::chrono::milliseconds(0), "partial-after-error");
      require(partial_after_error.catalog_json()["state"] == "degraded" &&
                  partial_after_error_events["events"].size() == 1 &&
                  partial_after_error_events["events"][0]["kind"] == "error",
              "a partial recovery must remain degraded and must not emit recovered");

      PeripheralCatalog isolated_catalog("isolated-instance", 16);
      bool audio_fails = false;
      PeripheralCatalogManager isolated_manager(
          isolated_catalog,
          {
              {"test.camera",
               {"media", "video4linux"},
               [] { return std::vector{camera("isolated-camera")}; }},
              {"test.microphone",
               {"sound"},
               [&] {
                 if (audio_fails)
                   throw std::runtime_error("ALSA enumeration failed");
                 return std::vector{microphone("isolated-microphone")};
               }},
          },
          0);
      isolated_manager.initial_scan();
      require(isolated_catalog.catalog_json()["devices"].size() == 2,
              "independent providers must contribute to one catalog");
      audio_fails = true;
      isolated_manager.initial_scan();
      const auto partial = isolated_catalog.catalog_json();
      require(partial["state"] == "degraded" && partial["stale"] && partial["devices"].size() == 2,
              "a provider failure must retain its last-good devices");
      require(partial["issues"].size() == 1 &&
                  partial["issues"][0]["provider"] == "test.microphone" &&
                  partial["issues"][0]["retained_last_good"],
              "a provider failure must identify its isolated stale contribution");
      audio_fails = false;
      isolated_manager.initial_scan();
      require(isolated_catalog.catalog_json()["state"] == "ready",
              "provider recovery must restore a healthy catalog");

      PeripheralCatalog invalid_result_catalog("invalid-result", 16);
      std::atomic<std::uint32_t> valid_camera_width{1920};
      int invalid_audio_result = 0;
      PeripheralCatalogManager invalid_result_manager(
          invalid_result_catalog,
          {
              {"test.camera",
               {"media"},
               [&] { return std::vector{camera("valid-camera", valid_camera_width.load())}; }},
              {"test.microphone",
               {"sound"},
               [&] {
                 auto record = microphone("validated-microphone");
                 if (invalid_audio_result == 1)
                   record.id.clear();
                 if (invalid_audio_result == 2)
                   return std::vector{record, record};
                 if (invalid_audio_result == 3)
                   record.type = "id";
                 return std::vector{record};
               }},
          },
          0);
      invalid_result_manager.initial_scan();
      valid_camera_width = 2048;
      invalid_audio_result = 1;
      invalid_result_manager.initial_scan();
      auto invalid_result = invalid_result_catalog.catalog_json();
      require(invalid_result["state"] == "degraded" && invalid_result["devices"].size() == 2 &&
                  invalid_result["devices"][0]["camera"]["modes"][0]["width"] == 2048 &&
                  invalid_result["issues"][0]["code"] == "peripherals.invalid_provider_result" &&
                  invalid_result["issues"][0]["retained_last_good"],
              "malformed provider output must retain only that provider's last-good data "
              "while healthy providers continue updating");
      invalid_audio_result = 2;
      invalid_result_manager.initial_scan();
      require(invalid_result_catalog.catalog_json()["state"] == "degraded",
              "duplicate provider identities must remain an isolated provider issue");
      invalid_audio_result = 3;
      invalid_result_manager.initial_scan();
      require(invalid_result_catalog.catalog_json()["state"] == "degraded",
              "reserved envelope keys must not be accepted as peripheral types");
      invalid_audio_result = 0;
      invalid_result_manager.initial_scan();
      require(invalid_result_catalog.catalog_json()["state"] == "ready",
              "a provider must recover after returning valid records again");

      PeripheralCatalog conflicting_identity_catalog("conflicting-identity", 8);
      PeripheralCatalogManager conflicting_identity_manager(
          conflicting_identity_catalog,
          {
              {"test.camera", {"media"}, [] { return std::vector{camera("shared-identity")}; }},
              {"test.microphone",
               {"sound"},
               [] {
                 auto record = microphone("other-device");
                 record.id = "camera:shared-identity";
                 return std::vector{record};
               }},
          },
          0);
      conflicting_identity_manager.initial_scan();
      const auto conflicting_identity = conflicting_identity_catalog.catalog_json();
      require(!conflicting_identity["ready"] && conflicting_identity["state"] == "degraded" &&
                  conflicting_identity["devices"].empty() &&
                  conflicting_identity["issues"].size() == 2,
              "an identity with no established owner must reject every ambiguous "
              "provider result without depending on registry order");

      for (const bool reverse_provider_order : {false, true}) {
        bool camera_claims_identity = true;
        bool microphone_claims_identity = false;
        auto owned_record = [](const std::string& provider) {
          PeripheralRecord record;
          record.id = "shared:transferable";
          record.type = "camera";
          record.provider = provider;
          record.details = {{"name", "transferable"}};
          return record;
        };
        PeripheralProvider camera_provider{"test.camera", {"media"}, [&] {
                                             return camera_claims_identity
                                                        ? std::vector{owned_record("test.camera")}
                                                        : std::vector<PeripheralRecord>{};
                                           }};
        PeripheralProvider microphone_provider{
            "test.microphone", {"sound"}, [&] {
              return microphone_claims_identity ? std::vector{owned_record("test.microphone")}
                                                : std::vector<PeripheralRecord>{};
            }};
        std::vector<PeripheralProvider> transfer_providers;
        if (reverse_provider_order) {
          transfer_providers.push_back(std::move(microphone_provider));
          transfer_providers.push_back(std::move(camera_provider));
        } else {
          transfer_providers.push_back(std::move(camera_provider));
          transfer_providers.push_back(std::move(microphone_provider));
        }

        PeripheralCatalog transfer_catalog(
            reverse_provider_order ? "transfer-reverse" : "transfer-forward", 8);
        PeripheralCatalogManager transfer_manager(transfer_catalog, std::move(transfer_providers),
                                                  0);
        transfer_manager.initial_scan();
        require(transfer_catalog.catalog_json()["devices"][0]["provider"] == "test.camera",
                "the initial provider must establish ownership of its identity");

        camera_claims_identity = false;
        microphone_claims_identity = true;
        transfer_manager.initial_scan();
        const auto transferred = transfer_catalog.catalog_json();
        require(transferred["ready"] && transferred["state"] == "ready" &&
                    transferred["issues"].empty() && transferred["devices"].size() == 1 &&
                    transferred["devices"][0]["provider"] == "test.microphone",
                "an identity must transfer in one scan after its previous owner "
                "successfully relinquishes it, regardless of registry order");
      }

      PeripheralCatalog partial_startup_catalog("partial-startup", 8);
      PeripheralCatalogManager partial_startup_manager(
          partial_startup_catalog,
          {
              {"test.camera", {"media"}, [] { return std::vector{camera("healthy-camera")}; }},
              {"test.microphone",
               {"sound"},
               []() -> std::vector<PeripheralRecord> {
                 throw std::runtime_error("audio unavailable");
               }},
          },
          0);
      partial_startup_manager.initial_scan();
      const auto partial_startup = partial_startup_catalog.catalog_json();
      require(partial_startup["ready"] && partial_startup["state"] == "degraded" &&
                  !partial_startup["stale"] && partial_startup["devices"].size() == 1 &&
                  !partial_startup["issues"][0]["retained_last_good"],
              "a first-scan provider failure must not hide healthy current devices or "
              "claim stale data was retained");

      PeripheralCatalog unavailable_catalog("unavailable-startup", 8);
      PeripheralCatalogManager unavailable_manager(
          unavailable_catalog,
          {
              {"test.camera",
               {"media"},
               []() -> std::vector<PeripheralRecord> {
                 throw std::runtime_error("camera unavailable");
               }},
              {"test.microphone",
               {"sound"},
               []() -> std::vector<PeripheralRecord> {
                 throw std::runtime_error("audio unavailable");
               }},
          },
          0);
      unavailable_manager.initial_scan();
      const auto unavailable = unavailable_catalog.catalog_json();
      require(!unavailable["ready"] && unavailable["state"] == "degraded" &&
                  !unavailable["stale"] && unavailable["devices"].empty() &&
                  unavailable["issues"].size() == 2 && unavailable["scan_sequence"] == 1,
              "an all-provider startup failure must complete its scan without claiming a "
              "usable catalog");

      PeripheralCatalog bounded("instance-b", 2);
      bounded.apply_success({camera("a"), camera("b")});
      bounded.apply_success({camera("c"), camera("d")});
      require(bounded.events_json(0, std::chrono::milliseconds(0), "instance-b")["resync_required"],
              "event replay overflow must require a snapshot resync");

      PeripheralCatalog coalesced_catalog("coalesced-instance", 8);
      PeripheralCatalogManager coalesced_manager(
          coalesced_catalog,
          {{"test.camera", {"media"}, [] { return std::vector<PeripheralRecord>{}; }}}, 0);
      const auto first_target = coalesced_manager.request_refresh();
      const auto second_target = coalesced_manager.request_refresh();
      require(first_target == 1 && second_target == first_target,
              "queued refresh requests must share the next scan target");
      coalesced_manager.start();
      wait_for_scan_sequence(coalesced_catalog, first_target);
      coalesced_manager.stop();

      const std::string socket_path =
          "/tmp/simaai-peripherals-unit-" + std::to_string(::getpid()) + ".sock";
      PeripheralCatalog api_catalog("api-instance", 16);
      api_catalog.apply_success({camera("api-camera")});
      std::atomic<int> refresh_requests{0};
      PeripheralApiServer server(
          socket_path, api_catalog,
          [&] {
            ++refresh_requests;
            return api_catalog.scan_sequence() + 1;
          },
          2);
      server.start();

      PeripheralApiServer duplicate_server(socket_path, api_catalog, [] { return 1; });
      bool active_socket_rejected = false;
      try {
        duplicate_server.start();
      } catch (const std::runtime_error&) {
        active_socket_rejected = true;
      }
      require(active_socket_rejected, "a second daemon must not replace an active API socket");

      const auto health = request(socket_path, "GET", "/v1/health");
      require(health.status == 200 && health.body["state"] == "ready",
              "health endpoint must expose ready state");
      const auto api_snapshot = request(socket_path, "GET", "/v1/catalog");
      require(api_snapshot.status == 200 && api_snapshot.body["devices"].size() == 1,
              "catalog endpoint must expose one consistent snapshot");
      require(request(socket_path, "POST", "/v1/catalog").status == 405,
              "known routes must reject unsupported methods");
      require(request(socket_path, "GET", "/v1/unknown").status == 404,
              "unknown routes must return not found");
      require(request(socket_path, "GET", "/v1/events?wait_ms=30001").status == 400,
              "event waits above the protocol limit must be rejected");
      require(raw_request(socket_path, "not-an-http-request\r\n\r\n").status == 400,
              "malformed requests must return a bounded JSON error");
      require(raw_request(socket_path, "GET /v1/health HTTP/1.1\r\nX-Oversized: " +
                                           std::string(9000, 'x') + "\r\n\r\n")
                      .status == 413,
              "requests above the header limit must be rejected");
      const auto refresh = request(socket_path, "POST", "/v1/refresh");
      require(refresh.status == 202 && refresh_requests == 1 &&
                  refresh.body["target_scan_sequence"] == 2,
              "refresh endpoint must schedule one daemon-owned scan");

      auto first_client = std::async(std::launch::async, [&] {
        return request(socket_path, "GET",
                       "/v1/events?after_sequence=0&wait_ms=4000&instance_id=api-instance");
      });
      auto second_client = std::async(std::launch::async, [&] {
        return request(socket_path, "GET",
                       "/v1/events?after_sequence=0&wait_ms=4000&instance_id=api-instance");
      });
      std::this_thread::sleep_for(std::chrono::milliseconds(50));
      wait_for_status(socket_path, 503);
      api_catalog.apply_success({camera("api-camera", 2048)});
      const auto first_events = first_client.get();
      const auto second_events = second_client.get();
      require(first_events.status == 200 && second_events.status == 200,
              "simultaneous clients must both receive an event response");
      require(first_events.body["events"].size() == 1 && second_events.body["events"].size() == 1,
              "simultaneous clients must maintain independent cursors");
      require(first_events.body["events"] == second_events.body["events"],
              "simultaneous clients must observe the same catalog event");

      auto first_shutdown_client = std::async(std::launch::async, [&] {
        return request(socket_path, "GET",
                       "/v1/events?after_sequence=1&wait_ms=30000&instance_id=api-instance");
      });
      auto second_shutdown_client = std::async(std::launch::async, [&] {
        return request(socket_path, "GET",
                       "/v1/events?after_sequence=1&wait_ms=30000&instance_id=api-instance");
      });
      std::this_thread::sleep_for(std::chrono::milliseconds(50));
      wait_for_status(socket_path, 503);
      server.stop();
      const auto first_shutdown_events = first_shutdown_client.get();
      const auto second_shutdown_events = second_shutdown_client.get();
      require(first_shutdown_events.status == 200 && first_shutdown_events.body["shutting_down"] &&
                  second_shutdown_events.status == 200 &&
                  second_shutdown_events.body["shutting_down"],
              "shutdown must wake long-poll clients");

      for (int attempt = 0; attempt < 16; ++attempt) {
        const std::string race_socket_path = "/tmp/simaai-peripherals-race-unit-" +
                                             std::to_string(::getpid()) + "-" +
                                             std::to_string(attempt) + ".sock";
        PeripheralCatalog race_catalog("race-instance", 8);
        race_catalog.apply_success({});
        PeripheralApiServer first_racer(race_socket_path, race_catalog,
                                        [&] { return race_catalog.scan_sequence() + 1; });
        PeripheralApiServer second_racer(race_socket_path, race_catalog,
                                         [&] { return race_catalog.scan_sequence() + 1; });
        std::promise<void> start_race_promise;
        const auto start_race = start_race_promise.get_future().share();
        std::atomic<bool> first_started{false};
        std::atomic<bool> second_started{false};
        std::thread first_start([&] {
          start_race.wait();
          try {
            first_racer.start();
            first_started = true;
          } catch (const std::runtime_error&) {
          }
        });
        std::thread second_start([&] {
          start_race.wait();
          try {
            second_racer.start();
            second_started = true;
          } catch (const std::runtime_error&) {
          }
        });
        start_race_promise.set_value();
        first_start.join();
        second_start.join();
        require(first_started.load() != second_started.load(),
                "exactly one concurrent API server start must acquire a socket path");
        require(request(race_socket_path, "GET", "/v1/health").status == 200,
                "the winning server socket must remain reachable after the losing start "
                "returns");
        if (first_started)
          first_racer.stop();
        if (second_started)
          second_racer.stop();
        ::unlink((race_socket_path + ".lock").c_str());
      }

      const std::string overflow_socket_path =
          "/tmp/simaai-peripherals-overflow-unit-" + std::to_string(::getpid()) + ".sock";
      PeripheralRecord oversized = camera("oversized-camera");
      oversized.details["test_payload"] =
          std::string(simaai::neat::peripherals_internal::kMaximumResponseBytes, 'x');
      PeripheralCatalog overflow_catalog("overflow-instance", 8);
      overflow_catalog.apply_success({std::move(oversized)});
      PeripheralApiServer overflow_server(overflow_socket_path, overflow_catalog,
                                          [&] { return overflow_catalog.scan_sequence() + 1; });
      overflow_server.start();
      const auto overflow_response = request(overflow_socket_path, "GET", "/v1/catalog");
      require(overflow_response.status == 500 &&
                  overflow_response.body["error"] == "response_too_large",
              "oversized catalog responses must fail with a bounded JSON error");
      overflow_server.stop();

      const std::string managed_socket_path =
          "/tmp/simaai-peripherals-managed-unit-" + std::to_string(::getpid()) + ".sock";
      std::atomic<std::uint32_t> discovered_width{1920};
      std::atomic<bool> block_discovery{false};
      std::promise<void> discovery_entered_promise;
      auto discovery_entered = discovery_entered_promise.get_future();
      std::promise<void> release_discovery_promise;
      const auto release_discovery = release_discovery_promise.get_future().share();
      PeripheralCatalog managed_catalog("managed-instance", 16);
      PeripheralCatalogManager manager(managed_catalog,
                                       {{"test.camera",
                                         {"media", "video4linux"},
                                         [&] {
                                           if (block_discovery.load()) {
                                             discovery_entered_promise.set_value();
                                             release_discovery.wait();
                                           }
                                           return std::vector{
                                               camera("managed-camera", discovered_width.load())};
                                         }}},
                                       0);
      manager.initial_scan();
      manager.start();
      PeripheralApiServer managed_server(
          managed_socket_path, managed_catalog, [&] { return manager.request_refresh(); }, 1);
      managed_server.start();
      require(request(managed_socket_path, "GET", "/v1/health").body["state"] == "ready",
              "managed integration server must report ready after its initial scan");
      require(request(managed_socket_path, "GET", "/v1/catalog").body["devices"].size() == 1,
              "managed integration server must expose the fake provider snapshot");
      discovered_width = 2048;
      const auto managed_refresh = request(managed_socket_path, "POST", "/v1/refresh");
      require(managed_refresh.status == 202 && managed_refresh.body["target_scan_sequence"] == 2,
              "managed integration refresh must be accepted");
      const auto managed_events =
          request(managed_socket_path, "GET",
                  "/v1/events?after_sequence=0&wait_ms=2000&instance_id=managed-instance");
      require(managed_events.body["events"].size() == 1 &&
                  managed_events.body["events"][0]["kind"] == "changed",
              "managed refresh must publish the fake provider's structural change");
      require(request(managed_socket_path, "GET", "/v1/health").body["scan_sequence"] >=
                  managed_refresh.body["target_scan_sequence"],
              "refresh token must become observable after its scan completes");
      manager.throw_if_failed();
      managed_server.throw_if_failed();

      block_discovery = true;
      const auto blocked_refresh = request(managed_socket_path, "POST", "/v1/refresh");
      require(blocked_refresh.status == 202 && blocked_refresh.body["target_scan_sequence"] == 3,
              "blocked-provider refresh must be accepted");
      require(discovery_entered.wait_for(std::chrono::seconds(2)) == std::future_status::ready,
              "fake discovery provider did not enter its blocked state");
      const auto during_scan_refresh = request(managed_socket_path, "POST", "/v1/refresh");
      require(during_scan_refresh.status == 202 &&
                  during_scan_refresh.body["target_scan_sequence"] == 4,
              "a refresh accepted during discovery must target the following scan");
      require(request(managed_socket_path, "GET", "/v1/health").status == 200,
              "managed API must be idle before starting the shutdown long poll");
      auto blocked_shutdown_client = std::async(std::launch::async, [&] {
        return request(managed_socket_path, "GET",
                       "/v1/events?after_sequence=1&wait_ms=30000&instance_id=managed-instance");
      });
      try {
        std::this_thread::sleep_for(std::chrono::milliseconds(50));
        wait_for_status(managed_socket_path, 503);
        manager.request_stop();
        managed_catalog.shutdown();
        const auto shutdown_start = std::chrono::steady_clock::now();
        managed_server.stop();
        const auto shutdown_duration = std::chrono::steady_clock::now() - shutdown_start;
        block_discovery = false;
        release_discovery_promise.set_value();
        manager.join();
        const auto blocked_shutdown_events = blocked_shutdown_client.get();
        require(shutdown_duration < std::chrono::seconds(2),
                "API shutdown must not wait for a blocked discovery provider");
        require(blocked_shutdown_events.status == 200 &&
                    blocked_shutdown_events.body["shutting_down"],
                "blocked discovery must not prevent long-poll shutdown notification");
      } catch (...) {
        manager.request_stop();
        managed_catalog.shutdown();
        managed_server.stop();
        if (block_discovery.exchange(false))
          release_discovery_promise.set_value();
        manager.join();
        throw;
      }

      const std::string drip_socket_path =
          "/tmp/simaai-peripherals-drip-unit-" + std::to_string(::getpid()) + ".sock";
      PeripheralCatalog drip_catalog("drip-instance", 8);
      drip_catalog.apply_success({});
      PeripheralApiServer drip_server(
          drip_socket_path, drip_catalog, [&] { return drip_catalog.scan_sequence() + 1; }, 1);
      drip_server.start();
      const int drip_client = connect_client(drip_socket_path);
      require(::send(drip_client, "G", 1, 0) == 1, "failed to send partial drip-client request");
      std::this_thread::sleep_for(std::chrono::milliseconds(50));
      wait_for_status(drip_socket_path, 503);
      const auto drip_shutdown_start = std::chrono::steady_clock::now();
      drip_server.stop();
      const auto drip_shutdown_duration = std::chrono::steady_clock::now() - drip_shutdown_start;
      require(drip_shutdown_duration < std::chrono::seconds(2),
              "partial clients must not make API shutdown unbounded");
      char closed_byte = 0;
      require(::recv(drip_client, &closed_byte, 1, 0) <= 0,
              "partial client must be disconnected during shutdown");
      ::close(drip_client);
    }));
