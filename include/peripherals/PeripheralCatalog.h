/**
 * @file
 * @brief Typed client for the board-local peripheral catalog served by SiMa Sentinel.
 */
#pragma once

#include <cstddef>
#include <cstdint>
#include <optional>
#include <string>
#include <vector>

namespace simaai::neat::peripherals {

/** A structured daemon or provider failure. */
struct CatalogError {
  std::string code;
  std::string reason;
};

/** A provider-scoped refresh problem reported by the daemon. */
struct ProviderIssue {
  std::string provider;
  std::string code;
  std::string reason;
  bool retained_last_good = false;
};

/** A range of image sizes advertised by a camera backend. */
struct CameraSizeRange {
  std::uint32_t min_width = 0;
  std::uint32_t min_height = 0;
  std::uint32_t max_width = 0;
  std::uint32_t max_height = 0;
  std::uint32_t step_width = 1;
  std::uint32_t step_height = 1;
};

/** One daemon-classified camera mode. */
struct CameraMode {
  std::string format;
  std::uint32_t width = 0;
  std::uint32_t height = 0;
  std::optional<CameraSizeRange> size_range;
  std::uint32_t framerate_num = 0;
  std::uint32_t framerate_den = 1;
  bool supported = false;
  std::string reason;

  bool is_range() const noexcept {
    return size_range.has_value();
  }
};

/** Camera-specific details nested under a peripheral record. */
struct CameraDetails {
  /** Exact value accepted by CameraInputOptions when the backend is selectable. */
  std::optional<std::string> camera_name;
  std::optional<std::string> model;
  std::string backend;
  std::vector<CameraMode> modes;
};

/** Common identity plus type-specific details for one peripheral. */
struct Peripheral {
  std::string id;
  std::string type;
  std::string provider;
  /** Typed camera details; set only when `type == "camera"`. */
  std::optional<CameraDetails> camera;
  /**
   * Details for any peripheral type, as compact JSON.
   *
   * Holds the JSON object that Sentinel publishes under the record key named
   * by `type` (for example `"camera"`, `"microphone"`, or `"lidar"`), or
   * `"{}"` when that key is absent or null. Every field and value is preserved,
   * including fields this Core release does not know; the text is re-serialized,
   * so key order and whitespace may differ from the daemon response.
   *
   * This is the authoritative way to read peripheral types for which Core has
   * no typed struct: a new device type is usable as soon as Sentinel reports
   * it. Cameras carry both this field and the typed `camera` member.
   */
  std::string details_json = "{}";
};

/**
 * One internally consistent daemon catalog snapshot.
 *
 * The object is directly iterable over `devices` while retaining the daemon
 * identity, revision, freshness, and provider diagnostics that belong to the
 * same snapshot.
 */
struct Catalog {
  std::string instance_id;
  std::string state;
  bool stale = false;
  std::uint64_t revision = 0;
  std::uint64_t sequence = 0;
  std::uint64_t scan_sequence = 0;
  std::optional<std::string> last_success_at;
  std::optional<std::string> last_attempt_at;
  std::optional<CatalogError> error;
  std::vector<ProviderIssue> issues;
  std::vector<Peripheral> devices;

  using iterator = std::vector<Peripheral>::iterator;
  using const_iterator = std::vector<Peripheral>::const_iterator;

  iterator begin() noexcept {
    return devices.begin();
  }
  iterator end() noexcept {
    return devices.end();
  }
  const_iterator begin() const noexcept {
    return devices.begin();
  }
  const_iterator end() const noexcept {
    return devices.end();
  }
  const_iterator cbegin() const noexcept {
    return devices.cbegin();
  }
  const_iterator cend() const noexcept {
    return devices.cend();
  }
  std::size_t size() const noexcept {
    return devices.size();
  }
  bool empty() const noexcept {
    return devices.empty();
  }
  Peripheral& operator[](std::size_t index) noexcept {
    return devices[index];
  }
  const Peripheral& operator[](std::size_t index) const noexcept {
    return devices[index];
  }
};

/**
 * Read and validate one snapshot from the local SiMa Sentinel daemon.
 *
 * This function performs exactly one bounded `GET /v1/peripherals` request on
 * `/run/simaai-sentinel/api.sock`. It never scans hardware, caches results,
 * connects over SSH, or falls back when Sentinel is unavailable. Failures
 * throw `NeatError` with a structured `GraphReport::error_code`; when
 * Sentinel is not running, the message asks the user to install it with
 * `sima-cli neat install sentinel` or start `simaai-sentinel.service`.
 */
Catalog list();

} // namespace simaai::neat::peripherals
