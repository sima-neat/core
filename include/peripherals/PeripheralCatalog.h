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

/** A provider that failed in Sentinel's latest scan. */
struct CatalogError {
  std::string provider;
  std::string code;
  std::string reason;
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

/** One camera mode, classified by Core for CameraInput's default profile. */
struct CameraMode {
  std::string format;
  std::uint32_t width = 0;
  std::uint32_t height = 0;
  std::optional<CameraSizeRange> size_range;
  /** Fastest rate in the mode's frame intervals; 0/1 when it lists none. */
  std::uint32_t framerate_num = 0;
  std::uint32_t framerate_den = 1;
  /** Whether CameraInput's default libcamera profile accepts this mode. */
  bool supported = false;
  /** Why the mode is not supported; empty when it is. */
  std::string reason;

  bool is_range() const noexcept {
    return size_range.has_value();
  }
};

/** Camera fields of a peripheral record. */
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
  /** Typed camera details; set only when `type == "camera"`. */
  std::optional<CameraDetails> camera;
  /**
   * The whole device record Sentinel published, as compact JSON.
   *
   * Every field and value is preserved, including fields this Core release
   * does not know; the text is re-serialized, so key order and whitespace may
   * differ from the daemon response. This is the way to read peripheral types
   * for which Core has no typed struct, such as microphones: a new device type
   * is usable as soon as Sentinel reports it.
   */
  std::string details_json = "{}";
};

/**
 * One Sentinel catalog snapshot.
 *
 * The object is directly iterable over `devices`.
 */
struct Catalog {
  /** Changes whenever `devices` or `errors` change; compare for equality only. */
  std::uint64_t revision = 0;
  /** When the scan behind this snapshot started; unset until the first scan completes. */
  std::optional<std::string> observed_at;
  std::vector<CatalogError> errors;
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
 * Read one snapshot from the local SiMa Sentinel daemon.
 *
 * This function performs exactly one bounded `GET /v1/peripherals` request on
 * `/run/simaai-sentinel/api.sock` and classifies each camera mode for
 * CameraInput. It never scans hardware, caches results, connects over SSH, or
 * falls back when Sentinel is unavailable. Failures throw `NeatError` with a
 * structured `GraphReport::error_code`; when Sentinel is not running, the
 * message asks the user to install it with `sima-cli neat install sentinel`
 * or start `simaai-sentinel.service`.
 */
Catalog list();

} // namespace simaai::neat::peripherals
