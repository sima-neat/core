#pragma once

#include "PeripheralProtocol.h"

#include <cstdint>
#include <optional>
#include <string>
#include <vector>

namespace simaai::neat::peripherals_internal {

struct CameraSizeRange {
  std::uint32_t min_width = 0;
  std::uint32_t min_height = 0;
  std::uint32_t max_width = 0;
  std::uint32_t max_height = 0;
  std::uint32_t step_width = 1;
  std::uint32_t step_height = 1;
};

struct CameraMode {
  std::string format;
  std::uint32_t width = 0;
  std::uint32_t height = 0;
  std::optional<CameraSizeRange> size_range;
  std::uint32_t framerate_num = 30;
  std::uint32_t framerate_den = 1;
  bool supported = false;
  std::string reason;

  bool is_range() const noexcept {
    return size_range.has_value();
  }
};

struct CameraInfo {
  std::string name;
  std::string model;
  std::vector<CameraMode> modes;
};

std::vector<PeripheralRecord> camera_records(const std::vector<CameraInfo>& cameras);
std::vector<PeripheralRecord> discover_camera_peripherals();

} // namespace simaai::neat::peripherals_internal
