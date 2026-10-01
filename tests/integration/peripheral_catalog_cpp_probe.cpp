#include "neat/peripherals.h"

#include <nlohmann/json.hpp>

#include <iostream>
#include <optional>
#include <string>

namespace {

nlohmann::json encode_optional_string(const std::optional<std::string>& value) {
  return value ? nlohmann::json(*value) : nlohmann::json(nullptr);
}

nlohmann::json encode_mode(const simaai::neat::peripherals::CameraMode& mode) {
  nlohmann::json encoded = {
      {"format", mode.format},
      {"framerate_num", mode.framerate_num},
      {"framerate_den", mode.framerate_den},
      {"supported", mode.supported},
      {"reason", mode.reason},
  };
  if (mode.size_range) {
    encoded["size_range"] = {
        {"min_width", mode.size_range->min_width},   {"min_height", mode.size_range->min_height},
        {"max_width", mode.size_range->max_width},   {"max_height", mode.size_range->max_height},
        {"step_width", mode.size_range->step_width}, {"step_height", mode.size_range->step_height},
    };
  } else {
    encoded["width"] = mode.width;
    encoded["height"] = mode.height;
  }
  return encoded;
}

nlohmann::json encode_catalog(const simaai::neat::peripherals::Catalog& catalog) {
  nlohmann::json encoded = {
      {"instance_id", catalog.instance_id},
      {"state", catalog.state},
      {"stale", catalog.stale},
      {"revision", catalog.revision},
      {"sequence", catalog.sequence},
      {"scan_sequence", catalog.scan_sequence},
      {"last_success_at", encode_optional_string(catalog.last_success_at)},
      {"last_attempt_at", encode_optional_string(catalog.last_attempt_at)},
      {"error", nullptr},
      {"issues", nlohmann::json::array()},
      {"devices", nlohmann::json::array()},
  };
  if (catalog.error)
    encoded["error"] = {{"code", catalog.error->code}, {"reason", catalog.error->reason}};
  for (const auto& issue : catalog.issues) {
    encoded["issues"].push_back({
        {"provider", issue.provider},
        {"code", issue.code},
        {"reason", issue.reason},
        {"retained_last_good", issue.retained_last_good},
    });
  }
  for (const auto& device : catalog.devices) {
    nlohmann::json peripheral = {
        {"id", device.id},
        {"type", device.type},
        {"provider", device.provider},
        {"camera", nullptr},
    };
    if (device.camera) {
      peripheral["camera"] = {
          {"camera_name", encode_optional_string(device.camera->camera_name)},
          {"model", encode_optional_string(device.camera->model)},
          {"backend", device.camera->backend},
          {"modes", nlohmann::json::array()},
      };
      for (const auto& mode : device.camera->modes)
        peripheral["camera"]["modes"].push_back(encode_mode(mode));
    }
    encoded["devices"].push_back(std::move(peripheral));
  }
  return encoded;
}

} // namespace

int main() {
  try {
    std::cout << encode_catalog(simaai::neat::peripherals::list()).dump() << '\n';
    return 0;
  } catch (const std::exception& error) {
    std::cerr << error.what() << '\n';
    return 1;
  }
}
