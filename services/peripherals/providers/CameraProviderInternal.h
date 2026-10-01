#pragma once

#include "CameraProvider.h"

#include <cstdint>
#include <optional>
#include <string>
#include <vector>

namespace simaai::neat::peripherals_internal {

struct ProbeSize {
  std::uint32_t width = 0;
  std::uint32_t height = 0;
  std::string format = "NV12";
};

struct ProbeMode {
  std::string media_type;
  std::string format;
  std::uint32_t min_width = 0;
  std::uint32_t min_height = 0;
  std::uint32_t max_width = 0;
  std::uint32_t max_height = 0;
  std::uint32_t step_width = 1;
  std::uint32_t step_height = 1;
  bool size_is_range = false;

  static ProbeMode discrete(std::string media_type, std::string format, std::uint32_t width,
                            std::uint32_t height);
  static ProbeMode range(std::string media_type, std::string format, std::uint32_t min_width,
                         std::uint32_t min_height, std::uint32_t max_width,
                         std::uint32_t max_height, std::uint32_t step_width,
                         std::uint32_t step_height);
};

struct ProbeCamera {
  std::string name;
  std::string model;
  std::vector<ProbeMode> modes;
};

enum class ProbeFailure {
  None,
  BackendUnavailable,
  PermissionDenied,
};

struct CameraProbe {
  ProbeFailure failure = ProbeFailure::None;
  std::string failure_detail;
  std::vector<ProbeCamera> cameras;
  std::optional<std::vector<ProbeSize>> isp_sizes;
  std::string isp_error;
};

CameraProbe probe_camera_backend();
void canonicalize_camera_catalog(std::vector<CameraInfo>& cameras);
std::vector<CameraInfo> build_camera_catalog(const CameraProbe& probe,
                                             bool allow_no_cameras = false);

} // namespace simaai::neat::peripherals_internal
