#include "providers/CameraProviderInternal.h"
#include "pipeline/ErrorCodes.h"
#include "pipeline/NeatError.h"
#include "test_main.h"
#include "test_utils.h"

#include <algorithm>
#include <string>

namespace {

using simaai::neat::peripherals_internal::CameraInfo;
using simaai::neat::peripherals_internal::CameraMode;
using simaai::neat::peripherals_internal::CameraProbe;
using simaai::neat::peripherals_internal::ProbeCamera;
using simaai::neat::peripherals_internal::ProbeFailure;
using simaai::neat::peripherals_internal::ProbeMode;
using simaai::neat::peripherals_internal::ProbeSize;

const CameraMode& find_mode(const CameraInfo& camera, const std::string& format,
                            std::uint32_t width, std::uint32_t height) {
  const auto it =
      std::find_if(camera.modes.begin(), camera.modes.end(), [&](const CameraMode& mode) {
        return !mode.is_range() && mode.format == format && mode.width == width &&
               mode.height == height;
      });
  require(it != camera.modes.end(), "expected camera mode was not reported");
  return *it;
}

void require_error(const CameraProbe& probe, const std::string& expected_code,
                   const std::string& expected_text) {
  try {
    (void)simaai::neat::peripherals_internal::build_camera_catalog(probe);
  } catch (const simaai::neat::NeatError& error) {
    require(error.report().error_code == expected_code, "camera discovery error code mismatch");
    require_contains(error.what(), expected_text, "camera discovery actionable error");
    return;
  }
  throw std::runtime_error("camera discovery was expected to fail");
}

} // namespace

RUN_TEST("unit_camera_discovery_test", ([] {
           using namespace simaai::neat::peripherals_internal;

           CameraProbe probe;
           ProbeCamera camera;
           camera.name = "imx477 5-001a";
           camera.model = "imx477";
           camera.modes = {
               ProbeMode::discrete("video/x-raw", "NV12", 1920, 1080),
               ProbeMode::discrete("video/x-raw", "NV12", 1280, 720),
               ProbeMode::discrete("video/x-raw", "RGB", 1920, 1080),
               ProbeMode::discrete("video/x-bayer", "RG10", 2432, 2048),
               ProbeMode::range("video/x-raw", "NV12", 48, 32, 2432, 2048, 2, 2),
           };
           probe.cameras.push_back(std::move(camera));
           probe.isp_sizes = {
               ProbeSize{1920, 1080},
               ProbeSize{2048, 1080},
               ProbeSize{2432, 2048},
           };

           const auto cameras = build_camera_catalog(probe);
           require(cameras.size() == 1, "expected one camera");
           require(cameras[0].name == "imx477 5-001a",
                   "camera name must be the exact CameraInput camera_name");
           require(cameras[0].model == "imx477", "camera model mismatch");

           const CameraMode& full_hd = find_mode(cameras[0], "NV12", 1920, 1080);
           require(full_hd.supported, "ISP-backed 1080p NV12 mode should be supported");
           require(full_hd.reason.empty(), "supported mode should not have a rejection reason");
           require(full_hd.framerate_num == 30 && full_hd.framerate_den == 1,
                   "enumeration should use CameraInput's existing 30/1 default request");

           const CameraMode& hd = find_mode(cameras[0], "NV12", 1280, 720);
           require(!hd.supported, "1280x720 must not be reported as supported on this ISP");
           require_contains(hd.reason, "ISP output size", "unsupported size reason");

           const CameraMode& rgb = find_mode(cameras[0], "RGB", 1920, 1080);
           require(!rgb.supported, "unvalidated RGB camera output must not be reported supported");
           require_contains(rgb.reason, "NV12", "unsupported output format reason");

           const CameraMode& raw = find_mode(cameras[0], "RG10", 2432, 2048);
           require(!raw.supported, "raw Bayer must not be reported as CameraInput video/x-raw");
           require_contains(raw.reason, "video/x-raw", "unsupported media type reason");

           const auto range = std::find_if(cameras[0].modes.begin(), cameras[0].modes.end(),
                                           [](const CameraMode& mode) { return mode.is_range(); });
           require(range != cameras[0].modes.end(), "continuous size range must be explicit");
           require(range->size_range->min_width == 48 && range->size_range->max_width == 2432 &&
                       range->size_range->step_width == 2,
                   "continuous size range was not preserved");
           require(!range->supported, "an advisory continuous range cannot be wholly supported");
           require_contains(range->reason, "advisory", "continuous range reason");

           CameraProbe missing_isp = probe;
           missing_isp.isp_sizes.reset();
           missing_isp.isp_error = "no Modalix ISP output node was found";
           const auto unverified = build_camera_catalog(missing_isp);
           require(!find_mode(unverified[0], "NV12", 1920, 1080).supported,
                   "mode must not be called supported without ISP evidence");
           require_contains(find_mode(unverified[0], "NV12", 1920, 1080).reason,
                            "no Modalix ISP output node", "missing ISP reason");

           CameraProbe wrong_isp_format = probe;
           wrong_isp_format.isp_sizes = {ProbeSize{1920, 1080, "RGB3"}};
           const auto format_mismatch = build_camera_catalog(wrong_isp_format);
           require(!find_mode(format_mismatch[0], "NV12", 1920, 1080).supported,
                   "a size exposed only for another ISP format must not validate NV12");

           CameraProbe backend_failure;
           backend_failure.failure = ProbeFailure::BackendUnavailable;
           backend_failure.failure_detail = "libcameraprovider is not installed";
           require_error(backend_failure, simaai::neat::error_codes::kPluginMissing,
                         "libcameraprovider");

           CameraProbe permission_failure;
           permission_failure.failure = ProbeFailure::PermissionDenied;
           permission_failure.failure_detail = "permission denied opening /dev/media0";
           require_error(permission_failure, simaai::neat::error_codes::kPermissionDenied,
                         "video group");

           CameraProbe no_cameras;
           require_error(no_cameras, simaai::neat::error_codes::kCameraNotFound,
                         "No CameraInput cameras");
           require(build_camera_catalog(no_cameras, true).empty(),
                   "the daemon provider must treat no connected camera as a ready empty result");
         }));
