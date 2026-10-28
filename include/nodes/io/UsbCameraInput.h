/**
 * @file
 * @ingroup nodes_io
 * @brief USB (UVC) MJPEG camera source node: the camera's JPEG frames.
 *
 * @see CameraInput
 * @see SimaDecode
 */
#pragma once

#include "builder/Node.h"
#include "pipeline/FormatSpec.h"

#include <cstdint>
#include <memory>
#include <string>
#include <vector>

namespace simaai::neat {

/**
 * @brief Options for UsbCameraInput, with CameraInputOptions' field names and defaults.
 *
 * `camera_name` is required here, and there is no `format`: the node outputs the camera's
 * JPEG frames, and `usb_camera_info()` says which layout they decode to.
 */
struct UsbCameraInputOptions {
  // Required stable USB camera ID from usb_camera_ids().
  std::string camera_name;

  // An exact MJPEG mode of the camera.
  std::uint32_t width = 1920;
  std::uint32_t height = 1080;
  std::uint32_t framerate_num = 30;
  std::uint32_t framerate_den = 1;

  // Name used by downstream CVU/MLA configs through GstSimaMeta.
  std::string buffer_name = "camera";

  // Insert a small live-source queue by default to avoid unbounded camera backpressure.
  bool insert_queue = true;
  bool leaky_queue = true;
  std::uint32_t queue_depth = 2;
};

/// A USB camera as UsbCameraInput opens it.
struct UsbCameraInfo {
  std::string device;                         ///< /dev/videoN that captures its MJPEG.
  FormatTag decoded_format = FormatTag::Auto; ///< NV12 for 4:2:0 JPEGs, NV16 for 4:2:2.
  int node_index = -1; ///< sysfs index of `device` in its USB interface; -1 unknown.
};

/// Stable IDs of the connected USB (UVC) cameras, for `UsbCameraInputOptions::camera_name`.
/// IDs follow libcamera's uvcvideo handler: "<controller>-<ports:config.interface>-<vid:pid>",
/// for example "/base/pci@7000000000-3.1:1.0-046d:08e5". An ID is stable across boots and
/// plug order and changes when the camera moves to another USB port.
std::vector<std::string> usb_camera_ids();

/**
 * @brief Validate `opt`, find its camera, check the mode and read one frame.
 *
 * `decoded_format` is the layout the camera's JPEGs decode to: set it as a `SimaDecode`
 * `out_format`. A model reads NV16 directly: set its preprocessing input format to match
 * (`PreprocessColorFormat::NV16`). Pass the result to `UsbCameraInput(opt, info)` so the
 * camera is not opened a second time.
 * @throws std::invalid_argument and std::runtime_error as UsbCameraInput does.
 */
UsbCameraInfo usb_camera_info(const UsbCameraInputOptions& opt);

} // namespace simaai::neat

namespace simaai::neat::nodes {

/**
 * @brief Live USB (UVC) MJPEG camera source.
 *
 * Opens the camera whose stable ID is `opt.camera_name` (see `usb_camera_ids()`),
 * requires the exact MJPEG mode `opt.width` x `opt.height` at
 * `opt.framerate_num`/`opt.framerate_den`, and captures through V4L2.
 *
 * The node outputs the camera's JPEG frames for `VideoSender` passthrough (MJPEG) and
 * `SimaDecode` (MJPEG) on the same Graph. The camera captures into CMA DMA-BUFs that the
 * hardware decoder reads in place, so no frame is copied between capture and decode.
 *
 * Building a Graph again after the camera was unplugged, moved or renumbered
 * throws; call UsbCameraInput again to open it.
 *
 * @throws std::invalid_argument for a missing camera ID or buffer name, or a zero size
 *         or rate.
 * @throws std::runtime_error when the camera is not connected, is in use, lacks the mode,
 *         or sends no decodable frame. The message names the camera ID and lists the
 *         connected IDs or the camera's MJPEG modes.
 */
std::shared_ptr<simaai::neat::Node> UsbCameraInput(simaai::neat::UsbCameraInputOptions opt);

/**
 * @brief UsbCameraInput for a camera that `usb_camera_info(opt)` already checked.
 *
 * Reuses `info` instead of opening the camera again: it validates `opt` like the overload
 * above and confirms, from sysfs only, that `info.device` still belongs to the camera ID
 * `opt.camera_name`. Pass the same `opt` that produced `info`; the mode is not checked again.
 *
 * @throws std::invalid_argument for invalid options, as the overload above.
 * @throws std::runtime_error when `info` is stale (the camera was unplugged, moved or
 *         renumbered since) or describes another camera; call usb_camera_info again.
 */
std::shared_ptr<simaai::neat::Node> UsbCameraInput(simaai::neat::UsbCameraInputOptions opt,
                                                   const simaai::neat::UsbCameraInfo& info);

} // namespace simaai::neat::nodes
