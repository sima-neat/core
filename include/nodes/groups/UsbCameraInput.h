/**
 * @file
 * @ingroup nodes_groups
 * @brief USB (UVC) camera input for cameras that send MJPEG, decoded to NV12.
 *
 * @see CameraInput
 * @see SimaDecode
 */
#pragma once

#include "pipeline/Graph.h"

#include <cstdint>
#include <string>
#include <vector>

namespace simaai::neat {

/**
 * @brief Options for UsbCameraInput, with CameraInputOptions' field names and defaults.
 *
 * `camera_name` is required here, and there is no `format`: the output is always NV12 in
 * SiMaAI memory.
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

  // No effect here; kept so the options match CameraInputOptions. Capture always
  // lands in the decoder's own CMA buffers, so no frame needs a copy.
  bool allow_cpu_fallback = false;
};

/// Stable IDs of the connected USB (UVC) cameras, for `UsbCameraInputOptions::camera_name`.
/// IDs follow libcamera's uvcvideo handler: "<controller>-<ports:config.interface>-<vid:pid>",
/// for example "/base/pci@7000000000-3.1:1.0-046d:08e5". An ID is stable across boots and
/// plug order and changes when the camera moves to another USB port.
std::vector<std::string> usb_camera_ids();

} // namespace simaai::neat

namespace simaai::neat::nodes::groups {

/**
 * @brief Live USB (UVC) MJPEG camera input that outputs NV12 frames in SiMaAI memory.
 *
 * Opens the camera whose stable ID is `opt.camera_name` (see `usb_camera_ids()`),
 * requires the exact MJPEG mode `opt.width` x `opt.height` at
 * `opt.framerate_num`/`opt.framerate_den`, and captures through V4L2. The camera
 * writes each frame into CMA buffers the hardware decoder reads in place, so no
 * frame is copied before decode. Frames with bytes after their JPEG end marker are
 * trimmed; incomplete or corrupt frames are dropped. A camera that sends 4:2:2 JPEGs
 * gets one EV74 ColorConvert from NV16 to NV12, read from its first frame when the
 * Graph is created.
 *
 * Building the Graph again after the camera was unplugged, moved or renumbered
 * throws; call UsbCameraInput again to open it.
 *
 * @param capture_buffer_count CMA capture buffers shared by the camera and the
 *        decoder; 0 uses the queue depth plus the decoder's input slots. At most 64.
 * @throws std::invalid_argument for a missing camera ID or buffer name, a zero size
 *         or rate, or more than 64 capture buffers.
 * @throws std::runtime_error when the installed decoder is too old, or the camera
 *         is not connected, is in use, lacks the mode, or sends no decodable frame.
 *         The message names the camera ID and lists the connected IDs or the
 *         camera's MJPEG modes.
 */
simaai::neat::Graph UsbCameraInput(simaai::neat::UsbCameraInputOptions opt,
                                   std::uint32_t capture_buffer_count = 0);

} // namespace simaai::neat::nodes::groups
