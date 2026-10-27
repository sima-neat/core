/**
 * @file
 * @ingroup nodes_io
 * @brief USB (UVC) MJPEG camera source node: decoded frames, or the camera's JPEGs.
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

/// What UsbCameraInput outputs.
enum class UsbCameraOutput {
  Frames, ///< Frames decoded in place to the camera's native layout (UsbCameraInfo).
  Jpeg,   ///< The camera's JPEG frames, not decoded.
};

/**
 * @brief Options for UsbCameraInput, with CameraInputOptions' field names and defaults.
 *
 * `camera_name` is required here, and there is no `format`: decoded frames keep the
 * camera's native chroma layout (see `usb_camera_info()`), in SiMaAI memory.
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

  // No effect here; kept so the options match CameraInputOptions. Frames capture
  // always lands in the decoder's own CMA buffers, so no frame needs a copy.
  bool allow_cpu_fallback = false;

  UsbCameraOutput output = UsbCameraOutput::Frames;
};

/// A USB camera as UsbCameraInput opens it.
struct UsbCameraInfo {
  std::string device;                         ///< /dev/videoN that captures its MJPEG.
  FormatTag decoded_format = FormatTag::Auto; ///< NV12 for 4:2:0 JPEGs, NV16 for 4:2:2.
};

/// Stable IDs of the connected USB (UVC) cameras, for `UsbCameraInputOptions::camera_name`.
/// IDs follow libcamera's uvcvideo handler: "<controller>-<ports:config.interface>-<vid:pid>",
/// for example "/base/pci@7000000000-3.1:1.0-046d:08e5". An ID is stable across boots and
/// plug order and changes when the camera moves to another USB port.
std::vector<std::string> usb_camera_ids();

/**
 * @brief Validate `opt`, find its camera, check the mode and read one frame.
 *
 * `decoded_format` is the format UsbCameraOutput::Frames outputs. A model reads NV16
 * directly: set its preprocessing input format to match (`PreprocessColorFormat::NV16`).
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
 * UsbCameraOutput::Frames: the camera writes each frame into CMA buffers the hardware
 * decoder reads in place, so no frame is copied before decode. Frames come out in the
 * camera's native layout: NV12 for 4:2:0 JPEGs, NV16 for 4:2:2, read from its first
 * frame when the node is created. Frames with bytes after their JPEG end marker are
 * trimmed; incomplete or corrupt frames are dropped.
 *
 * UsbCameraOutput::Jpeg: the camera's JPEG frames, exported by the driver as DMA-BUF,
 * for `VideoSender` passthrough (MJPEG) and `SimaDecode` (MJPEG) on the same Graph. The
 * decoder usually cannot import the camera's buffers in place, so it copies each frame.
 *
 * Building a Graph again after the camera was unplugged, moved or renumbered
 * throws; call UsbCameraInput again to open it.
 *
 * @param capture_buffer_count Frames only: CMA capture buffers shared by the camera and
 *        the decoder; 0 uses the queue depth plus the decoder's input slots. At most 64.
 * @throws std::invalid_argument for a missing camera ID or buffer name, a zero size
 *         or rate, more than 64 capture buffers (a default pool needs a queue depth
 *         of at most 62), or a nonzero capture_buffer_count with UsbCameraOutput::Jpeg.
 * @throws std::runtime_error when the installed decoder is too old (Frames), or the
 *         camera is not connected, is in use, lacks the mode, or sends no decodable
 *         frame. The message names the camera ID and lists the connected IDs or the
 *         camera's MJPEG modes.
 */
std::shared_ptr<simaai::neat::Node> UsbCameraInput(simaai::neat::UsbCameraInputOptions opt,
                                                   std::uint32_t capture_buffer_count = 0);

} // namespace simaai::neat::nodes
