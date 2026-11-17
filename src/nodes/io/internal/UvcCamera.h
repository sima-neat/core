#pragma once

#include "nodes/io/UsbCameraInput.h"
#include "pipeline/FormatSpec.h"

#include <cstdint>
#include <memory>
#include <string>
#include <vector>

namespace simaai::neat::internal::uvc {

/// A USB video node and the stable camera ID libcamera's uvcvideo handler
/// would give it: "<controller firmware node>-<ports:config.interface>-<vid:pid>".
struct CaptureNode {
  std::string id;     ///< Stable across boots and plug order; changes with the USB port.
  std::string device; ///< /dev/videoN for this boot.
};

/// Stable ID of the USB interface at `interface_dir` (a sysfs device path such
/// as .../usb1/1-3/1-3.1/1-3.1:1.0), as libcamera's uvcvideo handler builds it.
/// Empty when the controller firmware node or the USB IDs cannot be read.
std::string camera_id(const std::string& interface_dir, const std::string& sysfs_root = "/sys");

/// uvcvideo nodes under `sysfs_root`, sorted by device. One USB interface can
/// expose several nodes (capture plus metadata), which then share an ID.
std::vector<CaptureNode> list_nodes(const std::string& sysfs_root = "/sys");

/// The distinct camera IDs of `nodes`, sorted by ID so the order does not depend on plug order.
std::vector<std::string> camera_ids(const std::vector<CaptureNode>& nodes);

/// The camera's MJPEG capture node and the chroma layout of its decoded frames.
struct ResolvedCamera {
  std::string device;     ///< /dev/videoN that captures MJPEG for the camera ID.
  FormatTag sampling;     ///< NV12 for 4:2:0 JPEGs, NV16 for 4:2:2.
  std::string id;         ///< The camera ID; empty skips require_still_connected().
  std::string sysfs_root; ///< Where `id` was resolved.
  int index = -1;         ///< sysfs index of `device` in its interface (capture 0); -1 unknown.
};

/// Throw when `camera.device` no longer belongs to `camera.id` or is no longer the node at
/// `camera.index`, for example after the camera was unplugged, moved to another port or
/// renumbered since it was resolved (its metadata node can take the old device number).
void require_still_connected(const ResolvedCamera& camera);

/// The camera that `info` (from usb_camera_info) describes, checked against sysfs only: no
/// device is opened. Throws std::runtime_error naming usb_camera_info when `info.device` no
/// longer belongs to `camera_id` or is no longer the node at `info.node_index`.
ResolvedCamera camera_from_info(const std::string& camera_id, const UsbCameraInfo& info,
                                const std::string& sysfs_root = "/sys");

/// Find the camera by ID, require the exact MJPEG mode, and read one frame to
/// learn its chroma sampling. Throws std::runtime_error with the camera ID and
/// what was found (other IDs, supported modes) when any step fails.
ResolvedCamera resolve_mjpeg_camera(const std::string& camera_id, std::uint32_t width,
                                    std::uint32_t height, std::uint32_t framerate_num,
                                    std::uint32_t framerate_den,
                                    const std::string& sysfs_root = "/sys");

/// The UsbCameraInput node for an already resolved camera (validated options).
std::shared_ptr<Node> camera_node(const UsbCameraInputOptions& opt, const ResolvedCamera& camera);

} // namespace simaai::neat::internal::uvc
