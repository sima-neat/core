#pragma once

#include "nodes/groups/UsbCameraInput.h"
#include "pipeline/FormatSpec.h"
#include "pipeline/Graph.h"

#include <cstdint>
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

/// The camera's MJPEG capture node and the chroma layout of its decoded frames.
struct ResolvedCamera {
  std::string device; ///< /dev/videoN that captures MJPEG for the camera ID.
  FormatTag sampling; ///< NV12 for 4:2:0 JPEGs, NV16 for 4:2:2.
};

/// Find the camera by ID, require the exact MJPEG mode, and read one frame to
/// learn its chroma sampling. Throws std::runtime_error with the camera ID and
/// what was found (other IDs, supported modes) when any step fails.
ResolvedCamera resolve_mjpeg_camera(const std::string& camera_id, std::uint32_t width,
                                    std::uint32_t height, std::uint32_t framerate_num,
                                    std::uint32_t framerate_den,
                                    const std::string& sysfs_root = "/sys");

/// The UsbCameraInput Graph for an already resolved camera (validated options).
simaai::neat::Graph mjpeg_camera_graph(const UsbCameraInputOptions& opt,
                                       const ResolvedCamera& camera,
                                       std::uint32_t capture_buffer_count);

} // namespace simaai::neat::internal::uvc
