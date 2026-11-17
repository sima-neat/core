#include "gst/NeatCameraMemoryBridge.h"
#include <simaai/runtime/gstreamer/elements.h>

namespace simaai::neat {
bool register_neat_camera_memory_bridge() {
  return simaai::runtime::gstreamer::register_neat_camera_memory_bridge();
}
} // namespace simaai::neat
