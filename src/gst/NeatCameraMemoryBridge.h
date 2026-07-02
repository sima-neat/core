#pragma once
#ifndef SIMA_NEAT_INTERNAL
#error "Internal header. Not part of the public API."
#endif

namespace simaai::neat {

inline constexpr const char* kNeatCameraMemoryBridgeFactory = "neatcamerabridge";

// Registers Neat's private adaptive camera memory bridge element in the process-local
// GStreamer registry.  The element is intentionally not a shipped public plugin; Neat's
// graph materializer inserts it for CameraInput fallback/adaptive device-memory handoff.
bool register_neat_camera_memory_bridge();

} // namespace simaai::neat
