#pragma once
#ifndef SIMA_NEAT_INTERNAL
#error "Internal header. Not part of the public API."
#endif

#include <gst/gst.h>

#include <optional>
#include <string>

namespace simaai::neat::pipeline_internal {

inline constexpr const char* kCameraFrameSizeMismatchDiagnosticId =
    "neatcamera.frame_size_mismatch";

// Returns the user-facing reason when a camera frame's size differs from the negotiated
// caps size, or nullopt when they match. Sizes are written as "W by H" because the
// diagnostic normalizer rejects reasons that contain "0x" (as in "1920x1080").
std::optional<std::string> camera_frame_size_mismatch_reason(int caps_width, int caps_height,
                                                             int frame_width, int frame_height);

// Checks the first buffer leaving `element`'s src pad. libcamera can accept a size the
// sensor has no mode for and deliver frames of the sensor's current size instead, with
// only the buffer's GstVideoMeta telling the truth (core #881). On a mismatch the probe
// posts one structured error from `element` and drops buffers until the pipeline stops;
// otherwise, or when the buffer carries no GstVideoMeta, it removes itself.
void attach_camera_frame_size_check(GstElement* element);

} // namespace simaai::neat::pipeline_internal
