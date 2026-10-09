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

// "W by H", not "WxH": the diagnostic normalizer drops reasons containing "0x".
std::optional<std::string> camera_frame_size_mismatch_reason(int caps_width, int caps_height,
                                                             int frame_width, int frame_height);

// Compares the first buffer's GstVideoMeta on `element`'s src pad with its caps. On a mismatch
// posts one error and drops buffers; otherwise removes itself.
void attach_camera_frame_size_check(GstElement* element);

} // namespace simaai::neat::pipeline_internal
