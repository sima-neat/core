/**
 * @file
 * @ingroup nodes_common
 * @brief `ImageFreeze` Node — turn a single still image into a continuous video stream.
 *
 * Wraps GStreamer's `imagefreeze`. Useful for testing detection models against a single
 * frame, or for synthesizing a video stream from a static asset.
 */
#pragma once

#include "builder/Node.h"

#include <memory>

namespace simaai::neat::nodes {
/**
 * @brief Repeat one input frame as a continuous stream.
 * @param num_buffers Number of frames to emit. `-1` means unlimited (until EOS).
 */
std::shared_ptr<simaai::neat::Node> ImageFreeze(int num_buffers = -1);
} // namespace simaai::neat::nodes
