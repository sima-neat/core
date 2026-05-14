/**
 * @file
 * @ingroup nodes_common
 * @brief `VideoRate` Node — adjust framerate (drop / duplicate frames as needed).
 *
 * Wraps GStreamer's `videorate`. Combine with a downstream caps filter that pins
 * `framerate=…/1` to coerce the stream to a target rate.
 */
#pragma once

#include "builder/Node.h"

#include <memory>

namespace simaai::neat::nodes {
/// Convenience factory for a `VideoRate` Node — pair with a downstream framerate caps filter.
std::shared_ptr<simaai::neat::Node> VideoRate();
} // namespace simaai::neat::nodes
