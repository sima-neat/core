/**
 * @file
 * @ingroup nodes_common
 * @brief `VideoRate` Node — limit framerate by dropping frames.
 *
 * Wraps GStreamer's `videorate`. Combine with a downstream caps filter that pins
 * `framerate=…/1` to select the output rate. Always uses drop-only mode, which
 * preserves input timestamps and does not duplicate frames.
 */
#pragma once

#include "builder/Node.h"

#include <memory>

namespace simaai::neat::nodes {
/// Convenience factory for a `VideoRate` Node — pair with a downstream framerate caps filter.
std::shared_ptr<simaai::neat::Node> VideoRate();
} // namespace simaai::neat::nodes
