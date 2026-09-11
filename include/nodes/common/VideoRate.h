/**
 * @file
 * @ingroup nodes_common
 * @brief `VideoRate` Node — limit framerate by dropping frames by default.
 *
 * Wraps GStreamer's `videorate`. Combine with a downstream caps filter that pins
 * `framerate=…/1` to select the output rate. Drop-only mode preserves input
 * timestamps; pass false to allow duplication and regularize timestamps.
 */
#pragma once

#include "builder/Node.h"

#include <memory>

namespace simaai::neat::nodes {
/// Convenience factory for a `VideoRate` Node — pair with a downstream framerate caps filter.
std::shared_ptr<simaai::neat::Node> VideoRate();
/// Set drop_only=false to permit frame duplication. The no-argument overload uses true.
std::shared_ptr<simaai::neat::Node> VideoRate(bool drop_only);
} // namespace simaai::neat::nodes
