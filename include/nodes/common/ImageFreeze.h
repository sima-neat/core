/**
 * @file
 * @ingroup nodes_common
 * @brief ImageFreeze node wrapper.
 */
#pragma once

#include "builder/Node.h"

#include <memory>

namespace simaai::neat::nodes {
std::shared_ptr<simaai::neat::Node>
ImageFreeze(int num_buffers = -1); // implemented as raw gst fragment (imagefreeze ...)
} // namespace simaai::neat::nodes
