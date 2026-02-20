/**
 * @file
 * @ingroup nodes_common
 * @brief Caps helpers and raw fragment escape hatch.
 */
#pragma once

#include "builder/Node.h"
#include "contracts/ContractTypes.h"

#include <memory>
#include <string>

namespace simaai::neat::nodes {

// Escape hatch (implemented via a raw gst fragment node)
std::shared_ptr<simaai::neat::Node>
Custom(std::string fragment, simaai::neat::InputRole role = simaai::neat::InputRole::None);

// Caps helpers (typed; always produce a capsfilter)
std::shared_ptr<simaai::neat::Node>
CapsRaw(std::string format, int width = -1, int height = -1, int fps = -1,
        simaai::neat::CapsMemory memory = simaai::neat::CapsMemory::Any);

std::shared_ptr<simaai::neat::Node> CapsNV12SysMem(int w, int h, int fps);
std::shared_ptr<simaai::neat::Node>
CapsI420(int w, int h, int fps, simaai::neat::CapsMemory memory = simaai::neat::CapsMemory::Any);

} // namespace simaai::neat::nodes
