#pragma once

#include "pipeline/FormatSpec.h"

#include <cstdint>
#include <span>

namespace simaai::neat::internal {

// Native SimaDecode output format for a JPEG frame: NV12 for 4:2:0, NV16 for 4:2:2.
// Throws std::invalid_argument if the data is not a JPEG or uses another sampling.
FormatTag jpeg_output_format(std::span<const std::uint8_t> jpeg);

} // namespace simaai::neat::internal
