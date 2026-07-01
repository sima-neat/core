#pragma once
#ifndef SIMA_NEAT_INTERNAL
#error "Internal header. Not part of the public API."
#endif

#include "pipeline/PayloadType.h"

#include <string>
#include <string_view>

namespace simaai::neat::pipeline_internal {

std::string caps_media_type(std::string_view caps);
PayloadType payload_type_from_caps_string(std::string_view caps);

} // namespace simaai::neat::pipeline_internal
