#pragma once

#include <string>

namespace simaai::neat::gst {

// Public diagnostic URLs carry only the HTTP scheme, host, and optional port.
std::string http_origin(const char* location);

} // namespace simaai::neat::gst
