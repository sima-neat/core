#pragma once

#include "peripherals/PeripheralCatalog.h"

#include <chrono>
#include <cstddef>
#include <limits>
#include <string>

namespace simaai::neat::peripherals_internal {

// maximum_send_bytes is an internal seam for deterministic partial-write tests;
// production callers use the unbounded default.
peripherals::Catalog
list_from_socket(const std::string& socket_path, std::chrono::milliseconds timeout,
                 std::size_t maximum_send_bytes = std::numeric_limits<std::size_t>::max());

} // namespace simaai::neat::peripherals_internal
