#pragma once

#include "peripherals/PeripheralCatalog.h"

#include <chrono>
#include <cstddef>
#include <limits>
#include <string>

namespace simaai::neat::peripherals_internal {

inline constexpr std::size_t kMaximumResponseBytes = 4 * 1024 * 1024;

// maximum_send_bytes lets tests force partial writes; production uses the default.
peripherals::Catalog
list_from_socket(const std::string& socket_path, std::chrono::milliseconds timeout,
                 std::size_t maximum_send_bytes = std::numeric_limits<std::size_t>::max());

} // namespace simaai::neat::peripherals_internal
