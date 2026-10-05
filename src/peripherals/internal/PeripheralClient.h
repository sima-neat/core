#pragma once

#include "peripherals/PeripheralCatalog.h"

#include <chrono>
#include <cstddef>
#include <string>

namespace simaai::neat::peripherals_internal {

inline constexpr std::size_t kMaximumResponseBytes = 4 * 1024 * 1024;

peripherals::Catalog list_from_socket(const std::string& socket_path,
                                      std::chrono::milliseconds timeout);

} // namespace simaai::neat::peripherals_internal
