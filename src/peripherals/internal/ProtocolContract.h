#pragma once

#include <cstddef>
#include <cstdint>

namespace simaai::neat::peripherals_internal {

inline constexpr std::uint32_t kPeripheralSchemaVersion = 1;
inline constexpr const char* kPeripheralSocketPath = "/run/simaai-sentinel/api.sock";
inline constexpr const char* kCatalogPath = "/v1/peripherals";
inline constexpr std::size_t kMaximumResponseBytes = 4 * 1024 * 1024;

} // namespace simaai::neat::peripherals_internal
