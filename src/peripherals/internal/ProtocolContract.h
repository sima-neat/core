#pragma once

#include <cstddef>
#include <cstdint>

namespace simaai::neat::peripherals_internal {

inline constexpr std::uint32_t kPeripheralSchemaVersion = 1;
inline constexpr const char* kPeripheralApiVersion = "v1";
inline constexpr const char* kPeripheralSocketPath = "/run/simaai-peripherals/api.sock";
inline constexpr const char* kHealthPath = "/v1/health";
inline constexpr const char* kCatalogPath = "/v1/catalog";
inline constexpr const char* kEventsPath = "/v1/events";
inline constexpr const char* kRefreshPath = "/v1/refresh";
inline constexpr std::uint64_t kMaximumEventWaitMs = 30000;
inline constexpr std::size_t kMaximumResponseBytes = 4 * 1024 * 1024;

} // namespace simaai::neat::peripherals_internal
