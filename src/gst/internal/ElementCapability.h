/**
 * @file
 * @brief Private helpers for querying typed, read-only GStreamer capabilities.
 */
#pragma once

#include <optional>

namespace simaai::neat::internal {

/**
 * Read a boolean element property without changing element state.
 *
 * `std::nullopt` means the factory/property is absent or is not boolean.  This
 * distinction lets callers safely reject old plugins instead of assuming a
 * capability from the plugin version string.
 */
std::optional<bool> element_boolean_capability(const char* factory, const char* property);

} // namespace simaai::neat::internal
