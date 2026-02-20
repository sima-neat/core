/**
 * @file
 * @ingroup contracts
 * @brief Contract enums for memory and caps constraints.
 */
#pragma once

#include <string>

namespace simaai::neat {

// =============================
// Memory contract
// =============================
enum class MemoryContract {
  // Must be CPU-mappable (typically SystemMemory); violations are hard errors.
  RequireSystemMemoryMappable = 0,

  // Prefer device/zero-copy (runner may avoid forcing SystemMemory); still report contract
  // mismatches.
  PreferDeviceZeroCopy,

  // Allow either; if non-mappable, return empty payload with explicit reason + location.
  AllowEitherButReport,
};

// =============================
// Caps helper types
// =============================
enum class CapsMemory {
  Any = 0,
  SystemMemory,
};

} // namespace simaai::neat
