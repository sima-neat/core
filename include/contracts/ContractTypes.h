/**
 * @file
 * @ingroup contracts
 * @brief Contract enums for memory and caps constraints.
 *
 * Vocabulary types used across the contracts and builder layers to express
 * what kind of memory a Node accepts/emits and what gst caps memory class to
 * advertise. Kept enum-only so the header is cheap to include and STL-pure.
 */
#pragma once

#include <string>

namespace simaai::neat {

// =============================
// Memory contract
// =============================

/**
 * @brief How a Node (or Graph) wants buffer memory to be sourced/handed back.
 *
 * Drives the runner's behavior at output time and contributes to caps
 * negotiation: requiring CPU-mappable memory forces a (potentially copying)
 * conversion, while `PreferDeviceZeroCopy` lets the runner keep device
 * memory whenever possible.
 *
 * @ingroup contracts
 */
enum class MemoryContract {
  /// Must be CPU-mappable (typically SystemMemory); violations are hard errors.
  RequireSystemMemoryMappable = 0,

  /// Prefer device/zero-copy; runner may avoid forcing SystemMemory but still reports mismatches.
  PreferDeviceZeroCopy,

  /// Allow either; if non-mappable, return empty payload with explicit reason + location.
  AllowEitherButReport,
};

// =============================
// Caps helper types
// =============================

/**
 * @brief Memory class to advertise in GStreamer caps.
 *
 * `Any` means "don't constrain in caps"; `SystemMemory` forces the caps to
 * carry `memory:SystemMemory` so upstream/downstream negotiation picks
 * CPU-mappable buffers.
 *
 * @ingroup contracts
 */
enum class CapsMemory {
  Any = 0,      ///< Don't constrain memory class in caps.
  SystemMemory, ///< Force caps to advertise `memory:SystemMemory`.
};

} // namespace simaai::neat
