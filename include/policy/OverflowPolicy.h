/**
 * @file
 * @ingroup pipeline
 * @brief Overflow behavior shared by bounded Neat queues.
 */
#pragma once

namespace simaai::neat {

/**
 * @brief What a bounded queue does when it is full.
 *
 * The right choice depends on the input source. File batches normally block,
 * while live sources may prefer freshness over completeness.
 */
enum class OverflowPolicy {
  Block = 0,    ///< Wait for space and preserve every item.
  KeepLatest,   ///< Drop the oldest queued item to make room.
  DropIncoming, ///< Drop the new item and preserve what is already queued.
};

} // namespace simaai::neat
