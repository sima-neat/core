/**
 * @file
 * @ingroup builder
 * @brief Memory-side policy: ceiling on buffer-pool allocations.
 *
 * Caps the total bytes the framework will let a single buffer pool reserve. Used by the
 * memory-aware nodes (PCIe sink/src, MLA-shared paths) to refuse oversized requests
 * before they reach the allocator. Default is 1 GiB.
 *
 * @see Policy.h
 * @see DefaultPolicy
 */
#pragma once

#include "policy/Policy.h"

#include <cstddef>

namespace simaai::neat::policy {

/**
 * @brief Memory-pool policy parameters and validation.
 * @ingroup builder
 */
struct MemoryPolicy {
  std::size_t max_pool_bytes = 1ULL << 30; ///< Maximum bytes a single pool may request (default 1 GiB).

  /**
   * @brief Validate a requested pool size against the configured ceiling.
   * @param requested_bytes Pool size in bytes.
   * @return `Evaluation` allowing the request or denying with a reason.
   */
  Evaluation validate_pool_bytes(std::size_t requested_bytes) const;
};

} // namespace simaai::neat::policy
