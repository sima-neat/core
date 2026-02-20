/**
 * @file
 * @ingroup policy
 * @brief Memory policy checks.
 */
#pragma once

#include "policy/Policy.h"

#include <cstddef>

namespace simaai::neat::policy {

struct MemoryPolicy {
  std::size_t max_pool_bytes = 1ULL << 30; // 1 GiB

  Evaluation validate_pool_bytes(std::size_t requested_bytes) const;
};

} // namespace simaai::neat::policy
