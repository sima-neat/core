#include "policy/MemoryPolicy.h"

namespace simaai::neat::policy {

Evaluation MemoryPolicy::validate_pool_bytes(std::size_t requested_bytes) const {
  if (requested_bytes == 0) {
    return {Decision::Deny, "memory_policy: requested bytes must be > 0"};
  }
  if (requested_bytes > max_pool_bytes) {
    return {Decision::Deny, "memory_policy: requested bytes exceed configured limit"};
  }
  return {Decision::Allow, "memory_policy: accepted"};
}

} // namespace simaai::neat::policy
