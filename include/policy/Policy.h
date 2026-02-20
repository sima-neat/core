/**
 * @file
 * @ingroup policy
 * @brief Minimal runtime policy contract.
 */
#pragma once

#include <string>

namespace simaai::neat::policy {

enum class Decision {
  Allow,
  Deny,
};

struct Evaluation {
  Decision decision = Decision::Allow;
  std::string reason;

  bool ok() const noexcept {
    return decision == Decision::Allow;
  }
};

} // namespace simaai::neat::policy
