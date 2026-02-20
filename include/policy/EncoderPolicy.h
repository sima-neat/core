/**
 * @file
 * @ingroup policy
 * @brief Encoder policy checks.
 */
#pragma once

#include "policy/Policy.h"

namespace simaai::neat::policy {

struct EncoderPolicy {
  int min_bitrate_kbps = 100;
  int max_bitrate_kbps = 100000;

  Evaluation validate_bitrate(int bitrate_kbps) const;
};

} // namespace simaai::neat::policy
