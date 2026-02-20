/**
 * @file
 * @ingroup policy
 * @brief Decoder policy checks.
 */
#pragma once

#include "policy/Policy.h"

namespace simaai::neat::policy {

struct DecoderPolicy {
  int max_width = 3840;
  int max_height = 2160;
  bool allow_dynamic_caps = true;

  Evaluation validate_resolution(int width, int height) const;
};

} // namespace simaai::neat::policy
