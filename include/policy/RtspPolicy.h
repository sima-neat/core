/**
 * @file
 * @ingroup policy
 * @brief RTSP policy checks.
 */
#pragma once

#include "policy/Policy.h"

namespace simaai::neat::policy {

struct RtspPolicy {
  int min_port = 1024;
  int max_port = 65535;

  Evaluation validate_port(int port) const;
};

} // namespace simaai::neat::policy
