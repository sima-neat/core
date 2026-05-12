/**
 * @file
 * @ingroup builder
 * @brief RTSP-side policy: legal port range for RTSP servers and clients.
 *
 * Bounds the TCP/UDP port numbers the framework will let RTSP-bearing nodes bind to.
 * Defaults exclude the privileged-port range (0-1023) so unprivileged processes can
 * bind without elevated capabilities.
 *
 * @see Policy.h
 * @see DefaultPolicy
 */
#pragma once

#include "policy/Policy.h"

namespace simaai::neat::policy {

/**
 * @brief RTSP-side policy parameters and validation.
 * @ingroup builder
 */
struct RtspPolicy {
  int min_port = 1024;  ///< Lowest allowed RTSP port (excludes privileged ports).
  int max_port = 65535; ///< Highest allowed RTSP port.

  /**
   * @brief Validate a requested RTSP port against the configured range.
   * @param port TCP/UDP port to bind.
   * @return `Evaluation` allowing the port or denying with a reason.
   */
  Evaluation validate_port(int port) const;
};

} // namespace simaai::neat::policy
