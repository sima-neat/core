#include "policy/RtspPolicy.h"

namespace simaai::neat::policy {

Evaluation RtspPolicy::validate_port(int port) const {
  if (port < min_port || port > max_port) {
    return {Decision::Deny, "rtsp_policy: port out of supported range"};
  }
  return {Decision::Allow, "rtsp_policy: accepted"};
}

} // namespace simaai::neat::policy
