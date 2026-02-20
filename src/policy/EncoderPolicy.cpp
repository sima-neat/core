#include "policy/EncoderPolicy.h"

namespace simaai::neat::policy {

Evaluation EncoderPolicy::validate_bitrate(int bitrate_kbps) const {
  if (bitrate_kbps < min_bitrate_kbps || bitrate_kbps > max_bitrate_kbps) {
    return {Decision::Deny, "encoder_policy: bitrate out of supported bounds"};
  }
  return {Decision::Allow, "encoder_policy: accepted"};
}

} // namespace simaai::neat::policy
