#include "policy/DecoderPolicy.h"

#include <string>

namespace simaai::neat::policy {

Evaluation DecoderPolicy::validate_resolution(int width, int height) const {
  if (width <= 0 || height <= 0) {
    return {Decision::Deny, "decoder_policy: width/height must be > 0"};
  }
  if (width > max_width || height > max_height) {
    return {Decision::Deny, "decoder_policy: resolution exceeds maximum bounds"};
  }
  return {Decision::Allow, "decoder_policy: accepted"};
}

} // namespace simaai::neat::policy
