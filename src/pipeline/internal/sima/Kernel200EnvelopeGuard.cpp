#include "pipeline/internal/sima/Kernel200EnvelopeGuard.h"

#include <sstream>

namespace simaai::neat::pipeline_internal::sima {

namespace {

void set_violation(Kernel200EnvelopeViolation* violation, std::string code, std::string message) {
  if (!violation) {
    return;
  }
  violation->code = std::move(code);
  violation->message = std::move(message);
}

} // namespace

bool validate_kernel200_envelope(const Kernel200EnvelopeLimits& limits,
                                 const Kernel200EnvelopeActual& actual,
                                 Kernel200EnvelopeViolation* violation) {
  if (violation) {
    violation->code.clear();
    violation->message.clear();
  }

  if (actual.actual_w <= 0 || actual.actual_h <= 0 || actual.actual_stride <= 0) {
    std::ostringstream oss;
    oss << "invalid actual dimensions (w=" << actual.actual_w << ", h=" << actual.actual_h
        << ", stride=" << actual.actual_stride << ")";
    set_violation(violation, "invalid_actual_dims", oss.str());
    return false;
  }

  if (limits.max_w > 0 && actual.actual_w > limits.max_w) {
    std::ostringstream oss;
    oss << "actual width " << actual.actual_w << " exceeds max " << limits.max_w;
    set_violation(violation, "actual_gt_max_w", oss.str());
    return false;
  }
  if (limits.max_h > 0 && actual.actual_h > limits.max_h) {
    std::ostringstream oss;
    oss << "actual height " << actual.actual_h << " exceeds max " << limits.max_h;
    set_violation(violation, "actual_gt_max_h", oss.str());
    return false;
  }
  if (limits.max_stride > 0 && actual.actual_stride > limits.max_stride) {
    std::ostringstream oss;
    oss << "actual stride " << actual.actual_stride << " exceeds max " << limits.max_stride;
    set_violation(violation, "actual_gt_max_stride", oss.str());
    return false;
  }

  if (limits.allocated_bytes > 0 && actual.required_bytes > limits.allocated_bytes) {
    std::ostringstream oss;
    oss << "required bytes " << actual.required_bytes << " exceeds allocated bytes "
        << limits.allocated_bytes;
    set_violation(violation, "required_gt_allocated_bytes", oss.str());
    return false;
  }

  return true;
}

} // namespace simaai::neat::pipeline_internal::sima

