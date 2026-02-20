#pragma once
#ifndef SIMA_NEAT_INTERNAL
#error "Internal header. Not part of the public API."
#endif

#include <cstddef>
#include <string>

namespace simaai::neat::pipeline_internal::sima {

struct Kernel200EnvelopeLimits {
  int max_w = 0;
  int max_h = 0;
  int max_stride = 0;
  std::size_t allocated_bytes = 0;
};

struct Kernel200EnvelopeActual {
  int actual_w = 0;
  int actual_h = 0;
  int actual_stride = 0;
  std::size_t required_bytes = 0;
};

struct Kernel200EnvelopeViolation {
  std::string code;
  std::string message;
};

bool validate_kernel200_envelope(const Kernel200EnvelopeLimits& limits,
                                 const Kernel200EnvelopeActual& actual,
                                 Kernel200EnvelopeViolation* violation = nullptr);

} // namespace simaai::neat::pipeline_internal::sima

