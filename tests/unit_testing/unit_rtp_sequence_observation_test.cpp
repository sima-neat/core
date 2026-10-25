#include "pipeline/internal/RtpSequenceObservation.h"

#include <stdexcept>

void require(bool condition) {
  if (!condition)
    throw std::runtime_error("RTP sequence observation mismatch");
}

int main() {
  using simaai::neat::pipeline_internal::RtpSequenceObservation;
  RtpSequenceObservation observation;
  auto first = observation.observe(1, 65534);
  require(first.forward_gap == 0 && !first.ssrc_changed);
  require(observation.observe(1, 65535).forward_gap == 0);
  require(observation.observe(1, 0).forward_gap == 0);
  require(observation.observe(1, 3).forward_gap == 2);
  require(observation.observe(1, 2).reordered_or_duplicate);
  require(observation.observe(1, 3).reordered_or_duplicate);
  require(observation.observe(1, 4).forward_gap == 0);
  auto changed = observation.observe(2, 12000);
  require(changed.ssrc_changed && changed.forward_gap == 0);
  require(observation.observe(2, 12001).forward_gap == 0);
}
