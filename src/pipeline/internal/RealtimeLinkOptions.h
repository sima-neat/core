#pragma once

#include "pipeline/GraphOptions.h"

#include <algorithm>

namespace simaai::neat::pipeline_internal {

inline constexpr int kDefaultRealtimeMaxInflightPerStream = 4;
inline constexpr int kDefaultRealtimeMaxInflightTotal = 8;

inline int
resolved_realtime_max_inflight_per_stream(const RealtimeGraphLinkOptions& options) noexcept {
  return options.max_inflight_per_stream > 0 ? options.max_inflight_per_stream
                                             : kDefaultRealtimeMaxInflightPerStream;
}

inline int default_realtime_max_inflight_total(int total_per_stream_capacity) noexcept {
  return std::min(std::max(0, total_per_stream_capacity), kDefaultRealtimeMaxInflightTotal);
}

} // namespace simaai::neat::pipeline_internal
