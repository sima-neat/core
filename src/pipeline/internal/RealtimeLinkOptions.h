#pragma once

#include "pipeline/GraphOptions.h"

namespace simaai::neat::pipeline_internal {

inline constexpr int kDefaultRealtimeMaxInflightPerStream = 4;

inline int resolved_realtime_max_inflight_per_stream(const GraphLinkOptions& options) noexcept {
  return options.max_inflight_per_stream > 0 ? options.max_inflight_per_stream
                                             : kDefaultRealtimeMaxInflightPerStream;
}

} // namespace simaai::neat::pipeline_internal
