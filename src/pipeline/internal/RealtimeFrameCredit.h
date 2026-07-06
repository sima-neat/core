#pragma once
#ifndef SIMA_NEAT_INTERNAL
#error "Internal header. Not part of the public API."
#endif

#include "pipeline/GraphOptions.h"

#include <cstdint>
#include <string>
#include <vector>

namespace simaai::neat::pipeline_internal {

struct RealtimeFrameCredit {
  std::uint64_t namespace_id = 0;
  std::string stream_id;
  std::int64_t frame_id = -1;
};

std::vector<RealtimeFrameCredit> realtime_frame_credits_for_sample(const Sample& sample);
void release_realtime_frame_credits(const std::vector<RealtimeFrameCredit>& credits,
                                    const char* mode);
void release_realtime_frame_credits_for_sample(const Sample& sample, const char* mode);

} // namespace simaai::neat::pipeline_internal
