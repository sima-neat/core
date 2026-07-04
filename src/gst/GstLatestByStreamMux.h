#pragma once

#include <cstdint>
#include <string>

typedef struct _GstBuffer GstBuffer;

namespace simaai::neat {

struct Sample;

namespace pipeline_internal {

// Release the realtime mux credit associated with a frame after the downstream
// graph has produced its output.  This is intentionally internal: users still
// see a normal graph Output, while the fused live path gets a true completion
// signal instead of guessing from GstBuffer refcounts.
void release_latest_by_stream_mux_loan(const std::string& stream_id, std::int64_t frame_id);
bool release_latest_by_stream_mux_loan_for_buffer(GstBuffer* buffer);
void release_latest_by_stream_mux_loan_for_sample(const Sample& sample);

} // namespace pipeline_internal
} // namespace simaai::neat

namespace simaai::neat {

bool register_latest_by_stream_mux();

} // namespace simaai::neat
