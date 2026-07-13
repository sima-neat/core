#pragma once

#include <cstdint>
#include <string>

typedef struct _GstBuffer GstBuffer;
typedef struct _GstCaps GstCaps;
typedef struct _GstElement GstElement;

namespace simaai::neat {
namespace pipeline_internal {

// Release the realtime mux credit associated with a frame after the downstream
// graph has produced its output.  This is intentionally internal: users still
// see a normal graph Output, while the fused live path gets a true completion
// signal instead of guessing from GstBuffer refcounts.
void release_latest_by_stream_mux_loan(const std::string& stream_id, std::int64_t frame_id);
bool release_latest_by_stream_mux_loan_for_buffer(GstBuffer* buffer,
                                                  std::uint64_t namespace_hint = 0);
void dispatch_latest_by_stream_encoded_frame_for_buffer(GstBuffer* buffer, GstCaps* caps,
                                                        const char* stream_id);
bool latest_by_stream_encoded_frame_callback_enabled();

} // namespace pipeline_internal
} // namespace simaai::neat

namespace simaai::neat {

bool register_latest_by_stream_mux();
std::uint64_t latest_by_stream_mux_namespace(GstElement* element);

} // namespace simaai::neat
