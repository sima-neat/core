#pragma once

#include <cstdint>
#include <string>

typedef struct _GstBuffer GstBuffer;
typedef struct _GstCaps GstCaps;
typedef struct _GstElement GstElement;
typedef struct _GstPad GstPad;

namespace simaai::neat {
namespace pipeline_internal {

// Complete the realtime mux timing/identity record after the downstream graph
// has produced its output. This also returns admission when a graph-owned raw
// completion boundary has not already done so. It is intentionally internal:
// users still see a normal graph Output.
void release_latest_by_stream_mux_loan(const std::string& stream_id, std::int64_t frame_id);
bool release_latest_by_stream_mux_loan_for_buffer(GstBuffer* buffer,
                                                  std::uint64_t namespace_hint = 0);
// Return only the decoder-backed raw-input admission credit at a graph-owned
// completion boundary. The registry/timing record remains live until the
// terminal output probe restores identity and consumes it.
bool release_latest_by_stream_mux_raw_input_credit_for_buffer(GstBuffer* buffer,
                                                              std::uint64_t namespace_hint);
// Buffer-finalize guards are only valid while downstream preserves the same
// GstBuffer (or copies this mux's lifecycle meta).  Fused hardware stages
// replace buffers without invoking arbitrary GstMeta transforms, so their
// terminal probe is the authoritative completion signal instead.
bool set_latest_by_stream_mux_lifetime_guard_enabled(GstElement* element, bool enabled);

} // namespace pipeline_internal
} // namespace simaai::neat

namespace simaai::neat {

bool register_latest_by_stream_mux();
std::uint64_t latest_by_stream_mux_namespace(GstElement* element);

} // namespace simaai::neat
