#include "gst/GstLatestByStreamMux.h"
#include <simaai/runtime/gstreamer/elements.h>
namespace simaai::neat {
bool register_latest_by_stream_mux() {
  return simaai::runtime::gstreamer::register_latest_by_stream_mux();
}
std::uint64_t latest_by_stream_mux_namespace(GstElement* element) {
  return simaai::runtime::gstreamer::latest_by_stream_mux_namespace(element);
}
namespace pipeline_internal {
void release_latest_by_stream_mux_loan(const std::string& stream_id, std::int64_t frame_id) {
  simaai::runtime::gstreamer::release_latest_by_stream_mux_loan(stream_id, frame_id);
}
bool release_latest_by_stream_mux_loan_for_buffer(GstBuffer* buffer, std::uint64_t hint) {
  return simaai::runtime::gstreamer::release_latest_by_stream_mux_loan_for_buffer(buffer, hint);
}
bool release_latest_by_stream_mux_raw_input_credit_for_buffer(GstBuffer* buffer,
                                                              std::uint64_t hint) {
  return simaai::runtime::gstreamer::release_latest_by_stream_mux_raw_input_credit_for_buffer(
      buffer, hint);
}
bool set_latest_by_stream_mux_lifetime_guard_enabled(GstElement* element, bool enabled) {
  return simaai::runtime::gstreamer::set_latest_by_stream_mux_lifetime_guard_enabled(element,
                                                                                     enabled);
}
} // namespace pipeline_internal
} // namespace simaai::neat
