#pragma once

namespace sima_test::output_contract {

// Call after Gst initialization, before requesting named Runtime storage.
bool initialize_runtime_segment_allocator_fixture();

} // namespace sima_test::output_contract
