#include "runtime_segment_allocator_fixture.h"

#include <gstsimaaisegmentallocator.h>

namespace sima_test::output_contract {

bool initialize_runtime_segment_allocator_fixture() {
  gst_neat_segment_memory_init_once();
  GstAllocator* allocator = gst_neat_memory_get_segment_allocator();
  if (!allocator) {
    return false;
  }
  gst_object_unref(allocator);
  return true;
}

} // namespace sima_test::output_contract
