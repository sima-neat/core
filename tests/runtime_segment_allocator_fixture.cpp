#include "runtime_segment_allocator_fixture.h"

#include <simaai/scheduler/testing/segment_allocator.h>

namespace sima_test::output_contract {

bool initialize_runtime_segment_allocator_fixture() {
  return simaai::scheduler::testing::initialize_segment_allocator();
}

} // namespace sima_test::output_contract
