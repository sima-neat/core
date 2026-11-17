// Executable-only negative control. Simulate a successful registration call
// that did not register the required factory. Production has no test switch.
#include <simaai/scheduler/client/capabilities.h>
namespace simaai::scheduler::client {
bool register_kernel_runner() {
  return true;
}
} // namespace simaai::scheduler::client
