// Executable-only negative control. Simulate a successful registration call
// that did not register the required factory. Production has no test switch.
#include <simaai/runtime/gstreamer/stage.h>
namespace simaai::runtime::gstreamer {
bool register_element(GstPlugin*) {
  return true;
}
} // namespace simaai::runtime::gstreamer
