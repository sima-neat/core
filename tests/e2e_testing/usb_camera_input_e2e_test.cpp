// groups::UsbCameraInput on a DevKit USB camera. Set SIMA_NEAT_TEST_USB_CAMERA to an ID
// from usb_camera_ids(); the camera must offer MJPEG 1920x1080 at 30 fps. The decoder
// runs with copy-allowed=false, so a frame that needed a copy would stop the stream:
// every pulled frame was decoded in place.
#include "nodes/common/Output.h"
#include "nodes/groups/UsbCameraInput.h"
#include "pipeline/Graph.h"
#include "test_utils.h"

#include <cstdlib>
#include <iostream>
#include <stdexcept>
#include <string>

int main() {
  const char* id = std::getenv("SIMA_NEAT_TEST_USB_CAMERA");
  if (!id || !*id)
    return skip_test("set SIMA_NEAT_TEST_USB_CAMERA to a USB camera ID from usb_camera_ids()");
  using simaai::neat::nodes::groups::UsbCameraInput;
  try {
    simaai::neat::UsbCameraInputOptions opt;
    opt.camera_name = id;
    // Two builds: stopping the first must release the camera and its capture pool.
    for (int round = 1; round <= 2; ++round) {
      simaai::neat::Graph graph("usb_camera_e2e");
      graph.add(UsbCameraInput(opt));
      graph.add(simaai::neat::nodes::Output());
      simaai::neat::Run run = graph.build();
      for (int frame = 0; frame < 90; ++frame) {
        const auto sample = run.pull(5000);
        if (!sample)
          return fail_test("round " + std::to_string(round) + ": no frame " +
                           std::to_string(frame) + " within 5 s");
        if (frame == 0) {
          require_contains(sample->caps_string, "format=(string)NV12", "NV12 output");
          require_contains(sample->caps_string, "width=(int)1920", "requested width");
          require_contains(sample->caps_string, "height=(int)1080", "requested height");
        }
      }
      run.stop();
    }
    opt.framerate_num = 1000;
    try {
      (void)UsbCameraInput(opt);
      return fail_test("a mode the camera lacks must be rejected");
    } catch (const std::runtime_error& e) {
      require_contains(e.what(), "has no MJPEG mode", "the mode error lists the camera's modes");
    }
  } catch (const std::exception& e) {
    return fail_test(e.what());
  }
  std::cout << "[OK] usb_camera_input_e2e_test\n";
  return 0;
}
