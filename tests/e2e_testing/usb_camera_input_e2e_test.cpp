// nodes::UsbCameraInput on a DevKit USB camera. Set SIMA_NEAT_TEST_USB_CAMERA to an ID
// from usb_camera_ids(); the camera must offer MJPEG 1920x1080 at 30 fps. The camera's JPEGs
// are pulled as they are, then fed to a VideoSender passthrough and a SimaDecode on one Graph;
// that decoder copies each frame.
#include "nodes/common/Output.h"
#include "nodes/groups/VideoSender.h"
#include "nodes/io/UsbCameraInput.h"
#include "nodes/sima/SimaDecode.h"
#include "pipeline/FormatSpec.h"
#include "pipeline/Graph.h"
#include "pipeline/GraphOptions.h"
#include "test_utils.h"

#include <cstdlib>
#include <iostream>
#include <stdexcept>
#include <string>

int main() {
  const char* id = std::getenv("SIMA_NEAT_TEST_USB_CAMERA");
  if (!id || !*id)
    return skip_test("set SIMA_NEAT_TEST_USB_CAMERA to a USB camera ID from usb_camera_ids()");
  using simaai::neat::nodes::UsbCameraInput;
  try {
    simaai::neat::UsbCameraInputOptions opt;
    opt.camera_name = id;
    const auto info = simaai::neat::usb_camera_info(opt);
    const std::string format = simaai::neat::format_tag_to_string(info.decoded_format);
    require(format == "NV12" || format == "NV16", "decoded format is NV12 or NV16: " + format);
    std::cout << "[INFO] " << id << " at " << info.device << " decodes to " << format << "\n";
    // Two builds: stopping the first must release the camera.
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
          require_contains(sample->caps_string, "image/jpeg", "JPEG output");
          require_contains(sample->caps_string, "width=(int)1920", "requested width");
          require_contains(sample->caps_string, "height=(int)1080", "requested height");
        }
      }
      run.stop();
    }
    // The passthrough pattern: the camera's JPEGs go unchanged to a VideoSender (nothing
    // needs to listen) and through SimaDecode to the app.
    namespace groups = simaai::neat::nodes::groups;
    simaai::neat::Graph source("usb_camera_jpeg");
    source.add(UsbCameraInput(opt));
    simaai::neat::SimaDecodeOptions decode;
    decode.type = simaai::neat::SimaDecodeType::MJPEG;
    decode.out_format = info.decoded_format;
    decode.dec_width = 1920;
    decode.dec_height = 1080;
    decode.dec_fps = 30;
    simaai::neat::Graph decoder("usb_camera_decode");
    decoder.add(simaai::neat::nodes::SimaDecode(decode));
    decoder.add(simaai::neat::nodes::Output());
    auto sender = groups::VideoSenderOptions::Passthrough(groups::RtspCodec::MJPEG);
    sender.host = "127.0.0.1";
    simaai::neat::GraphLinkOptions video_link;
    video_link.policy = simaai::neat::GraphLinkPolicy::RealtimeLatestByStream;
    simaai::neat::Graph split("usb_camera_jpeg_split");
    split.connect(source, decoder);
    split.connect(source, groups::VideoSender(sender), video_link);
    simaai::neat::Run run = split.build();
    for (int frame = 0; frame < 60; ++frame) {
      const auto sample = run.pull(5000);
      if (!sample)
        return fail_test("Jpeg split: no decoded frame " + std::to_string(frame) + " within 5 s");
      require_contains(sample->caps_string, "format=(string)" + format,
                       "decoded frames in the camera's native layout");
      require_contains(sample->caps_string, "width=(int)1920", "decoded width");
      require_contains(sample->caps_string, "height=(int)1080", "decoded height");
    }
    run.stop();
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
