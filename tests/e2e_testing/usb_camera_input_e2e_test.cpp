// nodes::UsbCameraInput on a DevKit USB camera. Set SIMA_NEAT_TEST_USB_CAMERA to an ID
// from usb_camera_ids(); the camera must offer MJPEG 1920x1080 at 30 fps. The camera's JPEGs
// are pulled as they are, then fed to a VideoSender passthrough and a SimaDecode on one Graph;
// that decoder reads each frame in place from the camera's CMA buffer. When the camera lists a
// YUYV mode with an even width and an integer frame rate, its packed frames also feed a model's
// Preproc and, through ColorConvert, a strict H.264 VideoSender.
#include "asset_utils.h"
#include "model/Model.h"
#include "nodes/common/Output.h"
#include "nodes/groups/VideoSender.h"
#include "nodes/io/UsbCameraInput.h"
#include "nodes/sima/ImageTransform.h"
#include "nodes/sima/Preproc.h"
#include "nodes/sima/SimaDecode.h"
#include "pipeline/FormatSpec.h"
#include "pipeline/Graph.h"
#include "pipeline/GraphOptions.h"
#include "test_utils.h"
#include "udp_test_utils.h"

#include <fcntl.h>
#include <linux/videodev2.h>
#include <sys/ioctl.h>
#include <unistd.h>

#include <cstdlib>
#include <iostream>
#include <optional>
#include <stdexcept>
#include <string>

namespace {

// The first YUYV mode `device` lists with an even width and an integer frame rate, which the
// strict H.264 viewer requires.
std::optional<simaai::neat::UsbCameraInputOptions>
yuyv_mode(const std::string& device, simaai::neat::UsbCameraInputOptions opt) {
  const int fd = ::open(device.c_str(), O_RDWR | O_CLOEXEC);
  if (fd < 0)
    throw std::runtime_error("cannot open " + device);
  std::optional<simaai::neat::UsbCameraInputOptions> mode;
  v4l2_frmsizeenum size{};
  size.pixel_format = V4L2_PIX_FMT_YUYV;
  for (; !mode && ::ioctl(fd, VIDIOC_ENUM_FRAMESIZES, &size) == 0; ++size.index) {
    if (size.type != V4L2_FRMSIZE_TYPE_DISCRETE || (size.discrete.width & 1U))
      continue;
    v4l2_frmivalenum ival{};
    ival.pixel_format = V4L2_PIX_FMT_YUYV;
    ival.width = size.discrete.width;
    ival.height = size.discrete.height;
    for (; ::ioctl(fd, VIDIOC_ENUM_FRAMEINTERVALS, &ival) == 0; ++ival.index) {
      if (ival.type != V4L2_FRMIVAL_TYPE_DISCRETE || ival.discrete.numerator != 1)
        continue;
      opt.format = simaai::neat::UsbCameraFormat::YUYV;
      opt.width = size.discrete.width;
      opt.height = size.discrete.height;
      opt.framerate_num = ival.discrete.denominator;
      opt.framerate_den = 1;
      mode = opt;
      break;
    }
  }
  ::close(fd);
  return mode;
}

// capture -> Preproc -> model, and capture -> ColorConvert -> strict H.264 VideoSender.
int run_yuyv(const simaai::neat::UsbCameraInputOptions& opt) {
  namespace neat = simaai::neat;
  const auto info = neat::usb_camera_info(opt);
  require(info.row_stride >= opt.width * 2 && !info.colorimetry.empty(), "YUYV capture info");
  std::cout << "[INFO] YUYV " << opt.width << "x" << opt.height << " at " << opt.framerate_num
            << " fps, colorimetry " << info.colorimetry << "\n";

  const auto model_path = sima_test::resolve_yolov8s_tar();
  require(!model_path.empty(), "missing model archive; set SIMA_YOLO_TAR or install yolo_v8s");
  neat::Model::Options model_options;
  model_options.preprocess.kind = neat::InputKind::Image;
  model_options.preprocess.enable = neat::AutoFlag::On;
  model_options.preprocess.color_convert.input_format = neat::PreprocessColorFormat::YUYV;
  model_options.preprocess.input_max_width = static_cast<int>(opt.width);
  model_options.preprocess.input_max_height = static_cast<int>(opt.height);
  const neat::Model model(model_path, model_options);
  {
    neat::Graph graph("usb_camera_yuyv_model");
    graph.add(neat::nodes::UsbCameraInput(opt, info));
    neat::PreprocOptions preproc(model);
    preproc.input_color = info.color;
    graph.add(neat::nodes::Preproc(preproc));
    graph.add(model.inference());
    graph.add(model.postprocess());
    graph.add(neat::nodes::Output());
    neat::Run run = graph.build();
    for (int frame = 0; frame < 30; ++frame) {
      if (!run.pull(5000))
        return fail_test("YUYV model: no result " + std::to_string(frame) + " within 5 s");
    }
    run.stop();
  }

  sima_test::UdpReceiver receiver;
  neat::ColorConvertOptions convert;
  convert.output_format = neat::ImageFormat::NV12;
  convert.surface = neat::ImageSurfaceLayout::EncoderNv12;
  convert.color = info.color;
  convert.color.output_siting = {neat::ImageChromaSiting::Cosited,
                                 neat::ImageChromaSiting::Centered};
  convert.input_colorimetry = info.colorimetry;
  auto sender = neat::nodes::groups::VideoSenderOptions::FromRaw(
      {.fps = static_cast<int>(opt.framerate_num),
       .input_policy = neat::EncoderInputPolicy::RequireDirect});
  sender.host = "127.0.0.1";
  sender.video_port_base = receiver.port();
  neat::Graph viewer("usb_camera_yuyv_h264");
  viewer.add(neat::nodes::UsbCameraInput(opt, info));
  viewer.add(neat::nodes::ColorConvert(convert));
  viewer.add(neat::nodes::groups::VideoSender(sender));
  neat::Run run = viewer.build();
  std::string packet;
  for (int i = 0; i < 30; ++i) {
    if (!receiver.recv_one(&packet, 5000))
      return fail_test("YUYV H.264: no RTP packet " + std::to_string(i) + " within 5 s");
    require(packet.size() > 12 && (static_cast<unsigned char>(packet[0]) >> 6) == 2,
            "RTP version 2 packet");
  }
  run.stop();
  return 0;
}

} // namespace

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
    source.add(UsbCameraInput(opt, info)); // probe once: reuse usb_camera_info's result
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
    opt.framerate_num = 30;
    if (const auto mode = yuyv_mode(info.device, opt)) {
      if (const int rc = run_yuyv(*mode))
        return rc;
    } else {
      std::cout << "[INFO] " << id << " lists no YUYV mode with an even width and integer fps\n";
    }
  } catch (const std::exception& e) {
    return fail_test(e.what());
  }
  std::cout << "[OK] usb_camera_input_e2e_test\n";
  return 0;
}
