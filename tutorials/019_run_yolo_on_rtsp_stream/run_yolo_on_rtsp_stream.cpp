// 019: Run YOLOv8 on decoded frames from a live RTSP stream.
//
// Usage:
//   tutorial_v2_019_run_yolo_on_rtsp_stream --url rtsp://host/path --mpk /path/to/yolo_v8s.tar.gz
//   [--frames 5]

#include "neat.h"
#include "nodes/groups/RtspDecodedInput.h"

#include <exception>
#include <iostream>
#include <stdexcept>
#include <string>

namespace {

bool get_arg(int argc, char** argv, const std::string& key, std::string& out) {
  for (int i = 1; i + 1 < argc; ++i) {
    if (key == argv[i]) {
      out = argv[i + 1];
      return true;
    }
  }
  return false;
}

int parse_int_arg(int argc, char** argv, const std::string& key, int def) {
  std::string v;
  if (!get_arg(argc, argv, key, v))
    return def;
  return std::stoi(v);
}

simaai::neat::Model::Options yolo_options() {
  simaai::neat::Model::Options opt;
  opt.format = "BGR";
  opt.decode_type = "yolov8";
  opt.input_max_width = 1920;
  opt.input_max_height = 1080;
  opt.input_max_depth = 3;
  return opt;
}

simaai::neat::nodes::groups::RtspDecodedInputOptions rtsp_options(const std::string& url) {
  simaai::neat::nodes::groups::RtspDecodedInputOptions opt;
  opt.url = url;
  opt.tcp = true;
  opt.out_format = "BGR";
  opt.output_caps.enable = true;
  opt.output_caps.format = "BGR";
  return opt;
}

} // namespace

int main(int argc, char** argv) {
  try {
    std::string url, mpk;
    if (!get_arg(argc, argv, "--url", url) || !get_arg(argc, argv, "--mpk", mpk)) {
      std::cerr << "Usage: tutorial_v2_019_run_yolo_on_rtsp_stream --url <rtsp://...> --mpk <path> "
                   "[--frames <n>]\n";
      return 1;
    }
    const int frames = parse_int_arg(argc, argv, "--frames", 5);

    // CORE LOGIC
    // Session 1: decode the RTSP stream into BGR frames.
    simaai::neat::Session rtsp;
    rtsp.add(simaai::neat::nodes::groups::RtspDecodedInput(rtsp_options(url)));
    rtsp.add(simaai::neat::nodes::Output());
    auto rtsp_run = rtsp.build(simaai::neat::RunOptions{});

    // Session 2: YOLOv8 model.
    simaai::neat::Model model(mpk, yolo_options());

    for (int i = 0; i < frames; ++i) {
      auto frame = rtsp_run.pull(/*timeout_ms=*/5000);
      if (!frame.has_value() || !frame->tensor.has_value()) {
        std::cout << "frame=" << i << " rtsp_timeout\n";
        break;
      }
      auto out = model.run(*frame->tensor, /*timeout_ms=*/2000);
      const auto bbox_bytes =
          out.tensor.has_value() ? static_cast<long long>(out.tensor->shape[0]) : -1LL;
      std::cout << "frame=" << i << " fields=" << out.fields.size() << " bbox_bytes=" << bbox_bytes
                << "\n";
    }
    // END CORE LOGIC

    return 0;
  } catch (const std::exception& e) {
    std::cerr << "[FAIL] " << e.what() << "\n";
    return 1;
  }
}
