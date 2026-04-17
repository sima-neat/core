// tutorial_0014_rtsp_server.cpp
// Story: run a minimal RTSP server pipeline.
// What you learn:
// - StillImageInput feeds a static image into a live RTSP pipeline.
// - run_rtsp() starts a background RTSP server.
// - H264 encode + parse + pay are typical RTSP server components.

#include "neat/session.h"
#include "neat/nodes.h"
#include "gst/GstHelpers.h"

#include "tutorial_common.h"

#include <chrono>
#include <filesystem>
#include <iostream>
#include <string>
#include <thread>

namespace fs = std::filesystem;

namespace {

void print_help(const char* argv0) {
  std::cout << "Usage: " << argv0
            << " [--image <path>] [--port <p>] [--mount <name>] [--duration-ms <ms>]\n";
  sima_tutorial::print_common_flags(std::cout);
  std::cout << "  --image <path>       Image path (default: shipped tutorial sample)\n";
  std::cout << "  --port <p>           RTSP port (default: 8554)\n";
  std::cout << "  --mount <name>       RTSP mount (default: image)\n";
  std::cout << "  --duration-ms <ms>   How long to keep the server alive (default: 2000)\n";
  std::cout << "\nValidate with:\n";
  std::cout << "  gst-launch-1.0 -v rtspsrc location=rtsp://127.0.0.1:8554/image latency=0 \\\n";
  std::cout << "    ! rtph264depay ! h264parse ! avdec_h264 ! autovideosink\n";
}

int parse_int_arg(int argc, char** argv, const std::string& key, int def) {
  std::string val;
  if (!sima_tutorial::get_arg(argc, argv, key, val))
    return def;
  try {
    return std::stoi(val);
  } catch (...) {
    return def;
  }
}

fs::path default_image_path() {
  return sima_tutorial::find_asset_root() / "ilena_488.jpg";
}

} // namespace

int main(int argc, char** argv) {
  try {
    if (sima_tutorial::wants_help(argc, argv)) {
      print_help(argv[0]);
      return 0;
    }

    std::string image_arg;
    fs::path image_path = sima_tutorial::get_arg(argc, argv, "--image", image_arg)
                              ? fs::path(image_arg)
                              : default_image_path();
    if (!fs::exists(image_path)) {
      return sima_tutorial::skip("missing image for RTSP server");
    }

    if (!simaai::neat::element_exists("appsrc") || !simaai::neat::element_exists("rtph264pay") ||
        !simaai::neat::element_exists("h264parse") ||
        !simaai::neat::element_exists("neatencoder")) {
      return sima_tutorial::skip(
          "missing RTSP/H264 elements (appsrc/rtph264pay/h264parse/neatencoder)");
    }

    const int port = parse_int_arg(argc, argv, "--port", 8554);
    std::string mount = "image";
    sima_tutorial::get_arg(argc, argv, "--mount", mount);
    const int duration_ms = parse_int_arg(argc, argv, "--duration-ms", 2000);

    // StillImageInput expects even dimensions for NV12.
    const int content_w = 256;
    const int content_h = 256;
    const int enc_w = 256;
    const int enc_h = 256;
    const int fps = 30;

    simaai::neat::Session p;
    p.add(simaai::neat::nodes::StillImageInput(image_path.string(), content_w, content_h, enc_w,
                                               enc_h, fps));
    p.add(simaai::neat::nodes::H264EncodeSima(enc_w, enc_h, fps, /*bitrate_kbps=*/400, "baseline",
                                              "4.0"));
    p.add(simaai::neat::nodes::H264Parse(/*config_interval=*/1));
    p.add(simaai::neat::nodes::H264Packetize(/*pt=*/96, /*config_interval=*/1));

    if (sima_tutorial::wants_print_gst(argc, argv)) {
      std::cout << p.describe_backend() << "\n";
      return 0;
    }

    simaai::neat::RtspServerHandle server = p.run_rtsp({.mount = mount, .port = port});
    std::cout << "RTSP URL: " << server.url() << "\n";

    std::this_thread::sleep_for(std::chrono::milliseconds(duration_ms));
    server.stop();

    std::cout << "[OK] tutorial_0014 complete\n";
    return 0;
  } catch (const std::exception& e) {
    std::cerr << "[FAIL] " << e.what() << "\n";
    return 1;
  }
}
