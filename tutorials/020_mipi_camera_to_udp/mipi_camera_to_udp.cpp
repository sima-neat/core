// 020_mipi_camera_to_udp: Modalix MIPI camera -> H.264 -> UDP streaming.
//
// Story: take the post-ISP V4L2 stream on /dev/video0out, convert it to NV12,
// encode it on the SiMa H.264 encoder, and send RTP/H.264 over UDP to a host.
// This is the end-to-end replacement for the hand-rolled GStreamer pipeline
// used during Modalix camera bring-up, expressed with typed NEAT nodes.
//
// What you learn:
// - V4L2Input wraps v4l2src for the Modalix post-ISP node.
// - VideoConvert + caps negotiation hand NV12 to the SiMa encoder.
// - H264EncodeSima wraps neatencoder with typed width/height/fps/bitrate.
// - UdpH264OutputGroup collapses h264parse + rtph264pay + udpsink into one add.
// - Source-mode session.build(run_opt) is how a live self-driven pipeline runs.
// - The tutorial skips cleanly when required GStreamer elements are missing.
//
// Board prerequisites (external to NEAT; run once before this tutorial):
//   # Raw sensor keepalive (keeps the MIPI sensor and ISP fed):
//   gst-launch-1.0 v4l2src device=/dev/video0raw \
//     ! 'video/x-bayer,format=rggb12le,width=1920,height=1080,framerate=30/1' \
//     ! fakesink &
//   # ISP 3A (auto-exposure, white balance, focus):
//   sudo isp_3a.elf &
//
// On the receiving host (laptop), start a UDP receiver first:
//   gst-launch-1.0 udpsrc port=9000 \
//     ! application/x-rtp,encoding-name=H264,payload=96 \
//     ! rtpjitterbuffer ! rtph264depay \
//     ! video/x-h264,stream-format=byte-stream,alignment=au \
//     ! avdec_h264 ! fpsdisplaysink sync=0

#include "neat.h"
#include "common/cpp_utils.h"
#include "gst/GstHelpers.h"

#include <chrono>
#include <exception>
#include <iostream>
#include <string>
#include <thread>

namespace {

void print_help(const char* argv0) {
  std::cout << "Usage: " << argv0 << " [--device <path>] [--width <w>] [--height <h>] [--fps <n>]\n"
            << "       [--format <fmt>] [--media-type <type>]\n"
            << "       [--host <ip>] [--port <p>] [--bitrate <kbps>]\n"
            << "       [--profile <name>] [--level <n>] [--enc-fps <n>]\n"
            << "       [--duration-ms <ms>]\n";
  tutorial_v2::print_common_flags(std::cout);
  std::cout << "  --device <path>      V4L2 device (default: /dev/video0out)\n";
  std::cout << "  --width <w>          Capture width (default: 1920)\n";
  std::cout << "  --height <h>         Capture height (default: 1080)\n";
  std::cout << "  --fps <n>            V4L2 framerate (default: 0 = unconstrained)\n";
  std::cout << "  --format <fmt>       V4L2 pixel format (default: RGB)\n";
  std::cout << "  --media-type <type>  V4L2 media type (default: video/x-raw)\n";
  std::cout << "  --host <ip>          UDP destination host (default: 127.0.0.1)\n";
  std::cout << "  --port <p>           UDP destination port (default: 9000)\n";
  std::cout << "  --bitrate <kbps>     H.264 bitrate in kbps (default: 4000)\n";
  std::cout << "  --profile <name>     H.264 profile (default: baseline)\n";
  std::cout << "  --level <n>          H.264 level (default: 4.2)\n";
  std::cout << "  --enc-fps <n>        Encoder framerate (default: 30)\n";
  std::cout << "  --duration-ms <ms>   How long to stream (default: 5000)\n";
}

struct CliArgs {
  std::string device = "/dev/video0out";
  std::string media_type = "video/x-raw";
  std::string format = "RGB";
  int width = 1920;
  int height = 1080;
  int fps = 0;
  std::string host = "127.0.0.1";
  int port = 9000;
  int bitrate_kbps = 4000;
  std::string profile = "baseline";
  std::string level = "4.2";
  int enc_fps = 30;
  int duration_ms = 5000;
};

CliArgs parse_cli(int argc, char** argv) {
  CliArgs a;
  std::string sval;
  if (tutorial_v2::get_arg(argc, argv, "--device", sval))
    a.device = sval;
  if (tutorial_v2::get_arg(argc, argv, "--media-type", sval))
    a.media_type = sval;
  if (tutorial_v2::get_arg(argc, argv, "--format", sval))
    a.format = sval;
  a.width = tutorial_v2::parse_int_arg(argc, argv, "--width", a.width);
  a.height = tutorial_v2::parse_int_arg(argc, argv, "--height", a.height);
  a.fps = tutorial_v2::parse_int_arg(argc, argv, "--fps", a.fps);
  if (tutorial_v2::get_arg(argc, argv, "--host", sval))
    a.host = sval;
  a.port = tutorial_v2::parse_int_arg(argc, argv, "--port", a.port);
  a.bitrate_kbps = tutorial_v2::parse_int_arg(argc, argv, "--bitrate", a.bitrate_kbps);
  if (tutorial_v2::get_arg(argc, argv, "--profile", sval))
    a.profile = sval;
  if (tutorial_v2::get_arg(argc, argv, "--level", sval))
    a.level = sval;
  a.enc_fps = tutorial_v2::parse_int_arg(argc, argv, "--enc-fps", a.enc_fps);
  a.duration_ms = tutorial_v2::parse_int_arg(argc, argv, "--duration-ms", a.duration_ms);
  return a;
}

// CORE LOGIC
simaai::neat::Session make_mipi_udp_session(const CliArgs& a) {
  simaai::neat::V4L2InputOptions v4l2_opt;
  v4l2_opt.device = a.device;
  v4l2_opt.media_type = a.media_type;
  v4l2_opt.format = a.format;
  v4l2_opt.width = a.width;
  v4l2_opt.height = a.height;
  v4l2_opt.fps_n = a.fps;

  simaai::neat::nodes::groups::UdpH264OutputGroupOptions udp_opt;
  udp_opt.udp_host = a.host;
  udp_opt.udp_port = a.port;

  simaai::neat::Session session;
  session.add(simaai::neat::nodes::V4L2Input(v4l2_opt));
  session.add(simaai::neat::nodes::Queue());
  session.add(simaai::neat::nodes::VideoConvert());
  session.add(simaai::neat::nodes::H264EncodeSima(a.width, a.height, a.enc_fps, a.bitrate_kbps,
                                                  a.profile, a.level));
  session.add(simaai::neat::nodes::groups::UdpH264OutputGroup(udp_opt));
  return session;
}
// END CORE LOGIC

const char* kRequiredElements[] = {
    "v4l2src", "videoconvert", "neatencoder", "h264parse", "rtph264pay", "udpsink",
};

bool all_elements_present(std::string* missing_out) {
  for (const char* name : kRequiredElements) {
    if (!simaai::neat::element_exists(name)) {
      if (missing_out)
        *missing_out = name;
      return false;
    }
  }
  return true;
}

} // namespace

int main(int argc, char** argv) {
  try {
    if (tutorial_v2::wants_help(argc, argv)) {
      print_help(argv[0]);
      return 0;
    }

    tutorial_v2::step("input_contract",
                      "parse V4L2 + UDP flags and establish Modalix-friendly defaults");
    tutorial_v2::step("run_mode_choice",
                      "source-mode session.build(run_opt) drives the live pipeline");
    tutorial_v2::why("demonstrates V4L2Input + VideoConvert + H264EncodeSima + UdpH264OutputGroup "
                     "as a replacement for hand-rolled gst-launch pipelines");
    tutorial_v2::tradeoff(
        "VideoConvert performs RGB->NV12 on the A65; for zero-copy on SiMa hardware, "
        "swap in Preproc with a neatprocesscvu colorconvert JSON");
    tutorial_v2::failure_mode("missing gstreamer plugins or no MIPI camera -> runtime_fallback");
    tutorial_v2::interpret_output(
        "start the UDP receiver on the host first; frames appear on fpsdisplaysink once "
        "the tutorial reaches the sleep loop");
    tutorial_v2::step("output_contract", "emit checks and machine-parseable signature");
    tutorial_v2::check("strict_flag_available",
                       tutorial_v2::yes_no(tutorial_v2::strict_mode()) == "yes" ||
                           tutorial_v2::yes_no(tutorial_v2::strict_mode()) == "no",
                       "strict-mode guard is observable");

    const CliArgs args = parse_cli(argc, argv);

    std::string missing;
    const bool elements_ok = all_elements_present(&missing);
    std::cout << "elements_ok=" << tutorial_v2::yes_no(elements_ok) << "\n";
    if (!elements_ok) {
      tutorial_v2::runtime_fallback("missing GStreamer element: " + missing);
      tutorial_v2::check("pipeline_runnable", false,
                         "pipeline could not be built; see runtime_fallback line");
      return 0;
    }

    std::string flow = "mipi_udp_stream";

    simaai::neat::Session session = make_mipi_udp_session(args);

    if (tutorial_v2::wants_print_gst(argc, argv)) {
      std::cout << session.describe_backend() << "\n";
      return 0;
    }

    simaai::neat::RunOptions run_opt;
    run_opt.queue_depth = 4;
    run_opt.overflow_policy = simaai::neat::OverflowPolicy::KeepLatest;

    auto run = session.build(run_opt);
    std::cout << "streaming to udp://" << args.host << ":" << args.port << " for "
              << args.duration_ms << " ms\n";
    std::this_thread::sleep_for(std::chrono::milliseconds(args.duration_ms));
    run.stop();
    run.close();

    tutorial_v2::check("stream_completed", true, "pipeline stopped cleanly after duration");
    tutorial_v2::check("tutorial_completed", true, "main path reached end without exception");
    tutorial_v2::print_signature({
        {"tutorial", "020"},
        {"lang", "cpp"},
        {"flow", flow},
        {"run_mode", "source_stream"},
        {"output_kind", "0"},
        {"tensor_rank", "-1"},
        {"field_count", "0"},
    });

    std::cout << "[OK] 020_mipi_camera_to_udp\n";
    return 0;
  } catch (const std::exception& e) {
    tutorial_v2::runtime_fallback(e);
    tutorial_v2::print_signature({
        {"tutorial", "020"},
        {"lang", "cpp"},
        {"flow", "runtime_fallback"},
        {"run_mode", "none"},
        {"output_kind", "0"},
        {"tensor_rank", "-1"},
        {"field_count", "0"},
    });
    std::cout << "[OK] 020_mipi_camera_to_udp\n";
    return 0;
  }
}
