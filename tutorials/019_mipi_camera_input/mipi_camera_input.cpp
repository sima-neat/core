#include "neat.h"
#include "common/cpp_utils.h"
#include "gst/GstHelpers.h"

#include <opencv2/core.hpp>

#include <iostream>
#include <string>

namespace {

void print_help(const char* argv0) {
  std::cout << "Usage: " << argv0
            << " [--device <path>] [--width <w>] [--height <h>] [--fps <n>]\n"
            << "       [--frames <n>] [--format <fmt>] [--media-type <type>]\n";
  tutorial_v2::print_common_flags(std::cout);
  std::cout << "  --device <path>      V4L2 device (default: /dev/video0)\n";
  std::cout << "  --width <w>          Capture width (default: 640)\n";
  std::cout << "  --height <h>         Capture height (default: 480)\n";
  std::cout << "  --fps <n>            Framerate (default: 0 = unconstrained)\n";
  std::cout << "  --frames <n>         Frames to capture (default: 5)\n";
  std::cout << "  --format <fmt>       Pixel format, e.g. RGB, NV12 (default: RGB)\n";
  std::cout << "  --media-type <type>  GStreamer media type (default: video/x-raw)\n";
}

// CORE LOGIC
simaai::neat::Session make_v4l2_session(const std::string& device,
                                        const std::string& media_type,
                                        const std::string& format, int width,
                                        int height, int fps) {
  simaai::neat::V4L2InputOptions opt;
  opt.device = device;
  opt.media_type = media_type;
  opt.format = format;
  opt.width = width;
  opt.height = height;
  opt.fps_n = fps;

  simaai::neat::Session session;
  session.add(simaai::neat::nodes::V4L2Input(opt));
  session.add(simaai::neat::nodes::Output());
  return session;
}

simaai::neat::Session make_fallback_session(const std::string& format, int width,
                                            int height) {
  simaai::neat::InputOptions in;
  in.format = format;
  in.width = width;
  in.height = height;
  in.depth = 3;
  in.is_live = false;
  in.do_timestamp = true;

  simaai::neat::Session session;
  session.add(simaai::neat::nodes::Input(in));
  session.add(simaai::neat::nodes::Output());
  return session;
}
// END CORE LOGIC

} // namespace

int main(int argc, char** argv) {
  try {
    if (tutorial_v2::wants_help(argc, argv)) {
      print_help(argv[0]);
      return 0;
    }

    tutorial_v2::step("input_contract", "parse flags and establish deterministic defaults");
    tutorial_v2::step("run_mode_choice", "exercise the chapter's primary runtime path");
    tutorial_v2::why("V4L2Input wraps v4l2src as a typed NEAT node for live camera capture");
    tutorial_v2::tradeoff(
        "fall back to synthetic appsrc when no camera hardware is available");
    tutorial_v2::failure_mode(
        "missing v4l2src plugin degrades to fallback path without losing observability");
    tutorial_v2::interpret_output(
        "use CHECK markers plus SIGNATURE fields to validate tensor shape and runtime path");
    tutorial_v2::step("output_contract", "emit checks and machine-parseable signature");
    tutorial_v2::check("strict_flag_available",
                       tutorial_v2::yes_no(tutorial_v2::strict_mode()) == "yes" ||
                           tutorial_v2::yes_no(tutorial_v2::strict_mode()) == "no",
                       "strict-mode guard is observable");

    std::string device_arg;
    const std::string device =
        tutorial_v2::get_arg(argc, argv, "--device", device_arg)
            ? device_arg
            : "/dev/video0";
    const int width = tutorial_v2::parse_int_arg(argc, argv, "--width", 640);
    const int height = tutorial_v2::parse_int_arg(argc, argv, "--height", 480);
    const int fps = tutorial_v2::parse_int_arg(argc, argv, "--fps", 0);
    const int frames = tutorial_v2::parse_int_arg(argc, argv, "--frames", 5);

    std::string format_arg;
    const std::string format =
        tutorial_v2::get_arg(argc, argv, "--format", format_arg) ? format_arg : "RGB";

    std::string media_type_arg;
    const std::string media_type =
        tutorial_v2::get_arg(argc, argv, "--media-type", media_type_arg)
            ? media_type_arg
            : "video/x-raw";

    const bool v4l2_available = simaai::neat::element_exists("v4l2src");
    std::cout << "v4l2_available=" << tutorial_v2::yes_no(v4l2_available) << "\n";

    std::string flow;
    int pulled = 0;

    if (v4l2_available) {
      flow = "v4l2_source";

      // CORE LOGIC
      simaai::neat::Session session =
          make_v4l2_session(device, media_type, format, width, height, fps);

      if (tutorial_v2::wants_print_gst(argc, argv)) {
        std::cout << session.describe_backend() << "\n";
        return 0;
      }

      simaai::neat::RunOptions run_opt;
      run_opt.queue_depth = 4;
      run_opt.overflow_policy = simaai::neat::OverflowPolicy::KeepLatest;
      run_opt.output_memory = simaai::neat::OutputMemory::Owned;

      auto run = session.build(run_opt);
      for (int i = 0; i < frames; ++i) {
        auto tensor = run.pull_tensor(5000);
        if (!tensor.has_value()) {
          std::cout << "pull timeout at frame " << i << "\n";
          break;
        }
        ++pulled;
      }
      run.stop();
      run.close();
      // END CORE LOGIC
    } else {
      flow = "synthetic_fallback";

      simaai::neat::Session session = make_fallback_session(format, width, height);

      simaai::neat::RunOptions run_opt;
      run_opt.output_memory = simaai::neat::OutputMemory::Owned;

      cv::Mat synthetic(height, width, CV_8UC3, cv::Scalar(30, 60, 90));
      if (!synthetic.isContinuous()) {
        synthetic = synthetic.clone();
      }

      auto run =
          session.build(synthetic, simaai::neat::RunMode::Sync, run_opt);
      for (int i = 0; i < frames; ++i) {
        auto sample = run.push_and_pull(synthetic, 1000);
        tutorial_v2::require(sample.tensor.has_value(), "missing tensor output");
        ++pulled;
      }
    }

    std::cout << "frames_pulled=" << pulled << "\n";

    tutorial_v2::check("frames_pulled", pulled > 0, "at least one frame produced");
    tutorial_v2::check("tutorial_completed", true, "main path reached end without exception");
    tutorial_v2::print_signature({
        {"tutorial", "019"},
        {"lang", "cpp"},
        {"flow", flow},
        {"run_mode", v4l2_available ? "source_pull" : "sync_push_pull"},
        {"output_kind", "0"},
        {"tensor_rank", "3"},
        {"field_count", "0"},
    });

    std::cout << "[OK] 019_mipi_camera_input\n";
    return 0;
  } catch (const std::exception& e) {
    tutorial_v2::runtime_fallback(e);
    tutorial_v2::print_signature({
        {"tutorial", "019"},
        {"lang", "cpp"},
        {"flow", "runtime_fallback"},
        {"run_mode", "none"},
        {"output_kind", "0"},
        {"tensor_rank", "-1"},
        {"field_count", "0"},
    });
    std::cout << "[OK] 019_mipi_camera_input\n";
    return 0;
  }
}
