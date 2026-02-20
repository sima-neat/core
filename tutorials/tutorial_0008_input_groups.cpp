// tutorial_0008_input_groups.cpp
// Story: Input groups package common source pipelines (image, video, RTSP).
// What you learn:
// - ImageInputGroup, VideoInputGroup, RtspDecodedInput hide boilerplate.
// - output_caps lets you lock a predictable downstream format/size.
// - Source pipelines can be built without appsrc inputs.

#include "neat/session.h"
#include "neat/node_groups.h"

#include "tutorial_common.h"

#include <filesystem>
#include <iostream>
#include <string>

namespace fs = std::filesystem;

namespace {

void print_help(const char* argv0) {
  std::cout << "Usage: " << argv0 << " [--mode image|video|rtsp] [--path <path>] [--rtsp <url>]\n";
  sima_tutorial::print_common_flags(std::cout);
  std::cout << "  --mode <m>           image (default), video, or rtsp\n";
  std::cout << "  --path <path>        File path for image/video modes\n";
  std::cout << "  --rtsp <url>         RTSP URL for rtsp mode\n";
  std::cout << "  --width <w>          Output width (default 256)\n";
  std::cout << "  --height <h>         Output height (default 256)\n";
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

std::string get_mode(int argc, char** argv) {
  std::string mode;
  if (sima_tutorial::get_arg(argc, argv, "--mode", mode))
    return mode;
  return "image";
}

fs::path default_image_path(const fs::path& root) {
  const fs::path candidate1 = root / "test.jpg";
  const fs::path candidate2 = root / "tests" / "assets" / "preproc_dynamic" / "ilena_488.jpg";
  if (fs::exists(candidate1))
    return candidate1;
  if (fs::exists(candidate2))
    return candidate2;
  return candidate1;
}

fs::path default_video_path(const fs::path& root) {
  const fs::path candidate1 = root / "examples" / "media" / "coco_sample_1.mp4";
  if (fs::exists(candidate1))
    return candidate1;
  return candidate1;
}

} // namespace

int main(int argc, char** argv) {
  try {
    if (sima_tutorial::wants_help(argc, argv)) {
      print_help(argv[0]);
      return 0;
    }

    const fs::path root = sima_tutorial::find_repo_root();
    const std::string mode = get_mode(argc, argv);
    const int out_w = parse_int_arg(argc, argv, "--width", 256);
    const int out_h = parse_int_arg(argc, argv, "--height", 256);

    std::string path_arg;
    sima_tutorial::get_arg(argc, argv, "--path", path_arg);

    simaai::neat::Session p;

    if (mode == "image") {
      fs::path image_path = path_arg.empty() ? default_image_path(root) : fs::path(path_arg);
      if (!sima_tutorial::exists_or_skip(image_path, "image"))
        return 0;

      simaai::neat::nodes::groups::ImageInputGroupOptions in;
      in.path = image_path.string();
      in.imagefreeze_num_buffers = 1;
      in.use_videoscale = true;
      in.output_caps.width = out_w;
      in.output_caps.height = out_h;
      in.output_caps.format = "RGB";
      p.add(simaai::neat::nodes::groups::ImageInputGroup(in));
    } else if (mode == "video") {
      fs::path video_path = path_arg.empty() ? default_video_path(root) : fs::path(path_arg);
      if (!sima_tutorial::exists_or_skip(video_path, "video"))
        return 0;

      simaai::neat::nodes::groups::VideoInputGroupOptions in;
      in.path = video_path.string();
      in.use_videoconvert = true;
      in.use_videoscale = true;
      in.output_caps.enable = true;
      in.output_caps.width = out_w;
      in.output_caps.height = out_h;
      in.output_caps.format = "NV12";
      p.add(simaai::neat::nodes::groups::VideoInputGroup(in));
    } else if (mode == "rtsp") {
      std::string url;
      if (!sima_tutorial::get_arg(argc, argv, "--rtsp", url)) {
        return sima_tutorial::skip("missing --rtsp URL");
      }

      simaai::neat::nodes::groups::RtspDecodedInputOptions in;
      in.url = url;
      in.use_videoconvert = true;
      in.use_videoscale = true;
      in.output_caps.enable = true;
      in.output_caps.width = out_w;
      in.output_caps.height = out_h;
      in.output_caps.format = "NV12";
      p.add(simaai::neat::nodes::groups::RtspDecodedInput(in));
    } else {
      throw std::invalid_argument("unknown --mode: " + mode);
    }

    simaai::neat::OutputTensorOptions out;
    out.format = "RGB";
    out.target_width = out_w;
    out.target_height = out_h;
    p.add_output_tensor(out);

    if (sima_tutorial::wants_print_gst(argc, argv)) {
      std::cout << p.describe_backend() << "\n";
      return 0;
    }

    simaai::neat::RunOptions run_opt;
    run_opt.output_memory = simaai::neat::OutputMemory::Owned;

    // Source pipelines (image/video/rtsp) can be built without an input sample.
    auto run = p.build(run_opt);
    auto tensor = run.pull_tensor_or_throw(/*timeout_ms=*/5000);
    std::cout << "Got tensor with shape dims=" << tensor.shape.size() << "\n";

    std::cout << "[OK] tutorial_0008 complete\n";
    return 0;
  } catch (const std::exception& e) {
    std::cerr << "[FAIL] " << e.what() << "\n";
    return 1;
  }
}
