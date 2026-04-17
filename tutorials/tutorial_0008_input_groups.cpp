// tutorial_0008_input_groups.cpp
// Story: Input groups package common source pipelines (image, video, RTSP).
// What you learn:
// - ImageInputGroup, VideoInputGroup, RtspDecodedInput hide boilerplate.
// - output_caps lets you lock a predictable downstream format/size.
// - Source pipelines can be built without appsrc inputs.

#include "neat/session.h"
#include "neat/node_groups.h"


#include <filesystem>
#include <iostream>
#include <string>
#include <array>
#include <cstdlib>
#include <exception>
#include <initializer_list>
#include <stdexcept>
#include <utility>
#include <vector>

namespace {

bool has_flag(int argc, char** argv, const std::string& key) {
  for (int i = 1; i < argc; ++i) {
    if (key == argv[i]) return true;
  }
  return false;
}

bool get_arg(int argc, char** argv, const std::string& key, std::string& out) {
  for (int i = 1; i + 1 < argc; ++i) {
    if (key == argv[i]) {
      out = argv[i + 1];
      return true;
    }
  }
  return false;
}

bool wants_help(int argc, char** argv) {
  return has_flag(argc, argv, "--help") || has_flag(argc, argv, "-h");
}

bool wants_print_gst(int argc, char** argv) {
  return has_flag(argc, argv, "--print-gst");
}

void print_common_flags(std::ostream& os) {
  os << "  --help               Show this help message\n";
  os << "  --print-gst          Print the gst-launch string and exit\n";
}

int skip(const std::string& reason) {
  std::cout << "SKIP: " << reason << "\n";
  return 0;
}

bool exists_or_skip(const std::filesystem::path& path, const std::string& label) {
  if (!std::filesystem::exists(path)) {
    std::cout << "SKIP: missing " << label << " at " << path.string() << "\n";
    return false;
  }
  return true;
}

std::filesystem::path find_repo_root() {
  namespace fs = std::filesystem;
  fs::path cur = fs::current_path();
  for (int i = 0; i < 6; ++i) {
    if (fs::exists(cur / "CMakeLists.txt") && fs::exists(cur / "include") &&
        fs::exists(cur / "tests")) {
      return cur;
    }
    if (!cur.has_parent_path()) break;
    cur = cur.parent_path();
  }
  return fs::current_path();
}

std::filesystem::path find_asset_root() {
  namespace fs = std::filesystem;
  if (const char* env = std::getenv("SIMA_NEAT_TUTORIAL_ASSETS")) {
    fs::path p{env};
    if (fs::exists(p)) return p;
  }
  for (const fs::path& p : {
           fs::path{"/usr/share/sima-neat/tutorials/assets"},
           fs::path{"/neat-resources/core-src/tutorials/assets"},
       }) {
    if (fs::exists(p)) return p;
  }
  return find_repo_root() / "tutorials" / "assets";
}

} // namespace

namespace fs = std::filesystem;

namespace {

void print_help(const char* argv0) {
  std::cout << "Usage: " << argv0 << " [--mode image|video|rtsp] [--path <path>] [--rtsp <url>]\n";
  print_common_flags(std::cout);
  std::cout << "  --mode <m>           image (default), video, or rtsp\n";
  std::cout << "  --path <path>        File path for image/video modes\n";
  std::cout << "  --rtsp <url>         RTSP URL for rtsp mode\n";
  std::cout << "  --width <w>          Output width (default 256)\n";
  std::cout << "  --height <h>         Output height (default 256)\n";
}

int parse_int_arg(int argc, char** argv, const std::string& key, int def) {
  std::string val;
  if (!get_arg(argc, argv, key, val))
    return def;
  try {
    return std::stoi(val);
  } catch (...) {
    return def;
  }
}

std::string get_mode(int argc, char** argv) {
  std::string mode;
  if (get_arg(argc, argv, "--mode", mode))
    return mode;
  return "image";
}

fs::path default_image_path() {
  return find_asset_root() / "ilena_488.jpg";
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
    if (wants_help(argc, argv)) {
      print_help(argv[0]);
      return 0;
    }

    const fs::path root = find_repo_root();
    const std::string mode = get_mode(argc, argv);
    const int out_w = parse_int_arg(argc, argv, "--width", 256);
    const int out_h = parse_int_arg(argc, argv, "--height", 256);

    std::string path_arg;
    get_arg(argc, argv, "--path", path_arg);

    simaai::neat::Session p;

    if (mode == "image") {
      fs::path image_path = path_arg.empty() ? default_image_path() : fs::path(path_arg);
      if (!exists_or_skip(image_path, "image"))
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
      if (!exists_or_skip(video_path, "video"))
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
      if (!get_arg(argc, argv, "--rtsp", url)) {
        return skip("missing --rtsp URL");
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

    if (wants_print_gst(argc, argv)) {
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
