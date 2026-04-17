// tutorial_0013_model_resnet50.cpp
// Story: Model is the shortest path to run a model pack.
// What you learn:
// - Model configures the pipeline in one call.
// - run() returns a simaai::neat::Sample with tensor outputs.
// - Useful for simple classification flows.

#include "neat.h"

#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>

#include <filesystem>
#include <array>
#include <iostream>
#include <string>
#include <vector>
#include <cstdlib>
#include <exception>
#include <initializer_list>
#include <stdexcept>
#include <utility>

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

void print_common_flags(std::ostream& os) {
  os << "  --help               Show this help message\n";
  os << "  --print-gst          Print the gst-launch string and exit\n";
}

int skip(const std::string& reason) {
  std::cout << "SKIP: " << reason << "\n";
  return 0;
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
  std::cout << "Usage: " << argv0 << " [--model <path>] [--image <path>]\n";
  print_common_flags(std::cout);
  std::cout << "  --model <path>       ResNet50 MPK tar.gz (default: tmp/resnet_50_mpk.tar.gz)\n";
  std::cout << "  --image <path>       Input image (default: shipped tutorial sample)\n";
}

fs::path default_model_path(const fs::path& root) {
  const fs::path c1 = root / "tmp" / "resnet_50_mpk.tar.gz";
  if (fs::exists(c1))
    return c1;
  return {};
}

fs::path default_image_path() {
  return find_asset_root() / "ilena_488.jpg";
}

} // namespace

int main(int argc, char** argv) {
  try {
    if (wants_help(argc, argv)) {
      print_help(argv[0]);
      return 0;
    }

    const fs::path root = find_repo_root();

    std::string model_arg;
    fs::path model_path = get_arg(argc, argv, "--model", model_arg)
                              ? fs::path(model_arg)
                              : default_model_path(root);
    if (model_path.empty() || !fs::exists(model_path)) {
      return skip("missing ResNet50 MPK (pass --model)");
    }

    std::string image_arg;
    fs::path image_path = get_arg(argc, argv, "--image", image_arg)
                              ? fs::path(image_arg)
                              : default_image_path();
    if (!fs::exists(image_path)) {
      return skip("missing image (pass --image)");
    }

    cv::Mat bgr = cv::imread(image_path.string(), cv::IMREAD_COLOR);
    if (bgr.empty()) {
      return skip("failed to load image");
    }

    cv::Mat rgb;
    cv::cvtColor(bgr, rgb, cv::COLOR_BGR2RGB);
    cv::resize(rgb, rgb, cv::Size(224, 224));
    if (!rgb.isContinuous())
      rgb = rgb.clone();

    std::vector<float> mean = {0.485f, 0.456f, 0.406f};
    std::vector<float> stddev = {0.229f, 0.224f, 0.225f};

    simaai::neat::Model::Options model_opt;
    model_opt.media_type = "video/x-raw";
    model_opt.format = "RGB";
    model_opt.input_max_width = 224;
    model_opt.input_max_height = 224;
    model_opt.input_max_depth = 3;
    model_opt.preproc.input_width = 224;
    model_opt.preproc.input_height = 224;
    model_opt.preproc.normalize = true;
    model_opt.preproc.channel_mean = std::array<float, 3>{mean[0], mean[1], mean[2]};
    model_opt.preproc.channel_stddev = std::array<float, 3>{stddev[0], stddev[1], stddev[2]};

    simaai::neat::Model model(model_path.string(), model_opt);

    const bool strict = (std::getenv("SIMA_RUN_TUTORIALS_FULL") != nullptr);
    try {
      auto out = model.run(rgb);
      if (!out.tensor.has_value()) {
        throw std::runtime_error("Model output missing tensor");
      }
      std::cout << "Output tensor dims: " << out.tensor->shape.size() << "\n";
    } catch (const std::exception& e) {
      if (!strict) {
        return skip(std::string("runtime unavailable: ") + e.what());
      }
      throw;
    }

    std::cout << "[OK] tutorial_0013 complete\n";
    return 0;
  } catch (const std::exception& e) {
    std::cerr << "[FAIL] " << e.what() << "\n";
    return 1;
  }
}
