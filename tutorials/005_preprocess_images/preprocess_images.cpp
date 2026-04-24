// Run preprocessing standalone via stages::Preproc and inspect the resulting tensor.
//
// Usage:
//   tutorial_v2_005_preprocess_images --mpk /path/to/resnet_50.tar.gz [--size 224]

#include "neat.h"

#include "pipeline/StageRun.h"

#include <opencv2/core.hpp>

#include <array>
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
  std::string value;
  if (!get_arg(argc, argv, key, value))
    return def;
  return std::stoi(value);
}

} // namespace

int main(int argc, char** argv) {
  try {
    std::string mpk;
    if (!get_arg(argc, argv, "--mpk", mpk)) {
      std::cerr << "Usage: tutorial_v2_005_preprocess_images --mpk <path> [--size <n>]\n";
      return 1;
    }
    const int size = parse_int_arg(argc, argv, "--size", 224);

    simaai::neat::Model::Options opt;
    opt.format = "BGR";
    opt.input_max_width = size;
    opt.input_max_height = size;
    opt.input_max_depth = 3;
    opt.preproc.input_width = size;
    opt.preproc.input_height = size;
    opt.preproc.output_width = size;
    opt.preproc.output_height = size;
    opt.preproc.normalize = true;
    opt.preproc.channel_mean = std::array<float, 3>{0.5f, 0.5f, 0.5f};
    opt.preproc.channel_stddev = std::array<float, 3>{0.5f, 0.5f, 0.5f};

    simaai::neat::Model model(mpk, opt);

    cv::Mat bgr(size, size, CV_8UC3, cv::Scalar(40, 80, 120));
    if (!bgr.isContinuous())
      bgr = bgr.clone();

    // CORE LOGIC
    // stages::Preproc runs just the preprocessing step from the model's Options
    // and returns the preprocessed Tensor.
    simaai::neat::Tensor pre = simaai::neat::stages::Preproc(bgr, model);
    // END CORE LOGIC

    std::cout << "preproc_rank=" << pre.shape.size() << "\n";
    std::cout << "preproc_dtype=" << static_cast<int>(pre.dtype) << "\n";
    std::cout << "[OK] 005_preprocess_images\n";
    return 0;
  } catch (const std::exception& e) {
    std::cerr << "[FAIL] " << e.what() << "\n";
    return 1;
  }
}
