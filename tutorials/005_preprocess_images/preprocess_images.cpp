// Run preprocessing standalone via stages::Preproc and inspect the resulting tensor.
//
// Usage:
//   tutorial_005_preprocess_images --model /path/to/resnet_50.tar.gz [--size 224]

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
    std::string model_path;
    if (!get_arg(argc, argv, "--model", model_path)) {
      std::cerr << "Usage: tutorial_005_preprocess_images --model <path> [--size <n>]\n";
      return 1;
    }
    const int size = parse_int_arg(argc, argv, "--size", 224);

    // STEP configure-preproc
    simaai::neat::Model::Options opt;
    opt.preprocess.color_convert.input_format = simaai::neat::PreprocessColorFormat::BGR;
    opt.preprocess.input_max_width = size;
    opt.preprocess.input_max_height = size;
    opt.preprocess.input_max_depth = 3;
    opt.preprocess.resize.width = size;
    opt.preprocess.resize.height = size;
    opt.preprocess.resize.width = size;
    opt.preprocess.resize.height = size;
    opt.preprocess.normalize.enable = simaai::neat::AutoFlag::On;
    opt.preprocess.normalize.mean = std::array<float, 3>{0.5f, 0.5f, 0.5f};
    opt.preprocess.normalize.stddev = std::array<float, 3>{0.5f, 0.5f, 0.5f};
    // END STEP

    // STEP load-model
    simaai::neat::Model model(model_path, opt);
    // END STEP

    cv::Mat bgr(size, size, CV_8UC3, cv::Scalar(40, 80, 120));
    if (!bgr.isContinuous())
      bgr = bgr.clone();

    // CORE LOGIC
    // stages::Preproc runs just the preprocessing step from the model's Options
    // and returns the preprocessed Tensor.
    // STEP inspect-preproc
    simaai::neat::Tensor pre =
        simaai::neat::stages::Preproc(std::vector<cv::Mat>{bgr}, model).front();
    // END STEP
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
