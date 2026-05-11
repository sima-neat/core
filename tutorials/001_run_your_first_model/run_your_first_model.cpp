// Run a ResNet-50 model on an image in three lines of Neat.
//
// Usage:
//   tutorial_001_run_your_first_model --mpk /path/to/resnet_50.tar.gz [--image /path/to.jpg]

#include "neat.h"

#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>

#include <cstring>
#include <filesystem>
#include <iostream>
#include <stdexcept>
#include <string>

namespace fs = std::filesystem;

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

cv::Mat load_rgb(const fs::path& image_path, int size) {
  cv::Mat bgr = cv::imread(image_path.string(), cv::IMREAD_COLOR);
  if (bgr.empty())
    throw std::runtime_error("failed to read image: " + image_path.string());
  if (bgr.cols != size || bgr.rows != size) {
    cv::resize(bgr, bgr, cv::Size(size, size), 0, 0, cv::INTER_AREA);
  }
  cv::Mat rgb;
  cv::cvtColor(bgr, rgb, cv::COLOR_BGR2RGB);
  if (!rgb.isContinuous())
    rgb = rgb.clone();
  return rgb;
}

simaai::neat::Model::Options build_options(int size) {
  simaai::neat::Model::Options opt;
  opt.preprocess.color_convert.input_format = simaai::neat::PreprocessColorFormat::RGB;
  opt.preprocess.input_max_width = size;
  opt.preprocess.input_max_height = size;
  opt.preprocess.input_max_depth = 3;
  opt.preprocess.normalize.mean = {0.485f, 0.456f, 0.406f};
  opt.preprocess.normalize.stddev = {0.229f, 0.224f, 0.225f};
  return opt;
}

int top1_from_output(const simaai::neat::TensorList& out) {
  if (out.empty())
    throw std::runtime_error("no tensor output");
  const simaai::neat::Mapping m = out.front().map_read();
  const size_t n = m.size_bytes / sizeof(float);
  const float* p = reinterpret_cast<const float*>(m.data);
  int best = 0;
  for (size_t i = 1; i < n && i < 1000; ++i) {
    if (p[i] > p[best])
      best = static_cast<int>(i);
  }
  return best;
}

} // namespace

int main(int argc, char** argv) {
  try {
    std::string mpk, image;
    if (!get_arg(argc, argv, "--mpk", mpk)) {
      std::cerr << "Usage: tutorial_001_run_your_first_model --mpk <path> [--image <path>]\n";
      return 1;
    }
    get_arg(argc, argv, "--image", image);

    const int size = 224;

    // CORE LOGIC
    // The three-line Neat story:
    simaai::neat::Model model(mpk, build_options(size));
    cv::Mat input = image.empty() ? cv::Mat(size, size, CV_8UC3, cv::Scalar(99, 99, 99))
                                  : load_rgb(image, size);
    simaai::neat::TensorList sample = model.run(std::vector<cv::Mat>{input}, /*timeout_ms=*/2000);
    // END CORE LOGIC

    std::cout << "top1=" << top1_from_output(sample) << "\n";
    std::cout << "[OK] 001_run_your_first_model\n";
    return 0;
  } catch (const std::exception& e) {
    std::cerr << "[FAIL] " << e.what() << "\n";
    return 1;
  }
}
