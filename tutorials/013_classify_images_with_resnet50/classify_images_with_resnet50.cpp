// ResNet quickstart: ImageNet normalization via Model::Options, then one-shot model.run.
//
// Usage:
//   tutorial_v2_013_resnet_quickstart --mpk /path/to/resnet_50.tar.gz --image /path/to.jpg

#include "neat.h"

#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>

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

} // namespace

int main(int argc, char** argv) {
  try {
    std::string mpk, image;
    if (!get_arg(argc, argv, "--mpk", mpk) || !get_arg(argc, argv, "--image", image)) {
      std::cerr << "Usage: tutorial_v2_013_resnet_quickstart --mpk <path> --image <path>\n";
      return 1;
    }

    cv::Mat bgr = cv::imread(image, cv::IMREAD_COLOR);
    if (bgr.empty())
      throw std::runtime_error("failed to load image: " + image);

    cv::Mat rgb;
    cv::cvtColor(bgr, rgb, cv::COLOR_BGR2RGB);
    cv::resize(rgb, rgb, cv::Size(224, 224));
    if (!rgb.isContinuous())
      rgb = rgb.clone();

    // CORE LOGIC
    // ImageNet-trained ResNet needs per-channel mean/stddev normalization.
    simaai::neat::Model::Options opt;
    opt.media_type = "video/x-raw";
    opt.format = "RGB";
    opt.input_max_width = 224;
    opt.input_max_height = 224;
    opt.input_max_depth = 3;
    opt.preproc.normalize = true;
    opt.preproc.channel_mean = std::array<float, 3>{0.485f, 0.456f, 0.406f};
    opt.preproc.channel_stddev = std::array<float, 3>{0.229f, 0.224f, 0.225f};

    simaai::neat::Model model(mpk, opt);
    auto out = model.run(rgb, /*timeout_ms=*/2000);
    // END CORE LOGIC

    if (!out.tensor.has_value())
      throw std::runtime_error("missing output tensor");
    std::cout << "output_rank=" << out.tensor->shape.size() << "\n";
    if (!out.tensor->shape.empty())
      std::cout << "first_dim=" << out.tensor->shape.front() << "\n";
    std::cout << "[OK] 013_classify_images_with_resnet50\n";
    return 0;
  } catch (const std::exception& e) {
    std::cerr << "[FAIL] " << e.what() << "\n";
    return 1;
  }
}
