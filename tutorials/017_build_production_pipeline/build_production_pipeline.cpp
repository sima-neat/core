// Production blueprint: wrap a Model in a Runner with production-grade RunOptions.
//
// Usage:
//   tutorial_017_build_production_pipeline --model /path/to/resnet_50.tar.gz [--iters 4]

#include "neat.h"

#include <opencv2/core.hpp>

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
      std::cerr << "Usage: tutorial_017_build_production_pipeline --model <path> [--iters <n>]\n";
      return 1;
    }
    const int iters = parse_int_arg(argc, argv, "--iters", 4);

    cv::Mat rgb(224, 224, CV_8UC3, cv::Scalar(16, 96, 196));
    if (!rgb.isContinuous())
      rgb = rgb.clone();

    // CORE LOGIC
    // Production defaults: bounded queue, blocking overflow, owned output memory,
    // metrics on. Model::build returns a Runner that owns the async pipeline.
    // STEP configure-run-options
    simaai::neat::RunOptions run_opt;
    run_opt.queue_depth = 8;
    run_opt.overflow_policy = simaai::neat::OverflowPolicy::Block;
    run_opt.output_memory = simaai::neat::OutputMemory::Owned;
    run_opt.enable_metrics = true;
    // END STEP

    // STEP configure-model
    simaai::neat::Model::Options model_opt;
    model_opt.preprocess.kind = simaai::neat::InputKind::Image;
    model_opt.preprocess.enable = simaai::neat::AutoFlag::On;
    model_opt.preprocess.color_convert.input_format = simaai::neat::PreprocessColorFormat::RGB;
    model_opt.preprocess.input_max_width = rgb.cols;
    model_opt.preprocess.input_max_height = rgb.rows;
    model_opt.preprocess.input_max_depth = rgb.channels();
    model_opt.preprocess.normalize.enable = simaai::neat::AutoFlag::On;
    model_opt.preprocess.normalize.mean = {0.485f, 0.456f, 0.406f};
    model_opt.preprocess.normalize.stddev = {0.229f, 0.224f, 0.225f};
    model_opt.preprocess.normalize.has_explicit_stats = true;
    model_opt.name_suffix = "_prod";

    simaai::neat::Model model(model_path, model_opt);
    // END STEP

    // STEP build-runner
    simaai::neat::Model::RouteOptions sess_opt;
    sess_opt.include_input = true;
    sess_opt.include_output = true;
    sess_opt.name_suffix = "_prod";

    auto runner = model.build(
        simaai::neat::TensorList{simaai::neat::Tensor::from_cv_mat(
            rgb, simaai::neat::ImageSpec::PixelFormat::RGB, simaai::neat::TensorMemory::EV74)},
        sess_opt, run_opt);
    // END STEP

    // STEP run-loop
    int ok = 0;
    for (int i = 0; i < iters; ++i) {
      if (!runner.push(simaai::neat::TensorList{simaai::neat::Tensor::from_cv_mat(
              rgb, simaai::neat::ImageSpec::PixelFormat::RGB, simaai::neat::TensorMemory::EV74)}))
        continue;
      auto out = runner.pull(/*timeout_ms=*/2000);
      if (!out.empty())
        ++ok;
    }
    runner.close();
    // END STEP
    // END CORE LOGIC

    std::cout << "outputs=" << ok << "\n";
    std::cout << "[OK] 017_build_production_pipeline\n";
    return 0;
  } catch (const std::exception& e) {
    std::cerr << "[FAIL] " << e.what() << "\n";
    return 1;
  }
}
