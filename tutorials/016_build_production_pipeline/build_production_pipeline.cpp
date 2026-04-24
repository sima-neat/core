// Production blueprint: wrap a Model in a Runner with production-grade RunOptions.
//
// Usage:
//   tutorial_v2_016_build_production_pipeline --mpk /path/to/model.tar.gz [--iters 4]

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
    std::string mpk;
    if (!get_arg(argc, argv, "--mpk", mpk)) {
      std::cerr << "Usage: tutorial_v2_016_build_production_pipeline --mpk <path> [--iters <n>]\n";
      return 1;
    }
    const int iters = parse_int_arg(argc, argv, "--iters", 4);

    cv::Mat rgb(224, 224, CV_8UC3, cv::Scalar(16, 96, 196));
    if (!rgb.isContinuous())
      rgb = rgb.clone();

    // CORE LOGIC
    // Production defaults: bounded queue, blocking overflow, owned output memory,
    // metrics on. Model::build returns a Runner that owns the async pipeline.
    simaai::neat::RunOptions run_opt;
    run_opt.queue_depth = 8;
    run_opt.overflow_policy = simaai::neat::OverflowPolicy::Block;
    run_opt.output_memory = simaai::neat::OutputMemory::Owned;
    run_opt.enable_metrics = true;

    simaai::neat::Model::Options model_opt;
    model_opt.input_max_width = rgb.cols;
    model_opt.input_max_height = rgb.rows;
    model_opt.input_max_depth = rgb.channels();
    model_opt.name_suffix = "_prod";

    simaai::neat::Model model(mpk, model_opt);

    simaai::neat::Model::SessionOptions sess_opt;
    sess_opt.include_appsrc = true;
    sess_opt.include_appsink = true;
    sess_opt.name_suffix = "_prod";

    auto runner =
        model.build(simaai::neat::from_cv_mat(rgb, simaai::neat::ImageSpec::PixelFormat::RGB, true),
                    sess_opt, run_opt);

    int ok = 0;
    for (int i = 0; i < iters; ++i) {
      if (!runner.push(
              simaai::neat::from_cv_mat(rgb, simaai::neat::ImageSpec::PixelFormat::RGB, true)))
        continue;
      auto out = runner.pull(/*timeout_ms=*/2000);
      if (out.has_value())
        ++ok;
    }
    runner.close();
    // END CORE LOGIC

    std::cout << "outputs=" << ok << "\n";
    std::cout << "[OK] 016_build_production_pipeline\n";
    return 0;
  } catch (const std::exception& e) {
    std::cerr << "[FAIL] " << e.what() << "\n";
    return 1;
  }
}
