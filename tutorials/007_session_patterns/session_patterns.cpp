// Three Session composition patterns: direct nodes, model.session(), attached session.
//
// Usage:
//   tutorial_v2_007_session_patterns [--mpk /path/to/model.tar.gz]

#include "neat.h"

#include <opencv2/core.hpp>

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

} // namespace

int main(int argc, char** argv) {
  try {
    const int width = 224;
    const int height = 224;

    cv::Mat rgb(height, width, CV_8UC3, cv::Scalar(80, 40, 160));
    if (!rgb.isContinuous())
      rgb = rgb.clone();

    // CORE LOGIC
    // Pattern 1: build a Session by adding Input/Output nodes directly.
    simaai::neat::Session direct;
    simaai::neat::InputOptions in;
    in.format = "RGB";
    in.width = width;
    in.height = height;
    in.depth = 3;
    in.do_timestamp = true;
    direct.add(simaai::neat::nodes::Input(in));
    direct.add(simaai::neat::nodes::Output());

    std::string mpk;
    if (get_arg(argc, argv, "--mpk", mpk) && fs::exists(mpk)) {
      simaai::neat::Model model(mpk);

      // Pattern 2: ingest the model's default session group.
      simaai::neat::Session from_model;
      from_model.add(model.session());
      std::cout << "model_session_size=" << model.session().size() << "\n";

      // Pattern 3: attach the model under an upstream name with custom options.
      simaai::neat::Model::SessionOptions sopt;
      sopt.include_appsrc = false;
      sopt.include_appsink = true;
      sopt.upstream_name = "camera0";
      sopt.name_suffix = "_camera0";
      sopt.buffer_name = "camera0";

      simaai::neat::Session attached;
      attached.add(model.session(sopt));
      std::cout << "attached_session_size=" << attached.size() << "\n";
    }

    auto run = direct.build(rgb, simaai::neat::RunMode::Sync);
    auto out = run.push_and_pull(rgb, /*timeout_ms=*/1000);
    // END CORE LOGIC

    if (!out.tensor.has_value())
      throw std::runtime_error("direct session output missing tensor");
    std::cout << "direct_rank=" << out.tensor->shape.size() << "\n";
    std::cout << "[OK] 007_session_patterns\n";
    return 0;
  } catch (const std::exception& e) {
    std::cerr << "[FAIL] " << e.what() << "\n";
    return 1;
  }
}
