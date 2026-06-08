// Three Graph composition patterns: direct nodes, model.graph(), attached Graph.
//
// Usage:
//   tutorial_007_plug_model_into_pipeline [--model /path/to/model.tar.gz]

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

    // Pattern 1: build a Graph by adding Input/Output nodes directly.
    simaai::neat::InputOptions in;
    in.format = "RGB";
    in.width = width;
    in.height = height;
    in.depth = 3;
    in.do_timestamp = true;

    // CORE LOGIC
    simaai::neat::Graph direct;
    direct.add(simaai::neat::nodes::Input(in));
    direct.add(simaai::neat::nodes::Output());
    // END CORE LOGIC

    std::string model_path;
    if (get_arg(argc, argv, "--model", model_path) && fs::exists(model_path)) {
      simaai::neat::Model model(model_path);

      // Pattern 2: ask model.graph() to include explicit public Input/Output boundaries.
      // STEP model-graph
      simaai::neat::Model::RouteOptions runnable_opt;
      runnable_opt.include_input = true;
      runnable_opt.include_output = true;
      // CORE LOGIC
      simaai::neat::Graph from_model;
      from_model.add(model.graph(runnable_opt));
      // END CORE LOGIC
      // END STEP
      std::cout << "model_graph_backend=\n" << from_model.describe_backend() << "\n";

      // Pattern 3: attach the model under an upstream name with custom options.
      // STEP route-options
      simaai::neat::Model::RouteOptions sopt;
      sopt.include_input = false;
      sopt.include_output = true;
      sopt.upstream_name = "camera0";
      sopt.name_suffix = "_camera0";
      sopt.buffer_name = "camera0";
      // END STEP

      // CORE LOGIC
      // STEP attached-graph
      simaai::neat::Graph attached;
      attached.add(model.graph(sopt));
      // END STEP
      // END CORE LOGIC
      std::cout << "attached_graph_backend=\n" << attached.describe_backend() << "\n";
    }

    auto run = direct.build(std::vector<cv::Mat>{rgb}, simaai::neat::RunMode::Sync);
    simaai::neat::TensorList out = run.run(std::vector<cv::Mat>{rgb}, /*timeout_ms=*/1000);

    if (out.empty())
      throw std::runtime_error("direct Graph output missing tensor");
    std::cout << "direct_rank=" << out.front().shape.size() << "\n";
    std::cout << "[OK] 007_plug_model_into_pipeline\n";
    return 0;
  } catch (const std::exception& e) {
    std::cerr << "[FAIL] " << e.what() << "\n";
    return 1;
  }
}
