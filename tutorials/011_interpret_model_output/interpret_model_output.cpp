// Inspect a Sample returned by a Graph: kind, tensor, fields, rank.
//
// Usage:
//   tutorial_011_interpret_model_output

#include "neat.h"

#include <opencv2/core.hpp>

#include <iostream>
#include <stdexcept>

int main() {
  try {
    cv::Mat rgb(120, 160, CV_8UC3, cv::Scalar(110, 40, 30));
    if (!rgb.isContinuous())
      rgb = rgb.clone();

    // STEP configure-input
    simaai::neat::InputOptions in;
    in.format = "RGB";
    in.width = rgb.cols;
    in.height = rgb.rows;
    in.depth = rgb.channels();
    // END STEP

    // STEP compose-graph
    simaai::neat::Graph graph;
    graph.add(simaai::neat::nodes::Input(in));
    graph.add(simaai::neat::nodes::Output());
    // Use Graph::run(...) for a one-shot synchronous frame.
    // END STEP

    // CORE LOGIC
    // STEP run-frame
    // Graph::run is the one-frame synchronous shortcut.
    simaai::neat::TensorList out = graph.run(std::vector<cv::Mat>{rgb});
    // END STEP

    // STEP inspect-sample
    std::cout << "outputs=" << out.size() << " has_tensor=" << (!out.empty() ? "yes" : "no")
              << " fields=" << 0 << "\n";
    if (out.empty())
      throw std::runtime_error("expected tensor output");
    if (out.front().shape.empty())
      throw std::runtime_error("output tensor shape is empty");
    std::cout << "rank=" << out.front().shape.size() << "\n";
    // END STEP
    // END CORE LOGIC
    std::cout << "[OK] 011_interpret_model_output\n";
    return 0;
  } catch (const std::exception& e) {
    std::cerr << "[FAIL] " << e.what() << "\n";
    return 1;
  }
}
