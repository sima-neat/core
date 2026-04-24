// Inspect a Sample returned by a Session: kind, tensor, fields, rank.
//
// Usage:
//   tutorial_v2_010_interpret_model_output

#include "neat.h"

#include <opencv2/core.hpp>

#include <iostream>
#include <stdexcept>

int main() {
  try {
    cv::Mat rgb(120, 160, CV_8UC3, cv::Scalar(110, 40, 30));
    if (!rgb.isContinuous())
      rgb = rgb.clone();

    // CORE LOGIC
    simaai::neat::Session session;
    simaai::neat::InputOptions in;
    in.format = "RGB";
    in.width = rgb.cols;
    in.height = rgb.rows;
    in.depth = rgb.channels();
    session.add(simaai::neat::nodes::Input(in));
    session.add(simaai::neat::nodes::Output());

    auto run = session.build(rgb, simaai::neat::RunMode::Sync);

    // push_and_pull is the one-frame synchronous shortcut; Sample has .kind,
    // .tensor (optional), .fields (named sub-tensors for bundles).
    simaai::neat::Sample out = run.push_and_pull(rgb, /*timeout_ms=*/1000);
    // END CORE LOGIC

    std::cout << "kind=" << static_cast<int>(out.kind)
              << " has_tensor=" << (out.tensor.has_value() ? "yes" : "no")
              << " fields=" << out.fields.size() << "\n";
    if (!out.tensor.has_value())
      throw std::runtime_error("expected tensor output");
    if (out.tensor->shape.empty())
      throw std::runtime_error("output tensor shape is empty");
    std::cout << "rank=" << out.tensor->shape.size() << "\n";
    std::cout << "[OK] 010_interpret_model_output\n";
    return 0;
  } catch (const std::exception& e) {
    std::cerr << "[FAIL] " << e.what() << "\n";
    return 1;
  }
}
