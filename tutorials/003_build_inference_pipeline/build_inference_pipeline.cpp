// Build a minimal Session (Input -> Output), run a frame, read the tensor rank.
//
// Usage:
//   tutorial_v2_003_session_build_and_run [--width <w>] [--height <h>]

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
    const int width = parse_int_arg(argc, argv, "--width", 320);
    const int height = parse_int_arg(argc, argv, "--height", 240);

    cv::Mat input(height, width, CV_8UC3, cv::Scalar(30, 60, 90));
    if (!input.isContinuous())
      input = input.clone();

    // CORE LOGIC
    // Compose a Session from Input and Output nodes, then build+run one frame.
    simaai::neat::Session session;

    simaai::neat::InputOptions in;
    in.format = "RGB";
    in.width = width;
    in.height = height;
    in.depth = 3;
    in.is_live = false;
    in.do_timestamp = true;

    session.add(simaai::neat::nodes::Input(in));
    session.add(simaai::neat::nodes::Output());

    simaai::neat::RunOptions run_opt;
    run_opt.output_memory = simaai::neat::OutputMemory::Owned;

    auto run = session.build(input, simaai::neat::RunMode::Sync, run_opt);
    auto sample = run.push_and_pull(input, /*timeout_ms=*/1000);
    // END CORE LOGIC

    if (!sample.tensor.has_value())
      throw std::runtime_error("missing tensor output");
    std::cout << "tensor_rank=" << sample.tensor->shape.size() << "\n";
    std::cout << "[OK] 003_build_inference_pipeline\n";
    return 0;
  } catch (const std::exception& e) {
    std::cerr << "[FAIL] " << e.what() << "\n";
    return 1;
  }
}
