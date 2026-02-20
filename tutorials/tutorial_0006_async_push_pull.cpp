// tutorial_0006_async_push_pull.cpp
// Story: async pipelines separate producer and consumer threads.
// What you learn:
// - Async mode uses independent push() and pull() calls.
// - Queue depth and drop policy are configured via RunOptions.
// - close_input() signals the end of the stream.
// - Caps renegotiation is automatic for dynamic input shapes.
// - caps_override disables renegotiation (see tutorial_0007).

#include "neat.h"

#include "tutorial_common.h"

#include <opencv2/core.hpp>

#include <iostream>
#include <string>

namespace {

void print_help(const char* argv0) {
  std::cout << "Usage: " << argv0 << " [--iters <n>] [--width <w>] [--height <h>]\n";
  sima_tutorial::print_common_flags(std::cout);
  std::cout << "  --iters <n>          Number of frames to push (default 16)\n";
  std::cout << "  --width <w>          Input width (default 320)\n";
  std::cout << "  --height <h>         Input height (default 240)\n";
}

int parse_int_arg(int argc, char** argv, const std::string& key, int def) {
  std::string val;
  if (!sima_tutorial::get_arg(argc, argv, key, val))
    return def;
  try {
    return std::stoi(val);
  } catch (...) {
    return def;
  }
}

} // namespace

int main(int argc, char** argv) {
  try {
    if (sima_tutorial::wants_help(argc, argv)) {
      print_help(argv[0]);
      return 0;
    }

    const int iters = parse_int_arg(argc, argv, "--iters", 16);
    const int w = parse_int_arg(argc, argv, "--width", 320);
    const int h = parse_int_arg(argc, argv, "--height", 240);

    cv::Mat img(h, w, CV_8UC3, cv::Scalar(0, 90, 180));
    if (!img.isContinuous())
      img = img.clone();

    simaai::neat::Session p;
    simaai::neat::InputOptions in;
    in.format = "RGB";
    in.width = w;
    in.height = h;
    in.depth = 3;
    in.is_live = true;
    in.do_timestamp = true;
    p.add(simaai::neat::nodes::Input(in));
    p.add(simaai::neat::nodes::Output());

    if (sima_tutorial::wants_print_gst(argc, argv)) {
      std::cout << p.describe_backend() << "\n";
      return 0;
    }

    simaai::neat::RunOptions run_opt;
    run_opt.queue_depth = iters;
    run_opt.overflow_policy = simaai::neat::OverflowPolicy::Block;
    run_opt.output_memory = simaai::neat::OutputMemory::Owned;

    auto run = p.build(img, simaai::neat::RunMode::Async, run_opt);

    for (int i = 0; i < iters; ++i) {
      sima_tutorial::require(run.push(img), "async push failed");
    }
    run.close_input();

    int outputs = 0;
    while (true) {
      auto sample = run.pull(/*timeout_ms=*/2000);
      if (!sample.has_value())
        break;
      outputs += 1;
    }

    sima_tutorial::require(outputs == iters, "async output count mismatch");
    std::cout << "[OK] tutorial_0006 complete\n";
    return 0;
  } catch (const std::exception& e) {
    std::cerr << "[FAIL] " << e.what() << "\n";
    return 1;
  }
}
