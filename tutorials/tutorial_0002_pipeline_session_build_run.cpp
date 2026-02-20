// tutorial_0002_pipeline_session_build_run.cpp
// Story: build vs run, and how to inspect the pipeline you constructed.
// What you learn:
// - Session::describe() gives a human-readable graph view.
// - Session::describe_backend() prints the gst-launch string.
// - build() returns a Run; run() is the sync convenience path.
//
// This tutorial uses synthetic frames (no external assets).

#include "neat.h"

#include "tutorial_common.h"

#include <opencv2/core.hpp>

#include <cstdlib>
#include <iostream>
#include <string>

namespace {

void print_help(const char* argv0) {
  std::cout << "Usage: " << argv0 << " [--width <w>] [--height <h>]\n";
  sima_tutorial::print_common_flags(std::cout);
  std::cout << "  --width <w>          Input width (default 320)\n";
  std::cout << "  --height <h>         Input height (default 240)\n";
  std::cout << "  --use-run            Execute Session::run() demo\n";
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

simaai::neat::Session make_session(int w, int h) {
  simaai::neat::Session p;

  // Input describes what kind of frames we will push at runtime.
  simaai::neat::InputOptions in;
  in.format = "RGB";
  in.width = w;
  in.height = h;
  in.depth = 3;
  in.is_live = false;
  in.do_timestamp = true;
  p.add(simaai::neat::nodes::Input(in));

  // Output is the standard sink for pull-based pipelines.
  p.add(simaai::neat::nodes::Output());

  return p;
}

} // namespace

int main(int argc, char** argv) {
  try {
    if (sima_tutorial::wants_help(argc, argv)) {
      print_help(argv[0]);
      return 0;
    }

    const int w = parse_int_arg(argc, argv, "--width", 320);
    const int h = parse_int_arg(argc, argv, "--height", 240);

    // Keep run() bounded even if plugins are missing or negotiation fails.
    if (!std::getenv("SIMA_GST_RUN_INPUT_TIMEOUT_MS")) {
      setenv("SIMA_GST_RUN_INPUT_TIMEOUT_MS", "2000", 1);
    }

    // Synthetic frame: a simple solid color image.
    cv::Mat img(h, w, CV_8UC3, cv::Scalar(40, 120, 200));
    if (!img.isContinuous())
      img = img.clone();

    // 1) Build a session and inspect it.
    simaai::neat::Session p = make_session(w, h);
    std::cout << "--- describe() ---\n" << p.describe() << "\n";

    if (sima_tutorial::wants_print_gst(argc, argv)) {
      std::cout << p.describe_backend() << "\n";
      return 0;
    }

    // 2) build(): returns a Run you can push/pull on.
    simaai::neat::RunOptions run_opt;
    run_opt.output_memory = simaai::neat::OutputMemory::Owned;
    auto run = p.build(img, simaai::neat::RunMode::Sync, run_opt);
    auto sample = run.push_and_pull(img, /*timeout_ms=*/1000);
    sima_tutorial::require(sample.tensor.has_value(), "build(): missing tensor output");

    // 3) run(): convenience path for a single sync push/pull.
    if (sima_tutorial::has_flag(argc, argv, "--use-run")) {
      simaai::neat::Session p2 = make_session(w, h);
      auto sample2 = p2.run(img);
      sima_tutorial::require(sample2.tensor.has_value(), "run(): missing tensor output");
    } else {
      std::cout << "Skipping run() demo (pass --use-run to execute it).\n";
    }

    std::cout << "[OK] tutorial_0002 complete\n";
    return 0;
  } catch (const std::exception& e) {
    std::cerr << "[FAIL] " << e.what() << "\n";
    return 1;
  }
}
