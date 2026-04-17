// tutorial_0015_multi_stream_multi_output.cpp
// Story: scale out by running multiple PipelineRuns side-by-side.
// What you learn:
// - Multiple Sessions can run concurrently in one process.
// - element_name_suffix helps keep pipeline element names distinct.
// - You can aggregate outputs at the application layer.

#include "neat/session.h"
#include "neat/nodes.h"


#include <opencv2/core.hpp>

#include <iostream>
#include <string>
#include <vector>
#include <array>
#include <cstdlib>
#include <exception>
#include <filesystem>
#include <initializer_list>
#include <stdexcept>
#include <utility>

namespace {

bool has_flag(int argc, char** argv, const std::string& key) {
  for (int i = 1; i < argc; ++i) {
    if (key == argv[i]) return true;
  }
  return false;
}

bool get_arg(int argc, char** argv, const std::string& key, std::string& out) {
  for (int i = 1; i + 1 < argc; ++i) {
    if (key == argv[i]) {
      out = argv[i + 1];
      return true;
    }
  }
  return false;
}

bool wants_help(int argc, char** argv) {
  return has_flag(argc, argv, "--help") || has_flag(argc, argv, "-h");
}

void print_common_flags(std::ostream& os) {
  os << "  --help               Show this help message\n";
  os << "  --print-gst          Print the gst-launch string and exit\n";
}

void require(bool ok, const std::string& msg) {
  if (!ok) throw std::runtime_error(msg);
}

} // namespace

namespace {

void print_help(const char* argv0) {
  std::cout << "Usage: " << argv0 << " [--iters <n>] [--width <w>] [--height <h>]\n";
  print_common_flags(std::cout);
  std::cout << "  --iters <n>          Frames per stream (default 4)\n";
  std::cout << "  --width <w>          Input width (default 160)\n";
  std::cout << "  --height <h>         Input height (default 120)\n";
}

int parse_int_arg(int argc, char** argv, const std::string& key, int def) {
  std::string val;
  if (!get_arg(argc, argv, key, val))
    return def;
  try {
    return std::stoi(val);
  } catch (...) {
    return def;
  }
}

simaai::neat::Session make_session(const std::string& suffix, int w, int h) {
  simaai::neat::SessionOptions opt;
  opt.element_name_suffix = suffix;
  simaai::neat::Session p(opt);

  simaai::neat::InputOptions in;
  in.format = "RGB";
  in.width = w;
  in.height = h;
  in.depth = 3;
  in.is_live = true;
  in.do_timestamp = true;
  p.add(simaai::neat::nodes::Input(in));
  p.add(simaai::neat::nodes::Output());
  return p;
}

} // namespace

int main(int argc, char** argv) {
  try {
    if (wants_help(argc, argv)) {
      print_help(argv[0]);
      return 0;
    }

    const int iters = parse_int_arg(argc, argv, "--iters", 4);
    const int w = parse_int_arg(argc, argv, "--width", 160);
    const int h = parse_int_arg(argc, argv, "--height", 120);

    cv::Mat img_a(h, w, CV_8UC3, cv::Scalar(10, 200, 40));
    cv::Mat img_b(h, w, CV_8UC3, cv::Scalar(200, 40, 10));
    if (!img_a.isContinuous())
      img_a = img_a.clone();
    if (!img_b.isContinuous())
      img_b = img_b.clone();

    auto session_a = make_session("_cam0", w, h);
    auto session_b = make_session("_cam1", w, h);

    simaai::neat::RunOptions run_opt;
    run_opt.queue_depth = iters;
    run_opt.overflow_policy = simaai::neat::OverflowPolicy::Block;
    run_opt.output_memory = simaai::neat::OutputMemory::Owned;

    auto run_a = session_a.build(img_a, simaai::neat::RunMode::Async, run_opt);
    auto run_b = session_b.build(img_b, simaai::neat::RunMode::Async, run_opt);

    for (int i = 0; i < iters; ++i) {
      require(run_a.push(img_a), "push failed for stream A");
      require(run_b.push(img_b), "push failed for stream B");
    }
    run_a.close_input();
    run_b.close_input();

    int out_a = 0;
    int out_b = 0;
    while (true) {
      auto s = run_a.pull(/*timeout_ms=*/2000);
      if (!s.has_value())
        break;
      out_a += 1;
    }
    while (true) {
      auto s = run_b.pull(/*timeout_ms=*/2000);
      if (!s.has_value())
        break;
      out_b += 1;
    }

    std::cout << "Stream A outputs: " << out_a << "\n";
    std::cout << "Stream B outputs: " << out_b << "\n";

    // Aggregate outputs at the app layer (simple example).
    std::vector<int> totals = {out_a, out_b};
    std::cout << "Aggregated outputs: [" << totals[0] << ", " << totals[1] << "]\n";

    std::cout << "[OK] tutorial_0015 complete\n";
    return 0;
  } catch (const std::exception& e) {
    std::cerr << "[FAIL] " << e.what() << "\n";
    return 1;
  }
}
