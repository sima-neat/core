// tutorial_0007_caps_negotiation.cpp
// Story: caps are inferred from input, and resolution changes renegotiate.
// What you learn:
// - Input caps are derived from the first pushed frame.
// - Resolution changes are accepted by default for dynamic input shapes.
// - Presets tune how aggressively renegotiation stabilizes.

#include "neat/session.h"
#include "neat/nodes.h"


#include <opencv2/core.hpp>

#include <iostream>
#include <string>
#include <array>
#include <cstdlib>
#include <exception>
#include <filesystem>
#include <initializer_list>
#include <stdexcept>
#include <utility>
#include <vector>

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

bool wants_print_gst(int argc, char** argv) {
  return has_flag(argc, argv, "--print-gst");
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
  std::cout << "Usage: " << argv0 << " [--small-width <w>] [--small-height <h>] "
            << "[--large-width <w>] [--large-height <h>] [--stability <n>]\n";
  print_common_flags(std::cout);
  std::cout << "  --small-width <w>    First size width (default 320)\n";
  std::cout << "  --small-height <h>   First size height (default 240)\n";
  std::cout << "  --large-width <w>    Second size width (default 640)\n";
  std::cout << "  --large-height <h>   Second size height (default 480)\n";
  std::cout << "  --stability <n>      1 => realtime behavior, 2+ => balanced behavior\n";
  std::cout << "  --renegotiate        Actually push the second size to trigger caps changes\n";
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

} // namespace

int main(int argc, char** argv) {
  try {
    if (wants_help(argc, argv)) {
      print_help(argv[0]);
      return 0;
    }

    const int sw = parse_int_arg(argc, argv, "--small-width", 320);
    const int sh = parse_int_arg(argc, argv, "--small-height", 240);
    const int lw = parse_int_arg(argc, argv, "--large-width", 640);
    const int lh = parse_int_arg(argc, argv, "--large-height", 480);
    const int stability = parse_int_arg(argc, argv, "--stability", 2);

    cv::Mat small(sh, sw, CV_8UC3, cv::Scalar(20, 200, 30));
    cv::Mat large(lh, lw, CV_8UC3, cv::Scalar(200, 30, 20));
    if (!small.isContinuous())
      small = small.clone();
    if (!large.isContinuous())
      large = large.clone();

    simaai::neat::Session p;
    simaai::neat::InputOptions in;
    in.format = "RGB";
    in.width = sw;
    in.height = sh;
    in.depth = 3;
    in.is_live = false;
    in.block = false;
    p.add(simaai::neat::nodes::Input(in));
    p.add(simaai::neat::nodes::Output());

    if (wants_print_gst(argc, argv)) {
      std::cout << p.describe_backend() << "\n";
      return 0;
    }

    simaai::neat::RunOptions run_opt;
    run_opt.output_memory = simaai::neat::OutputMemory::ZeroCopy;
    run_opt.preset =
        (stability <= 1) ? simaai::neat::RunPreset::Realtime : simaai::neat::RunPreset::Balanced;
    run_opt.queue_depth = 4;
    run_opt.overflow_policy = simaai::neat::OverflowPolicy::DropIncoming;

    auto run = p.build(small, simaai::neat::RunMode::Async, run_opt);

    const bool do_reneg = has_flag(argc, argv, "--renegotiate");

    // Push a stable regime at the original size.
    require(run.try_push(small), "try_push(small) failed");
    require(run.try_push(small), "try_push(small) failed");

    if (do_reneg) {
      // Now switch to a new resolution. Preset-driven stability handling
      // controls when the caps transition is accepted.
      require(run.try_push(large), "try_push(large) failed");
      require(run.try_push(large), "try_push(large) failed");
    } else {
      std::cout << "Skipping size change (pass --renegotiate to trigger caps changes).\n";
    }

    run.close_input();

    int outputs = 0;
    while (true) {
      auto sample = run.pull(/*timeout_ms=*/2000);
      if (!sample.has_value())
        break;
      outputs += 1;
    }

    const auto stats = run.input_stats();
    std::cout << "Outputs pulled: " << outputs << "\n";
    std::cout << "Renegotiations observed: " << stats.renegotiations << "\n";
    std::cout << "[OK] tutorial_0007 complete\n";
    return 0;
  } catch (const std::exception& e) {
    std::cerr << "[FAIL] " << e.what() << "\n";
    return 1;
  }
}
