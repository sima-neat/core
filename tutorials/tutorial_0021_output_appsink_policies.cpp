// tutorial_0021_output_appsink_policies.cpp
// Story: appsink policies control buffering and drop behavior.
// What you learn:
// - OutputOptions::Latest/EveryFrame/Clocked presets.
// - add_output_tensor() uses run-time policies from RunOptions.
// - RunOptions maps user intent to runtime buffering/drop behavior.

#include "neat/session.h"
#include "neat/nodes.h"

#include <opencv2/core.hpp>

#include <algorithm>
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
    if (key == argv[i])
      return true;
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

} // namespace

namespace {

void print_help(const char* argv0) {
  std::cout << "Usage: " << argv0
            << " [--mode latest|every|clocked] [--max-buffers <n>] [--tensor]\n";
  print_common_flags(std::cout);
  std::cout << "  --mode <policy>      latest|every|clocked (default latest)\n";
  std::cout << "  --max-buffers <n>    Max buffers for appsink (default 1 or policy default)\n";
  std::cout << "  --tensor             Use add_output_tensor() instead of Output\n";
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

simaai::neat::OutputOptions parse_policy(int argc, char** argv, int max_buffers) {
  std::string mode;
  get_arg(argc, argv, "--mode", mode);
  if (mode == "every") {
    return simaai::neat::OutputOptions::EveryFrame(max_buffers > 0 ? max_buffers : 30);
  }
  if (mode == "clocked") {
    return simaai::neat::OutputOptions::Clocked(max_buffers > 0 ? max_buffers : 1);
  }
  return simaai::neat::OutputOptions::Latest();
}

} // namespace

int main(int argc, char** argv) {
  try {
    if (wants_help(argc, argv)) {
      print_help(argv[0]);
      return 0;
    }

    const int max_buffers = parse_int_arg(argc, argv, "--max-buffers", -1);
    const bool use_tensor = has_flag(argc, argv, "--tensor");
    simaai::neat::OutputOptions policy = parse_policy(argc, argv, max_buffers);

    const int w = 160;
    const int h = 120;
    cv::Mat img(h, w, CV_8UC3, cv::Scalar(20, 80, 200));
    if (!img.isContinuous())
      img = img.clone();

    simaai::neat::Session p;
    simaai::neat::InputOptions in;
    in.format = "RGB";
    in.width = w;
    in.height = h;
    in.depth = 3;
    p.add(simaai::neat::nodes::Input(in));

    if (use_tensor) {
      simaai::neat::OutputTensorOptions out;
      out.format = "RGB";
      out.target_width = w;
      out.target_height = h;
      p.add_output_tensor(out);
    } else {
      p.add(simaai::neat::nodes::Output(policy));
    }

    std::cout << "Policy: max_buffers=" << policy.max_buffers
              << " drop=" << (policy.drop ? "true" : "false")
              << " sync=" << (policy.sync ? "true" : "false") << "\n";

    if (wants_print_gst(argc, argv)) {
      std::cout << p.describe_backend() << "\n";
      return 0;
    }

    simaai::neat::RunOptions run_opt;
    run_opt.output_memory = simaai::neat::OutputMemory::Owned;
    if (use_tensor) {
      run_opt.queue_depth = std::max(1, policy.max_buffers);
      run_opt.overflow_policy = policy.drop ? simaai::neat::OverflowPolicy::KeepLatest
                                            : simaai::neat::OverflowPolicy::Block;
    }

    auto run = p.build(img, simaai::neat::RunMode::Sync, run_opt);
    auto out = run.push_and_pull(img, /*timeout_ms=*/1000);
    std::cout << "Output kind: " << static_cast<int>(out.kind) << "\n";

    std::cout << "[OK] tutorial_0021 complete\n";
    return 0;
  } catch (const std::exception& e) {
    std::cerr << "[FAIL] " << e.what() << "\n";
    return 1;
  }
}
