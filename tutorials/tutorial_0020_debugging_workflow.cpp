// tutorial_0020_debugging_workflow.cpp
// Story: practical debugging workflow for runtime issues.
// What you learn:
// - describe_backend() gives a reproducible gst-launch string.
// - Run::report() summarizes runtime diagnostics.
// - Env toggles enable extra telemetry (DOT graphs, pipeline strings).

#include "neat/session.h"
#include "neat/nodes.h"

#include <opencv2/core.hpp>

#include <cstdlib>
#include <filesystem>
#include <iostream>
#include <string>
#include <array>
#include <exception>
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

std::filesystem::path find_repo_root() {
  namespace fs = std::filesystem;
  fs::path cur = fs::current_path();
  for (int i = 0; i < 6; ++i) {
    if (fs::exists(cur / "CMakeLists.txt") && fs::exists(cur / "include") &&
        fs::exists(cur / "tests")) {
      return cur;
    }
    if (!cur.has_parent_path())
      break;
    cur = cur.parent_path();
  }
  return fs::current_path();
}

} // namespace

namespace fs = std::filesystem;

namespace {

void print_help(const char* argv0) {
  std::cout << "Usage: " << argv0 << " [--width <w>] [--height <h>] [--debug-env] [--bad-input]\n";
  print_common_flags(std::cout);
  std::cout << "  --width <w>          Width (default 160)\n";
  std::cout << "  --height <h>         Height (default 120)\n";
  std::cout << "  --debug-env          Enable common debug env vars\n";
  std::cout << "  --bad-input          Push a bad frame to show error handling\n";
  std::cout << "\n";
  std::cout << "Tips:\n";
  std::cout << "  SIMA_PIPELINE_STRING_DEBUG=1 prints the final gst pipeline string.\n";
  std::cout << "  SIMA_GST_DOT_DIR=<dir> writes DOT graphs for failures/debug.\n";
  std::cout << "  SIMA_INPUTSTREAM_DOT_ON_TIMEOUT=1 dumps DOT on timeouts.\n";
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

void enable_debug_env(const fs::path& root) {
  const fs::path dot_dir = root / "tmp" / "gst_dot";
  std::error_code ec;
  fs::create_directories(dot_dir, ec);
  setenv("SIMA_PIPELINE_STRING_DEBUG", "1", 1);
  setenv("SIMA_GST_DOT_DIR", dot_dir.string().c_str(), 1);
  setenv("SIMA_GST_ELEMENT_TIMINGS", "1", 1);
  setenv("SIMA_GST_FLOW_DEBUG", "1", 1);
  std::cout << "Debug env enabled. DOT dir: " << dot_dir.string() << "\n";
}

} // namespace

int main(int argc, char** argv) {
  try {
    if (wants_help(argc, argv)) {
      print_help(argv[0]);
      return 0;
    }

    const int w = parse_int_arg(argc, argv, "--width", 160);
    const int h = parse_int_arg(argc, argv, "--height", 120);
    const bool use_debug_env = has_flag(argc, argv, "--debug-env");
    const bool bad_input = has_flag(argc, argv, "--bad-input");

    const fs::path root = find_repo_root();
    if (use_debug_env) {
      enable_debug_env(root);
    }

    cv::Mat img(h, w, CV_8UC3, cv::Scalar(20, 120, 200));
    if (!img.isContinuous())
      img = img.clone();

    simaai::neat::Session p;
    simaai::neat::InputOptions in;
    in.format = "RGB";
    in.width = w;
    in.height = h;
    in.depth = 3;
    in.is_live = true;
    p.add(simaai::neat::nodes::Input(in));
    p.add(simaai::neat::nodes::Output());

    if (wants_print_gst(argc, argv)) {
      std::cout << p.describe_backend() << "\n";
      return 0;
    }

    simaai::neat::RunOptions run_opt;
    run_opt.output_memory = simaai::neat::OutputMemory::Owned;
    run_opt.enable_metrics = true;

    auto run = p.build(img, simaai::neat::RunMode::Async, run_opt);

    if (bad_input) {
      try {
        cv::Mat gray(h, w, CV_8UC1, cv::Scalar(80));
        run.push(gray);
      } catch (const std::exception& e) {
        std::cout << "Expected push failure: " << e.what() << "\n";
      }
    }

    run.push(img);
    auto quick = run.pull(/*timeout_ms=*/0);
    if (!quick.has_value()) {
      std::cout << "No output yet (0ms timeout). last_error=" << run.last_error() << "\n";
    }

    auto out = run.pull(/*timeout_ms=*/2000);
    if (!out.has_value()) {
      std::cout << "Timeout waiting for output. last_error=" << run.last_error() << "\n";
      std::cout << run.report() << "\n";
      return 1;
    }

    std::cout << "Output kind: " << static_cast<int>(out->kind) << "\n";
    std::cout << run.report() << "\n";

    const std::string last_err = run.last_error();
    if (!last_err.empty()) {
      std::cout << "last_error: " << last_err << "\n";
    }

    std::cout << "[OK] tutorial_0020 complete\n";
    return 0;
  } catch (const std::exception& e) {
    std::cerr << "[FAIL] " << e.what() << "\n";
    return 1;
  }
}
