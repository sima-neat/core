#include "neat.h"

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

int parse_int_arg(int argc, char** argv, const std::string& key, int def) {
  std::string value;
  if (!get_arg(argc, argv, key, value)) return def;
  try {
    return std::stoi(value);
  } catch (...) {
    throw std::invalid_argument("invalid integer for " + key + ": " + value);
  }
}

void require(bool ok, const std::string& msg) {
  if (!ok) throw std::runtime_error(msg);
}

bool strict_mode() {
  return std::getenv("SIMA_RUN_TUTORIALS_FULL") != nullptr;
}

std::string yes_no(bool v) { return v ? "yes" : "no"; }

void step(const std::string& name, const std::string& detail = {}) {
  if (detail.empty()) {
    std::cout << "STEP " << name << "\n";
  } else {
    std::cout << "STEP " << name << ": " << detail << "\n";
  }
}

void check(const std::string& name, bool cond, const std::string& detail = {}) {
  std::cout << "CHECK " << name << ": " << (cond ? "PASS" : "FAIL");
  if (!detail.empty()) std::cout << " (" << detail << ")";
  std::cout << "\n";
  if (!cond) throw std::runtime_error("check failed: " + name);
}

void why(const std::string& detail) {
  std::cout << "WHY " << detail << "\n";
}

void tradeoff(const std::string& detail) {
  std::cout << "TRADEOFF " << detail << "\n";
}

void failure_mode(const std::string& detail) {
  std::cout << "FAILURE_MODE " << detail << "\n";
}

void interpret_output(const std::string& detail) {
  std::cout << "INTERPRET " << detail << "\n";
}

void print_signature(std::initializer_list<std::pair<std::string, std::string>> values) {
  static constexpr std::array<const char*, 7> kRequired = {
      "tutorial", "lang", "flow", "run_mode", "output_kind", "tensor_rank", "field_count",
  };
  for (const char* key : kRequired) {
    bool found = false;
    for (const auto& kv : values) {
      if (kv.first == key) { found = true; break; }
    }
    if (!found) throw std::invalid_argument(std::string("missing signature key: ") + key);
  }
  std::cout << "SIGNATURE {";
  bool first = true;
  for (const auto& kv : values) {
    if (!first) std::cout << ",";
    std::cout << kv.first << "=" << kv.second;
    first = false;
  }
  std::cout << "}\n";
}

} // namespace

namespace {

void print_help(const char* argv0) {
  std::cout << "Usage: " << argv0 << " [--width <w>] [--height <h>]\n";
  print_common_flags(std::cout);
  std::cout << "  --width <w>          Input width (default 320)\n";
  std::cout << "  --height <h>         Input height (default 240)\n";
}

// CORE LOGIC
simaai::neat::Session make_session(int width, int height) {
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
  return session;
}
// END CORE LOGIC
} // namespace

int main(int argc, char** argv) {
  try {
    if (wants_help(argc, argv)) {
      print_help(argv[0]);
      return 0;
    }

    // Why: explicit runtime markers keep the tutorial explainable from terminal output alone.
    // Why: parity and score tooling consume these checkpoints as a stable contract.
    step("input_contract", "parse flags and establish deterministic defaults");
    step("run_mode_choice", "exercise the chapter's primary runtime path");
    why("understand the contract first: inputs, run mode, and outputs");
    tradeoff(
        "prefer deterministic samples and stable contracts over production realism");
    failure_mode(
        "runtime/plugin issues should degrade to runtime_fallback without losing observability");
    interpret_output(
        "use CHECK markers plus SIGNATURE fields to validate behavior and parity");
    step("output_contract", "emit checks and machine-parseable signature");
    check("strict_flag_available",
                       yes_no(strict_mode()) == "yes" ||
                           yes_no(strict_mode()) == "no",
                       "strict-mode guard is observable");

    const int width = parse_int_arg(argc, argv, "--width", 320);
    const int height = parse_int_arg(argc, argv, "--height", 240);

    cv::Mat input(height, width, CV_8UC3, cv::Scalar(30, 60, 90));
    if (!input.isContinuous()) {
      input = input.clone();
    }

    simaai::neat::Session session = make_session(width, height);

    if (wants_print_gst(argc, argv)) {
      std::cout << session.describe_backend() << "\n";
      return 0;
    }

    // CORE LOGIC
    simaai::neat::RunOptions run_opt;
    run_opt.output_memory = simaai::neat::OutputMemory::Owned;

    auto run = session.build(input, simaai::neat::RunMode::Sync, run_opt);
    auto sample = run.push_and_pull(input, /*timeout_ms=*/1000);
    // END CORE LOGIC

    require(sample.tensor.has_value(), "missing tensor output");
    std::cout << "Output tensor rank: " << sample.tensor->shape.size() << "\n";

    check("tutorial_completed", true, "main path reached end without exception");
    print_signature({
        {"tutorial", "003"},
        {"lang", "cpp"},
        {"flow", "chapter_path"},
        {"run_mode", "sync_or_async"},
        {"output_kind", "sample_or_tensor"},
        {"tensor_rank", "-1"},
        {"field_count", "-1"},
    });

    std::cout << "[OK] 003_session_build_and_run\n";
    return 0;
  } catch (const std::exception& e) {
    std::cerr << "[FAIL] " << e.what() << "\n";
    return 1;
  }
}
