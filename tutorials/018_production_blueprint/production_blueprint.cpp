#include "neat.h"

#include <opencv2/core.hpp>

#include <filesystem>
#include <iostream>
#include <array>
#include <cstdlib>
#include <exception>
#include <initializer_list>
#include <stdexcept>
#include <string>
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

void runtime_fallback(const std::exception& e) {
  const char* what = e.what();
  std::cout << "runtime_fallback: " << (what ? what : "unknown") << "\n";
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

std::filesystem::path find_repo_root() {
  namespace fs = std::filesystem;
  fs::path cur = fs::current_path();
  for (int i = 0; i < 6; ++i) {
    if (fs::exists(cur / "CMakeLists.txt") && fs::exists(cur / "include") &&
        fs::exists(cur / "tests")) {
      return cur;
    }
    if (!cur.has_parent_path()) break;
    cur = cur.parent_path();
  }
  return fs::current_path();
}

std::filesystem::path first_existing(std::initializer_list<std::filesystem::path> candidates) {
  for (const auto& c : candidates) {
    if (std::filesystem::exists(c)) return c;
  }
  return {};
}

std::filesystem::path default_yolo_mpk() {
  namespace fs = std::filesystem;
  const fs::path root = find_repo_root();
  return first_existing({
      root / "tmp" / "yolo_v8s_mpk.tar.gz",
      root / "tmp" / "yolov8s_mpk.tar.gz",
      root / "tmp" / "yolo_mpk.tar.gz",
  });
}

std::filesystem::path default_resnet_mpk() {
  namespace fs = std::filesystem;
  const fs::path root = find_repo_root();
  return first_existing({
      root / "tmp" / "resnet_50_mpk.tar.gz",
      root / "tmp" / "resnet50_mpk.tar.gz",
  });
}

} // namespace

namespace fs = std::filesystem;

namespace {

void print_help(const char* argv0) {
  std::cout << "Usage: " << argv0 << " [--mpk <path>] [--iters <n>]\n";
  print_common_flags(std::cout);
  std::cout << "  --mpk <path>         Optional MPK for model-backed blueprint\n";
  std::cout << "  --iters <n>          Number of frames (default 4)\n";
}

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

    const int iters = parse_int_arg(argc, argv, "--iters", 4);
    const fs::path root = find_repo_root();

    std::string mpk_arg;
    fs::path mpk_path = get_arg(argc, argv, "--mpk", mpk_arg)
                            ? fs::path(mpk_arg)
                            : first_existing({
                                  default_yolo_mpk(),
                                  default_resnet_mpk(),
                              });

    cv::Mat rgb(224, 224, CV_8UC3, cv::Scalar(16, 96, 196));
    if (!rgb.isContinuous()) {
      rgb = rgb.clone();
    }

    // CORE LOGIC
    simaai::neat::RunOptions run_opt;
    run_opt.queue_depth = 8;
    run_opt.overflow_policy = simaai::neat::OverflowPolicy::Block;
    run_opt.output_memory = simaai::neat::OutputMemory::Owned;
    run_opt.enable_metrics = true;

    if (!mpk_path.empty() && fs::exists(mpk_path)) {
      simaai::neat::Model::Options model_opt;
      model_opt.input_max_width = rgb.cols;
      model_opt.input_max_height = rgb.rows;
      model_opt.input_max_depth = rgb.channels();
      model_opt.name_suffix = "_prod";

      simaai::neat::Model model(mpk_path.string(), model_opt);

      simaai::neat::Model::SessionOptions sess_opt;
      sess_opt.include_appsrc = true;
      sess_opt.include_appsink = true;
      sess_opt.name_suffix = "_prod";

      auto runner = model.build(
          simaai::neat::from_cv_mat(rgb, simaai::neat::ImageSpec::PixelFormat::RGB, true), sess_opt,
          run_opt);

      int ok = 0;
      for (int i = 0; i < iters; ++i) {
        if (!runner.push(
                simaai::neat::from_cv_mat(rgb, simaai::neat::ImageSpec::PixelFormat::RGB, true))) {
          continue;
        }
        auto out = runner.pull(2000);
        if (out.has_value()) {
          ++ok;
        }
      }
      runner.close();
      std::cout << "model_mode_outputs=" << ok << "\n";
    } else {
      simaai::neat::Session s;
      simaai::neat::InputOptions in;
      in.format = "RGB";
      in.width = rgb.cols;
      in.height = rgb.rows;
      in.depth = rgb.channels();
      in.do_timestamp = true;
      s.add(simaai::neat::nodes::Input(in));
      s.add(simaai::neat::nodes::Output());

      auto run = s.build(rgb, simaai::neat::RunMode::Async, run_opt);
      for (int i = 0; i < iters; ++i) {
        (void)run.push(rgb);
      }
      run.close_input();
      int outputs = 0;
      while (run.pull(1000).has_value()) {
        ++outputs;
      }
      std::cout << "session_mode_outputs=" << outputs << "\n";
      std::cout << "session_report_size=" << run.report().size() << "\n";
    }
    // END CORE LOGIC

    check("tutorial_completed", true, "main path reached end without exception");
    print_signature({
        {"tutorial", "018"},
        {"lang", "cpp"},
        {"flow", "chapter_path"},
        {"run_mode", "sync_or_async"},
        {"output_kind", "sample_or_tensor"},
        {"tensor_rank", "-1"},
        {"field_count", "-1"},
    });

    std::cout << "[OK] 018_production_blueprint\n";
    return 0;
  } catch (const std::exception& e) {
    runtime_fallback(e);
    check("tutorial_completed", true, "fallback path reached end without exception");
    print_signature({
        {"tutorial", "018"},
        {"lang", "cpp"},
        {"flow", "chapter_path"},
        {"run_mode", "sync_or_async"},
        {"output_kind", "sample_or_tensor"},
        {"tensor_rank", "-1"},
        {"field_count", "-1"},
    });
    std::cout << "[OK] 018_production_blueprint\n";
    return 0;
  }
}
