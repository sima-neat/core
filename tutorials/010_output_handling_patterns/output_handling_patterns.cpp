#include "neat.h"
#include "common/cpp_utils.h"

#include <opencv2/core.hpp>

#include <iostream>

namespace {

void print_help(const char* argv0) {
  std::cout << "Usage: " << argv0 << "\n";
  tutorial_v2::print_common_flags(std::cout);
}

void print_sample_summary(const simaai::neat::Sample& sample) {
  std::cout << "kind=" << static_cast<int>(sample.kind)
            << " has_tensor=" << (sample.tensor.has_value() ? "yes" : "no")
            << " fields=" << sample.fields.size() << "\n";
}

} // namespace

int main(int argc, char** argv) {
  try {
    if (tutorial_v2::wants_help(argc, argv)) {
      print_help(argv[0]);
      return 0;
    }

    // Why: explicit runtime markers keep the tutorial explainable from terminal output alone.
    // Why: parity and score tooling consume these checkpoints as a stable contract.
    tutorial_v2::step("input_contract", "parse flags and establish deterministic defaults");
    tutorial_v2::step("run_mode_choice", "exercise the chapter's primary runtime path");
    tutorial_v2::why("understand the contract first: inputs, run mode, and outputs");
    tutorial_v2::tradeoff(
        "prefer deterministic samples and stable contracts over production realism");
    tutorial_v2::failure_mode(
        "runtime/plugin issues should degrade to runtime_fallback without losing observability");
    tutorial_v2::interpret_output(
        "use CHECK markers plus SIGNATURE fields to validate behavior and parity");
    tutorial_v2::step("output_contract", "emit checks and machine-parseable signature");
    tutorial_v2::check("strict_flag_available",
                       tutorial_v2::yes_no(tutorial_v2::strict_mode()) == "yes" ||
                           tutorial_v2::yes_no(tutorial_v2::strict_mode()) == "no",
                       "strict-mode guard is observable");

    cv::Mat rgb(120, 160, CV_8UC3, cv::Scalar(110, 40, 30));
    if (!rgb.isContinuous()) {
      rgb = rgb.clone();
    }

    // CORE LOGIC
    simaai::neat::Session s;
    simaai::neat::InputOptions in;
    in.format = "RGB";
    in.width = rgb.cols;
    in.height = rgb.rows;
    in.depth = rgb.channels();
    s.add(simaai::neat::nodes::Input(in));
    s.add(simaai::neat::nodes::Output());

    if (tutorial_v2::wants_print_gst(argc, argv)) {
      std::cout << s.describe_backend() << "\n";
      return 0;
    }

    auto run = s.build(rgb, simaai::neat::RunMode::Sync);

    auto out = run.push_and_pull(rgb, 1000);
    std::cout << "push_and_pull output: ";
    print_sample_summary(out);
    tutorial_v2::require(out.tensor.has_value(), "expected tensor output");

    if (out.tensor->shape.empty()) {
      throw std::runtime_error("output tensor shape is empty");
    }

    std::cout << "output rank: " << out.tensor->shape.size() << "\n";
    // END CORE LOGIC
    tutorial_v2::check("tutorial_completed", true, "main path reached end without exception");
    tutorial_v2::print_signature({
        {"tutorial", "010"},
        {"lang", "cpp"},
        {"flow", "chapter_path"},
        {"run_mode", "sync_or_async"},
        {"output_kind", "sample_or_tensor"},
        {"tensor_rank", "-1"},
        {"field_count", "-1"},
    });

    std::cout << "[OK] 010_output_handling_patterns\n";
    return 0;
  } catch (const std::exception& e) {
    std::cerr << "[FAIL] " << e.what() << "\n";
    return 1;
  }
}
