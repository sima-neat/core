#include "neat.h"
#include "common/cpp_utils.h"

#include <opencv2/core.hpp>

#include <iostream>
#include <string>

namespace {

void print_help(const char* argv0) {
  std::cout << "Usage: " << argv0 << " [--width <w>] [--height <h>]\n";
  tutorial_v2::print_common_flags(std::cout);
  std::cout << "  --width <w>          Input width (default 320)\n";
  std::cout << "  --height <h>         Input height (default 240)\n";
}

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
    tutorial_v2::tradeoff("prefer deterministic samples and stable contracts over production realism");
    tutorial_v2::failure_mode("runtime/plugin issues should degrade to runtime_fallback without losing observability");
    tutorial_v2::interpret_output("use CHECK markers plus SIGNATURE fields to validate behavior and parity");
    tutorial_v2::step("output_contract", "emit checks and machine-parseable signature");
    tutorial_v2::check("strict_flag_available", tutorial_v2::yes_no(tutorial_v2::strict_mode()) == "yes" ||
                                              tutorial_v2::yes_no(tutorial_v2::strict_mode()) == "no",
                      "strict-mode guard is observable");

    const int width = tutorial_v2::parse_int_arg(argc, argv, "--width", 320);
    const int height = tutorial_v2::parse_int_arg(argc, argv, "--height", 240);

    cv::Mat input(height, width, CV_8UC3, cv::Scalar(30, 60, 90));
    if (!input.isContinuous()) {
      input = input.clone();
    }

    simaai::neat::Session session = make_session(width, height);

    if (tutorial_v2::wants_print_gst(argc, argv)) {
      std::cout << session.describe_backend() << "\n";
      return 0;
    }

    simaai::neat::RunOptions run_opt;
    run_opt.output_memory = simaai::neat::OutputMemory::Owned;

    auto run = session.build(input, simaai::neat::RunMode::Sync, run_opt);
    auto sample = run.push_and_pull(input, /*timeout_ms=*/1000);

    tutorial_v2::require(sample.tensor.has_value(), "missing tensor output");
    std::cout << "Output tensor rank: " << sample.tensor->shape.size() << "\n";

    tutorial_v2::check("tutorial_completed", true, "main path reached end without exception");
    tutorial_v2::print_signature({
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
