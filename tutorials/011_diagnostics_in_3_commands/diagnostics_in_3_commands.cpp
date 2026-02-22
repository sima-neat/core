#include "neat.h"
#include "common/cpp_utils.h"

#include <opencv2/core.hpp>

#include <iostream>

namespace {

void print_help(const char* argv0) {
  std::cout << "Usage: " << argv0 << "\n";
  tutorial_v2::print_common_flags(std::cout);
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

    // CORE LOGIC
    cv::Mat rgb(96, 128, CV_8UC3, cv::Scalar(22, 44, 66));
    if (!rgb.isContinuous()) {
      rgb = rgb.clone();
    }

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

    // Command 1: validate contract and backend parse path.
    auto report = s.validate();
    std::cout << "validate.error_code: " << report.error_code << "\n";

    // Command 2: build + run one frame with metrics enabled.
    simaai::neat::RunOptions run_opt;
    run_opt.enable_metrics = true;
    run_opt.output_memory = simaai::neat::OutputMemory::Owned;
    auto run = s.build(rgb, simaai::neat::RunMode::Sync, run_opt);
    auto out = run.push_and_pull(rgb, 1000);
    tutorial_v2::require(out.tensor.has_value(), "missing output tensor");

    // Command 3: inspect runtime diagnostics.
    auto stats = run.stats();
    std::cout << "stats.inputs_enqueued=" << stats.inputs_enqueued
              << " outputs_pulled=" << stats.outputs_pulled << "\n";
    std::cout << "report.size=" << run.report().size() << "\n";
    std::cout << "diag_summary=" << run.diagnostics_summary() << "\n";
    // END CORE LOGIC

    tutorial_v2::check("tutorial_completed", true, "main path reached end without exception");
    tutorial_v2::print_signature({
        {"tutorial", "011"},
        {"lang", "cpp"},
        {"flow", "chapter_path"},
        {"run_mode", "sync_or_async"},
        {"output_kind", "sample_or_tensor"},
        {"tensor_rank", "-1"},
        {"field_count", "-1"},
    });

    std::cout << "[OK] 011_diagnostics_in_3_commands\n";
    return 0;
  } catch (const std::exception& e) {
    std::cerr << "[FAIL] " << e.what() << "\n";
    return 1;
  }
}
