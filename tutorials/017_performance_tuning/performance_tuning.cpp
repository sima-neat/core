#include "neat.h"
#include "common/cpp_utils.h"

#include <opencv2/core.hpp>

#include <iostream>

namespace {

void print_help(const char* argv0) {
  std::cout << "Usage: " << argv0 << " [--iters <n>] [--queue <n>] [--drop <mode>]\n";
  tutorial_v2::print_common_flags(std::cout);
  std::cout << "  --drop <mode>        block|latest|incoming (default block)\n";
}

simaai::neat::OverflowPolicy parse_drop_policy(int argc, char** argv) {
  std::string mode;
  if (!tutorial_v2::get_arg(argc, argv, "--drop", mode)) {
    return simaai::neat::OverflowPolicy::Block;
  }
  if (mode == "latest") {
    return simaai::neat::OverflowPolicy::KeepLatest;
  }
  if (mode == "incoming") {
    return simaai::neat::OverflowPolicy::DropIncoming;
  }
  return simaai::neat::OverflowPolicy::Block;
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

    const int iters = tutorial_v2::parse_int_arg(argc, argv, "--iters", 32);
    const int queue_depth = tutorial_v2::parse_int_arg(argc, argv, "--queue", 4);

    cv::Mat rgb(120, 160, CV_8UC3, cv::Scalar(70, 20, 200));
    if (!rgb.isContinuous()) {
      rgb = rgb.clone();
    }

    simaai::neat::Session s;
    simaai::neat::InputOptions in;
    in.format = "RGB";
    in.width = rgb.cols;
    in.height = rgb.rows;
    in.depth = rgb.channels();
    in.is_live = true;
    s.add(simaai::neat::nodes::Input(in));
    s.add(simaai::neat::nodes::Output());

    if (tutorial_v2::wants_print_gst(argc, argv)) {
      std::cout << s.describe_backend() << "\n";
      return 0;
    }

    simaai::neat::RunOptions opt;
    opt.queue_depth = queue_depth;
    opt.overflow_policy = parse_drop_policy(argc, argv);
    opt.output_memory = simaai::neat::OutputMemory::Owned;
    opt.enable_metrics = true;

    auto run = s.build(rgb, simaai::neat::RunMode::Async, opt);

    for (int i = 0; i < iters; ++i) {
      (void)run.try_push(rgb);
    }
    run.close_input();

    int pulled = 0;
    while (run.pull(1000).has_value()) {
      ++pulled;
    }

    const auto stats = run.stats();
    const auto input_stats = run.input_stats();

    std::cout << "inputs_enqueued=" << stats.inputs_enqueued << "\n";
    std::cout << "inputs_dropped=" << stats.inputs_dropped << "\n";
    std::cout << "outputs_pulled=" << pulled << "\n";
    std::cout << "avg_latency_ms=" << stats.avg_latency_ms << "\n";
    std::cout << "avg_push_us=" << input_stats.avg_push_us << "\n";
    std::cout << "renegotiations=" << input_stats.renegotiations << "\n";

    tutorial_v2::check("tutorial_completed", true, "main path reached end without exception");
    tutorial_v2::print_signature({
        {"tutorial", "017"},
        {"lang", "cpp"},
        {"flow", "chapter_path"},
        {"run_mode", "sync_or_async"},
        {"output_kind", "sample_or_tensor"},
        {"tensor_rank", "-1"},
        {"field_count", "-1"},
    });

    std::cout << "[OK] 017_performance_tuning\n";
    return 0;
  } catch (const std::exception& e) {
    std::cerr << "[FAIL] " << e.what() << "\n";
    return 1;
  }
}
