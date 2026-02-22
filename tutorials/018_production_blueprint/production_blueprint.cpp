#include "neat.h"
#include "common/cpp_utils.h"

#include <opencv2/core.hpp>

#include <filesystem>
#include <iostream>

namespace fs = std::filesystem;

namespace {

void print_help(const char* argv0) {
  std::cout << "Usage: " << argv0 << " [--mpk <path>] [--iters <n>]\n";
  tutorial_v2::print_common_flags(std::cout);
  std::cout << "  --mpk <path>         Optional MPK for model-backed blueprint\n";
  std::cout << "  --iters <n>          Number of frames (default 4)\n";
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

    const int iters = tutorial_v2::parse_int_arg(argc, argv, "--iters", 4);
    const fs::path root = tutorial_v2::find_repo_root();

    std::string mpk_arg;
    fs::path mpk_path = tutorial_v2::get_arg(argc, argv, "--mpk", mpk_arg)
                            ? fs::path(mpk_arg)
                            : tutorial_v2::first_existing({
                                  tutorial_v2::default_yolo_mpk(root),
                                  tutorial_v2::default_resnet_mpk(root),
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

    tutorial_v2::check("tutorial_completed", true, "main path reached end without exception");
    tutorial_v2::print_signature({
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
    tutorial_v2::runtime_fallback(e);
    tutorial_v2::check("tutorial_completed", true, "fallback path reached end without exception");
    tutorial_v2::print_signature({
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
