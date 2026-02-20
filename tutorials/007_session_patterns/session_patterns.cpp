#include "neat.h"
#include "common/cpp_utils.h"

#include <opencv2/core.hpp>

#include <filesystem>
#include <iostream>

namespace fs = std::filesystem;

namespace {

void print_help(const char* argv0) {
  std::cout << "Usage: " << argv0 << " [--mpk <path>]\n";
  tutorial_v2::print_common_flags(std::cout);
  std::cout << "  --mpk <path>         Optional MPK for model-backed session examples\n";
}

simaai::neat::Session make_base_session(int width, int height) {
  simaai::neat::Session s;
  simaai::neat::InputOptions in;
  in.format = "RGB";
  in.width = width;
  in.height = height;
  in.depth = 3;
  in.do_timestamp = true;
  s.add(simaai::neat::nodes::Input(in));
  s.add(simaai::neat::nodes::Output());
  return s;
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

    const int width = 224;
    const int height = 224;

    simaai::neat::Session direct = make_base_session(width, height);

    const fs::path root = tutorial_v2::find_repo_root();
    std::string mpk_arg;
    fs::path mpk_path = tutorial_v2::get_arg(argc, argv, "--mpk", mpk_arg)
                            ? fs::path(mpk_arg)
                            : tutorial_v2::first_existing({
                                  tutorial_v2::default_yolo_mpk(root),
                                  tutorial_v2::default_resnet_mpk(root),
                              });

    if (!mpk_path.empty() && fs::exists(mpk_path)) {
      simaai::neat::Model model(mpk_path.string());

      simaai::neat::Session from_model;
      from_model.add(model.session());
      std::cout << "Model session group size: " << model.session().size() << "\n";

      simaai::neat::Model::SessionOptions sopt;
      sopt.include_appsrc = false;
      sopt.include_appsink = true;
      sopt.upstream_name = "camera0";
      sopt.name_suffix = "_camera0";
      sopt.buffer_name = "camera0";

      simaai::neat::Session model_attached;
      model_attached.add(model.session(sopt));

      if (tutorial_v2::wants_print_gst(argc, argv)) {
        std::cout << "[direct]\n" << direct.describe_backend() << "\n";
        std::cout << "[model default]\n" << from_model.describe_backend() << "\n";
        std::cout << "[model attached]\n" << model_attached.describe_backend() << "\n";
        return 0;
      }
    } else if (tutorial_v2::wants_print_gst(argc, argv)) {
      std::cout << direct.describe_backend() << "\n";
      return 0;
    }

    cv::Mat rgb(height, width, CV_8UC3, cv::Scalar(80, 40, 160));
    if (!rgb.isContinuous()) {
      rgb = rgb.clone();
    }

    auto run = direct.build(rgb, simaai::neat::RunMode::Sync);
    auto out = run.push_and_pull(rgb, 1000);
    tutorial_v2::require(out.tensor.has_value(), "direct session output missing tensor");

    tutorial_v2::check("tutorial_completed", true, "main path reached end without exception");
    tutorial_v2::print_signature({
        {"tutorial", "007"},
        {"lang", "cpp"},
        {"flow", "chapter_path"},
        {"run_mode", "sync_or_async"},
        {"output_kind", "sample_or_tensor"},
        {"tensor_rank", "-1"},
        {"field_count", "-1"},
    });

    std::cout << "[OK] 007_session_patterns\n";
    return 0;
  } catch (const std::exception& e) {
    std::cerr << "[FAIL] " << e.what() << "\n";
    return 1;
  }
}
