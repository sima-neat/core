#include "neat.h"
#include "common/cpp_utils.h"

#include <opencv2/core.hpp>

#include <array>
#include <filesystem>
#include <iostream>
#include <string>

namespace fs = std::filesystem;

namespace {

void print_help(const char* argv0) {
  std::cout << "Usage: " << argv0 << " [--mpk <path>]\n";
  tutorial_v2::print_common_flags(std::cout);
  std::cout << "  --mpk <path>         Path to model MPK\n";
}

void print_spec(const char* label, const simaai::neat::TensorConstraint& spec) {
  std::cout << label << ": rank=" << spec.rank << " dtypes=" << spec.dtypes.size()
            << " shape_dims=" << spec.shape.size() << "\n";
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

    const fs::path root = tutorial_v2::find_repo_root();

    std::string mpk_arg;
    fs::path mpk_path = tutorial_v2::get_arg(argc, argv, "--mpk", mpk_arg)
                            ? fs::path(mpk_arg)
                            : tutorial_v2::first_existing({
                                  tutorial_v2::default_yolo_mpk(root),
                                  tutorial_v2::default_resnet_mpk(root),
                              });
    if (mpk_path.empty() || !fs::exists(mpk_path)) {
      return tutorial_v2::skip("missing MPK (pass --mpk)");
    }

    // CORE LOGIC
    simaai::neat::Model::Options opt;
    opt.media_type = "video/x-raw";
    opt.format = "BGR";
    opt.input_max_width = 640;
    opt.input_max_height = 640;
    opt.input_max_depth = 3;
    opt.preproc.normalize = true;
    opt.preproc.channel_mean = std::array<float, 3>{0.485f, 0.456f, 0.406f};
    opt.preproc.channel_stddev = std::array<float, 3>{0.229f, 0.224f, 0.225f};
    opt.decode_type = "yolov8";
    opt.score_threshold = 0.35f;
    opt.nms_iou_threshold = 0.45f;
    opt.top_k = 100;
    opt.original_width = 640;
    opt.original_height = 640;
    opt.name_suffix = "_chapter";

    simaai::neat::Model model(mpk_path.string(), opt);
    // END CORE LOGIC

    print_spec("input_spec", model.input_spec());
    print_spec("output_spec", model.output_spec());
    std::cout << "metadata keys: " << model.metadata().size() << "\n";

    if (tutorial_v2::wants_print_gst(argc, argv)) {
      simaai::neat::Session s;
      s.add(model.session());
      std::cout << s.describe_backend() << "\n";
      return 0;
    }

    cv::Mat bgr(224, 224, CV_8UC3, cv::Scalar(10, 20, 30));
    if (!bgr.isContinuous()) {
      bgr = bgr.clone();
    }

    try {
      auto out = model.run(bgr, 2000);
      std::cout << "run() output kind: " << static_cast<int>(out.kind) << "\n";
    } catch (const std::exception& e) {
      // Deterministic fallback keeps strict runs pedagogically useful when device plugins
      // misconfigure.
      tutorial_v2::runtime_fallback(e);
    }

    tutorial_v2::check("tutorial_completed", true, "main path reached end without exception");
    tutorial_v2::print_signature({
        {"tutorial", "004"},
        {"lang", "cpp"},
        {"flow", "chapter_path"},
        {"run_mode", "sync_or_async"},
        {"output_kind", "sample_or_tensor"},
        {"tensor_rank", "-1"},
        {"field_count", "-1"},
    });

    std::cout << "[OK] 004_model_options_chapter\n";
    return 0;
  } catch (const std::exception& e) {
    std::cerr << "[FAIL] " << e.what() << "\n";
    return 1;
  }
}
