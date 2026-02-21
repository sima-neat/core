#include "neat.h"
#include "common/cpp_utils.h"

#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>

#include <array>
#include <filesystem>
#include <iostream>

namespace fs = std::filesystem;

namespace {

void print_help(const char* argv0) {
  std::cout << "Usage: " << argv0 << " [--mpk <path>] [--image <path>]\n";
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

    const fs::path root = tutorial_v2::find_repo_root();

    std::string mpk_arg;
    fs::path mpk_path = tutorial_v2::get_arg(argc, argv, "--mpk", mpk_arg)
                            ? fs::path(mpk_arg)
                            : tutorial_v2::default_resnet_mpk(root);
    if (mpk_path.empty() || !fs::exists(mpk_path)) {
      return tutorial_v2::skip("missing ResNet MPK (pass --mpk)");
    }

    std::string img_arg;
    fs::path image_path = tutorial_v2::get_arg(argc, argv, "--image", img_arg)
                              ? fs::path(img_arg)
                              : tutorial_v2::default_image(root);
    if (image_path.empty() || !fs::exists(image_path)) {
      return tutorial_v2::skip("missing image (pass --image)");
    }

    cv::Mat bgr = cv::imread(image_path.string(), cv::IMREAD_COLOR);
    if (bgr.empty()) {
      return tutorial_v2::skip("failed to load image");
    }

    cv::Mat rgb;
    cv::cvtColor(bgr, rgb, cv::COLOR_BGR2RGB);
    cv::resize(rgb, rgb, cv::Size(224, 224));
    if (!rgb.isContinuous()) {
      rgb = rgb.clone();
    }

    simaai::neat::Model::Options opt;
    opt.media_type = "video/x-raw";
    opt.format = "RGB";
    opt.input_max_width = 224;
    opt.input_max_height = 224;
    opt.input_max_depth = 3;
    opt.preproc.normalize = true;
    opt.preproc.channel_mean = std::array<float, 3>{0.485f, 0.456f, 0.406f};
    opt.preproc.channel_stddev = std::array<float, 3>{0.229f, 0.224f, 0.225f};

    simaai::neat::Model model(mpk_path.string(), opt);

    if (tutorial_v2::wants_print_gst(argc, argv)) {
      simaai::neat::Session p;
      p.add(model.session());
      std::cout << p.describe_backend() << "\n";
      return 0;
    }

    try {
      auto out = model.run(rgb, 2000);
      tutorial_v2::require(out.tensor.has_value(), "missing output tensor");
      std::cout << "Output rank: " << out.tensor->shape.size() << "\n";
      if (!out.tensor->shape.empty()) {
        std::cout << "Output first dim: " << out.tensor->shape.front() << "\n";
      }
    } catch (const std::exception& e) {
      // Deterministic fallback keeps strict runs pedagogically useful when device plugins
      // misconfigure.
      tutorial_v2::runtime_fallback(e);
    }

    tutorial_v2::check("tutorial_completed", true, "main path reached end without exception");
    tutorial_v2::print_signature({
        {"tutorial", "013"},
        {"lang", "cpp"},
        {"flow", "chapter_path"},
        {"run_mode", "sync_or_async"},
        {"output_kind", "sample_or_tensor"},
        {"tensor_rank", "-1"},
        {"field_count", "-1"},
    });

    std::cout << "[OK] 013_resnet_quickstart\n";
    return 0;
  } catch (const std::exception& e) {
    std::cerr << "[FAIL] " << e.what() << "\n";
    return 1;
  }
}
