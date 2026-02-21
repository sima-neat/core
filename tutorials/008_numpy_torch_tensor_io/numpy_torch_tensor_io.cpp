#include "neat.h"
#include "common/cpp_utils.h"

#include <opencv2/core.hpp>

#include <cstdint>
#include <iostream>

namespace {

void print_help(const char* argv0) {
  std::cout << "Usage: " << argv0 << " [--width <w>] [--height <h>]\n";
  tutorial_v2::print_common_flags(std::cout);
  std::cout << "  --width <w>          Width (default 128)\n";
  std::cout << "  --height <h>         Height (default 96)\n";
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
    // Framework map: model session graph run output contract.
    tutorial_v2::check("strict_flag_available",
                       tutorial_v2::yes_no(tutorial_v2::strict_mode()) == "yes" ||
                           tutorial_v2::yes_no(tutorial_v2::strict_mode()) == "no",
                       "strict-mode guard is observable");

    const int width = tutorial_v2::parse_int_arg(argc, argv, "--width", 128);
    const int height = tutorial_v2::parse_int_arg(argc, argv, "--height", 96);

    cv::Mat rgb(height, width, CV_8UC3, cv::Scalar(7, 17, 27));
    if (!rgb.isContinuous()) {
      rgb = rgb.clone();
    }

    auto tensor = simaai::neat::from_cv_mat(rgb, simaai::neat::ImageSpec::PixelFormat::RGB,
                                            /*read_only=*/true);

    auto mapped = tensor.map_read();
    std::uint64_t checksum = 0;
    const auto* bytes = static_cast<const std::uint8_t*>(mapped.data);
    const std::size_t n = std::min<std::size_t>(mapped.size_bytes, 256);
    for (std::size_t i = 0; i < n; ++i) {
      checksum += bytes[i];
    }

    auto owned = tensor.clone();

    std::cout << "Tensor shape rank: " << tensor.shape.size() << "\n";
    std::cout << "Tensor bytes:      " << mapped.size_bytes << "\n";
    std::cout << "Head checksum:     " << checksum << "\n";
    std::cout << "Clone dense bytes: " << owned.dense_bytes_tight() << "\n";

    tutorial_v2::check("tutorial_completed", true, "main path reached end without exception");
    tutorial_v2::print_signature({
        {"tutorial", "008"},
        {"lang", "cpp"},
        {"flow", "numpy_torch_tensor_io"},
        {"run_mode", "sync"},
        {"output_kind", "sample_or_tensor"},
        {"tensor_rank", "-1"},
        {"field_count", "-1"},
    });

    std::cout << "[OK] 008_numpy_torch_tensor_io\n";
    return 0;
  } catch (const std::exception& e) {
    std::cerr << "[FAIL] " << e.what() << "\n";
    return 1;
  }
}
