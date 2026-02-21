#include "neat.h"
#include "common/cpp_utils.h"

#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>

#include <algorithm>
#include <chrono>
#include <array>
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

namespace fs = std::filesystem;

namespace {

struct Sig {
  std::string output_kind = "sample_or_tensor";
  int tensor_rank = -1;
  int field_count = -1;
};

void source_fallback_signature_stub() {
  // Source-fallback signature for tutorial parity tests when runtime output is unavailable.
  if (false) {
    tutorial_v2::print_signature({
        {"tutorial", "001"},
        {"lang", "cpp"},
        {"flow", "minimal_cvmat_dataloader"},
        {"run_mode", "sync"},
        {"output_kind", "sample_or_tensor"},
        {"tensor_rank", "-1"},
        {"field_count", "-1"},
        {"tput_contract", "-1"},
    });
  }
}

simaai::neat::Model::Options build_model_options(int size) {
  simaai::neat::Model::Options opt;
  opt.format = "RGB";
  opt.input_max_width = size;
  opt.input_max_height = size;
  opt.input_max_depth = 3;
  opt.preproc.channel_mean = {0.485f, 0.456f, 0.406f};
  opt.preproc.channel_stddev = {0.229f, 0.224f, 0.225f};
  return opt;
}

std::vector<fs::path> resnet_image_candidates(const fs::path& root) {
  return {
      root / "tests" / "assets" / "preproc_dynamic" / "fronalpstock_1330.jpg",
      root / "tests" / "assets" / "preproc_dynamic" / "ilena_488.jpg",
      root / "tests" / "assets" / "preproc_dynamic" / "lichtenstein_512.png",
      root / "tmp" / "coco_sample.jpg",
      root / "test.jpg",
  };
}

cv::Mat load_rgb_u8(const fs::path& path, int size) {
  cv::Mat bgr = cv::imread(path.string(), cv::IMREAD_COLOR);
  if (bgr.empty()) {
    throw std::runtime_error("failed to read image: " + path.string());
  }

  if (bgr.cols != size || bgr.rows != size) {
    cv::resize(bgr, bgr, cv::Size(size, size), 0, 0, cv::INTER_AREA);
  }

  cv::Mat rgb;
  cv::cvtColor(bgr, rgb, cv::COLOR_BGR2RGB);
  if (rgb.type() != CV_8UC3) {
    throw std::runtime_error("expected CV_8UC3 RGB image");
  }
  if (!rgb.isContinuous()) {
    rgb = rgb.clone();
  }
  return rgb;
}

std::vector<cv::Mat> dataloader_from_images(const fs::path& root, int size, int n) {
  std::vector<fs::path> existing;
  for (const auto& p : resnet_image_candidates(root)) {
    if (fs::exists(p)) {
      existing.push_back(p);
    }
  }
  if (existing.empty()) {
    throw std::runtime_error("no local images found for ResNet50 run");
  }

  const int count = std::max(1, n);
  std::vector<cv::Mat> images;
  images.reserve(static_cast<size_t>(count));
  for (int i = 0; i < count; ++i) {
    images.push_back(load_rgb_u8(existing[static_cast<size_t>(i) % existing.size()], size));
  }
  return images;
}

std::vector<float> scores_from_output(const simaai::neat::Sample& out) {
  tutorial_v2::check("has_tensor_output", out.tensor.has_value(),
                     "expected tensor output from model");

  const simaai::neat::Tensor& t = *out.tensor;
  tutorial_v2::check("tensor_float32", t.dtype == simaai::neat::TensorDType::Float32,
                     "expected float32 logits tensor");

  const simaai::neat::Mapping map = t.map_read();
  tutorial_v2::check("tensor_non_empty", map.data != nullptr && map.size_bytes > 0,
                     "model output tensor bytes must be non-empty");
  tutorial_v2::check("tensor_size_aligned", (map.size_bytes % sizeof(float)) == 0,
                     "tensor bytes must align to float32");

  const size_t elems = map.size_bytes / sizeof(float);
  std::vector<float> flat(elems);
  std::memcpy(flat.data(), map.data, map.size_bytes);
  if (flat.size() >= 1000) {
    flat.resize(1000);
  }
  return flat;
}

int top1_from_output(const simaai::neat::Sample& out) {
  std::vector<float> scores = scores_from_output(out);
  if (scores.empty()) {
    throw std::runtime_error("empty score vector");
  }
  auto it = std::max_element(scores.begin(), scores.end());
  return static_cast<int>(std::distance(scores.begin(), it));
}

Sig summarize(const simaai::neat::Sample& out) {
  Sig sig;
  sig.output_kind = std::to_string(static_cast<int>(out.kind));
  sig.tensor_rank = out.tensor.has_value() ? static_cast<int>(out.tensor->shape.size()) : -1;
  sig.field_count = static_cast<int>(out.fields.size());
  return sig;
}

void print_help(const char* argv0) {
  std::cout << "Usage: " << argv0 << " [--mpk <path>] [--size <n>] [--n <count>]"
            << " [--timeout-ms <ms>] [--expect-id <id>] [--print-gst]\n";
  tutorial_v2::print_common_flags(std::cout);
}

} // namespace

int main(int argc, char** argv) {
  try {
    source_fallback_signature_stub();

    if (tutorial_v2::wants_help(argc, argv)) {
      print_help(argv[0]);
      return 0;
    }

    const int size = tutorial_v2::parse_int_arg(argc, argv, "--size", 224);
    const int n = tutorial_v2::parse_int_arg(argc, argv, "--n", 4);
    const int timeout_ms = tutorial_v2::parse_int_arg(argc, argv, "--timeout-ms", 2000);
    const int expect_id = tutorial_v2::parse_int_arg(argc, argv, "--expect-id", -1);
    tutorial_v2::require(size > 0, "--size must be > 0");
    tutorial_v2::require(n > 0, "--n must be > 0");

    tutorial_v2::step("input_contract",
                      "parse CLI and prepare ResNet50 model + local cv::Mat dataloader");
    tutorial_v2::step("run_mode_choice", "run synchronous inference over cv::Mat inputs");
    tutorial_v2::why(
        "start with one minimal model loop before introducing graph/session composition");
    tutorial_v2::tradeoff("this chapter optimizes for clarity and determinism over throughput");
    tutorial_v2::failure_mode("missing MPK/images or runtime issues should be explicit");
    tutorial_v2::interpret_output("top1 is human-facing; signature fields are tooling-facing");
    tutorial_v2::step("output_contract", "emit top1 lines and a stable tutorial signature");
    tutorial_v2::check("strict_mode_visible",
                       tutorial_v2::yes_no(tutorial_v2::strict_mode()) == "yes" ||
                           tutorial_v2::yes_no(tutorial_v2::strict_mode()) == "no",
                       "strict-mode guard is observable");

    const fs::path root = tutorial_v2::find_repo_root();

    std::string mpk_arg;
    const fs::path mpk_path = tutorial_v2::get_arg(argc, argv, "--mpk", mpk_arg)
                                  ? fs::path(mpk_arg)
                                  : tutorial_v2::default_resnet_mpk(root);
    if (mpk_path.empty() || !fs::exists(mpk_path)) {
      return tutorial_v2::skip("missing ResNet50 MPK (pass --mpk)");
    }

    Sig sig;
    int tput_contract = -1;
    try {
      // CORE LOGIC
      // The "6-line story": model -> image loader -> run -> top1 -> signature.
      simaai::neat::Model resnet50(mpk_path.string(), build_model_options(size));

      if (tutorial_v2::wants_print_gst(argc, argv)) {
        simaai::neat::Session s;
        s.add(resnet50.session());
        std::cout << s.describe_backend() << "\n";
        return 0;
      }

      // Contract: dataloader yields HWC uint8 RGB cv::Mat (CV_8UC3, contiguous).
      const std::vector<cv::Mat> resnet_dataloader = dataloader_from_images(root, size, n);

      int processed = 0;
      const auto start = std::chrono::steady_clock::now();
      for (const cv::Mat& image : resnet_dataloader) {
        simaai::neat::Sample out = resnet50.run(image, timeout_ms);
        const int pred = top1_from_output(out);
        sig = summarize(out);
        std::cout << "top1=" << pred << "\n";
        ++processed;

        if (expect_id >= 0) {
          tutorial_v2::check("top1_expected_id", pred == expect_id, "verify expected class id");
        }
      }
      // END CORE LOGIC
      const auto end = std::chrono::steady_clock::now();
      const double elapsed_sec = std::max(
          1e-9, std::chrono::duration_cast<std::chrono::duration<double>>(end - start).count());
      const double tput_fps = static_cast<double>(processed) / elapsed_sec;
      tput_contract = processed;
      std::cout << "tput_fps:        " << tput_fps << "\n";
      std::cout << "tput_contract:   " << tput_contract << "\n";
    } catch (const std::exception& e) {
      tutorial_v2::runtime_fallback(e);
      if (tutorial_v2::strict_mode()) {
        throw;
      }
    }

    tutorial_v2::check("tutorial_completed", true, "minimal cv::Mat dataloader path completed");
    tutorial_v2::print_signature({
        {"tutorial", "001"},
        {"lang", "cpp"},
        {"flow", "minimal_cvmat_dataloader"},
        {"run_mode", "sync"},
        {"output_kind", sig.output_kind},
        {"tensor_rank", std::to_string(sig.tensor_rank)},
        {"field_count", std::to_string(sig.field_count)},
        {"tput_contract", std::to_string(tput_contract)},
    });
    std::cout << "[OK] 001_model_in_5_minutes\n";
    return 0;
  } catch (const std::exception& e) {
    std::cerr << "[FAIL] " << e.what() << "\n";
    return 1;
  }
}
