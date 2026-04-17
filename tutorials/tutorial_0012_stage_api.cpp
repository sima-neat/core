// tutorial_0012_stage_api.cpp
// Story: stage APIs let you run Preproc/Infer/BoxDecode without a full pipeline.
// What you learn:
// - stages::Preproc and stages::Infer operate on simaai::neat::Tensor payloads.
// - stages::BoxDecode converts tensors into BoxDecodeResult.
// - Useful for hybrid flows or custom scheduling.

#include "neat/models.h"
#include "gst/GstHelpers.h"

#include "tutorial_common.h"

#include <opencv2/imgcodecs.hpp>

#include <filesystem>
#include <iostream>
#include <string>

namespace fs = std::filesystem;

namespace {

void print_help(const char* argv0) {
  std::cout << "Usage: " << argv0 << " [--mpk <path>] [--image <path>]\n";
  sima_tutorial::print_common_flags(std::cout);
  std::cout << "  --mpk <path>         Path to YOLOv8 MPK tar.gz (default: search tmp/)\n";
  std::cout << "  --image <path>       Input image (default: shipped tutorial sample)\n";
}

fs::path find_default_mpk(const fs::path& root) {
  const fs::path c1 = root / "tmp" / "yolo_v8s_mpk.tar.gz";
  const fs::path c2 = root / "tmp" / "yolov8s_mpk.tar.gz";
  if (fs::exists(c1))
    return c1;
  if (fs::exists(c2))
    return c2;
  return {};
}

fs::path find_default_image() {
  return sima_tutorial::find_asset_root() / "ilena_488.jpg";
}

} // namespace

int main(int argc, char** argv) {
  try {
    if (sima_tutorial::wants_help(argc, argv)) {
      print_help(argv[0]);
      return 0;
    }

    const fs::path root = sima_tutorial::find_repo_root();

    std::string mpk_arg;
    fs::path mpk_path = sima_tutorial::get_arg(argc, argv, "--mpk", mpk_arg)
                            ? fs::path(mpk_arg)
                            : find_default_mpk(root);
    if (mpk_path.empty() || !fs::exists(mpk_path)) {
      return sima_tutorial::skip("missing YOLOv8 MPK (pass --mpk)");
    }

    std::string img_arg;
    fs::path image_path = sima_tutorial::get_arg(argc, argv, "--image", img_arg)
                              ? fs::path(img_arg)
                              : find_default_image();
    if (!fs::exists(image_path)) {
      return sima_tutorial::skip("missing image (pass --image)");
    }

    if (!simaai::neat::element_exists("simaaiprocesscvu") ||
        !simaai::neat::element_exists("simaaiprocessmla") ||
        !simaai::neat::element_exists("simaaiboxdecode")) {
      return sima_tutorial::skip("missing SimaAI plugins (simaaiprocesscvu/mla/boxdecode)");
    }

    cv::Mat img_bgr = cv::imread(image_path.string(), cv::IMREAD_COLOR);
    if (img_bgr.empty()) {
      return sima_tutorial::skip("failed to load image");
    }

    simaai::neat::Model::Options model_opt;
    model_opt.input_max_width = img_bgr.cols;
    model_opt.input_max_height = img_bgr.rows;
    model_opt.input_max_depth = img_bgr.channels();
    simaai::neat::Model model(mpk_path.string(), model_opt);

    const bool strict = (std::getenv("SIMA_RUN_TUTORIALS_FULL") != nullptr);
    try {
      auto pre = simaai::neat::stages::Preproc(img_bgr, model);
      auto infer = simaai::neat::stages::Infer(pre, model);

      simaai::neat::stages::BoxDecodeOptions opt;
      opt.decode_type = "yolov8";
      opt.original_width = img_bgr.cols;
      opt.original_height = img_bgr.rows;
      opt.detection_threshold = 0.52;
      opt.nms_iou_threshold = 0.5;
      opt.top_k = 100;

      auto boxes = simaai::neat::stages::BoxDecode(infer, model, opt);
      std::cout << "Boxes decoded: " << boxes.boxes.size() << "\n";
    } catch (const std::exception& e) {
      if (!strict) {
        return sima_tutorial::skip(std::string("runtime unavailable: ") + e.what());
      }
      throw;
    }

    std::cout << "[OK] tutorial_0012 complete\n";
    return 0;
  } catch (const std::exception& e) {
    std::cerr << "[FAIL] " << e.what() << "\n";
    return 1;
  }
}
