// Decompose model execution into stages: Preproc -> Infer -> BoxDecode.
//
// Usage:
//   tutorial_v2_006_postproc_boxdecode --mpk /path/to/yolo_v8s.tar.gz --image /path/to.jpg

#include "neat.h"

#include "pipeline/StageRun.h"

#include <opencv2/imgcodecs.hpp>

#include <iostream>
#include <stdexcept>
#include <string>

namespace {

bool get_arg(int argc, char** argv, const std::string& key, std::string& out) {
  for (int i = 1; i + 1 < argc; ++i) {
    if (key == argv[i]) {
      out = argv[i + 1];
      return true;
    }
  }
  return false;
}

} // namespace

int main(int argc, char** argv) {
  try {
    std::string mpk, image;
    if (!get_arg(argc, argv, "--mpk", mpk) || !get_arg(argc, argv, "--image", image)) {
      std::cerr << "Usage: tutorial_v2_006_postproc_boxdecode --mpk <path> --image <path>\n";
      return 1;
    }

    cv::Mat bgr = cv::imread(image, cv::IMREAD_COLOR);
    if (bgr.empty())
      throw std::runtime_error("failed to load image: " + image);

    simaai::neat::Model::Options opt;
    opt.format = "BGR";
    opt.input_max_width = bgr.cols;
    opt.input_max_height = bgr.rows;
    opt.input_max_depth = bgr.channels();

    simaai::neat::Model model(mpk, opt);

    // CORE LOGIC
    // Stage-by-stage: each stages::* call runs one piece of the model pipeline.
    simaai::neat::Tensor pre = simaai::neat::stages::Preproc(bgr, model);
    simaai::neat::Tensor infer = simaai::neat::stages::Infer(pre, model);

    simaai::neat::stages::BoxDecodeOptions box;
    box.decode_type = "yolov8";
    box.original_width = bgr.cols;
    box.original_height = bgr.rows;
    box.detection_threshold = 0.52;
    box.nms_iou_threshold = 0.5;
    box.top_k = 100;

    // BoxDecode parses the "BBOX" tensor into {x1, y1, x2, y2, score, class_id}
    // entries clamped to original_width x original_height source pixels.
    simaai::neat::stages::BoxDecodeResult decoded =
        simaai::neat::stages::BoxDecode(infer, model, box);
    // END CORE LOGIC

    std::cout << "boxes=" << decoded.boxes.size() << "\n";
    std::cout << "[OK] 006_postproc_boxdecode\n";
    return 0;
  } catch (const std::exception& e) {
    std::cerr << "[FAIL] " << e.what() << "\n";
    return 1;
  }
}
