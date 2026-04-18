// Model::Options chapter: configure input/preproc/boxdecode via Options, inspect specs.
//
// Usage:
//   tutorial_v2_004_model_options_chapter --mpk /path/to/yolo_v8s.tar.gz

#include "neat.h"

#include <opencv2/core.hpp>

#include <array>
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

void print_spec(const char* label, const simaai::neat::TensorConstraint& spec) {
  std::cout << label << ": rank=" << spec.rank << " dtypes=" << spec.dtypes.size()
            << " shape_dims=" << spec.shape.size() << "\n";
}

} // namespace

int main(int argc, char** argv) {
  try {
    std::string mpk;
    if (!get_arg(argc, argv, "--mpk", mpk)) {
      std::cerr << "Usage: tutorial_v2_004_model_options_chapter --mpk <path>\n";
      return 1;
    }

    // CORE LOGIC
    // Model::Options groups input caps, preproc, and box-decode into one struct.
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

    simaai::neat::Model model(mpk, opt);
    // END CORE LOGIC

    print_spec("input_spec", model.input_spec());
    print_spec("output_spec", model.output_spec());
    std::cout << "metadata_keys=" << model.metadata().size() << "\n";

    cv::Mat bgr(224, 224, CV_8UC3, cv::Scalar(10, 20, 30));
    if (!bgr.isContinuous())
      bgr = bgr.clone();
    auto out = model.run(bgr, /*timeout_ms=*/2000);
    std::cout << "output_kind=" << static_cast<int>(out.kind) << "\n";
    std::cout << "[OK] 004_model_options_chapter\n";
    return 0;
  } catch (const std::exception& e) {
    std::cerr << "[FAIL] " << e.what() << "\n";
    return 1;
  }
}
