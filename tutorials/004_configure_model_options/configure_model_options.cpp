// Model::Options chapter: configure input/preproc/boxdecode via Options, inspect specs.
//
// Usage:
//   tutorial_004_configure_model_options --model /path/to/yolo_v8s.tar.gz

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
    std::string model_path;
    if (!get_arg(argc, argv, "--model", model_path)) {
      std::cerr << "Usage: tutorial_004_configure_model_options --model <path>\n";
      return 1;
    }

    // Model::Options groups input caps, preproc, and box-decode into one struct.
    simaai::neat::Model::Options opt;
    // STEP set-input-preproc
    opt.preprocess.kind = simaai::neat::InputKind::Image;
    opt.preprocess.color_convert.input_format = simaai::neat::PreprocessColorFormat::BGR;
    opt.preprocess.input_max_width = 640;
    opt.preprocess.input_max_height = 640;
    opt.preprocess.input_max_depth = 3;
    opt.preprocess.normalize.enable = simaai::neat::AutoFlag::On;
    opt.preprocess.normalize.mean = std::array<float, 3>{0.485f, 0.456f, 0.406f};
    opt.preprocess.normalize.stddev = std::array<float, 3>{0.229f, 0.224f, 0.225f};
    // END STEP
    // STEP set-postproc
    opt.decode_type = simaai::neat::BoxDecodeType::YoloV8;
    opt.score_threshold = 0.55f;
    opt.nms_iou_threshold = 0.45f;
    opt.top_k = 100;
    opt.boxdecode_original_width = 640;
    opt.boxdecode_original_height = 640;
    opt.name_suffix = "_chapter";
    // END STEP

    // CORE LOGIC
    // STEP load-and-inspect
    simaai::neat::Model model(model_path, opt);
    print_spec("input_spec", model.input_spec());
    print_spec("output_spec", model.output_spec());
    std::cout << "metadata_keys=" << model.metadata().size() << "\n";
    // END STEP
    // END CORE LOGIC

    // STEP run-inference
    cv::Mat bgr(640, 640, CV_8UC3, cv::Scalar(10, 20, 30));
    if (!bgr.isContinuous())
      bgr = bgr.clone();
    auto out = model.run(std::vector<cv::Mat>{bgr}, /*timeout_ms=*/2000);
    std::cout << "outputs=" << out.size() << "\n";
    // END STEP
    std::cout << "[OK] 004_configure_model_options\n";
    return 0;
  } catch (const std::exception& e) {
    std::cerr << "[FAIL] " << e.what() << "\n";
    return 1;
  }
}
