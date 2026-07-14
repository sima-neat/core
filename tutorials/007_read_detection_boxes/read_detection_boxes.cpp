// Decompose model execution into stages: Preproc -> Infer -> BoxDecode.
//
// Usage:
//   tutorial_007_read_detection_boxes --model /path/to/yolo_v8s.tar.gz --image /path/to.jpg

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
    std::string model_path, image;
    if (!get_arg(argc, argv, "--model", model_path) || !get_arg(argc, argv, "--image", image)) {
      std::cerr << "Usage: tutorial_007_read_detection_boxes --model <path> --image <path>\n";
      return 1;
    }

    cv::Mat bgr = cv::imread(image, cv::IMREAD_COLOR);
    if (bgr.empty())
      throw std::runtime_error("failed to load image: " + image);

    // STEP configure-decode
    simaai::neat::Model::Options opt;
    opt.preprocess.kind = simaai::neat::InputKind::Image;
    opt.preprocess.enable = simaai::neat::AutoFlag::On;
    opt.preprocess.color_convert.input_format = simaai::neat::PreprocessColorFormat::BGR;
    opt.preprocess.input_max_width = bgr.cols;
    opt.preprocess.input_max_height = bgr.rows;
    opt.preprocess.input_max_depth = bgr.channels();
    opt.preprocess.normalize.enable = simaai::neat::AutoFlag::On;
    opt.decode_type = simaai::neat::BoxDecodeType::YoloV8;
    // END STEP

    // STEP load-model
    simaai::neat::Model model(model_path, opt);
    // END STEP

    // CORE LOGIC
    // Stage-by-stage: each stages::* call runs one piece of the model pipeline.
    // STEP run-decode
    simaai::neat::TensorList pre = simaai::neat::stages::Preproc(std::vector<cv::Mat>{bgr}, model);
    simaai::neat::Sample infer_samples = simaai::neat::stages::Infer(
        simaai::neat::Sample{simaai::neat::sample_from_tensors(pre)}, model);
    if (infer_samples.empty())
      throw std::runtime_error("infer stage returned no samples");
    simaai::neat::Sample infer = infer_samples.front();

    simaai::neat::stages::BoxDecodeOptions box(simaai::neat::BoxDecodeType::YoloV8);
    (void)box.decode_type;
    (void)bgr.cols;
    (void)bgr.rows;
    box.detection_threshold = 0.55;
    box.nms_iou_threshold = 0.5;
    box.top_k = 100;
    // END STEP

    // STEP read-boxes
    // BoxDecode parses the "BBOX" tensor into {x1, y1, x2, y2, score, class_id}
    // entries clamped to original_width x original_height source pixels.
    simaai::neat::BoxDecodeResultList decoded_results =
        simaai::neat::stages::BoxDecodeResults(simaai::neat::Sample{infer}, model, box);
    if (decoded_results.empty())
      throw std::runtime_error("boxdecode result parser returned no results");
    const simaai::neat::BoxDecodeResult& decoded = decoded_results.front();
    // END STEP
    // END CORE LOGIC

    std::cout << "boxes=" << decoded.boxes.size() << "\n";
    for (std::size_t i = 0; i < decoded.boxes.size(); ++i) {
      const auto& detection = decoded.boxes[i];
      std::cout << "box[" << i << "] class=" << detection.class_id
                << " score=" << detection.score << " xyxy=[" << detection.x1 << ","
                << detection.y1 << "," << detection.x2 << "," << detection.y2 << "]\n";
    }
    if (decoded.boxes.empty())
      throw std::runtime_error("image produced no detections above threshold");
    std::cout << "[OK] 007_read_detection_boxes\n";
    return 0;
  } catch (const std::exception& e) {
    std::cerr << "[FAIL] " << e.what() << "\n";
    return 1;
  }
}
