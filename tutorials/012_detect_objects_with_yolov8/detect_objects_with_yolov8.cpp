// YOLO quickstart: compose a full detection Session via node groups.
//
// Usage:
//   tutorial_v2_012_yolo_quickstart --mpk /path/to/yolo_v8s.tar.gz --image /path/to.jpg

#include "neat.h"

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
      std::cerr << "Usage: tutorial_v2_012_yolo_quickstart --mpk <path> --image <path>\n";
      return 1;
    }

    cv::Mat bgr = cv::imread(image, cv::IMREAD_COLOR);
    if (bgr.empty())
      throw std::runtime_error("failed to load image: " + image);

    simaai::neat::Model::Options mopt;
    mopt.input_max_width = bgr.cols;
    mopt.input_max_height = bgr.rows;
    mopt.input_max_depth = bgr.channels();

    simaai::neat::Model model(mpk, mopt);

    // CORE LOGIC
    // Explicit detection pipeline: Input -> Preprocess -> MLA -> SimaBoxDecode -> Output.
    simaai::neat::Session session;
    session.add(simaai::neat::nodes::Input());
    session.add(simaai::neat::nodes::groups::Preprocess(model));
    session.add(simaai::neat::nodes::groups::MLA(model));
    session.add(simaai::neat::nodes::SimaBoxDecode(model, "yolov8", bgr.cols, bgr.rows,
                                                   /*score_threshold=*/0.52f,
                                                   /*nms_iou_threshold=*/0.5f,
                                                   /*top_k=*/100));
    session.add(simaai::neat::nodes::Output());

    auto run = session.build(bgr, simaai::neat::RunMode::Sync);
    auto out = run.push_and_pull(bgr, /*timeout_ms=*/2000);
    // END CORE LOGIC

    std::cout << "output_kind=" << static_cast<int>(out.kind) << "\n";
    std::cout << "fields=" << out.fields.size() << "\n";
    std::cout << "[OK] 012_detect_objects_with_yolov8\n";
    return 0;
  } catch (const std::exception& e) {
    std::cerr << "[FAIL] " << e.what() << "\n";
    return 1;
  }
}
