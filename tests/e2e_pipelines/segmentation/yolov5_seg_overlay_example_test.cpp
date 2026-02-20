#include "e2e_pipelines/segmentation/segmentation_example_test_utils.h"
#include "test_utils.h"

#include <iostream>

namespace fs = std::filesystem;

int main(int argc, char** argv) {
  try {
    const fs::path root = sima_example_test::resolve_root(argc, argv);
    const std::string model_tar =
        sima_example_test::resolve_model_tar_or_throw("yolov5n", root);
    const sima_example_test::ExampleIoPaths io =
        sima_example_test::prepare_single_input(root, "yolov5_seg_overlay_example_test");
    const fs::path example_bin =
        sima_example_test::resolve_example_binary(root, argv[0], "yolov5_seg_overlay");

    sima_example_test::run_example_or_throw(example_bin, model_tar, io.input_dir, io.output_dir);

    const fs::path mask_path = io.output_dir / (io.stem + "_mask_overlay.png");
    const fs::path bbox_path = io.output_dir / (io.stem + "_bbox_overlay.png");
    const cv::Mat mask_img = sima_example_test::require_image(mask_path, "yolov5 mask overlay");
    const cv::Mat bbox_img = sima_example_test::require_image(bbox_path, "yolov5 bbox overlay");
    require(mask_img.cols == 640 && mask_img.rows == 640, "expected 640x640 mask overlay output");
    require(bbox_img.cols == 640 && bbox_img.rows == 640, "expected 640x640 bbox overlay output");

    std::cout << "YOLOV5_SEG_OVERLAY_EXAMPLE ok=1 mask=" << mask_path << " bbox=" << bbox_path
              << " size=" << mask_img.cols << "x" << mask_img.rows << "\n";
    return 0;
  } catch (const SkipTest& e) {
    std::cout << "[SKIP] " << e.what() << "\n";
    return skip_long_test(e.what());
  } catch (const std::exception& e) {
    if (is_dispatcher_unavailable(e.what())) {
      return skip_long_test("dispatcher unavailable");
    }
    std::cerr << "[ERR] " << e.what() << "\n";
    return 1;
  }
}
