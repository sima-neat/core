#include "e2e_pipelines/segmentation/segmentation_example_test_utils.h"
#include "test_utils.h"

#include <iostream>

namespace fs = std::filesystem;

int main(int argc, char** argv) {
  try {
    const fs::path root = sima_example_test::resolve_root(argc, argv);
    const std::string model_tar =
        sima_example_test::resolve_model_tar_or_throw("yolo_v8n_seg", root);
    const sima_example_test::ExampleIoPaths io =
        sima_example_test::prepare_single_input(root, "yolov8n_seg_detess_simple_example_test");
    const fs::path example_bin =
        sima_example_test::resolve_example_binary(root, argv[0], "yolov8n_seg_detess_simple");

    sima_example_test::run_example_or_throw(example_bin, model_tar, io.input_dir, io.output_dir);

    const cv::Mat input_img = sima_example_test::require_image(io.input_image, "test input image");
    const fs::path out_path = io.output_dir / (io.stem + "_overlay.jpg");
    const cv::Mat out_img = sima_example_test::require_image(out_path, "yolov8n-seg overlay output");
    require(out_img.cols == input_img.cols && out_img.rows == input_img.rows,
            "expected overlay output to preserve original image size");

    std::cout << "YOLOV8N_SEG_DETESS_SIMPLE_EXAMPLE ok=1 output=" << out_path
              << " size=" << out_img.cols << "x" << out_img.rows << "\n";
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
