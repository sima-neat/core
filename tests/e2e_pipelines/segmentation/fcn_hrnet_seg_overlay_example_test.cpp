#include "e2e_pipelines/segmentation/segmentation_example_test_utils.h"
#include "test_utils.h"

#include <iostream>

namespace fs = std::filesystem;

int main(int argc, char** argv) {
  try {
    const fs::path root = sima_example_test::resolve_root(argc, argv);
    const std::string model_tar =
        sima_example_test::resolve_model_tar_or_throw("fcn_hrnet18", root);
    const sima_example_test::ExampleIoPaths io =
        sima_example_test::prepare_single_input(root, "fcn_hrnet_seg_overlay_example_test");
    const fs::path example_bin =
        sima_example_test::resolve_example_binary(root, argv[0], "fcn_hrnet_seg_overlay");

    sima_example_test::run_example_or_throw(example_bin, model_tar, io.input_dir, io.output_dir);

    const fs::path out_path = io.output_dir / (io.stem + "_overlay.png");
    const cv::Mat out_img = sima_example_test::require_image(out_path, "fcn-hrnet overlay output");
    require(out_img.cols == 512 && out_img.rows == 512, "expected 512x512 overlay output");

    std::cout << "FCN_HRNET_SEG_OVERLAY_EXAMPLE ok=1 output=" << out_path
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
