#include "pipeline/TensorAdapters.h"

#include "test_utils.h"

#if defined(SIMA_WITH_OPENCV)
#include <opencv2/core.hpp>
#endif

#include <iostream>

int main() {
  try {
#if !defined(SIMA_WITH_OPENCV)
    return fail_test("unit_tensortensor_cvmat_test requires SIMA_WITH_OPENCV");
#else
    cv::Mat img(4, 4, CV_8UC3);
    img.setTo(cv::Scalar(1, 2, 3));

    cv::Rect roi(1, 1, 2, 2);
    cv::Mat view = img(roi);

    simaai::neat::Tensor t = simaai::neat::from_cv_mat(view);
    require(t.shape.size() == 3, "shape rank mismatch");
    require(t.shape[0] == 2 && t.shape[1] == 2 && t.shape[2] == 3, "shape mismatch");
    require(!t.is_contiguous(), "ROI view should be non-contiguous");

    auto v = t.map_cv_mat_view(simaai::neat::ImageSpec::PixelFormat::BGR);
    require(v.has_value(), "map_cv_mat_view failed");
    require(v->mat.data == view.data, "expected zero-copy ROI view");
    const cv::Vec3b px = v->mat.at<cv::Vec3b>(0, 0);
    require(px[0] == 1 && px[1] == 2 && px[2] == 3, "view pixel mismatch");
    cv::Mat clone = v->mat.clone();
    require(clone.rows == view.rows && clone.cols == view.cols, "clone shape mismatch");

    std::cout << "[OK] unit_tensortensor_cvmat_test passed\n";
    return 0;
#endif
  } catch (const std::exception& e) {
    std::cerr << "[FAIL] " << e.what() << "\n";
    return 1;
  }
}
