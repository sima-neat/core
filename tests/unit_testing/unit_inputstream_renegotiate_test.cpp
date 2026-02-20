#include "inputstream_test_utils.h"
#include "test_main.h"

#include <opencv2/core/mat.hpp>

RUN_TEST("unit_inputstream_renegotiate_test", [] {
  sima_test::InputstreamRenegotiateSpec spec;
  spec.format = "BGR";
  spec.first_w = 16;
  spec.first_h = 16;
  spec.second_w = 32;
  spec.second_h = 32;
  spec.stability_frames = 2;
  spec.max_input_bytes = 1024 * 1024;

  sima_test::run_inputstream_renegotiate_test(spec, [](int w, int h) {
    cv::Mat img(h, w, CV_8UC3, cv::Scalar(0, 0, 0));
    if (!img.isContinuous())
      img = img.clone();
    return img;
  });
});
