#include "inputstream_test_utils.h"
#include "test_main.h"

#include <opencv2/core.hpp>

RUN_TEST("inputstream_videoscale_test", [] {
  sima_test::InputstreamRenegotiateSpec spec;
  spec.format = "RGB";
  spec.gst_element = "videoscale";
  spec.first_w = 40;
  spec.first_h = 24;
  spec.second_w = 56;
  spec.second_h = 32;
  spec.stability_frames = 1;
  spec.check_output = true;
  spec.max_input_bytes = static_cast<size_t>(spec.second_w * spec.second_h * 3);

  sima_test::run_inputstream_renegotiate_test(spec, [](int w, int h) {
    cv::Mat img(h, w, CV_8UC3, cv::Scalar(12, 34, 56));
    if (!img.isContinuous())
      img = img.clone();
    return img;
  });
});
