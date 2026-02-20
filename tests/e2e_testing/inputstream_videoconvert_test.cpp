#include "inputstream_test_utils.h"
#include "test_main.h"

#include <opencv2/core.hpp>

RUN_TEST("inputstream_videoconvert_test", [] {
  sima_test::InputstreamRenegotiateSpec spec;
  spec.format = "BGR";
  spec.gst_element = "videoconvert";
  spec.first_w = 32;
  spec.first_h = 32;
  spec.second_w = 64;
  spec.second_h = 48;
  spec.stability_frames = 1;
  spec.check_output = true;
  spec.max_input_bytes = static_cast<size_t>(spec.second_w * spec.second_h * 3);

  sima_test::run_inputstream_renegotiate_test(spec, [](int w, int h) {
    cv::Mat img(h, w, CV_8UC3, cv::Scalar(10, 20, 30));
    if (!img.isContinuous())
      img = img.clone();
    return img;
  });
});
