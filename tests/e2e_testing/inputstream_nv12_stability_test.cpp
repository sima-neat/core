#include "inputstream_test_utils.h"
#include "test_main.h"

RUN_TEST("inputstream_nv12_stability_test", [] {
  sima_test::InputstreamRenegotiateSpec spec;
  spec.format = "NV12";
  spec.first_w = 1280;
  spec.first_h = 720;
  spec.second_w = 1920;
  spec.second_h = 1080;
  spec.stability_frames = 2;
  spec.max_input_bytes = static_cast<size_t>(1920 * 1080 * 3 / 2);

  sima_test::run_inputstream_renegotiate_test(spec,
                                              [](int w, int h) { return make_nv12_tensor(w, h); });
});
