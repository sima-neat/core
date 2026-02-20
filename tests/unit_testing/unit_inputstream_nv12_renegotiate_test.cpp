#include "inputstream_test_utils.h"
#include "test_main.h"

RUN_TEST("unit_inputstream_nv12_renegotiate_test", [] {
  sima_test::InputstreamRenegotiateSpec spec;
  spec.format = "NV12";
  spec.first_w = 16;
  spec.first_h = 16;
  spec.second_w = 32;
  spec.second_h = 32;
  spec.stability_frames = 2;
  spec.max_input_bytes = 1024 * 1024;

  sima_test::run_inputstream_renegotiate_test(spec,
                                              [](int w, int h) { return make_nv12_tensor(w, h); });
});
