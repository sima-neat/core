#include "inputstream_test_utils.h"
#include "test_main.h"

#include <vector>

RUN_TEST("caps_negotiation_matrix_regression_test", ([] {
           using sima_test::InputstreamFormatChangeSpec;
           using sima_test::InputstreamRenegotiateSpec;

           const std::vector<InputstreamRenegotiateSpec> rgb_matrix = {
               InputstreamRenegotiateSpec{
                   .format = "RGB",
                   .gst_element = "",
                   .first_w = 32,
                   .first_h = 24,
                   .second_w = 64,
                   .second_h = 48,
                   .stability_frames = 1,
                   .timeout_ms = 2000,
                   .max_input_bytes = 0,
                   .check_output = false,
               },
               InputstreamRenegotiateSpec{
                   .format = "RGB",
                   .gst_element = "",
                   .first_w = 40,
                   .first_h = 30,
                   .second_w = 56,
                   .second_h = 42,
                   .stability_frames = 2,
                   .timeout_ms = 2000,
                   .max_input_bytes = 0,
                   .check_output = false,
               },
           };

           for (const auto& spec : rgb_matrix) {
             sima_test::run_inputstream_renegotiate_test(spec, [](int w, int h) {
               return make_color_tensor(w, h, simaai::neat::ImageSpec::PixelFormat::RGB);
             });
           }

           const std::vector<InputstreamRenegotiateSpec> nv12_matrix = {
               InputstreamRenegotiateSpec{
                   .format = "NV12",
                   .gst_element = "",
                   .first_w = 48,
                   .first_h = 32,
                   .second_w = 80,
                   .second_h = 48,
                   .stability_frames = 1,
                   .timeout_ms = 2000,
                   .max_input_bytes = 0,
                   .check_output = false,
               },
               InputstreamRenegotiateSpec{
                   .format = "NV12",
                   .gst_element = "",
                   .first_w = 64,
                   .first_h = 48,
                   .second_w = 96,
                   .second_h = 64,
                   .stability_frames = 2,
                   .timeout_ms = 2000,
                   .max_input_bytes = 0,
                   .check_output = false,
               },
           };

           for (const auto& spec : nv12_matrix) {
             sima_test::run_inputstream_renegotiate_test(
                 spec, [](int w, int h) { return make_nv12_tensor(w, h); });
           }

           InputstreamFormatChangeSpec format_change;
           format_change.width = 64;
           format_change.height = 48;
           format_change.expect_throw = false;
           sima_test::run_inputstream_format_change_test(format_change);
         }));
