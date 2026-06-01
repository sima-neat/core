#include "inputstream_test_utils.h"
#include "test_main.h"

#include <opencv2/core.hpp>

#include <vector>

RUN_TEST("caps_negotiation_matrix_regression_test", ([] {
           using sima_test::InputstreamFormatChangeSpec;
           using sima_test::InputstreamRenegotiateSpec;

           const std::vector<InputstreamRenegotiateSpec> rgb_matrix = {
               // Baseline RGB scenarios.
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
                   .stability_frames = 1,
                   .timeout_ms = 2000,
                   .max_input_bytes = 0,
                   .check_output = false,
               },
               // Former inputstream_videoscale_test wrapper scenario.
               InputstreamRenegotiateSpec{
                   .format = "RGB",
                   .gst_element = "videoscale",
                   .first_w = 40,
                   .first_h = 24,
                   .second_w = 56,
                   .second_h = 32,
                   .stability_frames = 1,
                   .timeout_ms = 2000,
                   .max_input_bytes = static_cast<size_t>(56 * 32 * 3),
                   .check_output = true,
               },
           };

           for (const auto& spec : rgb_matrix) {
             sima_test::run_inputstream_renegotiate_test(spec, [](int w, int h) {
               return make_color_tensor(w, h, simaai::neat::ImageSpec::PixelFormat::RGB);
             });
           }

           const std::vector<InputstreamRenegotiateSpec> bgr_matrix = {
               // Former unit_inputstream_renegotiate_test wrapper scenario.
               InputstreamRenegotiateSpec{
                   .format = "BGR",
                   .gst_element = "",
                   .first_w = 16,
                   .first_h = 16,
                   .second_w = 32,
                   .second_h = 32,
                   .stability_frames = 1,
                   .timeout_ms = 2000,
                   .max_input_bytes = 1024 * 1024,
                   .check_output = false,
               },
               // Former inputstream_videoconvert_test wrapper scenario.
               InputstreamRenegotiateSpec{
                   .format = "BGR",
                   .gst_element = "videoconvert",
                   .first_w = 32,
                   .first_h = 32,
                   .second_w = 64,
                   .second_h = 48,
                   .stability_frames = 1,
                   .timeout_ms = 2000,
                   .max_input_bytes = static_cast<size_t>(64 * 48 * 3),
                   .check_output = true,
               },
           };

           for (const auto& spec : bgr_matrix) {
             sima_test::run_inputstream_renegotiate_test(spec, [](int w, int h) {
               cv::Mat img(h, w, CV_8UC3, cv::Scalar(10, 20, 30));
               if (!img.isContinuous()) {
                 img = img.clone();
               }
               return img;
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
                   .stability_frames = 1,
                   .timeout_ms = 2000,
                   .max_input_bytes = 0,
                   .check_output = false,
               },
               // Former unit_inputstream_nv12_renegotiate_test wrapper scenario.
               InputstreamRenegotiateSpec{
                   .format = "NV12",
                   .gst_element = "",
                   .first_w = 16,
                   .first_h = 16,
                   .second_w = 32,
                   .second_h = 32,
                   .stability_frames = 1,
                   .timeout_ms = 2000,
                   .max_input_bytes = 1024 * 1024,
                   .check_output = false,
               },
               // Former inputstream_nv12_dynamic renegotiate wrapper scenario.
               InputstreamRenegotiateSpec{
                   .format = "NV12",
                   .gst_element = "",
                   .first_w = 1280,
                   .first_h = 720,
                   .second_w = 1920,
                   .second_h = 1080,
                   .stability_frames = 1,
                   .timeout_ms = 2000,
                   .max_input_bytes = static_cast<size_t>(1920 * 1080 * 3 / 2),
                   .check_output = false,
               },
           };

           for (const auto& spec : nv12_matrix) {
             sima_test::run_inputstream_renegotiate_test(
                 spec, [](int w, int h) { return make_nv12_tensor(w, h); });
           }

           const std::vector<InputstreamFormatChangeSpec> format_change_matrix = {
               // Former unit_inputstream_format_change_test wrapper scenario.
               InputstreamFormatChangeSpec{
                   .width = 16,
                   .height = 16,
                   .max_input_bytes = 0,
                   .expect_throw = false,
                   .timeout_ms = 1000,
                   .reneg_timeout_ms = 1000,
                   .rgb_fill = 0x11,
               },
               // Baseline matrix scenario.
               InputstreamFormatChangeSpec{
                   .width = 64,
                   .height = 48,
                   .max_input_bytes = 0,
                   .expect_throw = false,
                   .timeout_ms = 1000,
                   .reneg_timeout_ms = 1000,
                   .rgb_fill = 0x7f,
               },
               // Former inputstream_nv12_format_change_test wrapper scenario.
               InputstreamFormatChangeSpec{
                   .width = 1280,
                   .height = 720,
                   .max_input_bytes = static_cast<size_t>(1280 * 720 * 3),
                   .expect_throw = false,
                   .timeout_ms = 1000,
                   .reneg_timeout_ms = 1000,
                   .rgb_fill = 0x7f,
               },
           };

           for (const auto& spec : format_change_matrix) {
             sima_test::run_inputstream_format_change_test(spec);
           }
         }));
