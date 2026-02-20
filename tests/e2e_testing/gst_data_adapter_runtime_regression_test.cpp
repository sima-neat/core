#include "nodes/common/Output.h"
#include "nodes/io/Input.h"
#include "pipeline/Session.h"
#include "runtime_test_utils.h"
#include "test_main.h"
#include "test_utils.h"

#include <opencv2/core.hpp>

RUN_TEST("gst_data_adapter_runtime_regression_test", ([] {
           using namespace simaai::neat;

           Session session;
           InputOptions src_opt;
           src_opt.media_type = "video/x-raw";
           src_opt.format = "RGB";
           src_opt.use_simaai_pool = false;
           src_opt.max_width = 96;
           src_opt.max_height = 96;
           src_opt.max_depth = 3;
           session.add(nodes::Input(src_opt));
           session.add(nodes::Output(OutputOptions::EveryFrame(64)));

           RunOptions run_opt;
           run_opt.queue_depth = 16;

           const Tensor seed_tensor = make_color_tensor(64, 48, ImageSpec::PixelFormat::RGB, 0x55);

           // Tensor path through runtime adapter.
           Run tensor_run;
           try {
             tensor_run = session.build(seed_tensor, RunMode::Async, run_opt);
           } catch (const std::exception& e) {
             if (sima_test::likely_runtime_missing(e.what())) {
               throw std::runtime_error(
                   "Skipping GstDataAdapter runtime regression due runtime limitations: " +
                   std::string(e.what()));
             }
             throw;
           }
           require(tensor_run.push(seed_tensor), "GstDataAdapter tensor path push failed");
           auto tensor_out = tensor_run.pull(1000);
           require(tensor_out.has_value(), "GstDataAdapter tensor path pull timed out");
           require(tensor_out->tensor.has_value(),
                   "GstDataAdapter tensor path output missing tensor");
           tensor_run.stop();

           // Raw video (cv::Mat) path through runtime adapter.
           cv::Mat rgb(48, 64, CV_8UC3, cv::Scalar(70, 60, 50));
           Run mat_run = session.build(rgb, RunMode::Async, run_opt);
           require(mat_run.push(rgb), "GstDataAdapter cv::Mat path push failed");
           auto mat_out = mat_run.pull(1000);
           require(mat_out.has_value(), "GstDataAdapter cv::Mat path pull timed out");
           require(mat_out->tensor.has_value(),
                   "GstDataAdapter cv::Mat path output missing tensor");
           mat_run.stop();

           // Error path: empty frame should fail deterministically.
           require(sima_test::throws_with(
                       [&]() {
                         cv::Mat empty;
                         (void)session.build(empty, RunMode::Async, run_opt);
                       },
                       "input frame is empty"),
                   "GstDataAdapter runtime empty-frame error text mismatch");
         }));
