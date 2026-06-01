#include "nodes/common/Output.h"
#include "nodes/io/Input.h"
#include "pipeline/Graph.h"
#include "runtime_test_utils.h"
#include "test_main.h"
#include "test_utils.h"

#include <opencv2/core.hpp>

RUN_TEST("graph_naming_pipeline_integration_test", ([] {
           using namespace simaai::neat;

           GraphOptions sopt;
           sopt.element_name_prefix = "named_";
           sopt.element_name_suffix = "_rt";

           Graph graph(sopt);
           InputOptions src_opt;
           src_opt.payload_type = simaai::neat::PayloadType::Image;
           src_opt.format = simaai::neat::FormatTag::RGB;
           src_opt.use_simaai_pool = false;
           src_opt.max_width = 96;
           src_opt.max_height = 96;
           src_opt.max_depth = 3;
           graph.add(nodes::Input(src_opt));
           graph.add(nodes::Output(OutputOptions::EveryFrame(64)));

           const Tensor seed = make_color_tensor(64, 48, ImageSpec::PixelFormat::RGB, 0x43);

           RunOptions run_opt;
           run_opt.queue_depth = 16;

           Run run;
           try {
             run = graph.build(TensorList{seed}, RunMode::Async, run_opt);
           } catch (const std::exception& e) {
             if (sima_test::likely_runtime_missing(e.what())) {
               throw std::runtime_error("Skipping naming integration due runtime limitations: " +
                                        std::string(e.what()));
             }
             throw;
           }

           TensorList outs = run.run(TensorList{seed}, 1000);
           require(outs.size() == 1, "naming integration: expected one output tensor");

           const std::string backend = graph.last_pipeline();
           require_contains(backend, "named_mysrc_rt",
                            "naming integration: transformed mysrc missing");
           require_contains(backend, "named_mysink_rt",
                            "naming integration: transformed mysink missing");

           const std::string report = run.report();
           require_contains(report, "named_mysrc_rt",
                            "naming integration: report missing transformed mysrc");
           require_contains(report, "named_mysink_rt",
                            "naming integration: report missing transformed mysink");

           ValidateOptions vopt;
           vopt.parse_launch = true;
           vopt.enforce_names = true;
           cv::Mat frame(48, 64, CV_8UC3, cv::Scalar(11, 22, 33));
           const GraphReport vrep = graph.validate(vopt, frame);
           require_contains(vrep.pipeline_string, "named_mysrc_rt",
                            "naming integration: validate pipeline string missing mysrc");
           require_contains(vrep.pipeline_string, "named_mysink_rt",
                            "naming integration: validate pipeline string missing mysink");

           run.stop();
         }));
