#include "asset_utils.h"
#include "nodes/common/Output.h"
#include "nodes/io/Input.h"
#include "nodes/io/StillImageInput.h"
#include "pipeline/Graph.h"
#include "runtime_test_utils.h"
#include "test_main.h"
#include "test_utils.h"

#include <opencv2/core.hpp>

#include <algorithm>
#include <chrono>
#include <cstdlib>
#include <filesystem>
#include <string>
#include <thread>

namespace {

int env_int(const char* key, int fallback) {
  const char* raw = std::getenv(key);
  if (!raw || !*raw) {
    return fallback;
  }
  return std::atoi(raw);
}

int clamp_iters(int value) {
  return std::max(10, std::min(value, 500));
}

} // namespace

RUN_TEST("stress_graph_validate_rtsp_churn_test", ([] {
           using namespace simaai::neat;
           namespace fs = std::filesystem;

           const int iters = clamp_iters(env_int("SIMA_STRESS_ITERS", 120));
           const int validate_cycles = std::max(2, std::min(8, (iters / 30) + 1));
           const int rtsp_cycles = std::max(1, std::min(5, (iters / 60) + 1));

           // Validate/report churn loop.
           for (int i = 0; i < validate_cycles; ++i) {
             Graph graph;
             InputOptions src_opt;
             src_opt.payload_type = simaai::neat::PayloadType::Image;
             src_opt.format = simaai::neat::FormatTag::RGB;
             src_opt.memory_policy = simaai::neat::InputMemoryPolicy::SystemMemory;
             src_opt.max_width = 96;
             src_opt.max_height = 96;
             src_opt.max_depth = 3;
             graph.add(nodes::Input(src_opt));
             graph.add(nodes::Output(OutputOptions::Latest()));

             ValidateOptions vopt;
             vopt.parse_launch = true;
             vopt.enforce_names = true;
             cv::Mat frame(48, 64, CV_8UC3, cv::Scalar(9, 19, 29));

             GraphReport rep;
             try {
               rep = graph.validate(vopt, frame);
             } catch (const std::exception& e) {
               if (sima_test::likely_runtime_missing(e.what())) {
                 skip_long_test_exception("Skipping validate/rtsp churn due runtime limitations: " +
                                          std::string(e.what()));
               }
               throw;
             }

             require(!rep.pipeline_string.empty(),
                     "stress_graph_validate_rtsp_churn: empty pipeline string in validate report");
             require(!rep.repro_note.empty(),
                     "stress_graph_validate_rtsp_churn: empty repro note in validate report");
           }

           const fs::path image = sima_test::test_image_fixture_path();
           if (!fs::exists(image)) {
             skip_long_test_exception("Skipping RTSP churn stress: test.jpg not found at " +
                                      image.string());
           }

           // RTSP start/stop churn loop.
           for (int i = 0; i < rtsp_cycles; ++i) {
             Graph rtsp_graph;
             rtsp_graph.add(nodes::StillImageInput(image.string(), 640, 480, 640, 480, 10));

             RtspServerOptions ropt;
             ropt.mount = "stress_" + std::to_string(i);
             ropt.port = sima_test::allocate_local_rtsp_port();

             RtspServerHandle handle;
             try {
               handle = rtsp_graph.run_rtsp(ropt);
             } catch (const std::exception& e) {
               if (sima_test::likely_runtime_missing(e.what())) {
                 skip_long_test_exception("Skipping RTSP churn due runtime limitations: " +
                                          std::string(e.what()));
               }
               throw;
             }

             require(handle.running(), "stress_graph_validate_rtsp_churn: RTSP handle not running");
             std::this_thread::sleep_for(std::chrono::milliseconds(60));
             handle.stop();
             handle.stop();
             require(!handle.running(),
                     "stress_graph_validate_rtsp_churn: RTSP handle should be stopped");
           }
         }));
