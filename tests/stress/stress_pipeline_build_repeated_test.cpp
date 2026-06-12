#include "nodes/common/Output.h"
#include "nodes/io/Input.h"
#include "pipeline/Graph.h"
#include "runtime_test_utils.h"
#include "test_main.h"
#include "test_utils.h"

#include <algorithm>
#include <cstdlib>
#include <string>

namespace {

int env_int(const char* key, int fallback) {
  const char* raw = std::getenv(key);
  if (!raw || !*raw) {
    return fallback;
  }
  return std::atoi(raw);
}

int clamp_iters(int value) {
  return std::max(10, std::min(value, 600));
}

} // namespace

RUN_TEST("stress_pipeline_build_repeated_test", ([] {
           using namespace simaai::neat;

           const int iters = clamp_iters(env_int("SIMA_STRESS_ITERS", 150));
           const int cycles = std::max(3, std::min(10, (iters / 20) + 1));
           const Tensor seed = make_color_tensor(64, 48, ImageSpec::PixelFormat::RGB, 0x6b);

           int completed = 0;
           for (int i = 0; i < cycles; ++i) {
             GraphOptions sopt;
             sopt.element_name_prefix = "rep_" + std::to_string(i) + "_";
             sopt.element_name_suffix = "_x";

             Graph graph(sopt);
             InputOptions src_opt;
             src_opt.payload_type = simaai::neat::PayloadType::Image;
             src_opt.format = simaai::neat::FormatTag::RGB;
             src_opt.use_simaai_pool = false;
             src_opt.max_width = 96;
             src_opt.max_height = 96;
             src_opt.max_depth = 3;
             graph.add(nodes::Input(src_opt));
             graph.add(nodes::Output(OutputOptions::EveryFrame(32)));

             const std::string plain_a = graph.describe_backend(false);
             const std::string plain_b = graph.describe_backend(false);
             require(plain_a == plain_b,
                     "stress_pipeline_build_repeated: non-deterministic pipeline string");
             require_contains(plain_a, "rep_" + std::to_string(i) + "_mysrc_x",
                              "stress_pipeline_build_repeated: transformed source name missing");

             const std::string bounded = graph.describe_backend(true);
             require_contains(bounded, "identity name=rep_" + std::to_string(i) + "_sima_b0_x",
                              "stress_pipeline_build_repeated: boundary insertion missing");

             RunOptions run_opt;
             run_opt.queue_depth = 12;

             Run run;
             try {
               run = graph.build(TensorList{seed}, run_opt);
             } catch (const std::exception& e) {
               if (sima_test::likely_runtime_missing(e.what())) {
                 skip_long_test_exception(
                     "Skipping pipeline build repeated stress due runtime limitations: " +
                     std::string(e.what()));
               }
               throw;
             }

             TensorList outs = run.run(TensorList{seed}, 1000);
             require(outs.size() == 1,
                     "stress_pipeline_build_repeated: expected one output tensor");
             run.stop();
             ++completed;
           }

           require(completed == cycles,
                   "stress_pipeline_build_repeated: did not complete all iterations");
         }));
