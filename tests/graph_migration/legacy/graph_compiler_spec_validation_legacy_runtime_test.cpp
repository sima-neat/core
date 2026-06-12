#include "graph/Compiler.h"
#include "graph_test_utils.h"
#include "test_main.h"

namespace {

void expect_input_spec_mismatch(const simaai::neat::OutputSpec& a,
                                const simaai::neat::OutputSpec& b, const std::string& field_name) {
  using simaai::neat::graph::Compiler;
  using simaai::neat::graph::Graph;

  Compiler compiler;
  Graph g;
  const auto src_a = g.add(sima_test::make_stage_source("src_a", a));
  const auto src_b = g.add(sima_test::make_stage_source("src_b", b));
  const auto merge = g.add(sima_test::make_stage_passthrough("merge", 0));

  g.connect(src_a, merge, "out", "in");
  g.connect(src_b, merge, "out", "in");

  require(sima_test::throws_with([&]() { (void)compiler.compile(g); }, field_name),
          "Compiler should reject mixed input spec mismatch for field: " + field_name);
}

} // namespace

RUN_TEST("graph_migration_legacy_graph_compiler_spec_validation_test", ([] {
           using simaai::neat::OutputSpec;

           // media_type mismatch
           expect_input_spec_mismatch(
               OutputSpec{.media_type = "video/x-raw", .format = "RGB", .width = 32, .height = 24},
               OutputSpec{.media_type = "application/vnd.simaai.tensor",
                          .format = "RGB",
                          .width = 32,
                          .height = 24},
               "media_type");

           // format mismatch
           expect_input_spec_mismatch(
               OutputSpec{.media_type = "video/x-raw", .format = "RGB", .width = 32, .height = 24},
               OutputSpec{.media_type = "video/x-raw", .format = "NV12", .width = 32, .height = 24},
               "format");

           // width mismatch
           expect_input_spec_mismatch(
               OutputSpec{.media_type = "video/x-raw", .format = "RGB", .width = 32, .height = 24},
               OutputSpec{.media_type = "video/x-raw", .format = "RGB", .width = 64, .height = 24},
               "width");

           // height mismatch
           expect_input_spec_mismatch(
               OutputSpec{.media_type = "video/x-raw", .format = "RGB", .width = 32, .height = 24},
               OutputSpec{.media_type = "video/x-raw", .format = "RGB", .width = 32, .height = 48},
               "height");

           // depth mismatch
           expect_input_spec_mismatch(OutputSpec{.media_type = "application/vnd.simaai.tensor",
                                                 .format = "FP32",
                                                 .width = 16,
                                                 .height = 16,
                                                 .depth = 4,
                                                 .dtype = "Float32"},
                                      OutputSpec{.media_type = "application/vnd.simaai.tensor",
                                                 .format = "FP32",
                                                 .width = 16,
                                                 .height = 16,
                                                 .depth = 8,
                                                 .dtype = "Float32"},
                                      "depth");

           // dtype mismatch
           expect_input_spec_mismatch(OutputSpec{.media_type = "application/vnd.simaai.tensor",
                                                 .format = "FP32",
                                                 .width = 16,
                                                 .height = 16,
                                                 .depth = 4,
                                                 .dtype = "Float32"},
                                      OutputSpec{.media_type = "application/vnd.simaai.tensor",
                                                 .format = "FP32",
                                                 .width = 16,
                                                 .height = 16,
                                                 .depth = 4,
                                                 .dtype = "UInt8"},
                                      "dtype");

           // layout mismatch
           expect_input_spec_mismatch(OutputSpec{.media_type = "application/vnd.simaai.tensor",
                                                 .format = "FP32",
                                                 .width = 16,
                                                 .height = 16,
                                                 .depth = 4,
                                                 .layout = "CHW"},
                                      OutputSpec{.media_type = "application/vnd.simaai.tensor",
                                                 .format = "FP32",
                                                 .width = 16,
                                                 .height = 16,
                                                 .depth = 4,
                                                 .layout = "HWC"},
                                      "layout");

           // memory mismatch
           expect_input_spec_mismatch(OutputSpec{.media_type = "video/x-raw",
                                                 .format = "RGB",
                                                 .width = 32,
                                                 .height = 24,
                                                 .memory = "SystemMemory"},
                                      OutputSpec{.media_type = "video/x-raw",
                                                 .format = "RGB",
                                                 .width = 32,
                                                 .height = 24,
                                                 .memory = "SimaAI"},
                                      "memory");
         }));
