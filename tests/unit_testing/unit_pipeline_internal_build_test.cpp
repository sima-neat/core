#include "pipeline/internal/PipelineBuild.h"
#include "test_main.h"
#include "test_utils.h"

RUN_TEST("unit_pipeline_internal_build_test", ([] {
           using namespace simaai::neat;
           using namespace simaai::neat::pipeline_internal;

           NameTransform t;
           t.prefix = "ns_";
           t.suffix = "_v1";

           require(resolve_buffer_name("decoder", t) == "ns_decoder_v1",
                   "resolve_buffer_name should apply transform to single buffer");

           require(resolve_buffer_name("decoder, parser", t) == "ns_decoder_v1,ns_parser_v1",
                   "resolve_buffer_name should transform comma-separated buffer names");

           require(resolve_buffer_name("", t) == "ns_decoder_v1",
                   "resolve_buffer_name should default empty buffer name to decoder");

           const auto expected = resolve_expected_buffer_names("decoder, parser", t);
           require(expected.size() == 4,
                   "resolve_expected_buffer_names should include transformed and legacy aliases");
           require(expected[0] == "ns_decoder_v1", "expected transformed decoder name missing");
           require(expected[1] == "decoder", "expected legacy decoder alias missing");
           require(expected[2] == "ns_parser_v1", "expected transformed parser name missing");
           require(expected[3] == "parser", "expected legacy parser alias missing");

           GraphOptions opt;
           opt.element_name_prefix = "aa_";
           opt.element_name_suffix = "_zz";
           PipelineBuildContext ctx(opt);

           require(ctx.resolve_buffer_name("sink") == "aa_sink_zz",
                   "PipelineBuildContext::resolve_buffer_name mismatch");

           const auto expected2 = ctx.resolve_expected_buffer_names("sink,bridge");
           require(expected2.size() == 4, "PipelineBuildContext::resolve_expected_buffer_names "
                                          "should include transformed and base aliases");
           require(expected2[0] == "aa_sink_zz", "PipelineBuildContext expected[0] mismatch");
           require(expected2[1] == "sink",
                   "PipelineBuildContext expected[1] legacy alias mismatch");
           require(expected2[2] == "aa_bridge_zz", "PipelineBuildContext expected[2] mismatch");
           require(expected2[3] == "bridge",
                   "PipelineBuildContext expected[3] legacy alias mismatch");
         }));
