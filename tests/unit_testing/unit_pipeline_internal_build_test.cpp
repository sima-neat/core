#include "pipeline/internal/PipelineBuild.h"
#include "pipeline/Graph.h"
#include "nodes/common/Output.h"
#include "nodes/io/Input.h"
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

           // Preserve the cold renderer wiring contract after Session/JSON hook retirement.
           Graph graph(opt);
           InputOptions input;
           input.payload_type = PayloadType::Image;
           input.format = FormatTag::RGB;
           input.memory_policy = InputMemoryPolicy::SystemMemory;
           input.max_width = 96;
           input.max_height = 96;
           input.max_depth = 3;
           graph.add(nodes::Input(input));
           graph.add(nodes::Output(OutputOptions::Latest()));
           const std::string plain = graph.describe_backend(false);
           require(plain == graph.describe_backend(false),
                   "cold graph rendering should be deterministic");
           require_contains(plain, "aa_mysrc_zz", "transformed source name missing");
           require_contains(plain, "aa_mysink_zz", "transformed sink name missing");
           require(plain.find("aa_sima_b0_zz") == std::string::npos,
                   "boundary-disabled rendering should omit the boundary identity");
           const std::string bounded = graph.describe_backend(true);
           require_contains(bounded, "identity name=aa_sima_b0_zz",
                            "boundary-enabled rendering should preserve the name transform");
           require(plain == graph.describe_backend(false),
                   "boundary rendering must not mutate subsequent plain rendering");
         }));
