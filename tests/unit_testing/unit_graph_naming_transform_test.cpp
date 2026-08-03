#include "gst/GstInit.h"
#include "pipeline/ErrorCodes.h"
#include "pipeline/Graph.h"
#include "pipeline/NeatError.h"
#include "pipeline/internal/PipelineBuild.h"
#include "pipeline/graph/GraphDetail.h"
#include "test_main.h"
#include "test_utils.h"

#include <memory>
#include <string>
#include <utility>
#include <vector>

RUN_TEST("unit_graph_naming_transform_test", ([] {
           using namespace simaai::neat;

           NameTransform t;
           t.prefix = "pre_";
           t.suffix = "_suf";

           require(name_transform_enabled(t),
                   "name_transform_enabled should be true for non-empty transform");
           require(apply_name_transform(t, "decoder") == "pre_decoder_suf",
                   "apply_name_transform failed basic prefix/suffix transform");
           require(apply_name_transform(t, "pre_decoder_suf") == "pre_decoder_suf",
                   "apply_name_transform should be idempotent");

           // Graph naming keeps RTSP payloader names unchanged.
           require(apply_name_transform(t, "pay0") == "pay0",
                   "RTSP payloader name should not be transformed");

           // PipelineBuild transform does not special-case payN.
           require(apply_name_transform_once("pay0", t) == "pre_pay0_suf",
                   "apply_name_transform_once should transform payloader names");

           const std::vector<std::string> names = {"a", "pre_b_suf"};
           const auto transformed = apply_name_transform(t, names);
           require(transformed.size() == 2, "vector transform size mismatch");
           require(transformed[0] == "pre_a_suf", "vector transform first element mismatch");
           require(transformed[1] == "pre_b_suf",
                   "vector transform should preserve already transformed names");

           {
             const std::unordered_map<std::string, std::string> mapping = {
                 {"boxdecode", "pre_boxdecode_suf"}};
             const std::string fragment =
                 "neatobjectdecode name=boxdecode stage-id=boxdecode silent=true ! fakesink";
             const std::string rewritten = rewrite_fragment_names(fragment, mapping);
             require_contains(rewritten, "name=pre_boxdecode_suf",
                              "rewrite_fragment_names should rewrite boxdecode element name");
             require_contains(rewritten, "stage-id=pre_boxdecode_suf",
                              "rewrite_fragment_names should rewrite matching stage-id");
           }

           {
             const std::unordered_map<std::string, std::string> mapping = {
                 {"boxdecode", "pre_boxdecode_suf"}};
             const std::string fragment =
                 "neatobjectdecode name=boxdecode stage-id=logical_box silent=true ! fakesink";
             const std::string rewritten = rewrite_fragment_names(fragment, mapping);
             require_contains(rewritten, "name=pre_boxdecode_suf",
                              "rewrite_fragment_names should still rewrite element name");
             require_contains(rewritten, "stage-id=logical_box",
                              "rewrite_fragment_names should preserve explicit logical stage-id");
           }

           GraphOptions opt;
           opt.element_name_prefix = "x_";
           opt.element_name_suffix = "_y";

           Graph graph(opt);
           graph.custom("videotestsrc num-buffers=1 pattern=black", InputRole::Source);
           graph.custom(
               "appsink name=mysink emit-signals=false sync=false max-buffers=1 drop=true");

           const std::string backend = graph.describe_backend(false);
           require_contains(backend, "x_mysink_y",
                            "describe_backend should include transformed appsink element name");

           gst_init_once();
           GError* parse_error = nullptr;
           GstElement* pipeline = gst_parse_launch(
               "identity name=owned ! identity name=orphan ! fakesink name=sink", &parse_error);
           if (!pipeline) {
             const std::string message =
                 parse_error && parse_error->message ? parse_error->message : "unknown parse error";
             if (parse_error) {
               g_error_free(parse_error);
             }
             throw std::runtime_error("naming-contract fixture failed to parse: " + message);
           }
           if (parse_error) {
             g_error_free(parse_error);
           }

           BuildResult build_result;
           build_result.diag = std::make_shared<pipeline_internal::DiagCtx>();
           NodeReport node_report;
           node_report.elements = {"owned", "sink"};
           build_result.diag->node_reports.push_back(std::move(node_report));

           bool rejected_orphan = false;
           try {
             enforce_names_contract(pipeline, build_result);
           } catch (const NeatError& error) {
             rejected_orphan = true;
             require(error.report().error_code == error_codes::kGraphElementName,
                     "naming ownership violations should use graph-element-name code");
             require_contains(error.what(), "orphan",
                              "naming ownership violation should identify the element");
           }
           gst_element_set_state(pipeline, GST_STATE_NULL);
           gst_object_unref(pipeline);
           require(rejected_orphan, "naming contract should reject an unowned graph element");
         }));
