#include "builder/Node.h"
#include "nodes/io/Input.h"
#include "nodes/sima/QuantTess.h"
#include "pipeline/Session.h"
#include "pipeline/internal/PipelineBuild.h"
#include "test_main.h"
#include "test_utils.h"

#include <nlohmann/json.hpp>

#include <memory>
#include <vector>

RUN_TEST("unit_pipeline_build_wiring_test", ([] {
           using namespace simaai::neat;
           using namespace simaai::neat::pipeline_internal;

           SessionOptions opt;
           opt.element_name_prefix = "pre_";
           opt.element_name_suffix = "_suf";

           InputOptions src_opt;
           src_opt.media_type = "video/x-raw";
           src_opt.format = "RGB";
           src_opt.use_simaai_pool = false;
           src_opt.max_width = 96;
           src_opt.max_height = 96;
           src_opt.max_depth = 3;

           QuantTessOptions qopt;
           qopt.config_json = nlohmann::json{
               {"node_name", "quant0"},
               {"input_buffers", nlohmann::json::array({nlohmann::json{{"name", "decoder"}}})},
           };
           qopt.keep_config = true;
           qopt.config_dir = "/tmp";
           qopt.element_name = "quant_wire";

           // Deterministic pipeline text + boundary insertion toggle.
           Session session(opt);
           session.add(nodes::Input(src_opt));
           session.add(nodes::QuantTess(qopt));
           session.add(nodes::Output(OutputOptions::Latest()));

           const std::string no_bound_a = session.describe_backend(false);
           const std::string no_bound_b = session.describe_backend(false);
           require_contains(no_bound_a, "pre_mysrc_suf",
                            "PipelineBuild wiring: transformed mysrc missing");
           require_contains(no_bound_a, "pre_mysink_suf",
                            "PipelineBuild wiring: transformed mysink missing");
           require_contains(no_bound_b, "pre_mysrc_suf",
                            "PipelineBuild wiring: transformed mysrc missing");
           require_contains(no_bound_b, "pre_mysink_suf",
                            "PipelineBuild wiring: transformed mysink missing");

           const std::string with_bound = session.describe_backend(true);
           require_contains(with_bound, "identity name=pre_sima_b0_suf",
                            "PipelineBuild wiring: boundary insertion toggle mismatch");

           // Legacy config wiring hooks are no-op; build must not mutate config JSON.
           const auto src = nodes::Input(src_opt);
           const auto quant_node = nodes::QuantTess(qopt);
           std::vector<std::shared_ptr<Node>> nodes_vec = {src, quant_node};

           PipelineBuildContext ctx(opt);
           ctx.apply_name_transform_to_configs(nodes_vec);
           ctx.wire_configs_by_order(nodes_vec);
           const BuildWiringReport report = ctx.check_config_wiring(nodes_vec);
           require(report.issues.empty(),
                   "PipelineBuild wiring: config wiring report should be clean");

           auto* quant_raw = dynamic_cast<simaai::neat::QuantTess*>(quant_node.get());
           require(quant_raw != nullptr, "PipelineBuild wiring: QuantTess cast failed");
           const nlohmann::json* cfg = quant_raw->config_json();
           require(cfg != nullptr, "PipelineBuild wiring: missing QuantTess config json");
           const std::string wired = (*cfg)["input_buffers"][0]["name"].get<std::string>();
           require(wired == "decoder", "PipelineBuild wiring: stage-aware configs should not "
                                       "depend on framework input_buffers rewrites");
           require((*cfg)["node_name"].get<std::string>() == "quant0",
                   "PipelineBuild wiring: framework should not rewrite node_name");
         }));
