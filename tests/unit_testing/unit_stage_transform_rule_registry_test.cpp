#include "pipeline/internal/sima/StageTransformRuleRegistry.h"
#include "test_main.h"
#include "test_utils.h"

RUN_TEST("unit_stage_transform_rule_registry_test", ([] {
           using namespace simaai::neat::pipeline_internal::sima;

           const StageTransformRuleRegistry& rules = default_stage_transform_rules();

           require(rules.is_pre_adapter("preproc"),
                   "preproc should be classified as pre-adapter");
           require(rules.is_pre_adapter("QuantTess"),
                   "QuantTess case-insensitive pre-adapter classification failed");

           require(rules.is_post_adapter("boxdecode"),
                   "boxdecode should be classified as post-adapter");
           require(rules.is_post_adapter("DetessDequant"),
                   "DetessDequant case-insensitive post-adapter classification failed");

           require(!rules.is_pre_adapter("processmla"),
                   "processmla must not be pre-adapter");
           require(!rules.is_post_adapter("processmla"),
                   "processmla must not be post-adapter");

           const auto pre_rule = rules.lookup("preproc");
           require(pre_rule.has_value(), "preproc lookup must return a transform rule");
           require(pre_rule->output_source == StageTensorSource::MlaInputs,
                   "preproc output source should be MLA inputs");
           require(pre_rule->input_source == StageTensorSource::None,
                   "preproc input source should remain None");
           require(!pre_rule->propagate_output_quant,
                   "preproc should not propagate output quant");

           const auto post_rule = rules.lookup("detessdequant");
           require(post_rule.has_value(), "detessdequant lookup must return a transform rule");
           require(post_rule->input_source == StageTensorSource::MlaOutputs,
                   "detessdequant input source should be MLA outputs");
           require(post_rule->propagate_output_quant,
                   "detessdequant should propagate output quant");

           const auto missing_rule = rules.lookup("processmla");
           require(!missing_rule.has_value(), "processmla must not have a transform rule");
         }));
