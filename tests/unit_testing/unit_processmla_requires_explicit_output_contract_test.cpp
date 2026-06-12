#include "gst/GstHelpers.h"
#include "pipeline/internal/sima/SimaPluginStaticManifest.h"
#include "test_main.h"
#include "test_utils.h"
#include "unit_testing/manifest_plugin_startup_test_utils.h"

namespace {

using simaai::neat::pipeline_internal::sima::ProcessMlaStagePayload;
using simaai::neat::pipeline_internal::sima::SimaPluginStaticManifest;
using simaai::neat::pipeline_internal::sima::StagePayloadKind;
using simaai::neat::pipeline_internal::sima::StageStaticSpec;

SimaPluginStaticManifest make_processmla_manifest_missing_explicit_contract() {
  SimaPluginStaticManifest manifest;
  manifest.session_id = "mla-explicit-contract-test";
  manifest.model_id = "mla-explicit-contract-model";

  StageStaticSpec stage;
  stage.element_name = "mla_explicit_contract";
  stage.logical_stage_id = "stage_mla_explicit";
  stage.plugin_kind = "neatprocessmla";
  stage.kernel_kind = "mla";
  stage.payload_kind = StagePayloadKind::ProcessMla;
  stage.processmla = ProcessMlaStagePayload{
      .model_path = "/tmp/mla_explicit_contract_unused_model.tar.gz",
      .batch_size = 1,
      .batch_sz_model = 1,
  };
  manifest.stages.push_back(stage);
  return manifest;
}

} // namespace

RUN_TEST("unit_processmla_requires_explicit_output_contract_test", ([] {
           sima_test::require_plugin_or_skip("neatprocessmla");

           const auto manifest = make_processmla_manifest_missing_explicit_contract();
           const auto result = sima_test::run_raw_gst_pipeline(
               "processmla_requires_explicit_output_contract",
               "fakesrc num-buffers=1 ! neatprocessmla name=mla_explicit_contract "
               "stage-id=stage_mla_explicit silent=true ! fakesink",
               &manifest, GST_STATE_PAUSED);

           if (result.error.find("Unable to get dispatcher") != std::string::npos ||
               is_dispatcher_unavailable(result.error)) {
             skip_test_exception("MLA dispatcher unavailable for explicit output-contract test");
           }
           require(!result.error.empty(),
                   "processmla should fail when explicit output contract fields are missing");

           require_contains(
               result.error, "Unable to build typed stage config",
               "processmla should reject manifest stages without explicit v3 output contract");
           require_contains(result.error, "canonical_contract_missing",
                            "processmla should report canonical contract fields as missing");
           require_contains(result.error, "source_used='manifest_context'",
                            "processmla should report manifest context as the semantic source");
           require_contains(result.error, "fallback_chain='manifest_context_only'",
                            "processmla should report manifest-only resolution");
         }));
