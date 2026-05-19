#include "pipeline/BoxDecodeType.h"
#include "pipeline/internal/sima/SimaPluginStaticManifest.h"
#include "test_main.h"
#include "unit_testing/manifest_plugin_startup_test_utils.h"

namespace {

using simaai::neat::BoxDecodeType;
using simaai::neat::pipeline_internal::sima::BoxDecodeStagePayload;
using simaai::neat::pipeline_internal::sima::SimaPluginStaticManifest;
using simaai::neat::pipeline_internal::sima::StagePayloadKind;
using simaai::neat::pipeline_internal::sima::StageStaticSpec;

SimaPluginStaticManifest make_model_managed_boxdecode_manifest() {
  SimaPluginStaticManifest manifest;
  manifest.session_id = "agg-model-managed-test";
  manifest.model_id = "agg-model-managed-model";

  StageStaticSpec stage;
  stage.element_name = "boxdecode_model_managed";
  stage.logical_stage_id = "stage_boxdecode_model_managed";
  stage.plugin_kind = "neatboxdecode";
  stage.kernel_kind = "boxdecode";
  stage.payload_kind = StagePayloadKind::BoxDecode;
  stage.boxdecode = BoxDecodeStagePayload{
      .decode_type = BoxDecodeType::YoloV8,
      .input_dtype = "INT8",
      .tess_needed = false,
      .quant_needed = true,
      .model_owned_flags = true,
      .quant_contract_required = true,
      .detection_threshold = 0.25,
      .nms_iou_threshold = 0.45,
      .topk = 100,
  };
  manifest.stages.push_back(stage);
  return manifest;
}

} // namespace

RUN_TEST("unit_agg_model_managed_no_standalone_json_test", ([] {
           sima_test::require_plugin_or_skip("neatboxdecode");

           const auto manifest = make_model_managed_boxdecode_manifest();
           // neatboxdecode still exposes a legacy config property, so it is a good
           // regression target for "stage contract present means no standalone config read".
           const auto result = sima_test::run_raw_gst_pipeline(
               "agg_model_managed_no_standalone_json",
               "fakesrc num-buffers=1 ! neatboxdecode name=boxdecode_model_managed "
               "stage-id=stage_boxdecode_model_managed config=/tmp/does_not_exist_boxdecode.json "
               "silent=true ! fakesink",
               &manifest, GST_STATE_READY);

           require(result.error.empty(), "model-managed startup should not require legacy "
                                         "standalone config when stage contract is present: " +
                                             result.error);
           require(result.set_state_result != GST_STATE_CHANGE_FAILURE,
                   "model-managed startup should not fail during NULL->READY");
           require(result.wait_state_result != GST_STATE_CHANGE_FAILURE,
                   "model-managed startup should not fail while waiting for READY");
         }));
