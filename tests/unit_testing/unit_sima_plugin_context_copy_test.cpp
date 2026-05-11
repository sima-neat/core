#include "pipeline/internal/sima/SimaPluginStaticManifest.h"
#include "gst/SimaPluginStaticManifestAbi.h"
#include "test_main.h"
#include "test_utils.h"

#include <gst/gst.h>

namespace {

using namespace simaai::neat::pipeline_internal::sima;

SimaPluginStaticManifest make_manifest(const char* element_name,
                                       const char* stage_id,
                                       int topk) {
  SimaPluginStaticManifest manifest;
  manifest.session_id = "ctx-copy";
  manifest.model_id = "model-copy";

  StageStaticSpec stage;
  stage.element_name = element_name;
  stage.logical_stage_id = stage_id;
  stage.plugin_kind = "neatboxdecode";
  stage.kernel_kind = "boxdecode";
  stage.payload_kind = StagePayloadKind::BoxDecode;
  stage.boxdecode.decode_type = simaai::neat::BoxDecodeType::YoloV8;
  stage.boxdecode.topk = topk;
  manifest.stages.push_back(stage);
  return manifest;
}

} // namespace

RUN_TEST("unit_sima_plugin_context_copy_test", ([] {
           gst_init(nullptr, nullptr);

           GstElement* pipeline = GST_ELEMENT(gst_pipeline_new("ctx_copy_pipeline"));
           require(pipeline != nullptr, "pipeline must be created");

           std::string attach_error;
           require(attach_manifest_context(pipeline, make_manifest("stage_copy", "stage_copy_id", 77),
                                          &attach_error),
                   "attach_manifest_context should succeed: " + attach_error);

           GstContext* context =
               gst_element_get_context(pipeline, SIMA_PLUGIN_STATIC_MANIFEST_CONTEXT_TYPE);
           require(context != nullptr, "pipeline should expose manifest context");

           GstContext* copied = gst_context_copy(context);
           require(copied != nullptr, "gst_context_copy should succeed");

           gst_context_unref(context);
           gst_object_unref(pipeline);

           SimaPluginManifestLookupStatus status = SIMA_PLUGIN_MANIFEST_LOOKUP_STATUS_NO_CONTEXT;
           const auto* stage = sima_plugin_manifest_context_stage_lookup_typed_checked(
               copied, "stage_copy_id", "stage_copy", SIMA_PLUGIN_STAGE_PAYLOAD_BOXDECODE,
               &status);
           require(stage != nullptr, "copied context should resolve stage after pipeline teardown");
           require(status == SIMA_PLUGIN_MANIFEST_LOOKUP_STATUS_OK,
                   "copied context lookup status mismatch");
           require(stage->payload.boxdecode.topk == 77, "copied context payload mismatch");

           gst_context_unref(copied);
         }));
