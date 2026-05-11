#include "pipeline/internal/sima/SimaPluginStaticManifest.h"
#include "gst/SimaPluginStaticManifestAbi.h"
#include "test_main.h"
#include "test_utils.h"

#include <gst/gst.h>

RUN_TEST("unit_sima_plugin_context_destroyed_pipeline_test", ([] {
           using namespace simaai::neat::pipeline_internal::sima;

           gst_init(nullptr, nullptr);

           GstElement* pipeline = GST_ELEMENT(gst_pipeline_new("ctx_destroy_pipeline"));
           require(pipeline != nullptr, "pipeline must be created");

           SimaPluginStaticManifest manifest;
           manifest.session_id = "ctx-destroy";
           manifest.model_id = "model-destroy";
           StageStaticSpec stage;
           stage.element_name = "stage_destroy";
           stage.logical_stage_id = "stage_destroy_id";
           stage.plugin_kind = "neatboxdecode";
           stage.kernel_kind = "boxdecode";
           stage.payload_kind = StagePayloadKind::BoxDecode;
           stage.boxdecode.decode_type = simaai::neat::BoxDecodeType::YoloV8;
           stage.boxdecode.topk = 55;
           manifest.stages.push_back(stage);

           std::string attach_error;
           require(attach_manifest_context(pipeline, manifest, &attach_error),
                   "attach_manifest_context should succeed: " + attach_error);

           GstContext* context =
               gst_element_get_context(pipeline, SIMA_PLUGIN_STATIC_MANIFEST_CONTEXT_TYPE);
           require(context != nullptr, "pipeline should expose manifest context");

           gst_object_unref(pipeline);

           SimaPluginManifestLookupStatus status = SIMA_PLUGIN_MANIFEST_LOOKUP_STATUS_NO_CONTEXT;
           const auto* resolved = sima_plugin_manifest_context_stage_lookup_typed_checked(
               context, "stage_destroy_id", "stage_destroy",
               SIMA_PLUGIN_STAGE_PAYLOAD_BOXDECODE, &status);
           require(resolved != nullptr,
                   "context should remain valid after original pipeline teardown");
           require(status == SIMA_PLUGIN_MANIFEST_LOOKUP_STATUS_OK,
                   "destroyed pipeline lookup status mismatch");
           require(resolved->payload.boxdecode.topk == 55,
                   "destroyed pipeline payload mismatch");

           gst_context_unref(context);
         }));
