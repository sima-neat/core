#include "pipeline/internal/sima/SimaPluginStaticManifest.h"
#include "gst/SimaPluginStaticManifestAbi.h"
#include "test_main.h"
#include "test_utils.h"

#include <gst/gst.h>

RUN_TEST("unit_sima_plugin_context_late_child_injection_test", ([] {
           using namespace simaai::neat::pipeline_internal::sima;

           gst_init(nullptr, nullptr);

           GstElement* pipeline = GST_ELEMENT(gst_pipeline_new("ctx_test_pipeline_late"));
           require(pipeline != nullptr, "pipeline must be created");
           GstElement* existing = GST_ELEMENT(gst_bin_new("stage_a"));
           require(existing != nullptr, "existing child should be created");
           require(gst_bin_add(GST_BIN(pipeline), existing),
                   "existing child should be addable to the pipeline");

           SimaPluginStaticManifest manifest;
           manifest.session_id = "ctx-test-late";
           manifest.model_id = "model-x";

           StageStaticSpec stage_a;
           stage_a.element_name = "stage_a";
           stage_a.logical_stage_id = "stage_a_id";
           stage_a.plugin_kind = "neatboxdecode";
           stage_a.kernel_kind = "boxdecode";
           stage_a.payload_kind = StagePayloadKind::BoxDecode;
           stage_a.boxdecode.decode_type = simaai::neat::BoxDecodeType::YoloV8;
           stage_a.boxdecode.topk = 100;
           manifest.stages.push_back(stage_a);

           StageStaticSpec late_stage;
           late_stage.element_name = "late_stage";
           late_stage.logical_stage_id = "late_stage_id";
           late_stage.plugin_kind = "neatboxdecode";
           late_stage.kernel_kind = "boxdecode";
           late_stage.payload_kind = StagePayloadKind::BoxDecode;
           late_stage.boxdecode.decode_type = simaai::neat::BoxDecodeType::YoloV8;
           late_stage.boxdecode.topk = 50;
           manifest.stages.push_back(late_stage);

           std::string attach_error;
           require(attach_manifest_context(pipeline, manifest, &attach_error),
                   "attach_manifest_context should succeed: " + attach_error);

           GstElement* late = GST_ELEMENT(gst_bin_new("late_stage"));
           require(late != nullptr, "late child element should be created");
           require(gst_bin_add(GST_BIN(pipeline), late),
                   "late child element should be addable to the pipeline");

           GstContext* late_context =
               gst_element_get_context(late, SIMA_PLUGIN_STATIC_MANIFEST_CONTEXT_TYPE);
           require(late_context != nullptr, "late child should inherit manifest context");
           require(sima_plugin_manifest_context_matches(late_context),
                   "late child context should match manifest type");

           const auto* late_accessor = sima_plugin_manifest_context_accessor(late_context);
           require(late_accessor != nullptr,
                   "late child manifest context should expose ABI-safe accessor");
           const SimaPluginStageSpec* late_payload =
               sima_plugin_manifest_context_stage_lookup_typed(
                   late_context, "late_stage_id", "late_stage",
                   SIMA_PLUGIN_STAGE_PAYLOAD_BOXDECODE);
           require(late_payload != nullptr,
                   "late child manifest context should resolve the newly added stage");
           require(late_payload->payload.boxdecode.topk == 50,
                   "late child stage payload should match the manifest entry");

           gst_context_unref(late_context);
           gst_object_unref(pipeline);
         }));
