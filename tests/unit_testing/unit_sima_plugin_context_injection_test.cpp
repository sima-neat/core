#include "pipeline/internal/sima/SimaPluginStaticManifest.h"
#include "gst/SimaPluginStaticManifestAbi.h"
#include "test_main.h"
#include "test_utils.h"

#include <gst/gst.h>

RUN_TEST("unit_sima_plugin_context_injection_test", ([] {
           using namespace simaai::neat::pipeline_internal::sima;

           gst_init(nullptr, nullptr);

           GstElement* pipeline = GST_ELEMENT(gst_pipeline_new("ctx_test_pipeline"));
           require(pipeline != nullptr, "pipeline must be created");
           GstElement* child = GST_ELEMENT(gst_bin_new("stage_a"));
           require(child != nullptr, "matched child should exist");
           require(gst_bin_add(GST_BIN(pipeline), child),
                   "child stage should be addable to the pipeline");

           SimaPluginStaticManifest manifest;
           manifest.session_id = "ctx-test";
           manifest.model_id = "model-x";
           StageStaticSpec stage;
           stage.element_name = "stage_a";
           stage.logical_stage_id = "stage_a_id";
           stage.plugin_kind = "neatboxdecode";
           stage.kernel_kind = "boxdecode";
           stage.payload_kind = StagePayloadKind::BoxDecode;
           stage.boxdecode.decode_type = simaai::neat::BoxDecodeType::YoloV8;
           stage.boxdecode.topk = 100;
           manifest.stages.push_back(stage);

           std::string attach_error;
           require(attach_manifest_context(pipeline, manifest, &attach_error),
                   "attach_manifest_context should succeed: " + attach_error);

           GstContext* context =
               gst_element_get_context(pipeline, SIMA_PLUGIN_STATIC_MANIFEST_CONTEXT_TYPE);
           require(context != nullptr, "pipeline should expose manifest context");

           const auto* accessor = sima_plugin_manifest_context_accessor(context);
           require(accessor != nullptr, "manifest context should expose ABI-safe accessor");
           const SimaPluginStageSpec* stage_payload =
               sima_plugin_manifest_stage_by_element_name(accessor, "stage_a");
           require(stage_payload != nullptr,
                   "manifest accessor should resolve stage by element_name");
           const SimaPluginStageSpec* stage_payload_by_id =
               sima_plugin_manifest_stage_by_logical_id(accessor, "stage_a_id");
           require(stage_payload_by_id != nullptr,
                   "manifest accessor should resolve stage by logical_stage_id");
           require(stage_payload == stage_payload_by_id,
                   "stage lookup by id and name should resolve same stage");
           require(stage_payload->payload_kind == SIMA_PLUGIN_STAGE_PAYLOAD_BOXDECODE,
                   "manifest payload kind mismatch");
           require(stage_payload->payload.boxdecode.decode_type != nullptr,
                   "decode type should be populated from stage payload");
           require(std::string(stage_payload->payload.boxdecode.decode_type) == "yolov8",
                   "decode type token should be populated from stage payload");
           require(stage_payload->payload.boxdecode.topk == 100,
                   "topk should be populated from stage payload");

           GstContext* child_context =
               gst_element_get_context(child, SIMA_PLUGIN_STATIC_MANIFEST_CONTEXT_TYPE);
           require(child_context != nullptr, "matched child should expose manifest context");
           require(sima_plugin_manifest_context_matches(child_context),
                   "manifest context type mismatch on child element");
           const auto* child_accessor = sima_plugin_manifest_context_accessor(child_context);
           require(child_accessor != nullptr,
                   "child manifest context should expose ABI-safe accessor");
           const SimaPluginStageSpec* child_stage = sima_plugin_manifest_context_stage_lookup_typed(
               child_context, "stage_a_id", "stage_a", SIMA_PLUGIN_STAGE_PAYLOAD_BOXDECODE);
           require(child_stage == stage_payload,
                   "child manifest context should resolve the same stage payload");
           const gchar* stage_key = sima_plugin_manifest_stage_key(child_stage);
           require(stage_key != nullptr && std::string(stage_key) == "stage_a_id",
                   "manifest stage key should prefer logical stage id");

           gst_context_unref(child_context);
           gst_context_unref(context);
           gst_object_unref(pipeline);
         }));
