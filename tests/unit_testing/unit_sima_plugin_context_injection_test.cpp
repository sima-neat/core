#include "pipeline/internal/sima/SimaPluginStaticManifest.h"
#include "gst/SimaPluginStaticManifestAbi.h"
#include "test_main.h"
#include "test_utils.h"

#include <gst/gst.h>

RUN_TEST("unit_sima_plugin_context_injection_test", ([] {
           using namespace simaai::neat::pipeline_internal::sima;

           gst_init(nullptr, nullptr);

           GError* err = nullptr;
           GstElement* pipeline = gst_parse_launch("fakesrc ! fakesink", &err);
           require(err == nullptr, "gst_parse_launch should succeed for context injection test");
           require(pipeline != nullptr, "pipeline must be created");

           SimaPluginStaticManifest manifest;
           manifest.manifest_version = 2;
           manifest.session_id = "ctx-test";
           manifest.model_id = "model-x";
           StageStaticSpec stage;
           stage.element_name = "stage_a";
           stage.logical_stage_id = "stage_a_id";
           stage.plugin_kind = "neatboxdecode";
           stage.kernel_kind = "boxdecode";
           stage.runtime_defaults = {{"decode_type", "yolov8"}, {"topk", 100}};
           manifest.stages.push_back(stage);

           std::string attach_error;
           require(attach_manifest_context(pipeline, manifest, &attach_error),
                   "attach_manifest_context should succeed: " + attach_error);

           GstContext* context =
               gst_element_get_context(pipeline, SIMA_PLUGIN_STATIC_MANIFEST_CONTEXT_TYPE);
           require(context != nullptr, "pipeline should expose manifest context");

           const auto* accessor = sima_plugin_manifest_context_accessor(context);
           require(accessor != nullptr, "manifest context should expose ABI-safe accessor");
           require(accessor->manifest_version != nullptr,
                   "manifest accessor should expose manifest_version callback");
           require(accessor->manifest_version(accessor->user_data) == 2,
                   "manifest accessor version callback mismatch");
           const gchar* stage_payload =
               sima_plugin_manifest_lookup_stage_by_element_name(accessor, "stage_a");
           require(stage_payload != nullptr && *stage_payload,
                   "manifest accessor should resolve stage JSON by element_name");
           const gchar* stage_payload_by_id =
               sima_plugin_manifest_lookup_stage_by_logical_id(accessor, "stage_a_id");
           require(stage_payload_by_id != nullptr && *stage_payload_by_id,
                   "manifest accessor should resolve stage JSON by logical_stage_id");

           std::string payload;
           guint version = 0;
           require(extract_manifest_json_from_context(context, payload, &version),
                   "context payload extraction should succeed");
           require(version == 2, "manifest context version should be 2");

           std::string parse_error;
           const auto parsed = parse_manifest_json(payload, &parse_error);
           require(parsed.has_value(), "manifest JSON should parse: " + parse_error);
           require(parsed->session_id == "ctx-test", "manifest session_id roundtrip mismatch");
           require(parsed->stages.size() == 1, "manifest stage count mismatch");
           require(parsed->stages[0].element_name == "stage_a", "manifest stage key mismatch");
           require(parsed->stages[0].runtime_defaults.contains("decode_type"),
                   "runtime defaults should roundtrip via context payload");

           gst_context_unref(context);
           gst_object_unref(pipeline);
         }));
