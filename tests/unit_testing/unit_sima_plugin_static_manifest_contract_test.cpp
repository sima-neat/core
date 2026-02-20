#include "pipeline/internal/sima/SimaPluginStaticManifest.h"
#include "gst/SimaPluginStaticManifestAbi.h"
#include "test_main.h"
#include "test_utils.h"

#include <gst/gst.h>

#include <string>
#include <vector>

RUN_TEST("unit_sima_plugin_static_manifest_contract_test", ([] {
           using namespace simaai::neat::pipeline_internal::sima;

           gst_init(nullptr, nullptr);

           GError* err = nullptr;
           GstElement* pipeline = gst_parse_launch("fakesrc ! fakesink", &err);
           require(err == nullptr, "gst_parse_launch should succeed");
           require(pipeline != nullptr, "pipeline must be created");

           SimaPluginStaticManifest manifest;
           manifest.manifest_version = 2;
           manifest.session_id = "contract-sess";
           manifest.model_id = "contract-model";

           StageStaticSpec pre;
           pre.element_name = "pre";
           pre.logical_stage_id = "stage_pre";
           pre.plugin_kind = "neatprocesscvu";
           pre.kernel_kind = "preproc";
           pre.sink_pad_tensor_index_map = {0};
           pre.runtime_defaults = {{"config_path", "/tmp/pre.json"}};
           manifest.stages.push_back(pre);

           StageStaticSpec box;
           box.element_name = "box";
           box.logical_stage_id = "stage_box";
           box.plugin_kind = "neatboxdecode";
           box.kernel_kind = "boxdecode";
           box.runtime_defaults = {{"decode_type", "yolov8"}, {"topk", 100}};
           manifest.stages.push_back(box);

           std::string attach_error;
           require(attach_manifest_context(pipeline, manifest, &attach_error),
                   "attach_manifest_context should succeed: " + attach_error);

           GstContext* context =
               gst_element_get_context(pipeline, SIMA_PLUGIN_STATIC_MANIFEST_CONTEXT_TYPE);
           require(context != nullptr, "pipeline should expose manifest context");

           const auto* accessor = sima_plugin_manifest_context_accessor(context);
           require(accessor != nullptr, "context must expose accessor pointer");
           require(accessor->abi_version == SIMA_PLUGIN_STATIC_MANIFEST_ABI_VERSION,
                   "unexpected accessor ABI version");

           require(accessor->manifest_version != nullptr,
                   "accessor manifest_version callback should exist");
           require(accessor->manifest_version(accessor->user_data) == 2,
                   "manifest version callback mismatch");

           require(accessor->session_id != nullptr, "accessor session_id callback should exist");
           require(accessor->model_id != nullptr, "accessor model_id callback should exist");
           require(std::string(accessor->session_id(accessor->user_data)) == "contract-sess",
                   "session id callback mismatch");
           require(std::string(accessor->model_id(accessor->user_data)) == "contract-model",
                   "model id callback mismatch");

           const gchar* pre_stage =
               sima_plugin_manifest_lookup_stage_by_element_name(accessor, "pre");
           const gchar* box_stage =
               sima_plugin_manifest_lookup_stage_by_logical_id(accessor, "stage_box");
           require(pre_stage != nullptr && *pre_stage,
                   "stage lookup by element name should return JSON");
           require(box_stage != nullptr && *box_stage,
                   "stage lookup by logical id should return JSON");

           std::string payload;
           guint version = 0;
           require(extract_manifest_json_from_context(context, payload, &version),
                   "extract_manifest_json_from_context should succeed");
           require(version == 2, "context version should be 2");
           const auto parsed = parse_manifest_json(payload);
           require(parsed.has_value(), "parsed manifest payload should be valid");
           require(parsed->stages.size() == 2, "parsed manifest stage count mismatch");
           require(parsed->stages[0].sink_pad_tensor_index_map == std::vector<int>({0}),
                   "sink_pad_tensor_index_map roundtrip mismatch");

           gst_context_unref(context);
           gst_object_unref(pipeline);
         }));
